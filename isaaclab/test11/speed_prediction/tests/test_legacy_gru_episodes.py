"""Exercise full-episode training against the deployed legacy GRU contract."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

TEST11 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST11.parent))

from test11.speed_prediction.common.time_aware_velocity_data import (
    TAQ_LABEL_SEMANTICS, TAQ_SCHEMA_VERSION, TAQ_STREAM_FORMAT,
)
from test11.speed_prediction.legacy_gru.online_gru_velocity import OnlineLegacyGRU, load_legacy_checkpoint
from test11.speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU
from test11.speed_prediction.legacy_gru.train_gru_velocity import WeightedSmoothL1Loss
from test11.speed_prediction.legacy_gru.train_gru_velocity_episodes import (
    EpisodeDataset, collate_episodes, discover_episodes, input_norm,
    predict_episode_frames, read_episode, run_epoch,
    prepare_initialization,
)


def stream_payload(uid="episode_0", count=20):
    x = torch.zeros(count, 24)
    x[:, 0] = 80 + torch.arange(count) * .04
    x[:, 1] = torch.arange(count) * .02
    x[:, 6] = .1
    x[:, 11] = .01
    x[1:, 12] = .02
    x[:, 22:24] = 1
    sources = torch.ones(count, dtype=torch.long)
    sources[:2] = 4
    x[:2, :3] = 0
    x[:2, 22:24] = 0
    sources[10] = 4
    x[10, :3] = 0
    x[10, 22:24] = 0
    return {
        "schema_version": TAQ_SCHEMA_VERSION, "stream_format": TAQ_STREAM_FORMAT,
        "metadata": {"episode_uid": uid, "scenario": "sd", "episode_complete": True,
                     "label_semantics": TAQ_LABEL_SEMANTICS,
                     "angular_velocity_z_semantics": "yaw_only_heading_derivative"},
        "timestamps_s": torch.arange(count, dtype=torch.float64) * .02,
        "sequence_features": x, "sequence_source_ids": sources,
        "sequence_valid_mask": torch.ones(count, dtype=torch.bool),
        "target_valid_mask": torch.arange(count) >= 6,
        "target_velocity_base": torch.tensor([2., 1., 0.]).repeat(count, 1),
        # Observational resets must not cut the full-episode GRU history.
        "state_reset_mask": torch.ones(count, dtype=torch.bool),
    }


def save_stream(root, data):
    root.mkdir(parents=True, exist_ok=True)
    path = root / (data["metadata"]["episode_uid"] + ".taqstream")
    torch.save(data, path)
    return path


@pytest.mark.parametrize("smooth_window", [1, 4])
def test_full_episode_predictions_match_online_with_loss_and_initial_wait(tmp_path, smooth_window):
    torch.manual_seed(41)
    data = stream_payload()
    path = save_stream(tmp_path, data)
    episode = read_episode(path, smooth_window)
    model = VelocityPredictorGRU(hidden_dim=16).eval()
    model.set_input_norm_stats(torch.arange(13).float(), torch.arange(13).float() + 1)
    adapter = OnlineLegacyGRU(model, context="full", smooth_window=smooth_window)
    actual = []
    for t, features, source in zip(data["timestamps_s"], data["sequence_features"], data["sequence_source_ids"]):
        result = adapter.update(timestamp_s=float(t), position_base_m=features[:3].numpy(),
                                robot_linear_velocity_base_mps=features[6:9].numpy(),
                                robot_angular_velocity_base_rps=features[9:12].numpy(),
                                episode_key="episode_0", direct_observation=int(source) <= 2)
        if result is not None:
            actual.append(torch.from_numpy(result["velocity"]))
    with torch.no_grad():
        expected = predict_episode_frames(model, episode["x"][None])[0]
    torch.testing.assert_close(torch.stack(actual), expected, rtol=1e-5, atol=1e-6)
    assert episode["record"]["initialization_discarded_frames"] == 2
    assert episode["record"]["input_frames"] == 17
    assert int(episode["supervised"].sum()) == 14


def test_labels_never_affect_inputs_and_discovery_ignores_chunks(tmp_path):
    data = stream_payload()
    path = save_stream(tmp_path, data)
    original = read_episode(path, 1)["x"]
    data["target_velocity_base"] += 10000
    torch.save(data, path)
    torch.testing.assert_close(read_episode(path, 1)["x"], original, rtol=0, atol=0)
    torch.save({"unsupported": True}, tmp_path / "taq_velocity_chunk_0000.pt")
    assert discover_episodes([tmp_path, tmp_path]) == [path.resolve()]


@pytest.mark.parametrize("problem", ["incomplete", "time", "mask", "schema"])
def test_reject_invalid_streams(tmp_path, problem):
    data = stream_payload()
    if problem == "incomplete":
        data["metadata"]["episode_complete"] = False
    elif problem == "time":
        data["timestamps_s"][8] = data["timestamps_s"][7]
    elif problem == "mask":
        data["sequence_valid_mask"][8] = False
    else:
        data["schema_version"] = "taq_velocity_v1"
    with pytest.raises(ValueError):
        read_episode(save_stream(tmp_path, data), 1)


def test_frame_loss_excludes_padding_and_burn_in_and_gradient_reaches_episode_start(tmp_path):
    torch.manual_seed(12)
    files = [save_stream(tmp_path, stream_payload(f"ep_{i}", count))
             for i, count in enumerate((13, 20))]
    dataset = EpisodeDataset(files)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_episodes)
    model = VelocityPredictorGRU(hidden_dim=16)
    mean, std = input_norm(dataset)
    values = torch.cat([e["x"] for e in dataset.episodes])
    torch.testing.assert_close(mean, values.double().mean(0).float())
    model.set_input_norm_stats(mean, std)
    metrics = run_epoch(model, loader, torch.device("cpu"), WeightedSmoothL1Loss())
    assert metrics["frame_count"] == 7 + 14
    batch = collate_episodes(dataset.episodes)
    with torch.no_grad():
        original = predict_episode_frames(model, batch["x"])
        batch["x"][0, 10:] = 1e8
        changed = predict_episode_frames(model, batch["x"])
    torch.testing.assert_close(original[batch["supervised"]], changed[batch["supervised"]])
    x = dataset[1]["x"].clone().requires_grad_()
    predict_episode_frames(model, x[None])[0, -1].sum().backward()
    assert x.grad[0].abs().sum() > 0


def test_cli_preparation_training_and_checkpoint_live_compatibility(tmp_path):
    for i in range(6):
        save_stream(tmp_path / "data", stream_payload(f"ep_{i}"))
    script = TEST11 / "speed_prediction/legacy_gru/train_gru_velocity_episodes.py"
    args = [sys.executable, str(script), "--data_dirs", str(tmp_path / "data"),
            "--train_ratio", ".5", "--test_ratio", ".2", "--device", "cpu",
            "--hidden_dim", "8", "--batch_size", "2", "--epochs", "1"]
    prepared = subprocess.run(args + ["--prepare_only", "--save_dir", str(tmp_path / "prepared")],
                              capture_output=True, text=True, timeout=30)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    assert not list((tmp_path / "prepared").rglob("*.pt"))
    trained = subprocess.run(args + ["--save_dir", str(tmp_path / "trained")],
                             capture_output=True, text=True, timeout=30)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    run = next((tmp_path / "trained").iterdir())
    model, checkpoint = load_legacy_checkpoint(run / "best_model.pt")
    assert model.input_dim == 13
    assert checkpoint["training_protocol"]["context"] == "full_episode"
    manifest = json.loads((run / "data_manifest.json").read_text())
    identities = [{e["episode_uid"] for e in manifest["episodes"][split]}
                  for split in ("train", "val", "test")]
    assert all(not identities[i] & identities[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    assert [manifest["counts"][s]["episodes"] for s in ("train", "val", "test")] == [3, 2, 1]
    metrics = json.loads((run / "test_metrics.json").read_text())
    assert metrics["frame_count"] == 14
    assert all(f"err_norm_p{p}" in metrics for p in (50, 75, 80, 85, 90, 95))


def save_initial_checkpoint(path, enabled=True):
    torch.manual_seed(91)
    config = dict(input_dim=13, hidden_dim=8, num_layers=1, dropout=0.)
    model = VelocityPredictorGRU(**config)
    mean, std = torch.arange(13).float() / 10, torch.arange(13).float() / 10 + 1
    model.set_input_norm_stats(mean, std)
    model.norm_enabled.fill_(enabled)
    torch.save({"config": config, "model_state_dict": model.state_dict(),
                "input_norm": {"mean": mean, "std": std},
                "data_schema": TAQ_SCHEMA_VERSION, "label_semantics": TAQ_LABEL_SEMANTICS,
                "preprocessing": {"pos_smooth_window": 1,
                                  "taq_moving_flag": "horizontal_base_speed_gt_0.05_mps"}}, path)
    return model


@pytest.mark.parametrize("enabled", [True, False])
def test_initialization_preserves_predictions_and_all_normalization_buffers(tmp_path, enabled):
    path = tmp_path / "source.pt"
    source = save_initial_checkpoint(path, enabled)
    args = SimpleNamespace(init_checkpoint=str(path), hidden_dim=None, num_layers=None,
                           dropout=None, pos_smooth_window=None)
    model, info = prepare_initialization(args)
    assert (args.hidden_dim, args.num_layers, args.dropout, args.pos_smooth_window) == (8, 1, 0., 1)
    assert info["normalization_source"] == "init_checkpoint"
    assert info["optimizer_state"] == "new_AdamW"
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    x = torch.randn(2, 15, 13)
    with torch.no_grad():
        torch.testing.assert_close(model(x), source(x), rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [("hidden_dim", 16), ("pos_smooth_window", 3)])
def test_initialization_rejects_architecture_or_preprocessing_changes(tmp_path, field, value):
    path = tmp_path / "source.pt"
    save_initial_checkpoint(path)
    args = SimpleNamespace(init_checkpoint=str(path), hidden_dim=None, num_layers=None,
                           dropout=None, pos_smooth_window=None)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="differs"):
        prepare_initialization(args)


def test_finetune_cli_keeps_epoch_zero_and_source_norm_when_no_validation_improvement(tmp_path):
    path = tmp_path / "source.pt"
    source = save_initial_checkpoint(path)
    original_bytes = path.read_bytes()
    for i in range(6):
        save_stream(tmp_path / "data", stream_payload(f"ep_{i}"))
    script = TEST11 / "speed_prediction/legacy_gru/train_gru_velocity_episodes.py"
    args = [sys.executable, str(script), "--data_dirs", str(tmp_path / "data"),
            "--init_checkpoint", str(path), "--save_dir", str(tmp_path / "fine_tuned"),
            "--train_ratio", ".5", "--test_ratio", ".2", "--device", "cpu",
            "--batch_size", "2", "--epochs", "1", "--lr", "1e-30"]
    trained = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    run = next((tmp_path / "fine_tuned").iterdir())
    model, checkpoint = load_legacy_checkpoint(run / "best_model.pt")
    assert checkpoint["best_epoch"] == 0
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    norm = json.loads((run / "input_norm.json").read_text())
    assert norm["source"] == "init_checkpoint"
    assert norm["mean"] == source.input_mean.flatten().tolist()
    assert norm["std"] == source.input_std.flatten().tolist()
    baseline = json.loads((run / "initial_val_metrics.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["best_val_err_norm_mean"] == baseline["err_norm_mean"]
    assert len(json.loads((run / "history.json").read_text())) == 1
    assert path.read_bytes() == original_bytes

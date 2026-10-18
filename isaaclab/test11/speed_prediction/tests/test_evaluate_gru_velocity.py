"""Check held-out GRU evaluation with the same preprocessing as training."""

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test11.speed_prediction.common.time_aware_velocity_data import (
    TAQ_LABEL_SEMANTICS, TAQ_SCHEMA_VERSION, TAQ_STREAM_FORMAT, TAQ_YAW_RATE_SEMANTICS,
)
from test11.speed_prediction.legacy_gru import evaluate_gru_velocity as evaluator
from test11.speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU
from test11.speed_prediction.legacy_gru.train_gru_velocity import WeightedSmoothL1Loss
from test11.speed_prediction.legacy_gru.train_gru_velocity_episodes import (
    EpisodeDataset, collate_episodes, run_epoch,
)


def create_data(root, uid, context="full"):
    t = torch.arange(8, dtype=torch.float32) * .02
    x = torch.zeros(8, 24)
    x[:, 0] = 80 + t
    x[1:, 12] = .02
    x[:, 22:24] = 1
    valid = torch.ones(8, dtype=torch.bool)
    metadata = dict(episode_uid=uid, scenario="sd", episode_complete=True,
                    label_semantics=TAQ_LABEL_SEMANTICS,
                    angular_velocity_z_semantics=TAQ_YAW_RATE_SEMANTICS)
    payload = dict(schema_version=TAQ_SCHEMA_VERSION, label_semantics=TAQ_LABEL_SEMANTICS,
                   sequence_features=x, sequence_valid_mask=valid,
                   sequence_source_ids=torch.ones(8, dtype=torch.long),
                   target_velocity_base=torch.ones(8, 3), target_valid_mask=valid,
                   state_reset_mask=torch.tensor([True] + [False] * 7),
                   timestamps_s=t.double(), stream_format=TAQ_STREAM_FORMAT, metadata=metadata)
    folder = root / uid
    folder.mkdir(parents=True)
    if context == "full":
        file = folder / "episode.taqstream"
    else:
        file = folder / "taq_velocity_chunk_0000.pt"
        for key in ("sequence_features", "sequence_valid_mask", "sequence_source_ids",
                    "target_velocity_base", "target_valid_mask"):
            payload[key] = payload[key].unsqueeze(0)
    torch.save(payload, file)
    return file


def checkpoint(path):
    config = dict(input_dim=13, hidden_dim=8, num_layers=1, dropout=0.)
    model = VelocityPredictorGRU(**config).eval()
    torch.save(dict(config=config, model_state_dict=model.state_dict(),
                    data_schema=TAQ_SCHEMA_VERSION, label_semantics=TAQ_LABEL_SEMANTICS,
                    input_norm=dict(mean=model.input_mean.flatten(), std=model.input_std.flatten()),
                    preprocessing=dict(pos_smooth_window=1),
                    train_args=dict(smooth_l1_beta=.05)), path)
    return model


def test_manifest_evaluation_uses_only_test_frames_and_training_preprocessing(tmp_path, monkeypatch):
    groups = {split: [str(create_data(tmp_path, split))] for split in ("train", "val", "test")}
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(json.dumps(dict(files=groups)))
    weights, result = tmp_path / "model.pt", tmp_path / "metrics.json"
    model = checkpoint(weights)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(weights),
                                     "--manifest", str(manifest), "--device", "cpu",
                                     "--out_json", str(result)])
    evaluator.main()
    actual = json.loads(result.read_text())
    expected = run_epoch(model, torch.utils.data.DataLoader(
        EpisodeDataset([Path(groups["test"][0])]), batch_size=1, collate_fn=collate_episodes),
        torch.device("cpu"), WeightedSmoothL1Loss(beta=.05))
    assert actual["split"] == "test"
    assert actual["file_count"] == 1
    assert actual["frame_count"] == 7
    assert actual["loss"] == pytest.approx(expected["loss"])
    assert actual["err_norm_mean"] == pytest.approx(expected["err_norm_mean"])


def test_window_evaluation_and_format_mismatch(tmp_path, monkeypatch):
    file = create_data(tmp_path, "test", "window")
    weights, result = tmp_path / "model.pt", tmp_path / "metrics.json"
    checkpoint(weights)
    args = ["evaluate", "--checkpoint", str(weights), "--data_dirs", str(file.parent),
            "--device", "cpu", "--out_json", str(result)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(FileNotFoundError, match=".taqstream"):
        evaluator.main()
    monkeypatch.setattr(sys, "argv", args + ["--context", "window"])
    evaluator.main()
    actual = json.loads(result.read_text())
    assert actual["context"] == "window"
    assert actual["sample_count"] == 1
    assert actual["err_norm_mean"] >= 0


def test_manifest_rejects_cross_split_episode_and_missing_test(tmp_path):
    train = create_data(tmp_path, "same-episode")
    duplicate = tmp_path / "duplicate.taqstream"
    duplicate.write_bytes(train.read_bytes())
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(json.dumps(dict(files=dict(train=[str(train)], val=[], test=[str(duplicate)]))))
    with pytest.raises(ValueError, match="cross-split"):
        evaluator.manifest_files(manifest, "test")
    manifest.write_text(json.dumps(dict(files=dict(train=[str(train)], val=[], test=[]))))
    with pytest.raises(ValueError, match="no 'test'"):
        evaluator.manifest_files(manifest, "test")

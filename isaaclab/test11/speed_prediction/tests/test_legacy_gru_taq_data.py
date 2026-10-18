import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.utils.data import DataLoader


TEST11_DIR = Path(__file__).resolve().parents[2]
if str(TEST11_DIR.parent) not in sys.path:
    sys.path.insert(0, str(TEST11_DIR.parent))

from test11.speed_prediction.common.time_aware_velocity_data import TAQ_LABEL_SEMANTICS, TAQ_SCHEMA_VERSION
from test11.speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU
from test11.speed_prediction.legacy_gru.taq_dataset import (
    TAQLegacyGRUDataset,
    chunk_identity,
    collate_velocity_windows,
    discover_chunks,
    predict_velocity_windows,
    split_episode_files,
)
from test11.speed_prediction.legacy_gru.train_gru_velocity import compute_input_norm_stats, evaluate, WeightedSmoothL1Loss


def payload(episode="run:ss:episode_000000", scenario="ss"):
    features = torch.zeros(2, 5, 24)
    features[:, :, 0] = torch.arange(5)
    features[:, :, 3:6] = 999
    features[:, :, 6] = 0.1
    features[:, :, 9:12] = torch.tensor([0.2, 0.3, 0.4])
    features[:, 1:, 12] = 0.02
    valid = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    targets = torch.zeros(2, 5, 3)
    targets[:, :, 0] = torch.arange(5)
    supervised = valid.clone()
    supervised[:, 0] = False
    return dict(
        schema_version=TAQ_SCHEMA_VERSION,
        sequence_features=features,
        sequence_valid_mask=valid,
        target_velocity_base=targets,
        target_valid_mask=supervised,
        metadata=dict(episode_uid=episode, scenario=scenario, label_semantics=TAQ_LABEL_SEMANTICS,
                      angular_velocity_z_semantics="yaw_only_heading_derivative"),
    )


def write_chunk(root, data, index=0):
    metadata = data["metadata"]
    path = root / metadata["scenario"] / "episodes" / metadata["episode_uid"].replace(":", "_")
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"taq_velocity_chunk_{index:04d}.pt"
    torch.save(data, file)
    return file


class LegacyGRUTAQDataTests(unittest.TestCase):
    def test_features_are_13_dimensional_position_increments_not_taq_fd(self):
        with TemporaryDirectory() as directory:
            path = write_chunk(Path(directory), payload())
            dataset = TAQLegacyGRUDataset(path)
            inputs, target = dataset[0]
            self.assertEqual(inputs.shape, (3, 13))
            torch.testing.assert_close(inputs[:, 0], torch.tensor([1., 2., 3.]))
            torch.testing.assert_close(inputs[:, 3], torch.ones(3))
            torch.testing.assert_close(inputs[:, 6], torch.full((3,), 0.1))
            torch.testing.assert_close(inputs[:, 12], torch.ones(3))
            torch.testing.assert_close(target, torch.tensor([3., 0., 0.]))

    def test_input_ends_at_last_supervised_frame_not_later_valid_frames(self):
        with TemporaryDirectory() as directory:
            data = payload()
            data["target_valid_mask"][0, 3:] = False
            data["sequence_features"][0, 3:] = float("nan")
            data["target_velocity_base"][0, 3:] = float("nan")
            dataset = TAQLegacyGRUDataset(write_chunk(Path(directory), data))
            self.assertEqual(dataset[0][0].shape, (2, 13))
            torch.testing.assert_close(dataset[0][1], torch.tensor([2., 0., 0.]))

    def test_unsupervised_windows_are_skipped(self):
        with TemporaryDirectory() as directory:
            data = payload()
            data["target_valid_mask"][0] = False
            dataset = TAQLegacyGRUDataset(write_chunk(Path(directory), data))
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.skipped_windows, 1)

    def test_normalization_excludes_padding(self):
        with TemporaryDirectory() as directory:
            dataset = TAQLegacyGRUDataset(write_chunk(Path(directory), payload()))
            expected = torch.cat([dataset[index][0] for index in range(len(dataset))])
            dataset.x[0, 3:] = 1e8
            mean, std = compute_input_norm_stats(dataset)
            torch.testing.assert_close(mean, expected.mean(0))
            torch.testing.assert_close(std, expected.std(0, unbiased=False).clamp_min(1e-6))

    def test_variable_length_batch_matches_native_forward_without_network_changes(self):
        torch.manual_seed(1)
        model = VelocityPredictorGRU(hidden_dim=8).eval()
        sequences = [torch.randn(2, 13), torch.randn(5, 13)]
        inputs, targets, lengths = collate_velocity_windows([(sequence, torch.zeros(3)) for sequence in sequences])
        with torch.no_grad():
            actual = predict_velocity_windows(model, inputs, lengths)
            expected = torch.cat([model(sequence[None]) for sequence in sequences])
            torch.testing.assert_close(actual, expected)
            inputs[0, 2:] = 1e8
            torch.testing.assert_close(predict_velocity_windows(model, inputs, lengths), expected)
        self.assertEqual(targets.shape, (2, 3))

    def test_episode_splits_are_stratified_disjoint_and_deterministic(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario in ("ss", "sd"):
                for episode in range(5):
                    for chunk in range(2):
                        write_chunk(root, payload(f"run:{scenario}:{episode}", scenario), chunk)
            files = discover_chunks([root, root])
            self.assertEqual(len(files), 20)
            splits, info = split_episode_files(files, 0.6, 0.2, 42)
            self.assertEqual(splits, split_episode_files(files, 0.6, 0.2, 42)[0])
            episodes = {split: {chunk_identity(path)[2] for path in paths} for split, paths in splits.items()}
            for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
                self.assertFalse(episodes[first] & episodes[second])
            self.assertEqual(info["episode_counts"]["train"], {"ss": 3, "sd": 3})
            self.assertEqual(info["episode_counts"]["val"], {"ss": 1, "sd": 1})
            self.assertEqual(info["episode_counts"]["test"], {"ss": 1, "sd": 1})

    def test_v1_and_incompatible_labels_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for field in ("schema", "labels", "heading", "episode"):
                data = payload()
                if field == "schema":
                    data["schema_version"] = "taq_velocity_v1"
                elif field == "labels":
                    data["metadata"]["label_semantics"] = "rotating_frame_derivative"
                elif field == "heading":
                    data["metadata"]["angular_velocity_z_semantics"] = "gyro_z"
                else:
                    data["metadata"]["episode_uid"] = ""
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        chunk_identity(write_chunk(root, data))

    def test_holes_in_padding_mask_are_rejected(self):
        with TemporaryDirectory() as directory:
            data = payload()
            data["sequence_valid_mask"][0] = torch.tensor([True, False, True, True, False])
            data["target_valid_mask"][0, 1] = False
            with self.assertRaisesRegex(ValueError, "prefix"):
                TAQLegacyGRUDataset(write_chunk(Path(directory), data))

    def test_evaluation_scores_one_prediction_per_window(self):
        with TemporaryDirectory() as directory:
            dataset = TAQLegacyGRUDataset(write_chunk(Path(directory), payload()))
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_velocity_windows)
            result = evaluate(VelocityPredictorGRU(hidden_dim=8), loader, torch.device("cpu"), WeightedSmoothL1Loss())
            self.assertEqual(result["sample_count"], 2)
            self.assertTrue(result["err_norm_p95"] >= result["err_norm_mean"])

    def test_prepare_and_synthetic_training_cli(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            for scenario in ("ss", "sd"):
                for episode in range(5):
                    write_chunk(data_root, payload(f"run:{scenario}:{episode}", scenario))
            script = TEST11_DIR / "speed_prediction/legacy_gru/train_gru_velocity.py"
            arguments = [sys.executable, str(script), "--data_dirs", str(data_root),
                         "--train_ratio", "0.6", "--test_ratio", "0.2", "--device", "cpu",
                         "--num_workers", "0", "--hidden_dim", "8", "--batch_size", "4", "--epochs", "1"]
            prepared_root = root / "prepared"
            prepared = subprocess.run(arguments + ["--save_dir", str(prepared_root), "--prepare_only"],
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            self.assertFalse(list(prepared_root.rglob("*.pt")))
            manifest = json.loads(next(prepared_root.rglob("data_manifest.json")).read_text())
            self.assertEqual(manifest["sample_counts"], {"train": 12, "val": 4, "test": 4})
            trained_root = root / "synthetic_train"
            trained = subprocess.run(arguments + ["--save_dir", str(trained_root)],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            checkpoint = torch.load(next(trained_root.rglob("best_model.pt")), map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["input_dim"], 13)
            self.assertEqual(checkpoint["label_semantics"], TAQ_LABEL_SEMANTICS)
            metrics = json.loads(next(trained_root.rglob("test_metrics.json")).read_text())
            self.assertEqual(metrics["sample_count"], 4)

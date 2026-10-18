from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.utils.data import TensorDataset


TEST11_DIR = Path(__file__).resolve().parents[2]
if str(TEST11_DIR.parent) not in sys.path:
    sys.path.insert(0, str(TEST11_DIR.parent))

from test11.speed_prediction.common.time_aware_velocity_data import (
    LEGACY_INPUT_TRANSFORM_VERSION,
    FeatureIndex,
    TAQVelocityChunkDataset,
    TrackerSource,
    build_taq_dataset,
    build_time_aware_features,
    collate_taq_velocity_batch,
    compute_taq_feature_norm_stats,
    load_torch_payload,
    save_taq_velocity_chunk,
    robust_finite_difference,
    TAQ_LABEL_SEMANTICS,
    TAQ_BASELINE_VERSION,
    TAQ_SCHEMA_VERSION,
    TAQ_STREAM_FORMAT,
    relative_translation_velocity_yaw_base,
    target_position_derivative_base_mps,
    relative_velocity_base_to_world_mps,
    weighted_velocity_baseline,
)
from test11.speed_prediction.taq_gru.time_aware_velocity_model import TimeAwareQualityGatedGRU
from test11.speed_prediction.taq_gru.online_taq_velocity import OnlineTAQVelocityEstimator, TAQObservation
from test11.speed_prediction.taq_gru.taq_velocity_training import masked_taq_velocity_loss
from test11.speed_prediction.taq_gru.train_stage2_balanced import DataGroup, build_balanced_sampler, parse_env_spec
from test11.speed_prediction.taq_gru.train_taq_velocity import update_top_p95_checkpoints
from test11.speed_prediction.taq_gru.evaluate_taq_velocity_streaming import evaluate_streaming


class TimeAwareVelocityTests(unittest.TestCase):
    def test_relative_velocity_label_is_physical_not_rotating_frame_derivative(self) -> None:
        target = torch.tensor([[0.0, 0.0, 0.0]])
        base = torch.tensor([[0.0, 0.0, 0.0]])
        relative = relative_translation_velocity_yaw_base(target, base, torch.tensor([0.7]))
        self.assertTrue(torch.allclose(relative, torch.zeros_like(relative)))

        base = torch.tensor([[1.5, -0.5, 0.0]])
        relative = relative_translation_velocity_yaw_base(target, base, 0.0)
        self.assertTrue(torch.allclose(relative, -base))
        position = torch.tensor([[100.0, 0.0, 0.0]])
        derivative = target_position_derivative_base_mps(relative, torch.tensor([0.2]), position)
        self.assertTrue(torch.allclose(derivative, torch.tensor([[-1.5, -19.5, 0.0]])))

    def test_relative_velocity_world_round_trip(self) -> None:
        target = torch.tensor([[4.0, -2.0, 0.5]])
        base_world = torch.tensor([[1.0, 0.5, -0.2]])
        yaw = torch.tensor([0.73])
        relative = relative_translation_velocity_yaw_base(target, base_world, yaw)
        base_velocity_base = relative_translation_velocity_yaw_base(
            base_world, torch.zeros_like(base_world), yaw
        )
        restored = relative_velocity_base_to_world_mps(relative, base_velocity_base, yaw)
        self.assertLess(float((restored - target).abs().max()), 1e-4)
    def test_stage2_balanced_sampler_uses_target_mass_not_dataset_size(self) -> None:
        groups = [
            DataGroup(
                key="ideal/ss",
                domain="ideal",
                scenario="ss",
                profile="ideal",
                dataset=TensorDataset(torch.arange(1000)),
                target_mass=0.45,
                paths=(),
            ),
            DataGroup(
                key="stage2/ss/mixed",
                domain="stage2",
                scenario="ss",
                profile="mixed",
                dataset=TensorDataset(torch.arange(10)),
                target_mass=0.55,
                paths=(),
            ),
        ]
        _, sampler = build_balanced_sampler(groups, samples_per_epoch=20_000, seed=42)
        stage2_fraction = sum(index >= 1000 for index in sampler) / 20_000
        self.assertAlmostEqual(stage2_fraction, 0.55, delta=0.015)

    def test_stage2_environment_range_parser(self) -> None:
        self.assertEqual(parse_env_spec("0-2,5,7-8"), (0, 1, 2, 5, 7, 8))
        with self.assertRaises(ValueError):
            parse_env_spec("3-1")

    def _features(self):
        timestamps = torch.tensor([0.0, 0.02, 0.06, 0.12, 0.20])
        position = torch.stack((1.5 * timestamps, -2.0 * timestamps, 0.5 * timestamps), dim=-1)
        zeros = torch.zeros_like(position)
        features, valid = build_time_aware_features(
            timestamps_s=timestamps,
            position_base_m=position,
            robot_linear_velocity_base_mps=zeros,
            robot_angular_velocity_base_rps=zeros,
            source_ids=torch.tensor(
                [
                    int(TrackerSource.INITIAL),
                    int(TrackerSource.FLOW),
                    int(TrackerSource.PREDICTION),
                    int(TrackerSource.OSTRACK),
                    int(TrackerSource.FLOW),
                ]
            ),
            quality=torch.ones((5, 7)),
        )
        return timestamps, features, valid

    def test_feature_builder_uses_actual_time_delta(self) -> None:
        _, features, valid = self._features()
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(features[1:, 3:6], torch.tensor([[1.5, -2.0, 0.5]]).repeat(4, 1)))
        self.assertAlmostEqual(float(features[2, FeatureIndex.DT]), 0.04, places=6)
        self.assertEqual(float(features[2, FeatureIndex.OBSERVED_MASK]), 0.0)
        self.assertEqual(float(features[3, FeatureIndex.OBSERVED_MASK]), 1.0)

    def test_feature_builder_preserves_dt_at_large_absolute_timestamp(self) -> None:
        timestamps = [10_000.0, 10_000.015, 10_000.035]
        position = torch.zeros((3, 3), dtype=torch.float32)
        features, _ = build_time_aware_features(
            timestamps_s=timestamps,
            position_base_m=position,
            robot_linear_velocity_base_mps=position,
            robot_angular_velocity_base_rps=position,
            source_ids=[
                int(TrackerSource.INITIAL),
                int(TrackerSource.FLOW),
                int(TrackerSource.FLOW),
            ],
        )
        self.assertAlmostEqual(float(features[1, FeatureIndex.DT]), 0.015, places=7)
        self.assertAlmostEqual(float(features[2, FeatureIndex.DT]), 0.020, places=7)
        self.assertEqual(features.dtype, torch.float32)

    def test_weighted_baseline_recovers_constant_velocity(self) -> None:
        _, features, valid = self._features()
        baseline = weighted_velocity_baseline(features.unsqueeze(0), valid.unsqueeze(0))
        expected = torch.tensor([1.5, -2.0, 0.5])
        self.assertTrue(torch.allclose(baseline[0, -1], expected, atol=1e-5))

    def test_weighted_baseline_removes_yaw_rotation_pseudo_velocity(self) -> None:
        yaw_rate = 0.1
        timestamps = torch.arange(32, dtype=torch.float64) * 0.015
        position = torch.stack(
            (
                100.0 * torch.cos(yaw_rate * timestamps),
                -100.0 * torch.sin(yaw_rate * timestamps),
                torch.zeros_like(timestamps),
            ),
            dim=-1,
        ).to(torch.float32)
        zeros = torch.zeros_like(position)
        angular = torch.zeros_like(position)
        angular[:, 2] = yaw_rate
        features, valid = build_time_aware_features(
            timestamps_s=timestamps,
            position_base_m=position,
            robot_linear_velocity_base_mps=zeros,
            robot_angular_velocity_base_rps=angular,
            source_ids=torch.full((32,), int(TrackerSource.OSTRACK)),
            quality=torch.ones((32, 7)),
        )
        baseline = weighted_velocity_baseline(features.unsqueeze(0), valid.unsqueeze(0))[0, -1]
        self.assertLess(float(torch.linalg.vector_norm(baseline)), 0.3)

    def test_robust_finite_difference_vector_clip_and_reliability(self) -> None:
        _, features, _ = self._features()
        features[2, 3:6] = torch.tensor([60.0, 80.0, 0.0])
        clipped, reliability = robust_finite_difference(features, clip_mps=30.0)
        self.assertTrue(torch.allclose(clipped[2], torch.tensor([18.0, 24.0, 0.0])))
        self.assertAlmostEqual(float(reliability[2]), 0.3, places=6)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(clipped[2])), 30.0, places=5)
        self.assertEqual(float(reliability[1]), 1.0)

    def test_robust_baseline_reduces_position_jump_effect(self) -> None:
        timestamps = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
        # Put the jump away from the exact center: a centered single spike is
        # an accidental least-squares symmetry whose slope cancels even
        # without robust weights.
        position = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]])
        zeros = torch.zeros_like(position)
        features, valid = build_time_aware_features(
            timestamps_s=timestamps,
            position_base_m=position,
            robot_linear_velocity_base_mps=zeros,
            robot_angular_velocity_base_rps=zeros,
            source_ids=torch.full((5,), int(TrackerSource.FLOW)),
            quality=torch.ones((5, 7)),
        )
        robust = weighted_velocity_baseline(features.unsqueeze(0), valid.unsqueeze(0))[0, -1]
        legacy = weighted_velocity_baseline(
            features.unsqueeze(0),
            valid.unsqueeze(0),
            finite_difference_clip_mps=None,
        )[0, -1]
        expected = torch.tensor([1.0, 0.0, 0.0])
        self.assertLess(torch.linalg.vector_norm(robust - expected), torch.linalg.vector_norm(legacy - expected))

    def test_model_handles_padding_and_returns_uncertainty(self) -> None:
        _, features, valid = self._features()
        padded_features = torch.zeros((2, 7, features.shape[-1]))
        padded_sources = torch.full((2, 7), int(TrackerSource.INVALID), dtype=torch.long)
        padded_mask = torch.zeros((2, 7), dtype=torch.bool)
        padded_features[0, :5] = features
        padded_sources[0, :5] = torch.tensor([0, 1, 3, 2, 1])
        padded_mask[0, :5] = valid
        padded_features[1, :3] = features[:3]
        padded_sources[1, :3] = torch.tensor([0, 1, 3])
        padded_mask[1, :3] = valid[:3]
        model = TimeAwareQualityGatedGRU(dropout=0.0)
        mean, log_sigma, aux = model(padded_features, padded_sources, padded_mask, return_aux=True)
        self.assertEqual(tuple(mean.shape), (2, 7, 3))
        self.assertEqual(tuple(log_sigma.shape), (2, 7, 3))
        self.assertEqual(tuple(aux["reliability_gate"].shape), (2, 7, 1))
        self.assertEqual(tuple(aux["finite_difference_reliability"].shape), (2, 7, 1))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(log_sigma).all())

    def test_model_gate_is_bounded_by_finite_difference_reliability(self) -> None:
        _, features, valid = self._features()
        features[2, 3:6] = torch.tensor([60.0, 80.0, 0.0])
        sources = torch.tensor([[0, 1, 3, 2, 1]])
        model = TimeAwareQualityGatedGRU(dropout=0.0)
        _, _, aux = model(features.unsqueeze(0), sources, valid.unsqueeze(0), return_aux=True)
        reliability = aux["finite_difference_reliability"][0, 2, 0]
        gate = aux["reliability_gate"][0, 2, 0]
        self.assertAlmostEqual(float(reliability), 0.3, places=6)
        self.assertLessEqual(float(gate), float(reliability) + 1e-7)

    def test_chunk_schema_and_collation(self) -> None:
        _, features, valid = self._features()
        with TemporaryDirectory() as directory:
            chunk_path = Path(directory) / "chunk.pt"
            save_taq_velocity_chunk(
                chunk_path,
                sequence_features=features.unsqueeze(0),
                sequence_source_ids=torch.tensor([[0, 1, 3, 2, 1]]),
                sequence_valid_mask=valid.unsqueeze(0),
                target_velocity_base=torch.tensor([[[1.5, -2.0, 0.5]] * 5]),
                target_valid_mask=torch.ones((1, 5), dtype=torch.bool),
                sample_window_start_s=torch.tensor([10.0], dtype=torch.float64),
                sample_window_end_s=torch.tensor([10.2], dtype=torch.float64),
                metadata={"coordinate_frame": "yaw_only_base", "scenario": "ss"},
            )
            payload = load_torch_payload(chunk_path)
            self.assertEqual(payload["sample_window_start_s"].dtype, torch.float64)
            self.assertEqual(payload["sample_window_start_s"].tolist(), [10.0])
            self.assertEqual(payload["sample_window_end_s"].tolist(), [10.2])
            dataset = TAQVelocityChunkDataset(chunk_path)
            batch = collate_taq_velocity_batch([dataset[0], dataset[0]])
            self.assertEqual(tuple(batch["features"].shape), (2, 5, 24))
            self.assertEqual(int(batch["target_valid_mask"].sum()), 10)

    def test_chunk_rejects_invalid_window_timestamps(self) -> None:
        _, features, valid = self._features()
        with TemporaryDirectory() as directory:
            common = dict(
                sequence_features=features.unsqueeze(0),
                sequence_source_ids=torch.tensor([[0, 1, 3, 2, 1]]),
                sequence_valid_mask=valid.unsqueeze(0),
                target_velocity_base=torch.tensor([[[1.5, -2.0, 0.5]] * 5]),
                target_valid_mask=torch.ones((1, 5), dtype=torch.bool),
            )
            with self.assertRaisesRegex(ValueError, "provided together"):
                save_taq_velocity_chunk(
                    Path(directory) / "missing_end.pt",
                    sample_window_start_s=[0.0],
                    **common,
                )
            with self.assertRaisesRegex(ValueError, "later than its start"):
                save_taq_velocity_chunk(
                    Path(directory) / "reversed.pt",
                    sample_window_start_s=[1.0],
                    sample_window_end_s=[0.5],
                    **common,
                )

    def test_normalization_stats_use_clipped_finite_difference(self) -> None:
        features = torch.zeros((1, 1, 24), dtype=torch.float32)
        features[0, 0, 3:6] = torch.tensor([60.0, 80.0, 0.0])
        valid = torch.ones((1, 1), dtype=torch.bool)
        with TemporaryDirectory() as directory:
            chunk_path = Path(directory) / "outlier.pt"
            save_taq_velocity_chunk(
                chunk_path,
                sequence_features=features,
                sequence_source_ids=torch.tensor([[int(TrackerSource.FLOW)]]),
                sequence_valid_mask=valid,
                target_velocity_base=torch.zeros((1, 1, 3)),
                target_valid_mask=torch.ones((1, 1), dtype=torch.bool),
                metadata={"scenario": "ss"},
            )
            mean, _ = compute_taq_feature_norm_stats(TAQVelocityChunkDataset(chunk_path))
        self.assertTrue(torch.allclose(mean[3:6], torch.tensor([18.0, 24.0, 0.0])))

    def test_legacy_model_architecture_remains_available(self) -> None:
        legacy = TimeAwareQualityGatedGRU(
            dropout=0.0,
            input_transform_version=LEGACY_INPUT_TRANSFORM_VERSION,
        )
        config = legacy.config
        restored = TimeAwareQualityGatedGRU(**config)
        restored.load_state_dict(legacy.state_dict(), strict=True)
        self.assertFalse(restored.robust_input_enabled)

    def test_dataset_discovers_episode_subdirectories(self) -> None:
        _, features, valid = self._features()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            chunk_path = root / "episodes" / "env_000" / "episode_000001" / "chunk.pt"
            save_taq_velocity_chunk(
                chunk_path,
                sequence_features=features.unsqueeze(0),
                sequence_source_ids=torch.tensor([[0, 1, 3, 2, 1]]),
                sequence_valid_mask=valid.unsqueeze(0),
                target_velocity_base=torch.tensor([[[1.5, -2.0, 0.5]] * 5]),
                target_valid_mask=torch.ones((1, 5), dtype=torch.bool),
                metadata={"scenario": "ss"},
            )
            self.assertEqual(len(build_taq_dataset([root])), 1)

    def test_top_p95_checkpoints_keep_the_correct_epoch_payloads(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            records: list[tuple[float, int, Path]] = []
            for epoch, score in ((1, 0.30), (2, 0.50), (3, 0.20), (4, 0.40)):
                records = update_top_p95_checkpoints(
                    payload={"epoch": epoch},
                    macro_p95=score,
                    epoch=epoch,
                    output_dir=output_dir,
                    records=records,
                )
            self.assertEqual([(score, epoch) for score, epoch, _ in records], [(0.20, 3), (0.30, 1), (0.40, 4)])
            for _, epoch, path in records:
                self.assertEqual(load_torch_payload(path)["epoch"], epoch)
            self.assertFalse((output_dir / "p95_epoch_002.pt").exists())

    def test_streaming_evaluator_requires_and_uses_complete_episode_streams(self) -> None:
        _, features, valid = self._features()
        model = TimeAwareQualityGatedGRU(hidden_dim=32, dropout=0.0)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            torch.save(
                {
                    "schema_version": TAQ_SCHEMA_VERSION,
                    "label_semantics": TAQ_LABEL_SEMANTICS,
                    "baseline_version": model.baseline_version,
                    "model_config": model.config,
                    "model_state_dict": model.state_dict(),
                    "input_norm": {"mean": torch.zeros(24), "std": torch.ones(24)},
                    "train_args": {"baseline_window_s": 0.48},
                },
                checkpoint,
            )
            timestamps = torch.tensor([0.0, 0.02, 0.06, 0.12, 0.20], dtype=torch.float64)
            for scenario in ("ss", "sd"):
                stream_path = root / scenario / "episode.taqstream"
                stream_path.parent.mkdir(parents=True)
                torch.save(
                    {
                        "schema_version": TAQ_SCHEMA_VERSION,
                        "stream_format": TAQ_STREAM_FORMAT,
                        "label_semantics": TAQ_LABEL_SEMANTICS,
                        "timestamps_s": timestamps,
                        "sequence_features": features,
                        "sequence_source_ids": torch.tensor([0, 1, 3, 2, 1]),
                        "sequence_valid_mask": valid,
                        "target_velocity_base": torch.zeros((5, 3)),
                        "target_valid_mask": torch.tensor([False, False, False, True, True]),
                        "state_reset_mask": torch.tensor([True, False, False, False, False]),
                        "metadata": {"scenario": scenario},
                    },
                    stream_path,
                )
            single_scenario = evaluate_streaming(checkpoint, root / "ss", torch.device("cpu"))
            self.assertEqual(single_scenario["episodes"], 1)
            self.assertEqual(single_scenario["streaming_steps"], 2)
            self.assertIn("err_norm_p95_ss", single_scenario)
            self.assertNotIn("err_norm_p95_sd", single_scenario)
            metrics = evaluate_streaming(checkpoint, root, torch.device("cpu"))
            self.assertEqual(metrics["episodes"], 2)
            self.assertEqual(metrics["streaming_steps"], 4)
            self.assertIn("err_norm_p95_ss", metrics)
            self.assertIn("err_norm_p95_sd", metrics)

    def test_feature_builder_accepts_persistent_age_anchors(self) -> None:
        timestamps = torch.tensor([1.0, 1.05], dtype=torch.float64)
        zeros = torch.zeros((2, 3))
        features, _ = build_time_aware_features(
            timestamps_s=timestamps,
            position_base_m=zeros,
            robot_linear_velocity_base_mps=zeros,
            robot_angular_velocity_base_rps=zeros,
            source_ids=[int(TrackerSource.PREDICTION)] * 2,
            initial_last_observation_time_s=0.9,
            initial_last_fine_time_s=0.8,
        )
        self.assertTrue(torch.allclose(features[:, FeatureIndex.OBSERVATION_AGE], torch.tensor([0.1, 0.15]), atol=1e-6))
        self.assertTrue(torch.allclose(features[:, FeatureIndex.FINE_AGE], torch.tensor([0.2, 0.25]), atol=1e-6))

    def test_online_features_match_offline_frame_by_frame(self) -> None:
        timestamps, expected, _ = self._features()
        model = TimeAwareQualityGatedGRU(hidden_dim=16, dropout=0.0)
        captured: list[torch.Tensor] = []
        original_step = model.step

        def capture_step(features, source_ids, **kwargs):
            captured.append(features.detach().cpu().clone())
            return original_step(features, source_ids, **kwargs)

        model.step = capture_step
        estimator = OnlineTAQVelocityEstimator(
            model, device="cpu", history_window_s=1.0, max_history_steps=32
        )
        source_names = ("initial", "flow", "prediction", "ostrack", "flow")
        for index, timestamp in enumerate(timestamps.tolist()):
            estimator.update(
                3,
                TAQObservation(
                    timestamp_s=timestamp,
                    position_base_m=tuple(expected[index, :3].tolist()),
                    robot_linear_velocity_base_mps=tuple(expected[index, 6:9].tolist()),
                    robot_angular_velocity_base_rps=tuple(expected[index, 9:12].tolist()),
                    source=source_names[index],
                    quality=tuple(expected[index, 15:22].tolist()),
                ),
            )
        online_rows = torch.cat(captured, dim=0)
        self.assertTrue(torch.allclose(online_rows, expected, atol=1e-6, rtol=0.0))

    def test_online_age_survives_history_cropping(self) -> None:
        model = TimeAwareQualityGatedGRU(hidden_dim=16, dropout=0.0, baseline_window_s=0.04)
        captured: list[torch.Tensor] = []
        original_step = model.step

        def capture_step(features, source_ids, **kwargs):
            captured.append(features.detach().cpu().clone())
            return original_step(features, source_ids, **kwargs)

        model.step = capture_step
        estimator = OnlineTAQVelocityEstimator(
            model, device="cpu", history_window_s=0.04, max_history_steps=2
        )
        for timestamp, source in ((0.0, "flow"), (0.05, "prediction"), (0.10, "prediction"), (0.15, "prediction")):
            estimator.update(
                7,
                TAQObservation(
                    timestamp_s=timestamp,
                    position_base_m=(timestamp, 0.0, 0.0),
                    robot_linear_velocity_base_mps=(0.0, 0.0, 0.0),
                    robot_angular_velocity_base_rps=(0.0, 0.0, 0.0),
                    source=source,
                ),
            )
        self.assertAlmostEqual(
            float(captured[-1][0, FeatureIndex.OBSERVATION_AGE]), 0.15, places=6
        )
        self.assertEqual(len(estimator._states[7].observations), 2)

    def test_loss_masks_unlabeled_steps(self) -> None:
        prediction = torch.zeros((1, 3, 3))
        target = torch.zeros_like(prediction)
        target[:, 0] = 100.0
        log_sigma = torch.zeros_like(prediction)
        mask = torch.tensor([[False, False, True]])
        loss, parts = masked_taq_velocity_loss(prediction, log_sigma, target, mask)
        self.assertLess(float(parts["huber"]), 1e-8)
        self.assertTrue(torch.isfinite(loss))

    def test_online_lost_preserves_state_and_hard_reset_clears_it(self) -> None:
        model = TimeAwareQualityGatedGRU(hidden_dim=16, dropout=0.0)
        estimator = OnlineTAQVelocityEstimator(model, device="cpu", history_window_s=0.5)
        common = dict(
            robot_linear_velocity_base_mps=(0.0, 0.0, 0.0),
            robot_angular_velocity_base_rps=(0.0, 0.0, 0.0),
        )
        estimator.update(
            9,
            TAQObservation(
                timestamp_s=0.0,
                position_base_m=(1.0, 0.0, 0.0),
                source="flow",
                **common,
            ),
        )
        hidden_before_lost = estimator._states[9].hidden.clone()
        estimator.update(
            9,
            TAQObservation(
                timestamp_s=0.1,
                position_base_m=(0.0, 0.0, 0.0),
                source="lost",
                range_valid=False,
                **common,
            ),
        )
        self.assertEqual(len(estimator._states[9].observations), 2)
        self.assertFalse(torch.equal(estimator._states[9].hidden, hidden_before_lost))
        estimator.reset(9)
        self.assertNotIn(9, estimator._states)

    def test_online_estimator_keeps_independent_target_state(self) -> None:
        model = TimeAwareQualityGatedGRU(hidden_dim=32, dropout=0.0)
        estimator = OnlineTAQVelocityEstimator(model, device="cpu", history_window_s=0.5)
        common = dict(
            robot_linear_velocity_base_mps=(0.0, 0.0, 0.0),
            robot_angular_velocity_base_rps=(0.0, 0.0, 0.0),
            source="flow",
            quality=(0.9, 0.8, 0.8, 0.8, 0.2, 0.1, 0.9),
        )
        first = estimator.update(
            1,
            TAQObservation(timestamp_s=0.0, position_base_m=(1.0, 0.0, 0.0), **common),
        )
        second = estimator.update(
            1,
            TAQObservation(timestamp_s=0.02, position_base_m=(1.02, 0.0, 0.0), **common),
        )
        other = estimator.update(
            2,
            TAQObservation(timestamp_s=0.02, position_base_m=(3.0, 0.0, 0.0), **common),
        )

        self.assertEqual(tuple(first.velocity_mean_mps.shape), (3,))
        self.assertEqual(tuple(second.velocity_log_sigma.shape), (3,))
        self.assertEqual(other.target_id, 2)
        self.assertEqual(len(estimator._states), 2)


if __name__ == "__main__":
    unittest.main()

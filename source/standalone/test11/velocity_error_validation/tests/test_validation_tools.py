from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_error_sources import (  # noqa: E402
    _derivative,
    _evaluation_started_mask,
    _paired_initialization_starts,
    metric_values,
    paired_effects,
    validate_pair_coverage,
    validate_trajectory_consistency,
)
from collect_paired_conditions import _validate_paired_state_rows  # noqa: E402
from validation_io import load_audit_rows  # noqa: E402
from speed_prediction.time_aware_velocity_data import (  # noqa: E402
    FeatureIndex, TrackerSource, TAQ_LABEL_SEMANTICS, TAQ_SCHEMA_VERSION, TAQ_STREAM_FORMAT,
)


def _metric_row(condition: str, pair: str, error: float) -> dict[str, object]:
    return {
        "condition": condition,
        "trajectory_uid": pair,
        "stream_uid": f"{pair}:{condition}",
        "episode_uid": f"episode:{pair}:{condition}",
        "valid": 1,
        "error_norm": error,
    }


class ValidationToolTests(unittest.TestCase):
    def test_initialization_is_excluded_but_later_loss_is_scored(self) -> None:
        features = torch.zeros((6, 24), dtype=torch.float32)
        features[2, FeatureIndex.OBSERVED_MASK] = 1.0
        features[3, FeatureIndex.OBSERVED_MASK] = 1.0
        features[3, FeatureIndex.RANGE_VALID_MASK] = 1.0
        sources = torch.tensor([
            TrackerSource.INVALID, TrackerSource.INVALID, TrackerSource.INITIAL,
            TrackerSource.FLOW, TrackerSource.INVALID, TrackerSource.INVALID,
        ])
        started = _evaluation_started_mask(
            features, sources, torch.ones(6, dtype=torch.bool),
            torch.arange(6, dtype=torch.float64) * 0.02,
        )
        self.assertEqual(started.tolist(), [False, False, False, True, True, True])
        rows = [
            {"valid": int(is_started), "error_norm": error}
            for is_started, error in zip(started.tolist(), (8.0, 9.0, 10.0, 2.0, 3.0, 4.0))
        ]
        result = metric_values(rows)
        self.assertEqual(result["valid_frames"], 3)
        self.assertAlmostEqual(result["mean"], 3.0)

    def test_paired_conditions_share_start_and_keep_later_missing_frames(self) -> None:
        timestamps = torch.arange(6, dtype=torch.float64) * 0.02

        def payload(first_observation: int, condition: str) -> dict[str, object]:
            features = torch.zeros((6, 24), dtype=torch.float32)
            features[first_observation, FeatureIndex.OBSERVED_MASK] = 1.0
            features[first_observation, FeatureIndex.RANGE_VALID_MASK] = 1.0
            sources = torch.full((6,), int(TrackerSource.INVALID), dtype=torch.long)
            sources[first_observation] = int(TrackerSource.INITIAL)
            return {
                "schema_version": TAQ_SCHEMA_VERSION,
                "stream_format": TAQ_STREAM_FORMAT,
                "label_semantics": TAQ_LABEL_SEMANTICS,
                "sequence_features": features,
                "sequence_source_ids": sources,
                "sequence_valid_mask": torch.ones(6, dtype=torch.bool),
                "target_velocity_base": torch.zeros((6, 3), dtype=torch.float32),
                "target_valid_mask": torch.ones(6, dtype=torch.bool),
                "state_reset_mask": torch.tensor([True, False, False, False, False, False]),
                "timestamps_s": timestamps,
                "metadata": {"trajectory_uid": "same-trajectory", "condition": condition},
            }

        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "A.taqstream"
            c = Path(directory) / "C.taqstream"
            torch.save(payload(2, "A"), a)
            torch.save(payload(4, "C"), c)
            starts = _paired_initialization_starts([a, c])
            self.assertAlmostEqual(starts[a], 0.04)
            self.assertAlmostEqual(starts[c], 0.04)
            c_payload = payload(4, "C")
            started = _evaluation_started_mask(
                c_payload["sequence_features"], c_payload["sequence_source_ids"],
                c_payload["sequence_valid_mask"], timestamps,
                initialization_start_s=starts[c],
            )
            self.assertEqual(started.tolist(), [False, False, True, True, True, True])

    def test_vector_derivative_returns_one_norm_per_frame(self) -> None:
        values = np.asarray([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]])
        result = _derivative(values, np.asarray([0.0, 1.0, 3.0]))
        np.testing.assert_allclose(result[1:], [5.0, 2.0])
        self.assertTrue(np.isnan(result[0]))

    def test_jsonl_audit_is_loaded_line_by_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            rows = [
                {"episode_uid": "ep", "frame_index": 0, "value": 1},
                {"episode_uid": "ep", "frame_index": 1, "value": 2},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            loaded = load_audit_rows(path)
            self.assertEqual(loaded[("ep", 0)]["value"], 1)
            self.assertEqual(loaded[("ep", 1)]["value"], 2)

    def test_camera_and_perception_factorial_effects(self) -> None:
        rows = [
            _metric_row("A", "body", 1.0),
            _metric_row("B", "body", 3.0),
            _metric_row("C", "body", 2.0),
            _metric_row("D", "body", 5.0),
            _metric_row("P0", "perception", 4.0),
            _metric_row("P1", "perception", 1.0),
            _metric_row("P2", "perception", 2.0),
            _metric_row("P3", "perception", 0.5),
        ]
        coverage = validate_pair_coverage(rows)
        self.assertEqual(coverage["complete_pairs"], 2)
        effects = {
            (item["effect"], item["metric"]): item["mean_difference"]
            for item in paired_effects(rows, bootstrap=0, seed=1)
        }
        self.assertAlmostEqual(effects[("tracking_fixed", "p95")], 2.0)
        self.assertAlmostEqual(effects[("camera_motion_with_oracle_tracking", "p95")], 1.0)
        self.assertAlmostEqual(effects[("camera_motion_with_real_tracking", "p95")], 2.0)
        self.assertAlmostEqual(effects[("camera_tracking_interaction", "p95")], 1.0)
        self.assertAlmostEqual(effects[("direction_path", "p95")], 3.0)
        self.assertAlmostEqual(effects[("range_path", "p95")], 2.0)
        self.assertAlmostEqual(effects[("direction_range_interaction", "p95")], 1.5)

    def test_trajectory_audit_detects_mismatched_replay(self) -> None:
        rows: list[dict[str, object]] = []
        for condition in ("A", "B", "C", "D"):
            for frame, timestamp in enumerate((0.0, 0.02)):
                rows.append(
                    {
                        **_metric_row(condition, "trajectory", 1.0),
                        "frame_index": frame,
                        "timestamp_s": timestamp,
                        "target_position_world_m": [timestamp, 0.0, 1.0],
                    }
                )
        result = validate_trajectory_consistency(rows)
        self.assertEqual(result["checked_pairs"], 1)
        rows[-1]["target_position_world_m"] = [1.0, 0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "same target trajectory"):
            validate_trajectory_consistency(rows)

    def test_incomplete_pairs_are_reported_but_never_enter_effects(self) -> None:
        rows = [
            _metric_row("A", "complete", 1.0),
            _metric_row("B", "complete", 3.0),
            _metric_row("C", "complete", 2.0),
            _metric_row("D", "complete", 5.0),
            _metric_row("C", "incomplete", 10.0),
            _metric_row("D", "incomplete", 100.0),
        ]
        coverage = validate_pair_coverage(rows)
        self.assertEqual(coverage["complete_pairs"], 1)
        self.assertEqual(coverage["excluded_incomplete_pairs"], 1)
        effects = {
            (item["effect"], item["metric"]): item
            for item in paired_effects(rows, bootstrap=0, seed=1)
        }
        tracking_moving = effects[("tracking_moving", "p95")]
        self.assertEqual(tracking_moving["pairs"], 1)
        self.assertAlmostEqual(tracking_moving["mean_difference"], 3.0)

    def test_paired_state_validation_checks_camera_and_root_interventions(self) -> None:
        rows: dict[str, list[dict[str, object]]] = {}
        for condition in ("A", "B", "C", "D"):
            condition_rows: list[dict[str, object]] = []
            for step in (10, 11):
                mounted_position = [0.01 * step, 0.0, 1.0]
                fixed = condition in {"A", "B"}
                condition_rows.append(
                    {
                        "physics_step_index": step,
                        "physics_timestamp_s": 0.02 * step,
                        "camera_mode": (
                            "fixed_world_proxy" if fixed else "torso_mounted_proxy"
                        ),
                        "target_position_world_m": [100.0 + 0.1 * step, 0.0, 1.0],
                        "base_position_world_m": [0.01 * step, 0.0, 0.5],
                        "base_quaternion_world": [1.0, 0.0, 0.0, 0.0],
                        "robot_linear_velocity_base_mps": [0.01, 0.0, 0.0],
                        "robot_angular_velocity_base_rps": [0.0, 0.0, 0.0],
                        "camera_position_world_m": (
                            [0.0, 0.0, 1.0] if fixed else mounted_position
                        ),
                        "camera_quaternion_world": [1.0, 0.0, 0.0, 0.0],
                        "mounted_camera_position_world_m": mounted_position,
                        "mounted_camera_quaternion_world": [1.0, 0.0, 0.0, 0.0],
                        "background_global_flow_px": 0.0,
                    }
                )
            rows[condition] = condition_rows
        kwargs = {
            "trajectory_position_tolerance_m": 1e-4,
            "root_position_tolerance_m": 1e-4,
            "root_orientation_tolerance_rad": 1e-4,
            "root_velocity_tolerance": 1e-4,
            "fixed_camera_position_tolerance_m": 1e-5,
            "fixed_camera_orientation_tolerance_rad": 1e-4,
            "mounted_camera_position_tolerance_m": 1e-5,
            "mounted_camera_orientation_tolerance_rad": 1e-4,
            "fixed_camera_background_flow_median_px": 0.1,
            "fixed_camera_background_flow_p95_px": 0.5,
            "fixed_camera_background_flow_min_samples": 4,
        }
        result = _validate_paired_state_rows(rows, ("A", "B", "C", "D"), **kwargs)
        self.assertEqual(result["samples"], 2)
        rows["D"][1]["base_position_world_m"] = [1.0, 0.0, 0.5]
        with self.assertRaisesRegex(RuntimeError, "root_position_difference_m"):
            _validate_paired_state_rows(rows, ("A", "B", "C", "D"), **kwargs)

    def test_fixed_camera_flow_gate_skips_only_when_samples_are_insufficient(self) -> None:
        rows: dict[str, list[dict[str, object]]] = {}
        for condition in ("A", "B", "C", "D"):
            fixed = condition in {"A", "B"}
            rows[condition] = [
                {
                    "physics_step_index": 0,
                    "physics_timestamp_s": 0.0,
                    "camera_mode": "fixed_world_proxy" if fixed else "torso_mounted_proxy",
                    "target_position_world_m": [10.0, 0.0, 1.0],
                    "base_position_world_m": [0.0, 0.0, 0.5],
                    "base_quaternion_world": [1.0, 0.0, 0.0, 0.0],
                    "robot_linear_velocity_base_mps": [0.0, 0.0, 0.0],
                    "robot_angular_velocity_base_rps": [0.0, 0.0, 0.0],
                    "camera_position_world_m": [0.0, 0.0, 1.0],
                    "camera_quaternion_world": [1.0, 0.0, 0.0, 0.0],
                    "mounted_camera_position_world_m": [0.0, 0.0, 1.0],
                    "mounted_camera_quaternion_world": [1.0, 0.0, 0.0, 0.0],
                    "background_global_flow_px": 2.0 if fixed else None,
                }
            ]
        kwargs = {
            "trajectory_position_tolerance_m": 1e-4,
            "root_position_tolerance_m": 1e-4,
            "root_orientation_tolerance_rad": 1e-4,
            "root_velocity_tolerance": 1e-4,
            "fixed_camera_position_tolerance_m": 1e-5,
            "fixed_camera_orientation_tolerance_rad": 1e-4,
            "mounted_camera_position_tolerance_m": 1e-5,
            "mounted_camera_orientation_tolerance_rad": 1e-4,
            "fixed_camera_background_flow_median_px": 0.1,
            "fixed_camera_background_flow_p95_px": 0.5,
        }
        result = _validate_paired_state_rows(
            rows,
            ("A", "B", "C", "D"),
            fixed_camera_background_flow_min_samples=3,
            **kwargs,
        )
        self.assertEqual(result["fixed_camera_background_flow_samples"], 2)
        self.assertEqual(result["fixed_camera_background_flow_pairs_checked"], 0)
        with self.assertRaisesRegex(RuntimeError, "background flow median"):
            _validate_paired_state_rows(
                rows,
                ("A", "B", "C", "D"),
                fixed_camera_background_flow_min_samples=2,
                **kwargs,
            )


if __name__ == "__main__":
    unittest.main()

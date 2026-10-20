from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np

from test_lk_ostrack.isaacsim_dataset import (
    IsaacSimSequenceWriter,
    validate_collection_dataset,
    write_dataset_manifest,
)
from test_lk_ostrack.multi_target_io import load_initial_targets


def _ground_truth(target_count: int) -> dict[str, np.ndarray]:
    boxes = np.asarray(
        [[10.0 + 30.0 * index, 15.0, 30.0 + 30.0 * index, 35.0] for index in range(target_count)]
    )
    centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    positions = np.zeros((target_count, 3), dtype=np.float32)
    positions[:, 2] = 8.0
    return {
        "bbox_xyxy": boxes,
        "bbox_xyxy_raw": boxes.copy(),
        "center_uv": centers,
        "direction": np.zeros((target_count, 2), dtype=np.float32),
        "distance_m": np.full((target_count,), 8.0, dtype=np.float32),
        "lidar_range_m": np.full((target_count,), 8.0, dtype=np.float32),
        "lidar_valid": np.ones((target_count,), dtype=bool),
        "target_pos_cam": positions.copy(),
        "target_vel_cam": np.zeros((target_count, 3), dtype=np.float32),
        "camera_linear_velocity_cam": np.zeros((target_count, 3), dtype=np.float32),
        "camera_angular_velocity_cam": np.zeros((target_count, 3), dtype=np.float32),
        "visible": np.ones((target_count,), dtype=bool),
        "fully_visible": np.ones((target_count,), dtype=bool),
    }


class IsaacSimDatasetTests(TestCase):
    def test_sequence_writer_produces_tracker_compatible_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            boxes = _ground_truth(2)["bbox_xyxy"]
            writer = IsaacSimSequenceWriter(
                dataset_root=root,
                sequence_id=0,
                env_id=1,
                target_count=2,
                fps=50.0,
                intrinsics=np.asarray(
                    [[200.0, 0.0, 80.0], [0.0, 200.0, 45.0], [0.0, 0.0, 1.0]]
                ),
                initial_boxes_xyxy=boxes,
                expected_frames=3,
            )
            image = np.zeros((90, 160, 3), dtype=np.uint8)
            for frame_index in range(3):
                writer.add_frame(
                    image,
                    _ground_truth(2),
                    timestamp_s=frame_index / 50.0,
                )
            manifest_row = writer.finish()
            write_dataset_manifest(root, [manifest_row], {"test": True})

            summary = validate_collection_dataset(root)
            self.assertEqual(summary["sequences"], 1)
            self.assertEqual(summary["frames"], 3)
            sequence_dir = root / manifest_row["seq_dir"]
            initial_boxes, _ = load_initial_targets(
                sequence_dir / "initial_targets.json"
            )
            self.assertEqual(len(initial_boxes), 2)
            with (sequence_dir / "ground_truth.csv").open(
                "r", newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            self.assertEqual({int(row["target_id"]) for row in rows}, {0, 1})
            self.assertIn("x_cam_m", rows[0])
            self.assertEqual(float(rows[0]["lidar_range_m"]), 8.0)
            self.assertEqual(int(rows[0]["lidar_valid"]), 1)
            self.assertNotIn("x_world_m", rows[0])

    def test_abort_removes_partial_sequence(self) -> None:
        with TemporaryDirectory() as directory:
            writer = IsaacSimSequenceWriter(
                dataset_root=directory,
                sequence_id=3,
                env_id=0,
                target_count=2,
                fps=50.0,
                intrinsics=np.eye(3),
                initial_boxes_xyxy=_ground_truth(2)["bbox_xyxy"],
                expected_frames=5,
            )
            sequence_dir = writer.sequence_dir
            writer.add_frame(
                np.zeros((40, 80, 3), dtype=np.uint8),
                _ground_truth(2),
                timestamp_s=0.0,
            )
            writer.abort()
            self.assertFalse(sequence_dir.exists())

    def test_yaw_only_ground_truth_is_written_with_capture_frame_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth = _ground_truth(2)
            ground_truth["target_pos_yaw"] = np.asarray(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
            )
            ground_truth["target_vel_yaw"] = np.asarray(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64
            )
            ground_truth["camera_position_yaw"] = np.asarray(
                [[0.2, 0.0, 0.4], [0.2, 0.0, 0.4]], dtype=np.float64
            )
            ground_truth["rotation_yaw_from_camera"] = np.broadcast_to(
                np.eye(3), (2, 3, 3)
            ).copy()
            writer = IsaacSimSequenceWriter(
                dataset_root=root,
                sequence_id=8,
                env_id=0,
                target_count=2,
                fps=50.0,
                intrinsics=np.eye(3),
                initial_boxes_xyxy=_ground_truth(2)["bbox_xyxy"],
                expected_frames=1,
                extra_meta={"tracking_frame": "yaw_only"},
            )
            writer.add_frame(
                np.zeros((40, 80, 3), dtype=np.uint8),
                ground_truth,
                timestamp_s=0.0,
            )
            writer.finish()
            with (writer.sequence_dir / "ground_truth.csv").open(
                "r", newline="", encoding="utf-8"
            ) as file:
                row = next(csv.DictReader(file))
            self.assertEqual(float(row["x_yaw_m"]), 1.0)
            self.assertEqual(float(row["vx_yaw_mps"]), 0.1)
            self.assertEqual(float(row["camera_position_yaw_x_m"]), 0.2)
            with (writer.sequence_dir / "sequence_meta.json").open(
                "r", encoding="utf-8"
            ) as file:
                self.assertIn('"tracking_frame": "yaw_only"', file.read())

    def test_csv_only_sequence_contains_no_rgb_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = IsaacSimSequenceWriter(
                dataset_root=root,
                sequence_id=4,
                env_id=2,
                target_count=2,
                fps=50.0,
                intrinsics=np.eye(3),
                initial_boxes_xyxy=_ground_truth(2)["bbox_xyxy"],
                expected_frames=3,
                save_images=False,
            )
            for frame_index in range(3):
                writer.add_frame(
                    None,
                    _ground_truth(2),
                    timestamp_s=frame_index / 50.0,
                )
            manifest_row = writer.finish()
            write_dataset_manifest(root, [manifest_row], {"storage_format": "csv_only"})

            summary = validate_collection_dataset(root)
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(manifest_row["rgb_dir"], "")
            self.assertFalse((writer.sequence_dir / "rgb").exists())
            self.assertTrue((writer.sequence_dir / "ground_truth.csv").is_file())

    def test_primary_only_sequence_records_one_stable_target(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = IsaacSimSequenceWriter(
                dataset_root=root,
                sequence_id=5,
                env_id=0,
                target_count=3,
                fps=25.0,
                intrinsics=np.eye(3),
                initial_boxes_xyxy=_ground_truth(3)["bbox_xyxy"],
                expected_frames=4,
                save_images=False,
                recorded_target_ids=(2,),
                extra_meta={"primary_target_only": True, "primary_target_id": 2},
            )
            for frame_index in range(4):
                writer.add_frame(
                    None,
                    _ground_truth(3),
                    timestamp_s=frame_index / 25.0,
                )
            manifest_row = writer.finish()
            write_dataset_manifest(root, [manifest_row], {"recorded_targets": "primary_only"})

            summary = validate_collection_dataset(root)
            self.assertEqual(summary["frames"], 4)
            self.assertEqual(manifest_row["target_count"], 3)
            self.assertEqual(manifest_row["recorded_target_count"], 1)
            with (writer.sequence_dir / "ground_truth.csv").open(
                "r", newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertEqual({int(row["target_id"]) for row in rows}, {2})


if __name__ == "__main__":
    main()

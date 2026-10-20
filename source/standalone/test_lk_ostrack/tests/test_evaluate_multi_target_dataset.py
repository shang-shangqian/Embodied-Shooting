from __future__ import annotations

import math
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.evaluate_multi_target_dataset import (
    _SpeedRangeAccumulator,
    _angle_difference,
    _angular_error_rad,
    _bbox_iou,
    _fused_position_error,
    _position_error_components,
    _read_ground_truth,
)


class MultiTargetDatasetEvaluationTests(TestCase):
    def test_bbox_iou(self) -> None:
        self.assertAlmostEqual(_bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(_bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1.0 / 3.0)
        self.assertEqual(_bbox_iou((0, 0, 2, 2), (4, 4, 6, 6)), 0.0)

    def test_angle_difference_wraps(self) -> None:
        error = _angle_difference(math.radians(-179), math.radians(179))
        self.assertAlmostEqual(math.degrees(error), 2.0)

    def test_angular_error(self) -> None:
        self.assertAlmostEqual(_angular_error_rad(0.0, 0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(
            math.degrees(_angular_error_rad(math.radians(1.0), 0.0, 0.0, 0.0)),
            1.0,
        )

    def test_position_error_components_separate_radial_and_cross_track(self) -> None:
        radial = _position_error_components(0.0, 0.0, 12.0, (0.0, 0.0, 10.0))
        self.assertAlmostEqual(radial["position_error_m"], 2.0)
        self.assertAlmostEqual(radial["radial_error_m"], 2.0)
        self.assertAlmostEqual(radial["cross_track_error_m"], 0.0)
        self.assertAlmostEqual(radial["z_error_m"], 2.0)

        cross_track = _position_error_components(
            math.pi / 2.0, 0.0, 10.0, (0.0, 0.0, 10.0)
        )
        self.assertAlmostEqual(cross_track["position_error_m"], math.sqrt(200.0))
        self.assertAlmostEqual(cross_track["radial_error_m"], -10.0)
        self.assertAlmostEqual(cross_track["cross_track_error_m"], 10.0)

    def test_fused_position_error_applies_range_adjustment(self) -> None:
        truth = {
            "lidar_valid": True,
            "lidar_range_m": 10.25,
            "x_cam_m": 0.0,
            "y_cam_m": 0.0,
            "z_cam_m": 10.0,
        }
        error = _fused_position_error(0.0, 0.0, truth, -0.25)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertAlmostEqual(error["position_error_m"], 0.0)
        self.assertAlmostEqual(error["range_error_m"], 0.0)

    def test_ground_truth_is_reprojected_from_camera_position(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ground_truth.csv"
            fieldnames = (
                "frame_index",
                "target_id",
                "visible",
                "fully_visible",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "center_u",
                "center_v",
                "yaw_rad",
                "pitch_rad",
                "distance_m",
                "lidar_range_m",
                "lidar_valid",
                "x_cam_m",
                "y_cam_m",
                "z_cam_m",
                "vx_cam_mps",
                "vy_cam_mps",
                "vz_cam_mps",
            )
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "frame_index": 0,
                        "target_id": 0,
                        "visible": 0,
                        "fully_visible": 0,
                        "bbox_x1": 1,
                        "bbox_y1": 2,
                        "bbox_x2": 3,
                        "bbox_y2": 4,
                        "center_u": 2,
                        "center_v": 3,
                        "yaw_rad": 1,
                        "pitch_rad": 1,
                        "distance_m": 0,
                        "lidar_range_m": 11,
                        "lidar_valid": 1,
                        "x_cam_m": 1,
                        "y_cam_m": -2,
                        "z_cam_m": 10,
                        "vx_cam_mps": 3,
                        "vy_cam_mps": 4,
                        "vz_cam_mps": 0,
                    }
                )
            truth = _read_ground_truth(
                path,
                intrinsics=CameraIntrinsics(100, 100, 50, 40, 100, 80),
                target_radius_m=0.5,
            )[(0, 0)]

            self.assertTrue(truth["visible"])
            self.assertTrue(truth["fully_visible"])
            self.assertAlmostEqual(truth["center_u"], 60.0)
            self.assertAlmostEqual(truth["center_v"], 20.0)
            self.assertAlmostEqual(truth["bbox_x1"], 55.0)
            self.assertAlmostEqual(truth["bbox_y1"], 15.0)
            self.assertAlmostEqual(truth["bbox_x2"], 65.0)
            self.assertAlmostEqual(truth["bbox_y2"], 25.0)
            self.assertAlmostEqual(truth["speed_m_s"], 5.0)
            self.assertTrue(truth["lidar_valid"])
            self.assertAlmostEqual(truth["lidar_range_m"], 11.0)
            self.assertAlmostEqual(truth["x_cam_m"], 1.0)
            self.assertAlmostEqual(truth["y_cam_m"], -2.0)
            self.assertAlmostEqual(truth["z_cam_m"], 10.0)
            self.assertAlmostEqual(truth["distance_m"], math.sqrt(105.0))

    def test_speed_range_accumulator_finds_continuous_limit(self) -> None:
        accumulator = _SpeedRangeAccumulator(
            bin_width_mps=1.0,
            min_observations=2,
            min_tracking_availability=1.0,
            min_success_iou_0_5=0.5,
            max_angular_p95_deg=1.5,
        )
        results = []
        ground_truth = {}
        samples = (
            (0.2, "tracking", (0.0, 0.0, 10.0, 10.0)),
            (0.3, "tracking", (0.0, 0.0, 10.0, 10.0)),
            (1.2, "tracking", (0.0, 0.0, 10.0, 10.0)),
            (1.3, "lost", (20.0, 20.0, 30.0, 30.0)),
        )
        for frame_index, (speed, status, bbox) in enumerate(samples):
            estimate = SimpleNamespace(
                target_id=0,
                status=status,
                bbox_xyxy=bbox,
                center_xy=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                yaw_rad=0.0,
                pitch_rad=0.0,
            )
            results.append(SimpleNamespace(frame_index=frame_index, targets=[estimate]))
            ground_truth[(frame_index, 0)] = {
                "visible": True,
                "bbox_x1": 0.0,
                "bbox_y1": 0.0,
                "bbox_x2": 10.0,
                "bbox_y2": 10.0,
                "center_u": 5.0,
                "center_v": 5.0,
                "yaw_rad": 0.0,
                "pitch_rad": 0.0,
                "speed_m_s": speed,
                "distance_m": 10.0,
                "lidar_range_m": 10.0,
                "lidar_valid": True,
                "x_cam_m": 0.0,
                "y_cam_m": 0.0,
                "z_cam_m": 10.0,
            }
        accumulator.add_sequence(results=results, ground_truth=ground_truth)

        summary = accumulator.summary()

        self.assertTrue(summary["bins"][0]["passes"])
        self.assertAlmostEqual(summary["bins"][0]["angular_p50_deg"], 0.0)
        self.assertAlmostEqual(summary["bins"][0]["angular_p80_deg"], 0.0)
        self.assertAlmostEqual(summary["bins"][0]["angular_p85_deg"], 0.0)
        self.assertAlmostEqual(summary["bins"][0]["angular_p99_deg"], 0.0)
        self.assertAlmostEqual(summary["bins"][0]["angular_max_deg"], 0.0)
        self.assertEqual(summary["bins"][0]["position_samples"], 2)
        self.assertAlmostEqual(summary["bins"][0]["position_range_availability"], 1.0)
        self.assertAlmostEqual(summary["bins"][0]["position_3d_mae_m"], 0.0)
        self.assertFalse(summary["bins"][1]["passes"])
        self.assertIn("tracking_availability", summary["bins"][1]["failure_reasons"])
        self.assertAlmostEqual(summary["maximum_continuous_supported_speed_mps"], 0.3)

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock, patch

import cv2
import numpy as np

from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.multi_target_io import load_initial_targets, write_multi_target_csv
from test_lk_ostrack.multi_target_tracker import (
    FineTrackResult,
    MultiTargetConfig,
    MultiTargetVisionTracker,
    TargetKinematics,
)
from test_lk_ostrack.ostrack_tracker import OSTrackFrameTracker


def _textured_patch(seed: int, height: int = 40, width: int = 50) -> np.ndarray:
    random = np.random.RandomState(seed)
    patch_image = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(60):
        x = int(random.randint(2, width - 2))
        y = int(random.randint(2, height - 2))
        value = int(random.randint(100, 255))
        cv2.circle(patch_image, (x, y), 1, (value, value, value), -1)
    return patch_image


PATCH_A = _textured_patch(7)
PATCH_B = _textured_patch(13)


def _two_target_frame(frame_index: int) -> np.ndarray:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    first_x = 30 + 3 * frame_index
    second_x = 220 - 2 * frame_index
    image[45:85, first_x : first_x + 50] = PATCH_A
    image[100:140, second_x : second_x + 50] = PATCH_B
    return image


class _FineTrackerStub:
    def __init__(self) -> None:
        self.initialized_boxes: list[tuple[float, ...]] = []
        self.track_calls = 0
        self.reset_calls = 0
        self.box: tuple[float, ...] | None = None

    def initialize(self, image_rgb: np.ndarray, bbox_xyxy) -> None:
        del image_rgb
        self.box = tuple(float(value) for value in bbox_xyxy)
        self.initialized_boxes.append(self.box)

    def track(self, image_rgb: np.ndarray) -> FineTrackResult:
        del image_rgb
        self.track_calls += 1
        if self.box is None:
            raise RuntimeError("stub is not initialized")
        x1, y1, x2, y2 = self.box
        self.box = (x1 + 1.0, y1, x2 + 1.0, y2)
        return FineTrackResult(self.box, score=0.95)

    def reset(self) -> None:
        self.reset_calls += 1
        self.box = None


class _LowScoreFineTrackerStub(_FineTrackerStub):
    def track(self, image_rgb: np.ndarray) -> FineTrackResult:
        del image_rgb
        self.track_calls += 1
        if self.box is None:
            raise RuntimeError("stub is not initialized")
        return FineTrackResult(self.box, score=0.01)


class _JumpingFineTrackerStub(_FineTrackerStub):
    def track(self, image_rgb: np.ndarray) -> FineTrackResult:
        del image_rgb
        self.track_calls += 1
        return FineTrackResult((230.0, 100.0, 280.0, 140.0), score=0.95)


class _ModerateCorrectionFineTrackerStub(_FineTrackerStub):
    def track(self, image_rgb: np.ndarray) -> FineTrackResult:
        del image_rgb
        self.track_calls += 1
        if self.box is None:
            raise RuntimeError("stub is not initialized")
        x1, y1, x2, y2 = self.box
        self.box = (x1 + 20.0, y1, x2 + 20.0, y2)
        return FineTrackResult(self.box, score=0.95)


class MultiTargetTrackerTests(TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(250.0, 250.0, 160.0, 90.0, 320, 180)

    def test_tracks_all_ids_with_one_forward_backward_lk_pair(self) -> None:
        tracker = MultiTargetVisionTracker(self.intrinsics)
        initial = tracker.initialize(
            _two_target_frame(0),
            {10: (30, 45, 80, 85), 20: (220, 100, 270, 140)},
        )
        self.assertTrue(all(9 <= target.corners <= 16 for target in initial.targets))

        with patch(
            "test_lk_ostrack.multi_target_tracker.cv2.calcOpticalFlowPyrLK",
            wraps=cv2.calcOpticalFlowPyrLK,
        ) as optical_flow:
            result = tracker.update(_two_target_frame(1), dt_s=1.0 / 30.0)
        self.assertEqual(optical_flow.call_count, 2)

        targets = {target.target_id: target for target in result.targets}
        self.assertAlmostEqual(targets[10].center_xy[0], 58.0, delta=0.8)
        self.assertAlmostEqual(targets[20].center_xy[0], 243.0, delta=0.8)
        self.assertGreaterEqual(targets[10].ransac_inliers, 9)
        self.assertGreaterEqual(targets[20].ransac_inliers, 9)
        self.assertGreater(targets[10].yaw_rad, -1.0)
        self.assertGreater(targets[20].yaw_rad, 0.0)

    def test_reuses_lk_points_between_three_frame_redetections(self) -> None:
        tracker = MultiTargetVisionTracker(self.intrinsics)
        with patch.object(
            tracker, "_extract_points", wraps=tracker._extract_points
        ) as extract_points:
            tracker.initialize(
                _two_target_frame(0),
                {10: (30, 45, 80, 85), 20: (220, 100, 270, 140)},
            )
            call_counts = [extract_points.call_count]
            for frame_index in range(1, 7):
                tracker.update(_two_target_frame(frame_index), dt_s=1.0 / 30.0)
                call_counts.append(extract_points.call_count)

        self.assertEqual(call_counts, [2, 2, 2, 4, 4, 4, 6])

    def test_extracts_corners_from_target_roi(self) -> None:
        tracker = MultiTargetVisionTracker(self.intrinsics)
        image = _two_target_frame(0)
        with patch(
            "test_lk_ostrack.multi_target_tracker.cv2.goodFeaturesToTrack",
            wraps=cv2.goodFeaturesToTrack,
        ) as detect_corners:
            result = tracker.initialize(image, {10: (30, 45, 80, 85)})

        self.assertEqual(detect_corners.call_count, 1)
        roi = detect_corners.call_args.args[0]
        self.assertEqual(roi.shape, (38, 48))
        self.assertIsNone(detect_corners.call_args.kwargs["mask"])
        for x, y in tracker._states[10].points:
            self.assertGreaterEqual(x, 31.0)
            self.assertLess(x, 79.0)
            self.assertGreaterEqual(y, 46.0)
            self.assertLess(y, 84.0)
        self.assertGreaterEqual(result.targets[0].corners, 9)

    def test_occlusion_decays_confidence_then_marks_target_lost(self) -> None:
        config = MultiTargetConfig(max_missed_frames=2)
        tracker = MultiTargetVisionTracker(self.intrinsics, config=config)
        tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})
        blank = np.zeros_like(_two_target_frame(0))

        first = tracker.update(blank, dt_s=1.0 / 30.0).targets[0]
        second = tracker.update(blank, dt_s=1.0 / 30.0).targets[0]
        third_result = tracker.update(blank, dt_s=1.0 / 30.0)
        third = third_result.targets[0]

        self.assertEqual(first.status, "predicted")
        self.assertLess(second.confidence, first.confidence)
        self.assertEqual(third.status, "lost")
        self.assertIsNone(third_result.selected_target_id)

    def test_rejects_disjoint_high_innovation_flow_measurement(self) -> None:
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=MultiTargetConfig(
                measurement_rejection_innovation_threshold=0.80,
                measurement_rejection_iou_threshold=0.05,
            ),
        )
        tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})
        state = tracker._states[10]
        kalman_before = state.kalman.x.copy()
        rejected_bbox = (210.0, 100.0, 260.0, 140.0)
        fake_measurement = (
            rejected_bbox,
            0.95,
            9,
            0.1,
            np.asarray([[220.0, 110.0]] * 9, dtype=np.float32),
            1.0,
            1.0,
        )

        with (
            patch.object(tracker, "_track_all_points", return_value={10: object()}),
            patch.object(tracker, "_fit_motion", return_value=fake_measurement),
        ):
            selected = tracker.update(
                _two_target_frame(1), dt_s=1.0 / 30.0
            ).selected_target

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "prediction")
        self.assertEqual(selected.status, "predicted")
        self.assertTrue(selected.measurement_rejected)
        self.assertGreaterEqual(selected.bbox_innovation, 0.80)
        self.assertLessEqual(selected.bbox_iou_with_prediction, 0.05)
        self.assertNotEqual(selected.bbox_xyxy, rejected_bbox)
        # Initial velocity is zero, so predict() must not change box state;
        # importantly, the rejected measurement never enters correct().
        np.testing.assert_allclose(state.kalman.x, kalman_before, atol=1e-12)

    def test_priority_switch_reinitializes_fine_tracker_for_new_id(self) -> None:
        fine_tracker = _FineTrackerStub()
        config = MultiTargetConfig(fine_tracker_interval_frames=1)
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=config,
            fine_tracker=fine_tracker,
        )
        initial = tracker.initialize(
            _two_target_frame(0),
            {10: (30, 45, 80, 85), 20: (220, 100, 270, 140)},
        )
        first_id = initial.selected_target_id
        self.assertIsNotNone(first_id)
        self.assertEqual(len(fine_tracker.initialized_boxes), 1)

        tracked = tracker.update(_two_target_frame(1), dt_s=1.0 / 30.0)
        self.assertEqual(fine_tracker.track_calls, 1)
        self.assertEqual(tracked.selected_target.source, "ostrack")

        other_id = 10 if first_id == 20 else 20
        tracker.set_priority_bias(other_id, 2.0)
        switched = tracker.update(_two_target_frame(2), dt_s=1.0 / 30.0)
        self.assertEqual(switched.selected_target_id, other_id)
        self.assertEqual(fine_tracker.track_calls, 1)
        self.assertEqual(len(fine_tracker.initialized_boxes), 2)

    def test_estimates_distance_and_speed_from_box_state(self) -> None:
        config = MultiTargetConfig(
            target_width_m=0.70,
            target_height_m=0.70,
            kinematic_estimation_horizon_s=0.10,
        )
        tracker = MultiTargetVisionTracker(self.intrinsics, config=config)
        result = tracker.initialize(
            _two_target_frame(0),
            {10: (135, 65, 185, 115)},
        )

        self.assertAlmostEqual(result.targets[0].distance_m, 3.5, places=6)
        self.assertAlmostEqual(result.targets[0].speed_m_s, 0.0, places=6)

        tracker._states[10].kalman.x[4] = 50.0
        _, speed_m_s = tracker._estimate_kinematics(tracker._states[10])
        self.assertAlmostEqual(speed_m_s, 0.70, places=6)

    def test_priority_can_prefer_nearer_or_faster_target(self) -> None:
        config = MultiTargetConfig(
            priority_confidence_weight=0.0,
            priority_center_weight=0.0,
            priority_area_weight=0.0,
            priority_stability_weight=0.0,
            priority_distance_weight=1.0,
            priority_speed_weight=0.0,
            priority_distance_near_m=2.0,
            priority_distance_far_m=10.0,
            priority_switch_margin=0.0,
        )
        tracker = MultiTargetVisionTracker(self.intrinsics, config=config)
        initial = tracker.initialize(
            _two_target_frame(0),
            {10: (30, 45, 80, 85), 20: (220, 100, 245, 120)},
        )
        self.assertEqual(initial.selected_target_id, 10)

        tracker.config = MultiTargetConfig(
            priority_confidence_weight=0.0,
            priority_center_weight=0.0,
            priority_area_weight=0.0,
            priority_stability_weight=0.0,
            priority_distance_weight=0.0,
            priority_speed_weight=1.0,
            priority_switch_margin=0.0,
        )
        tracker._states[10].kalman.x[4:] = 0.0
        tracker._states[20].kalman.x[4] = 100.0
        priorities = tracker._compute_priorities(320, 180)
        self.assertGreater(priorities[20], priorities[10])

    def test_external_kinematics_override_visual_estimates_each_frame(self) -> None:
        config = MultiTargetConfig(
            priority_confidence_weight=0.0,
            priority_center_weight=0.0,
            priority_area_weight=0.0,
            priority_stability_weight=0.0,
            priority_distance_weight=1.0,
            priority_speed_weight=0.0,
            priority_distance_near_m=2.0,
            priority_distance_far_m=10.0,
            priority_switch_margin=0.0,
        )
        tracker = MultiTargetVisionTracker(self.intrinsics, config=config)
        initial = tracker.initialize(
            _two_target_frame(0),
            {10: (30, 45, 80, 85), 20: (220, 100, 245, 120)},
            target_kinematics={
                10: TargetKinematics(9.0, 1.0, source="ground_truth"),
                20: TargetKinematics(3.0, 2.0, source="ground_truth"),
            },
        )
        self.assertEqual(initial.selected_target_id, 20)
        self.assertEqual(initial.selected_target.kinematics_source, "ground_truth")
        self.assertEqual(initial.selected_target.distance_m, 3.0)

        updated = tracker.update(
            _two_target_frame(1),
            dt_s=1.0 / 30.0,
            target_kinematics={10: (2.5, 3.0), 20: (9.5, 0.5)},
        )
        self.assertEqual(updated.selected_target_id, 10)
        self.assertEqual(updated.selected_target.kinematics_source, "external")
        self.assertEqual(updated.selected_target.speed_m_s, 3.0)

    def test_runs_fine_tracker_every_three_frames(self) -> None:
        fine_tracker = _FineTrackerStub()
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=MultiTargetConfig(fine_tracker_interval_frames=3),
            fine_tracker=fine_tracker,
        )
        tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})

        track_calls = []
        sources = []
        for frame_index in range(1, 7):
            result = tracker.update(
                _two_target_frame(frame_index), dt_s=1.0 / 30.0
            )
            track_calls.append(fine_tracker.track_calls)
            sources.append(result.selected_target.source)

        self.assertEqual(track_calls, [0, 0, 1, 1, 1, 2])
        self.assertEqual(sources[2], "ostrack")
        self.assertEqual(sources[5], "ostrack")

    def test_adaptive_policy_uses_max_interval_for_stable_tracking(self) -> None:
        fine_tracker = _FineTrackerStub()
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=MultiTargetConfig(
                fine_tracker_policy="adaptive",
                adaptive_min_interval_frames=2,
                adaptive_max_interval_frames=4,
            ),
            fine_tracker=fine_tracker,
        )
        initial = tracker.initialize(
            _two_target_frame(0), {10: (30, 45, 80, 85)}
        )
        self.assertEqual(initial.selected_target.fine_action, "initialize")
        self.assertFalse(initial.selected_target.fine_applied)

        calls = []
        reasons = []
        for frame_index in range(1, 9):
            selected = tracker.update(
                _two_target_frame(frame_index), dt_s=1.0 / 30.0
            ).selected_target
            calls.append(fine_tracker.track_calls)
            reasons.append(selected.fine_trigger_reason)

        self.assertEqual(calls, [0, 0, 0, 1, 1, 1, 1, 2])
        self.assertEqual(reasons[3], "max_interval")
        self.assertEqual(reasons[7], "max_interval")

    def test_adaptive_policy_hard_triggers_after_lk_failure(self) -> None:
        fine_tracker = _FineTrackerStub()
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=MultiTargetConfig(
                fine_tracker_policy="adaptive",
                adaptive_min_interval_frames=2,
                adaptive_max_interval_frames=100,
            ),
            fine_tracker=fine_tracker,
        )
        tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})
        blank = np.zeros_like(_two_target_frame(0))

        first = tracker.update(blank, dt_s=1.0 / 30.0).selected_target
        second = tracker.update(blank, dt_s=1.0 / 30.0).selected_target

        self.assertEqual(first.fine_trigger_reason, "min_interval")
        self.assertEqual(second.fine_trigger_reason, "hard_tracking_failure")
        self.assertTrue(second.fine_attempted)
        self.assertTrue(second.fine_applied)
        self.assertEqual(fine_tracker.track_calls, 1)

    def test_adaptive_policy_requires_consecutive_soft_risk_frames(self) -> None:
        fine_tracker = _FineTrackerStub()
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=MultiTargetConfig(
                fine_tracker_policy="adaptive",
                adaptive_min_interval_frames=1,
                adaptive_max_interval_frames=100,
                adaptive_bad_frames=2,
                adaptive_risk_threshold=0.30,
                adaptive_critical_risk_threshold=1.0,
                adaptive_risk_clear_threshold=0.20,
            ),
            fine_tracker=fine_tracker,
        )
        tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})
        state = tracker._states[10]
        state.status = "tracking"
        state.confidence = 0.60
        state.last_motion_confidence = 0.60
        state.ransac_inliers = 4
        state.median_fb_error_px = 1.4
        state.last_bbox_innovation = 0.70
        state.last_bbox_iou_with_prediction = 0.20

        tracker._frame_index = 1
        first_run, first_reason = tracker._decide_fine_tracker_action(state)
        tracker._frame_index = 2
        second_run, second_reason = tracker._decide_fine_tracker_action(state)

        self.assertFalse(first_run)
        self.assertEqual(first_reason, "stable")
        self.assertTrue(second_run)
        self.assertEqual(second_reason, "soft_quality_streak")

    def test_adaptive_policy_backs_off_rejected_results(self) -> None:
        fine_tracker = _LowScoreFineTrackerStub()
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=MultiTargetConfig(
                fine_tracker_policy="adaptive",
                adaptive_min_interval_frames=1,
                adaptive_max_interval_frames=2,
                adaptive_failure_backoff_initial_frames=1,
                adaptive_failure_backoff_max_frames=8,
            ),
            fine_tracker=fine_tracker,
        )
        tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})

        calls = []
        results = []
        for frame_index in range(1, 6):
            selected = tracker.update(
                _two_target_frame(frame_index), dt_s=1.0 / 30.0
            ).selected_target
            calls.append(fine_tracker.track_calls)
            results.append(selected)

        self.assertEqual(calls, [0, 1, 1, 1, 2])
        self.assertEqual(results[1].fine_action, "track_rejected")
        self.assertEqual(
            results[2].fine_trigger_reason, "reinitialize_after_failure"
        )
        self.assertEqual(results[2].fine_failure_count, 1)
        self.assertEqual(results[4].fine_failure_count, 2)

    def test_rejects_implausible_fine_tracker_jump(self) -> None:
        fine_tracker = _JumpingFineTrackerStub()
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=MultiTargetConfig(
                fine_tracker_policy="adaptive",
                adaptive_min_interval_frames=1,
                adaptive_max_interval_frames=2,
            ),
            fine_tracker=fine_tracker,
        )
        tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})
        tracker.update(_two_target_frame(1), dt_s=1.0 / 30.0)
        selected = tracker.update(
            _two_target_frame(2), dt_s=1.0 / 30.0
        ).selected_target

        self.assertEqual(selected.fine_action, "track_rejected")
        self.assertFalse(selected.fine_applied)
        self.assertNotEqual(selected.source, "ostrack")
        self.assertLess(selected.center_xy[0], 100.0)

    def test_smooths_and_marks_moderate_fine_tracker_correction(self) -> None:
        baseline = MultiTargetVisionTracker(self.intrinsics)
        baseline.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})
        lk_center_x = baseline.update(
            _two_target_frame(1), dt_s=1.0 / 30.0
        ).selected_target.center_xy[0]

        fine_tracker = _ModerateCorrectionFineTrackerStub()
        tracker = MultiTargetVisionTracker(
            self.intrinsics,
            config=MultiTargetConfig(
                fine_tracker_interval_frames=1,
                fine_discontinuity_center_jump_box_diagonals=0.20,
            ),
            fine_tracker=fine_tracker,
        )
        tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})
        selected = tracker.update(
            _two_target_frame(1), dt_s=1.0 / 30.0
        ).selected_target

        raw_ostrack_center_x = 75.0
        self.assertEqual(selected.source, "ostrack")
        self.assertTrue(selected.fine_applied)
        self.assertTrue(selected.measurement_discontinuity)
        self.assertGreater(selected.fine_correction_center_px, 0.0)
        self.assertGreater(selected.bbox_innovation, 0.0)
        self.assertLess(selected.bbox_iou_with_prediction, 1.0)
        self.assertGreater(selected.center_xy[0], lk_center_x)
        self.assertLess(selected.center_xy[0], raw_ostrack_center_x)
        np.testing.assert_allclose(
            tracker._states[10].kalman.x[4:],
            baseline._states[10].kalman.x[4:],
            rtol=1e-9,
            atol=1e-9,
        )

    def test_target_file_and_long_csv(self) -> None:
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            target_path = directory_path / "targets.json"
            target_path.write_text(
                '{"targets": [{"id": 4, "bbox": [10, 20, 40, 60], "priority": 0.3}]}',
                encoding="utf-8",
            )
            boxes, biases = load_initial_targets(target_path)
            self.assertEqual(boxes[4], (10.0, 20.0, 40.0, 60.0))
            self.assertEqual(biases[4], 0.3)

            tracker = MultiTargetVisionTracker(self.intrinsics)
            result = tracker.initialize(_two_target_frame(0), {10: (30, 45, 80, 85)})
            csv_path = directory_path / "tracks.csv"
            write_multi_target_csv([result], csv_path)
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("target_id", text)
            self.assertIn("predicted_yaw_rad", text)
            self.assertIn("distance_m", text)
            self.assertIn("speed_m_s", text)
            self.assertIn("kinematics_source", text)
            self.assertIn("fine_trigger_reason", text)
            self.assertIn("fine_latency_ms", text)
            self.assertIn("measurement_discontinuity", text)

    def test_ostrack_target_switch_reuses_loaded_network(self) -> None:
        runtime_tracker = OSTrackFrameTracker("/unused/repo", "/unused/checkpoint")
        official_tracker = Mock()
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        with (
            patch.object(
                runtime_tracker.adapter,
                "_build_tracker",
                return_value=official_tracker,
            ) as build_tracker,
            patch.object(
                runtime_tracker,
                "_call_in_repo",
                side_effect=lambda function: function(),
            ),
        ):
            runtime_tracker.initialize(image, (5, 5, 25, 25))
            runtime_tracker.reset()
            runtime_tracker.initialize(image, (40, 20, 65, 50))

        self.assertEqual(build_tracker.call_count, 1)
        self.assertEqual(official_tracker.initialize.call_count, 2)

    def test_ostrack_runtime_preserves_subpixel_bbox(self) -> None:
        runtime_tracker = OSTrackFrameTracker("/unused/repo", "/unused/checkpoint")
        official_tracker = Mock()
        official_tracker.track.return_value = {
            "target_bbox": [10.25, 20.5, 15.5, 12.25],
            "best_score": 0.9,
        }
        runtime_tracker._tracker = official_tracker
        runtime_tracker._initialized = True
        image = np.zeros((60, 80, 3), dtype=np.uint8)

        with patch.object(
            runtime_tracker,
            "_call_in_repo",
            side_effect=lambda function: function(),
        ):
            result = runtime_tracker.track(image)

        self.assertEqual(result.bbox_xyxy, (10.25, 20.5, 25.75, 32.75))


if __name__ == "__main__":
    main()

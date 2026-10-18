from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


TEST11_DIR = Path(__file__).resolve().parents[2]
if str(TEST11_DIR.parent) not in sys.path:
    sys.path.insert(0, str(TEST11_DIR.parent))

from test11.speed_prediction.common import stage2_degradation as stage2_module
from test11.speed_prediction.common.stage2_degradation import Stage2Degrader


class Stage2DegradationTests(unittest.TestCase):
    @staticmethod
    def _degrader(profile: str, severity: str = "hard", seed: int = 123) -> Stage2Degrader:
        degrader = Stage2Degrader(
            profile=profile,
            severity=severity,
            seed=seed,
            num_envs=1,
            nominal_dt_s=0.02,
        )
        degrader.reset_episode(0, 7)
        return degrader

    def test_clean_profile_is_identity(self) -> None:
        degrader = self._degrader("clean")
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        box = (2.0, 3.0, 10.0, 11.0)
        linear = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        angular = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)

        self.assertEqual(degrader.perturb_bootstrap_box(0, box, image_width=16, image_height=12), box)
        np.testing.assert_array_equal(degrader.degrade_image(0, image), image)
        self.assertEqual(degrader.degrade_direction(0, 0.1, -0.2), (0.1, -0.2))
        self.assertEqual(degrader.degrade_range(0, 100.0, True), (100.0, True))
        degraded_linear, degraded_angular = degrader.degrade_proprioception(0, linear, angular)
        np.testing.assert_array_equal(degraded_linear, linear)
        np.testing.assert_array_equal(degraded_angular, angular)
        self.assertEqual(degrader.next_timestamp(0, 0.0), (0.0, 0.02))

    def test_same_seed_and_episode_are_reproducible(self) -> None:
        first = self._degrader("range", "medium", seed=918)
        second = self._degrader("range", "medium", seed=918)
        first_values = [first.degrade_range(0, 80.0 + index, True) for index in range(50)]
        second_values = [second.degrade_range(0, 80.0 + index, True) for index in range(50)]
        np.testing.assert_allclose(
            [value for value, _ in first_values],
            [value for value, _ in second_values],
            equal_nan=True,
        )
        self.assertEqual(
            [valid for _, valid in first_values],
            [valid for _, valid in second_values],
        )
        self.assertEqual(first.episode_metadata(0), second.episode_metadata(0))

    def test_range_profile_changes_measurements_and_records_failures(self) -> None:
        degrader = self._degrader("range")
        samples = [degrader.degrade_range(0, 100.0, True) for _ in range(1000)]
        valid_values = [value for value, valid in samples if valid]
        self.assertTrue(valid_values)
        self.assertTrue(any(abs(value - 100.0) > 1e-4 for value in valid_values))
        counters = degrader.episode_metadata(0)["event_counts"]
        self.assertGreater(counters["range_degraded_frames"], 0)
        self.assertGreater(counters.get("range_dropout_frames", 0), 0)

    def test_correlated_range_noise_avoids_unphysical_frame_difference(self) -> None:
        degrader = self._degrader("range", "medium", seed=512)
        values = [degrader.degrade_range(0, 120.0, True) for _ in range(1000)]
        valid_ranges = np.asarray([value for value, valid in values if valid], dtype=np.float64)
        frame_speeds = np.abs(np.diff(valid_ranges)) / 0.02
        self.assertLess(float(np.median(frame_speeds)), 2.0)
        self.assertLess(float(np.quantile(frame_speeds, 0.95)), 5.0)

    @unittest.skipIf(stage2_module.cv2 is None, "OpenCV is unavailable in this Python environment")
    def test_image_profile_really_changes_pixels(self) -> None:
        degrader = self._degrader("image")
        image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        outputs = [degrader.degrade_image(0, image) for _ in range(300)]
        self.assertTrue(any(not np.array_equal(output, image) for output in outputs))
        counters = degrader.episode_metadata(0)["event_counts"]
        self.assertGreater(counters["image_degraded_frames"], 0)
        self.assertGreater(counters["image_degradation_segments"], 0)

    def test_tracking_profile_generates_detector_and_outage_events(self) -> None:
        degrader = self._degrader("tracking")
        box = (10.0, 10.0, 30.0, 30.0)
        proposals = [
            degrader.perturb_bootstrap_box(0, box, image_width=64, image_height=48)
            for _ in range(200)
        ]
        self.assertTrue(any(proposal is None for proposal in proposals))
        self.assertTrue(any(proposal is not None and proposal != box for proposal in proposals))

        image = np.full((48, 64, 3), 127, dtype=np.uint8)
        outputs = [
            degrader.degrade_image(0, image, target_bbox_xyxy=box) for _ in range(1000)
        ]
        self.assertTrue(any(np.count_nonzero(output) == 0 for output in outputs))
        counters = degrader.episode_metadata(0)["event_counts"]
        self.assertGreater(counters["bootstrap_misses"], 0)
        self.assertGreater(counters["perturbed_bootstrap_boxes"], 0)
        self.assertGreater(counters["image_dropout_frames"], 0)
        self.assertGreater(counters["occlusion_frames"], 0)

    def test_records_tracker_measurement_rejection(self) -> None:
        degrader = self._degrader("tracking")
        degrader.record_tracker_measurement_rejection(0)
        degrader.record_tracker_measurement_rejection(0)
        counters = degrader.episode_metadata(0)["event_counts"]
        self.assertEqual(counters["tracker_measurement_rejected"], 2)

    def test_timing_profile_delays_image_range_and_uses_actual_dt(self) -> None:
        degrader = self._degrader("timing", "medium")
        first_image = np.full((8, 8, 3), 11, dtype=np.uint8)
        second_image = np.full((8, 8, 3), 22, dtype=np.uint8)
        third_image = np.full((8, 8, 3), 33, dtype=np.uint8)
        np.testing.assert_array_equal(degrader.degrade_image(0, first_image), first_image)
        np.testing.assert_array_equal(degrader.degrade_image(0, second_image), first_image)
        np.testing.assert_array_equal(degrader.degrade_image(0, third_image), first_image)

        first_range, first_valid = degrader.degrade_range(0, 10.0, True)
        second_range, second_valid = degrader.degrade_range(0, 20.0, True)
        self.assertTrue(np.isnan(first_range))
        self.assertFalse(first_valid)
        self.assertEqual((second_range, second_valid), (10.0, True))
        self.assertEqual(degrader.degrade_range(0, 30.0, True), (20.0, True))
        metadata = degrader.episode_metadata(0)
        self.assertEqual(metadata["visual_range_delay_offset_steps"], 1)

        timestamps = [degrader.next_timestamp(0, index * 0.02) for index in range(100)]
        observed_times = [timestamp for timestamp, _ in timestamps]
        self.assertTrue(all(right > left for left, right in zip(observed_times, observed_times[1:])))
        self.assertTrue(any(abs(dt_s - 0.02) > 1e-8 for _, dt_s in timestamps[1:]))
        self.assertGreaterEqual(min(dt_s for _, dt_s in timestamps), 0.015)
        self.assertEqual(degrader.config_metadata()["min_timestamp_dt_ratio"], 0.75)

    def test_timestamp_dt_floor_is_configurable_and_validated(self) -> None:
        degrader = Stage2Degrader(
            profile="timing",
            severity="hard",
            seed=456,
            num_envs=1,
            nominal_dt_s=0.02,
            min_timestamp_dt_ratio=0.9,
        )
        degrader.reset_episode(0, 2)
        observed = [degrader.next_timestamp(0, index * 0.02) for index in range(500)]
        self.assertGreaterEqual(min(dt_s for _, dt_s in observed), 0.018)
        with self.assertRaisesRegex(ValueError, "min_timestamp_dt_ratio"):
            Stage2Degrader(
                profile="timing",
                severity="hard",
                seed=456,
                num_envs=1,
                nominal_dt_s=0.02,
                min_timestamp_dt_ratio=1.1,
            )

    def test_proprio_profile_keeps_truth_separate_from_degraded_copy(self) -> None:
        degrader = self._degrader("proprio")
        linear = np.zeros(3, dtype=np.float32)
        angular = np.zeros(3, dtype=np.float32)
        degraded_linear, degraded_angular = degrader.degrade_proprioception(0, linear, angular)
        np.testing.assert_array_equal(linear, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(angular, np.zeros(3, dtype=np.float32))
        self.assertFalse(np.array_equal(degraded_linear, linear))
        self.assertFalse(np.array_equal(degraded_angular, angular))

    def test_mixed_profile_enables_two_to_four_components(self) -> None:
        degrader = self._degrader("mixed", "random")
        metadata = degrader.episode_metadata(0)
        self.assertIn(metadata["episode_severity"], ("mild", "medium", "hard"))
        self.assertGreaterEqual(len(metadata["enabled_components"]), 2)
        self.assertLessEqual(len(metadata["enabled_components"]), 4)


if __name__ == "__main__":
    unittest.main()

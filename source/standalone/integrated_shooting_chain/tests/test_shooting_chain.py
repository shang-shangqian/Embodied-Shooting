from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import unittest

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from integrated_shooting_chain.shooting_chain import (  # noqa: E402
    CameraExtrinsics,
    RobotState,
    ShootingChain,
    build_policy_observation,
    tracker_quality_features,
)


@dataclass
class FakeTarget:
    target_id: int = 7
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    distance_m: float = 100.0
    source: str = "initial"
    status: str = "initialized"
    confidence: float = 1.0
    motion_confidence: float = 1.0
    point_retention: float = 1.0
    ransac_inlier_ratio: float = 1.0
    median_fb_error_px: float = 0.0
    bbox_innovation: float = 0.0
    bbox_iou_with_prediction: float = 1.0


@dataclass
class FakeFrame:
    timestamp_s: float
    selected_target: FakeTarget | None


@dataclass
class FakeTrackerConfig:
    fb_max_error_px: float = 2.0


class FakeVision:
    initialized = False

    def initialize(self, image_rgb, initial_boxes, **kwargs):
        self.initialized = True
        return FakeFrame(float(kwargs["timestamp_s"]), FakeTarget())

    def update(self, image_rgb, **kwargs):
        timestamp = kwargs.get("timestamp_s")
        return FakeFrame(float(0.1 if timestamp is None else timestamp), FakeTarget(source="flow", status="tracking"))


class FakeEstimate:
    velocity_mean_mps = np.asarray([1.0, 2.0, 0.5], dtype=np.float32)
    reliability_gate = 0.9


class FakeVelocity:
    def __init__(self):
        self.observations = []

    def update(self, target_id, observation):
        self.observations.append((target_id, observation))
        return FakeEstimate()

    def reset(self, target_id=None):
        return None


class ShootingChainTest(unittest.TestCase):
    def setUp(self):
        self.robot = RobotState(
            turret_yaw_rad=0.0,
            turret_pitch_rad=0.0,
            turret_joint_limits_rad=((-1.0, 1.0), (-0.8, 0.8)),
        )

    def test_observation_is_28_dimensions_and_finite(self):
        obs = build_policy_observation([0.0, 0.0, 100.0], [1.0, 2.0, 0.5], self.robot)
        self.assertEqual(obs.shape, (28,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertAlmostEqual(float(obs[0]), 0.0, places=6)
        self.assertAlmostEqual(float(obs[2]), 1.0, places=6)

    def test_camera_extrinsic_is_applied_before_speed_model(self):
        extrinsics = CameraExtrinsics(
            rotation_base_from_camera=np.asarray(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            camera_position_base_m=np.asarray([1.0, 2.0, 3.0]),
        )
        velocity = FakeVelocity()
        chain = ShootingChain(
            FakeVision(),
            velocity,
            lambda obs: np.asarray([0.0, 0.0, 1.0]),
            camera_extrinsics=extrinsics,
        )
        command = chain.initialize(np.zeros((8, 8, 3), dtype=np.uint8), {7: (1, 1, 2, 2)}, self.robot)
        self.assertTrue(command.prediction_ready)
        # Camera point [0, 0, 100] -> base [101, 2, 3].
        np.testing.assert_allclose(command.target_position_base_m, [101.0, 2.0, 3.0])
        self.assertEqual(len(velocity.observations), 1)
        np.testing.assert_allclose(velocity.observations[0][1].position_base_m, [101.0, 2.0, 3.0])

    def test_fire_intent_requires_two_consecutive_frames(self):
        chain = ShootingChain(
            FakeVision(),
            FakeVelocity(),
            lambda obs: np.asarray([0.0, 0.0, 1.0]),
        )
        first = chain.initialize(np.zeros((8, 8, 3), dtype=np.uint8), {7: (1, 1, 2, 2)}, self.robot)
        second = chain.update(np.zeros((8, 8, 3), dtype=np.uint8), self.robot, dt_s=0.1)
        self.assertFalse(first.fire_command)
        self.assertTrue(second.fire_command)

    def test_taq_quality_matches_collector_normalization(self):
        target = FakeTarget(
            confidence=1.2,
            motion_confidence=-0.1,
            point_retention=0.7,
            ransac_inlier_ratio=0.8,
            median_fb_error_px=1.0,
            bbox_innovation=0.25,
            bbox_iou_with_prediction=0.9,
        )
        quality = tracker_quality_features(target, FakeTrackerConfig())
        np.testing.assert_allclose(quality, [1.0, 0.0, 0.7, 0.8, 0.5, 0.25, 0.9])


if __name__ == "__main__":
    unittest.main()

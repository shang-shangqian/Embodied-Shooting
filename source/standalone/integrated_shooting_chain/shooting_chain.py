"""Runtime orchestration for the three-module shooting chain.

Data flow::

    RGB frame + robot state
        -> test_lk_ostrack.MultiTargetVisionTracker
        -> yaw-only-base target position
        -> test11 OnlineTAQVelocityEstimator
        -> 28-D test7_2_weight actor observation
        -> decision policy
        -> turret action + confirmed fire command

The module is numpy-only so that geometry, observation construction and offline
integration tests can run without importing Isaac Sim or torch.  The actual
vision tracker, velocity estimator and policy are injected as objects with the
small interfaces documented below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


class VisionTracker(Protocol):
    initialized: bool

    def initialize(self, image_rgb: np.ndarray, initial_boxes: Any, **kwargs: Any) -> Any: ...

    def update(self, image_rgb: np.ndarray, **kwargs: Any) -> Any: ...


class VelocityEstimator(Protocol):
    def update(self, target_id: int, observation: Any) -> Any: ...

    def reset(self, target_id: int | None = None) -> None: ...


Policy = Callable[[np.ndarray], Sequence[float]]


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _wrap_to_pi(angle: float) -> float:
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


def tracker_quality_features(target: Any, tracker_config: Any) -> tuple[float, ...]:
    """Build the seven TAQ quality values exactly as the v2 collector does."""
    fb_max_error_px = max(float(getattr(tracker_config, "fb_max_error_px", 1.5)), 1.0e-6)
    fb_error_raw = getattr(target, "median_fb_error_px", None)
    fb_error = 0.0 if fb_error_raw is None else float(fb_error_raw)
    bbox_innovation = float(getattr(target, "bbox_innovation", 0.0))
    bbox_iou = float(getattr(target, "bbox_iou_with_prediction", 1.0))
    if bool(getattr(target, "fine_applied", False)):
        x1, y1, x2, y2 = (float(value) for value in getattr(target, "bbox_xyxy"))
        bbox_diagonal = max(float(math.hypot(x2 - x1, y2 - y1)), 1.0)
        fine_center_ratio = float(getattr(target, "fine_correction_center_px", 0.0)) / bbox_diagonal
        bbox_innovation = max(bbox_innovation, fine_center_ratio)
        bbox_iou = min(bbox_iou, float(getattr(target, "fine_correction_iou", 1.0)))

    values = np.asarray(
        [
            getattr(target, "confidence", 0.0),
            getattr(target, "motion_confidence", 0.0),
            getattr(target, "point_retention", 0.0),
            getattr(target, "ransac_inlier_ratio", 0.0),
            fb_error / fb_max_error_px,
            bbox_innovation,
            bbox_iou,
        ],
        dtype=np.float64,
    )
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    return tuple(float(value) for value in np.clip(values, 0.0, 1.0))


@dataclass(frozen=True)
class CameraExtrinsics:
    """Fixed camera extrinsics in the yaw-only base frame.

    ``rotation_base_from_camera`` maps OpenCV camera vectors
    ``[right, down, forward]`` to yaw-only base vectors ``[forward, left, up]``.
    ``camera_position_base_m`` is the camera origin relative to the base origin.
    These are exactly the quantities exposed by the vision dataset as
    ``rotation_yaw_from_camera`` and ``camera_position_yaw``.
    """

    rotation_base_from_camera: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    camera_position_base_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_base_from_camera, dtype=np.float64)
        position = _vec3(self.camera_position_base_m, "camera_position_base_m")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("rotation_base_from_camera must be a finite 3x3 matrix")
        # A malformed extrinsic silently corrupts both position and velocity.
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-3):
            raise ValueError("rotation_base_from_camera must be approximately orthonormal")
        object.__setattr__(self, "rotation_base_from_camera", rotation)
        object.__setattr__(self, "camera_position_base_m", position)

    def camera_to_base(self, position_camera_m: Sequence[float]) -> np.ndarray:
        return self.camera_position_base_m + self.rotation_base_from_camera @ _vec3(
            position_camera_m, "position_camera_m"
        )


@dataclass(frozen=True)
class RobotState:
    """Robot/turret state required by the trained test7_2_weight Actor."""

    turret_yaw_rad: float
    turret_pitch_rad: float
    turret_yaw_velocity_rps: float = 0.0
    turret_pitch_velocity_rps: float = 0.0
    base_linear_velocity_base_mps: Sequence[float] = (0.0, 0.0, 0.0)
    base_angular_velocity_base_rps: Sequence[float] = (0.0, 0.0, 0.0)
    projected_gravity_base: Sequence[float] = (0.0, 0.0, -1.0)
    turret_joint_limits_rad: Sequence[Sequence[float]] = ((-math.pi, math.pi), (-math.pi / 2, math.pi / 2))
    muzzle_yaw_base_rad: float | None = None
    muzzle_pitch_base_rad: float | None = None
    prev_action: Sequence[float] = (0.0, 0.0, 0.0)
    fire_intent: float = 0.0
    fire_cooldown_steps: int = 0

    def validate(self) -> None:
        for value, name in (
            (self.turret_yaw_rad, "turret_yaw_rad"),
            (self.turret_pitch_rad, "turret_pitch_rad"),
            (self.turret_yaw_velocity_rps, "turret_yaw_velocity_rps"),
            (self.turret_pitch_velocity_rps, "turret_pitch_velocity_rps"),
        ):
            _finite(value, name)
        _vec3(self.base_linear_velocity_base_mps, "base_linear_velocity_base_mps")
        _vec3(self.base_angular_velocity_base_rps, "base_angular_velocity_base_rps")
        _vec3(self.projected_gravity_base, "projected_gravity_base")
        limits = np.asarray(self.turret_joint_limits_rad, dtype=np.float64)
        if limits.shape != (2, 2) or not np.isfinite(limits).all() or np.any(limits[:, 1] <= limits[:, 0]):
            raise ValueError("turret_joint_limits_rad must be a finite [2, 2] min/max array")
        prev = np.asarray(self.prev_action, dtype=np.float64)
        if prev.shape != (3,) or not np.isfinite(prev).all():
            raise ValueError("prev_action must have shape (3,)")


@dataclass(frozen=True)
class ShootingChainConfig:
    """Normalization and fire-gating values copied from TurretAimEnvCfg."""

    target_distance_range_m: tuple[float, float] = (80.0, 150.0)
    target_speed_range_mps: tuple[float, float] = (0.6, 5.0)
    aim_error_obs_scale_rad: float = 0.05
    aim_error_obs_clip: float = 5.0
    joint_velocity_scale_rps: float = 5.0
    base_linear_velocity_scale_mps: float = 2.0
    base_angular_velocity_scale_rps: float = 3.0
    fire_intent_threshold: float = 0.65
    fire_intent_hold_steps: int = 2
    fire_cooldown_steps: int = 5
    require_tracking_status: tuple[str, ...] = ("initialized", "initial", "tracking", "predicted")

    def __post_init__(self) -> None:
        r_min, r_max = self.target_distance_range_m
        if r_max <= r_min or r_min <= 0.0:
            raise ValueError("target_distance_range_m must be positive and increasing")
        if self.aim_error_obs_scale_rad <= 0.0 or self.aim_error_obs_clip < 1.0:
            raise ValueError("invalid aim-error normalization")
        if self.fire_intent_hold_steps < 1 or self.fire_cooldown_steps < 0:
            raise ValueError("invalid fire-gating steps")


@dataclass(frozen=True)
class DecisionCommand:
    target_id: int | None
    action: np.ndarray
    fire_command: bool
    fire_intent: float
    prediction_ready: bool
    target_position_base_m: np.ndarray
    target_velocity_base_mps: np.ndarray
    observation: np.ndarray
    vision_status: str
    vision_source: str
    velocity_reliability_gate: float


def _soft_norm(values: np.ndarray, scale: float) -> np.ndarray:
    return np.tanh(values / max(float(scale), 1e-9))


def build_policy_observation(
    target_position_base_m: Sequence[float],
    target_velocity_base_mps: Sequence[float],
    robot: RobotState,
    *,
    prediction_ready: bool = True,
    config: ShootingChainConfig | None = None,
) -> np.ndarray:
    """Build the exact 28-D actor observation used by ``turret_aim_env_gt``."""

    cfg = config or ShootingChainConfig()
    robot.validate()
    position = _vec3(target_position_base_m, "target_position_base_m")
    velocity = _vec3(target_velocity_base_mps, "target_velocity_base_mps")
    if not prediction_ready:
        position = np.zeros(3, dtype=np.float64)
        velocity = np.zeros(3, dtype=np.float64)

    distance = float(np.linalg.norm(position))
    distance_valid = bool(prediction_ready and distance > 1e-6)
    distance_safe = max(distance, 1e-6)
    position_n = position / distance_safe if distance_valid else np.zeros(3)

    r_min, r_max = cfg.target_distance_range_m
    distance_n = (distance - 0.5 * (r_min + r_max)) / max(0.5 * (r_max - r_min), 1.0)
    distance_n = distance_n if distance_valid else 0.0

    target_speed_scale = max(
        abs(cfg.target_speed_range_mps[0]),
        abs(cfg.target_speed_range_mps[1]),
        1.0,
    )
    target_vel_n = _soft_norm(velocity, target_speed_scale + cfg.base_linear_velocity_scale_mps)

    x, y, z = position
    target_yaw = math.atan2(y, x) if distance_valid else 0.0
    target_pitch = math.atan2(-z, max(math.hypot(x, y), 1e-6)) if distance_valid else 0.0
    muzzle_yaw = robot.muzzle_yaw_base_rad if robot.muzzle_yaw_base_rad is not None else robot.turret_yaw_rad
    muzzle_pitch = robot.muzzle_pitch_base_rad if robot.muzzle_pitch_base_rad is not None else robot.turret_pitch_rad
    yaw_error_n = np.clip(_wrap_to_pi(target_yaw - muzzle_yaw) / cfg.aim_error_obs_scale_rad, -cfg.aim_error_obs_clip, cfg.aim_error_obs_clip)
    pitch_error_n = np.clip((target_pitch - muzzle_pitch) / cfg.aim_error_obs_scale_rad, -cfg.aim_error_obs_clip, cfg.aim_error_obs_clip)

    limits = np.asarray(robot.turret_joint_limits_rad, dtype=np.float64)
    centers = 0.5 * (limits[:, 0] + limits[:, 1])
    half_ranges = np.maximum(0.5 * (limits[:, 1] - limits[:, 0]), 1e-6)
    joint_pos_n = np.clip(
        (np.asarray([robot.turret_yaw_rad, robot.turret_pitch_rad]) - centers) / half_ranges,
        -1.0,
        1.0,
    )
    joint_vel_n = _soft_norm(
        np.asarray([robot.turret_yaw_velocity_rps, robot.turret_pitch_velocity_rps]),
        cfg.joint_velocity_scale_rps,
    )
    base_lin = _soft_norm(_vec3(robot.base_linear_velocity_base_mps, "base_linear_velocity_base_mps"), cfg.base_linear_velocity_scale_mps)
    base_ang = _soft_norm(_vec3(robot.base_angular_velocity_base_rps, "base_angular_velocity_base_rps"), cfg.base_angular_velocity_scale_rps)
    gravity = np.clip(_vec3(robot.projected_gravity_base, "projected_gravity_base"), -1.0, 1.0)
    moving = float(np.linalg.norm(np.asarray(robot.base_linear_velocity_base_mps, dtype=np.float64)[:2]) > 0.05)
    prev = np.clip(np.asarray(robot.prev_action, dtype=np.float64), -1.0, 1.0)
    intent = np.clip(float(robot.fire_intent), 0.0, 1.0)
    cooldown = np.clip(float(robot.fire_cooldown_steps) / max(float(cfg.fire_cooldown_steps), 1.0), 0.0, 1.0)

    observation = np.concatenate(
        [
            position_n,
            target_vel_n,
            np.asarray([distance_n, yaw_error_n, pitch_error_n]),
            joint_pos_n,
            joint_vel_n,
            base_lin,
            base_ang,
            gravity,
            np.asarray([moving]),
            prev,
            np.asarray([intent, cooldown]),
        ]
    ).astype(np.float32)
    if observation.shape != (28,) or not np.isfinite(observation).all():
        raise RuntimeError(f"Actor observation must be finite with shape (28,), got {observation.shape}")
    return observation


class ShootingChain:
    """Connect the existing vision, velocity and RL decision modules."""

    def __init__(
        self,
        vision_tracker: VisionTracker,
        velocity_estimator: VelocityEstimator,
        policy: Policy,
        *,
        camera_extrinsics: CameraExtrinsics | None = None,
        config: ShootingChainConfig | None = None,
    ) -> None:
        self.vision_tracker = vision_tracker
        self.velocity_estimator = velocity_estimator
        self.policy = policy
        self.camera_extrinsics = camera_extrinsics or CameraExtrinsics()
        self.config = config or ShootingChainConfig()
        self._fire_high_steps = 0
        self._cooldown_steps = 0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._last_target_id: int | None = None

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def reset(self, target_id: int | None = None) -> None:
        self._fire_high_steps = 0
        self._cooldown_steps = 0
        self._last_action.fill(0.0)
        self._last_target_id = None
        self.velocity_estimator.reset(target_id)
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()

    def initialize(
        self,
        image_rgb: np.ndarray,
        initial_boxes: Any,
        robot: RobotState,
        *,
        timestamp_s: float = 0.0,
        target_kinematics: Mapping[int, Any] | None = None,
    ) -> DecisionCommand:
        # A new tracker sequence also starts fresh velocity history, fire gate,
        # and recurrent policy state. Otherwise reused target IDs can inherit
        # stale GRU state or fail the estimator's monotonic timestamp check.
        self.reset()
        result = self.vision_tracker.initialize(
            image_rgb,
            initial_boxes,
            timestamp_s=_finite(timestamp_s, "timestamp_s"),
            target_kinematics=target_kinematics,
        )
        return self._decide(result, robot)

    def update(
        self,
        image_rgb: np.ndarray,
        robot: RobotState,
        *,
        dt_s: float,
        timestamp_s: float | None = None,
        target_kinematics: Mapping[int, Any] | None = None,
    ) -> DecisionCommand:
        dt = _finite(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        result = self.vision_tracker.update(
            image_rgb,
            dt_s=dt,
            timestamp_s=None if timestamp_s is None else _finite(timestamp_s, "timestamp_s"),
            target_kinematics=target_kinematics,
        )
        return self._decide(result, robot)

    def _update_velocity(self, target_id: int, timestamp_s: float, robot: RobotState,
                         position_base: Any, source: str, range_valid: bool,
                         quality: Sequence[float] = (0.0,) * 7) -> Any:
        from types import SimpleNamespace

        return self.velocity_estimator.update(target_id, SimpleNamespace(
            timestamp_s=timestamp_s, position_base_m=np.asarray(position_base).tolist(),
            robot_linear_velocity_base_mps=_vec3(robot.base_linear_velocity_base_mps, "base_linear_velocity_base_mps").tolist(),
            robot_angular_velocity_base_rps=_vec3(robot.base_angular_velocity_base_rps, "base_angular_velocity_base_rps").tolist(),
            yaw_rate_rps=float(robot.base_angular_velocity_base_rps[2]), source=source,
            quality=quality, range_valid=range_valid,
        ))

    def _decide(self, frame_result: Any, robot: RobotState) -> DecisionCommand:
        robot.validate()
        target = getattr(frame_result, "selected_target", None)
        preserve_missing = bool(getattr(self.velocity_estimator, "preserve_state_on_missing", False))
        if target is None:
            self._fire_high_steps = 0
            if self._last_target_id is not None:
                if preserve_missing:
                    self._update_velocity(self._last_target_id, float(frame_result.timestamp_s), robot,
                                          np.full(3, np.nan), "lost", False)
                else:
                    self.velocity_estimator.reset(self._last_target_id)
            if not preserve_missing:
                self._last_target_id = None
            self._cooldown_steps = max(self._cooldown_steps - 1, 0)
            empty = np.zeros(3, dtype=np.float32)
            return DecisionCommand(None, empty, False, 0.0, False, empty, empty, np.zeros(28, dtype=np.float32), "lost", "invalid", 0.0)

        target_id = int(target.target_id)
        if self._last_target_id is not None and target_id != self._last_target_id:
            self._fire_high_steps = 0
        self._last_target_id = target_id

        # The tracker reports angles in camera coordinates and range from its
        # configured kinematics source. Reconstruct the 3-D camera ray here.
        distance = float(getattr(target, "distance_m", 0.0))
        if not math.isfinite(distance) or distance <= 0.0:
            distance = 0.0
        yaw_cam = float(getattr(target, "yaw_rad"))
        pitch_cam = float(getattr(target, "pitch_rad"))
        cp, sp = math.cos(pitch_cam), math.sin(pitch_cam)
        cy, sy = math.cos(yaw_cam), math.sin(yaw_cam)
        position_cam = np.asarray([distance * cp * sy, -distance * sp, distance * cp * cy])
        position_base = self.camera_extrinsics.camera_to_base(position_cam) if distance > 0.0 else np.zeros(3)

        source = str(getattr(target, "source", "invalid"))
        status = str(getattr(target, "status", "lost"))
        ready = bool(distance > 0.0 and status in self.config.require_tracking_status)
        reliability_gate = 0.0
        velocity = np.zeros(3, dtype=np.float64)
        if ready or preserve_missing:
            quality = tracker_quality_features(
                target, getattr(self.vision_tracker, "config", None)
            )
            estimate = self._update_velocity(
                target_id, float(frame_result.timestamp_s), robot,
                position_base if ready else np.full(3, np.nan), source, ready, quality,
            )
            if estimate is None:
                ready = False
            else:
                raw_velocity = getattr(estimate, "velocity_mean_mps")
                velocity = np.asarray(raw_velocity.detach().cpu().numpy() if hasattr(raw_velocity, "detach") else raw_velocity, dtype=np.float64).reshape(3)
                reliability_gate = float(getattr(estimate, "reliability_gate", 0.0))
        else:
            self.velocity_estimator.reset(target_id)

        policy_obs = build_policy_observation(position_base, velocity, robot, prediction_ready=ready, config=self.config)
        raw_action = np.asarray(self.policy(policy_obs), dtype=np.float64).reshape(-1)
        if raw_action.shape != (3,) or not np.isfinite(raw_action).all():
            raise RuntimeError(f"Decision policy must return 3 finite actions, got {raw_action.shape}")
        action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
        fire_intent = float(np.clip((float(action[2]) + 1.0) * 0.5, 0.0, 1.0))
        if ready and fire_intent > self.config.fire_intent_threshold:
            self._fire_high_steps = min(self._fire_high_steps + 1, self.config.fire_intent_hold_steps)
        else:
            self._fire_high_steps = 0
        fire_now = bool(
            ready
            and self._cooldown_steps <= 0
            and self._fire_high_steps >= self.config.fire_intent_hold_steps
        )
        if fire_now:
            self._cooldown_steps = self.config.fire_cooldown_steps
        else:
            self._cooldown_steps = max(self._cooldown_steps - 1, 0)
        self._last_action = action.copy()
        return DecisionCommand(
            target_id=target_id,
            action=action,
            fire_command=fire_now,
            fire_intent=fire_intent,
            prediction_ready=ready,
            target_position_base_m=position_base.astype(np.float32),
            target_velocity_base_mps=velocity.astype(np.float32),
            observation=policy_obs,
            vision_status=status,
            vision_source=source,
            velocity_reliability_gate=reliability_gate,
        )

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Tuple

import cv2
import numpy as np

from .camera_geometry import CameraIntrinsics


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class MultiTargetConfig:
    """Parameters for sparse-flow tracking and engagement decisions."""

    min_corners: int = 9
    max_corners: int = 16
    corner_redetect_interval_frames: int = 3
    corner_quality: float = 0.01
    corner_min_distance_px: float = 4.0
    lk_window_px: int = 21
    lk_max_level: int = 3
    fb_max_error_px: float = 1.5
    ransac_reproj_threshold_px: float = 2.5
    min_ransac_inliers: int = 4
    min_motion_scale: float = 0.65
    max_motion_scale: float = 1.50
    # Optional cumulative scale guard for fixed-distance, static-camera pools.
    # Moving-target/general tracking leaves this disabled.
    max_initial_scale_ratio: float | None = None
    confidence_smoothing: float = 0.35
    confidence_decay: float = 0.72
    max_missed_frames: int = 8
    process_noise: float = 8.0
    measurement_noise: float = 4.0
    # Reject an LK measurement only when it is both far from the Kalman prior
    # and essentially disjoint from it.  The conjunction keeps legitimate
    # fast motion from being rejected merely because one metric is marginal.
    measurement_rejection_innovation_threshold: float = 0.80
    measurement_rejection_iou_threshold: float = 0.05
    prediction_horizon_s: float = 0.08
    target_width_m: float = 0.70
    target_height_m: float = 0.70
    kinematic_estimation_horizon_s: float = 0.10
    priority_confidence_weight: float = 0.35
    priority_center_weight: float = 0.20
    priority_area_weight: float = 0.05
    priority_stability_weight: float = 0.10
    priority_distance_weight: float = 0.20
    priority_speed_weight: float = 0.10
    priority_distance_near_m: float = 80.0
    priority_distance_far_m: float = 150.0
    priority_speed_reference_mps: float = 5.0
    priority_switch_margin: float = 0.08
    selection_policy: str = "priority"
    fine_tracker_interval_frames: int = 3
    fine_tracker_min_score: float = 0.20
    fine_tracker_policy: str = "fixed"
    adaptive_min_interval_frames: int = 2
    adaptive_max_interval_frames: int = 12
    adaptive_bad_frames: int = 2
    adaptive_warning_confidence: float = 0.75
    adaptive_critical_confidence: float = 0.55
    adaptive_risk_threshold: float = 0.55
    adaptive_critical_risk_threshold: float = 0.85
    adaptive_risk_clear_threshold: float = 0.35
    adaptive_desired_inliers: int = 9
    adaptive_innovation_critical: float = 0.75
    adaptive_prediction_iou_threshold: float = 0.55
    adaptive_failure_backoff_initial_frames: int = 1
    adaptive_failure_backoff_max_frames: int = 8
    # OSTrack refines an already current-frame LK estimate.  Its correction
    # therefore needs to be much smaller than the broad re-detection gate that
    # was previously used here, especially for 10--20 px distant targets.
    fine_max_center_jump_box_diagonals: float = 0.50
    fine_discontinuity_center_jump_box_diagonals: float = 0.25
    fine_max_scale_change: float = 1.75

    def __post_init__(self) -> None:
        if self.selection_policy not in {"priority", "nearest_until_removed"}:
            raise ValueError("Unknown target selection_policy")
        if not 9 <= self.min_corners <= self.max_corners:
            raise ValueError("Require 9 <= min_corners <= max_corners")
        if not 9 <= self.max_corners <= 16:
            raise ValueError("max_corners must be in [9, 16]")
        if self.corner_redetect_interval_frames < 1:
            raise ValueError("corner_redetect_interval_frames must be positive")
        if self.lk_window_px <= 1 or self.lk_max_level < 0:
            raise ValueError("Invalid Lucas-Kanade pyramid parameters")
        if self.fb_max_error_px <= 0.0 or self.ransac_reproj_threshold_px <= 0.0:
            raise ValueError("Optical-flow error thresholds must be positive")
        if self.min_ransac_inliers < 3:
            raise ValueError("min_ransac_inliers must be at least 3")
        if not 0.0 < self.min_motion_scale <= 1.0 <= self.max_motion_scale:
            raise ValueError("Motion scale bounds must contain 1.0")
        if self.max_initial_scale_ratio is not None and (
            not math.isfinite(self.max_initial_scale_ratio) or self.max_initial_scale_ratio <= 1.0
        ):
            raise ValueError("max_initial_scale_ratio must exceed 1")
        if not 0.0 < self.confidence_smoothing <= 1.0:
            raise ValueError("confidence_smoothing must be in (0, 1]")
        if not 0.0 < self.confidence_decay < 1.0:
            raise ValueError("confidence_decay must be in (0, 1)")
        if self.max_missed_frames < 1:
            raise ValueError("max_missed_frames must be positive")
        if not 0.0 < self.corner_quality <= 1.0 or self.corner_min_distance_px <= 0.0:
            raise ValueError("Corner quality and spacing must be positive")
        if self.process_noise <= 0.0 or self.measurement_noise <= 0.0:
            raise ValueError("Kalman noise values must be positive")
        if self.measurement_rejection_innovation_threshold <= 0.0:
            raise ValueError(
                "measurement_rejection_innovation_threshold must be positive"
            )
        if not 0.0 <= self.measurement_rejection_iou_threshold <= 1.0:
            raise ValueError(
                "measurement_rejection_iou_threshold must be in [0, 1]"
            )
        if self.prediction_horizon_s < 0.0:
            raise ValueError("prediction_horizon_s must be non-negative")
        if self.target_width_m <= 0.0 or self.target_height_m <= 0.0:
            raise ValueError("Target physical dimensions must be positive")
        if self.kinematic_estimation_horizon_s <= 0.0:
            raise ValueError("kinematic_estimation_horizon_s must be positive")
        priority_weights = (
            self.priority_confidence_weight,
            self.priority_center_weight,
            self.priority_area_weight,
            self.priority_stability_weight,
            self.priority_distance_weight,
            self.priority_speed_weight,
        )
        if not all(math.isfinite(weight) for weight in priority_weights):
            raise ValueError("Priority weights must be finite")
        if sum(abs(weight) for weight in priority_weights) <= 0.0:
            raise ValueError("At least one priority weight must be non-zero")
        if self.priority_distance_near_m < 0.0:
            raise ValueError("priority_distance_near_m must be non-negative")
        if self.priority_distance_far_m <= self.priority_distance_near_m:
            raise ValueError("priority_distance_far_m must exceed priority_distance_near_m")
        if self.priority_speed_reference_mps <= 0.0:
            raise ValueError("priority_speed_reference_mps must be positive")
        if self.priority_switch_margin < 0.0:
            raise ValueError("priority_switch_margin must be non-negative")
        if self.fine_tracker_interval_frames < 1:
            raise ValueError("fine_tracker_interval_frames must be positive")
        if not 0.0 <= self.fine_tracker_min_score <= 1.0:
            raise ValueError("fine_tracker_min_score must be in [0, 1]")
        if self.fine_tracker_policy not in {"fixed", "adaptive"}:
            raise ValueError("fine_tracker_policy must be 'fixed' or 'adaptive'")
        if self.adaptive_min_interval_frames < 1:
            raise ValueError("adaptive_min_interval_frames must be positive")
        if self.adaptive_max_interval_frames < self.adaptive_min_interval_frames:
            raise ValueError(
                "adaptive_max_interval_frames must be at least adaptive_min_interval_frames"
            )
        if self.adaptive_bad_frames < 1:
            raise ValueError("adaptive_bad_frames must be positive")
        unit_interval_fields = {
            "adaptive_warning_confidence": self.adaptive_warning_confidence,
            "adaptive_critical_confidence": self.adaptive_critical_confidence,
            "adaptive_risk_threshold": self.adaptive_risk_threshold,
            "adaptive_critical_risk_threshold": self.adaptive_critical_risk_threshold,
            "adaptive_risk_clear_threshold": self.adaptive_risk_clear_threshold,
            "adaptive_prediction_iou_threshold": self.adaptive_prediction_iou_threshold,
        }
        for name, value in unit_interval_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.adaptive_critical_confidence >= self.adaptive_warning_confidence:
            raise ValueError(
                "adaptive_critical_confidence must be below adaptive_warning_confidence"
            )
        if self.adaptive_risk_clear_threshold >= self.adaptive_risk_threshold:
            raise ValueError(
                "adaptive_risk_clear_threshold must be below adaptive_risk_threshold"
            )
        if self.adaptive_critical_risk_threshold < self.adaptive_risk_threshold:
            raise ValueError(
                "adaptive_critical_risk_threshold must be at least adaptive_risk_threshold"
            )
        if self.adaptive_desired_inliers < self.min_ransac_inliers:
            raise ValueError(
                "adaptive_desired_inliers must be at least min_ransac_inliers"
            )
        if self.adaptive_innovation_critical <= 0.0:
            raise ValueError("adaptive_innovation_critical must be positive")
        if self.adaptive_failure_backoff_initial_frames < 1:
            raise ValueError("adaptive_failure_backoff_initial_frames must be positive")
        if (
            self.adaptive_failure_backoff_max_frames
            < self.adaptive_failure_backoff_initial_frames
        ):
            raise ValueError(
                "adaptive_failure_backoff_max_frames must be at least the initial backoff"
            )
        if self.fine_max_center_jump_box_diagonals <= 0.0:
            raise ValueError("fine_max_center_jump_box_diagonals must be positive")
        if not (
            0.0
            < self.fine_discontinuity_center_jump_box_diagonals
            <= self.fine_max_center_jump_box_diagonals
        ):
            raise ValueError(
                "fine_discontinuity_center_jump_box_diagonals must be positive "
                "and no greater than fine_max_center_jump_box_diagonals"
            )
        if self.fine_max_scale_change <= 1.0:
            raise ValueError("fine_max_scale_change must exceed 1")


@dataclass(frozen=True)
class FineTrackResult:
    bbox_xyxy: BBox | None
    score: float | None = None
    ok: bool = True


class FineTracker(Protocol):
    """Minimal interface used to hand only the selected target to OSTrack."""

    def initialize(self, image_rgb: np.ndarray, bbox_xyxy: Iterable[float]) -> None: ...

    def track(self, image_rgb: np.ndarray) -> FineTrackResult: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class TargetKinematics:
    distance_m: float
    speed_m_s: float
    source: str = "external"

    def __post_init__(self) -> None:
        if not math.isfinite(self.distance_m) or self.distance_m < 0.0:
            raise ValueError("distance_m must be finite and non-negative")
        if not math.isfinite(self.speed_m_s) or self.speed_m_s < 0.0:
            raise ValueError("speed_m_s must be finite and non-negative")
        if not self.source:
            raise ValueError("Target kinematics source must be non-empty")


@dataclass(frozen=True)
class TargetEstimate:
    target_id: int
    bbox_xyxy: BBox
    predicted_bbox_xyxy: BBox
    center_xy: tuple[float, float]
    predicted_center_xy: tuple[float, float]
    yaw_rad: float
    pitch_rad: float
    predicted_yaw_rad: float
    predicted_pitch_rad: float
    distance_m: float
    speed_m_s: float
    kinematics_source: str
    distance_priority_score: float
    speed_priority_score: float
    confidence: float
    priority: float
    status: str
    source: str
    corners: int
    ransac_inliers: int
    median_fb_error_px: float | None
    missed_frames: int
    selected: bool
    fine_score: float | None = None
    motion_confidence: float = 0.0
    point_retention: float = 0.0
    ransac_inlier_ratio: float = 0.0
    bbox_innovation: float = 0.0
    bbox_iou_with_prediction: float = 1.0
    fine_policy: str = "disabled"
    fine_action: str = "none"
    fine_trigger_reason: str = "not_selected"
    fine_risk: float = 0.0
    fine_bad_frame_streak: int = 0
    frames_since_fine_attempt: int = -1
    frames_since_fine_success: int = -1
    fine_attempted: bool = False
    fine_applied: bool = False
    fine_latency_ms: float = 0.0
    fine_failure_count: int = 0
    fine_correction_center_px: float = 0.0
    fine_correction_iou: float = 1.0
    measurement_discontinuity: bool = False
    measurement_rejected: bool = False

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self.pitch_rad)


@dataclass(frozen=True)
class MultiTargetFrameResult:
    frame_index: int
    timestamp_s: float
    targets: tuple[TargetEstimate, ...]
    selected_target_id: int | None

    @property
    def selected_target(self) -> TargetEstimate | None:
        for target in self.targets:
            if target.selected:
                return target
        return None


class _BoxKalman:
    """Constant-velocity Kalman filter over center, size, and their velocities."""

    def __init__(self, bbox_xyxy: Iterable[float], process_noise: float, measurement_noise: float) -> None:
        cx, cy, width, height = _box_to_measurement(bbox_xyxy)
        self.x = np.asarray([cx, cy, width, height, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # Velocity is initially unknown in pixel/second units, so its prior
        # variance must be large enough to learn RobotMaster-speed motion in a
        # few camera frames.
        self.p = np.diag(
            [16.0, 16.0, 16.0, 16.0, 10000.0, 10000.0, 2500.0, 2500.0]
        )
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

    def predict(self, dt: float) -> BBox:
        dt = max(float(dt), 1e-4)
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4, dtype=np.float64) * dt
        q_base = np.asarray(
            [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]],
            dtype=np.float64,
        )
        process_cov = np.zeros((8, 8), dtype=np.float64)
        for index in range(4):
            process_cov[np.ix_([index, index + 4], [index, index + 4])] = q_base * self.process_noise
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + process_cov
        self.x[2:4] = np.maximum(self.x[2:4], 2.0)
        return self.bbox()

    def correct(
        self,
        bbox_xyxy: Iterable[float],
        noise_scale: float = 1.0,
        *,
        snap_to_measurement: bool = True,
        preserve_velocity: bool = False,
    ) -> BBox:
        measurement = np.asarray(_box_to_measurement(bbox_xyxy), dtype=np.float64)
        observation = np.zeros((4, 8), dtype=np.float64)
        observation[:, :4] = np.eye(4, dtype=np.float64)
        measurement_cov = np.eye(4, dtype=np.float64) * self.measurement_noise * max(float(noise_scale), 0.1)
        innovation = measurement - observation @ self.x
        innovation_cov = observation @ self.p @ observation.T + measurement_cov
        gain = np.linalg.solve(innovation_cov.T, (self.p @ observation.T).T).T
        previous_velocity = self.x[4:].copy()
        self.x = self.x + gain @ innovation
        identity = np.eye(8, dtype=np.float64)
        # Joseph form prevents covariance from becoming asymmetric after many frames.
        residual = identity - gain @ observation
        self.p = residual @ self.p @ residual.T + gain @ measurement_cov @ gain.T
        # LK is an incremental, sub-pixel current-frame measurement and remains
        # the exact box used by the fast path.  An occasional absolute OSTrack
        # correction can be discontinuous, so its caller keeps the Kalman-
        # blended state instead of snapping the position to the measurement.
        if snap_to_measurement:
            self.x[:4] = measurement
        if preserve_velocity:
            # An absolute re-localization correction is not an instantaneous
            # target motion measurement.  Let it correct position/scale without
            # injecting the correction jump into the box velocity state.
            self.x[4:] = previous_velocity
        self.x[2:4] = np.maximum(self.x[2:4], 2.0)
        return self.bbox()

    def bbox(self, horizon_s: float = 0.0) -> BBox:
        state = self.x.copy()
        state[:4] += state[4:] * max(float(horizon_s), 0.0)
        cx, cy = state[:2]
        width, height = np.maximum(state[2:4], 2.0)
        return (
            float(cx - 0.5 * width),
            float(cy - 0.5 * height),
            float(cx + 0.5 * width),
            float(cy + 0.5 * height),
        )


@dataclass
class _TargetState:
    target_id: int
    bbox_xyxy: BBox
    kalman: _BoxKalman
    points: np.ndarray
    initial_size: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.75
    priority_bias: float = 0.0
    missed_frames: int = 0
    status: str = "initialized"
    source: str = "initial"
    ransac_inliers: int = 0
    median_fb_error_px: float | None = None
    fine_score: float | None = None
    external_kinematics: TargetKinematics | None = None
    last_motion_confidence: float = 0.0
    last_point_retention: float = 0.0
    last_inlier_ratio: float = 0.0
    last_bbox_innovation: float = 0.0
    last_bbox_iou_with_prediction: float = 1.0
    fine_action: str = "none"
    fine_trigger_reason: str = "not_selected"
    fine_risk: float = 0.0
    fine_attempted: bool = False
    fine_applied: bool = False
    fine_latency_ms: float = 0.0
    fine_correction_center_px: float = 0.0
    fine_correction_iou: float = 1.0
    measurement_discontinuity: bool = False
    measurement_rejected: bool = False


class MultiTargetVisionTracker:
    """Track many boxes cheaply and refine only the selected ID with OSTrack.

    All target points are concatenated into one forward/backward Lucas-Kanade
    call. Motion models and Kalman states remain independent per target ID.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        config: MultiTargetConfig | None = None,
        fine_tracker: FineTracker | None = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.config = config or MultiTargetConfig()
        self.fine_tracker = fine_tracker
        self._states: dict[int, _TargetState] = {}
        self._previous_gray: np.ndarray | None = None
        self._selected_target_id: int | None = None
        self._fine_target_id: int | None = None
        self._fine_last_run_frame_index: int | None = None
        self._fine_last_attempt_frame: int | None = None
        self._fine_last_success_frame: int | None = None
        self._fine_bad_frame_streak = 0
        self._fine_failure_count = 0
        self._fine_backoff_until_frame = 0
        self._fine_last_trigger_reason = "none"
        self._fine_scheduler_target_id: int | None = None
        self._frame_index = -1
        self._timestamp_s = 0.0
        self._image_shape: tuple[int, int] | None = None

    @property
    def initialized(self) -> bool:
        return self._previous_gray is not None and bool(self._states)

    def initialize(
        self,
        image_rgb: np.ndarray,
        initial_boxes: Mapping[int, Iterable[float]] | Iterable[Iterable[float]],
        *,
        priority_biases: Mapping[int, float] | None = None,
        target_kinematics: Mapping[
            int, TargetKinematics | tuple[float, float]
        ]
        | None = None,
        timestamp_s: float = 0.0,
    ) -> MultiTargetFrameResult:
        image = _as_rgb_uint8(image_rgb)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        boxes = _normalize_initial_boxes(initial_boxes)
        if not boxes:
            raise ValueError("At least one initial target box is required")
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")

        self.reset()
        self._image_shape = (height, width)
        for target_id, raw_box in boxes.items():
            bbox = _clip_box(raw_box, width, height)
            points = self._extract_points(gray, bbox)
            bias = 0.0 if priority_biases is None else float(priority_biases.get(target_id, 0.0))
            self._states[target_id] = _TargetState(
                target_id=target_id,
                bbox_xyxy=bbox,
                kalman=_BoxKalman(bbox, self.config.process_noise, self.config.measurement_noise),
                points=points,
                initial_size=_box_to_measurement(bbox)[2:],
                priority_bias=bias,
            )

        self._set_target_kinematics(target_kinematics)
        self._previous_gray = gray
        self._frame_index = 0
        self._timestamp_s = float(timestamp_s)
        priorities = self._compute_priorities(width, height)
        self._selected_target_id = self._choose_target(priorities)
        self._initialize_fine_tracker(image, trigger_reason="initialization")
        return self._build_result(priorities)

    def update(
        self,
        image_rgb: np.ndarray,
        *,
        dt_s: float,
        timestamp_s: float | None = None,
        target_kinematics: Mapping[
            int, TargetKinematics | tuple[float, float]
        ]
        | None = None,
    ) -> MultiTargetFrameResult:
        if not self.initialized or self._previous_gray is None:
            raise RuntimeError("Call initialize() before update()")
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        image = _as_rgb_uint8(image_rgb)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        if self._image_shape != (height, width):
            raise ValueError("Input image size changed after tracker initialization")

        self._frame_index += 1
        next_timestamp = self._timestamp_s + dt_s if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(next_timestamp):
            raise ValueError("timestamp_s must be finite")
        self._timestamp_s = next_timestamp
        self._set_target_kinematics(target_kinematics)
        previous_boxes = {
            target_id: state.bbox_xyxy for target_id, state in self._states.items()
        }
        predicted_boxes: dict[int, BBox] = {}
        for state in self._states.values():
            predicted_box = _clip_box(state.kalman.predict(dt_s), width, height)
            if not self._scale_is_plausible(predicted_box, state):
                # Bound fallback predictions as well as measurements. An old
                # positive size velocity must not keep inflating a rejected box.
                cx, cy, bw, bh = _box_to_measurement(predicted_box)
                ratio = self.config.max_initial_scale_ratio
                iw, ih = state.initial_size
                bw, bh = np.clip([bw, bh], [iw / ratio, ih / ratio], [iw * ratio, ih * ratio])
                state.kalman.x[2:4] = [bw, bh]
                state.kalman.x[6:8] = 0.0
                predicted_box = _clip_box((cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2), width, height)
            predicted_boxes[state.target_id] = predicted_box
            state.bbox_xyxy = predicted_box
            state.source = "prediction"
            state.fine_score = None
            state.ransac_inliers = 0
            state.median_fb_error_px = None
            state.last_motion_confidence = 0.0
            state.last_point_retention = 0.0
            state.last_inlier_ratio = 0.0
            state.last_bbox_innovation = 1.0
            state.last_bbox_iou_with_prediction = 0.0
            state.fine_action = "none"
            state.fine_trigger_reason = "not_selected"
            state.fine_risk = 0.0
            state.fine_attempted = False
            state.fine_applied = False
            state.fine_latency_ms = 0.0
            state.fine_correction_center_px = 0.0
            state.fine_correction_iou = 1.0
            state.measurement_discontinuity = False
            state.measurement_rejected = False

        grouped_flow = self._track_all_points(self._previous_gray, gray)
        for target_id, state in self._states.items():
            flow = grouped_flow.get(target_id)
            measurement = self._fit_motion(
                previous_boxes[target_id], flow, width, height
            )
            if measurement is None:
                state.points = np.empty((0, 2), dtype=np.float32)
                state.missed_frames += 1
                state.confidence *= self.config.confidence_decay
                state.status = (
                    "lost" if state.missed_frames > self.config.max_missed_frames else "predicted"
                )
                continue

            (
                bbox,
                motion_confidence,
                inliers,
                median_fb,
                tracked_points,
                point_retention,
                inlier_ratio,
            ) = measurement
            bbox_innovation = _bbox_innovation(
                bbox, predicted_boxes[target_id]
            )
            bbox_iou = _box_iou(bbox, predicted_boxes[target_id])
            state.ransac_inliers = inliers
            state.median_fb_error_px = median_fb
            state.last_motion_confidence = motion_confidence
            state.last_point_retention = point_retention
            state.last_inlier_ratio = inlier_ratio
            state.last_bbox_innovation = bbox_innovation
            state.last_bbox_iou_with_prediction = bbox_iou

            reject_measurement = (
                bbox_innovation
                >= self.config.measurement_rejection_innovation_threshold
                and bbox_iou
                <= self.config.measurement_rejection_iou_threshold
            ) or not self._scale_is_plausible(bbox, state)
            if reject_measurement:
                # The Kalman filter has already completed predict(), but the
                # implausible visual measurement must not enter correct().
                # Keep the predicted box for this frame and expose the failed
                # consistency metrics so downstream quality gating can see why
                # the source was downgraded.
                state.points = np.empty((0, 2), dtype=np.float32)
                state.missed_frames += 1
                state.confidence *= self.config.confidence_decay
                state.status = (
                    "lost"
                    if state.missed_frames > self.config.max_missed_frames
                    else "predicted"
                )
                state.source = "prediction"
                state.measurement_rejected = True
                continue

            noise_scale = 1.0 + 3.0 * (1.0 - motion_confidence)
            state.kalman.correct(bbox, noise_scale)
            # Use the image measurement for the current frame. The Kalman state
            # supplies velocity, look-ahead, and fallback boxes when vision fails.
            state.bbox_xyxy = bbox
            alpha = self.config.confidence_smoothing
            state.confidence = float(np.clip((1.0 - alpha) * state.confidence + alpha * motion_confidence, 0.0, 1.0))
            state.missed_frames = 0
            state.status = "tracking"
            state.source = "flow"
            # These are the current-frame LK points that survived both the
            # forward/backward test and the per-target RANSAC fit. Reuse them
            # on the next frame instead of detecting a fresh set every time.
            state.points = tracked_points

        priorities = self._compute_priorities(width, height)
        self._selected_target_id = self._choose_target(priorities)
        self._run_fine_tracker(image, width, height)

        # Re-detect periodically, or immediately when the reusable set became
        # too small/invalid. The latter prevents a failed flow fit, a recovery,
        # or a large OSTrack correction from carrying stale points forward.
        periodic_redetect = (
            self._frame_index % self.config.corner_redetect_interval_frames == 0
        )
        for state in self._states.values():
            if state.status == "lost":
                state.points = np.empty((0, 2), dtype=np.float32)
                continue
            state.points = self._points_inside_box(state.points, state.bbox_xyxy)
            needs_redetect = (
                periodic_redetect
                or state.status == "predicted"
                or len(state.points) < self.config.min_corners
            )
            if needs_redetect:
                state.points = self._extract_points(gray, state.bbox_xyxy)

        priorities = self._compute_priorities(width, height)
        self._previous_gray = gray
        return self._build_result(priorities)

    def set_priority_bias(self, target_id: int, bias: float) -> None:
        """Add application knowledge such as armor class or threat level."""
        if int(target_id) not in self._states:
            raise KeyError(f"Unknown target ID: {target_id}")
        if not math.isfinite(bias):
            raise ValueError("Priority bias must be finite")
        self._states[int(target_id)].priority_bias = float(bias)

    def remove_targets(self, target_ids: Iterable[int]) -> None:
        """Remove confirmed destroyed IDs while retaining all other track histories."""
        removed = {int(target_id) for target_id in target_ids}
        for target_id in removed:
            self._states.pop(target_id, None)
        if self._selected_target_id in removed:
            self._selected_target_id = None
        if self._fine_target_id in removed:
            if self.fine_tracker is not None:
                self.fine_tracker.reset()
            self._fine_target_id = None
            self._reset_fine_scheduler()

    def set_selection_candidates(self, target_ids: Iterable[int] | None) -> None:
        """Restrict engagement without deleting or resetting any visual tracks."""
        self._selection_candidates = None if target_ids is None else {int(i) for i in target_ids}

    def selectable_target_distances(self) -> dict[int, float]:
        """Return RGB-derived distances of surviving, non-lost visual tracks."""
        return {
            target_id: self._resolve_kinematics(state).distance_m
            for target_id, state in self._states.items() if state.status != "lost"
        }

    def _scale_is_plausible(self, bbox: BBox, state: _TargetState) -> bool:
        ratio = self.config.max_initial_scale_ratio
        if ratio is None:
            return True
        _, _, width, height = _box_to_measurement(bbox)
        iw, ih = state.initial_size
        return iw / ratio <= width <= iw * ratio and ih / ratio <= height <= ih * ratio

    def _set_target_kinematics(
        self,
        measurements: Mapping[int, TargetKinematics | tuple[float, float]] | None,
    ) -> None:
        for state in self._states.values():
            state.external_kinematics = None
        if measurements is None:
            return
        for raw_target_id, raw_measurement in measurements.items():
            target_id = int(raw_target_id)
            if target_id not in self._states:
                raise KeyError(f"Unknown target ID in kinematics input: {target_id}")
            if isinstance(raw_measurement, TargetKinematics):
                measurement = raw_measurement
            else:
                try:
                    distance_m, speed_m_s = raw_measurement
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Target kinematics tuple must contain distance_m and speed_m_s"
                    ) from exc
                measurement = TargetKinematics(
                    distance_m=float(distance_m),
                    speed_m_s=float(speed_m_s),
                )
            self._states[target_id].external_kinematics = measurement

    def reset(self) -> None:
        if self.fine_tracker is not None:
            self.fine_tracker.reset()
        self._states.clear()
        self._previous_gray = None
        self._selected_target_id = None
        self._selection_candidates = None
        self._fine_target_id = None
        self._fine_last_run_frame_index = None
        self._reset_fine_scheduler()
        self._frame_index = -1
        self._timestamp_s = 0.0
        self._image_shape = None

    def _extract_points(self, gray: np.ndarray, bbox_xyxy: BBox) -> np.ndarray:
        height, width = gray.shape
        x1, y1, x2, y2 = _clip_box(bbox_xyxy, width, height)
        ix1 = max(0, int(math.floor(x1)) + 1)
        iy1 = max(0, int(math.floor(y1)) + 1)
        ix2 = min(width, int(math.ceil(x2)) - 1)
        iy2 = min(height, int(math.ceil(y2)) - 1)
        if ix2 <= ix1 or iy2 <= iy1:
            return np.empty((0, 2), dtype=np.float32)
        roi = np.ascontiguousarray(gray[iy1:iy2, ix1:ix2])
        detected = None
        if roi.shape[0] >= 3 and roi.shape[1] >= 3:
            detected = cv2.goodFeaturesToTrack(
                roi,
                mask=None,
                maxCorners=self.config.max_corners,
                qualityLevel=self.config.corner_quality,
                minDistance=self.config.corner_min_distance_px,
                blockSize=3,
                useHarrisDetector=False,
            )
        points = (
            np.empty((0, 2), dtype=np.float32)
            if detected is None
            else detected.reshape(-1, 2).astype(np.float32)
            + np.asarray([ix1, iy1], dtype=np.float32)
        )
        if len(points) >= self.config.min_corners:
            return points[: self.config.max_corners]

        # Low-texture armor plates still need a spatially distributed LK set.
        # Fill missing locations with an interior grid; FB checking will reject
        # locations that cannot be tracked reliably.
        grid_side = max(3, int(math.ceil(math.sqrt(self.config.min_corners))))
        grid_x = np.linspace(ix1, max(ix1, ix2 - 1), grid_side, dtype=np.float32)
        grid_y = np.linspace(iy1, max(iy1, iy2 - 1), grid_side, dtype=np.float32)
        candidates = np.asarray([(x, y) for y in grid_y for x in grid_x], dtype=np.float32)
        result = [point for point in points]
        minimum_spacing = max(1.0, 0.5 * self.config.corner_min_distance_px)
        for candidate in candidates:
            if result and min(float(np.linalg.norm(candidate - point)) for point in result) < minimum_spacing:
                continue
            result.append(candidate)
            if len(result) >= self.config.min_corners:
                break
        return np.asarray(result[: self.config.max_corners], dtype=np.float32).reshape(-1, 2)

    @staticmethod
    def _points_inside_box(points: np.ndarray, bbox_xyxy: BBox) -> np.ndarray:
        """Keep reusable current-frame points that still belong to the box."""
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        x1, y1, x2, y2 = bbox_xyxy
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        inside = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= float(x1))
            & (points[:, 0] <= float(x2))
            & (points[:, 1] >= float(y1))
            & (points[:, 1] <= float(y2))
        )
        return np.ascontiguousarray(points[inside], dtype=np.float32)

    def _track_all_points(
        self, previous_gray: np.ndarray, gray: np.ndarray
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        owners: list[tuple[int, int, int]] = []
        point_batches: list[np.ndarray] = []
        offset = 0
        for target_id, state in self._states.items():
            count = len(state.points)
            if count == 0 or state.status == "lost":
                continue
            point_batches.append(state.points)
            owners.append((target_id, offset, offset + count))
            offset += count
        if not point_batches:
            return {}

        previous_points = np.concatenate(point_batches, axis=0).reshape(-1, 1, 2).astype(np.float32)
        lk_params = {
            "winSize": (self.config.lk_window_px, self.config.lk_window_px),
            "maxLevel": self.config.lk_max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, previous_points, None, **lk_params
        )
        if next_points is None or forward_status is None:
            return {}
        back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous_gray, next_points, None, **lk_params
        )
        if back_points is None or backward_status is None:
            return {}

        old_flat = previous_points.reshape(-1, 2)
        new_flat = next_points.reshape(-1, 2)
        back_flat = back_points.reshape(-1, 2)
        fb_error = np.linalg.norm(old_flat - back_flat, axis=1)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(new_flat).all(axis=1)
            & np.isfinite(fb_error)
            & (fb_error <= self.config.fb_max_error_px)
        )

        grouped: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        for target_id, start, end in owners:
            local_valid = valid[start:end]
            grouped[target_id] = (
                old_flat[start:end][local_valid],
                new_flat[start:end][local_valid],
                fb_error[start:end][local_valid],
                end - start,
            )
        return grouped

    def _fit_motion(
        self,
        previous_bbox: BBox,
        flow: tuple[np.ndarray, np.ndarray, np.ndarray, int] | None,
        width: int,
        height: int,
    ) -> tuple[BBox, float, int, float, np.ndarray, float, float] | None:
        if flow is None:
            return None
        old_points, new_points, fb_error, original_count = flow
        if len(old_points) < self.config.min_ransac_inliers:
            return None
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            old_points,
            new_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reproj_threshold_px,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inlier_mask is None or not np.isfinite(affine).all():
            return None
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < self.config.min_ransac_inliers:
            return None
        singular_values = np.linalg.svd(affine[:, :2], compute_uv=False)
        scale = float(np.sqrt(max(float(np.prod(singular_values)), 0.0)))
        if not self.config.min_motion_scale <= scale <= self.config.max_motion_scale:
            return None

        # Transform the center, but update size only by the similarity scale.
        # Repeatedly rotating an axis-aligned box and taking its enclosing box
        # multiplies its dimensions by abs(cos)+abs(sin), even for pure jitter.
        cx, cy, box_width, box_height = _box_to_measurement(previous_bbox)
        center = affine[:, :2] @ np.asarray([cx, cy]) + affine[:, 2]
        half_width, half_height = 0.5 * scale * box_width, 0.5 * scale * box_height
        measured_box = _clip_box(
            (
                float(center[0] - half_width),
                float(center[1] - half_height),
                float(center[0] + half_width),
                float(center[1] + half_height),
            ),
            width,
            height,
        )
        retention = len(old_points) / max(original_count, 1)
        inlier_ratio = inlier_count / max(len(old_points), 1)
        median_fb = float(np.median(fb_error[inliers]))
        error_quality = math.exp(-median_fb / self.config.fb_max_error_px)
        motion_confidence = float(np.clip(retention * inlier_ratio * error_quality, 0.0, 1.0))
        return (
            measured_box,
            motion_confidence,
            inlier_count,
            median_fb,
            np.ascontiguousarray(new_points[inliers], dtype=np.float32),
            float(retention),
            float(inlier_ratio),
        )

    def _compute_priorities(self, width: int, height: int) -> dict[int, float]:
        principal = np.asarray([self.intrinsics.cx, self.intrinsics.cy], dtype=np.float64)
        half_diagonal = max(0.5 * math.hypot(width, height), 1.0)
        priority_weight_sum = sum(
            abs(weight)
            for weight in (
                self.config.priority_confidence_weight,
                self.config.priority_center_weight,
                self.config.priority_area_weight,
                self.config.priority_stability_weight,
                self.config.priority_distance_weight,
                self.config.priority_speed_weight,
            )
        )
        priorities: dict[int, float] = {}
        for target_id, state in self._states.items():
            if state.status == "lost":
                priorities[target_id] = float("-inf")
                continue
            cx, cy, box_width, box_height = _box_to_measurement(state.bbox_xyxy)
            center_quality = max(0.0, 1.0 - float(np.linalg.norm(np.asarray([cx, cy]) - principal)) / half_diagonal)
            area_quality = min(1.0, (box_width * box_height) / max(0.08 * width * height, 1.0))
            stability = max(0.0, 1.0 - state.missed_frames / self.config.max_missed_frames)
            kinematics = self._resolve_kinematics(state)
            distance_quality = self._distance_priority_score(kinematics.distance_m)
            speed_quality = self._speed_priority_score(kinematics.speed_m_s)
            weighted_priority = (
                self.config.priority_confidence_weight * state.confidence
                + self.config.priority_center_weight * center_quality
                + self.config.priority_area_weight * area_quality
                + self.config.priority_stability_weight * stability
                + self.config.priority_distance_weight * distance_quality
                + self.config.priority_speed_weight * speed_quality
            )
            priorities[target_id] = weighted_priority / priority_weight_sum + state.priority_bias
        return priorities

    def _resolve_kinematics(self, state: _TargetState) -> TargetKinematics:
        if state.external_kinematics is not None:
            return state.external_kinematics
        distance_m, speed_m_s = self._estimate_kinematics(state)
        return TargetKinematics(distance_m, speed_m_s, source="vision")

    def _estimate_kinematics(self, state: _TargetState) -> tuple[float, float]:
        # Known target dimensions convert apparent box size into depth. Moving
        # the Kalman box a short time forward then gives camera-relative speed.
        current_position = self._camera_position_from_bbox(state.bbox_xyxy)
        future_position = self._camera_position_from_bbox(
            state.kalman.bbox(self.config.kinematic_estimation_horizon_s)
        )
        distance_m = float(np.linalg.norm(current_position))
        speed_m_s = float(
            np.linalg.norm(future_position - current_position)
            / self.config.kinematic_estimation_horizon_s
        )
        return distance_m, speed_m_s

    def _camera_position_from_bbox(self, bbox_xyxy: BBox) -> np.ndarray:
        cx, cy, box_width, box_height = _box_to_measurement(bbox_xyxy)
        depth_from_width = (
            self.intrinsics.fx * self.config.target_width_m / max(box_width, 2.0)
        )
        depth_from_height = (
            self.intrinsics.fy * self.config.target_height_m / max(box_height, 2.0)
        )
        depth_m = 0.5 * (depth_from_width + depth_from_height)
        return self.intrinsics.backproject(cx, cy, depth_m)

    def _distance_priority_score(self, distance_m: float) -> float:
        span = self.config.priority_distance_far_m - self.config.priority_distance_near_m
        return float(
            np.clip(
                (self.config.priority_distance_far_m - distance_m) / span,
                0.0,
                1.0,
            )
        )

    def _speed_priority_score(self, speed_m_s: float) -> float:
        return float(
            np.clip(speed_m_s / self.config.priority_speed_reference_mps, 0.0, 1.0)
        )

    def _choose_target(self, priorities: Mapping[int, float]) -> int | None:
        allowed = getattr(self, "_selection_candidates", None)
        finite = {target_id: score for target_id, score in priorities.items()
                  if math.isfinite(score) and (allowed is None or target_id in allowed)}
        if not finite:
            return None
        if self.config.selection_policy == "nearest_until_removed":
            # Distances come from apparent box size and known target dimensions,
            # never from simulation positions or bootstrap range labels.
            if self._selected_target_id in finite:
                return self._selected_target_id
            return min(finite, key=lambda target_id: (
                self._resolve_kinematics(self._states[target_id]).distance_m, target_id))
        best_id = max(finite, key=lambda target_id: (finite[target_id], -target_id))
        current_id = self._selected_target_id
        if current_id in finite and finite[best_id] < finite[current_id] + self.config.priority_switch_margin:
            return current_id
        return best_id

    def _reset_fine_scheduler(self) -> None:
        self._fine_last_attempt_frame = None
        self._fine_last_success_frame = None
        self._fine_bad_frame_streak = 0
        self._fine_failure_count = 0
        self._fine_backoff_until_frame = 0
        self._fine_last_trigger_reason = "none"
        self._fine_scheduler_target_id = None

    def _initialize_fine_tracker(
        self,
        image_rgb: np.ndarray,
        *,
        trigger_reason: str,
        preserve_failure_state: bool = False,
    ) -> None:
        if self.fine_tracker is None or self._selected_target_id is None:
            return
        state = self._states[self._selected_target_id]
        self._fine_scheduler_target_id = self._selected_target_id
        started = time.perf_counter()
        state.fine_action = "initialize"
        state.fine_trigger_reason = trigger_reason
        previous_failure_count = self._fine_failure_count
        previous_backoff_until = self._fine_backoff_until_frame
        try:
            self.fine_tracker.initialize(image_rgb, state.bbox_xyxy)
            self._fine_target_id = self._selected_target_id
            self._fine_last_run_frame_index = self._frame_index
            self._fine_last_attempt_frame = self._frame_index
            self._fine_last_success_frame = self._frame_index
            self._fine_bad_frame_streak = 0
            if preserve_failure_state:
                self._fine_failure_count = previous_failure_count
                self._fine_backoff_until_frame = previous_backoff_until
            else:
                self._fine_failure_count = 0
                self._fine_backoff_until_frame = self._frame_index
            self._fine_last_trigger_reason = trigger_reason
        except Exception:
            self.fine_tracker.reset()
            self._fine_target_id = None
            self._fine_last_run_frame_index = None
            self._register_fine_failure()
            state.fine_action = "initialize_failed"
        finally:
            state.fine_latency_ms = (time.perf_counter() - started) * 1000.0

    def _compute_fine_tracking_risk(self, state: _TargetState) -> float:
        confidence_span = (
            self.config.adaptive_warning_confidence
            - self.config.adaptive_critical_confidence
        )
        confidence_risk = float(
            np.clip(
                (self.config.adaptive_warning_confidence - state.confidence)
                / confidence_span,
                0.0,
                1.0,
            )
        )
        point_risk = 1.0 - float(
            np.clip(
                state.ransac_inliers / self.config.adaptive_desired_inliers,
                0.0,
                1.0,
            )
        )
        fb_risk = (
            1.0
            if state.median_fb_error_px is None and state.status == "predicted"
            else float(
                np.clip(
                    (state.median_fb_error_px or 0.0) / self.config.fb_max_error_px,
                    0.0,
                    1.0,
                )
            )
        )
        innovation_risk = float(
            np.clip(
                state.last_bbox_innovation
                / self.config.adaptive_innovation_critical,
                0.0,
                1.0,
            )
        )
        prediction_iou_risk = float(
            np.clip(
                (
                    self.config.adaptive_prediction_iou_threshold
                    - state.last_bbox_iou_with_prediction
                )
                / max(self.config.adaptive_prediction_iou_threshold, 1e-6),
                0.0,
                1.0,
            )
        )
        innovation_risk = max(innovation_risk, prediction_iou_risk)
        return float(
            np.clip(
                0.35 * confidence_risk
                + 0.20 * point_risk
                + 0.20 * fb_risk
                + 0.25 * innovation_risk,
                0.0,
                1.0,
            )
        )

    def _decide_fine_tracker_action(
        self, state: _TargetState
    ) -> tuple[bool, str]:
        if self.config.fine_tracker_policy == "fixed":
            frames_since_run = (
                self.config.fine_tracker_interval_frames
                if self._fine_last_run_frame_index is None
                else self._frame_index - self._fine_last_run_frame_index
            )
            if state.status == "predicted":
                return True, "hard_tracking_failure"
            if frames_since_run >= self.config.fine_tracker_interval_frames:
                return True, "fixed_interval"
            return False, "fixed_interval_wait"

        risk = self._compute_fine_tracking_risk(state)
        state.fine_risk = risk
        if risk >= self.config.adaptive_risk_threshold:
            self._fine_bad_frame_streak += 1
        elif risk <= self.config.adaptive_risk_clear_threshold:
            self._fine_bad_frame_streak = 0

        if self._frame_index < self._fine_backoff_until_frame:
            return False, "failure_backoff"

        frames_since_attempt = _frames_since(
            self._frame_index, self._fine_last_attempt_frame
        )
        if frames_since_attempt < self.config.adaptive_min_interval_frames:
            return False, "min_interval"

        hard_failure = (
            state.status == "predicted"
            or state.missed_frames > 0
            or state.last_motion_confidence
            <= self.config.adaptive_critical_confidence
        )
        if hard_failure:
            return True, "hard_tracking_failure"
        if risk >= self.config.adaptive_critical_risk_threshold:
            return True, "critical_quality"
        if self._fine_bad_frame_streak >= self.config.adaptive_bad_frames:
            return True, "soft_quality_streak"

        frames_since_success = _frames_since(
            self._frame_index, self._fine_last_success_frame
        )
        if frames_since_success >= self.config.adaptive_max_interval_frames:
            return True, "max_interval"
        return False, "stable"

    def _register_fine_failure(self) -> None:
        self._fine_failure_count += 1
        exponent = min(self._fine_failure_count - 1, 30)
        backoff = min(
            self.config.adaptive_failure_backoff_initial_frames * (2**exponent),
            self.config.adaptive_failure_backoff_max_frames,
        )
        self._fine_backoff_until_frame = self._frame_index + int(backoff)

    def _fine_candidate_is_plausible(
        self, candidate: BBox, prior: BBox
    ) -> bool:
        _, _, prior_w, prior_h = _box_to_measurement(prior)
        _, _, cand_w, cand_h = _box_to_measurement(candidate)
        center_jump = _center_jump_box_diagonals(candidate, prior)
        scale_change = max(
            cand_w / prior_w,
            prior_w / cand_w,
            cand_h / prior_h,
            prior_h / cand_h,
        )
        return bool(
            center_jump <= self.config.fine_max_center_jump_box_diagonals
            and scale_change <= self.config.fine_max_scale_change
        )

    def _run_fine_tracker(self, image_rgb: np.ndarray, width: int, height: int) -> None:
        selected_id = self._selected_target_id
        if self.fine_tracker is None:
            return
        if selected_id is None:
            if self._fine_target_id is not None:
                self.fine_tracker.reset()
                self._fine_target_id = None
                self._fine_last_run_frame_index = None
                self._reset_fine_scheduler()
            return
        state = self._states[selected_id]
        if self._fine_target_id != selected_id:
            target_changed = (
                self._fine_scheduler_target_id is not None
                and self._fine_scheduler_target_id != selected_id
            )
            if target_changed:
                self.fine_tracker.reset()
                self._fine_target_id = None
                self._fine_last_run_frame_index = None
                self._reset_fine_scheduler()
                self._initialize_fine_tracker(
                    image_rgb, trigger_reason="target_switch"
                )
                return
            if (
                self.config.fine_tracker_policy == "adaptive"
                and self._frame_index < self._fine_backoff_until_frame
            ):
                state.fine_action = "skip"
                state.fine_trigger_reason = "failure_backoff"
                return
            self._initialize_fine_tracker(
                image_rgb,
                trigger_reason="reinitialize_after_failure",
                preserve_failure_state=True,
            )
            return
        should_run, trigger_reason = self._decide_fine_tracker_action(state)
        state.fine_trigger_reason = trigger_reason
        self._fine_last_trigger_reason = trigger_reason
        if not should_run:
            state.fine_action = "skip"
            return
        state.fine_action = "track"
        state.fine_attempted = True
        self._fine_last_attempt_frame = self._frame_index
        started = time.perf_counter()
        try:
            result = self.fine_tracker.track(image_rgb)
        except Exception:
            self.fine_tracker.reset()
            self._fine_target_id = None
            self._fine_last_run_frame_index = None
            self._register_fine_failure()
            state.fine_action = "track_failed"
            state.fine_latency_ms = (time.perf_counter() - started) * 1000.0
            return
        state.fine_latency_ms = (time.perf_counter() - started) * 1000.0
        self._fine_last_run_frame_index = self._frame_index
        state.fine_score = result.score
        score_ok = result.score is None or (
            math.isfinite(result.score) and result.score >= self.config.fine_tracker_min_score
        )
        if not result.ok or result.bbox_xyxy is None or not score_ok:
            self.fine_tracker.reset()
            self._fine_target_id = None
            self._fine_last_run_frame_index = None
            self._register_fine_failure()
            state.fine_action = "track_rejected"
            return
        try:
            bbox = _clip_box(result.bbox_xyxy, width, height)
        except (TypeError, ValueError):
            self.fine_tracker.reset()
            self._fine_target_id = None
            self._fine_last_run_frame_index = None
            self._register_fine_failure()
            state.fine_action = "track_rejected"
            return
        prior_bbox = state.bbox_xyxy
        if not self._fine_candidate_is_plausible(bbox, prior_bbox) or not self._scale_is_plausible(bbox, state):
            self.fine_tracker.reset()
            self._fine_target_id = None
            self._fine_last_run_frame_index = None
            self._register_fine_failure()
            state.fine_action = "track_rejected"
            return
        state.fine_correction_center_px = math.dist(
            _box_center(bbox), _box_center(prior_bbox)
        )
        state.fine_correction_iou = _box_iou(bbox, prior_bbox)
        fine_innovation = _bbox_innovation(bbox, prior_bbox)
        center_jump = _center_jump_box_diagonals(bbox, prior_bbox)
        state.measurement_discontinuity = (
            center_jump
            >= self.config.fine_discontinuity_center_jump_box_diagonals
        )
        # Do not hard-snap the current LK box to an occasional absolute
        # OSTrack result.  The Kalman posterior preserves continuity while
        # still using the fine tracker to correct accumulated LK drift.
        state.bbox_xyxy = state.kalman.correct(
            bbox,
            noise_scale=0.35,
            snap_to_measurement=False,
            preserve_velocity=True,
        )
        # The final measurement source is now OSTrack, so quality must describe
        # the raw OSTrack-vs-LK correction rather than the preceding LK-vs-
        # prediction update.  The TAQ collector maps these fields directly to
        # its bbox innovation and prediction-IoU feature slots.
        state.last_bbox_innovation = fine_innovation
        state.last_bbox_iou_with_prediction = state.fine_correction_iou
        if result.score is not None:
            state.confidence = max(state.confidence, float(np.clip(result.score, 0.0, 1.0)))
        state.status = "tracking"
        state.source = "ostrack"
        state.points = np.empty((0, 2), dtype=np.float32)
        state.fine_applied = True
        self._fine_last_success_frame = self._frame_index
        self._fine_bad_frame_streak = 0
        self._fine_failure_count = 0
        self._fine_backoff_until_frame = self._frame_index
        if state.missed_frames > 0:
            state.missed_frames = 0

    def _build_result(self, priorities: Mapping[int, float]) -> MultiTargetFrameResult:
        if self._image_shape is None:
            raise RuntimeError("Tracker image shape is unavailable")
        height, width = self._image_shape
        estimates: list[TargetEstimate] = []
        for target_id in sorted(self._states):
            state = self._states[target_id]
            bbox = _clip_box(state.bbox_xyxy, width, height)
            predicted_bbox = _clip_box(
                state.kalman.bbox(self.config.prediction_horizon_s), width, height
            )
            center = _box_center(bbox)
            predicted_center = _box_center(predicted_bbox)
            yaw, pitch = self.intrinsics.direction_from_pixel(*center)
            predicted_yaw, predicted_pitch = self.intrinsics.direction_from_pixel(*predicted_center)
            kinematics = self._resolve_kinematics(state)
            selected = target_id == self._selected_target_id
            estimates.append(
                TargetEstimate(
                    target_id=target_id,
                    bbox_xyxy=bbox,
                    predicted_bbox_xyxy=predicted_bbox,
                    center_xy=center,
                    predicted_center_xy=predicted_center,
                    yaw_rad=float(yaw),
                    pitch_rad=float(pitch),
                    predicted_yaw_rad=float(predicted_yaw),
                    predicted_pitch_rad=float(predicted_pitch),
                    distance_m=kinematics.distance_m,
                    speed_m_s=kinematics.speed_m_s,
                    kinematics_source=kinematics.source,
                    distance_priority_score=self._distance_priority_score(
                        kinematics.distance_m
                    ),
                    speed_priority_score=self._speed_priority_score(
                        kinematics.speed_m_s
                    ),
                    confidence=float(np.clip(state.confidence, 0.0, 1.0)),
                    priority=float(priorities.get(target_id, float("-inf"))),
                    status=state.status,
                    source=state.source,
                    corners=len(state.points),
                    ransac_inliers=state.ransac_inliers,
                    median_fb_error_px=state.median_fb_error_px,
                    missed_frames=state.missed_frames,
                    selected=selected,
                    fine_score=state.fine_score,
                    motion_confidence=state.last_motion_confidence,
                    point_retention=state.last_point_retention,
                    ransac_inlier_ratio=state.last_inlier_ratio,
                    bbox_innovation=state.last_bbox_innovation,
                    bbox_iou_with_prediction=state.last_bbox_iou_with_prediction,
                    fine_policy=(
                        self.config.fine_tracker_policy
                        if self.fine_tracker is not None
                        else "disabled"
                    ),
                    fine_action=(
                        state.fine_action
                        if self.fine_tracker is not None
                        else "disabled"
                    ),
                    fine_trigger_reason=(
                        state.fine_trigger_reason
                        if self.fine_tracker is not None
                        else "disabled"
                    ),
                    fine_risk=state.fine_risk,
                    fine_bad_frame_streak=(
                        self._fine_bad_frame_streak if selected else 0
                    ),
                    frames_since_fine_attempt=(
                        _frames_since(self._frame_index, self._fine_last_attempt_frame)
                        if selected and self.fine_tracker is not None
                        else -1
                    ),
                    frames_since_fine_success=(
                        _frames_since(self._frame_index, self._fine_last_success_frame)
                        if selected and self.fine_tracker is not None
                        else -1
                    ),
                    fine_attempted=state.fine_attempted,
                    fine_applied=state.fine_applied,
                    fine_latency_ms=state.fine_latency_ms,
                    fine_failure_count=(self._fine_failure_count if selected else 0),
                    fine_correction_center_px=state.fine_correction_center_px,
                    fine_correction_iou=state.fine_correction_iou,
                    measurement_discontinuity=state.measurement_discontinuity,
                    measurement_rejected=state.measurement_rejected,
                )
            )
        return MultiTargetFrameResult(
            frame_index=self._frame_index,
            timestamp_s=self._timestamp_s,
            targets=tuple(estimates),
            selected_target_id=self._selected_target_id,
        )


def _normalize_initial_boxes(
    initial_boxes: Mapping[int, Iterable[float]] | Iterable[Iterable[float]],
) -> dict[int, BBox]:
    if isinstance(initial_boxes, Mapping):
        items = initial_boxes.items()
    else:
        items = enumerate(initial_boxes)
    normalized: dict[int, BBox] = {}
    for target_id_raw, box_raw in items:
        target_id = int(target_id_raw)
        if target_id in normalized:
            raise ValueError(f"Duplicate target ID: {target_id}")
        values = tuple(float(value) for value in box_raw)
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Target {target_id} box must contain four finite values")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"Target {target_id} box must satisfy x2>x1 and y2>y1")
        normalized[target_id] = values
    return normalized


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {array.shape}")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    converted = array.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("RGB image contains non-finite values")
    if converted.size and float(np.nanmax(converted)) <= 1.5:
        converted *= 255.0
    return np.ascontiguousarray(np.clip(converted, 0.0, 255.0).round().astype(np.uint8))


def _box_to_measurement(bbox_xyxy: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), max(x2 - x1, 2.0), max(y2 - y1, 2.0)


def _box_center(bbox_xyxy: Iterable[float]) -> tuple[float, float]:
    cx, cy, _, _ = _box_to_measurement(bbox_xyxy)
    return cx, cy


def _box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _bbox_innovation(measurement: Iterable[float], prediction: Iterable[float]) -> float:
    measured_cx, measured_cy, measured_w, measured_h = _box_to_measurement(measurement)
    predicted_cx, predicted_cy, predicted_w, predicted_h = _box_to_measurement(prediction)
    diagonal = max(math.hypot(measured_w, measured_h), 1.0)
    center_error = math.hypot(
        measured_cx - predicted_cx, measured_cy - predicted_cy
    ) / diagonal
    scale_error = max(
        abs(math.log(measured_w / predicted_w)),
        abs(math.log(measured_h / predicted_h)),
    )
    return float(center_error + scale_error)


def _center_jump_box_diagonals(candidate: Iterable[float], prior: Iterable[float]) -> float:
    """Return center displacement normalized by the prior box diagonal."""
    prior_cx, prior_cy, prior_w, prior_h = _box_to_measurement(prior)
    candidate_cx, candidate_cy, _, _ = _box_to_measurement(candidate)
    prior_diagonal = max(math.hypot(prior_w, prior_h), 1.0)
    return float(
        math.hypot(candidate_cx - prior_cx, candidate_cy - prior_cy)
        / prior_diagonal
    )


def _frames_since(frame_index: int, previous_frame: int | None) -> int:
    if previous_frame is None:
        return 2**31 - 1
    return max(0, int(frame_index) - int(previous_frame))


def _clip_box(bbox_xyxy: Iterable[float], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("Bounding box contains a non-finite value")
    x1 = min(max(x1, 0.0), max(float(width - 2), 0.0))
    y1 = min(max(y1, 0.0), max(float(height - 2), 0.0))
    x2 = min(max(x2, x1 + 2.0), float(width))
    y2 = min(max(y2, y1 + 2.0), float(height))
    return x1, y1, x2, y2


__all__ = [
    "BBox",
    "FineTrackResult",
    "FineTracker",
    "MultiTargetConfig",
    "MultiTargetFrameResult",
    "MultiTargetVisionTracker",
    "TargetEstimate",
]

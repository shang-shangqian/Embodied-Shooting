"""Reproducible Stage-2 sensor degradation for static-base TAQ collection.

The degradation is intentionally applied before TAQ feature construction.  It
never reads target ground truth and therefore cannot silently repair an invalid
visual/range measurement.  Ground-truth velocity remains a collector concern
and is used only as the supervised label.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import math
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # Pure schema/degradation tests need not install OpenCV.
    cv2 = None


DEGRADATION_PROFILES = (
    "clean",
    "image",
    "tracking",
    "range",
    "timing",
    "proprio",
    "mixed",
)
DEGRADATION_SEVERITIES = ("mild", "medium", "hard", "random")
DATA_SPLITS = ("train", "val", "test")

_COMPONENTS = ("image", "tracking", "range", "timing", "proprio")
_RANDOM_SEVERITY_VALUES = ("mild", "medium", "hard")
_RANDOM_SEVERITY_PROBABILITIES = (0.50, 0.35, 0.15)
_MIXED_SEVERITY_PROBABILITIES = (0.40, 0.40, 0.20)


@dataclass(frozen=True)
class _SeverityPreset:
    image_segment_start_probability: float
    image_segment_length: tuple[int, int]
    brightness_delta: float
    gamma_delta: float
    image_noise_std: float
    blur_kernel_max: int
    jpeg_quality_min: int
    image_shift_px: float
    occlusion_start_probability: float
    occlusion_length: tuple[int, int]
    occlusion_fraction: tuple[float, float]
    image_dropout_start_probability: float
    image_dropout_length: tuple[int, int]
    detector_miss_probability: float
    detector_center_jitter_diagonal: float
    detector_scale_fraction: float
    range_bias_fraction: float
    range_drift_fraction: float
    range_noise_fraction: float
    range_noise_correlation: float
    range_outlier_probability: float
    range_outlier_m: float
    range_dropout_start_probability: float
    range_dropout_length: tuple[int, int]
    timestamp_jitter_s: float
    timestamp_gap_probability: float
    timestamp_gap_steps: tuple[int, int]
    image_delay_steps: int
    range_delay_steps: int
    linear_velocity_noise_mps: float
    linear_velocity_bias_mps: float
    angular_velocity_noise_rps: float
    angular_velocity_bias_rps: float
    proprio_delay_steps: int
    extrinsic_angle_deg: float
    extrinsic_translation_m: float


_PRESETS = {
    "mild": _SeverityPreset(
        # Image parameters stay coherent for a short segment and use a smooth
        # fade-in/fade-out.  This avoids frame-wise flicker that is much less
        # realistic than exposure/blur changes on a physical camera.
        image_segment_start_probability=0.015,
        image_segment_length=(8, 20),
        brightness_delta=0.12,
        gamma_delta=0.10,
        image_noise_std=1.5,
        blur_kernel_max=3,
        jpeg_quality_min=80,
        image_shift_px=0.35,
        occlusion_start_probability=0.015,
        occlusion_length=(2, 5),
        occlusion_fraction=(0.10, 0.20),
        image_dropout_start_probability=0.004,
        image_dropout_length=(1, 3),
        detector_miss_probability=0.03,
        detector_center_jitter_diagonal=0.07,
        detector_scale_fraction=0.05,
        range_bias_fraction=0.002,
        range_drift_fraction=0.001,
        range_noise_fraction=0.0002,
        range_noise_correlation=0.98,
        range_outlier_probability=0.002,
        range_outlier_m=0.75,
        range_dropout_start_probability=0.002,
        range_dropout_length=(2, 4),
        timestamp_jitter_s=0.002,
        timestamp_gap_probability=0.002,
        timestamp_gap_steps=(1, 2),
        image_delay_steps=1,
        range_delay_steps=0,
        linear_velocity_noise_mps=0.02,
        linear_velocity_bias_mps=0.01,
        angular_velocity_noise_rps=0.002,
        angular_velocity_bias_rps=0.002,
        proprio_delay_steps=1,
        extrinsic_angle_deg=0.05,
        extrinsic_translation_m=0.005,
    ),
    "medium": _SeverityPreset(
        image_segment_start_probability=0.025,
        image_segment_length=(12, 30),
        brightness_delta=0.25,
        gamma_delta=0.18,
        image_noise_std=3.0,
        blur_kernel_max=5,
        jpeg_quality_min=60,
        image_shift_px=0.75,
        occlusion_start_probability=0.040,
        occlusion_length=(4, 8),
        occlusion_fraction=(0.25, 0.45),
        image_dropout_start_probability=0.010,
        image_dropout_length=(3, 6),
        detector_miss_probability=0.08,
        detector_center_jitter_diagonal=0.18,
        detector_scale_fraction=0.15,
        range_bias_fraction=0.0035,
        range_drift_fraction=0.002,
        range_noise_fraction=0.0005,
        range_noise_correlation=0.985,
        range_outlier_probability=0.005,
        range_outlier_m=1.5,
        range_dropout_start_probability=0.005,
        range_dropout_length=(3, 7),
        timestamp_jitter_s=0.005,
        timestamp_gap_probability=0.005,
        timestamp_gap_steps=(1, 4),
        image_delay_steps=2,
        range_delay_steps=1,
        linear_velocity_noise_mps=0.05,
        linear_velocity_bias_mps=0.03,
        angular_velocity_noise_rps=0.010,
        angular_velocity_bias_rps=0.006,
        proprio_delay_steps=2,
        extrinsic_angle_deg=0.10,
        extrinsic_translation_m=0.010,
    ),
    "hard": _SeverityPreset(
        image_segment_start_probability=0.050,
        image_segment_length=(16, 40),
        brightness_delta=0.45,
        gamma_delta=0.35,
        image_noise_std=6.0,
        blur_kernel_max=9,
        jpeg_quality_min=40,
        image_shift_px=1.5,
        occlusion_start_probability=0.070,
        occlusion_length=(5, 12),
        occlusion_fraction=(0.30, 0.55),
        image_dropout_start_probability=0.018,
        image_dropout_length=(4, 9),
        detector_miss_probability=0.12,
        detector_center_jitter_diagonal=0.30,
        detector_scale_fraction=0.30,
        range_bias_fraction=0.005,
        range_drift_fraction=0.003,
        range_noise_fraction=0.0010,
        range_noise_correlation=0.99,
        range_outlier_probability=0.010,
        range_outlier_m=3.0,
        range_dropout_start_probability=0.010,
        range_dropout_length=(5, 10),
        timestamp_jitter_s=0.008,
        timestamp_gap_probability=0.010,
        timestamp_gap_steps=(1, 6),
        image_delay_steps=3,
        range_delay_steps=1,
        linear_velocity_noise_mps=0.08,
        linear_velocity_bias_mps=0.05,
        angular_velocity_noise_rps=0.020,
        angular_velocity_bias_rps=0.010,
        proprio_delay_steps=3,
        extrinsic_angle_deg=0.20,
        extrinsic_translation_m=0.020,
    ),
}


@dataclass
class _EpisodeState:
    episode_id: int
    severity: str
    enabled_components: tuple[str, ...]
    rng: np.random.Generator
    range_bias_fraction: float = 0.0
    range_drift_fraction: float = 0.0
    range_noise_relative: float = 0.0
    linear_velocity_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_velocity_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    extrinsic_yaw_rad: float = 0.0
    extrinsic_pitch_rad: float = 0.0
    extrinsic_translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    range_delay_steps: int = 0
    image_delay_steps: int = 0
    proprio_delay_steps: int = 0
    image_dropout_left: int = 0
    image_segment_left: int = 0
    image_segment_total: int = 0
    image_segment_gain: float = 1.0
    image_segment_gamma: float = 1.0
    image_segment_noise_std: float = 0.0
    image_segment_blur_kernel: int = 0
    image_segment_jpeg_quality: int | None = None
    image_segment_shift_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    occlusion_left: int = 0
    occlusion_fraction: float = 0.0
    occlusion_anchor: tuple[float, float] = (0.5, 0.5)
    range_dropout_left: int = 0
    sensor_timestamp_s: float | None = None
    image_queue: deque[np.ndarray] = field(default_factory=deque)
    range_queue: deque[tuple[float, bool]] = field(default_factory=deque)
    proprio_queue: deque[tuple[np.ndarray, np.ndarray]] = field(default_factory=deque)
    counters: Counter = field(default_factory=Counter)


def _json_vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)]


class Stage2Degrader:
    """Apply deterministic, per-episode Stage-2 observation degradation."""

    def __init__(
        self,
        *,
        profile: str,
        severity: str,
        seed: int,
        num_envs: int,
        nominal_dt_s: float,
        min_timestamp_dt_ratio: float = 0.75,
    ) -> None:
        if profile not in DEGRADATION_PROFILES:
            raise ValueError(f"Unsupported degradation profile: {profile}")
        if severity not in DEGRADATION_SEVERITIES:
            raise ValueError(f"Unsupported degradation severity: {severity}")
        if num_envs < 1 or nominal_dt_s <= 0.0:
            raise ValueError("num_envs and nominal_dt_s must be positive")
        if not 0.0 < min_timestamp_dt_ratio <= 1.0:
            raise ValueError("min_timestamp_dt_ratio must be in (0, 1]")
        self.profile = str(profile)
        self.requested_severity = str(severity)
        self.seed = int(seed)
        self.num_envs = int(num_envs)
        self.nominal_dt_s = float(nominal_dt_s)
        self.min_timestamp_dt_ratio = float(min_timestamp_dt_ratio)
        self._states: list[_EpisodeState | None] = [None] * self.num_envs
        self._aggregate_counters: Counter = Counter()

    def _state(self, env_id: int) -> _EpisodeState:
        identifier = int(env_id)
        if identifier < 0 or identifier >= self.num_envs:
            raise IndexError(f"env_id out of range: {identifier}")
        state = self._states[identifier]
        if state is None:
            state = self.reset_episode(identifier, 0)
        return state

    def reset_episode(self, env_id: int, episode_id: int) -> _EpisodeState:
        identifier = int(env_id)
        if identifier < 0 or identifier >= self.num_envs:
            raise IndexError(f"env_id out of range: {identifier}")
        previous = self._states[identifier]
        if previous is not None:
            self._aggregate_counters.update(previous.counters)

        seed_sequence = np.random.SeedSequence([self.seed, identifier, int(episode_id)])
        rng = np.random.default_rng(seed_sequence)
        if self.requested_severity == "random":
            probabilities = (
                _MIXED_SEVERITY_PROBABILITIES
                if self.profile == "mixed"
                else _RANDOM_SEVERITY_PROBABILITIES
            )
            severity = str(
                rng.choice(
                    _RANDOM_SEVERITY_VALUES,
                    p=probabilities,
                )
            )
        else:
            severity = self.requested_severity

        if self.profile == "clean":
            components: tuple[str, ...] = ()
        elif self.profile == "mixed":
            count = int(rng.integers(2, min(4, len(_COMPONENTS)) + 1))
            components = tuple(sorted(str(item) for item in rng.choice(_COMPONENTS, size=count, replace=False)))
        else:
            components = (self.profile,)

        preset = _PRESETS[severity]
        uses_range = "range" in components
        uses_timing = "timing" in components
        uses_proprio = "proprio" in components
        state = _EpisodeState(
            episode_id=int(episode_id),
            severity=severity,
            enabled_components=components,
            rng=rng,
            range_bias_fraction=(
                float(rng.uniform(-preset.range_bias_fraction, preset.range_bias_fraction))
                if uses_range
                else 0.0
            ),
            linear_velocity_bias=(
                rng.uniform(
                    -preset.linear_velocity_bias_mps,
                    preset.linear_velocity_bias_mps,
                    size=3,
                ).astype(np.float32)
                if uses_proprio
                else np.zeros(3, dtype=np.float32)
            ),
            angular_velocity_bias=(
                rng.uniform(
                    -preset.angular_velocity_bias_rps,
                    preset.angular_velocity_bias_rps,
                    size=3,
                ).astype(np.float32)
                if uses_proprio
                else np.zeros(3, dtype=np.float32)
            ),
            extrinsic_yaw_rad=(
                math.radians(float(rng.uniform(-preset.extrinsic_angle_deg, preset.extrinsic_angle_deg)))
                if uses_proprio
                else 0.0
            ),
            extrinsic_pitch_rad=(
                math.radians(float(rng.uniform(-preset.extrinsic_angle_deg, preset.extrinsic_angle_deg)))
                if uses_proprio
                else 0.0
            ),
            extrinsic_translation=(
                rng.uniform(
                    -preset.extrinsic_translation_m,
                    preset.extrinsic_translation_m,
                    size=3,
                ).astype(np.float32)
                if uses_proprio
                else np.zeros(3, dtype=np.float32)
            ),
            image_delay_steps=(preset.image_delay_steps if uses_timing else 0),
            range_delay_steps=(preset.range_delay_steps if uses_timing else 0),
            proprio_delay_steps=(preset.proprio_delay_steps if uses_proprio else 0),
        )
        self._states[identifier] = state
        return state

    def _preset(self, state: _EpisodeState) -> _SeverityPreset:
        return _PRESETS[state.severity]

    @staticmethod
    def _uses(state: _EpisodeState, component: str) -> bool:
        return component in state.enabled_components

    def next_timestamp(self, env_id: int, nominal_timestamp_s: float) -> tuple[float, float]:
        """Return a strictly increasing sensor timestamp and its actual delta."""
        state = self._state(env_id)
        if state.sensor_timestamp_s is None:
            state.sensor_timestamp_s = float(nominal_timestamp_s)
            return state.sensor_timestamp_s, self.nominal_dt_s
        if self.profile == "clean":
            # Clean acquisition follows the actual simulation capture clock.
            # Do not reconstruct it by repeatedly adding nominal dt.
            dt_s = float(nominal_timestamp_s) - state.sensor_timestamp_s
            if not math.isfinite(dt_s) or dt_s <= 0:
                raise ValueError("Clean capture timestamps must increase strictly")
            state.sensor_timestamp_s = float(nominal_timestamp_s)
            return state.sensor_timestamp_s, dt_s
        jitter = 0.0
        if self._uses(state, "timing"):
            jitter = float(state.rng.normal(0.0, self._preset(state).timestamp_jitter_s))
        skipped_steps = 0
        if self._uses(state, "timing") and state.rng.random() < self._preset(state).timestamp_gap_probability:
            low, high = self._preset(state).timestamp_gap_steps
            skipped_steps = int(state.rng.integers(low, high + 1))
            state.counters["timestamp_gap_frames"] += 1
        dt_s = max(
            self.min_timestamp_dt_ratio * self.nominal_dt_s,
            (1 + skipped_steps) * self.nominal_dt_s + jitter,
        )
        state.sensor_timestamp_s += dt_s
        if abs(jitter) > 1e-12:
            state.counters["timestamp_jitter_frames"] += 1
        return float(state.sensor_timestamp_s), float(dt_s)

    def perturb_bootstrap_box(
        self,
        env_id: int,
        bbox_xyxy: Iterable[float],
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float] | None:
        """Turn a simulation bootstrap box into a detector-like proposal."""
        state = self._state(env_id)
        if not self._uses(state, "tracking"):
            return tuple(float(item) for item in bbox_xyxy)
        preset = self._preset(state)
        if state.rng.random() < preset.detector_miss_probability:
            state.counters["bootstrap_misses"] += 1
            return None

        x1, y1, x2, y2 = (float(item) for item in bbox_xyxy)
        width = max(x2 - x1, 2.0)
        height = max(y2 - y1, 2.0)
        diagonal = max(math.hypot(width, height), 1.0)
        radius = preset.detector_center_jitter_diagonal * diagonal
        angle = float(state.rng.uniform(-math.pi, math.pi))
        magnitude = float(state.rng.uniform(0.0, radius))
        center_x = 0.5 * (x1 + x2) + magnitude * math.cos(angle)
        center_y = 0.5 * (y1 + y2) + magnitude * math.sin(angle)
        width *= float(state.rng.uniform(1.0 - preset.detector_scale_fraction, 1.0 + preset.detector_scale_fraction))
        height *= float(state.rng.uniform(1.0 - preset.detector_scale_fraction, 1.0 + preset.detector_scale_fraction))
        x1 = max(0.0, center_x - 0.5 * width)
        y1 = max(0.0, center_y - 0.5 * height)
        x2 = min(float(image_width), center_x + 0.5 * width)
        y2 = min(float(image_height), center_y + 0.5 * height)
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            state.counters["bootstrap_misses"] += 1
            return None
        state.counters["perturbed_bootstrap_boxes"] += 1
        return x1, y1, x2, y2

    @staticmethod
    def _start_image_segment(state: _EpisodeState, preset: _SeverityPreset) -> None:
        """Sample one coherent camera-degradation segment for an episode."""
        total = int(
            state.rng.integers(
                preset.image_segment_length[0],
                preset.image_segment_length[1] + 1,
            )
        )
        state.image_segment_left = total
        state.image_segment_total = total
        state.image_segment_gain = float(
            state.rng.uniform(1.0 - preset.brightness_delta, 1.0 + preset.brightness_delta)
        )
        state.image_segment_gamma = float(
            state.rng.uniform(1.0 - preset.gamma_delta, 1.0 + preset.gamma_delta)
        )
        state.image_segment_noise_std = float(preset.image_noise_std)
        state.image_segment_blur_kernel = (
            int(state.rng.choice(list(range(3, preset.blur_kernel_max + 1, 2))))
            if preset.blur_kernel_max >= 3 and state.rng.random() < 0.45
            else 0
        )
        state.image_segment_jpeg_quality = (
            int(state.rng.integers(preset.jpeg_quality_min, 96))
            if state.rng.random() < 0.25
            else None
        )
        state.image_segment_shift_xy = state.rng.uniform(
            -preset.image_shift_px,
            preset.image_shift_px,
            size=2,
        ).astype(np.float32)
        state.counters["image_degradation_segments"] += 1

    @staticmethod
    def _segment_envelope(state: _EpisodeState) -> float:
        """Raised cosine-like segment envelope, zero at both boundaries."""
        if state.image_segment_total <= 0:
            return 0.0
        progress = (state.image_segment_total - state.image_segment_left + 1) / (
            state.image_segment_total + 1
        )
        return float(math.sin(math.pi * progress))

    def degrade_image(
        self,
        env_id: int,
        image_rgb: np.ndarray,
        *,
        target_bbox_xyxy: Iterable[float] | None = None,
        allow_dropout: bool = True,
    ) -> np.ndarray:
        """Apply image, occlusion, and frame-drop degradation to one RGB frame."""
        state = self._state(env_id)
        preset = self._preset(state)
        image = np.ascontiguousarray(np.asarray(image_rgb, dtype=np.uint8).copy())
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB uint8 image [H,W,3], got {image.shape}")

        if allow_dropout and self._uses(state, "tracking"):
            if state.image_dropout_left <= 0 and state.rng.random() < preset.image_dropout_start_probability:
                state.image_dropout_left = int(
                    state.rng.integers(
                        preset.image_dropout_length[0],
                        preset.image_dropout_length[1] + 1,
                    )
                )
            if state.image_dropout_left > 0:
                state.image_dropout_left -= 1
                state.counters["image_dropout_frames"] += 1
                return np.zeros_like(image)

        if self._uses(state, "image"):
            if (
                state.image_segment_left <= 0
                and state.rng.random() < preset.image_segment_start_probability
            ):
                self._start_image_segment(state, preset)
            if state.image_segment_left > 0:
                envelope = self._segment_envelope(state)
                gain = 1.0 + envelope * (state.image_segment_gain - 1.0)
                gamma = 1.0 + envelope * (state.image_segment_gamma - 1.0)
                normalized = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
                image = np.clip(255.0 * gain * np.power(normalized, gamma), 0.0, 255.0).astype(np.uint8)
                if state.image_segment_noise_std > 0.0:
                    noise = state.rng.normal(
                        0.0,
                        envelope * state.image_segment_noise_std,
                        size=image.shape,
                    )
                    image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
                if state.image_segment_blur_kernel > 0:
                    if cv2 is None:
                        raise RuntimeError("OpenCV is required for Stage-2 image degradation")
                    image = cv2.GaussianBlur(
                        image,
                        (state.image_segment_blur_kernel, state.image_segment_blur_kernel),
                        sigmaX=0.0,
                    )
                shift_x, shift_y = envelope * state.image_segment_shift_xy
                if abs(float(shift_x)) > 1e-6 or abs(float(shift_y)) > 1e-6:
                    if cv2 is None:
                        raise RuntimeError("OpenCV is required for Stage-2 image degradation")
                    transform = np.asarray([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
                    image = cv2.warpAffine(
                        image,
                        transform,
                        (image.shape[1], image.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                if state.image_segment_jpeg_quality is not None:
                    if cv2 is None:
                        raise RuntimeError("OpenCV is required for Stage-2 image degradation")
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, state.image_segment_jpeg_quality],
                    )
                    if ok:
                        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                state.image_segment_left -= 1
                state.counters["image_degraded_frames"] += 1

        if self._uses(state, "tracking") and target_bbox_xyxy is not None:
            if state.occlusion_left <= 0 and state.rng.random() < preset.occlusion_start_probability:
                state.occlusion_left = int(
                    state.rng.integers(
                        preset.occlusion_length[0],
                        preset.occlusion_length[1] + 1,
                    )
                )
                state.occlusion_fraction = float(
                    state.rng.uniform(
                        preset.occlusion_fraction[0],
                        preset.occlusion_fraction[1],
                    )
                )
                state.occlusion_anchor = tuple(float(item) for item in state.rng.uniform(0.2, 0.8, size=2))
            if state.occlusion_left > 0:
                x1, y1, x2, y2 = (float(item) for item in target_bbox_xyxy)
                box_width = max(x2 - x1, 1.0)
                box_height = max(y2 - y1, 1.0)
                side_ratio = math.sqrt(max(state.occlusion_fraction, 0.0))
                patch_width = max(1, int(round(box_width * side_ratio)))
                patch_height = max(1, int(round(box_height * side_ratio)))
                center_x = x1 + state.occlusion_anchor[0] * box_width
                center_y = y1 + state.occlusion_anchor[1] * box_height
                left = int(np.clip(round(center_x - 0.5 * patch_width), 0, image.shape[1] - 1))
                top = int(np.clip(round(center_y - 0.5 * patch_height), 0, image.shape[0] - 1))
                right = min(image.shape[1], left + patch_width)
                bottom = min(image.shape[0], top + patch_height)
                fill = np.median(image, axis=(0, 1)).astype(np.uint8)
                image[top:bottom, left:right] = fill
                state.occlusion_left -= 1
                state.counters["occlusion_frames"] += 1

        if self._uses(state, "timing") and state.image_delay_steps > 0:
            state.image_queue.append(np.ascontiguousarray(image))
            if len(state.image_queue) <= state.image_delay_steps:
                delayed_image = state.image_queue[0]
            else:
                delayed_image = state.image_queue.popleft()
            state.counters["image_delayed_frames"] += 1
            return np.ascontiguousarray(delayed_image.copy())

        return np.ascontiguousarray(image)

    def degrade_direction(self, env_id: int, yaw_rad: float, pitch_rad: float) -> tuple[float, float]:
        state = self._state(env_id)
        return (
            float(yaw_rad) + state.extrinsic_yaw_rad,
            float(pitch_rad) + state.extrinsic_pitch_rad,
        )

    def position_bias_base_m(self, env_id: int) -> np.ndarray:
        return self._state(env_id).extrinsic_translation.copy()

    def degrade_range(self, env_id: int, range_m: float, valid: bool) -> tuple[float, bool]:
        """Apply range delay, bias, noise, outliers, and burst dropout."""
        state = self._state(env_id)
        preset = self._preset(state)
        finite_valid = bool(valid) and math.isfinite(float(range_m)) and float(range_m) > 0.0
        state.range_queue.append((float(range_m), finite_valid))
        if len(state.range_queue) <= state.range_delay_steps:
            state.counters["range_delay_invalid_frames"] += 1
            return float("nan"), False
        delayed_range, delayed_valid = state.range_queue.popleft()
        if not delayed_valid:
            return float("nan"), False

        if self._uses(state, "range"):
            if state.range_dropout_left <= 0 and state.rng.random() < preset.range_dropout_start_probability:
                state.range_dropout_left = int(
                    state.rng.integers(
                        preset.range_dropout_length[0],
                        preset.range_dropout_length[1] + 1,
                    )
                )
            if state.range_dropout_left > 0:
                state.range_dropout_left -= 1
                state.counters["range_dropout_frames"] += 1
                return float("nan"), False

            drift_step = float(state.rng.normal(0.0, preset.range_drift_fraction * 0.05))
            state.range_drift_fraction = float(
                np.clip(
                    0.995 * state.range_drift_fraction + drift_step,
                    -preset.range_drift_fraction,
                    preset.range_drift_fraction,
                )
            )
            # A real range sensor's short-term errors are correlated.  Independent
            # frame noise at 100+ m turns into tens of m/s after finite
            # differencing, which is neither useful nor realistic for TAQ.
            correlation = float(np.clip(preset.range_noise_correlation, 0.0, 0.999999))
            innovation_std = preset.range_noise_fraction * math.sqrt(
                max(1.0 - correlation * correlation, 0.0)
            )
            state.range_noise_relative = float(
                np.clip(
                    correlation * state.range_noise_relative
                    + state.rng.normal(0.0, innovation_std),
                    -3.0 * preset.range_noise_fraction,
                    3.0 * preset.range_noise_fraction,
                )
            )
            degraded = delayed_range * (
                1.0
                + state.range_bias_fraction
                + state.range_drift_fraction
                + state.range_noise_relative
            )
            if state.rng.random() < preset.range_outlier_probability:
                degraded += float(state.rng.uniform(-preset.range_outlier_m, preset.range_outlier_m))
                state.counters["range_outlier_frames"] += 1
            state.counters["range_degraded_frames"] += 1
            if not math.isfinite(degraded) or degraded <= 0.0:
                return float("nan"), False
            return float(degraded), True

        return float(delayed_range), True

    def degrade_proprioception(
        self,
        env_id: int,
        linear_velocity_base_mps: np.ndarray,
        angular_velocity_base_rps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self._state(env_id)
        linear = np.asarray(linear_velocity_base_mps, dtype=np.float32).reshape(3).copy()
        angular = np.asarray(angular_velocity_base_rps, dtype=np.float32).reshape(3).copy()
        state.proprio_queue.append((linear, angular))
        if len(state.proprio_queue) <= state.proprio_delay_steps:
            delayed_linear, delayed_angular = state.proprio_queue[0]
        else:
            delayed_linear, delayed_angular = state.proprio_queue.popleft()
        linear = delayed_linear.copy()
        angular = delayed_angular.copy()

        if self._uses(state, "proprio"):
            preset = self._preset(state)
            linear += state.linear_velocity_bias
            angular += state.angular_velocity_bias
            linear += state.rng.normal(0.0, preset.linear_velocity_noise_mps, size=3).astype(np.float32)
            angular += state.rng.normal(0.0, preset.angular_velocity_noise_rps, size=3).astype(np.float32)
            state.counters["proprio_degraded_frames"] += 1
        return linear.astype(np.float32), angular.astype(np.float32)

    def record_tracker_source(self, env_id: int, source: str) -> None:
        state = self._state(env_id)
        state.counters[f"tracker_source_{str(source).lower()}"] += 1

    def record_tracker_measurement_rejection(self, env_id: int) -> None:
        """Record a visual measurement rejected by Kalman consistency gating."""
        self._state(env_id).counters["tracker_measurement_rejected"] += 1

    def episode_metadata(self, env_id: int) -> dict[str, object]:
        state = self._state(env_id)
        return {
            "profile": self.profile,
            "requested_severity": self.requested_severity,
            "episode_severity": state.severity,
            "enabled_components": list(state.enabled_components),
            "degradation_seed": self.seed,
            "severity_parameters": asdict(self._preset(state)),
            "range_bias_fraction": float(state.range_bias_fraction),
            "final_range_drift_fraction": float(state.range_drift_fraction),
            "final_range_noise_relative": float(state.range_noise_relative),
            "image_delay_steps": int(state.image_delay_steps),
            "range_delay_steps": int(state.range_delay_steps),
            "visual_range_delay_offset_steps": int(
                state.image_delay_steps - state.range_delay_steps
            ),
            "linear_velocity_bias_mps": _json_vector(state.linear_velocity_bias),
            "angular_velocity_bias_rps": _json_vector(state.angular_velocity_bias),
            "proprio_delay_steps": int(state.proprio_delay_steps),
            "extrinsic_yaw_error_rad": float(state.extrinsic_yaw_rad),
            "extrinsic_pitch_error_rad": float(state.extrinsic_pitch_rad),
            "extrinsic_translation_error_m": _json_vector(state.extrinsic_translation),
            "event_counts": {str(key): int(value) for key, value in sorted(state.counters.items())},
        }

    def aggregate_counters(self, *, include_active: bool = True) -> dict[str, int]:
        counts = Counter(self._aggregate_counters)
        if include_active:
            for state in self._states:
                if state is not None:
                    counts.update(state.counters)
        return {str(key): int(value) for key, value in sorted(counts.items())}

    def config_metadata(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "requested_severity": self.requested_severity,
            "seed": self.seed,
            "nominal_dt_s": self.nominal_dt_s,
            "min_timestamp_dt_ratio": self.min_timestamp_dt_ratio,
            "min_timestamp_dt_s": self.min_timestamp_dt_ratio * self.nominal_dt_s,
            "random_severity_probabilities": {
                name: probability
                for name, probability in zip(
                    _RANDOM_SEVERITY_VALUES,
                    _RANDOM_SEVERITY_PROBABILITIES,
                )
            },
            "mixed_random_severity_probabilities": {
                name: probability
                for name, probability in zip(
                    _RANDOM_SEVERITY_VALUES,
                    _MIXED_SEVERITY_PROBABILITIES,
                )
            },
        }

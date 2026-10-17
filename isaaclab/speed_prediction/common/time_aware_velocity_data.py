"""Data schema and causal feature utilities for the TAQ-GRU velocity model.

The v2 schema is deliberately strict.  It stores the target's relative
translation velocity (not the derivative of its coordinates in a rotating
frame) at every time step and never silently upgrades v1/window-end labels.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset


TAQ_SCHEMA_VERSION = "taq_velocity_v2"
TAQ_LABEL_SEMANTICS = "target_relative_translation_velocity_yaw_base"
TAQ_STREAM_FORMAT = "taq_episode_stream_v1"
TAQ_YAW_RATE_SEMANTICS = "yaw_only_heading_derivative"
LEGACY_BASELINE_VERSION = "physical_relative_velocity_v1"
# Preserve the historical public alias for existing experimental tools.
TAQ_BASELINE_VERSION = LEGACY_BASELINE_VERSION
ALIGNED_BASELINE_VERSION = "odometry_aligned_relative_velocity_v2"
DEFAULT_BASELINE_VERSION = ALIGNED_BASELINE_VERSION
SUPPORTED_BASELINE_VERSIONS = (LEGACY_BASELINE_VERSION, ALIGNED_BASELINE_VERSION)
CONTINUOUS_FEATURE_DIM = 24
DEFAULT_FINITE_DIFFERENCE_CLIP_MPS = 30.0
DEFAULT_NORMALIZED_FEATURE_CLIP = 8.0
ROBUST_INPUT_TRANSFORM_VERSION = "taq_robust_fd_v1"
LEGACY_INPUT_TRANSFORM_VERSION = "taq_legacy_v1"


class FeatureIndex(IntEnum):
    DT = 12
    OBSERVATION_AGE = 13
    FINE_AGE = 14
    CONFIDENCE = 15
    MOTION_CONFIDENCE = 16
    POINT_RETENTION = 17
    RANSAC_INLIER_RATIO = 18
    MEDIAN_FB_ERROR = 19
    BBOX_INNOVATION = 20
    BBOX_IOU_WITH_PREDICTION = 21
    OBSERVED_MASK = 22
    RANGE_VALID_MASK = 23


POSITION_SLICE = slice(0, 3)
FINITE_DIFFERENCE_SLICE = slice(3, 6)
ROBOT_LINEAR_VELOCITY_SLICE = slice(6, 9)
ROBOT_ANGULAR_VELOCITY_SLICE = slice(9, 12)
QUALITY_SLICE = slice(15, 22)


class TrackerSource(IntEnum):
    INITIAL = 0
    FLOW = 1
    OSTRACK = 2
    PREDICTION = 3
    INVALID = 4


SOURCE_NAME_TO_ID = {
    "initial": TrackerSource.INITIAL,
    "flow": TrackerSource.FLOW,
    "ostrack": TrackerSource.OSTRACK,
    "prediction": TrackerSource.PREDICTION,
    "lost": TrackerSource.INVALID,
    "invalid": TrackerSource.INVALID,
}


def robust_finite_difference(
    features: torch.Tensor,
    *,
    clip_mps: float = DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vector-clip finite differences and return their causal reliability.

    The raw 24-dimensional TAQ feature remains unchanged on disk.  For a
    finite-difference vector ``v``, the returned scale is
    ``min(1, clip_mps / ||v||)`` and the model consumes ``v * scale``.  The
    same scale is a derived reliability feature: normal samples are 1 and an
    implausibly large jump approaches 0.  Only the current and previous
    measured positions and timestamps are involved, so this transform is
    available unchanged during online deployment.
    """
    if features.ndim < 1 or features.shape[-1] != CONTINUOUS_FEATURE_DIM:
        raise ValueError(
            f"features must end in {CONTINUOUS_FEATURE_DIM} values, got {tuple(features.shape)}"
        )
    if clip_mps <= 0.0 or eps <= 0.0:
        raise ValueError("clip_mps and eps must be positive")
    finite_difference = features[..., FINITE_DIFFERENCE_SLICE]
    speed = torch.linalg.vector_norm(finite_difference, dim=-1)
    reliability = (float(clip_mps) / speed.clamp_min(float(eps))).clamp(max=1.0)
    clipped = finite_difference * reliability.unsqueeze(-1)
    return clipped, reliability


def load_torch_payload(path: str | Path) -> Any:
    """Load a local tensor payload on both old and new supported PyTorch releases."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the ``weights_only`` keyword.
        return torch.load(path, map_location="cpu")


def source_ids_from_names(names: Sequence[str]) -> torch.Tensor:
    """Map tracker source strings to the stable TAQ source vocabulary."""
    try:
        return torch.tensor(
            [int(SOURCE_NAME_TO_ID[str(name).lower()]) for name in names],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown tracker source: {exc.args[0]}") from exc


def relative_translation_velocity_yaw_base(
    target_velocity_world_mps: torch.Tensor,
    base_velocity_world_mps: torch.Tensor,
    yaw_rad: torch.Tensor | float,
) -> torch.Tensor:
    """Express ``v_target - v_base`` in the current yaw-only base frame."""
    target = torch.as_tensor(target_velocity_world_mps, dtype=torch.float32)
    base = torch.as_tensor(base_velocity_world_mps, dtype=target.dtype, device=target.device)
    yaw = torch.as_tensor(yaw_rad, dtype=target.dtype, device=target.device)
    if target.shape[-1] != 3 or base.shape != target.shape:
        raise ValueError("target_velocity_world_mps and base_velocity_world_mps must share [..., 3]")
    if yaw.shape not in {(), target.shape[:-1]}:
        raise ValueError("yaw_rad must be scalar or have shape target_velocity_world_mps.shape[:-1]")
    delta = target - base
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    # R_WB^T delta, with R_WB=[ [cos,-sin], [sin,cos] ].
    return torch.stack(
        (cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1],
         -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1],
         delta[..., 2]),
        dim=-1,
    )


def target_position_derivative_base_mps(
    relative_velocity_base_mps: torch.Tensor,
    yaw_rate_rps: torch.Tensor,
    position_base_m: torch.Tensor,
) -> torch.Tensor:
    """Return ``d(position_base)/dt = v_rel_base - omega_yaw x p_base``."""
    velocity = torch.as_tensor(relative_velocity_base_mps, dtype=torch.float32)
    position = torch.as_tensor(position_base_m, dtype=velocity.dtype, device=velocity.device)
    yaw_rate = torch.as_tensor(yaw_rate_rps, dtype=velocity.dtype, device=velocity.device)
    if velocity.shape[-1] != 3 or position.shape != velocity.shape:
        raise ValueError("relative velocity and position must share [..., 3]")
    if yaw_rate.shape not in {(), velocity.shape[:-1]}:
        raise ValueError("yaw_rate_rps must be scalar or have shape relative_velocity_base_mps.shape[:-1]")
    omega = torch.zeros_like(position)
    omega[..., 2] = yaw_rate
    return velocity - torch.cross(omega, position, dim=-1)


def relative_velocity_base_to_world_mps(
    relative_velocity_base_mps: torch.Tensor,
    base_velocity_base_mps: torch.Tensor,
    yaw_rad: torch.Tensor | float,
) -> torch.Tensor:
    """Return target world velocity from relative and base velocities in base axes."""
    relative = torch.as_tensor(relative_velocity_base_mps, dtype=torch.float32)
    base = torch.as_tensor(base_velocity_base_mps, dtype=relative.dtype, device=relative.device)
    yaw = torch.as_tensor(yaw_rad, dtype=relative.dtype, device=relative.device)
    if relative.shape[-1] != 3 or base.shape != relative.shape:
        raise ValueError("relative_velocity_base_mps and base_velocity_base_mps must share [..., 3]")
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    target_velocity_base = relative + base
    return torch.stack(
        (cos_yaw * target_velocity_base[..., 0] - sin_yaw * target_velocity_base[..., 1],
         sin_yaw * target_velocity_base[..., 0] + cos_yaw * target_velocity_base[..., 1],
         target_velocity_base[..., 2]),
        dim=-1,
    )


def _as_time_vector(value: torch.Tensor | Sequence[float], name: str) -> torch.Tensor:
    # Keep absolute timestamps in float64 until all relative-time quantities
    # have been computed.  At long collection times float32 absolute seconds
    # can quantize a configured 15 ms interval to roughly 14.95 ms.
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [T], got {tuple(tensor.shape)}")
    return tensor


def _as_time_matrix(
    value: torch.Tensor | Sequence[Sequence[float]], name: str, width: int
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(
            f"{name} must have shape [T, {width}], got {tuple(tensor.shape)}"
        )
    return tensor


def build_time_aware_features(
    *,
    timestamps_s: torch.Tensor | Sequence[float],
    position_base_m: torch.Tensor | Sequence[Sequence[float]],
    robot_linear_velocity_base_mps: torch.Tensor | Sequence[Sequence[float]],
    robot_angular_velocity_base_rps: torch.Tensor | Sequence[Sequence[float]],
    source_ids: torch.Tensor | Sequence[int],
    quality: torch.Tensor | Sequence[Sequence[float]] | None = None,
    range_valid_mask: torch.Tensor | Sequence[bool] | None = None,
    max_age_s: float = 1.0,
    initial_last_observation_time_s: float | None = None,
    initial_last_fine_time_s: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 24 continuous features and padding-free valid mask for one track.

    ``position_base_m`` and all robot velocities must already use the same
    yaw-only base frame as the target velocity label.  Direct observations are
    INITIAL/FLOW/OSTRACK; prediction is retained in the sequence but marked by
    ``observed_mask=0``.
    """
    timestamps = _as_time_vector(timestamps_s, "timestamps_s")
    position = _as_time_matrix(position_base_m, "position_base_m", 3)
    robot_linear = _as_time_matrix(
        robot_linear_velocity_base_mps, "robot_linear_velocity_base_mps", 3
    )
    robot_angular = _as_time_matrix(
        robot_angular_velocity_base_rps, "robot_angular_velocity_base_rps", 3
    )
    sources = torch.as_tensor(source_ids, dtype=torch.long)
    if sources.ndim != 1:
        raise ValueError(f"source_ids must have shape [T], got {tuple(sources.shape)}")
    time_steps = timestamps.shape[0]
    if time_steps < 1:
        raise ValueError("At least one time step is required")
    if any(tensor.shape[0] != time_steps for tensor in (position, robot_linear, robot_angular, sources)):
        raise ValueError("All time-dependent inputs must have the same length")
    if not torch.isfinite(timestamps).all():
        raise ValueError("timestamps_s contains non-finite values")
    if time_steps > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError("timestamps_s must be strictly increasing")
    if torch.any((sources < int(TrackerSource.INITIAL)) | (sources > int(TrackerSource.INVALID))):
        raise ValueError("source_ids contains an unsupported source ID")

    if quality is None:
        quality_tensor = torch.zeros((time_steps, 7), dtype=torch.float32)
    else:
        quality_tensor = _as_time_matrix(quality, "quality", 7)
        if quality_tensor.shape[0] != time_steps:
            raise ValueError("quality must have the same length as timestamps_s")
    quality_tensor = torch.nan_to_num(quality_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if range_valid_mask is None:
        range_valid = torch.ones(time_steps, dtype=torch.bool)
    else:
        range_valid = torch.as_tensor(range_valid_mask, dtype=torch.bool)
        if range_valid.ndim != 1 or range_valid.shape[0] != time_steps:
            raise ValueError("range_valid_mask must have shape [T]")

    position_finite = torch.isfinite(position).all(dim=-1)
    robot_linear = torch.nan_to_num(robot_linear, nan=0.0, posinf=0.0, neginf=0.0)
    robot_angular = torch.nan_to_num(robot_angular, nan=0.0, posinf=0.0, neginf=0.0)
    clean_position = torch.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
    range_valid = range_valid & position_finite

    direct_source = (
        (sources == int(TrackerSource.INITIAL))
        | (sources == int(TrackerSource.FLOW))
        | (sources == int(TrackerSource.OSTRACK))
    )
    observed = direct_source & range_valid

    dt_f64 = torch.zeros(time_steps, dtype=torch.float64)
    if time_steps > 1:
        dt_f64[1:] = timestamps[1:] - timestamps[:-1]

    finite_difference = torch.zeros((time_steps, 3), dtype=torch.float32)
    if time_steps > 1:
        pair_valid = range_valid[1:] & range_valid[:-1] & (dt_f64[1:] > 1e-6)
        finite_difference_f64 = (
            clean_position[1:][pair_valid].to(torch.float64)
            - clean_position[:-1][pair_valid].to(torch.float64)
        ) / dt_f64[1:][pair_valid].unsqueeze(-1)
        finite_difference[1:][pair_valid] = finite_difference_f64.to(torch.float32)

    observation_age = torch.zeros(time_steps, dtype=torch.float32)
    fine_age = torch.zeros(time_steps, dtype=torch.float32)
    first_time = float(timestamps[0].item())
    if initial_last_observation_time_s is not None and initial_last_observation_time_s > first_time:
        raise ValueError("initial_last_observation_time_s cannot be later than the first sequence timestamp")
    if initial_last_fine_time_s is not None and initial_last_fine_time_s > first_time:
        raise ValueError("initial_last_fine_time_s cannot be later than the first sequence timestamp")
    last_observation_time: float | None = initial_last_observation_time_s
    last_fine_time: float | None = initial_last_fine_time_s
    capped_age = max(float(max_age_s), 0.0)
    for index in range(time_steps):
        now = float(timestamps[index].item())
        if bool(observed[index]):
            last_observation_time = now
        if int(sources[index]) == int(TrackerSource.OSTRACK) and bool(range_valid[index]):
            last_fine_time = now
        observation_age[index] = min(
            0.0 if last_observation_time is None else now - last_observation_time,
            capped_age,
        )
        fine_age[index] = min(
            0.0 if last_fine_time is None else now - last_fine_time,
            capped_age,
        )

    features = torch.cat(
        (
            clean_position,
            finite_difference,
            robot_linear,
            robot_angular,
            dt_f64.to(dtype=torch.float32).unsqueeze(-1),
            observation_age.unsqueeze(-1),
            fine_age.unsqueeze(-1),
            quality_tensor,
            observed.to(torch.float32).unsqueeze(-1),
            range_valid.to(torch.float32).unsqueeze(-1),
        ),
        dim=-1,
    )
    if features.shape != (time_steps, CONTINUOUS_FEATURE_DIM):
        raise RuntimeError(f"Unexpected TAQ feature shape: {tuple(features.shape)}")
    return features.contiguous(), range_valid.contiguous()


@torch.no_grad()
def weighted_velocity_baseline(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    window_s: float = 0.48,
    finite_difference_clip_mps: float | None = DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
    eps: float = 1e-6,
    baseline_version: str = LEGACY_BASELINE_VERSION,
) -> torch.Tensor:
    """Versioned causal relative velocity; the legacy default preserves callers.

    New TAQ models explicitly select ``ALIGNED_BASELINE_VERSION``. Checkpoints
    record this choice so an old residual head cannot silently get a new base.
    """
    if baseline_version == ALIGNED_BASELINE_VERSION:
        return _aligned_velocity_baseline(
            features, valid_mask, window_s=window_s,
            finite_difference_clip_mps=finite_difference_clip_mps, eps=eps,
        )
    if baseline_version != LEGACY_BASELINE_VERSION:
        raise ValueError(f"Unsupported baseline_version={baseline_version!r}")
    return _legacy_velocity_baseline(
        features, valid_mask, window_s=window_s,
        finite_difference_clip_mps=finite_difference_clip_mps, eps=eps,
    )


def taq_model_config_from_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve historical defaults and reject conflicting baseline metadata."""
    version = payload.get("baseline_version")
    if version not in SUPPORTED_BASELINE_VERSIONS:
        raise ValueError(f"Missing or unsupported TAQ baseline_version={version!r}")
    config = dict(payload.get("model_config", {}))
    if config.get("baseline_version", version) != version:
        raise ValueError("Checkpoint baseline_version conflicts with model_config")
    config["baseline_version"] = version
    config.setdefault("input_transform_version", LEGACY_INPUT_TRANSFORM_VERSION)
    return config


@torch.no_grad()
def _aligned_velocity_baseline(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    window_s: float,
    finite_difference_clip_mps: float | None,
    eps: float,
) -> torch.Tensor:
    """Fit observed positions in one odometry frame, then recover v_target-v_base.

    Yaw and translation are causal trapezoidal integrals of existing robot
    motion features. The reference pose is arbitrary; no absolute or target
    truth pose is consumed. Float64 bounds drift/cancellation and makes a
    cropped online history equivalent to a full chronological sequence.
    ``valid_mask`` excludes padding, not prediction/lost observations.
    """
    if features.ndim != 3 or features.shape[-1] != CONTINUOUS_FEATURE_DIM:
        raise ValueError(f"features must have shape [B, T, {CONTINUOUS_FEATURE_DIM}]")
    if valid_mask.shape != features.shape[:2]:
        raise ValueError("valid_mask must have shape [B, T]")
    if features.shape[1] < 1 or window_s <= 0 or eps <= 0:
        raise ValueError("Nonempty sequences and positive window_s/eps are required")
    valid = valid_mask.to(device=features.device, dtype=torch.bool)
    # Right-padded rows contribute neither odometry nor regression points.
    raw = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features)).to(torch.float64)
    dt = raw[..., FeatureIndex.DT].clamp_min(0).clone()
    dt[:, 0] = 0.0
    time = torch.cumsum(dt, dim=1)
    rate = raw[..., 11]
    previous_rate = torch.cat((rate[:, :1], rate[:, :-1]), dim=1)
    yaw = torch.cumsum(0.5 * (previous_rate + rate) * dt, dim=1)

    def rotate(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(angle), torch.sin(angle)
        return torch.stack((c * vector[..., 0] - s * vector[..., 1],
                            s * vector[..., 0] + c * vector[..., 1], vector[..., 2]), dim=-1)

    base_velocity = raw[..., ROBOT_LINEAR_VELOCITY_SLICE]
    velocity_reference = rotate(base_velocity, yaw)
    previous_velocity = torch.cat((velocity_reference[:, :1], velocity_reference[:, :-1]), dim=1)
    displacement = torch.cumsum(0.5 * (velocity_reference + previous_velocity) * dt.unsqueeze(-1), dim=1)
    position_reference = rotate(raw[..., POSITION_SLICE], yaw) + displacement

    elapsed = time.unsqueeze(2) - time.unsqueeze(1)
    # Float32 dt encoding can shift a nominal boundary by several nanoseconds.
    tolerance = max(1e-9, float(window_s) * 1e-7)
    in_window = (elapsed >= 0) & (elapsed <= float(window_s) + tolerance)
    point_weight = (
        valid.to(raw.dtype)
        * raw[..., FeatureIndex.RANGE_VALID_MASK].clamp(0, 1)
        * (0.05 + 0.95 * raw[..., FeatureIndex.OBSERVED_MASK].clamp(0, 1))
        * (0.1 + 0.9 * raw[..., FeatureIndex.CONFIDENCE].clamp(0, 1))
    )
    if finite_difference_clip_mps is not None:
        _, reliability = robust_finite_difference(raw, clip_mps=finite_difference_clip_mps, eps=eps)
        point_weight = point_weight * reliability
    weights = in_window.to(raw.dtype) * point_weight.unsqueeze(1) * valid.unsqueeze(-1)
    total = weights.sum(dim=-1, keepdim=True)
    mean_time = (weights * time.unsqueeze(1)).sum(dim=-1, keepdim=True) / total.clamp_min(eps)
    centered = time.unsqueeze(1) - mean_time
    mean_position = torch.bmm(weights, position_reference) / total.clamp_min(eps)
    numerator = (
        weights.unsqueeze(-1) * centered.unsqueeze(-1)
        * (position_reference.unsqueeze(1) - mean_position.unsqueeze(2))
    ).sum(dim=2)
    denominator = (weights * centered.square()).sum(dim=-1, keepdim=True)
    target_velocity_reference = numerator / denominator.clamp_min(eps)
    # No additional omega x p: rotation was already removed before fitting.
    relative = rotate(target_velocity_reference, -yaw) - base_velocity
    enough_history = (weights > 0).sum(dim=-1, keepdim=True) >= 2
    return torch.where(enough_history & valid.unsqueeze(-1), relative, torch.zeros_like(relative)).to(features.dtype)


@torch.no_grad()
def _legacy_velocity_baseline(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    window_s: float = 0.48,
    finite_difference_clip_mps: float | None = DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Historical physical-relative-velocity baseline, unchanged for old weights.

    A weighted position fit first estimates ``d(position_base)/dt``.  Because
    the v2 label is physical relative translation rather than the derivative
    of coordinates in a rotating frame, the known yaw-frame term is then
    restored as ``omega_yaw x position_base``.  The z component of the stored
    robot angular velocity must therefore be the yaw-only frame heading rate.

    The function is vectorized over batch and time. Prediction frames receive
    a small non-zero weight so a short direct-observation outage degrades
    smoothly instead of resetting the baseline to zero.
    """
    if features.ndim != 3 or features.shape[-1] != CONTINUOUS_FEATURE_DIM:
        raise ValueError(
            "features must have shape [B, T, "
            f"{CONTINUOUS_FEATURE_DIM}], got {tuple(features.shape)}"
        )
    if valid_mask.shape != features.shape[:2]:
        raise ValueError("valid_mask must have shape [B, T]")
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")

    batch_size, time_steps, _ = features.shape
    dt = features[..., FeatureIndex.DT].clamp_min(0.0)
    absolute_time = torch.cumsum(dt, dim=1)
    position = features[..., POSITION_SLICE]
    observed = features[..., FeatureIndex.OBSERVED_MASK].clamp(0.0, 1.0)
    range_valid = features[..., FeatureIndex.RANGE_VALID_MASK].clamp(0.0, 1.0)
    confidence = features[..., FeatureIndex.CONFIDENCE].clamp(0.0, 1.0)
    if finite_difference_clip_mps is None:
        finite_difference_reliability = torch.ones_like(confidence)
    else:
        _, finite_difference_reliability = robust_finite_difference(
            features,
            clip_mps=finite_difference_clip_mps,
            eps=eps,
        )

    # current-time x history-time; only use causal points inside the time window.
    elapsed = absolute_time.unsqueeze(2) - absolute_time.unsqueeze(1)
    causal_window = (elapsed >= 0.0) & (elapsed <= float(window_s))
    history_valid = valid_mask.unsqueeze(1).to(dtype=features.dtype)
    current_valid = valid_mask.unsqueeze(2).to(dtype=features.dtype)
    source_weight = 0.05 + 0.95 * observed
    weights = (
        causal_window.to(dtype=features.dtype)
        * history_valid
        * current_valid
        * range_valid.unsqueeze(1)
        * source_weight.unsqueeze(1)
        * (0.1 + 0.9 * confidence).unsqueeze(1)
        * finite_difference_reliability.unsqueeze(1)
    )
    weight_sum = weights.sum(dim=-1, keepdim=True)
    history_time = absolute_time.unsqueeze(1)
    mean_time = (weights * history_time).sum(dim=-1, keepdim=True) / weight_sum.clamp_min(eps)
    centered_time = history_time - mean_time
    history_position = position.unsqueeze(1)
    mean_position = (weights.unsqueeze(-1) * history_position).sum(dim=2) / weight_sum.clamp_min(eps)
    numerator = (weights.unsqueeze(-1) * centered_time.unsqueeze(-1) * (history_position - mean_position.unsqueeze(2))).sum(dim=2)
    denominator = (weights * centered_time.square()).sum(dim=-1, keepdim=True)
    position_derivative = numerator / denominator.clamp_min(eps)

    enough_history = (weights > 0.0).sum(dim=-1, keepdim=True) >= 2
    yaw_omega = torch.zeros_like(position)
    yaw_omega[..., 2] = features[..., ROBOT_ANGULAR_VELOCITY_SLICE][..., 2]
    baseline = position_derivative + torch.cross(yaw_omega, position, dim=-1)
    baseline = torch.where(enough_history, baseline, torch.zeros_like(baseline))
    return baseline * valid_mask.unsqueeze(-1).to(dtype=features.dtype)


def save_taq_velocity_chunk(
    path: str | Path,
    *,
    sequence_features: torch.Tensor,
    sequence_source_ids: torch.Tensor,
    sequence_valid_mask: torch.Tensor,
    target_velocity_base: torch.Tensor,
    target_valid_mask: torch.Tensor,
    sample_window_start_s: torch.Tensor | Sequence[float] | None = None,
    sample_window_end_s: torch.Tensor | Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a schema-versioned training chunk for the TAQ-GRU."""
    features = torch.as_tensor(sequence_features, dtype=torch.float32).cpu()
    sources = torch.as_tensor(sequence_source_ids, dtype=torch.long).cpu()
    valid = torch.as_tensor(sequence_valid_mask, dtype=torch.bool).cpu()
    target = torch.as_tensor(target_velocity_base, dtype=torch.float32).cpu()
    if features.ndim != 3 or features.shape[-1] != CONTINUOUS_FEATURE_DIM:
        raise ValueError("sequence_features must have shape [N, T, 24]")
    if sources.shape != features.shape[:2] or valid.shape != features.shape[:2]:
        raise ValueError("source IDs and valid mask must have shape [N, T]")
    expected_target_shape = (features.shape[0], features.shape[1], 3)
    if target.shape != expected_target_shape:
        raise ValueError(f"target_velocity_base must have shape [N, T, 3], got {tuple(target.shape)}")
    target_mask = torch.as_tensor(target_valid_mask, dtype=torch.bool).cpu()
    if target_mask.shape != features.shape[:2]:
        raise ValueError("target_valid_mask must have shape [N, T]")
    if torch.any(target_mask & ~valid):
        raise ValueError("target_valid_mask cannot mark padded time steps as valid")
    if not torch.isfinite(features).all() or not torch.isfinite(target).all():
        raise ValueError("v2 chunks cannot contain non-finite features or labels")
    chunk_metadata = dict(metadata or {})
    label_semantics = chunk_metadata.get("label_semantics", TAQ_LABEL_SEMANTICS)
    if label_semantics != TAQ_LABEL_SEMANTICS:
        raise ValueError(
            f"v2 chunks require label_semantics={TAQ_LABEL_SEMANTICS!r}, got {label_semantics!r}"
        )
    chunk_metadata["label_semantics"] = TAQ_LABEL_SEMANTICS
    chunk_metadata.setdefault("angular_velocity_z_semantics", TAQ_YAW_RATE_SEMANTICS)
    payload: dict[str, Any] = {
        "schema_version": TAQ_SCHEMA_VERSION,
        "sequence_features": features,
        "sequence_source_ids": sources,
        "sequence_valid_mask": valid,
        "target_velocity_base": target,
        "target_valid_mask": target_mask,
        "metadata": chunk_metadata,
    }
    if (sample_window_start_s is None) != (sample_window_end_s is None):
        raise ValueError(
            "sample_window_start_s and sample_window_end_s must be provided together"
        )
    if sample_window_start_s is not None and sample_window_end_s is not None:
        window_start = torch.as_tensor(sample_window_start_s, dtype=torch.float64).cpu()
        window_end = torch.as_tensor(sample_window_end_s, dtype=torch.float64).cpu()
        expected_shape = (features.shape[0],)
        if window_start.shape != expected_shape or window_end.shape != expected_shape:
            raise ValueError("sample window timestamps must have shape [N]")
        if not torch.isfinite(window_start).all() or not torch.isfinite(window_end).all():
            raise ValueError("sample window timestamps must be finite")
        if torch.any(window_end <= window_start):
            raise ValueError("every sample window end must be later than its start")
        payload["sample_window_start_s"] = window_start
        payload["sample_window_end_s"] = window_end
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


class TAQVelocityChunkDataset(Dataset):
    """One schema-v2 chunk.  Samples are trimmed before collation."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = load_torch_payload(self.path)
        if payload.get("schema_version") != TAQ_SCHEMA_VERSION:
            found = payload.get("schema_version", "missing")
            raise ValueError(
                f"{self.path} has schema_version={found!r}; expected {TAQ_SCHEMA_VERSION!r}. "
                "v1/window-end chunks cannot be mixed with v2 training."
            )
        try:
            self.features = payload["sequence_features"].to(torch.float32).contiguous()
            self.source_ids = payload["sequence_source_ids"].to(torch.long).contiguous()
            self.valid_mask = payload["sequence_valid_mask"].to(torch.bool).contiguous()
            target = payload["target_velocity_base"].to(torch.float32)
        except KeyError as exc:
            raise KeyError(f"{self.path} is missing required key {exc.args[0]!r}") from exc
        if self.features.ndim != 3 or self.features.shape[-1] != CONTINUOUS_FEATURE_DIM:
            raise ValueError(f"{self.path} sequence_features must have shape [N, T, 24]")
        if self.source_ids.shape != self.features.shape[:2] or self.valid_mask.shape != self.features.shape[:2]:
            raise ValueError(f"{self.path} source IDs / valid mask shape mismatch")
        if not torch.all(self.valid_mask[:, :1]):
            raise ValueError(f"{self.path} requires the first time step of every sample to be valid")
        # Padding must be right-aligned to make per-sample trimming unambiguous.
        if torch.any(self.valid_mask[:, 1:] & ~self.valid_mask[:, :-1]):
            raise ValueError(f"{self.path} valid mask must be a contiguous prefix")
        if torch.any((self.source_ids < 0) | (self.source_ids > int(TrackerSource.INVALID))):
            raise ValueError(f"{self.path} contains invalid source IDs")

        sample_count, time_steps, _ = self.features.shape
        if target.shape != (sample_count, time_steps, 3):
            raise ValueError(
                f"{self.path} v2 target_velocity_base must have shape [N, T, 3], got {tuple(target.shape)}"
            )
        self.target = target.contiguous()
        target_mask = payload.get("target_valid_mask")
        if target_mask is None:
            raise KeyError(f"{self.path} is missing required v2 key 'target_valid_mask'")
        self.target_valid_mask = torch.as_tensor(target_mask, dtype=torch.bool).contiguous()
        if self.target_valid_mask.shape != self.valid_mask.shape:
            raise ValueError(f"{self.path} target_valid_mask shape mismatch")
        self.target_valid_mask &= self.valid_mask
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.target).all():
            raise ValueError(f"{self.path} contains non-finite features or labels")
        self.metadata = dict(payload.get("metadata", {}))
        if self.metadata.get("label_semantics") != TAQ_LABEL_SEMANTICS:
            raise ValueError(f"{self.path} has missing or incompatible v2 label_semantics")
        if self.metadata.get("angular_velocity_z_semantics") != TAQ_YAW_RATE_SEMANTICS:
            raise ValueError(f"{self.path} has missing or incompatible yaw-rate semantics")
        scenario = str(self.metadata.get("scenario", "unknown")).lower()
        if scenario not in {"ss", "sd"}:
            raise ValueError(f"{self.path} has unsupported or missing scenario={scenario!r}")
        self.scenario_id = 0 if scenario == "ss" else 1
        self.episode_uid = str(self.metadata.get("episode_uid", self.path.parent.as_posix()))
        self.speed_bin = self._compute_speed_bins()

    def _compute_speed_bins(self) -> torch.Tensor:
        # Horizontal speed is invariant under the yaw-only rotation.  Adding
        # the robot's base-frame translation reconstructs target world speed
        # without requiring yaw itself in the chunk.
        target_velocity_base = self.target + self.features[..., ROBOT_LINEAR_VELOCITY_SLICE]
        speed = torch.linalg.vector_norm(target_velocity_base[..., :2], dim=-1)
        valid = self.target_valid_mask
        mean_speed = (speed * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        # 0/1/2 are the prescribed SD bins; SS stays in bin 0.
        return torch.where(mean_speed < 2.0, 0, torch.where(mean_speed < 3.5, 1, 2)).to(torch.long)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        length = int(self.valid_mask[index].sum().item())
        return {
            "features": self.features[index, :length],
            "source_ids": self.source_ids[index, :length],
            "valid_mask": self.valid_mask[index, :length],
            "target": self.target[index, :length],
            "target_valid_mask": self.target_valid_mask[index, :length],
            "scenario_id": torch.tensor(self.scenario_id, dtype=torch.long),
        }


def list_taq_chunk_files(data_dirs: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for directory in data_dirs:
        path = Path(directory).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"TAQ data directory not found: {path}")
        # TAQ collection stores chunks below env/episode directories so that
        # complete trajectories can be split without leaking neighbouring
        # windows across train/validation/test.  ``rglob`` also keeps support
        # for a flat directory of chunks.
        files.extend(sorted(path.rglob("*.pt")))
    if not files:
        raise FileNotFoundError("No .pt TAQ velocity chunks found")
    return files


def build_taq_dataset(data_dirs: Iterable[str | Path]) -> Dataset:
    datasets = [TAQVelocityChunkDataset(path) for path in list_taq_chunk_files(data_dirs)]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _chunk_datasets(dataset: Dataset) -> list[TAQVelocityChunkDataset]:
    if isinstance(dataset, TAQVelocityChunkDataset):
        return [dataset]
    if isinstance(dataset, ConcatDataset):
        return [item for item in dataset.datasets if isinstance(item, TAQVelocityChunkDataset)]
    raise TypeError("Expected TAQVelocityChunkDataset or ConcatDataset")


def _flatten_taq_chunks(dataset: Dataset) -> list[TAQVelocityChunkDataset]:
    """Flatten a dataset while preserving the chunk metadata used for sampling."""
    return _chunk_datasets(dataset)


def taq_scenario_window_counts(dataset: Dataset) -> dict[str, int]:
    """Count SS/SD windows and fail if a dataset contains unknown scenarios."""
    counts = {"ss": 0, "sd": 0}
    for chunk in _chunk_datasets(dataset):
        name = "ss" if chunk.scenario_id == 0 else "sd" if chunk.scenario_id == 1 else None
        if name is None:
            raise ValueError(f"Unsupported scenario_id={chunk.scenario_id} in {chunk.path}")
        counts[name] += len(chunk)
    return counts


def compute_episode_stratified_weights(
    dataset: Dataset,
    *,
    ss_weight: float = 0.40,
    sd_speed_weights: tuple[float, float, float] = (0.30, 0.35, 0.35),
) -> torch.Tensor:
    """Return per-window weights for scenario/speed-bin/episode sampling."""
    if not 0.0 < ss_weight < 1.0 or len(sd_speed_weights) != 3:
        raise ValueError("Invalid scenario or SD speed-bin weights")
    if any(weight <= 0.0 for weight in sd_speed_weights):
        raise ValueError("SD speed-bin weights must be positive")
    chunks = _flatten_taq_chunks(dataset)
    entries: list[tuple[int, int, str]] = []
    for chunk in chunks:
        entries.extend(
            (chunk.scenario_id, int(speed_bin), chunk.episode_uid)
            for speed_bin in chunk.speed_bin.tolist()
        )
    group_counts: dict[tuple[int, int], int] = {}
    episode_counts: dict[tuple[int, int, str], int] = {}
    for scenario_id, speed_bin, episode_uid in entries:
        group = (scenario_id, speed_bin if scenario_id == 1 else 0)
        group_counts[group] = group_counts.get(group, 0) + 1
        episode_key = (*group, episode_uid)
        episode_counts[episode_key] = episode_counts.get(episode_key, 0) + 1
    group_episode_ids: dict[tuple[int, int], set[str]] = {}
    for scenario_id, speed_bin, episode_uid in entries:
        group = (scenario_id, speed_bin if scenario_id == 1 else 0)
        group_episode_ids.setdefault(group, set()).add(episode_uid)
    weights: list[float] = []
    for scenario_id, speed_bin, episode_uid in entries:
        group = (scenario_id, speed_bin if scenario_id == 1 else 0)
        if scenario_id == 0:
            group_mass = ss_weight
        elif scenario_id == 1:
            group_mass = (1.0 - ss_weight) * float(sd_speed_weights[speed_bin])
        else:  # TAQVelocityChunkDataset rejects this before sampling.
            raise ValueError(f"Unsupported scenario_id={scenario_id}")
        episode_key = (*group, episode_uid)
        weights.append(
            group_mass
            / max(len(group_episode_ids[group]), 1)
            / max(episode_counts[episode_key], 1)
        )
    result = torch.tensor(weights, dtype=torch.double)
    if not torch.isfinite(result).all() or torch.any(result <= 0.0):
        raise ValueError("Unable to construct positive finite episode-stratified weights")
    return result


def compute_taq_feature_norm_stats(
    dataset: Dataset,
    eps: float = 1e-6,
    *,
    finite_difference_clip_mps: float | None = DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute valid-frame statistics after the model's robust FD transform.

    ``sample_weights`` may contain the hierarchical training probability for
    each window; when supplied, normalization matches the sampler's quality
    distribution instead of the raw number of windows.
    """
    total = torch.zeros(CONTINUOUS_FEATURE_DIM, dtype=torch.float64)
    total_sq = torch.zeros(CONTINUOUS_FEATURE_DIM, dtype=torch.float64)
    count = 0
    offset = 0
    for chunk in _chunk_datasets(dataset):
        values = chunk.features.to(torch.float64)
        if finite_difference_clip_mps is not None:
            clipped, _ = robust_finite_difference(
                values,
                clip_mps=finite_difference_clip_mps,
                eps=eps,
            )
            values = values.clone()
            values[..., FINITE_DIFFERENCE_SLICE] = clipped
        mask = chunk.valid_mask
        if sample_weights is None:
            frame_weights = mask.to(torch.float64)
        else:
            weights = torch.as_tensor(sample_weights, dtype=torch.float64).reshape(-1)
            if weights.numel() != len(dataset):
                raise ValueError("sample_weights must have one value per dataset window")
            frame_weights = mask.to(torch.float64) * weights[offset : offset + len(chunk)].unsqueeze(-1)
        total += (values * frame_weights.unsqueeze(-1)).sum(dim=(0, 1))
        total_sq += (values.square() * frame_weights.unsqueeze(-1)).sum(dim=(0, 1))
        count += int(frame_weights.sum().item()) if sample_weights is None else float(frame_weights.sum().item())
        offset += len(chunk)
    if count <= 0:
        raise ValueError("No valid frames available for normalization")
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(0.0)
    return mean.to(torch.float32), variance.sqrt().clamp_min(eps).to(torch.float32)


def collate_taq_velocity_batch(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad variable-duration sequences for a DataLoader."""
    if not samples:
        raise ValueError("Cannot collate an empty TAQ batch")
    batch_size = len(samples)
    max_steps = max(int(sample["features"].shape[0]) for sample in samples)
    features = torch.zeros((batch_size, max_steps, CONTINUOUS_FEATURE_DIM), dtype=torch.float32)
    source_ids = torch.full((batch_size, max_steps), int(TrackerSource.INVALID), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    target = torch.zeros((batch_size, max_steps, 3), dtype=torch.float32)
    target_valid_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    scenario_id = torch.full((batch_size,), -1, dtype=torch.long)
    for index, sample in enumerate(samples):
        length = int(sample["features"].shape[0])
        features[index, :length] = sample["features"]
        source_ids[index, :length] = sample["source_ids"]
        valid_mask[index, :length] = sample["valid_mask"]
        target[index, :length] = sample["target"]
        target_valid_mask[index, :length] = sample["target_valid_mask"]
        if "scenario_id" in sample:
            scenario_id[index] = sample["scenario_id"]
    return {
        "features": features,
        "source_ids": source_ids,
        "valid_mask": valid_mask,
        "target": target,
        "target_valid_mask": target_valid_mask,
        "scenario_id": scenario_id,
    }

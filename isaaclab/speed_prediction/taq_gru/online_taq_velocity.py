"""Per-target online state manager for the TAQ-GRU velocity model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..common.time_aware_velocity_data import (
    LEGACY_BASELINE_VERSION,
    SOURCE_NAME_TO_ID,
    TrackerSource,
    build_time_aware_features,
)
from .time_aware_velocity_model import TimeAwareQualityGatedGRU


@dataclass(frozen=True)
class TAQObservation:
    """One target observation already expressed in yaw-only base coordinates.

    ``robot_angular_velocity_base_rps[2]`` must be the derivative of the
    yaw-only frame heading.  ``yaw_rate_rps`` can override that component when
    the caller computes the exact rate from orientation and 3-D angular
    velocity.
    """

    timestamp_s: float
    position_base_m: Sequence[float]
    robot_linear_velocity_base_mps: Sequence[float]
    robot_angular_velocity_base_rps: Sequence[float]
    source: str | int
    yaw_rate_rps: float | None = None
    quality: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    range_valid: bool = True


@dataclass(frozen=True)
class OnlineVelocityEstimate:
    target_id: int
    timestamp_s: float
    velocity_mean_mps: torch.Tensor
    velocity_log_sigma: torch.Tensor
    velocity_baseline_mps: torch.Tensor
    reliability_gate: float


@dataclass
class _TargetState:
    observations: list[TAQObservation] = field(default_factory=list)
    hidden: torch.Tensor | None = None
    last_dropped_observation_time_s: float | None = None
    last_dropped_fine_time_s: float | None = None


class OnlineTAQVelocityEstimator:
    """Maintain per-target state; aligned baselines retain the full time window.

    ``max_history_steps`` remains a hard cap for legacy baseline checkpoints.
    Aligned checkpoints use the time bound so higher frame rates do not
    silently shorten the baseline window relative to offline computation.
    """

    def __init__(
        self,
        model: TimeAwareQualityGatedGRU,
        *,
        history_window_s: float | None = None,
        max_history_steps: int = 32,
        device: torch.device | str | None = None,
    ) -> None:
        if history_window_s is None:
            history_window_s = model.baseline_window_s
        if history_window_s <= 0.0 or max_history_steps < 2:
            raise ValueError("history_window_s must be positive and max_history_steps must be >= 2")
        self.model = model
        self.history_window_s = float(history_window_s)
        if model.baseline_version != LEGACY_BASELINE_VERSION:
            self.history_window_s = max(self.history_window_s, model.baseline_window_s)
        self.max_history_steps = int(max_history_steps)
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self._states: dict[int, _TargetState] = {}

    @staticmethod
    def _source_id(source: str | int) -> int:
        if isinstance(source, str):
            try:
                return int(SOURCE_NAME_TO_ID[source.lower()])
            except KeyError as exc:
                raise ValueError(f"Unknown tracker source {source!r}") from exc
        source_id = int(source)
        if source_id < int(TrackerSource.INITIAL) or source_id > int(TrackerSource.INVALID):
            raise ValueError(f"Unsupported tracker source ID {source_id}")
        return source_id

    def reset(self, target_id: int | None = None) -> None:
        if target_id is None:
            self._states.clear()
        else:
            self._states.pop(int(target_id), None)

    @staticmethod
    def _drop_oldest_observation(state: _TargetState) -> None:
        item = state.observations.pop(0)
        source_id = OnlineTAQVelocityEstimator._source_id(item.source)
        if item.range_valid and source_id in {
            int(TrackerSource.INITIAL), int(TrackerSource.FLOW), int(TrackerSource.OSTRACK)
        }:
            state.last_dropped_observation_time_s = float(item.timestamp_s)
        if item.range_valid and source_id == int(TrackerSource.OSTRACK):
            state.last_dropped_fine_time_s = float(item.timestamp_s)

    def _trim_history(self, state: _TargetState) -> None:
        latest_time = state.observations[-1].timestamp_s
        threshold = latest_time - self.history_window_s
        while len(state.observations) > 2 and state.observations[1].timestamp_s < threshold:
            self._drop_oldest_observation(state)
        if self.model.baseline_version == LEGACY_BASELINE_VERSION:
            while len(state.observations) > self.max_history_steps:
                self._drop_oldest_observation(state)

    @torch.no_grad()
    def update(self, target_id: int, observation: TAQObservation) -> OnlineVelocityEstimate:
        """Update one target and return its current velocity estimate.

        The caller should invoke ``reset(target_id)`` after long loss or a
        confirmed re-identification.  A non-monotonic timestamp is rejected
        because it would invalidate the causal finite-difference feature.
        """
        identifier = int(target_id)
        state = self._states.setdefault(identifier, _TargetState())
        if state.observations and observation.timestamp_s <= state.observations[-1].timestamp_s:
            raise ValueError("Per-target observation timestamps must be strictly increasing")
        if len(observation.position_base_m) != 3 or len(observation.quality) != 7:
            raise ValueError("position_base_m must have 3 values and quality must have 7 values")
        state.observations.append(observation)
        self._trim_history(state)

        # Preserve sub-frame deltas even when the absolute monotonic timestamp
        # has grown large; the feature builder only casts relative dt to f32.
        timestamps = torch.tensor([item.timestamp_s for item in state.observations], dtype=torch.float64)
        position = torch.tensor([item.position_base_m for item in state.observations], dtype=torch.float32)
        robot_linear = torch.tensor(
            [item.robot_linear_velocity_base_mps for item in state.observations], dtype=torch.float32
        )
        robot_angular_rows = []
        for item in state.observations:
            angular = list(item.robot_angular_velocity_base_rps)
            if len(angular) != 3:
                raise ValueError("robot_angular_velocity_base_rps must have 3 values")
            if item.yaw_rate_rps is not None:
                angular[2] = float(item.yaw_rate_rps)
            robot_angular_rows.append(angular)
        robot_angular = torch.tensor(robot_angular_rows, dtype=torch.float32)
        source_ids = torch.tensor([self._source_id(item.source) for item in state.observations], dtype=torch.long)
        quality = torch.tensor([item.quality for item in state.observations], dtype=torch.float32)
        range_valid = torch.tensor([item.range_valid for item in state.observations], dtype=torch.bool)
        features, _ = build_time_aware_features(
            timestamps_s=timestamps,
            position_base_m=position,
            robot_linear_velocity_base_mps=robot_linear,
            robot_angular_velocity_base_rps=robot_angular,
            source_ids=source_ids,
            quality=quality,
            range_valid_mask=range_valid,
            initial_last_observation_time_s=state.last_dropped_observation_time_s,
            initial_last_fine_time_s=state.last_dropped_fine_time_s,
        )
        features = features.to(self.device)
        source_ids = source_ids.to(self.device)
        valid_mask = torch.ones((1, features.shape[0]), dtype=torch.bool, device=self.device)
        baseline = self.model.compute_velocity_baseline(
            features.unsqueeze(0),
            valid_mask,
            window_s=(self.history_window_s if self.model.baseline_version == LEGACY_BASELINE_VERSION
                      else self.model.baseline_window_s),
        )[:, -1]
        mean, log_sigma, hidden, gate = self.model.step(
            features[-1:].contiguous(),
            source_ids[-1:].contiguous(),
            hidden=state.hidden,
            velocity_baseline=baseline,
        )
        state.hidden = hidden.detach()
        return OnlineVelocityEstimate(
            target_id=identifier,
            timestamp_s=float(observation.timestamp_s),
            velocity_mean_mps=mean[0].detach().clone(),
            velocity_log_sigma=log_sigma[0].detach().clone(),
            velocity_baseline_mps=baseline[0].detach().clone(),
            reliability_gate=float(gate[0, 0].item()),
        )

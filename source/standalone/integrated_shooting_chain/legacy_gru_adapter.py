"""Adapt the frozen test11 ordinary GRU to the full-chain velocity interface."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LegacyGRUVelocityEstimate:
    target_id: int
    timestamp_s: float
    velocity_mean_mps: torch.Tensor
    reliability_gate: float


class OnlineLegacyGRUEstimator:
    """Share frozen weights while keeping independent per-target rolling histories."""

    preserve_state_on_missing = True

    def __init__(self, model: Any, *, device: str | torch.device = "cpu",
                 history_window_s: float = 0.48, max_history_steps: int = 32,
                 smooth_window: int = 1) -> None:
        if history_window_s <= 0 or max_history_steps < 2 or smooth_window < 1:
            raise ValueError("Invalid legacy GRU history configuration")
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.history_window_s = float(history_window_s)
        self.max_history_steps = int(max_history_steps)
        self.smooth_window = int(smooth_window)
        self._states: dict[int, Any] = {}

    def reset(self, target_id: int | None = None) -> None:
        if target_id is None:
            self._states.clear()
        else:
            self._states.pop(int(target_id), None)

    def update(self, target_id: int, observation: Any) -> LegacyGRUVelocityEstimate | None:
        from speed_prediction.common.time_aware_velocity_data import SOURCE_NAME_TO_ID, TrackerSource
        from speed_prediction.legacy_gru.online_gru_velocity import OnlineLegacyGRU

        identifier = int(target_id)
        source = observation.source
        source_id = int(SOURCE_NAME_TO_ID[source.lower()]) if isinstance(source, str) else int(source)
        if not int(TrackerSource.INITIAL) <= source_id <= int(TrackerSource.INVALID):
            raise ValueError(f"Invalid tracker source: {source}")
        position = np.asarray(observation.position_base_m, dtype=np.float32)
        angular = np.asarray(observation.robot_angular_velocity_base_rps, dtype=np.float32).copy()
        if position.shape != (3,) or angular.shape != (3,):
            raise ValueError("Position and robot angular velocity must be three-vectors")
        yaw_rate = getattr(observation, "yaw_rate_rps", None)
        if yaw_rate is not None:
            angular[2] = float(yaw_rate)
        valid_range = bool(observation.range_valid) and bool(np.isfinite(position).all())
        if not valid_range:
            position = np.full(3, np.nan, dtype=np.float32)
        direct = valid_range and source_id <= int(TrackerSource.OSTRACK)
        if identifier not in self._states:
            self._states[identifier] = OnlineLegacyGRU(
                self.model, device=self.device, context="rolling", window_s=self.history_window_s,
                max_history_frames=self.max_history_steps, smooth_window=self.smooth_window,
                initialization_policy="first_valid",
            )
        result = self._states[identifier].update(
            timestamp_s=observation.timestamp_s, position_base_m=position,
            robot_linear_velocity_base_mps=observation.robot_linear_velocity_base_mps,
            robot_angular_velocity_base_rps=angular, episode_key=identifier,
            direct_observation=direct,
        )
        if result is None:
            return None
        return LegacyGRUVelocityEstimate(
            target_id=identifier, timestamp_s=float(observation.timestamp_s),
            velocity_mean_mps=torch.as_tensor(result["velocity"], device=self.device),
            reliability_gate=float(direct),
        )

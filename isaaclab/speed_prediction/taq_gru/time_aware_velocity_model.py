"""Time-aware, quality-gated GRU for visual target velocity estimation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..common.time_aware_velocity_data import (
    CONTINUOUS_FEATURE_DIM,
    DEFAULT_BASELINE_VERSION,
    DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
    DEFAULT_NORMALIZED_FEATURE_CLIP,
    FINITE_DIFFERENCE_SLICE,
    FeatureIndex,
    LEGACY_INPUT_TRANSFORM_VERSION,
    POSITION_SLICE,
    QUALITY_SLICE,
    ROBUST_INPUT_TRANSFORM_VERSION,
    ROBOT_ANGULAR_VELOCITY_SLICE,
    ROBOT_LINEAR_VELOCITY_SLICE,
    SUPPORTED_BASELINE_VERSIONS,
    TrackerSource,
    robust_finite_difference,
    weighted_velocity_baseline,
)


class TimeAwareQualityGatedGRU(nn.Module):
    """Causal velocity estimator for LK/OSTrack/prediction observations.

    Input features are continuous TAQ v2 features ``[B, T, 24]`` plus source
    IDs and a padding mask.  The network outputs one mean velocity and one
    diagonal log-standard-deviation per time step.
    """

    def __init__(
        self,
        *,
        continuous_dim: int = CONTINUOUS_FEATURE_DIM,
        source_vocab_size: int = len(TrackerSource),
        source_embedding_dim: int = 4,
        measurement_hidden_dim: int = 48,
        measurement_dim: int = 64,
        context_hidden_dim: int = 48,
        context_dim: int = 48,
        hidden_dim: int = 128,
        head_hidden_dim: int = 64,
        dropout: float = 0.05,
        max_dt_s: float = 0.25,
        baseline_window_s: float = 0.48,
        baseline_version: str = DEFAULT_BASELINE_VERSION,
        input_transform_version: str = ROBUST_INPUT_TRANSFORM_VERSION,
        finite_difference_clip_mps: float = DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
        normalized_feature_clip: float = DEFAULT_NORMALIZED_FEATURE_CLIP,
    ) -> None:
        super().__init__()
        if continuous_dim != CONTINUOUS_FEATURE_DIM:
            raise ValueError(
                f"TAQ v2 requires continuous_dim={CONTINUOUS_FEATURE_DIM}, got {continuous_dim}"
            )
        if source_vocab_size < len(TrackerSource):
            raise ValueError("source_vocab_size is smaller than the TAQ source vocabulary")
        if hidden_dim < 1 or measurement_dim < 1 or context_dim < 1:
            raise ValueError("Network dimensions must be positive")
        if max_dt_s <= 0.0 or baseline_window_s <= 0.0:
            raise ValueError("max_dt_s and baseline_window_s must be positive")
        if baseline_version not in SUPPORTED_BASELINE_VERSIONS:
            raise ValueError(f"Unsupported baseline_version={baseline_version!r}")
        if input_transform_version not in {
            ROBUST_INPUT_TRANSFORM_VERSION,
            LEGACY_INPUT_TRANSFORM_VERSION,
        }:
            raise ValueError(f"Unsupported input_transform_version={input_transform_version!r}")
        if finite_difference_clip_mps <= 0.0 or normalized_feature_clip <= 0.0:
            raise ValueError("finite_difference_clip_mps and normalized_feature_clip must be positive")

        self.continuous_dim = int(continuous_dim)
        self.source_vocab_size = int(source_vocab_size)
        self.source_embedding_dim = int(source_embedding_dim)
        self.measurement_hidden_dim = int(measurement_hidden_dim)
        self.measurement_dim = int(measurement_dim)
        self.context_hidden_dim = int(context_hidden_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.dropout = float(dropout)
        self.max_dt_s = float(max_dt_s)
        self.baseline_window_s = float(baseline_window_s)
        self.baseline_version = str(baseline_version)
        self.input_transform_version = str(input_transform_version)
        self.finite_difference_clip_mps = float(finite_difference_clip_mps)
        self.normalized_feature_clip = float(normalized_feature_clip)
        self.robust_input_enabled = self.input_transform_version == ROBUST_INPUT_TRANSFORM_VERSION

        self.source_embedding = nn.Embedding(self.source_vocab_size, self.source_embedding_dim)
        self.measurement_encoder = nn.Sequential(
            nn.Linear(6, self.measurement_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.measurement_hidden_dim, self.measurement_dim),
            nn.SiLU(),
        )
        # Robust mode appends one causal finite-difference reliability value.
        context_continuous_dim = 18 + int(self.robust_input_enabled)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_continuous_dim + self.source_embedding_dim, self.context_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.context_hidden_dim, self.context_dim),
            nn.SiLU(),
        )
        self.reliability_gate = nn.Sequential(
            nn.Linear(context_continuous_dim + self.source_embedding_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.gru_cell = nn.GRUCell(self.measurement_dim + self.context_dim, self.hidden_dim)
        initial_tau = 0.15
        self.log_time_constant = nn.Parameter(
            torch.full((self.hidden_dim,), math.log(math.expm1(initial_tau)), dtype=torch.float32)
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.head_hidden_dim),
            nn.SiLU(),
        )
        self.velocity_delta_head = nn.Linear(self.head_hidden_dim, 3)
        self.log_sigma_head = nn.Linear(self.head_hidden_dim, 3)

        self.register_buffer("input_mean", torch.zeros(1, 1, self.continuous_dim))
        self.register_buffer("input_std", torch.ones(1, 1, self.continuous_dim))
        self.register_buffer("norm_enabled", torch.tensor(False, dtype=torch.bool))

    @property
    def config(self) -> dict[str, Any]:
        return {
            "continuous_dim": self.continuous_dim,
            "source_vocab_size": self.source_vocab_size,
            "source_embedding_dim": self.source_embedding_dim,
            "measurement_hidden_dim": self.measurement_hidden_dim,
            "measurement_dim": self.measurement_dim,
            "context_hidden_dim": self.context_hidden_dim,
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "head_hidden_dim": self.head_hidden_dim,
            "dropout": self.dropout,
            "max_dt_s": self.max_dt_s,
            "baseline_window_s": self.baseline_window_s,
            "baseline_version": self.baseline_version,
            "input_transform_version": self.input_transform_version,
            "finite_difference_clip_mps": self.finite_difference_clip_mps,
            "normalized_feature_clip": self.normalized_feature_clip,
        }

    @torch.no_grad()
    def compute_velocity_baseline(
        self, features: torch.Tensor, valid_mask: torch.Tensor, *, window_s: float | None = None,
    ) -> torch.Tensor:
        return weighted_velocity_baseline(
            features, valid_mask,
            window_s=self.baseline_window_s if window_s is None else window_s,
            finite_difference_clip_mps=self.finite_difference_clip_mps if self.robust_input_enabled else None,
            baseline_version=self.baseline_version,
        )

    @torch.no_grad()
    def set_input_norm_stats(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.input_mean.device).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32, device=self.input_std.device).reshape(-1)
        if mean.numel() != self.continuous_dim or std.numel() != self.continuous_dim:
            raise ValueError(
                f"Expected {self.continuous_dim} normalization values, got {mean.numel()} / {std.numel()}"
            )
        self.input_mean.copy_(mean.view(1, 1, -1))
        self.input_std.copy_(std.clamp_min(eps).view(1, 1, -1))
        self.norm_enabled.fill_(True)

    def normalize_input(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.norm_enabled.item()):
            return features
        if features.ndim == 2:
            return (features - self.input_mean[:, 0, :]) / self.input_std[:, 0, :]
        if features.ndim == 3:
            return (features - self.input_mean) / self.input_std
        raise ValueError(f"Expected features with shape [B, F] or [B, T, F], got {tuple(features.shape)}")

    def _prepare_input(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the checkpointed causal transform and return FD reliability."""
        if self.robust_input_enabled:
            clipped_fd, fd_reliability = robust_finite_difference(
                features,
                clip_mps=self.finite_difference_clip_mps,
            )
            transformed = features.clone()
            transformed[..., FINITE_DIFFERENCE_SLICE] = clipped_fd
        else:
            transformed = features
            fd_reliability = torch.ones(
                features.shape[:-1], dtype=features.dtype, device=features.device
            )
        normalized = self.normalize_input(transformed)
        if self.robust_input_enabled:
            normalized = normalized.clamp(
                -self.normalized_feature_clip,
                self.normalized_feature_clip,
            )
            # Binary masks are categorical indicators, not continuous sensor
            # values.  Keep their exact 0/1 semantics after normalization.
            normalized = normalized.clone()
            normalized[..., FeatureIndex.OBSERVED_MASK] = features[
                ..., FeatureIndex.OBSERVED_MASK
            ].clamp(0.0, 1.0)
            normalized[..., FeatureIndex.RANGE_VALID_MASK] = features[
                ..., FeatureIndex.RANGE_VALID_MASK
            ].clamp(0.0, 1.0)
        return normalized, fd_reliability

    def initial_hidden(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device or self.input_mean.device,
            dtype=dtype or self.input_mean.dtype,
        )

    def _encode_step(self, features: torch.Tensor, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.continuous_dim:
            raise ValueError(f"Expected step features [B, {self.continuous_dim}]")
        if source_ids.ndim != 1 or source_ids.shape[0] != features.shape[0]:
            raise ValueError("source_ids must have shape [B]")
        if torch.any((source_ids < 0) | (source_ids >= self.source_vocab_size)):
            raise ValueError("source_ids contains an out-of-vocabulary value")

        normalized, fd_reliability = self._prepare_input(features)
        measurement = torch.cat(
            (normalized[:, POSITION_SLICE], normalized[:, FINITE_DIFFERENCE_SLICE]), dim=-1
        )
        context_continuous = torch.cat(
            (
                normalized[:, ROBOT_LINEAR_VELOCITY_SLICE],
                normalized[:, ROBOT_ANGULAR_VELOCITY_SLICE],
                normalized[:, FeatureIndex.DT : FeatureIndex.FINE_AGE + 1],
                normalized[:, QUALITY_SLICE],
                normalized[:, FeatureIndex.OBSERVED_MASK : FeatureIndex.RANGE_VALID_MASK + 1],
                *(
                    (fd_reliability.unsqueeze(-1),)
                    if self.robust_input_enabled
                    else ()
                ),
            ),
            dim=-1,
        )
        source_embedding = self.source_embedding(source_ids)
        context_input = torch.cat((context_continuous, source_embedding), dim=-1)
        gate = self.reliability_gate(context_input)
        if self.robust_input_enabled:
            gate = gate * fd_reliability.unsqueeze(-1)
        measurement_encoded = self.measurement_encoder(measurement) * gate
        context_encoded = self.context_encoder(context_input)
        encoded = torch.cat((measurement_encoded, context_encoded), dim=-1)
        return encoded, gate, features[:, FeatureIndex.DT].clamp(0.0, self.max_dt_s)

    def step(
        self,
        features: torch.Tensor,
        source_ids: torch.Tensor,
        *,
        hidden: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        velocity_baseline: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one causal update for one or more target IDs.

        ``valid_mask`` is for padding only.  A prediction/lost observation must
        still be passed as a valid time step with its explicit source and masks.
        """
        encoded, gate, dt = self._encode_step(features, source_ids)
        batch_size = features.shape[0]
        if hidden is None:
            hidden = self.initial_hidden(batch_size, device=features.device, dtype=features.dtype)
        if hidden.shape != (batch_size, self.hidden_dim):
            raise ValueError(f"hidden must have shape [B, {self.hidden_dim}]")
        if valid_mask is None:
            valid = torch.ones(batch_size, dtype=torch.bool, device=features.device)
        else:
            valid = valid_mask.to(device=features.device, dtype=torch.bool)
            if valid.shape != (batch_size,):
                raise ValueError("valid_mask must have shape [B]")
        if velocity_baseline is None:
            baseline = torch.zeros((batch_size, 3), dtype=features.dtype, device=features.device)
        else:
            baseline = velocity_baseline.to(device=features.device, dtype=features.dtype)
            if baseline.shape != (batch_size, 3):
                raise ValueError("velocity_baseline must have shape [B, 3]")

        time_constant = functional.softplus(self.log_time_constant).to(dtype=features.dtype) + 1e-3
        decayed_hidden = hidden * torch.exp(-dt.unsqueeze(-1) / time_constant.unsqueeze(0))
        candidate_hidden = self.gru_cell(encoded, decayed_hidden)
        next_hidden = torch.where(valid.unsqueeze(-1), candidate_hidden, hidden)
        head_feature = self.head(next_hidden)
        mean_velocity = baseline + self.velocity_delta_head(head_feature)
        log_sigma = self.log_sigma_head(head_feature).clamp(-5.0, 2.0)
        return mean_velocity, log_sigma, next_hidden, gate

    def forward(
        self,
        features: torch.Tensor,
        source_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        velocity_baseline: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 3 or features.shape[-1] != self.continuous_dim:
            raise ValueError(f"features must have shape [B, T, {self.continuous_dim}]")
        if source_ids.shape != features.shape[:2] or valid_mask.shape != features.shape[:2]:
            raise ValueError("source_ids and valid_mask must have shape [B, T]")
        batch_size, time_steps, _ = features.shape
        if time_steps < 1:
            raise ValueError("At least one time step is required")
        if velocity_baseline is None:
            velocity_baseline = self.compute_velocity_baseline(features, valid_mask)
        if velocity_baseline.shape != (batch_size, time_steps, 3):
            raise ValueError("velocity_baseline must have shape [B, T, 3]")

        hidden = self.initial_hidden(batch_size, device=features.device, dtype=features.dtype)
        means: list[torch.Tensor] = []
        log_sigmas: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for index in range(time_steps):
            mean, log_sigma, hidden, gate = self.step(
                features[:, index],
                source_ids[:, index],
                hidden=hidden,
                valid_mask=valid_mask[:, index],
                velocity_baseline=velocity_baseline[:, index],
            )
            means.append(mean)
            log_sigmas.append(log_sigma)
            gates.append(gate)
        mean_tensor = torch.stack(means, dim=1)
        log_sigma_tensor = torch.stack(log_sigmas, dim=1)
        if return_aux:
            if self.robust_input_enabled:
                _, fd_reliability = robust_finite_difference(
                    features,
                    clip_mps=self.finite_difference_clip_mps,
                )
            else:
                fd_reliability = torch.ones_like(valid_mask, dtype=features.dtype)
            return mean_tensor, log_sigma_tensor, {
                "velocity_baseline": velocity_baseline,
                "reliability_gate": torch.stack(gates, dim=1),
                "finite_difference_reliability": fd_reliability.unsqueeze(-1),
            }
        return mean_tensor, log_sigma_tensor


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

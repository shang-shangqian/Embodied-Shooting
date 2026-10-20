"""Shared loss and metrics for TAQ-GRU training and evaluation."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as functional


def move_taq_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def masked_taq_velocity_loss(
    mean_velocity: torch.Tensor,
    log_sigma: torch.Tensor,
    target_velocity: torch.Tensor,
    target_valid_mask: torch.Tensor,
    *,
    huber_beta: float = 0.1,
    uncertainty_weight: float = 0.10,
    speed_norm_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Window-balanced masked Huber + uncertainty NLL + speed loss.

    Each valid time step is averaged inside its window first, then windows are
    averaged.  This prevents a longer/padded sequence from receiving a larger
    gradient merely because it contains more supervised frames.
    """
    if mean_velocity.shape != log_sigma.shape or mean_velocity.shape != target_velocity.shape:
        raise ValueError("mean_velocity, log_sigma and target_velocity must share shape [B, T, 3]")
    if target_valid_mask.shape != mean_velocity.shape[:2]:
        raise ValueError("target_valid_mask must have shape [B, T]")
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be positive")
    scalar_mask = target_valid_mask.to(dtype=mean_velocity.dtype)
    mask = scalar_mask.unsqueeze(-1)
    valid_steps_per_window = scalar_mask.sum(dim=1)
    window_mask = valid_steps_per_window > 0.0
    valid_windows = window_mask.to(dtype=mean_velocity.dtype).sum().clamp_min(1.0)
    valid_steps = scalar_mask.sum().clamp_min(1.0)

    error = mean_velocity - target_velocity
    huber = functional.smooth_l1_loss(
        mean_velocity, target_velocity, beta=float(huber_beta), reduction="none"
    )
    huber_per_step = huber.mean(dim=-1)
    huber_per_window = (huber_per_step * scalar_mask).sum(dim=1) / valid_steps_per_window.clamp_min(1.0)
    huber_loss = (huber_per_window * window_mask).sum() / valid_windows
    # log_sigma represents log standard deviation: 0.5 * (e/sigma)^2 + log(sigma).
    nll = 0.5 * torch.exp(-2.0 * log_sigma) * error.square() + log_sigma
    nll_per_step = nll.mean(dim=-1)
    nll_per_window = (nll_per_step * scalar_mask).sum(dim=1) / valid_steps_per_window.clamp_min(1.0)
    nll_loss = (nll_per_window * window_mask).sum() / valid_windows
    speed_norm_error = (
        torch.linalg.vector_norm(mean_velocity, dim=-1)
        - torch.linalg.vector_norm(target_velocity, dim=-1)
    ).abs()
    speed_norm_per_window = (speed_norm_error * scalar_mask).sum(dim=1) / valid_steps_per_window.clamp_min(1.0)
    speed_norm_loss = (speed_norm_per_window * window_mask).sum() / valid_windows
    total = huber_loss + float(uncertainty_weight) * nll_loss + float(speed_norm_weight) * speed_norm_loss
    return total, {
        "huber": huber_loss.detach(),
        "nll": nll_loss.detach(),
        "speed_norm": speed_norm_loss.detach(),
        "valid_steps": valid_steps.detach(),
        "valid_windows": valid_windows.detach(),
    }


@torch.no_grad()
def evaluate_taq_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    huber_beta: float = 0.1,
    uncertainty_weight: float = 0.10,
    speed_norm_weight: float = 0.05,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    huber_sum = 0.0
    nll_sum = 0.0
    speed_norm_sum = 0.0
    step_count = 0
    window_count = 0
    absolute_error_sum = torch.zeros(3, dtype=torch.float64, device=device)
    squared_error_sum = torch.zeros(3, dtype=torch.float64, device=device)
    error_norm_values: list[torch.Tensor] = []
    coverage_values: list[tuple[torch.Tensor, torch.Tensor]] = []
    group_error_values: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    group_step_counts: dict[int, int] = {0: 0, 1: 0}

    for raw_batch in loader:
        batch = move_taq_batch(raw_batch, device)
        mean, log_sigma = model(batch["features"], batch["source_ids"], batch["valid_mask"])
        total_loss, parts = masked_taq_velocity_loss(
            mean,
            log_sigma,
            batch["target"],
            batch["target_valid_mask"],
            huber_beta=huber_beta,
            uncertainty_weight=uncertainty_weight,
            speed_norm_weight=speed_norm_weight,
        )
        valid = batch["target_valid_mask"]
        count = int(valid.sum().item())
        if count == 0:
            continue
        error = mean[valid] - batch["target"][valid]
        sigma = torch.exp(log_sigma[valid])
        absolute_error_sum += error.abs().sum(dim=0).to(torch.float64)
        squared_error_sum += error.square().sum(dim=0).to(torch.float64)
        error_norm_values.append(torch.linalg.vector_norm(error, dim=-1).detach().cpu())
        coverage_values.append((error.abs().detach().cpu(), sigma.detach().cpu()))
        scenario_ids = batch.get("scenario_id")
        if scenario_ids is not None:
            for scenario_id in (0, 1):
                group_valid = valid & (scenario_ids[:, None] == scenario_id)
                if group_valid.any():
                    group_error_values[scenario_id].append(
                        torch.linalg.vector_norm(
                            (mean - batch["target"])[group_valid], dim=-1
                        ).detach().cpu()
                    )
                    group_step_counts[scenario_id] += int(group_valid.sum().item())
        valid_window_count = int(parts["valid_windows"].item())
        loss_sum += float(total_loss.item()) * valid_window_count
        huber_sum += float(parts["huber"].item()) * valid_window_count
        nll_sum += float(parts["nll"].item()) * valid_window_count
        speed_norm_sum += float(parts["speed_norm"].item()) * valid_window_count
        step_count += count
        window_count += valid_window_count

    if step_count == 0:
        raise RuntimeError("Evaluation received no target-valid time steps")
    norm = torch.cat(error_norm_values)
    error_abs = torch.cat([item[0] for item in coverage_values])
    sigma_values = torch.cat([item[1] for item in coverage_values])
    mae = absolute_error_sum / step_count
    rmse = torch.sqrt(squared_error_sum / step_count)
    def scalar_metrics(values: torch.Tensor) -> dict[str, float]:
        return {
            "err_norm_mean": float(values.mean().item()),
            "err_norm_p50": float(torch.quantile(values, 0.50).item()),
            "err_norm_p90": float(torch.quantile(values, 0.90).item()),
            "err_norm_p95": float(torch.quantile(values, 0.95).item()),
            "err_norm_p99": float(torch.quantile(values, 0.99).item()),
            "err_norm_max": float(values.max().item()),
            "error_over_0p5": float((values > 0.5).to(torch.float32).mean().item()),
            "error_over_1p0": float((values > 1.0).to(torch.float32).mean().item()),
            "error_over_2p0": float((values > 2.0).to(torch.float32).mean().item()),
        }
    result: dict[str, float] = {
        "loss": loss_sum / max(window_count, 1),
        "huber": huber_sum / max(window_count, 1),
        "nll": nll_sum / max(window_count, 1),
        "speed_norm_mae": speed_norm_sum / max(window_count, 1),
        "mae_x": float(mae[0].item()),
        "mae_y": float(mae[1].item()),
        "mae_z": float(mae[2].item()),
        "rmse_x": float(rmse[0].item()),
        "rmse_y": float(rmse[1].item()),
        "rmse_z": float(rmse[2].item()),
        **scalar_metrics(norm),
        "one_sigma_coverage": float((error_abs <= sigma_values).to(torch.float32).mean().item()),
        "two_sigma_coverage": float((error_abs <= 2.0 * sigma_values).to(torch.float32).mean().item()),
        "three_sigma_coverage": float((error_abs <= 3.0 * sigma_values).to(torch.float32).mean().item()),
        "valid_steps": float(step_count),
        "valid_windows": float(window_count),
    }
    # Coverage is recomputed from the retained per-step errors/sigmas below;
    # keep the existing one-sigma field above for backward-compatible logs.
    for scenario_id, values in group_error_values.items():
        if values:
            result.update({f"{name}_ss" if scenario_id == 0 else f"{name}_sd": value
                           for name, value in scalar_metrics(torch.cat(values)).items()})
            result["valid_steps_ss" if scenario_id == 0 else "valid_steps_sd"] = float(group_step_counts[scenario_id])
    return result


def json_ready_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    """Convert metric mapping to JSON-safe scalar values."""
    result: dict[str, float | int] = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        result[name] = int(value) if isinstance(value, int) else float(value)
    return result

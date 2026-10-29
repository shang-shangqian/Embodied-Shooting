"""Analyze TAQ episode streams for velocity-error source attribution.

This is intentionally an offline tool.  It evaluates the checkpoint in true
chronological streaming mode, resets hidden state at every stream reset, and
never uses audit-only fields as model inputs.  Optional audit CSV/JSONL files
are joined after inference by ``(episode_uid, frame_index)``.

Example::

  python tools/analyze_error_sources.py \
      --data-root outputs/paired_validation \
      --checkpoint ../../speed_prediction/stage1_v2_runs/09-17-17-16/best_macro_p95.pt \
      --out-dir reports/paired_validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - IsaacLab environments provide torch.
    raise SystemExit("This tool requires the IsaacLab/PyTorch Python environment.") from exc


TOOLS_DIR = Path(__file__).resolve().parent
# .../source/standalone/test11/velocity_error_validation/tools -> test11
TEST11_DIR = TOOLS_DIR.parents[1]
if str(TEST11_DIR) not in sys.path:
    sys.path.insert(0, str(TEST11_DIR))

from speed_prediction.time_aware_velocity_data import (  # noqa: E402
    FeatureIndex,
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION,
    TAQ_STREAM_FORMAT,
    TrackerSource,
    load_torch_payload,
    taq_model_config_from_checkpoint,
)
from speed_prediction.time_aware_velocity_model import TimeAwareQualityGatedGRU  # noqa: E402

try:
    from .validation_io import SOURCE_NAMES, finite_float, load_audit_rows, write_csv
except ImportError:  # pragma: no cover
    from validation_io import SOURCE_NAMES, finite_float, load_audit_rows, write_csv


CORE_METRICS = (
    "mean",
    "p95",
    "p99",
    "max",
    "over_0p5",
    "over_1p0",
    "over_2p0",
)

CONDITION_SETS: dict[str, tuple[str, ...]] = {
    "camera_tracking": ("A", "B", "C", "D"),
    "perception_factorial": ("P0", "P1", "P2", "P3"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--shift-max-ms", type=float, default=100.0)
    parser.add_argument("--shift-step-ms", type=float, default=10.0)
    parser.add_argument("--condition-field", default="condition")
    parser.add_argument("--max-event-gap-frames", type=int, default=1)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    return torch.device(name if "cuda" not in name or torch.cuda.is_available() else "cpu")


def _load_model(checkpoint: Path, device: torch.device) -> tuple[TimeAwareQualityGatedGRU, float]:
    payload = load_torch_payload(checkpoint)
    if payload.get("schema_version") != TAQ_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema must be {TAQ_SCHEMA_VERSION!r}")
    if payload.get("label_semantics") != TAQ_LABEL_SEMANTICS:
        raise ValueError("checkpoint has incompatible label semantics")
    model = TimeAwareQualityGatedGRU(**taq_model_config_from_checkpoint(payload)).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    norm = payload.get("input_norm")
    if norm is None:
        raise KeyError("checkpoint is missing input_norm")
    model.set_input_norm_stats(
        torch.as_tensor(norm["mean"], device=device),
        torch.as_tensor(norm["std"], device=device),
    )
    model.eval()
    window_s = float(payload.get("train_args", {}).get("baseline_window_s", model.baseline_window_s))
    return model, window_s


def _stream_arrays(payload: dict[str, Any], path: Path) -> tuple[torch.Tensor, ...]:
    if payload.get("schema_version") != TAQ_SCHEMA_VERSION or payload.get("stream_format") != TAQ_STREAM_FORMAT:
        raise ValueError(f"not a TAQ v2 stream: {path}")
    if payload.get("label_semantics") != TAQ_LABEL_SEMANTICS:
        raise ValueError(f"incompatible label semantics: {path}")
    features = torch.as_tensor(payload["sequence_features"], dtype=torch.float32)
    sources = torch.as_tensor(payload["sequence_source_ids"], dtype=torch.long)
    valid = torch.as_tensor(payload["sequence_valid_mask"], dtype=torch.bool)
    target = torch.as_tensor(payload["target_velocity_base"], dtype=torch.float32)
    target_valid = torch.as_tensor(payload["target_valid_mask"], dtype=torch.bool)
    resets = torch.as_tensor(payload["state_reset_mask"], dtype=torch.bool)
    timestamps = torch.as_tensor(payload["timestamps_s"], dtype=torch.float64)
    steps = int(features.shape[0]) if features.ndim == 2 else 0
    if features.shape != (steps, 24) or sources.shape != (steps,) or valid.shape != (steps,):
        raise ValueError(f"invalid feature shapes: {path}")
    if target.shape != (steps, 3) or target_valid.shape != (steps,) or resets.shape != (steps,):
        raise ValueError(f"invalid target/reset shapes: {path}")
    if timestamps.shape != (steps,) or steps == 0 or not bool(resets[0]):
        raise ValueError(f"stream must be non-empty and start with reset: {path}")
    if steps > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError(f"stream timestamps are not strictly increasing: {path}")
    if torch.any(target_valid & ~valid):
        raise ValueError(f"target_valid_mask includes invalid frames: {path}")
    if torch.any((sources < int(TrackerSource.INITIAL)) | (sources > int(TrackerSource.INVALID))):
        raise ValueError(f"stream contains invalid tracker source IDs: {path}")
    if not torch.isfinite(features).all() or not torch.isfinite(target).all():
        raise ValueError(f"stream contains non-finite features or targets: {path}")
    return features, sources, valid, target, target_valid, resets, timestamps


def _first_measurement_timestamp(
    features: torch.Tensor,
    sources: torch.Tensor,
    sequence_valid: torch.Tensor,
    timestamps: torch.Tensor,
) -> float | None:
    """Locate the first direct observation with a usable range."""
    measured = (
        sequence_valid
        & (sources <= int(TrackerSource.OSTRACK))
        & (features[:, FeatureIndex.OBSERVED_MASK] > 0.5)
        & (features[:, FeatureIndex.RANGE_VALID_MASK] > 0.5)
    )
    indices = torch.nonzero(measured, as_tuple=False).flatten()
    return None if len(indices) == 0 else float(timestamps[int(indices[0])].item())


def _evaluation_started_mask(
    features: torch.Tensor,
    sources: torch.Tensor,
    sequence_valid: torch.Tensor,
    timestamps: torch.Tensor,
    *,
    initialization_start_s: float | None = None,
) -> torch.Tensor:
    """Keep one shared scoring start for paired conditions, including later losses."""
    start = initialization_start_s
    if start is None:
        start = _first_measurement_timestamp(features, sources, sequence_valid, timestamps)
    return torch.zeros_like(sequence_valid) if start is None else timestamps >= start


def _paired_initialization_starts(streams: list[Path]) -> dict[Path, float | None]:
    """Use the earliest measurement in each trajectory pair as its scoring start."""
    group_by_stream: dict[Path, str] = {}
    start_by_group: dict[str, float] = {}
    for stream in streams:
        payload = load_torch_payload(stream)
        features, sources, valid, _, _, _, timestamps = _stream_arrays(payload, stream)
        trajectory_uid = str(dict(payload.get("metadata", {})).get("trajectory_uid", ""))
        group = trajectory_uid or str(stream.resolve())
        group_by_stream[stream] = group
        first = _first_measurement_timestamp(features, sources, valid, timestamps)
        if first is not None:
            start_by_group[group] = min(first, start_by_group.get(group, first))
    return {stream: start_by_group.get(group) for stream, group in group_by_stream.items()}


def _segmented_baseline(features: torch.Tensor, valid: torch.Tensor, resets: torch.Tensor, model: TimeAwareQualityGatedGRU, window_s: float) -> torch.Tensor:
    indices = torch.nonzero(resets, as_tuple=False).flatten().tolist()
    if not indices or indices[0] != 0:
        raise ValueError("reset mask must start with a reset")
    baseline = torch.zeros_like(features[:, :3])
    for start, end in zip(indices, [*indices[1:], len(features)]):
        if end <= start:
            continue
        baseline[start:end] = model.compute_velocity_baseline(
            features[start:end].unsqueeze(0),
            valid[start:end].unsqueeze(0),
            window_s=window_s,
        )[0]
    return baseline


def _derivative(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Return the norm of the first derivative as one scalar per frame."""
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < 2:
        return result
    dt = np.diff(timestamps)
    good = dt > 1e-8
    derivative = np.linalg.norm(np.diff(values, axis=0), axis=1) / np.maximum(dt, 1e-8)
    result[1:][good] = derivative[good]
    return result


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)


def _reset_segments(reset_mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(reset_mask).astype(int).tolist()
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return list(zip(starts, [*starts[1:], len(reset_mask)]))


def _json_number(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def infer_stream(path: Path, model: TimeAwareQualityGatedGRU, window_s: float, device: torch.device, audit: dict[tuple[str, int], dict[str, Any]], condition_field: str = "condition", initialization_start_s: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_torch_payload(path)
    features, sources, valid, target, target_valid, resets, timestamps = _stream_arrays(payload, path)
    metadata = dict(payload.get("metadata", {}))
    uid = str(metadata.get("episode_uid", path.stem))
    condition = str(metadata.get(condition_field, metadata.get("condition_name", "unknown")))
    pair_id = str(metadata.get("pair_id", ""))
    trajectory_uid = str(metadata.get("trajectory_uid", metadata.get("trajectory_id", "")))
    stream_uid = f"{uid}|condition={condition}|path={path.resolve()}"
    features_d = features.to(device)
    sources_d = sources.to(device)
    valid_d = valid.to(device)
    baseline = _segmented_baseline(features_d, valid_d, resets.to(device), model, window_s)
    predictions: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    gates: list[float] = []
    fd_reliability: list[float] = []
    with torch.no_grad():
        hidden = None
        for index in range(len(features)):
            if bool(resets[index]):
                hidden = None
            mean, log_sigma, hidden, gate = model.step(
                features_d[index:index + 1], sources_d[index:index + 1], hidden=hidden,
                valid_mask=valid_d[index:index + 1], velocity_baseline=baseline[index:index + 1],
            )
            predictions.append(mean[0].detach().cpu().numpy())
            sigmas.append(torch.exp(log_sigma[0]).detach().cpu().numpy())
            gates.append(float(gate[0, 0].item()))
            fd_reliability.append(float(model._prepare_input(features_d[index:index + 1])[1][0].item()))
    pred = np.asarray(predictions, dtype=np.float64)
    sigma = np.asarray(sigmas, dtype=np.float64)
    truth = target.numpy().astype(np.float64)
    ts = timestamps.numpy().astype(np.float64)
    pos = features[:, :3].numpy().astype(np.float64)
    robot_ang = features[:, 9:12].numpy().astype(np.float64)
    yaw_omega = np.zeros_like(robot_ang)
    yaw_omega[:, 2] = robot_ang[:, 2]
    errors = pred - truth
    baseline_np = baseline.detach().cpu().numpy().astype(np.float64)
    baseline_errors = baseline_np - truth
    supervision_valid_rows = target_valid.numpy().astype(bool) & valid.numpy().astype(bool)
    evaluation_started = _evaluation_started_mask(
        features, sources, valid, timestamps,
        initialization_start_s=initialization_start_s,
    ).numpy()
    valid_rows = supervision_valid_rows & evaluation_started
    reset_rows = resets.numpy().astype(bool)
    transition_valid = valid.numpy().astype(bool)
    transition_valid[0] = False
    transition_valid[1:] &= valid.numpy().astype(bool)[:-1] & ~reset_rows[1:]
    pred_jump = np.full(len(pred), np.nan)
    truth_jump = np.full(len(pred), np.nan)
    if len(pred) > 1:
        pred_jump[1:] = _norm(np.diff(pred, axis=0))
        truth_jump[1:] = _norm(np.diff(truth, axis=0))
        pred_jump[~transition_valid] = np.nan
        truth_jump[~transition_valid] = np.nan
    pos_error = np.full(len(pred), np.nan)
    kinematic = np.full(len(pred), np.nan)
    gt_position_fd_velocity = np.full_like(pred, np.nan)
    if any("position_gt_base_m" in audit.get((uid, index), {}) for index in range(len(pred))):
        gt_position_rows = []
        for index in range(len(pred)):
            value = audit.get((uid, index), {}).get("position_gt_base_m")
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                value = [np.nan] * 3
            gt_position_rows.append(value)
        gt_positions = np.asarray(gt_position_rows, dtype=np.float64)
        if gt_positions.shape == pos.shape:
            pos_error = _norm(pos - gt_positions)
            if len(gt_positions) > 1:
                dt = np.diff(ts)
                good = (dt > 1e-8) & transition_valid[1:]
                coordinate_derivative = np.diff(gt_positions, axis=0) / np.maximum(dt[:, None], 1e-8)
                physical_velocity = coordinate_derivative + np.cross(
                    yaw_omega[:-1], gt_positions[:-1]
                )
                gt_position_fd_velocity[1:][good] = physical_velocity[good]
    if len(pos) > 1:
        dt = np.diff(ts)
        expected = pos[:-1] + (truth[:-1] - np.cross(yaw_omega[:-1], pos[:-1])) * dt[:, None]
        good = (dt > 1e-8) & transition_valid[1:]
        kinematic[1:][good] = _norm(pos[1:][good] - expected[good]) / dt[good]
    linear_acceleration = _derivative(features[:, 6:9].numpy().astype(np.float64), ts)
    angular_acceleration = _derivative(robot_ang, ts)
    target_acceleration = _derivative(truth, ts)
    linear_acceleration[~transition_valid] = np.nan
    angular_acceleration[~transition_valid] = np.nan
    target_acceleration[~transition_valid] = np.nan
    extrapolation: dict[str, np.ndarray] = {}
    if "gt_positions" in locals() and gt_positions.shape == pos.shape:
        for horizon in (0.1, 0.2):
            values = np.full(len(ts), np.nan, dtype=np.float64)
            for start, end in _reset_segments(reset_rows):
                segment_ts = ts[start:end]
                segment_gt = gt_positions[start:end]
                if len(segment_ts) < 2 or not np.isfinite(segment_gt).all():
                    continue
                query = segment_ts + horizon
                future_gt = np.column_stack([
                    np.interp(query, segment_ts, segment_gt[:, dim], left=np.nan, right=np.nan)
                    for dim in range(3)
                ])
                predicted_position = pos[start:end] + (
                    pred[start:end] - np.cross(yaw_omega[start:end], pos[start:end])
                ) * horizon
                values[start:end] = _norm(predicted_position - future_gt)
            extrapolation[f"position_extrapolation_error_{int(horizon * 1000)}ms"] = values
    rows: list[dict[str, Any]] = []
    for index in range(len(pred)):
        extra = dict(audit.get((uid, index), {}))
        row: dict[str, Any] = {
            "episode_uid": uid, "stream_uid": stream_uid,
            "pair_id": pair_id, "trajectory_uid": trajectory_uid,
            "env_id": metadata.get("env_id", ""),
            "episode_id": metadata.get("episode_id", ""),
            "frame_index": index, "timestamp_s": ts[index],
            "condition": condition, "scenario": metadata.get("scenario", "unknown"),
            "target_id": metadata.get("target_id", ""),
            "valid": int(valid_rows[index]),
            "supervision_valid": int(supervision_valid_rows[index]),
            "evaluation_started": int(evaluation_started[index]),
            "initialization_frame": int(not evaluation_started[index]),
            "state_reset": int(resets[index]),
            "time_since_reset_s": ts[index] - ts[max(i for i in range(index + 1) if bool(resets[i]))],
            "source_id": int(sources[index]), "tracker_source": SOURCE_NAMES.get(int(sources[index]), "unknown"),
            "position_observed_x": pos[index, 0], "position_observed_y": pos[index, 1], "position_observed_z": pos[index, 2],
            "robot_linear_speed_mps": float(np.linalg.norm(features[index, 6:9].numpy())),
            "robot_angular_speed_rps": float(np.linalg.norm(robot_ang[index])),
            "robot_linear_acceleration_mps2": linear_acceleration[index],
            "robot_angular_acceleration_rps2": angular_acceleration[index],
            "target_acceleration_mps2": target_acceleration[index],
            "target_speed_mps": float(np.linalg.norm(truth[index])),
            "velocity_gt_x": truth[index, 0], "velocity_gt_y": truth[index, 1], "velocity_gt_z": truth[index, 2],
            "velocity_prediction_x": pred[index, 0], "velocity_prediction_y": pred[index, 1], "velocity_prediction_z": pred[index, 2],
            "velocity_baseline_x": baseline_np[index, 0], "velocity_baseline_y": baseline_np[index, 1], "velocity_baseline_z": baseline_np[index, 2],
            "sigma_x": sigma[index, 0], "sigma_y": sigma[index, 1], "sigma_z": sigma[index, 2],
            "reliability_gate": gates[index], "finite_difference_reliability": fd_reliability[index],
            "error_x": errors[index, 0], "error_y": errors[index, 1], "error_z": errors[index, 2],
            "error_norm": float(np.linalg.norm(errors[index])),
            "baseline_error_norm": float(np.linalg.norm(baseline_errors[index])),
            "zero_baseline_error_norm": float(np.linalg.norm(truth[index])),
            "gt_position_fd_velocity_x": gt_position_fd_velocity[index, 0],
            "gt_position_fd_velocity_y": gt_position_fd_velocity[index, 1],
            "gt_position_fd_velocity_z": gt_position_fd_velocity[index, 2],
            "gt_position_fd_oracle_error_norm": (
                float(np.linalg.norm(gt_position_fd_velocity[index] - truth[index]))
                if np.isfinite(gt_position_fd_velocity[index]).all() else np.nan
            ),
            "prediction_jump": pred_jump[index],
            "prediction_acceleration_mps2": (
                pred_jump[index] / max(float(features[index, 12].item()), 1e-8)
                if index > 0 else np.nan
            ),
            "gt_jump": truth_jump[index], "position_error": pos_error[index], "kinematic_residual": kinematic[index],
            "episode_age_s": ts[index] - ts[0],
        }
        for key, values in extrapolation.items():
            row[key] = values[index]
        for key, value in extra.items():
            if key in {"episode_uid", "frame_index"}:
                continue
            # Audit data is joined only after inference and must never replace
            # a value computed from the model stream.  Preserve colliding
            # fields under an explicit namespace for later consistency checks.
            row[f"audit_{key}" if key in row else key] = value
        measured_range = finite_float(row.get("range_measured_m"))
        gt_range = finite_float(row.get("range_gt_m"))
        row["range_absolute_error_m"] = (
            abs(measured_range - gt_range)
            if math.isfinite(measured_range) and math.isfinite(gt_range) else np.nan
        )
        row["audit_joined"] = int(bool(extra))
        rows.append(row)
    return rows, {"episode_uid": uid, "stream_uid": stream_uid, "condition": condition, "metadata": metadata, "rows": rows}


def _valid_values(rows: Iterable[dict[str, Any]], field: str = "error_norm") -> np.ndarray:
    values = np.asarray([finite_float(row.get(field)) for row in rows], dtype=np.float64)
    return values[np.isfinite(values) & np.asarray([bool(row.get("valid", 0)) for row in rows])]


def metric_values(rows: Iterable[dict[str, Any]], field: str = "error_norm") -> dict[str, float | int]:
    rows = list(rows)
    values = _valid_values(rows, field)
    if not len(values):
        return {"valid_frames": 0, **{name: None for name in CORE_METRICS}, "p50": None, "p90": None, "mae_x": None, "mae_y": None, "mae_z": None, "rmse_x": None, "rmse_y": None, "rmse_z": None, "prediction_jump_p95": None, "prediction_jump_p99": None, "prediction_acceleration_p95": None, "prediction_acceleration_p99": None, "extrapolation_100ms_p95": None, "extrapolation_200ms_p95": None, "one_sigma_coverage": None, "two_sigma_coverage": None, "three_sigma_coverage": None}
    result: dict[str, float | int | None] = {
        "valid_frames": int(len(values)), "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, .50)), "p90": float(np.quantile(values, .90)),
        "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)),
        "max": float(np.max(values)), "over_0p5": float(np.mean(values > .5)),
        "over_1p0": float(np.mean(values > 1.0)), "over_2p0": float(np.mean(values > 2.0)),
    }
    valid_rows = [row for row in rows if bool(row.get("valid", 0)) and math.isfinite(finite_float(row.get(field)))]
    for axis in "xyz":
        component = np.asarray([finite_float(row.get(f"error_{axis}")) for row in valid_rows], dtype=np.float64)
        component = component[np.isfinite(component)]
        result[f"mae_{axis}"] = None if not len(component) else float(np.mean(np.abs(component)))
        result[f"rmse_{axis}"] = None if not len(component) else float(np.sqrt(np.mean(component ** 2)))
    jumps = np.asarray([finite_float(row.get("prediction_jump")) for row in valid_rows], dtype=np.float64)
    jumps = jumps[np.isfinite(jumps)]
    result["prediction_jump_p95"] = None if not len(jumps) else float(np.quantile(jumps, .95))
    result["prediction_jump_p99"] = None if not len(jumps) else float(np.quantile(jumps, .99))
    accelerations = np.asarray(
        [finite_float(row.get("prediction_acceleration_mps2")) for row in valid_rows],
        dtype=np.float64,
    )
    accelerations = accelerations[np.isfinite(accelerations)]
    result["prediction_acceleration_p95"] = (
        None if not len(accelerations) else float(np.quantile(accelerations, .95))
    )
    result["prediction_acceleration_p99"] = (
        None if not len(accelerations) else float(np.quantile(accelerations, .99))
    )
    for horizon in (100, 200):
        extrap = np.asarray([finite_float(row.get(f"position_extrapolation_error_{horizon}ms")) for row in valid_rows], dtype=np.float64)
        extrap = extrap[np.isfinite(extrap)]
        result[f"extrapolation_{horizon}ms_p95"] = None if not len(extrap) else float(np.quantile(extrap, .95))
    component_error = np.asarray([[finite_float(row.get(f"error_{axis}")) for axis in "xyz"] for row in valid_rows], dtype=np.float64)
    component_sigma = np.asarray([[finite_float(row.get(f"sigma_{axis}")) for axis in "xyz"] for row in valid_rows], dtype=np.float64)
    for multiplier, name in ((1, "one_sigma_coverage"), (2, "two_sigma_coverage"), (3, "three_sigma_coverage")):
        good = np.isfinite(component_error) & np.isfinite(component_sigma)
        result[name] = None if not np.any(good) else float(np.mean(np.abs(component_error[good]) <= multiplier * component_sigma[good]))
    return result


def scalar_metric_values(rows: Iterable[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = _valid_values(rows, field)
    if not len(values):
        return {"valid_frames": 0, **{name: None for name in CORE_METRICS}, "p50": None, "p90": None}
    return {
        "valid_frames": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, .50)),
        "p90": float(np.quantile(values, .90)),
        "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)),
        "max": float(np.max(values)),
        "over_0p5": float(np.mean(values > .5)),
        "over_1p0": float(np.mean(values > 1.0)),
        "over_2p0": float(np.mean(values > 2.0)),
    }


def _bootstrap(episode_rows: list[list[dict[str, Any]]], metric: str, repeats: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    """Bootstrap the macro-average of an episode-level metric."""
    values = []
    for rows in episode_rows:
        summary = metric_values(rows)
        value = summary.get(metric)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None, None
    sample = np.asarray(values, dtype=np.float64)
    if len(sample) == 1 or repeats <= 0:
        return float(sample[0]), float(sample[0])
    draws = rng.choice(sample, size=(repeats, len(sample)), replace=True).mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def grouped_metrics(rows: list[dict[str, Any]], bootstrap: int, seed: int) -> list[dict[str, Any]]:
    for row in rows:
        measured = finite_float(row.get("range_measured_m"))
        ground_truth = finite_float(row.get("range_gt_m"))
        row["range_absolute_error_m"] = (
            abs(measured - ground_truth)
            if math.isfinite(measured) and math.isfinite(ground_truth) else float("nan")
        )

    def quantile_edges(field: str) -> np.ndarray | None:
        values = np.asarray(
            [finite_float(row.get(field)) for row in rows if bool(row.get("valid", 0))],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if len(values) < 4 or float(np.max(values) - np.min(values)) <= 1e-12:
            return None
        return np.quantile(values, [0.0, 0.25, 0.50, 0.75, 1.0])

    def quantile_bin(value: Any, edges: np.ndarray | None) -> str:
        number = finite_float(value)
        if edges is None or not math.isfinite(number):
            return "unknown"
        index = int(np.searchsorted(edges[1:-1], number, side="right"))
        return f"Q{index + 1}"

    quantile_fields = (
        "target_acceleration_mps2",
        "direction_angular_error_deg",
        "bbox_iou_gt",
        "range_absolute_error_m",
        "robot_angular_speed_rps",
        "robot_angular_acceleration_rps2",
        "background_global_flow_px",
        "kinematic_residual",
    )
    edges_by_field = {field: quantile_edges(field) for field in quantile_fields}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        distance = finite_float(row.get("range_measured_m"))
        if not math.isfinite(distance):
            distance = finite_float(row.get("range_gt_m"))
        distance_bin = "unknown" if not math.isfinite(distance) else ("80-100" if distance < 100 else "100-120" if distance < 120 else "120-150" if distance < 150 else "150+")
        target_speed = finite_float(row.get("target_speed_mps"))
        target_speed_bin = (
            "unknown" if not math.isfinite(target_speed) else
            "0-2" if target_speed < 2.0 else
            "2-3.5" if target_speed < 3.5 else "3.5+"
        )
        age = finite_float(row.get("episode_age_s"))
        stage = "startup" if age < .5 else "stable"
        group_values = (
            ("all", "all"), ("condition", row.get("condition", "unknown")),
            ("scenario", row.get("scenario", "unknown")),
            ("tracker_source", row.get("tracker_source", "unknown")),
            ("distance_bin_m", distance_bin), ("target_speed_bin_mps", target_speed_bin),
            ("startup_stage", stage),
        )
        for field in quantile_fields:
            group_values += ((f"{field}_quantile", quantile_bin(row.get(field), edges_by_field[field])),)
        for audit_field in ("state_reset", "fine_action", "measurement_discontinuity", "tracker_reset", "degradation_family"):
            if row.get(audit_field) not in (None, ""):
                group_values += ((audit_field, row[audit_field]),)
        for name, value in group_values:
            groups[(name, str(value))].append(row)
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for (group_name, group_value), group_rows in sorted(groups.items()):
        summary = metric_values(group_rows)
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_episode[str(row.get("stream_uid", row["episode_uid"]))].append(row)
        result = {"group": group_name, "value": group_value, "episodes": len(by_episode), **summary}
        for baseline_name, baseline_field in (
            ("weighted_baseline", "baseline_error_norm"),
            ("zero_baseline", "zero_baseline_error_norm"),
            ("gt_position_fd_oracle", "gt_position_fd_oracle_error_norm"),
        ):
            baseline = scalar_metric_values(group_rows, baseline_field)
            for metric in ("mean", "p95", "p99"):
                result[f"{baseline_name}_{metric}"] = baseline[metric]
        for metric in ("mean", "p95", "p99", "over_0p5", "over_1p0", "over_2p0"):
            episode_values = [
                metric_values(episode_rows).get(metric)
                for episode_rows in by_episode.values()
            ]
            episode_values = [
                float(value) for value in episode_values
                if value is not None and math.isfinite(float(value))
            ]
            low, high = _bootstrap(list(by_episode.values()), metric, bootstrap, rng)
            result[f"episode_macro_{metric}"] = (
                None if not episode_values else float(np.mean(episode_values))
            )
            result[f"episode_macro_{metric}_ci_low"] = low
            result[f"episode_macro_{metric}_ci_high"] = high
        output.append(result)
    return output


def _pair_key(row: dict[str, Any]) -> str | None:
    trajectory = row.get("trajectory_uid")
    if trajectory not in (None, ""):
        return str(trajectory)
    for key in ("episode_pair_id", "paired_episode_uid"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    pair_id = row.get("pair_id")
    env_id = row.get("env_id")
    episode_id = row.get("episode_id")
    if pair_id not in (None, "") and env_id not in (None, "") and episode_id not in (None, ""):
        return f"{pair_id}:env={env_id}:episode={episode_id}"
    return None


def paired_effects(rows: list[dict[str, Any]], bootstrap: int, seed: int) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pair_key = _pair_key(row)
        if pair_key is not None:
            by_pair[pair_key][str(row.get("condition", "unknown"))].append(row)
    for pair, conditions in by_pair.items():
        for condition, condition_rows in conditions.items():
            streams = {str(row.get("stream_uid", "")) for row in condition_rows}
            if len(streams) != 1:
                raise ValueError(
                    f"pair {pair!r} condition {condition!r} maps to {len(streams)} streams; "
                    "pair_id must identify one episode trajectory, not a whole collection seed"
                )
    # Positive differences mean that replacing the left-hand condition with
    # the right-hand oracle/controlled condition reduces the error.
    formulas = (
        ("tracking_fixed", "B", "A"),
        ("camera_motion_with_oracle_tracking", "C", "A"),
        ("tracking_moving", "D", "C"),
        ("camera_motion_with_real_tracking", "D", "B"),
        ("camera_tracking_interaction", "D-C_minus_B-A", ""),
        ("direction_path", "P0", "P1"),
        ("range_path", "P0", "P2"),
        ("all_perception", "P0", "P3"),
        ("direction_range_interaction", "P0-P1_minus_P2-P3", ""),
    )
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    pair_summaries = {
        pair: {condition: metric_values(condition_rows) for condition, condition_rows in conditions.items()}
        for pair, conditions in by_pair.items()
    }
    for metric in CORE_METRICS:
        for name, left, right in formulas:
            diffs: list[float] = []
            for conditions in pair_summaries.values():
                required_conditions = (
                    {"P0", "P1", "P2", "P3"}
                    if name in {
                        "direction_path", "range_path", "all_perception",
                        "direction_range_interaction",
                    }
                    else {"A", "B", "C", "D"}
                )
                if not required_conditions.issubset(conditions):
                    continue
                if name == "camera_tracking_interaction":
                    values = [conditions[key].get(metric) for key in ("A", "B", "C", "D")]
                    if any(value is None for value in values):
                        continue
                    a, b, c, d = (float(value) for value in values)
                    diffs.append((d - c) - (b - a))
                elif name == "direction_range_interaction":
                    values = [conditions[key].get(metric) for key in ("P0", "P1", "P2", "P3")]
                    if any(value is None for value in values):
                        continue
                    p0, p1, p2, p3 = (float(value) for value in values)
                    diffs.append((p0 - p1) - (p2 - p3))
                elif left in conditions and right in conditions:
                    left_value = conditions[left].get(metric)
                    right_value = conditions[right].get(metric)
                    if left_value is not None and right_value is not None:
                        diffs.append(float(left_value) - float(right_value))
            if not diffs:
                continue
            values = np.asarray(diffs, dtype=np.float64)
            draws = (
                rng.choice(values, size=(bootstrap, len(values)), replace=True).mean(axis=1)
                if bootstrap > 0 and len(values) > 1 else values
            )
            output.append({
                "effect": name, "metric": metric, "pairs": len(values),
                "mean_difference": float(values.mean()),
                "ci_low": float(np.quantile(draws, .025)),
                "ci_high": float(np.quantile(draws, .975)),
            })
    return output


def validate_pair_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = {str(row.get("condition", "unknown")) for row in rows}
    active_sets = {
        name: set(conditions)
        for name, conditions in CONDITION_SETS.items()
        if present & set(conditions)
    }
    if not active_sets:
        return {"required": False, "pairs": 0, "complete_pairs": 0, "conditions": sorted(present)}
    partial_sets = {
        name: sorted(expected - present)
        for name, expected in active_sets.items()
        if not expected.issubset(present)
    }
    if partial_sets:
        raise ValueError(f"partially present validation condition sets: {partial_sets}")
    expected_union = set().union(*active_sets.values())
    unknown_key_streams = {
        str(row.get("stream_uid", "")) for row in rows
        if str(row.get("condition", "unknown")) in expected_union and _pair_key(row) is None
    }
    if unknown_key_streams:
        raise ValueError(
            f"{len(unknown_key_streams)} validation streams lack trajectory_uid or "
            "pair_id+env_id+episode_id metadata"
        )
    set_summaries: dict[str, Any] = {}
    total_pairs = 0
    total_complete_pairs = 0
    for set_name, expected in active_sets.items():
        by_pair: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            condition = str(row.get("condition", "unknown"))
            pair = _pair_key(row)
            if condition in expected and pair is not None:
                by_pair[pair].add(condition)
        incomplete = {
            pair: sorted(expected - values)
            for pair, values in by_pair.items()
            if values != expected
        }
        complete_pairs = len(by_pair) - len(incomplete)
        if complete_pairs == 0:
            raise ValueError(f"no complete {set_name} episode pairs are available")
        set_summaries[set_name] = {
            "pairs": len(by_pair),
            "complete_pairs": complete_pairs,
            "excluded_incomplete_pairs": len(incomplete),
            "incomplete_preview": dict(list(incomplete.items())[:10]),
            "conditions": sorted(expected),
        }
        total_pairs += len(by_pair)
        total_complete_pairs += complete_pairs
    return {
        "required": True,
        "pairs": total_pairs,
        "complete_pairs": total_complete_pairs,
        "excluded_incomplete_pairs": total_pairs - total_complete_pairs,
        "status": "checked" if total_pairs == total_complete_pairs else "checked_with_exclusions",
        "conditions": sorted(present),
        "condition_sets": set_summaries,
    }


def validate_trajectory_consistency(
    rows: list[dict[str, Any]],
    *,
    position_tolerance_m: float = 1e-4,
) -> dict[str, Any]:
    """Check that paired conditions observed the same simulated target path.

    Tracker loss can remove different frames in different conditions, so the
    comparison uses common timestamps rather than frame indices.  Old streams
    without ``target_position_world_m`` remain analyzable, but are explicitly
    reported as unchecked instead of being called paired.
    """
    validation_conditions = set().union(*(set(values) for values in CONDITION_SETS.values()))
    pair_conditions_seen: dict[str, set[str]] = defaultdict(set)
    by_pair_condition: dict[str, dict[str, dict[int, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        pair = _pair_key(row)
        condition = str(row.get("condition", "unknown"))
        if pair is not None and condition in validation_conditions:
            pair_conditions_seen[pair].add(condition)
        position = row.get("target_position_world_m")
        timestamp = finite_float(row.get("timestamp_s"))
        if (
            pair is None
            or condition not in validation_conditions
            or not isinstance(position, (list, tuple))
            or len(position) != 3
            or not math.isfinite(timestamp)
        ):
            continue
        position_array = np.asarray(position, dtype=np.float64)
        if np.isfinite(position_array).all():
            by_pair_condition[pair][condition][int(round(timestamp * 1_000_000.0))] = position_array

    checked_pairs = 0
    skipped_pairs = 0
    common_samples = 0
    maximum_difference = 0.0
    for pair, seen_conditions in pair_conditions_seen.items():
        applicable = [
            set(condition_set)
            for condition_set in CONDITION_SETS.values()
            if set(condition_set).issubset(seen_conditions)
        ]
        for expected in applicable:
            conditions = by_pair_condition.get(pair, {})
            if any(not conditions.get(name) for name in expected):
                skipped_pairs += 1
                continue
            timestamp_sets = [set(conditions[name]) for name in expected]
            common = set.intersection(*timestamp_sets) if timestamp_sets else set()
            if not common:
                skipped_pairs += 1
                continue
            reference_name = sorted(expected)[0]
            for timestamp_key in common:
                reference = conditions[reference_name][timestamp_key]
                for name in expected - {reference_name}:
                    difference = float(np.linalg.norm(conditions[name][timestamp_key] - reference))
                    maximum_difference = max(maximum_difference, difference)
                    if difference > position_tolerance_m:
                        raise ValueError(
                            "paired conditions do not share the same target trajectory: "
                            f"difference={difference:.6g} m exceeds {position_tolerance_m:.6g} m"
                        )
            checked_pairs += 1
            common_samples += len(common)
    return {
        "checked_pairs": checked_pairs,
        "skipped_pairs_without_common_audit_samples": skipped_pairs,
        "common_timestamp_samples": common_samples,
        "maximum_target_position_difference_m": maximum_difference if checked_pairs else None,
        "position_tolerance_m": position_tolerance_m,
        "status": (
            "checked"
            if checked_pairs and not skipped_pairs
            else "partially_checked"
            if checked_pairs
            else "not_checked_missing_target_position_audit"
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    model, window_s = _load_model(args.checkpoint, device)
    audit_root = args.audit_root if args.audit_root is not None else args.data_root
    audits = load_audit_rows(audit_root)
    streams = sorted(args.data_root.rglob("*.taqstream"))
    if not streams:
        raise FileNotFoundError(f"no .taqstream files below {args.data_root}")
    initialization_starts = _paired_initialization_starts(streams)
    all_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    for stream in streams:
        rows, episode = infer_stream(
            stream, model, window_s, device, audits, args.condition_field,
            initialization_start_s=initialization_starts[stream],
        )
        all_rows.extend(rows)
        summary = {"episode_uid": episode["episode_uid"], "condition": episode["condition"], "source": str(stream), **metric_values(rows)}
        for baseline_name, baseline_field in (
            ("weighted_baseline", "baseline_error_norm"),
            ("zero_baseline", "zero_baseline_error_norm"),
            ("gt_position_fd_oracle", "gt_position_fd_oracle_error_norm"),
        ):
            baseline = scalar_metric_values(rows, baseline_field)
            summary[f"{baseline_name}_mean"] = baseline["mean"]
            summary[f"{baseline_name}_p95"] = baseline["p95"]
            summary[f"{baseline_name}_p99"] = baseline["p99"]
        episode_summaries.append(summary)
    write_csv(args.out_dir / "frame_metrics.csv", all_rows)
    write_csv(args.out_dir / "episode_metrics.csv", episode_summaries)
    grouped = grouped_metrics(all_rows, args.bootstrap, args.seed)
    write_csv(args.out_dir / "grouped_metrics.csv", grouped)
    pair_validation = validate_pair_coverage(all_rows)
    trajectory_validation = validate_trajectory_consistency(all_rows)
    effects = paired_effects(all_rows, args.bootstrap, args.seed + 1)
    write_csv(args.out_dir / "paired_effects.csv", effects)
    events = extract_events(all_rows, args.max_event_gap_frames)
    write_csv(args.out_dir / "event_catalog.csv", events)
    correlations = lagged_correlations(all_rows)
    write_csv(args.out_dir / "lagged_correlations.csv", correlations)
    shifts = scan_time_offsets(all_rows, args.shift_max_ms, args.shift_step_ms)
    write_csv(args.out_dir / "time_offset_scan.csv", shifts)
    summary = {
        "schema": "velocity_error_validation_v1", "data_root": str(args.data_root),
        "checkpoint": str(args.checkpoint), "device": str(device), "streams": len(streams),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "episodes": len(episode_summaries), "valid_frames": int(sum(item["valid_frames"] for item in episode_summaries)),
        "audit_rows_loaded": len(audits),
        "audit_frames_joined": int(sum(int(row.get("audit_joined", 0)) for row in all_rows)),
        "pair_validation": pair_validation,
        "trajectory_validation": trajectory_validation,
        "model_inputs": "sequence_features + sequence_source_ids + valid_mask only",
        "initialization_frames_excluded": int(sum(
            bool(row["initialization_frame"]) and bool(row["supervision_valid"])
            for row in all_rows
        )),
        "metrics": metric_values(all_rows), "event_count": len(events),
        "baseline_metrics": {
            "weighted_physics": scalar_metric_values(all_rows, "baseline_error_norm"),
            "zero_velocity": scalar_metric_values(all_rows, "zero_baseline_error_norm"),
            "gt_position_finite_difference_oracle": scalar_metric_values(
                all_rows, "gt_position_fd_oracle_error_norm"
            ),
        },
        "lagged_correlation_rows": len(correlations),
        "recognized_pair_condition_sets": CONDITION_SETS,
    }
    _write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def extract_events(rows: list[dict[str, Any]], max_gap_frames: int = 1) -> list[dict[str, Any]]:
    def event_record(
        event_type: str,
        threshold: float | None,
        active: list[dict[str, Any]],
        episode_rows: list[dict[str, Any]],
        *,
        trigger_field: str = "error_norm",
        trigger_threshold: float | None = None,
    ) -> dict[str, Any]:
        peak = max(active, key=lambda item: finite_float(item.get("error_norm")))
        peak_index = episode_rows.index(peak)
        recovery_time = None
        recovered = False
        consecutive = 0
        for candidate in episode_rows[peak_index + 1:]:
            if bool(candidate.get("valid", 0)) and finite_float(candidate.get("error_norm")) < 0.5:
                consecutive += 1
                if consecutive >= 5:
                    recovery_time = float(candidate["timestamp_s"]) - float(peak["timestamp_s"])
                    recovered = True
                    break
            else:
                consecutive = 0
        return {
            "event_type": event_type,
            "episode_uid": peak["episode_uid"], "stream_uid": peak.get("stream_uid", ""),
            "threshold_mps": threshold,
            "trigger_field": trigger_field,
            "trigger_threshold": trigger_threshold if trigger_threshold is not None else threshold,
            "start_frame": active[0]["frame_index"], "end_frame": active[-1]["frame_index"],
            "peak_frame": peak["frame_index"], "peak_error_norm": peak["error_norm"],
            "peak_timestamp_s": peak["timestamp_s"], "condition": peak.get("condition", "unknown"),
            "source": peak.get("tracker_source", "unknown"), "event_frames": len(active),
            "recovered_below_0p5_for_5_frames": int(recovered),
            "recovery_time_s": recovery_time,
            "not_recovered_within_0p5s": int(not recovered or float(recovery_time) > 0.5),
        }

    events: list[dict[str, Any]] = []
    for _, episode_rows in _group_rows(rows, "stream_uid"):
        episode_rows = sorted(episode_rows, key=lambda item: int(item["frame_index"]))
        for threshold in (.5, 1.0, 2.0):
            active: list[dict[str, Any]] = []
            last_index = None
            for row in episode_rows + [None]:
                is_hit = row is not None and finite_float(row.get("error_norm")) > threshold and bool(row.get("valid", 0))
                index = int(row["frame_index"]) if row is not None else None
                if is_hit and (not active or last_index is None or index - last_index <= max_gap_frames):
                    active.append(row)
                    last_index = index
                    continue
                if active:
                    events.append(event_record("threshold", threshold, active, episode_rows))
                    active = []
                if is_hit:
                    active = [row]
                    last_index = index
    valid = [row for row in rows if bool(row.get("valid", 0)) and math.isfinite(finite_float(row.get("error_norm")))]
    if valid:
        top_threshold = float(np.quantile([finite_float(row["error_norm"]) for row in valid], .99))
        for _, episode_rows in _group_rows(rows, "stream_uid"):
            episode_rows = sorted(episode_rows, key=lambda item: int(item["frame_index"]))
            active: list[dict[str, Any]] = []
            last_index: int | None = None
            for row in episode_rows + [None]:
                is_hit = row is not None and bool(row.get("valid", 0)) and finite_float(row.get("error_norm")) >= top_threshold
                index = int(row["frame_index"]) if row is not None else None
                if is_hit and (not active or last_index is None or index - last_index <= max_gap_frames):
                    active.append(row)
                    last_index = index
                    continue
                if active:
                    events.append(event_record("top_1_percent", top_threshold, active, episode_rows))
                    active = []
                if is_hit:
                    active = [row]
                    last_index = index
            episode_valid = [row for row in episode_rows if bool(row.get("valid", 0)) and math.isfinite(finite_float(row.get("error_norm")))]
            if episode_valid:
                peak = max(episode_valid, key=lambda item: finite_float(item.get("error_norm")))
                events.append(event_record("episode_max", None, [peak], episode_rows))

    # Also catalog the tails of candidate causes.  These are deliberately
    # separate from high-error events so lead/lag and false positives remain
    # visible instead of being selected on the outcome.
    for field in (
        "direction_angular_error_deg",
        "range_absolute_error_m",
        "robot_angular_acceleration_rps2",
        "background_global_flow_px",
    ):
        field_values = np.asarray(
            [finite_float(row.get(field)) for row in rows if bool(row.get("valid", 0))],
            dtype=np.float64,
        )
        field_values = field_values[np.isfinite(field_values)]
        if len(field_values) < 20 or float(np.max(field_values) - np.min(field_values)) <= 1e-12:
            continue
        trigger_threshold = float(np.quantile(field_values, .99))
        for _, episode_rows in _group_rows(rows, "stream_uid"):
            episode_rows = sorted(episode_rows, key=lambda item: int(item["frame_index"]))
            active: list[dict[str, Any]] = []
            last_index: int | None = None
            for row in episode_rows + [None]:
                value = finite_float(row.get(field)) if row is not None else float("nan")
                is_hit = row is not None and bool(row.get("valid", 0)) and value >= trigger_threshold
                index = int(row["frame_index"]) if row is not None else None
                if is_hit and (not active or last_index is None or index - last_index <= max_gap_frames):
                    active.append(row)
                    last_index = index
                    continue
                if active:
                    events.append(
                        event_record(
                            f"{field}_top_1_percent",
                            None,
                            active,
                            episode_rows,
                            trigger_field=field,
                            trigger_threshold=trigger_threshold,
                        )
                    )
                    active = []
                if is_hit:
                    active = [row]
                    last_index = index

    for _, episode_rows in _group_rows(rows, "stream_uid"):
        episode_rows = sorted(episode_rows, key=lambda item: int(item["frame_index"]))
        for row in episode_rows:
            if int(row.get("frame_index", 0)) == 0:
                continue
            if not bool(row.get("valid", 0)):
                continue
            event_type = None
            if bool(row.get("measurement_discontinuity", False)):
                event_type = "measurement_discontinuity"
            elif bool(row.get("state_reset", False)):
                event_type = "state_reset"
            if event_type is not None:
                events.append(
                    event_record(event_type, None, [row], episode_rows, trigger_field=event_type)
                )
    return events


def _group_rows(rows: list[dict[str, Any]], field: str) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return grouped.items()


def lagged_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "direction_angular_error_deg",
        "range_absolute_error_m",
        "position_error",
        "kinematic_residual",
        "robot_angular_speed_rps",
        "robot_angular_acceleration_rps2",
        "background_global_flow_px",
    )
    output: list[dict[str, Any]] = []
    for field in fields:
        for lag in range(-5, 11):
            sensor_values: list[float] = []
            error_values: list[float] = []
            for _, stream_rows in _group_rows(rows, "stream_uid"):
                ordered = sorted(stream_rows, key=lambda item: int(item["frame_index"]))
                starts = [index for index, item in enumerate(ordered) if bool(item.get("state_reset", 0))]
                if not starts or starts[0] != 0:
                    starts.insert(0, 0)
                for start, end in zip(starts, [*starts[1:], len(ordered)]):
                    segment = ordered[start:end]
                    if len(segment) <= abs(lag):
                        continue
                    sensor = np.asarray([finite_float(item.get(field)) for item in segment])
                    error = np.asarray([finite_float(item.get("error_norm")) for item in segment])
                    valid = np.asarray([bool(item.get("valid", 0)) for item in segment])
                    if lag == 0:
                        x, y, keep = sensor, error, valid
                    elif lag > 0:
                        x, y, keep = sensor[:-lag], error[lag:], valid[lag:]
                    else:
                        offset = -lag
                        x, y, keep = sensor[offset:], error[:-offset], valid[:-offset]
                    finite = keep & np.isfinite(x) & np.isfinite(y)
                    sensor_values.extend(x[finite].tolist())
                    error_values.extend(y[finite].tolist())
            x = np.asarray(sensor_values, dtype=np.float64)
            y = np.asarray(error_values, dtype=np.float64)
            correlation = None
            if len(x) >= 3 and float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
                correlation = float(np.corrcoef(x, y)[0, 1])
            output.append({
                "field": field, "lag_frames": lag, "samples": len(x),
                "pearson_correlation": correlation,
            })
    return output


def scan_time_offsets(rows: list[dict[str, Any]], max_ms: float, step_ms: float) -> list[dict[str, Any]]:
    offsets = np.arange(-max_ms, max_ms + step_ms * .5, step_ms, dtype=np.float64) / 1000.0
    output: list[dict[str, Any]] = []
    group_specs: list[tuple[str, str]] = [("all", "all")]
    group_specs.extend(("condition", value) for value in sorted({str(row.get("condition", "unknown")) for row in rows}))
    group_specs.extend(("tracker_source", value) for value in sorted({str(row.get("tracker_source", "unknown")) for row in rows}))
    for group_name, group_value in group_specs:
        for offset in offsets:
            errors: list[float] = []
            for _, stream_rows in _group_rows(rows, "stream_uid"):
                ordered = sorted(stream_rows, key=lambda item: int(item["frame_index"]))
                reset_starts = [index for index, row in enumerate(ordered) if bool(row.get("state_reset", 0))]
                if not reset_starts or reset_starts[0] != 0:
                    reset_starts.insert(0, 0)
                for start, end in zip(reset_starts, [*reset_starts[1:], len(ordered)]):
                    segment = ordered[start:end]
                    if len(segment) < 2:
                        continue
                    ts = np.asarray([float(item["timestamp_s"]) for item in segment])
                    truth = np.asarray([[finite_float(item.get(f"velocity_gt_{axis}")) for axis in "xyz"] for item in segment])
                    pred = np.asarray([[finite_float(item.get(f"velocity_prediction_{axis}")) for axis in "xyz"] for item in segment])
                    valid = np.asarray([bool(item.get("valid", 0)) for item in segment])
                    if group_name == "condition":
                        selected = np.asarray([str(item.get("condition", "unknown")) == group_value for item in segment])
                    elif group_name == "tracker_source":
                        selected = np.asarray([str(item.get("tracker_source", "unknown")) == group_value for item in segment])
                    else:
                        selected = np.ones(len(segment), dtype=bool)
                    query = ts + offset
                    shifted = np.column_stack([
                        np.interp(query, ts, truth[:, dim], left=np.nan, right=np.nan)
                        for dim in range(3)
                    ])
                    shifted_valid = np.interp(
                        query, ts, valid.astype(np.float64), left=0.0, right=0.0
                    ) >= 0.999999
                    values = np.linalg.norm(pred - shifted, axis=1)
                    keep = valid & shifted_valid & selected & np.isfinite(values)
                    errors.extend(values[keep].tolist())
            value = np.asarray(errors, dtype=np.float64)
            output.append({
                "group": group_name, "value": group_value,
                "offset_ms": float(offset * 1000.0), "valid_frames": int(len(value)),
                "mae_norm": None if not len(value) else float(value.mean()),
                "p95_norm": None if not len(value) else float(np.quantile(value, .95)),
                "p99_norm": None if not len(value) else float(np.quantile(value, .99)),
            })
    return output


if __name__ == "__main__":
    main()

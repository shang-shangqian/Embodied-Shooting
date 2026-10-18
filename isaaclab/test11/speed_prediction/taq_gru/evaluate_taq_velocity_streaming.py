"""Evaluate a v2 checkpoint on complete chronological episode streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "test11.speed_prediction.taq_gru"

from ..common.time_aware_velocity_data import (
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION,
    TAQ_STREAM_FORMAT,
    load_torch_payload,
    taq_model_config_from_checkpoint,
)
from .time_aware_velocity_model import TimeAwareQualityGatedGRU


def _stream_tensors(payload: dict, path: Path) -> tuple[torch.Tensor, ...]:
    if payload.get("schema_version") != TAQ_SCHEMA_VERSION:
        raise ValueError(f"Incompatible stream schema: {path}")
    if payload.get("stream_format") != TAQ_STREAM_FORMAT:
        raise ValueError(f"Not a complete TAQ episode stream: {path}")
    if payload.get("label_semantics") != TAQ_LABEL_SEMANTICS:
        raise ValueError(f"Incompatible stream label semantics: {path}")
    features = torch.as_tensor(payload["sequence_features"], dtype=torch.float32)
    source_ids = torch.as_tensor(payload["sequence_source_ids"], dtype=torch.long)
    valid = torch.as_tensor(payload["sequence_valid_mask"], dtype=torch.bool)
    targets = torch.as_tensor(payload["target_velocity_base"], dtype=torch.float32)
    target_valid = torch.as_tensor(payload["target_valid_mask"], dtype=torch.bool)
    reset_mask = torch.as_tensor(payload["state_reset_mask"], dtype=torch.bool)
    timestamps = torch.as_tensor(payload["timestamps_s"], dtype=torch.float64)
    steps = int(features.shape[0]) if features.ndim == 2 else 0
    expected = (steps,)
    if (
        steps < 1
        or features.shape != (steps, 24)
        or source_ids.shape != expected
        or valid.shape != expected
        or targets.shape != (steps, 3)
        or target_valid.shape != expected
        or reset_mask.shape != expected
        or timestamps.shape != expected
    ):
        raise ValueError(f"Invalid stream tensor shapes: {path}")
    if not bool(reset_mask[0]):
        raise ValueError(f"The first stream frame must reset state: {path}")
    if steps > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError(f"Stream timestamps are not strictly increasing: {path}")
    if torch.any(target_valid & ~valid):
        raise ValueError(f"Stream target mask includes invalid frames: {path}")
    return features, source_ids, valid, targets, target_valid, reset_mask, timestamps


def _segmented_baseline(
    features: torch.Tensor,
    valid: torch.Tensor,
    reset_mask: torch.Tensor,
    *,
    model: TimeAwareQualityGatedGRU,
    window_s: float,
) -> torch.Tensor:
    """Compute a causal baseline without allowing history across state resets."""
    reset_indices = torch.nonzero(reset_mask, as_tuple=False).flatten().tolist()
    if not reset_indices or reset_indices[0] != 0:
        raise ValueError("reset_mask must start with a reset")
    boundaries = [*reset_indices, int(features.shape[0])]
    baseline = torch.zeros_like(features[:, :3])
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        baseline[start:end] = model.compute_velocity_baseline(
            features[start:end].unsqueeze(0),
            valid[start:end].unsqueeze(0),
            window_s=window_s,
        )[0]
    return baseline


def evaluate_streaming(
    checkpoint: Path,
    data_root: Path,
    device: torch.device,
) -> dict[str, float | int | str]:
    checkpoint_payload = load_torch_payload(checkpoint)
    if checkpoint_payload.get("schema_version") != TAQ_SCHEMA_VERSION:
        raise ValueError("Streaming evaluation requires a TAQ v2 checkpoint")
    if checkpoint_payload.get("label_semantics") != TAQ_LABEL_SEMANTICS:
        raise ValueError("Checkpoint has missing or incompatible TAQ v2 label semantics")
    model = TimeAwareQualityGatedGRU(**taq_model_config_from_checkpoint(checkpoint_payload)).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    norm = checkpoint_payload.get("input_norm")
    if norm is None:
        raise KeyError("Checkpoint is missing input_norm")
    model.set_input_norm_stats(norm["mean"].to(device), norm["std"].to(device))
    model.eval()

    stream_files = sorted(data_root.rglob("*.taqstream"))
    if not stream_files:
        raise FileNotFoundError(
            f"No complete episode streams below {data_root}; recollect validation/test data "
            "with --save_episode_stream"
        )
    window_s = float(checkpoint_payload.get("train_args", {}).get("baseline_window_s", 0.48))
    errors: list[torch.Tensor] = []
    # Validation roots may intentionally contain one scenario (for example, a fixed-camera pilot).
    # Do not fail after a complete evaluation merely because another scenario is absent.
    group_errors: dict[str, list[torch.Tensor]] = {}
    startup_errors: dict[str, list[torch.Tensor]] = {
        "100_300ms": [],
        "300_500ms": [],
        "stable_after_500ms": [],
    }
    reset_recovery_errors: dict[str, list[torch.Tensor]] = {
        "0_100ms": [],
        "100_300ms": [],
        "300_500ms": [],
    }
    reset_count = 0

    with torch.no_grad():
        for path in stream_files:
            stream_payload = load_torch_payload(path)
            metadata = dict(stream_payload.get("metadata", {}))
            scenario = str(metadata.get("scenario", "")).lower()
            if scenario not in {"ss", "sd"}:
                raise ValueError(f"Stream has unsupported scenario={scenario!r}: {path}")
            group_errors.setdefault(scenario, [])
            tensors = _stream_tensors(stream_payload, path)
            features, source_ids, valid, targets, target_valid, reset_mask, timestamps = (
                tensor.to(device) for tensor in tensors
            )
            baseline = _segmented_baseline(
                features,
                valid,
                reset_mask,
                model=model,
                window_s=window_s,
            )
            hidden = None
            episode_start = float(timestamps[0].item())
            last_reset_time = episode_start
            for index in range(int(features.shape[0])):
                if bool(reset_mask[index]):
                    hidden = None
                    last_reset_time = float(timestamps[index].item())
                    reset_count += 1
                mean, _, hidden, _ = model.step(
                    features[index : index + 1],
                    source_ids[index : index + 1],
                    hidden=hidden,
                    valid_mask=valid[index : index + 1],
                    velocity_baseline=baseline[index : index + 1],
                )
                if not bool(target_valid[index] & valid[index]):
                    continue
                value = torch.linalg.vector_norm(
                    mean[0] - targets[index], dim=-1
                ).detach().cpu().reshape(1)
                errors.append(value)
                group_errors[scenario].append(value)
                now = float(timestamps[index].item())
                episode_age = now - episode_start
                if episode_age < 0.3:
                    startup_errors["100_300ms"].append(value)
                elif episode_age < 0.5:
                    startup_errors["300_500ms"].append(value)
                else:
                    startup_errors["stable_after_500ms"].append(value)
                reset_age = now - last_reset_time
                if reset_age < 0.1:
                    reset_recovery_errors["0_100ms"].append(value)
                elif reset_age < 0.3:
                    reset_recovery_errors["100_300ms"].append(value)
                elif reset_age < 0.5:
                    reset_recovery_errors["300_500ms"].append(value)

    if not errors:
        raise RuntimeError("Streaming evaluation received no valid labels")
    values = torch.cat(errors)
    result: dict[str, float | int | str] = {
        "schema_version": TAQ_SCHEMA_VERSION,
        "stream_format": TAQ_STREAM_FORMAT,
        "episodes": len(stream_files),
        "state_resets": reset_count,
        "streaming_steps": int(values.numel()),
        "err_norm_mean": float(values.mean()),
        "err_norm_p95": float(torch.quantile(values, 0.95)),
        "err_norm_p99": float(torch.quantile(values, 0.99)),
        "err_norm_max": float(values.max()),
    }
    for scenario, scenario_values in group_errors.items():
        result[f"err_norm_p95_{scenario}"] = float(
            torch.quantile(torch.cat(scenario_values), 0.95)
        )
        result[f"streaming_steps_{scenario}"] = int(torch.cat(scenario_values).numel())
    for name, bucket_values in startup_errors.items():
        if bucket_values:
            result[f"episode_err_norm_p95_{name}"] = float(
                torch.quantile(torch.cat(bucket_values), 0.95)
            )
    for name, bucket_values in reset_recovery_errors.items():
        if bucket_values:
            result[f"reset_err_norm_p95_{name}"] = float(
                torch.quantile(torch.cat(bucket_values), 0.95)
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(
        args.device if "cuda" not in args.device or torch.cuda.is_available() else "cpu"
    )
    result = evaluate_streaming(
        args.checkpoint.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
        device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

"""Launch and verify paired velocity-validation collection jobs.

The simulator-specific collection command is supplied by the user because
the current IsaacLab collector has local policy/checkpoint arguments.  This
launcher adds immutable condition/pair metadata to a manifest, assigns a
deterministic trajectory seed per pair, and optionally runs the commands.  Each
condition command must write streams whose metadata contains the same
``pair_id`` and a matching trajectory seed.  Condition-specific CLI options
are configured explicitly; the launcher does not pass unknown arguments to an
existing collector.

The config format is deliberately simple YAML/JSON and needs no PyYAML:

  base_command: "python source/standalone/test11/collect_vision_data.py"
  common_args: ["--headless", "--save_episode_stream"]
  condition_args: []
  conditions: [A, B, C, D]
  seeds: [72, 73, 74]
  output_root: "outputs/paired_validation"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONDITIONS = ("A", "B", "C", "D")


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("config must be a mapping")
        return payload
    try:
        import yaml  # type: ignore
        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            return payload
    except ImportError:
        pass
    # Fallback parser for the supplied config shape: scalars, one-line lists,
    # and quoted strings.  It avoids making PyYAML a repository dependency.
    result: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value.strip("'\"")
        result[key.strip()] = parsed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="execute commands; default only writes the manifest")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config.expanduser().resolve())
    base = config.get("base_command")
    expected_conditions = tuple(
        str(value) for value in config.get("expected_conditions", DEFAULT_CONDITIONS)
    )
    conditions = [str(value) for value in config.get("conditions", expected_conditions)]
    seeds = [int(value) for value in config.get("seeds", [72, 73, 74])]
    common_args = [str(value) for value in config.get("common_args", [])]
    condition_args = config.get("condition_args", [])
    minimum_complete_pairs = int(config.get("minimum_complete_pairs", 1))
    minimum_valid_frames_per_condition = int(
        config.get("minimum_valid_frames_per_condition", 1)
    )
    trajectory_position_tolerance_m = float(
        config.get("trajectory_position_tolerance_m", 1e-4)
    )
    root_position_tolerance_m = float(config.get("root_position_tolerance_m", 1e-4))
    root_orientation_tolerance_rad = float(
        config.get("root_orientation_tolerance_rad", 1e-4)
    )
    root_velocity_tolerance = float(config.get("root_velocity_tolerance", 1e-4))
    fixed_camera_position_tolerance_m = float(
        config.get("fixed_camera_position_tolerance_m", 1e-5)
    )
    fixed_camera_orientation_tolerance_rad = float(
        config.get("fixed_camera_orientation_tolerance_rad", 1e-4)
    )
    mounted_camera_position_tolerance_m = float(
        config.get("mounted_camera_position_tolerance_m", 1e-5)
    )
    mounted_camera_orientation_tolerance_rad = float(
        config.get("mounted_camera_orientation_tolerance_rad", 1e-4)
    )
    fixed_camera_background_flow_median_px = float(
        config.get("fixed_camera_background_flow_median_px", 0.1)
    )
    fixed_camera_background_flow_p95_px = float(
        config.get("fixed_camera_background_flow_p95_px", 0.5)
    )
    fixed_camera_background_flow_min_samples = int(
        config.get("fixed_camera_background_flow_min_samples", 20)
    )
    output_root = Path(str(config.get("output_root", "outputs/paired_validation"))).expanduser().resolve()
    if not base:
        raise ValueError("config requires base_command")
    if not seeds:
        raise ValueError("config requires at least one seed")
    if (
        minimum_complete_pairs < 1
        or minimum_valid_frames_per_condition < 1
        or trajectory_position_tolerance_m <= 0.0
        or root_position_tolerance_m <= 0.0
        or root_orientation_tolerance_rad <= 0.0
        or root_velocity_tolerance <= 0.0
        or fixed_camera_position_tolerance_m <= 0.0
        or fixed_camera_orientation_tolerance_rad <= 0.0
        or mounted_camera_position_tolerance_m <= 0.0
        or mounted_camera_orientation_tolerance_rad <= 0.0
        or fixed_camera_background_flow_median_px <= 0.0
        or fixed_camera_background_flow_p95_px <= 0.0
        or fixed_camera_background_flow_min_samples < 1
    ):
        raise ValueError("minimum pair/frame requirements must be positive")
    if not expected_conditions or conditions != list(expected_conditions):
        raise ValueError(f"paired collection requires conditions={list(expected_conditions)}")
    flattened_condition_args = json.dumps(condition_args, ensure_ascii=False)
    if "{condition}" not in flattened_condition_args or "{pair_id}" not in flattened_condition_args:
        raise ValueError("condition_args must pass both {condition} and {pair_id} to the collector")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    manifest_path = output_root / "collection_manifest.json"
    artifact_hashes = {
        flag.lstrip("-"): _sha256_if_file(_argument_value(common_args, flag))
        for flag in ("--locomotion_policy", "--ostrack_checkpoint")
    }
    manifest_header = {
        "schema": "velocity_error_collection_manifest_v2_fixed_camera",
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "artifact_sha256": artifact_hashes,
        "validation_tolerances": {
            "trajectory_position_m": trajectory_position_tolerance_m,
            "root_position_m": root_position_tolerance_m,
            "root_orientation_rad": root_orientation_tolerance_rad,
            "root_velocity": root_velocity_tolerance,
            "fixed_camera_position_m": fixed_camera_position_tolerance_m,
            "fixed_camera_orientation_rad": fixed_camera_orientation_tolerance_rad,
            "mounted_camera_position_m": mounted_camera_position_tolerance_m,
            "mounted_camera_orientation_rad": mounted_camera_orientation_tolerance_rad,
            "fixed_camera_background_flow_median_px": (
                fixed_camera_background_flow_median_px
            ),
            "fixed_camera_background_flow_p95_px": (
                fixed_camera_background_flow_p95_px
            ),
            "fixed_camera_background_flow_min_samples": (
                fixed_camera_background_flow_min_samples
            ),
        },
    }
    for seed in seeds:
        pair_id = f"pair_seed_{seed:06d}"
        for condition in conditions:
            out_dir = output_root / f"seed_{seed:06d}" / condition
            if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_existing:
                raise FileExistsError(f"refusing non-empty output directory: {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)
            substitutions = {
                "seed": str(seed), "out_dir": str(out_dir),
                "condition": condition, "pair_id": pair_id,
            }
            command_text = str(base).format(**substitutions)
            command = shlex.split(command_text) + common_args + [
                "--seed", str(seed), "--out_dir", str(out_dir),
            ]
            for item in condition_args:
                if isinstance(item, str):
                    command.append(item.format(**substitutions))
                elif isinstance(item, (list, tuple)):
                    command.extend(str(value).format(**substitutions) for value in item)
                else:
                    raise ValueError("condition_args entries must be strings or lists")
            entry = {"pair_id": pair_id, "seed": seed, "condition": condition, "out_dir": str(out_dir), "command": command, "created_at": datetime.now(timezone.utc).isoformat()}
            manifest.append(entry)
            manifest_path.write_text(
                json.dumps({**manifest_header, "jobs": manifest}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if args.run:
                subprocess.run(command, check=True)
                _verify_job_output(out_dir, condition, pair_id, seed)
    if args.run:
        _verify_complete_pairs(
            output_root,
            expected_conditions,
            minimum_complete_pairs=minimum_complete_pairs,
            minimum_valid_frames_per_condition=minimum_valid_frames_per_condition,
            trajectory_position_tolerance_m=trajectory_position_tolerance_m,
            root_position_tolerance_m=root_position_tolerance_m,
            root_orientation_tolerance_rad=root_orientation_tolerance_rad,
            root_velocity_tolerance=root_velocity_tolerance,
            fixed_camera_position_tolerance_m=fixed_camera_position_tolerance_m,
            fixed_camera_orientation_tolerance_rad=fixed_camera_orientation_tolerance_rad,
            mounted_camera_position_tolerance_m=mounted_camera_position_tolerance_m,
            mounted_camera_orientation_tolerance_rad=mounted_camera_orientation_tolerance_rad,
            fixed_camera_background_flow_median_px=(
                fixed_camera_background_flow_median_px
            ),
            fixed_camera_background_flow_p95_px=fixed_camera_background_flow_p95_px,
            fixed_camera_background_flow_min_samples=(
                fixed_camera_background_flow_min_samples
            ),
        )
    print(f"wrote {manifest_path}")
    if not args.run:
        for entry in manifest:
            print(shlex.join(entry["command"]))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _argument_value(arguments: list[str], flag: str) -> str | None:
    try:
        index = arguments.index(flag)
    except ValueError:
        return None
    return arguments[index + 1] if index + 1 < len(arguments) else None


def _sha256_if_file(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stream(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--run verification requires PyTorch") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old PyTorch.
        return torch.load(path, map_location="cpu")


def _verify_job_output(out_dir: Path, condition: str, pair_id: str, seed: int) -> None:
    streams = sorted(out_dir.rglob("*.taqstream"))
    if not streams:
        raise RuntimeError(f"collector produced no .taqstream files: {out_dir}")
    for path in streams:
        metadata = dict(_load_stream(path).get("metadata", {}))
        expected = {
            "condition": condition,
            "pair_id": pair_id,
            "trajectory_seed": int(seed),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"{path} metadata {key}={metadata.get(key)!r}; expected {value!r}"
                )
        if not metadata.get("trajectory_uid"):
            raise RuntimeError(f"{path} is missing trajectory_uid")
        expected_fixed_camera = condition in {"A", "B"}
        if metadata.get("validation_fixed_camera") is not expected_fixed_camera:
            raise RuntimeError(
                f"{path} validation_fixed_camera={metadata.get('validation_fixed_camera')!r}; "
                f"expected {expected_fixed_camera!r} for condition {condition}"
            )
        if metadata.get("robot_root_intervention") != "none":
            raise RuntimeError(
                f"{path} uses forbidden robot root intervention: "
                f"{metadata.get('robot_root_intervention')!r}"
            )
        pair_audit = path.with_name("paired_state_audit.jsonl")
        if not pair_audit.is_file():
            raise RuntimeError(f"{path} is missing paired_state_audit.jsonl")


def _vector(row: dict[str, Any], field: str, length: int) -> list[float]:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"paired audit field {field!r} must be a length-{length} list")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"paired audit field {field!r} contains non-finite values")
    return result


def _quaternion_distance_rad(first: list[float], second: list[float]) -> float:
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        raise RuntimeError("paired audit contains a zero quaternion")
    dot = abs(
        sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
    )
    return 2.0 * math.acos(max(0.0, min(1.0, dot)))


def _read_paired_state_audit(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing paired state audit: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"paired state audit is empty: {path}")
    return rows


def _validate_paired_state_rows(
    rows_by_condition: dict[str, list[dict[str, Any]]],
    expected_conditions: tuple[str, ...],
    *,
    trajectory_position_tolerance_m: float,
    root_position_tolerance_m: float,
    root_orientation_tolerance_rad: float,
    root_velocity_tolerance: float,
    fixed_camera_position_tolerance_m: float,
    fixed_camera_orientation_tolerance_rad: float,
    mounted_camera_position_tolerance_m: float,
    mounted_camera_orientation_tolerance_rad: float,
    fixed_camera_background_flow_median_px: float,
    fixed_camera_background_flow_p95_px: float,
    fixed_camera_background_flow_min_samples: int = 20,
    audit_label: str | None = None,
) -> dict[str, float | int]:
    """Validate one complete condition set on every paired physics step."""
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in expected_conditions:
        condition_rows = rows_by_condition.get(condition, [])
        by_step: dict[int, dict[str, Any]] = {}
        for row in condition_rows:
            step = row.get("physics_step_index")
            if step is None:
                raise RuntimeError(f"condition {condition} paired audit lacks physics_step_index")
            step_id = int(step)
            if step_id in by_step:
                raise RuntimeError(
                    f"condition {condition} has duplicate paired audit step {step_id}"
                )
            by_step[step_id] = row
        if not by_step:
            raise RuntimeError(f"condition {condition} has no paired physics frames")
        indexed[condition] = by_step

    reference_condition = expected_conditions[0]
    reference_steps = set(indexed[reference_condition])
    for condition in expected_conditions[1:]:
        steps = set(indexed[condition])
        if steps != reference_steps:
            missing = sorted(reference_steps - steps)[:10]
            extra = sorted(steps - reference_steps)[:10]
            raise RuntimeError(
                f"paired physics timeline mismatch {reference_condition} vs {condition}: "
                f"missing={missing}, extra={extra}"
            )

    maxima = {
        "samples": len(reference_steps),
        "target_difference_m": 0.0,
        "root_position_difference_m": 0.0,
        "root_orientation_difference_rad": 0.0,
        "root_linear_velocity_difference": 0.0,
        "root_angular_velocity_difference": 0.0,
        "fixed_camera_position_drift_m": 0.0,
        "fixed_camera_orientation_drift_rad": 0.0,
        "mounted_camera_position_error_m": 0.0,
        "mounted_camera_orientation_error_rad": 0.0,
        "fixed_camera_background_flow_median_px": 0.0,
        "fixed_camera_background_flow_p95_px": 0.0,
        "fixed_camera_background_flow_samples": 0,
        "fixed_camera_background_flow_pairs_checked": 0,
    }
    fixed_conditions = set(expected_conditions) & {"A", "B"}
    fixed_initial_pose: dict[str, tuple[list[float], list[float]]] = {}
    fixed_background_flow: list[float] = []
    for step in sorted(reference_steps):
        reference = indexed[reference_condition][step]
        reference_timestamp = float(reference.get("physics_timestamp_s", math.nan))
        reference_target = _vector(reference, "target_position_world_m", 3)
        reference_root_position = _vector(reference, "base_position_world_m", 3)
        reference_root_quaternion = _vector(reference, "base_quaternion_world", 4)
        reference_linear_velocity = _vector(
            reference, "robot_linear_velocity_base_mps", 3
        )
        reference_angular_velocity = _vector(
            reference, "robot_angular_velocity_base_rps", 3
        )
        for condition in expected_conditions:
            row = indexed[condition][step]
            timestamp = float(row.get("physics_timestamp_s", math.nan))
            if not math.isfinite(timestamp) or abs(timestamp - reference_timestamp) > 1e-9:
                raise RuntimeError(
                    f"paired physics timestamp mismatch at step {step}: "
                    f"{reference_condition}={reference_timestamp}, {condition}={timestamp}"
                )
            target_difference = math.dist(
                reference_target, _vector(row, "target_position_world_m", 3)
            )
            root_position_difference = math.dist(
                reference_root_position, _vector(row, "base_position_world_m", 3)
            )
            root_orientation_difference = _quaternion_distance_rad(
                reference_root_quaternion, _vector(row, "base_quaternion_world", 4)
            )
            linear_velocity_difference = math.dist(
                reference_linear_velocity,
                _vector(row, "robot_linear_velocity_base_mps", 3),
            )
            angular_velocity_difference = math.dist(
                reference_angular_velocity,
                _vector(row, "robot_angular_velocity_base_rps", 3),
            )
            checks = (
                ("target_difference_m", target_difference, trajectory_position_tolerance_m),
                ("root_position_difference_m", root_position_difference, root_position_tolerance_m),
                (
                    "root_orientation_difference_rad",
                    root_orientation_difference,
                    root_orientation_tolerance_rad,
                ),
                (
                    "root_linear_velocity_difference",
                    linear_velocity_difference,
                    root_velocity_tolerance,
                ),
                (
                    "root_angular_velocity_difference",
                    angular_velocity_difference,
                    root_velocity_tolerance,
                ),
            )
            for name, value, tolerance in checks:
                maxima[name] = max(float(maxima[name]), value)
                if value > tolerance:
                    raise RuntimeError(
                        f"paired state mismatch at step {step}, condition {condition}: "
                        f"{name}={value:.6g} exceeds {tolerance:.6g}"
                    )

            camera_position = _vector(row, "camera_position_world_m", 3)
            camera_quaternion = _vector(row, "camera_quaternion_world", 4)
            if condition in fixed_conditions:
                expected_mode = "fixed_world_proxy"
                flow = row.get("background_global_flow_px")
                if flow is not None and math.isfinite(float(flow)):
                    fixed_background_flow.append(float(flow))
                initial_position, initial_quaternion = fixed_initial_pose.setdefault(
                    condition, (camera_position, camera_quaternion)
                )
                position_drift = math.dist(initial_position, camera_position)
                orientation_drift = _quaternion_distance_rad(
                    initial_quaternion, camera_quaternion
                )
                maxima["fixed_camera_position_drift_m"] = max(
                    float(maxima["fixed_camera_position_drift_m"]), position_drift
                )
                maxima["fixed_camera_orientation_drift_rad"] = max(
                    float(maxima["fixed_camera_orientation_drift_rad"]), orientation_drift
                )
                if position_drift > fixed_camera_position_tolerance_m:
                    raise RuntimeError(
                        f"fixed camera position drift at step {step}, condition {condition}: "
                        f"{position_drift:.6g} m"
                    )
                if orientation_drift > fixed_camera_orientation_tolerance_rad:
                    raise RuntimeError(
                        f"fixed camera orientation drift at step {step}, condition {condition}: "
                        f"{orientation_drift:.6g} rad"
                    )
            else:
                expected_mode = "torso_mounted_proxy"
                mounted_position = _vector(
                    row, "mounted_camera_position_world_m", 3
                )
                mounted_quaternion = _vector(
                    row, "mounted_camera_quaternion_world", 4
                )
                position_error = math.dist(camera_position, mounted_position)
                orientation_error = _quaternion_distance_rad(
                    camera_quaternion, mounted_quaternion
                )
                maxima["mounted_camera_position_error_m"] = max(
                    float(maxima["mounted_camera_position_error_m"]), position_error
                )
                maxima["mounted_camera_orientation_error_rad"] = max(
                    float(maxima["mounted_camera_orientation_error_rad"]),
                    orientation_error,
                )
                if position_error > mounted_camera_position_tolerance_m:
                    raise RuntimeError(
                        f"mounted camera position mismatch at step {step}, condition {condition}: "
                        f"{position_error:.6g} m"
                    )
                if orientation_error > mounted_camera_orientation_tolerance_rad:
                    raise RuntimeError(
                        f"mounted camera orientation mismatch at step {step}, condition {condition}: "
                        f"{orientation_error:.6g} rad"
                    )
            if row.get("camera_mode") != expected_mode:
                raise RuntimeError(
                    f"condition {condition} camera_mode={row.get('camera_mode')!r}; "
                    f"expected {expected_mode!r}"
                )

    if fixed_conditions:
        flow_samples = len(fixed_background_flow)
        maxima["fixed_camera_background_flow_samples"] = flow_samples
        if flow_samples < fixed_camera_background_flow_min_samples:
            label = f" for {audit_label}" if audit_label else ""
            print(
                "warning: fixed-camera background-flow gate skipped"
                f"{label}: finite_samples={flow_samples} < "
                f"required={fixed_camera_background_flow_min_samples}; "
                "camera-pose hard checks still passed"
            )
        else:
            ordered_flow = sorted(fixed_background_flow)

            def quantile(q: float) -> float:
                index = q * (len(ordered_flow) - 1)
                lower = int(math.floor(index))
                upper = int(math.ceil(index))
                weight = index - lower
                return ordered_flow[lower] * (1.0 - weight) + ordered_flow[upper] * weight

            flow_median = quantile(0.5)
            flow_p95 = quantile(0.95)
            maxima["fixed_camera_background_flow_median_px"] = flow_median
            maxima["fixed_camera_background_flow_p95_px"] = flow_p95
            maxima["fixed_camera_background_flow_pairs_checked"] = 1
            if flow_median > fixed_camera_background_flow_median_px:
                raise RuntimeError(
                    "fixed-camera background flow median "
                    f"{flow_median:.6g} px exceeds {fixed_camera_background_flow_median_px:.6g} px"
                )
            if flow_p95 > fixed_camera_background_flow_p95_px:
                raise RuntimeError(
                    "fixed-camera background flow P95 "
                    f"{flow_p95:.6g} px exceeds {fixed_camera_background_flow_p95_px:.6g} px"
                )

    if {"A", "B"}.issubset(indexed):
        for step in reference_steps:
            position_difference = math.dist(
                _vector(indexed["A"][step], "camera_position_world_m", 3),
                _vector(indexed["B"][step], "camera_position_world_m", 3),
            )
            orientation_difference = _quaternion_distance_rad(
                _vector(indexed["A"][step], "camera_quaternion_world", 4),
                _vector(indexed["B"][step], "camera_quaternion_world", 4),
            )
            if position_difference > fixed_camera_position_tolerance_m:
                raise RuntimeError(
                    f"A/B fixed cameras differ at step {step}: {position_difference:.6g} m"
                )
            if orientation_difference > fixed_camera_orientation_tolerance_rad:
                raise RuntimeError(
                    f"A/B fixed cameras differ at step {step}: {orientation_difference:.6g} rad"
                )
    return maxima


def _verify_complete_pairs(
    output_root: Path,
    expected_conditions: tuple[str, ...],
    *,
    minimum_complete_pairs: int,
    minimum_valid_frames_per_condition: int,
    trajectory_position_tolerance_m: float,
    root_position_tolerance_m: float,
    root_orientation_tolerance_rad: float,
    root_velocity_tolerance: float,
    fixed_camera_position_tolerance_m: float,
    fixed_camera_orientation_tolerance_rad: float,
    mounted_camera_position_tolerance_m: float,
    mounted_camera_orientation_tolerance_rad: float,
    fixed_camera_background_flow_median_px: float,
    fixed_camera_background_flow_p95_px: float,
    fixed_camera_background_flow_min_samples: int,
) -> None:
    pairs: dict[str, set[str]] = {}
    duplicates: set[tuple[str, str]] = set()
    valid_frames_by_pair_condition: dict[tuple[str, str], int] = {}
    audit_path_by_pair_condition: dict[tuple[str, str], Path] = {}
    for path in sorted(output_root.rglob("*.taqstream")):
        payload = _load_stream(path)
        metadata = dict(payload.get("metadata", {}))
        trajectory_uid = str(metadata.get("trajectory_uid", ""))
        condition = str(metadata.get("condition", ""))
        if not trajectory_uid or condition not in expected_conditions:
            continue
        key = (trajectory_uid, condition)
        target_valid = payload.get("target_valid_mask")
        if target_valid is not None:
            source_ids = payload.get("sequence_source_ids")
            if source_ids is not None:
                # Source id 4 is INVALID/lost in the TAQ schema. Require real
                # observed/predicted coverage in addition to all-time labels.
                target_valid = target_valid & (source_ids != 4)
            valid_frames_by_pair_condition[key] = int(target_valid.sum().item())
        audit_path_by_pair_condition[key] = path.with_name("paired_state_audit.jsonl")
        if condition in pairs.setdefault(trajectory_uid, set()):
            duplicates.add(key)
        pairs[trajectory_uid].add(condition)
    if duplicates:
        raise RuntimeError(f"duplicate condition streams for trajectory pairs: {sorted(duplicates)[:10]}")
    incomplete = {
        trajectory: sorted(set(expected_conditions) - values)
        for trajectory, values in pairs.items() if values != set(expected_conditions)
    }
    complete_pair_ids = {
        pair for pair, values in pairs.items() if values == set(expected_conditions)
    }
    complete_pairs = len(complete_pair_ids)
    if complete_pairs < minimum_complete_pairs:
        raise RuntimeError(
            f"only {complete_pairs} complete trajectory pairs; "
            f"require at least {minimum_complete_pairs}"
        )
    valid_frames_by_condition = {
        condition: sum(
            valid_frames_by_pair_condition.get((pair, condition), 0)
            for pair in complete_pair_ids
        )
        for condition in expected_conditions
    }
    insufficient_frames = {
        condition: frames
        for condition, frames in valid_frames_by_condition.items()
        if frames < minimum_valid_frames_per_condition
    }
    if insufficient_frames:
        raise RuntimeError(
            "insufficient valid frames by condition: "
            f"{insufficient_frames}; require {minimum_valid_frames_per_condition} each"
        )
    aggregate_maxima: dict[str, float | int] = {
        "samples": 0,
        "target_difference_m": 0.0,
        "root_position_difference_m": 0.0,
        "root_orientation_difference_rad": 0.0,
        "root_linear_velocity_difference": 0.0,
        "root_angular_velocity_difference": 0.0,
        "fixed_camera_position_drift_m": 0.0,
        "fixed_camera_orientation_drift_rad": 0.0,
        "mounted_camera_position_error_m": 0.0,
        "mounted_camera_orientation_error_rad": 0.0,
        "fixed_camera_background_flow_median_px": 0.0,
        "fixed_camera_background_flow_p95_px": 0.0,
        "fixed_camera_background_flow_samples": 0,
        "fixed_camera_background_flow_pairs_checked": 0,
    }
    for pair in sorted(complete_pair_ids):
        rows_by_condition: dict[str, list[dict[str, Any]]] = {}
        for condition in expected_conditions:
            audit_path = audit_path_by_pair_condition[(pair, condition)]
            rows_by_condition[condition] = _read_paired_state_audit(audit_path)
        pair_maxima = _validate_paired_state_rows(
            rows_by_condition,
            expected_conditions,
            trajectory_position_tolerance_m=trajectory_position_tolerance_m,
            root_position_tolerance_m=root_position_tolerance_m,
            root_orientation_tolerance_rad=root_orientation_tolerance_rad,
            root_velocity_tolerance=root_velocity_tolerance,
            fixed_camera_position_tolerance_m=fixed_camera_position_tolerance_m,
            fixed_camera_orientation_tolerance_rad=(
                fixed_camera_orientation_tolerance_rad
            ),
            mounted_camera_position_tolerance_m=mounted_camera_position_tolerance_m,
            mounted_camera_orientation_tolerance_rad=(
                mounted_camera_orientation_tolerance_rad
            ),
            fixed_camera_background_flow_median_px=(
                fixed_camera_background_flow_median_px
            ),
            fixed_camera_background_flow_p95_px=(
                fixed_camera_background_flow_p95_px
            ),
            fixed_camera_background_flow_min_samples=(
                fixed_camera_background_flow_min_samples
            ),
            audit_label=pair,
        )
        summed_keys = {
            "samples",
            "fixed_camera_background_flow_samples",
            "fixed_camera_background_flow_pairs_checked",
        }
        for key, value in pair_maxima.items():
            if key in summed_keys:
                aggregate_maxima[key] = int(aggregate_maxima[key]) + int(value)
            else:
                aggregate_maxima[key] = max(float(aggregate_maxima[key]), float(value))
    print(
        "paired collection verified:",
        f"complete_pairs={complete_pairs}",
        f"excluded_incomplete_pairs={len(incomplete)}",
        f"valid_frames={valid_frames_by_condition}",
        f"paired_physics_samples={aggregate_maxima['samples']}",
        f"maxima={aggregate_maxima}",
    )
    if incomplete:
        print(
            "excluded incomplete trajectory pairs:",
            dict(list(incomplete.items())[:10]),
        )


if __name__ == "__main__":
    main()

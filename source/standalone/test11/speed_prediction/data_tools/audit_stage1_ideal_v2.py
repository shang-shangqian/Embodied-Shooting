"""Audit Stage-1 ideal-v2 TAQ chunks against the collection hard gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.data_tools"

from ..common.time_aware_velocity_data import (
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION,
    TAQ_STREAM_FORMAT,
    TAQ_YAW_RATE_SEMANTICS,
)


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def audit(root: Path, *, min_window_gap_s: float = 0.02, burn_in_s: float = 0.10) -> dict:
    files = sorted(root.rglob("*.pt"))
    stream_files = sorted(root.rglob("*.taqstream"))
    if not files:
        raise FileNotFoundError(f"No .pt chunks found below {root}")
    failures: list[str] = []
    episodes: dict[str, list[tuple[float, float, Path, int]]] = defaultdict(list)
    split_episodes: dict[str, set[str]] = defaultdict(set)
    chunk_episodes_by_split: dict[str, set[str]] = defaultdict(set)
    stream_episodes_by_split: dict[str, set[str]] = defaultdict(set)
    runs_by_split: dict[str, set[str]] = defaultdict(set)
    total_windows = 0
    total_supervised = 0
    min_supervised = None
    schemas: set[str] = set()
    semantics: set[str] = set()
    for path in files:
        payload = _load(path)
        schema = str(payload.get("schema_version", "missing"))
        schemas.add(schema)
        if schema != TAQ_SCHEMA_VERSION:
            failures.append(f"{path}: schema={schema!r}")
        metadata = dict(payload.get("metadata", {}))
        semantic = str(payload.get("label_semantics", metadata.get("label_semantics", "missing")))
        semantics.add(semantic)
        if semantic != TAQ_LABEL_SEMANTICS:
            failures.append(f"{path}: label_semantics={semantic!r}")
        if metadata.get("angular_velocity_z_semantics") != TAQ_YAW_RATE_SEMANTICS:
            failures.append(f"{path}: missing or incompatible yaw-rate semantics")
        split = str(metadata.get("split", "missing"))
        run_id = str(metadata.get("collection_run_id", "missing"))
        episode_uid = str(metadata.get("episode_uid", "missing"))
        for key, value in (("split", split), ("collection_run_id", run_id), ("episode_uid", episode_uid)):
            if value == "missing" or not value:
                failures.append(f"{path}: missing metadata {key!r}")
        split_episodes[split].add(episode_uid)
        chunk_episodes_by_split[split].add(episode_uid)
        runs_by_split[split].add(run_id)
        try:
            features = torch.as_tensor(payload["sequence_features"], dtype=torch.float32)
            sources = torch.as_tensor(payload["sequence_source_ids"], dtype=torch.long)
            valid = torch.as_tensor(payload["sequence_valid_mask"], dtype=torch.bool)
            labels = torch.as_tensor(payload["target_velocity_base"], dtype=torch.float32)
            target_valid = torch.as_tensor(payload["target_valid_mask"], dtype=torch.bool)
        except KeyError as exc:
            failures.append(f"{path}: missing key {exc.args[0]!r}")
            continue
        if features.ndim != 3 or features.shape[-1] != 24:
            failures.append(f"{path}: bad feature shape {tuple(features.shape)}")
            continue
        expected = features.shape[:2]
        if sources.shape != expected or valid.shape != expected or target_valid.shape != expected:
            failures.append(f"{path}: feature/source/mask shape mismatch")
            continue
        if labels.shape != (*expected, 3):
            failures.append(f"{path}: bad target shape {tuple(labels.shape)}")
            continue
        if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
            failures.append(f"{path}: non-finite feature or label")
        if torch.any(target_valid & ~valid):
            failures.append(f"{path}: target mask includes padding")
        supervised = target_valid.sum(dim=1)
        total_windows += int(features.shape[0])
        total_supervised += int(supervised.sum().item())
        min_supervised = int(supervised.min().item()) if min_supervised is None else min(min_supervised, int(supervised.min().item()))
        if torch.any(supervised == 0):
            failures.append(f"{path}: window without supervised time steps")
        elapsed = torch.cumsum(features[..., 12], dim=1)
        if torch.any(target_valid & (elapsed < float(burn_in_s) - 1e-6)):
            failures.append(f"{path}: target mask contains pre-burn-in supervision")
        starts = payload.get("sample_window_start_s")
        ends = payload.get("sample_window_end_s")
        if starts is None or ends is None:
            failures.append(f"{path}: missing sample window timestamps")
        else:
            starts = torch.as_tensor(starts, dtype=torch.float64).reshape(-1)
            ends = torch.as_tensor(ends, dtype=torch.float64).reshape(-1)
            if starts.numel() != features.shape[0] or ends.numel() != features.shape[0]:
                failures.append(f"{path}: window timestamp count mismatch")
                continue
            for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                episodes[episode_uid].append((start, end, path, row))

    stream_steps = 0
    stream_episode_uids: set[str] = set()
    for path in stream_files:
        payload = _load(path)
        metadata = dict(payload.get("metadata", {}))
        stream_split = str(metadata.get("split", "missing"))
        stream_run_id = str(metadata.get("collection_run_id", "missing"))
        stream_episode_uid = str(metadata.get("episode_uid", "missing"))
        split_episodes[stream_split].add(stream_episode_uid)
        stream_episodes_by_split[stream_split].add(stream_episode_uid)
        runs_by_split[stream_split].add(stream_run_id)
        if payload.get("schema_version") != TAQ_SCHEMA_VERSION:
            failures.append(f"{path}: incompatible stream schema")
        if payload.get("stream_format") != TAQ_STREAM_FORMAT:
            failures.append(f"{path}: incompatible stream format")
        if payload.get("label_semantics") != TAQ_LABEL_SEMANTICS:
            failures.append(f"{path}: incompatible stream label semantics")
        if metadata.get("angular_velocity_z_semantics") != TAQ_YAW_RATE_SEMANTICS:
            failures.append(f"{path}: missing or incompatible stream yaw-rate semantics")
        for key in ("split", "collection_run_id", "episode_uid"):
            if not metadata.get(key):
                failures.append(f"{path}: missing metadata {key!r}")
        try:
            features = torch.as_tensor(payload["sequence_features"], dtype=torch.float32)
            sources = torch.as_tensor(payload["sequence_source_ids"], dtype=torch.long)
            valid = torch.as_tensor(payload["sequence_valid_mask"], dtype=torch.bool)
            labels = torch.as_tensor(payload["target_velocity_base"], dtype=torch.float32)
            target_valid = torch.as_tensor(payload["target_valid_mask"], dtype=torch.bool)
            timestamps = torch.as_tensor(payload["timestamps_s"], dtype=torch.float64)
            reset_mask = torch.as_tensor(payload["state_reset_mask"], dtype=torch.bool)
        except KeyError as exc:
            failures.append(f"{path}: missing stream key {exc.args[0]!r}")
            continue
        steps = int(features.shape[0]) if features.ndim == 2 else 0
        expected = (steps,)
        shapes_ok = (
            features.ndim == 2
            and features.shape[-1] == 24
            and sources.shape == expected
            and valid.shape == expected
            and labels.shape == (steps, 3)
            and target_valid.shape == expected
            and timestamps.shape == expected
            and reset_mask.shape == expected
        )
        if not shapes_ok or steps == 0:
            failures.append(f"{path}: invalid episode stream shapes")
            continue
        if not bool(reset_mask[0]):
            failures.append(f"{path}: first stream frame must reset state")
        if steps > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
            failures.append(f"{path}: stream timestamps are not strictly increasing")
        if torch.any(target_valid & ~valid):
            failures.append(f"{path}: stream target mask includes invalid frames")
        if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
            failures.append(f"{path}: non-finite stream feature or label")
        elapsed = timestamps - timestamps[0]
        if torch.any(target_valid & (elapsed < float(burn_in_s) - 1e-6)):
            failures.append(f"{path}: stream target mask contains pre-burn-in supervision")
        stream_steps += steps
        stream_episode_uids.add(stream_episode_uid)

    for split in ("val", "test"):
        missing_streams = chunk_episodes_by_split[split] - stream_episodes_by_split[split]
        if missing_streams:
            failures.append(
                f"{len(missing_streams)} {split} episodes with training windows are missing complete streams"
            )
    duplicate_uids = sum(
        len({split for split, uids in split_episodes.items() if uid in uids}) > 1
        for uid in {uid for uids in split_episodes.values() for uid in uids}
    )
    if duplicate_uids:
        failures.append(f"{duplicate_uids} episode_uid values cross splits")
    run_split_membership: dict[str, set[str]] = defaultdict(set)
    for split, run_ids in runs_by_split.items():
        for run_id in run_ids:
            run_split_membership[run_id].add(split)
    duplicate_runs = sum(len(splits) > 1 for splits in run_split_membership.values())
    if duplicate_runs:
        failures.append(f"{duplicate_runs} collection_run_id values cross splits")
    overlap_count = 0
    gap_violations = 0
    max_windows = 0
    for episode_uid, intervals in episodes.items():
        max_windows = max(max_windows, len(intervals))
        if len(intervals) > 4:
            failures.append(f"{episode_uid}: {len(intervals)} windows exceeds cap 4")
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] <= previous[1] + 1e-9:
                overlap_count += 1
            elif current[0] - previous[1] < min_window_gap_s - 1e-6:
                gap_violations += 1
    if overlap_count:
        failures.append(f"{overlap_count} overlapping window pairs")
    if gap_violations:
        failures.append(f"{gap_violations} window gaps below {min_window_gap_s:g}s")
    result = {
        "root": str(root),
        "schema_versions": sorted(schemas),
        "label_semantics": sorted(semantics),
        "files": len(files),
        "stream_files": len(stream_files),
        "stream_episodes": len(stream_episode_uids),
        "stream_steps": stream_steps,
        "episodes": len(episodes),
        "windows": total_windows,
        "supervised_steps": total_supervised,
        "min_supervised_steps_per_window": min_supervised or 0,
        "max_windows_per_episode": max_windows,
        "overlap_pairs": overlap_count,
        "gap_violations": gap_violations,
        "duplicate_episode_uids_across_splits": duplicate_uids,
        "duplicate_collection_runs_across_splits": duplicate_runs,
        "splits": {name: len(values) for name, values in sorted(split_episodes.items())},
        "stream_episodes_by_split": {
            name: len(values) for name, values in sorted(stream_episodes_by_split.items())
        },
        "collection_runs_by_split": {name: sorted(values) for name, values in sorted(runs_by_split.items())},
        "failures": failures,
        "passed": not failures,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--min_window_gap_s", type=float, default=0.02)
    parser.add_argument("--supervision_burn_in_s", type=float, default=0.10)
    args = parser.parse_args()
    import json

    result = audit(args.data_root.expanduser().resolve(), min_window_gap_s=args.min_window_gap_s, burn_in_s=args.supervision_burn_in_s)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Print a compact sanity report for a Stage-2 collection directory."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch


SOURCE_NAMES = {
    0: "initial",
    1: "flow",
    2: "ostrack",
    3: "prediction",
    4: "invalid",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _profile_directories(root: Path) -> list[Path]:
    directories = {
        path.parents[3]
        for path in root.rglob("taq_velocity_chunk_*.pt")
        if len(path.parents) >= 4
    }
    return sorted(directories)


def _missing_chunk_message(root: Path) -> str:
    """Explain a failed collection when Isaac Sim returned a false success code."""
    logs = sorted(root.rglob("collector.log"))
    crashed_logs: list[Path] = []
    gpu_oom = False
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            "Traceback (most recent call last):" in text
            or "Failed to create simulation view backend" in text
            or "unable to create a PhysicsScene" in text
        ):
            crashed_logs.append(path)
        if (
            "failed to allocate memory" in text
            or "Not enough GPU memory" in text
            or "CUDA out of memory" in text
        ):
            gpu_oom = True

    lines = [f"No TAQ chunks were found below: {root}"]
    if crashed_logs:
        lines.append(
            f"The collector crashed in {len(crashed_logs)}/{len(logs)} run(s); "
            f"first failing log: {crashed_logs[0]}"
        )
    if gpu_oom:
        lines.append(
            "The collector log reports insufficient GPU memory. Close the process "
            "using that GPU or select another GPU with COLLECTION_DEVICE=cuda:N."
        )
    return "\n".join(lines)


def main() -> None:
    args = _arguments()
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory was not found: {root}")

    profile_directories = _profile_directories(root)
    if not profile_directories:
        raise FileNotFoundError(_missing_chunk_message(root))

    print(f"[stage2-audit] root={root}")
    sampling_failures: list[str] = []
    for profile_dir in profile_directories:
        files = sorted(profile_dir.rglob("taq_velocity_chunk_*.pt"))
        samples = 0
        valid_frames = 0
        range_invalid_frames = 0
        non_observed_frames = 0
        source_counts: Counter[str] = Counter()
        event_counts: Counter[str] = Counter()
        window_severity_counts: Counter[str] = Counter()
        episode_severity_counts: Counter[str] = Counter()
        episode_directories: set[Path] = set()
        dt_values: list[torch.Tensor] = []
        finite_difference_norms: list[torch.Tensor] = []
        label_speeds: list[torch.Tensor] = []
        all_finite = True
        schemas: set[str] = set()
        metadata_profiles: set[str] = set()
        environment_seeds: set[str] = set()
        configured_min_timestamp_dt_s: set[float] = set()
        configured_window_caps: set[int] = set()
        episode_sample_counts: Counter[Path] = Counter()
        episode_window_intervals: dict[Path, list[tuple[float, float]]] = defaultdict(list)
        window_timestamp_samples = 0
        raycast_eligible_frames = 0
        raycast_hit_frames = 0
        raycast_abs_error_sum = 0.0
        raycast_abs_error_p95_max = 0.0
        raycast_abs_error_max = 0.0
        raycast_rel_error_sum = 0.0
        raycast_rel_error_p95_max = 0.0
        raycast_rel_error_max = 0.0

        for path in files:
            payload = _load(path)
            features = torch.as_tensor(payload["sequence_features"])
            sources = torch.as_tensor(payload["sequence_source_ids"], dtype=torch.long)
            valid = torch.as_tensor(payload["sequence_valid_mask"], dtype=torch.bool)
            labels = torch.as_tensor(payload["target_velocity_base"], dtype=torch.float32)
            if features.ndim != 3 or features.shape[-1] != 24:
                raise ValueError(f"Unexpected feature shape in {path}: {tuple(features.shape)}")
            if sources.shape != features.shape[:2] or valid.shape != features.shape[:2]:
                raise ValueError(f"Mask/source shape mismatch in {path}")

            samples += int(features.shape[0])
            episode_sample_counts[path.parent] += int(features.shape[0])
            valid_frames += int(valid.sum().item())
            all_finite &= bool(torch.isfinite(features).all().item())
            range_invalid_frames += int(((features[..., 23] < 0.5) & valid).sum().item())
            non_observed_frames += int(((features[..., 22] < 0.5) & valid).sum().item())
            for source_id, source_name in SOURCE_NAMES.items():
                source_counts[source_name] += int(
                    ((sources == source_id) & valid).sum().item()
                )
            positive_dt = features[..., 12][valid & (features[..., 12] > 0.0)]
            if positive_dt.numel():
                dt_values.append(positive_dt.to(torch.float32))
            finite_difference_norms.append(
                torch.linalg.vector_norm(features[..., 3:6][valid], dim=-1).to(torch.float32)
            )
            if labels.ndim == 2:
                label_speeds.append(torch.linalg.vector_norm(labels, dim=-1))
            elif labels.ndim == 3:
                label_speeds.append(torch.linalg.vector_norm(labels[:, -1], dim=-1))

            schemas.add(str(payload.get("schema_version", "missing")))
            metadata = payload.get("metadata", {})
            environment_seeds.add(str(metadata.get("environment_seed", "missing")))
            degradation_config = metadata.get("degradation", {})
            configured_min_dt = degradation_config.get("min_timestamp_dt_s")
            if configured_min_dt is not None:
                configured_min_timestamp_dt_s.add(float(configured_min_dt))
            configured_cap = metadata.get("max_windows_per_episode")
            if configured_cap is not None:
                configured_window_caps.add(int(configured_cap))
            has_window_start = "sample_window_start_s" in payload
            has_window_end = "sample_window_end_s" in payload
            if has_window_start != has_window_end:
                raise ValueError(f"Incomplete sample window timestamps in {path}")
            if has_window_start:
                window_start = torch.as_tensor(
                    payload["sample_window_start_s"], dtype=torch.float64
                )
                window_end = torch.as_tensor(
                    payload["sample_window_end_s"], dtype=torch.float64
                )
                expected_shape = (features.shape[0],)
                if window_start.shape != expected_shape or window_end.shape != expected_shape:
                    raise ValueError(f"Sample window timestamp shape mismatch in {path}")
                if (
                    not torch.isfinite(window_start).all()
                    or not torch.isfinite(window_end).all()
                    or torch.any(window_end <= window_start)
                ):
                    raise ValueError(f"Invalid sample window timestamps in {path}")
                episode_window_intervals[path.parent].extend(
                    zip(window_start.tolist(), window_end.tolist())
                )
                window_timestamp_samples += int(features.shape[0])
            degradation = metadata.get("degradation_episode", {})
            metadata_profiles.add(str(degradation.get("profile", "missing")))
            episode_severity = str(degradation.get("episode_severity", "missing"))
            window_severity_counts[episode_severity] += int(features.shape[0])
            if path.parent not in episode_directories:
                episode_directories.add(path.parent)
                episode_severity_counts[episode_severity] += 1
                event_counts.update(degradation.get("event_counts", {}))
                raycast_audit = metadata.get("raycaster_vs_gt_audit", {})
                eligible = int(raycast_audit.get("eligible_tracker_frames", 0))
                hits = int(raycast_audit.get("raycast_hit_frames", 0))
                absolute = raycast_audit.get("absolute_distance_error_m") or {}
                relative = raycast_audit.get("relative_distance_error") or {}
                raycast_eligible_frames += eligible
                raycast_hit_frames += hits
                raycast_abs_error_sum += hits * float(absolute.get("mean", 0.0))
                raycast_abs_error_p95_max = max(
                    raycast_abs_error_p95_max, float(absolute.get("p95", 0.0))
                )
                raycast_abs_error_max = max(
                    raycast_abs_error_max, float(absolute.get("max", 0.0))
                )
                raycast_rel_error_sum += hits * float(relative.get("mean", 0.0))
                raycast_rel_error_p95_max = max(
                    raycast_rel_error_p95_max, float(relative.get("p95", 0.0))
                )
                raycast_rel_error_max = max(
                    raycast_rel_error_max, float(relative.get("max", 0.0))
                )

        dt = torch.cat(dt_values) if dt_values else torch.empty(0)
        finite_difference = (
            torch.cat(finite_difference_norms) if finite_difference_norms else torch.empty(0)
        )
        speeds = torch.cat(label_speeds) if label_speeds else torch.empty(0)
        window_overlap_count = 0
        window_gaps_s: list[float] = []
        window_durations_s: list[float] = []
        for intervals in episode_window_intervals.values():
            ordered = sorted(intervals)
            previous_end: float | None = None
            for window_start, window_end in ordered:
                window_durations_s.append(window_end - window_start)
                if previous_end is not None:
                    if window_start <= previous_end:
                        window_overlap_count += 1
                    else:
                        window_gaps_s.append(window_start - previous_end)
                previous_end = (
                    window_end if previous_end is None else max(previous_end, window_end)
                )
        max_windows_in_episode = max(episode_sample_counts.values(), default=0)
        min_windows_in_episode = min(episode_sample_counts.values(), default=0)
        mean_windows_in_episode = samples / max(len(episode_sample_counts), 1)
        timestamp_metadata_complete = window_timestamp_samples == samples
        cap_ok = bool(configured_window_caps) and all(
            max_windows_in_episode <= configured_cap
            for configured_cap in configured_window_caps
        )
        non_overlap_ok = timestamp_metadata_complete and window_overlap_count == 0
        relative_name = profile_dir.relative_to(root)
        print(f"\n[{relative_name}]")
        print(f"  chunks / episodes / samples : {len(files)} / {len(episode_directories)} / {samples}")
        print(f"  schemas                     : {sorted(schemas)}")
        print(f"  metadata profiles           : {sorted(metadata_profiles)}")
        print(f"  environment seeds           : {sorted(environment_seeds)}")
        print(f"  all feature values finite   : {all_finite}")
        if timestamp_metadata_complete:
            configured_cap_text = (
                str(next(iter(configured_window_caps)))
                if len(configured_window_caps) == 1
                else str(sorted(configured_window_caps))
                if len(configured_window_caps) > 1
                else "missing"
            )
            minimum_gap_text = (
                f"{min(window_gaps_s):.5f} s" if window_gaps_s else "n/a"
            )
            duration_tensor = torch.tensor(window_durations_s, dtype=torch.float64)
            print(
                "  windows/episode min/mean/max : "
                f"{min_windows_in_episode} / {mean_windows_in_episode:.2f} / "
                f"{max_windows_in_episode} (cap={configured_cap_text})"
            )
            print(
                "  window overlaps / min gap    : "
                f"{window_overlap_count} / {minimum_gap_text}"
            )
            print(
                "  window duration min/mean/max : "
                f"{float(duration_tensor.min()):.5f} / "
                f"{float(duration_tensor.mean()):.5f} / "
                f"{float(duration_tensor.max()):.5f} s"
            )
            print(
                "  sampling constraints valid   : "
                f"{non_overlap_ok and cap_ok}"
            )
            if not (non_overlap_ok and cap_ok):
                sampling_failures.append(str(relative_name))
        else:
            print(
                "  window sampling audit        : unavailable "
                f"({window_timestamp_samples}/{samples} samples have timestamps)"
            )
            if window_timestamp_samples:
                sampling_failures.append(str(relative_name))
        print(f"  source frame counts         : {dict(source_counts)}")
        print(f"  severity episode counts     : {dict(episode_severity_counts)}")
        print(f"  severity window counts      : {dict(window_severity_counts)}")
        print(
            "  range-invalid / non-observed: "
            f"{range_invalid_frames / max(valid_frames, 1):.2%} / "
            f"{non_observed_frames / max(valid_frames, 1):.2%}"
        )
        if dt.numel():
            minimum_dt = float(dt.min())
            print(
                "  dt min / mean / p95 / max (s): "
                f"{minimum_dt:.5f} / {float(dt.mean()):.5f} / "
                f"{float(torch.quantile(dt, 0.95)):.5f} / {float(dt.max()):.5f}"
            )
            if configured_min_timestamp_dt_s:
                floor_ok = all(
                    minimum_dt + 1e-6 >= configured_floor
                    for configured_floor in configured_min_timestamp_dt_s
                )
                print(
                    "  configured dt floor / valid : "
                    f"{sorted(configured_min_timestamp_dt_s)} / {floor_ok}"
                )
        if finite_difference.numel():
            print(
                "  finite-difference norm p50/p95/max (m/s): "
                f"{float(torch.quantile(finite_difference, 0.50)):.3f} / "
                f"{float(torch.quantile(finite_difference, 0.95)):.3f} / "
                f"{float(finite_difference.max()):.3f}"
            )
        if speeds.numel():
            print(
                "  label speed mean / p95 (m/s) : "
                f"{float(speeds.mean()):.4f} / {float(torch.quantile(speeds, 0.95)):.4f}"
            )
        if raycast_eligible_frames:
            print(
                "  RayCaster vs GT (audit only): "
                f"hit={raycast_hit_frames / raycast_eligible_frames:.2%}, "
                f"abs_mean={raycast_abs_error_sum / max(raycast_hit_frames, 1):.3f} m, "
                f"worst_episode_abs_p95={raycast_abs_error_p95_max:.3f} m, "
                f"abs_max={raycast_abs_error_max:.3f} m"
            )
            print(
                "  RayCaster relative error      : "
                f"mean={raycast_rel_error_sum / max(raycast_hit_frames, 1):.3%}, "
                f"worst_episode_p95={raycast_rel_error_p95_max:.3%}, "
                f"max={raycast_rel_error_max:.3%}"
            )
        print(f"  degradation event counts    : {dict(event_counts)}")

    if sampling_failures:
        raise ValueError(
            "Window sampling constraints failed for: "
            + ", ".join(sampling_failures)
        )


if __name__ == "__main__":
    main()

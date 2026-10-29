"""Plot error-event windows from ``analyze_error_sources.py`` output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--event-index", type=int, default=None, help="plot one event; default plots all selected events")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--before-s", type=float, default=.5)
    parser.add_argument("--after-s", type=float, default=1.0)
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("plot_error_events.py requires matplotlib in the active Python environment") from exc
    report = args.report_dir.expanduser().resolve()
    events = _read(report / "event_catalog.csv")
    frames = _read(report / "frame_metrics.csv")
    by_uid: dict[str, list[dict[str, str]]] = {}
    for row in frames:
        key = row.get("stream_uid") or row["episode_uid"]
        by_uid.setdefault(key, []).append(row)
    selected = events if args.event_index is None else [events[args.event_index]]
    output = report / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for number, event in enumerate(selected[:args.max_events]):
        peak = _float(event, "peak_timestamp_s")
        event_key = event.get("stream_uid") or event["episode_uid"]
        window = [row for row in by_uid.get(event_key, []) if peak - args.before_s <= _float(row, "timestamp_s") <= peak + args.after_s]
        if not window:
            continue
        window.sort(key=lambda item: _float(item, "timestamp_s"))
        t = [_float(row, "timestamp_s") - peak for row in window]
        fig, axes = plt.subplots(7, 1, figsize=(13, 17), sharex=True)
        for axis, color in zip("xyz", ("tab:blue", "tab:orange", "tab:green")):
            axes[0].plot(t, [_float(row, f"velocity_prediction_{axis}") for row in window], color=color, label=f"pred {axis}")
            axes[0].plot(t, [_float(row, f"velocity_gt_{axis}") for row in window], color=color, linestyle="--", alpha=.65, label=f"gt {axis}")
        axes[0].legend(ncol=3, fontsize=8)
        axes[0].set_ylabel("velocity (m/s)")
        axes[1].plot(t, [_float(row, "error_norm") for row in window], label="error")
        axes[1].plot(t, [_float(row, "prediction_jump") for row in window], label="prediction jump")
        axes[1].plot(t, [_float(row, "gt_jump") for row in window], label="GT jump", alpha=.7)
        threshold = _float(event, "threshold_mps")
        if threshold == threshold:
            axes[1].axhline(threshold, color="red", linestyle="--", alpha=.6)
        axes[1].legend(fontsize=8)
        axes[1].set_ylabel("m/s")
        axes[2].plot(t, [_float(row, "direction_angular_error_deg") for row in window], label="direction error (deg)")
        axes[2].plot(t, [_float(row, "bbox_iou_gt") for row in window], label="bbox IoU")
        axes[2].plot(t, [_float(row, "fine_correction_center_px") for row in window], label="fine correction (px)")
        axes[2].legend(fontsize=8)
        axes[3].plot(t, [_float(row, "range_measured_m") - _float(row, "range_gt_m") for row in window], label="range error (m)")
        axes[3].plot(t, [_float(row, "position_error") for row in window], label="position error (m)")
        axes[3].plot(t, [_float(row, "kinematic_residual") for row in window], label="kinematic residual (m/s)")
        axes[3].legend(fontsize=8)
        axes[4].plot(t, [_float(row, "robot_angular_speed_rps") for row in window], label="robot angular speed")
        axes[4].plot(t, [_float(row, "robot_angular_acceleration_rps2") for row in window], label="robot angular acceleration")
        axes[4].plot(t, [_float(row, "background_global_flow_px") for row in window], label="background flow")
        axes[4].legend(fontsize=8)
        axes[4].set_ylabel("body / image jitter")
        axes[5].plot(t, [_float(row, "sigma_x") for row in window], label="sigma x")
        axes[5].plot(t, [_float(row, "sigma_y") for row in window], label="sigma y")
        axes[5].plot(t, [_float(row, "sigma_z") for row in window], label="sigma z")
        axes[5].plot(t, [_float(row, "reliability_gate") for row in window], label="gate")
        axes[5].plot(t, [_float(row, "finite_difference_reliability") for row in window], label="FD reliability")
        axes[5].legend(fontsize=8)
        axes[6].step(t, [_float(row, "source_id") for row in window], where="post", label="source id")
        axes[6].step(t, [_float(row, "state_reset") for row in window], where="post", label="state reset")
        axes[6].step(t, [_float(row, "measurement_discontinuity") for row in window], where="post", label="measurement discontinuity")
        axes[6].legend(fontsize=8)
        axes[6].set_ylabel("source / reset")
        axes[6].set_xlabel("time from peak (s)")
        fig.suptitle(
            f"{event['episode_uid']}  {event.get('event_type', 'event')}  "
            f"peak error={event['peak_error_norm']} m/s"
        )
        fig.tight_layout()
        fig.savefig(output / f"event_{number:04d}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()

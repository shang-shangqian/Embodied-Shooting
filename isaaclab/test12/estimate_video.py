from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test12.camera_geometry import CameraIntrinsics

from test12.ostrack_tracker import OSTrackBoxTracker
from test12.pipeline import (
    estimate_target_direction_from_boxes,
    load_rgb_paths,
    track_timing_stats,
    tracked_boxes_to_map,
    write_estimates_csv,
    write_overlays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate target direction from an initial box and OSTrack bbox tracking. "
            "The output CSV is compatible with test12.evaluate_direction."
        )
    )
    parser.add_argument("--rgb_dir", required=True, type=Path, help="Directory of sorted RGB frames.")
    parser.add_argument("--out_csv", type=Path, default=Path("test12_outputs/estimates.csv"))
    parser.add_argument("--overlay_dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_frames", type=int, default=None)

    intrinsics = parser.add_argument_group("camera intrinsics")
    intrinsics.add_argument("--intrinsics_json", type=Path, default=None)
    intrinsics.add_argument("--fx", type=float, default=None)
    intrinsics.add_argument("--fy", type=float, default=None)
    intrinsics.add_argument("--cx", type=float, default=None)
    intrinsics.add_argument("--cy", type=float, default=None)
    intrinsics.add_argument("--width", type=int, default=None)
    intrinsics.add_argument("--height", type=int, default=None)

    tracker = parser.add_argument_group("OSTrack")
    tracker.add_argument(
        "--initial_box",
        nargs=4,
        type=float,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Initial target box in the first RGB frame.",
    )
    tracker.add_argument(
        "--initial_box_file",
        type=Path,
        default=None,
        help="Text file containing x1 y1 x2 y2. Used when --initial_box is omitted.",
    )
    tracker.add_argument(
        "--ostrack_repo",
        required=True,
        type=Path,
        help="Path to the official botaoye/OSTrack repository.",
    )
    tracker.add_argument(
        "--ostrack_checkpoint",
        required=True,
        type=Path,
        help="Path to an OSTrack .pth.tar checkpoint.",
    )
    tracker.add_argument(
        "--ostrack_config",
        type=str,
        default="vitb_256_mae_ce_32x4_ep300",
        help="OSTrack yaml name under experiments/ostrack or an explicit yaml path.",
    )
    tracker.add_argument("--device", type=str, default="cuda:0")
    tracker.add_argument("--debug", type=int, default=0)

    return parser.parse_args()


def build_intrinsics(args: argparse.Namespace) -> CameraIntrinsics:
    if args.intrinsics_json is not None:
        return CameraIntrinsics.from_json(args.intrinsics_json)

    missing = [
        name
        for name in ("fx", "fy", "cx", "cy")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Provide --intrinsics_json or all of --fx --fy --cx --cy. "
            f"Missing: {', '.join(missing)}"
        )
    return CameraIntrinsics.from_args(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=args.width,
        height=args.height,
    )


def load_initial_box(args: argparse.Namespace) -> list[float]:
    if args.initial_box is not None:
        return [float(v) for v in args.initial_box]
    if args.initial_box_file is None:
        raise ValueError("--initial_box or --initial_box_file is required.")
    text = args.initial_box_file.read_text(encoding="utf-8").strip()
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError(f"{args.initial_box_file} must contain exactly four numbers.")
    return values


def main() -> None:
    args = parse_args()
    intrinsics = build_intrinsics(args)
    initial_box = load_initial_box(args)
    rgb_paths = load_rgb_paths(args.rgb_dir)
    if args.max_frames is not None:
        rgb_paths = rgb_paths[: int(args.max_frames)]

    tracker = OSTrackBoxTracker(
        repo_path=args.ostrack_repo,
        checkpoint=args.ostrack_checkpoint,
        config=args.ostrack_config,
        device=args.device,
        debug=args.debug,
    )
    tracks = list(tracker.track(rgb_paths=rgb_paths, initial_box_xyxy=initial_box))
    tracks_by_frame = tracked_boxes_to_map(tracks)
    estimates = estimate_target_direction_from_boxes(
        rgb_paths=rgb_paths,
        tracks_by_frame=tracks_by_frame,
        intrinsics=intrinsics,
        fps=args.fps,
    )
    write_estimates_csv(estimates, args.out_csv)

    if args.overlay_dir is not None:
        write_overlays(rgb_paths, tracks_by_frame, estimates, args.overlay_dir)

    valid_count = sum(int(e.valid_measurement) for e in estimates)
    ok_count = sum(int(t.ok and t.bbox_xyxy is not None) for t in tracks)
    timing = track_timing_stats(tracks)
    print(f"[test12] mode=ostrack_rgb_direction frames={len(estimates)} boxes={ok_count} valid={valid_count}")
    print(
        "[test12] timing",
        f"total={timing['total_track_time_s']:.3f}s",
        f"mean={timing['mean_track_time_ms']:.2f}ms",
        f"median={timing['median_track_time_ms']:.2f}ms",
        f"p95={timing['p95_track_time_ms']:.2f}ms",
        f"fps={timing['track_fps']:.2f}",
    )
    print(f"[test12] wrote {args.out_csv}")
    if args.overlay_dir is not None:
        print(f"[test12] wrote overlays to {args.overlay_dir}")


if __name__ == "__main__":
    main()

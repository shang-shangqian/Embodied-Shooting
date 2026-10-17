from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.multi_target_io import (
    load_initial_targets,
    write_multi_target_csv,
    write_multi_target_overlays,
)
from test_lk_ostrack.multi_target_tracker import MultiTargetConfig, MultiTargetVisionTracker
from test_lk_ostrack.ostrack_tracker import OSTrackBoxTracker, OSTrackFrameTracker
from test_lk_ostrack.pipeline import (
    estimate_target_direction_from_boxes,
    load_rgb_paths,
    track_timing_stats,
    tracked_boxes_to_map,
    write_estimates_csv,
    write_overlays,
)


BUNDLED_OSTRACK_REPO = Path(__file__).resolve().parent / "ostrack" / "OSTrack-main"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track one target with OSTrack, or multiple target IDs with shared sparse "
            "optical flow and selected-target OSTrack refinement."
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
        "--initial_boxes_file",
        type=Path,
        default=None,
        help=(
            "Enable multi-target mode. JSON or text file containing multiple IDs and "
            "x1 y1 x2 y2 boxes."
        ),
    )
    tracker.add_argument(
        "--ostrack_repo",
        required=False,
        type=Path,
        default=BUNDLED_OSTRACK_REPO,
        help="Path to the official botaoye/OSTrack repository.",
    )
    tracker.add_argument(
        "--ostrack_checkpoint",
        required=False,
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

    multi = parser.add_argument_group("multi-target policy")
    multi.add_argument("--disable_fine_tracking", action="store_true")
    multi.add_argument("--fb_max_error_px", type=float, default=1.5)
    multi.add_argument("--ransac_threshold_px", type=float, default=2.5)
    multi.add_argument("--corner_redetect_interval_frames", type=int, default=3)
    multi.add_argument("--max_missed_frames", type=int, default=8)
    multi.add_argument("--prediction_horizon_s", type=float, default=0.08)
    multi.add_argument("--target_width_m", type=float, default=0.70)
    multi.add_argument("--target_height_m", type=float, default=0.70)
    multi.add_argument("--kinematic_estimation_horizon_s", type=float, default=0.10)
    multi.add_argument("--priority_confidence_weight", type=float, default=0.35)
    multi.add_argument("--priority_center_weight", type=float, default=0.20)
    multi.add_argument("--priority_area_weight", type=float, default=0.05)
    multi.add_argument("--priority_stability_weight", type=float, default=0.10)
    multi.add_argument("--priority_distance_weight", type=float, default=0.20)
    multi.add_argument("--priority_speed_weight", type=float, default=0.10)
    multi.add_argument("--priority_distance_near_m", type=float, default=80.0)
    multi.add_argument("--priority_distance_far_m", type=float, default=150.0)
    multi.add_argument("--priority_speed_reference_mps", type=float, default=5.0)
    multi.add_argument("--priority_switch_margin", type=float, default=0.08)
    multi.add_argument("--fine_tracker_interval_frames", type=int, default=3)
    multi.add_argument("--fine_tracker_min_score", type=float, default=0.20)
    multi.add_argument(
        "--fine_tracker_policy", choices=("fixed", "adaptive"), default="fixed"
    )
    multi.add_argument("--adaptive_min_interval_frames", type=int, default=2)
    multi.add_argument("--adaptive_max_interval_frames", type=int, default=12)
    multi.add_argument("--adaptive_bad_frames", type=int, default=2)
    multi.add_argument("--adaptive_warning_confidence", type=float, default=0.75)
    multi.add_argument("--adaptive_critical_confidence", type=float, default=0.55)
    multi.add_argument("--adaptive_risk_threshold", type=float, default=0.55)
    multi.add_argument("--adaptive_critical_risk_threshold", type=float, default=0.85)
    multi.add_argument("--adaptive_risk_clear_threshold", type=float, default=0.35)
    multi.add_argument("--adaptive_desired_inliers", type=int, default=9)
    multi.add_argument("--adaptive_innovation_critical", type=float, default=0.75)
    multi.add_argument("--adaptive_prediction_iou_threshold", type=float, default=0.55)
    multi.add_argument("--adaptive_failure_backoff_initial_frames", type=int, default=1)
    multi.add_argument("--adaptive_failure_backoff_max_frames", type=int, default=8)
    multi.add_argument("--fine_max_center_jump_box_diagonals", type=float, default=2.0)
    multi.add_argument("--fine_max_scale_change", type=float, default=2.5)

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
    if not math.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError("--fps must be finite and positive")
    uses_ostrack = args.initial_boxes_file is None or not args.disable_fine_tracking
    if uses_ostrack and (args.ostrack_repo is None or args.ostrack_checkpoint is None):
        raise ValueError(
            "--ostrack_repo and --ostrack_checkpoint are required unless multi-target "
            "mode uses --disable_fine_tracking"
        )
    intrinsics = build_intrinsics(args)
    rgb_paths = load_rgb_paths(args.rgb_dir)
    if args.max_frames is not None:
        rgb_paths = rgb_paths[: int(args.max_frames)]

    if not rgb_paths:
        raise ValueError("No RGB frames selected")

    if args.initial_boxes_file is not None:
        run_multi_target(args, intrinsics, rgb_paths)
        return

    initial_box = load_initial_box(args)
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


def run_multi_target(
    args: argparse.Namespace,
    intrinsics: CameraIntrinsics,
    rgb_paths: list[Path],
) -> None:
    boxes, priority_biases = load_initial_targets(args.initial_boxes_file)
    fine_tracker = None
    if not args.disable_fine_tracking:
        fine_tracker = OSTrackFrameTracker(
            repo_path=args.ostrack_repo,
            checkpoint=args.ostrack_checkpoint,
            config=args.ostrack_config,
            device=args.device,
            debug=args.debug,
        )
        # Report installation/device/checkpoint errors before the scheduler's
        # per-frame recovery logic can treat them as temporary tracking loss.
        fine_tracker.prepare()
    config = MultiTargetConfig(
        fb_max_error_px=args.fb_max_error_px,
        ransac_reproj_threshold_px=args.ransac_threshold_px,
        corner_redetect_interval_frames=args.corner_redetect_interval_frames,
        max_missed_frames=args.max_missed_frames,
        prediction_horizon_s=args.prediction_horizon_s,
        target_width_m=args.target_width_m,
        target_height_m=args.target_height_m,
        kinematic_estimation_horizon_s=args.kinematic_estimation_horizon_s,
        priority_confidence_weight=args.priority_confidence_weight,
        priority_center_weight=args.priority_center_weight,
        priority_area_weight=args.priority_area_weight,
        priority_stability_weight=args.priority_stability_weight,
        priority_distance_weight=args.priority_distance_weight,
        priority_speed_weight=args.priority_speed_weight,
        priority_distance_near_m=args.priority_distance_near_m,
        priority_distance_far_m=args.priority_distance_far_m,
        priority_speed_reference_mps=args.priority_speed_reference_mps,
        priority_switch_margin=args.priority_switch_margin,
        fine_tracker_interval_frames=args.fine_tracker_interval_frames,
        fine_tracker_min_score=args.fine_tracker_min_score,
        fine_tracker_policy=args.fine_tracker_policy,
        adaptive_min_interval_frames=args.adaptive_min_interval_frames,
        adaptive_max_interval_frames=args.adaptive_max_interval_frames,
        adaptive_bad_frames=args.adaptive_bad_frames,
        adaptive_warning_confidence=args.adaptive_warning_confidence,
        adaptive_critical_confidence=args.adaptive_critical_confidence,
        adaptive_risk_threshold=args.adaptive_risk_threshold,
        adaptive_critical_risk_threshold=args.adaptive_critical_risk_threshold,
        adaptive_risk_clear_threshold=args.adaptive_risk_clear_threshold,
        adaptive_desired_inliers=args.adaptive_desired_inliers,
        adaptive_innovation_critical=args.adaptive_innovation_critical,
        adaptive_prediction_iou_threshold=args.adaptive_prediction_iou_threshold,
        adaptive_failure_backoff_initial_frames=(
            args.adaptive_failure_backoff_initial_frames
        ),
        adaptive_failure_backoff_max_frames=args.adaptive_failure_backoff_max_frames,
        fine_max_center_jump_box_diagonals=args.fine_max_center_jump_box_diagonals,
        fine_max_scale_change=args.fine_max_scale_change,
    )
    tracker = MultiTargetVisionTracker(
        intrinsics,
        config=config,
        fine_tracker=fine_tracker,
    )
    first_image = read_rgb(rgb_paths[0])
    results = [
        tracker.initialize(
            first_image,
            boxes,
            priority_biases=priority_biases,
            timestamp_s=0.0,
        )
    ]
    dt_s = 1.0 / float(args.fps)
    for frame_index, rgb_path in enumerate(rgb_paths[1:], start=1):
        results.append(
            tracker.update(
                read_rgb(rgb_path),
                dt_s=dt_s,
                timestamp_s=frame_index * dt_s,
            )
        )

    write_multi_target_csv(results, args.out_csv)
    if args.overlay_dir is not None:
        write_multi_target_overlays(rgb_paths, results, args.overlay_dir)
    selected_frames = sum(result.selected_target_id is not None for result in results)
    fine_attempts = sum(
        int(target.fine_attempted)
        for result in results
        for target in result.targets
    )
    fine_applied = sum(
        int(target.fine_applied)
        for result in results
        for target in result.targets
    )
    print(
        "[multi_target]",
        f"frames={len(results)}",
        f"targets={len(boxes)}",
        f"selected_frames={selected_frames}",
        f"fine_tracking={int(fine_tracker is not None)}",
        f"fine_policy={config.fine_tracker_policy}",
        f"fine_attempts={fine_attempts}",
        f"fine_applied={fine_applied}",
    )
    print(f"[multi_target] wrote {args.out_csv}")
    if args.overlay_dir is not None:
        print(f"[multi_target] wrote overlays to {args.overlay_dir}")


def read_rgb(path: Path):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for multi-target video tracking") from exc
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()

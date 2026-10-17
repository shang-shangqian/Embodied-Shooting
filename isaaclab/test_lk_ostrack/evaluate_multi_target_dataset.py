from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import cv2
import numpy as np

import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "test_lk_ostrack"

from .camera_geometry import CameraIntrinsics
from .multi_target_io import load_initial_targets
from .multi_target_tracker import (
    MultiTargetConfig,
    MultiTargetFrameResult,
    MultiTargetVisionTracker,
    TargetEstimate,
    TargetKinematics,
)
from .ostrack_tracker import OSTrackFrameTracker


BUNDLED_OSTRACK_REPO = Path(__file__).resolve().parent / "ostrack" / "OSTrack-main"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the multi-target LK/RANSAC/Kalman/OSTrack pipeline against "
            "an Isaac Sim collection dataset without writing visualization images."
        )
    )
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=10)

    speed = parser.add_argument_group("speed-range analysis")
    speed.add_argument("--speed_bin_width_mps", type=float, default=0.5)
    speed.add_argument("--speed_min_observations", type=int, default=1000)
    speed.add_argument("--speed_min_tracking_availability", type=float, default=0.95)
    speed.add_argument("--speed_min_success_iou_0_5", type=float, default=0.80)
    speed.add_argument("--speed_max_angular_p95_deg", type=float, default=1.5)

    position = parser.add_argument_group("bearing/range position fusion")
    position.add_argument(
        "--position_range_adjustment_m",
        type=float,
        default=0.0,
        help=(
            "Constant added to recorded lidar_range_m before position fusion. "
            "Use a negative value to audit removal of a known center correction."
        ),
    )

    fine = parser.add_argument_group("OSTrack refinement")
    fine.add_argument("--disable_fine_tracking", action="store_true")
    fine.add_argument("--ostrack_repo", type=Path, default=BUNDLED_OSTRACK_REPO)
    fine.add_argument("--ostrack_checkpoint", type=Path, default=None)
    fine.add_argument(
        "--ostrack_config",
        type=str,
        default="vitb_256_mae_ce_32x4_ep300",
    )
    fine.add_argument("--device", type=str, default="cuda:0")
    fine.add_argument("--ostrack_debug", type=int, default=0)

    tracker = parser.add_argument_group("multi-target tracker")
    tracker.add_argument("--fb_max_error_px", type=float, default=1.5)
    tracker.add_argument("--ransac_threshold_px", type=float, default=2.5)
    tracker.add_argument("--corner_redetect_interval_frames", type=int, default=3)
    tracker.add_argument("--max_missed_frames", type=int, default=8)
    tracker.add_argument("--prediction_horizon_s", type=float, default=0.08)
    tracker.add_argument("--target_width_m", type=float, default=None)
    tracker.add_argument("--target_height_m", type=float, default=None)
    tracker.add_argument("--kinematic_estimation_horizon_s", type=float, default=0.10)
    tracker.add_argument("--priority_confidence_weight", type=float, default=0.35)
    tracker.add_argument("--priority_center_weight", type=float, default=0.20)
    tracker.add_argument("--priority_area_weight", type=float, default=0.05)
    tracker.add_argument("--priority_stability_weight", type=float, default=0.10)
    tracker.add_argument("--priority_distance_weight", type=float, default=0.20)
    tracker.add_argument("--priority_speed_weight", type=float, default=0.10)
    tracker.add_argument("--priority_distance_near_m", type=float, default=None)
    tracker.add_argument("--priority_distance_far_m", type=float, default=None)
    tracker.add_argument("--priority_speed_reference_mps", type=float, default=None)
    tracker.add_argument(
        "--kinematics_source",
        choices=("ground_truth", "vision"),
        default="ground_truth",
    )
    tracker.add_argument("--priority_switch_margin", type=float, default=0.08)
    tracker.add_argument("--fine_tracker_interval_frames", type=int, default=3)
    tracker.add_argument("--fine_tracker_min_score", type=float, default=0.20)
    tracker.add_argument(
        "--fine_tracker_policy", choices=("fixed", "adaptive"), default="fixed"
    )
    tracker.add_argument("--adaptive_min_interval_frames", type=int, default=2)
    tracker.add_argument("--adaptive_max_interval_frames", type=int, default=12)
    tracker.add_argument("--adaptive_bad_frames", type=int, default=2)
    tracker.add_argument("--adaptive_warning_confidence", type=float, default=0.75)
    tracker.add_argument("--adaptive_critical_confidence", type=float, default=0.55)
    tracker.add_argument("--adaptive_risk_threshold", type=float, default=0.55)
    tracker.add_argument("--adaptive_critical_risk_threshold", type=float, default=0.85)
    tracker.add_argument("--adaptive_risk_clear_threshold", type=float, default=0.35)
    tracker.add_argument("--adaptive_desired_inliers", type=int, default=9)
    tracker.add_argument("--adaptive_innovation_critical", type=float, default=0.75)
    tracker.add_argument("--adaptive_prediction_iou_threshold", type=float, default=0.55)
    tracker.add_argument("--adaptive_failure_backoff_initial_frames", type=int, default=1)
    tracker.add_argument("--adaptive_failure_backoff_max_frames", type=int, default=8)
    tracker.add_argument("--fine_max_center_jump_box_diagonals", type=float, default=2.0)
    tracker.add_argument("--fine_max_scale_change", type=float, default=2.5)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_sequences is not None and int(args.max_sequences) <= 0:
        raise ValueError("--max_sequences must be positive")
    if args.max_frames is not None and int(args.max_frames) <= 0:
        raise ValueError("--max_frames must be positive")
    if int(args.progress_every) <= 0:
        raise ValueError("--progress_every must be positive")
    if float(args.speed_bin_width_mps) <= 0.0:
        raise ValueError("--speed_bin_width_mps must be positive")
    if int(args.speed_min_observations) <= 0:
        raise ValueError("--speed_min_observations must be positive")
    for name in (
        "speed_min_tracking_availability",
        "speed_min_success_iou_0_5",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1]")
    if float(args.speed_max_angular_p95_deg) <= 0.0:
        raise ValueError("--speed_max_angular_p95_deg must be positive")
    if not math.isfinite(float(args.position_range_adjustment_m)):
        raise ValueError("--position_range_adjustment_m must be finite")
    if not args.disable_fine_tracking:
        if args.ostrack_repo is None or args.ostrack_checkpoint is None:
            raise ValueError(
                "--ostrack_repo and --ostrack_checkpoint are required unless "
                "--disable_fine_tracking is set"
            )
        if not args.ostrack_repo.is_dir():
            raise FileNotFoundError(f"OSTrack repository not found: {args.ostrack_repo}")
        if not args.ostrack_checkpoint.is_file():
            raise FileNotFoundError(
                f"OSTrack checkpoint not found: {args.ostrack_checkpoint}"
            )


def _read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")
    complete = [row for row in rows if str(row.get("complete", "1")) == "1"]
    if not complete:
        raise ValueError(f"No complete sequences in {manifest_path}")
    return complete


def _read_ground_truth(
    path: Path,
    *,
    intrinsics: CameraIntrinsics | None = None,
    target_radius_m: float | None = None,
    view_margin_px: float = 0.0,
) -> dict[tuple[int, int], dict[str, float | bool]]:
    """Read simulation labels, reconstructing image truth from target position.

    Recent collection datasets intentionally store online visual measurements in
    the bbox/direction columns.  The simulation target position and velocity
    remain supervision labels.  Reprojecting ``target_pos_cam`` prevents the
    evaluator from treating the collector's LK output as ground truth.  The
    fallback keeps compatibility with older or synthetic CSV files that do not
    contain camera-frame position columns.
    """
    ground_truth: dict[tuple[int, int], dict[str, float | bool]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frame_index = int(row["frame_index"])
            target_id = int(row["target_id"])
            key = (frame_index, target_id)
            if key in ground_truth:
                raise ValueError(f"Duplicate ground-truth key {key} in {path}")
            reconstruct = (
                intrinsics is not None
                and target_radius_m is not None
                and all(name in row for name in ("x_cam_m", "y_cam_m", "z_cam_m"))
            )
            if reconstruct:
                x = float(row["x_cam_m"])
                y = float(row["y_cam_m"])
                z = float(row["z_cam_m"])
                z_safe = max(z, 1e-6)
                center_u = intrinsics.fx * x / z_safe + intrinsics.cx
                center_v = intrinsics.fy * y / z_safe + intrinsics.cy
                radius_x = intrinsics.fx * float(target_radius_m) / z_safe
                radius_y = intrinsics.fy * float(target_radius_m) / z_safe
                raw_bbox = (
                    center_u - radius_x,
                    center_v - radius_y,
                    center_u + radius_x,
                    center_v + radius_y,
                )
                width = intrinsics.width
                height = intrinsics.height
                if width is None or height is None:
                    raise ValueError(
                        "Image width/height are required to reconstruct simulation truth"
                    )
                bbox = (
                    min(max(raw_bbox[0], 0.0), float(width)),
                    min(max(raw_bbox[1], 0.0), float(height)),
                    min(max(raw_bbox[2], 0.0), float(width)),
                    min(max(raw_bbox[3], 0.0), float(height)),
                )
                finite = all(
                    math.isfinite(value)
                    for value in (x, y, z, center_u, center_v, *raw_bbox)
                )
                intersects = (
                    raw_bbox[2] > 0.0
                    and raw_bbox[3] > 0.0
                    and raw_bbox[0] < float(width)
                    and raw_bbox[1] < float(height)
                )
                visible = (
                    finite
                    and z > 1e-4
                    and intersects
                    and radius_x >= 1.0
                    and radius_y >= 1.0
                )
                margin = max(0.0, float(view_margin_px))
                fully_visible = visible and (
                    raw_bbox[0] >= margin
                    and raw_bbox[1] >= margin
                    and raw_bbox[2] < float(width) - margin
                    and raw_bbox[3] < float(height) - margin
                )
                yaw_rad = math.atan2(x, z_safe)
                pitch_rad = math.atan2(-y, math.sqrt(x * x + z * z))
                distance_m = math.sqrt(x * x + y * y + z * z)
            else:
                bbox = tuple(
                    float(row[name])
                    for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
                )
                center_u = float(row["center_u"])
                center_v = float(row["center_v"])
                visible = bool(int(row["visible"]))
                fully_visible = bool(int(row["fully_visible"]))
                yaw_rad = float(row["yaw_rad"])
                pitch_rad = float(row["pitch_rad"])
                distance_m = float(row["distance_m"])
                truth_direction = _direction_vector(yaw_rad, pitch_rad)
                x, y, z = (truth_direction * distance_m).tolist()

            raw_lidar_range = row.get("lidar_range_m", "")
            try:
                lidar_range_m = float(raw_lidar_range)
            except (TypeError, ValueError):
                lidar_range_m = float("nan")
            raw_lidar_valid = row.get("lidar_valid", "0")
            try:
                lidar_valid = bool(int(raw_lidar_valid))
            except (TypeError, ValueError):
                lidar_valid = False
            lidar_valid = (
                lidar_valid
                and math.isfinite(lidar_range_m)
                and lidar_range_m >= 0.0
            )
            ground_truth[key] = {
                "visible": visible,
                "fully_visible": fully_visible,
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "center_u": center_u,
                "center_v": center_v,
                "yaw_rad": yaw_rad,
                "pitch_rad": pitch_rad,
                "distance_m": distance_m,
                "lidar_range_m": lidar_range_m,
                "lidar_valid": lidar_valid,
                "x_cam_m": x,
                "y_cam_m": y,
                "z_cam_m": z,
                "speed_m_s": math.sqrt(
                    float(row["vx_cam_mps"]) ** 2
                    + float(row["vy_cam_mps"]) ** 2
                    + float(row["vz_cam_mps"]) ** 2
                ),
            }
    return ground_truth


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _ground_truth_kinematics_by_frame(
    ground_truth: dict[tuple[int, int], dict[str, float | bool]],
) -> dict[int, dict[int, TargetKinematics]]:
    by_frame: dict[int, dict[int, TargetKinematics]] = {}
    for (frame_index, target_id), row in ground_truth.items():
        by_frame.setdefault(frame_index, {})[target_id] = TargetKinematics(
            distance_m=float(row["distance_m"]),
            speed_m_s=float(row["speed_m_s"]),
            source="ground_truth",
        )
    return by_frame


def _bbox_iou(prediction: Iterable[float], truth: Iterable[float]) -> float:
    px1, py1, px2, py2 = (float(value) for value in prediction)
    tx1, ty1, tx2, ty2 = (float(value) for value in truth)
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    truth_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = pred_area + truth_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _angle_difference(first: float, second: float) -> float:
    return (float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi


def _direction_vector(yaw: float, pitch: float) -> np.ndarray:
    cos_pitch = math.cos(float(pitch))
    return np.asarray(
        [
            cos_pitch * math.sin(float(yaw)),
            -math.sin(float(pitch)),
            cos_pitch * math.cos(float(yaw)),
        ],
        dtype=np.float64,
    )


def _angular_error_rad(
    predicted_yaw: float,
    predicted_pitch: float,
    truth_yaw: float,
    truth_pitch: float,
) -> float:
    first = _direction_vector(predicted_yaw, predicted_pitch)
    second = _direction_vector(truth_yaw, truth_pitch)
    return math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))


def _position_error_components(
    predicted_yaw: float,
    predicted_pitch: float,
    measured_range_m: float,
    truth_position_cam_m: Iterable[float],
) -> dict[str, float]:
    """Return camera-frame error components for bearing/range position fusion."""
    measured_range_m = float(measured_range_m)
    truth_position = np.asarray(list(truth_position_cam_m), dtype=np.float64)
    if truth_position.shape != (3,):
        raise ValueError(
            f"Expected truth_position_cam_m shape (3,), got {truth_position.shape}."
        )
    if not math.isfinite(measured_range_m) or measured_range_m < 0.0:
        raise ValueError("measured_range_m must be finite and non-negative")
    if not np.isfinite(truth_position).all():
        raise ValueError("truth_position_cam_m must be finite")
    truth_range_m = float(np.linalg.norm(truth_position))
    if truth_range_m <= 0.0:
        raise ValueError("truth_position_cam_m must have positive range")

    predicted_position = measured_range_m * _direction_vector(
        predicted_yaw, predicted_pitch
    )
    error = predicted_position - truth_position
    truth_direction = truth_position / truth_range_m
    radial_error_m = float(np.dot(error, truth_direction))
    cross_track_error_m = float(
        np.linalg.norm(error - radial_error_m * truth_direction)
    )
    return {
        "position_error_m": float(np.linalg.norm(error)),
        "position_relative_error": float(np.linalg.norm(error)) / truth_range_m,
        "x_error_m": float(error[0]),
        "y_error_m": float(error[1]),
        "z_error_m": float(error[2]),
        "radial_error_m": radial_error_m,
        "cross_track_error_m": cross_track_error_m,
        "range_error_m": measured_range_m - truth_range_m,
    }


def _fused_position_error(
    predicted_yaw: float,
    predicted_pitch: float,
    truth: dict[str, float | bool],
    range_adjustment_m: float = 0.0,
) -> dict[str, float] | None:
    """Fuse a predicted bearing with the recorded valid lidar range."""
    if not bool(truth.get("lidar_valid", False)):
        return None
    try:
        return _position_error_components(
            predicted_yaw,
            predicted_pitch,
            float(truth["lidar_range_m"]) + float(range_adjustment_m),
            (
                float(truth["x_cam_m"]),
                float(truth["y_cam_m"]),
                float(truth["z_cam_m"]),
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


class _MetricsAccumulator:
    def __init__(
        self,
        *,
        fps: float,
        prediction_horizon_s: float,
        position_range_adjustment_m: float = 0.0,
    ) -> None:
        self.fps = float(fps)
        self.horizon_frames = int(round(float(prediction_horizon_s) * self.fps))
        self.position_range_adjustment_m = float(position_range_adjustment_m)
        self.total_frames = 0
        self.target_observations = 0
        self.visible_observations = 0
        self.full_visible_observations = 0
        self.visible_trackable_observations = 0
        self.visible_lost_observations = 0
        self.trackable_observations = 0
        self.lost_observations = 0
        self.selected_frames = 0
        self.selection_switches = 0
        self.identity_confusions = 0
        self.identity_evaluations = 0
        self.identity_switch_events = 0
        self.status_counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.kinematics_source_counts: Counter[str] = Counter()
        self.fine_action_counts: Counter[str] = Counter()
        self.fine_trigger_reason_counts: Counter[str] = Counter()
        self.fine_initialize_count = 0
        self.fine_attempt_count = 0
        self.fine_applied_count = 0
        self.fine_initialize_latencies_ms: list[float] = []
        self.fine_track_latencies_ms: list[float] = []
        self.fine_risks: list[float] = []
        self.center_errors_px: list[float] = []
        self.normalized_center_errors: list[float] = []
        self.ious: list[float] = []
        self.yaw_errors_deg: list[float] = []
        self.pitch_errors_deg: list[float] = []
        self.angular_errors_deg: list[float] = []
        self.lookahead_angular_errors_deg: list[float] = []
        self.selected_angular_errors_deg: list[float] = []
        self.selected_lookahead_angular_errors_deg: list[float] = []
        self.distance_errors_m: list[float] = []
        self.distance_relative_errors: list[float] = []
        self.position_errors_m: list[float] = []
        self.position_relative_errors: list[float] = []
        self.position_x_errors_m: list[float] = []
        self.position_y_errors_m: list[float] = []
        self.position_z_errors_m: list[float] = []
        self.position_radial_errors_m: list[float] = []
        self.position_cross_track_errors_m: list[float] = []
        self.measured_range_errors_m: list[float] = []
        self.speed_errors_mps: list[float] = []
        self.confidences: list[float] = []
        self.corner_counts: list[float] = []
        self.ransac_inliers: list[float] = []
        self.fb_errors_px: list[float] = []
        self.update_times_s: list[float] = []
        self.initialize_times_s: list[float] = []

    def add_sequence(
        self,
        *,
        results: list[MultiTargetFrameResult],
        ground_truth: dict[tuple[int, int], dict[str, float | bool]],
        update_times_s: list[float],
        initialize_time_s: float,
    ) -> None:
        if not results:
            return
        self.total_frames += len(results)
        self.update_times_s.extend(update_times_s)
        self.initialize_times_s.append(float(initialize_time_s))
        previous_selected: int | None = None
        previous_identity: dict[int, int] = {}
        visible_ground_truth_by_frame: dict[
            int, dict[int, dict[str, float | bool]]
        ] = {}
        for (frame_index, target_id), row in ground_truth.items():
            if bool(row["visible"]):
                visible_ground_truth_by_frame.setdefault(frame_index, {})[target_id] = row

        for result in results:
            if result.selected_target_id is not None:
                self.selected_frames += 1
                if previous_selected is not None and result.selected_target_id != previous_selected:
                    self.selection_switches += 1
                previous_selected = result.selected_target_id
            visible_gt = visible_ground_truth_by_frame.get(result.frame_index, {})
            estimates_by_id = {
                estimate.target_id: estimate for estimate in result.targets
            }
            selected = estimates_by_id.get(result.selected_target_id)
            if selected is not None:
                self.fine_action_counts[selected.fine_action] += 1
                self.fine_trigger_reason_counts[selected.fine_trigger_reason] += 1
                self.fine_initialize_count += int(
                    selected.fine_action in {"initialize", "initialize_failed"}
                )
                self.fine_attempt_count += int(selected.fine_attempted)
                self.fine_applied_count += int(selected.fine_applied)
                self.fine_risks.append(float(selected.fine_risk))
                if selected.fine_latency_ms > 0.0:
                    if selected.fine_action in {"initialize", "initialize_failed"}:
                        self.fine_initialize_latencies_ms.append(
                            float(selected.fine_latency_ms)
                        )
                    elif selected.fine_attempted:
                        self.fine_track_latencies_ms.append(
                            float(selected.fine_latency_ms)
                        )
                selected_gt = ground_truth.get(
                    (result.frame_index, selected.target_id)
                )
                if selected_gt is not None and bool(selected_gt["visible"]):
                    self.selected_angular_errors_deg.append(
                        math.degrees(
                            _angular_error_rad(
                                selected.yaw_rad,
                                selected.pitch_rad,
                                float(selected_gt["yaw_rad"]),
                                float(selected_gt["pitch_rad"]),
                            )
                        )
                    )
                selected_future_gt = ground_truth.get(
                    (
                        result.frame_index + self.horizon_frames,
                        selected.target_id,
                    )
                )
                if selected_future_gt is not None and bool(selected_future_gt["visible"]):
                    selected_future_error = math.degrees(
                        _angular_error_rad(
                            selected.predicted_yaw_rad,
                            selected.predicted_pitch_rad,
                            float(selected_future_gt["yaw_rad"]),
                            float(selected_future_gt["pitch_rad"]),
                        )
                    )
                    self.selected_lookahead_angular_errors_deg.append(
                        selected_future_error
                    )

            for estimate in result.targets:
                self.target_observations += 1
                self.status_counts[estimate.status] += 1
                self.source_counts[estimate.source] += 1
                self.kinematics_source_counts[estimate.kinematics_source] += 1
                self.confidences.append(float(estimate.confidence))
                self.corner_counts.append(float(estimate.corners))
                self.ransac_inliers.append(float(estimate.ransac_inliers))
                if estimate.median_fb_error_px is not None and math.isfinite(
                    estimate.median_fb_error_px
                ):
                    self.fb_errors_px.append(float(estimate.median_fb_error_px))
                if estimate.status == "lost":
                    self.lost_observations += 1
                else:
                    self.trackable_observations += 1

                gt = ground_truth.get((result.frame_index, estimate.target_id))
                if gt is None or not bool(gt["visible"]):
                    continue
                self.visible_observations += 1
                if bool(gt["fully_visible"]):
                    self.full_visible_observations += 1
                if estimate.status == "lost":
                    self.visible_lost_observations += 1
                else:
                    self.visible_trackable_observations += 1

                truth_bbox = (
                    float(gt["bbox_x1"]),
                    float(gt["bbox_y1"]),
                    float(gt["bbox_x2"]),
                    float(gt["bbox_y2"]),
                )
                center_error = math.hypot(
                    estimate.center_xy[0] - float(gt["center_u"]),
                    estimate.center_xy[1] - float(gt["center_v"]),
                )
                truth_diagonal = max(
                    math.hypot(
                        truth_bbox[2] - truth_bbox[0],
                        truth_bbox[3] - truth_bbox[1],
                    ),
                    1e-6,
                )
                self.center_errors_px.append(center_error)
                self.normalized_center_errors.append(center_error / truth_diagonal)
                self.ious.append(_bbox_iou(estimate.bbox_xyxy, truth_bbox))
                yaw_error = math.degrees(
                    abs(_angle_difference(estimate.yaw_rad, float(gt["yaw_rad"])))
                )
                pitch_error = math.degrees(
                    abs(_angle_difference(estimate.pitch_rad, float(gt["pitch_rad"])))
                )
                angular_error = math.degrees(
                    _angular_error_rad(
                        estimate.yaw_rad,
                        estimate.pitch_rad,
                        float(gt["yaw_rad"]),
                        float(gt["pitch_rad"]),
                    )
                )
                self.yaw_errors_deg.append(yaw_error)
                self.pitch_errors_deg.append(pitch_error)
                self.angular_errors_deg.append(angular_error)
                distance_error = abs(estimate.distance_m - float(gt["distance_m"]))
                self.distance_errors_m.append(distance_error)
                self.distance_relative_errors.append(
                    distance_error / max(float(gt["distance_m"]), 1e-6)
                )
                position_error = _fused_position_error(
                    estimate.yaw_rad,
                    estimate.pitch_rad,
                    gt,
                    self.position_range_adjustment_m,
                )
                if position_error is not None:
                    self.position_errors_m.append(
                        position_error["position_error_m"]
                    )
                    self.position_relative_errors.append(
                        position_error["position_relative_error"]
                    )
                    self.position_x_errors_m.append(position_error["x_error_m"])
                    self.position_y_errors_m.append(position_error["y_error_m"])
                    self.position_z_errors_m.append(position_error["z_error_m"])
                    self.position_radial_errors_m.append(
                        position_error["radial_error_m"]
                    )
                    self.position_cross_track_errors_m.append(
                        position_error["cross_track_error_m"]
                    )
                    self.measured_range_errors_m.append(
                        position_error["range_error_m"]
                    )
                self.speed_errors_mps.append(
                    abs(estimate.speed_m_s - float(gt["speed_m_s"]))
                )

                future_gt = ground_truth.get(
                    (
                        result.frame_index + self.horizon_frames,
                        estimate.target_id,
                    )
                )
                if future_gt is not None and bool(future_gt["visible"]):
                    self.lookahead_angular_errors_deg.append(
                        math.degrees(
                            _angular_error_rad(
                                estimate.predicted_yaw_rad,
                                estimate.predicted_pitch_rad,
                                float(future_gt["yaw_rad"]),
                                float(future_gt["pitch_rad"]),
                            )
                        )
                    )

                if estimate.status != "lost" and visible_gt:
                    nearest_id = min(
                        visible_gt,
                        key=lambda candidate_id: math.hypot(
                            estimate.center_xy[0]
                            - float(visible_gt[candidate_id]["center_u"]),
                            estimate.center_xy[1]
                            - float(visible_gt[candidate_id]["center_v"]),
                        ),
                    )
                    self.identity_evaluations += 1
                    if nearest_id != estimate.target_id:
                        self.identity_confusions += 1
                    old_nearest = previous_identity.get(estimate.target_id, estimate.target_id)
                    if nearest_id != old_nearest:
                        self.identity_switch_events += 1
                    previous_identity[estimate.target_id] = nearest_id

    def summary(self) -> dict[str, Any]:
        latency_ms = [value * 1000.0 for value in self.update_times_s]
        init_ms = [value * 1000.0 for value in self.initialize_times_s]
        total_update_s = sum(self.update_times_s)
        angular_squared = [value * value for value in self.angular_errors_deg]
        position_squared = [value * value for value in self.position_errors_m]
        return {
            "frames": self.total_frames,
            "target_observations": self.target_observations,
            "visible_observations": self.visible_observations,
            "fully_visible_observations": self.full_visible_observations,
            "visible_tracking_availability": self.visible_trackable_observations
            / max(self.visible_observations, 1),
            "visible_lost_rate": self.visible_lost_observations
            / max(self.visible_observations, 1),
            "tracking_availability": self.trackable_observations
            / max(self.target_observations, 1),
            "lost_rate": self.lost_observations / max(self.target_observations, 1),
            "selected_frame_rate": self.selected_frames / max(self.total_frames, 1),
            "selection_switches": self.selection_switches,
            "selection_switches_per_1000_frames": 1000.0
            * self.selection_switches
            / max(self.total_frames, 1),
            "identity_confusion_rate": self.identity_confusions
            / max(self.identity_evaluations, 1),
            "identity_switch_events": self.identity_switch_events,
            "localization_samples": len(self.center_errors_px),
            "center_mae_px": _mean(self.center_errors_px),
            "center_median_px": _percentile(self.center_errors_px, 50.0),
            "center_p95_px": _percentile(self.center_errors_px, 95.0),
            "center_precision_5px": sum(value <= 5.0 for value in self.center_errors_px)
            / max(len(self.center_errors_px), 1),
            "center_precision_10px": sum(value <= 10.0 for value in self.center_errors_px)
            / max(len(self.center_errors_px), 1),
            "normalized_center_mae": _mean(self.normalized_center_errors),
            "mean_iou": _mean(self.ious),
            "success_iou_0_5": sum(value >= 0.5 for value in self.ious)
            / max(len(self.ious), 1),
            "success_iou_0_75": sum(value >= 0.75 for value in self.ious)
            / max(len(self.ious), 1),
            "yaw_mae_deg": _mean(self.yaw_errors_deg),
            "pitch_mae_deg": _mean(self.pitch_errors_deg),
            "angular_mae_deg": _mean(self.angular_errors_deg),
            "angular_rmse_deg": math.sqrt(_mean(angular_squared)),
            "angular_p50_deg": _percentile(self.angular_errors_deg, 50.0),
            "angular_p75_deg": _percentile(self.angular_errors_deg, 75.0),
            "angular_p80_deg": _percentile(self.angular_errors_deg, 80.0),
            "angular_p85_deg": _percentile(self.angular_errors_deg, 85.0),
            "angular_p90_deg": _percentile(self.angular_errors_deg, 90.0),
            "angular_p95_deg": _percentile(self.angular_errors_deg, 95.0),
            "angular_p99_deg": _percentile(self.angular_errors_deg, 99.0),
            "angular_max_deg": max(self.angular_errors_deg, default=0.0),
            "angular_precision_0_5deg": sum(
                value <= 0.5 for value in self.angular_errors_deg
            )
            / max(len(self.angular_errors_deg), 1),
            "angular_precision_1_0deg": sum(
                value <= 1.0 for value in self.angular_errors_deg
            )
            / max(len(self.angular_errors_deg), 1),
            "angular_precision_1_5deg": sum(
                value <= 1.5 for value in self.angular_errors_deg
            )
            / max(len(self.angular_errors_deg), 1),
            "lookahead_horizon_frames": self.horizon_frames,
            "lookahead_angular_mae_deg": _mean(
                self.lookahead_angular_errors_deg
            ),
            "selected_angular_mae_deg": _mean(
                self.selected_angular_errors_deg
            ),
            "selected_lookahead_angular_mae_deg": _mean(
                self.selected_lookahead_angular_errors_deg
            ),
            "distance_mae_m": _mean(self.distance_errors_m),
            "distance_p95_m": _percentile(self.distance_errors_m, 95.0),
            "distance_mape": _mean(self.distance_relative_errors),
            "position_samples": len(self.position_errors_m),
            "position_range_availability": len(self.position_errors_m)
            / max(self.visible_observations, 1),
            "position_3d_mae_m": _mean(self.position_errors_m),
            "position_3d_rmse_m": math.sqrt(_mean(position_squared)),
            "position_3d_p50_m": _percentile(self.position_errors_m, 50.0),
            "position_3d_p75_m": _percentile(self.position_errors_m, 75.0),
            "position_3d_p80_m": _percentile(self.position_errors_m, 80.0),
            "position_3d_p85_m": _percentile(self.position_errors_m, 85.0),
            "position_3d_p90_m": _percentile(self.position_errors_m, 90.0),
            "position_3d_p95_m": _percentile(self.position_errors_m, 95.0),
            "position_3d_p99_m": _percentile(self.position_errors_m, 99.0),
            "position_3d_max_m": max(self.position_errors_m, default=0.0),
            "position_precision_0_25m": sum(
                value <= 0.25 for value in self.position_errors_m
            )
            / max(len(self.position_errors_m), 1),
            "position_precision_0_5m": sum(
                value <= 0.5 for value in self.position_errors_m
            )
            / max(len(self.position_errors_m), 1),
            "position_precision_1_0m": sum(
                value <= 1.0 for value in self.position_errors_m
            )
            / max(len(self.position_errors_m), 1),
            "position_precision_2_0m": sum(
                value <= 2.0 for value in self.position_errors_m
            )
            / max(len(self.position_errors_m), 1),
            "position_relative_mae": _mean(self.position_relative_errors),
            "position_x_bias_m": _mean(self.position_x_errors_m),
            "position_y_bias_m": _mean(self.position_y_errors_m),
            "position_z_bias_m": _mean(self.position_z_errors_m),
            "position_x_mae_m": _mean(
                [abs(value) for value in self.position_x_errors_m]
            ),
            "position_y_mae_m": _mean(
                [abs(value) for value in self.position_y_errors_m]
            ),
            "position_z_mae_m": _mean(
                [abs(value) for value in self.position_z_errors_m]
            ),
            "position_radial_bias_m": _mean(self.position_radial_errors_m),
            "position_radial_mae_m": _mean(
                [abs(value) for value in self.position_radial_errors_m]
            ),
            "position_cross_track_mae_m": _mean(
                self.position_cross_track_errors_m
            ),
            "position_cross_track_p95_m": _percentile(
                self.position_cross_track_errors_m, 95.0
            ),
            "measured_range_bias_m": _mean(self.measured_range_errors_m),
            "measured_range_mae_m": _mean(
                [abs(value) for value in self.measured_range_errors_m]
            ),
            "measured_range_p95_m": _percentile(
                [abs(value) for value in self.measured_range_errors_m], 95.0
            ),
            "speed_mae_mps": _mean(self.speed_errors_mps),
            "speed_p95_mps": _percentile(self.speed_errors_mps, 95.0),
            "confidence_mean": _mean(self.confidences),
            "confidence_p10": _percentile(self.confidences, 10.0),
            "corners_mean": _mean(self.corner_counts),
            "ransac_inliers_mean": _mean(self.ransac_inliers),
            "fb_error_median_px": _percentile(self.fb_errors_px, 50.0),
            "status_counts": dict(sorted(self.status_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "kinematics_source_counts": dict(
                sorted(self.kinematics_source_counts.items())
            ),
            "fine_action_counts": dict(sorted(self.fine_action_counts.items())),
            "fine_trigger_reason_counts": dict(
                sorted(self.fine_trigger_reason_counts.items())
            ),
            "fine_initialize_count": self.fine_initialize_count,
            "fine_attempt_count": self.fine_attempt_count,
            "fine_applied_count": self.fine_applied_count,
            "fine_call_frame_rate": self.fine_attempt_count
            / max(self.total_frames, 1),
            "fine_apply_success_rate": self.fine_applied_count
            / max(self.fine_attempt_count, 1),
            "fine_risk_mean": _mean(self.fine_risks),
            "fine_initialize_latency_mean_ms": _mean(
                self.fine_initialize_latencies_ms
            ),
            "fine_initialize_latency_p95_ms": _percentile(
                self.fine_initialize_latencies_ms, 95.0
            ),
            "fine_latency_mean_ms": _mean(self.fine_track_latencies_ms),
            "fine_latency_p95_ms": _percentile(self.fine_track_latencies_ms, 95.0),
            "update_samples": len(self.update_times_s),
            "latency_mean_ms": _mean(latency_ms),
            "latency_median_ms": _percentile(latency_ms, 50.0),
            "latency_p95_ms": _percentile(latency_ms, 95.0),
            "latency_p99_ms": _percentile(latency_ms, 99.0),
            "deadline_20ms_miss_count": sum(value > 20.0 for value in latency_ms),
            "deadline_20ms_miss_rate": sum(value > 20.0 for value in latency_ms)
            / max(len(latency_ms), 1),
            "effective_fps": len(self.update_times_s) / total_update_s
            if total_update_s > 0.0
            else 0.0,
            "initialize_mean_ms": _mean(init_ms),
            "initialize_max_ms": max(init_ms, default=0.0),
        }


class _SpeedRangeAccumulator:
    """Aggregate localization quality by realized ground-truth target speed."""

    def __init__(
        self,
        *,
        bin_width_mps: float,
        min_observations: int,
        min_tracking_availability: float,
        min_success_iou_0_5: float,
        max_angular_p95_deg: float,
        position_range_adjustment_m: float = 0.0,
    ) -> None:
        self.bin_width_mps = float(bin_width_mps)
        self.position_range_adjustment_m = float(position_range_adjustment_m)
        self.criteria = {
            "min_observations": int(min_observations),
            "min_tracking_availability": float(min_tracking_availability),
            "min_success_iou_0_5": float(min_success_iou_0_5),
            "max_angular_p95_deg": float(max_angular_p95_deg),
        }
        self._bins: dict[int, dict[str, Any]] = {}

    def add_sequence(
        self,
        *,
        results: list[MultiTargetFrameResult],
        ground_truth: dict[tuple[int, int], dict[str, float | bool]],
    ) -> None:
        bins_seen: set[int] = set()
        for result in results:
            for estimate in result.targets:
                gt = ground_truth.get((result.frame_index, estimate.target_id))
                if gt is None or not bool(gt["visible"]):
                    continue
                speed_m_s = float(gt["speed_m_s"])
                if not math.isfinite(speed_m_s) or speed_m_s < 0.0:
                    continue
                bin_index = int(math.floor(speed_m_s / self.bin_width_mps))
                values = self._bins.setdefault(
                    bin_index,
                    {
                        "speeds_mps": [],
                        "lost_count": 0,
                        "ious": [],
                        "center_errors_px": [],
                        "angular_errors_deg": [],
                        "position_errors_m": [],
                        "sequence_count": 0,
                    },
                )
                bins_seen.add(bin_index)
                values["speeds_mps"].append(speed_m_s)
                values["lost_count"] += int(estimate.status == "lost")
                truth_bbox = (
                    float(gt["bbox_x1"]),
                    float(gt["bbox_y1"]),
                    float(gt["bbox_x2"]),
                    float(gt["bbox_y2"]),
                )
                values["ious"].append(_bbox_iou(estimate.bbox_xyxy, truth_bbox))
                values["center_errors_px"].append(
                    math.hypot(
                        estimate.center_xy[0] - float(gt["center_u"]),
                        estimate.center_xy[1] - float(gt["center_v"]),
                    )
                )
                values["angular_errors_deg"].append(
                    math.degrees(
                        _angular_error_rad(
                            estimate.yaw_rad,
                            estimate.pitch_rad,
                            float(gt["yaw_rad"]),
                            float(gt["pitch_rad"]),
                        )
                    )
                )
                position_error = _fused_position_error(
                    estimate.yaw_rad,
                    estimate.pitch_rad,
                    gt,
                    self.position_range_adjustment_m,
                )
                if position_error is not None:
                    values["position_errors_m"].append(
                        position_error["position_error_m"]
                    )
        for bin_index in bins_seen:
            self._bins[bin_index]["sequence_count"] += 1

    def summary(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for bin_index, values in sorted(self._bins.items()):
            speeds = values["speeds_mps"]
            ious = values["ious"]
            angular_errors = values["angular_errors_deg"]
            position_errors = values["position_errors_m"]
            observations = len(speeds)
            tracking_availability = 1.0 - values["lost_count"] / max(observations, 1)
            success_iou_0_5 = sum(value >= 0.5 for value in ious) / max(
                len(ious), 1
            )
            angular_p95_deg = _percentile(angular_errors, 95.0)
            failure_reasons: list[str] = []
            if observations < self.criteria["min_observations"]:
                failure_reasons.append("insufficient_observations")
            if tracking_availability < self.criteria["min_tracking_availability"]:
                failure_reasons.append("tracking_availability")
            if success_iou_0_5 < self.criteria["min_success_iou_0_5"]:
                failure_reasons.append("success_iou_0_5")
            if angular_p95_deg > self.criteria["max_angular_p95_deg"]:
                failure_reasons.append("angular_p95_deg")
            rows.append(
                {
                    "speed_bin_index": bin_index,
                    "speed_bin_start_mps": bin_index * self.bin_width_mps,
                    "speed_bin_end_mps": (bin_index + 1) * self.bin_width_mps,
                    "speed_observed_min_mps": min(speeds),
                    "speed_observed_mean_mps": _mean(speeds),
                    "speed_observed_max_mps": max(speeds),
                    "sequence_count": values["sequence_count"],
                    "visible_observations": observations,
                    "tracking_availability": tracking_availability,
                    "lost_rate": 1.0 - tracking_availability,
                    "mean_iou": _mean(ious),
                    "success_iou_0_5": success_iou_0_5,
                    "center_mae_px": _mean(values["center_errors_px"]),
                    "center_p95_px": _percentile(values["center_errors_px"], 95.0),
                    "angular_mae_deg": _mean(angular_errors),
                    "angular_p50_deg": _percentile(angular_errors, 50.0),
                    "angular_p75_deg": _percentile(angular_errors, 75.0),
                    "angular_p80_deg": _percentile(angular_errors, 80.0),
                    "angular_p85_deg": _percentile(angular_errors, 85.0),
                    "angular_p90_deg": _percentile(angular_errors, 90.0),
                    "angular_p95_deg": angular_p95_deg,
                    "angular_p99_deg": _percentile(angular_errors, 99.0),
                    "angular_max_deg": max(angular_errors, default=0.0),
                    "position_samples": len(position_errors),
                    "position_range_availability": len(position_errors)
                    / max(observations, 1),
                    "position_3d_mae_m": _mean(position_errors),
                    "position_3d_rmse_m": math.sqrt(
                        _mean([value * value for value in position_errors])
                    ),
                    "position_3d_p50_m": _percentile(position_errors, 50.0),
                    "position_3d_p75_m": _percentile(position_errors, 75.0),
                    "position_3d_p80_m": _percentile(position_errors, 80.0),
                    "position_3d_p85_m": _percentile(position_errors, 85.0),
                    "position_3d_p90_m": _percentile(position_errors, 90.0),
                    "position_3d_p95_m": _percentile(position_errors, 95.0),
                    "position_3d_p99_m": _percentile(position_errors, 99.0),
                    "position_3d_max_m": max(position_errors, default=0.0),
                    "passes": not failure_reasons,
                    "failure_reasons": failure_reasons,
                }
            )

        intervals: list[dict[str, float]] = []
        current: list[dict[str, Any]] = []
        for row in rows:
            is_adjacent = (
                current
                and row["speed_bin_index"]
                == current[-1]["speed_bin_index"] + 1
            )
            if row["passes"] and (not current or is_adjacent):
                current.append(row)
                continue
            if current:
                intervals.append(self._interval(current))
                current = []
            if row["passes"]:
                current = [row]
        if current:
            intervals.append(self._interval(current))

        eligible_rows = [
            row
            for row in rows
            if row["visible_observations"] >= self.criteria["min_observations"]
        ]
        continuous_from_min: dict[str, float] | None = None
        if eligible_rows and eligible_rows[0]["passes"]:
            first_index = eligible_rows[0]["speed_bin_index"]
            continuous_from_min = next(
                (
                    interval
                    for interval in intervals
                    if interval["first_bin_index"] == first_index
                ),
                None,
            )
        return {
            "speed_definition": (
                "ground-truth camera-frame relative 3D speed, "
                "sqrt(vx_cam_mps^2 + vy_cam_mps^2 + vz_cam_mps^2)"
            ),
            "bin_width_mps": self.bin_width_mps,
            "acceptance_criteria": self.criteria,
            "bins": rows,
            "continuous_supported_ranges_mps": intervals,
            "continuous_supported_range_from_min_mps": continuous_from_min,
            "maximum_continuous_supported_speed_mps": (
                continuous_from_min["observed_max_mps"]
                if continuous_from_min is not None
                else None
            ),
        }

    @staticmethod
    def _interval(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "first_bin_index": rows[0]["speed_bin_index"],
            "last_bin_index": rows[-1]["speed_bin_index"],
            "bin_start_mps": rows[0]["speed_bin_start_mps"],
            "bin_end_mps": rows[-1]["speed_bin_end_mps"],
            "observed_min_mps": rows[0]["speed_observed_min_mps"],
            "observed_max_mps": rows[-1]["speed_observed_max_mps"],
        }


def _build_config(
    args: argparse.Namespace,
    dataset_meta: dict[str, Any],
) -> MultiTargetConfig:
    target_radius_m = float(dataset_meta.get("target_radius_m", 0.35))
    distance_range = dataset_meta.get("target_distance_range_m", [80.0, 150.0])
    speed_range = dataset_meta.get("target_speed_range_mps", [0.0, 5.0])
    return MultiTargetConfig(
        fb_max_error_px=float(args.fb_max_error_px),
        ransac_reproj_threshold_px=float(args.ransac_threshold_px),
        corner_redetect_interval_frames=int(args.corner_redetect_interval_frames),
        max_missed_frames=int(args.max_missed_frames),
        prediction_horizon_s=float(args.prediction_horizon_s),
        target_width_m=(
            2.0 * target_radius_m
            if args.target_width_m is None
            else float(args.target_width_m)
        ),
        target_height_m=(
            2.0 * target_radius_m
            if args.target_height_m is None
            else float(args.target_height_m)
        ),
        kinematic_estimation_horizon_s=float(args.kinematic_estimation_horizon_s),
        priority_confidence_weight=float(args.priority_confidence_weight),
        priority_center_weight=float(args.priority_center_weight),
        priority_area_weight=float(args.priority_area_weight),
        priority_stability_weight=float(args.priority_stability_weight),
        priority_distance_weight=float(args.priority_distance_weight),
        priority_speed_weight=float(args.priority_speed_weight),
        priority_distance_near_m=(
            float(distance_range[0])
            if args.priority_distance_near_m is None
            else float(args.priority_distance_near_m)
        ),
        priority_distance_far_m=(
            float(distance_range[1])
            if args.priority_distance_far_m is None
            else float(args.priority_distance_far_m)
        ),
        priority_speed_reference_mps=(
            float(speed_range[1])
            if args.priority_speed_reference_mps is None
            else float(args.priority_speed_reference_mps)
        ),
        priority_switch_margin=float(args.priority_switch_margin),
        fine_tracker_interval_frames=int(args.fine_tracker_interval_frames),
        fine_tracker_min_score=float(args.fine_tracker_min_score),
        fine_tracker_policy=str(args.fine_tracker_policy),
        adaptive_min_interval_frames=int(args.adaptive_min_interval_frames),
        adaptive_max_interval_frames=int(args.adaptive_max_interval_frames),
        adaptive_bad_frames=int(args.adaptive_bad_frames),
        adaptive_warning_confidence=float(args.adaptive_warning_confidence),
        adaptive_critical_confidence=float(args.adaptive_critical_confidence),
        adaptive_risk_threshold=float(args.adaptive_risk_threshold),
        adaptive_critical_risk_threshold=float(
            args.adaptive_critical_risk_threshold
        ),
        adaptive_risk_clear_threshold=float(args.adaptive_risk_clear_threshold),
        adaptive_desired_inliers=int(args.adaptive_desired_inliers),
        adaptive_innovation_critical=float(args.adaptive_innovation_critical),
        adaptive_prediction_iou_threshold=float(
            args.adaptive_prediction_iou_threshold
        ),
        adaptive_failure_backoff_initial_frames=int(
            args.adaptive_failure_backoff_initial_frames
        ),
        adaptive_failure_backoff_max_frames=int(
            args.adaptive_failure_backoff_max_frames
        ),
        fine_max_center_jump_box_diagonals=float(
            args.fine_max_center_jump_box_diagonals
        ),
        fine_max_scale_change=float(args.fine_max_scale_change),
    )


def _evaluate_sequence(
    *,
    sequence_dir: Path,
    config: MultiTargetConfig,
    fine_tracker: OSTrackFrameTracker | None,
    max_frames: int | None,
    timing_device: str | None,
    kinematics_source: str,
    target_radius_m: float,
    view_margin_px: float,
) -> tuple[
    list[MultiTargetFrameResult],
    dict[tuple[int, int], dict[str, float | bool]],
    list[float],
    float,
    float,
]:
    intrinsics = CameraIntrinsics.from_json(sequence_dir / "intrinsics.json")
    boxes, biases = load_initial_targets(sequence_dir / "initial_targets.json")
    ground_truth = _read_ground_truth(
        sequence_dir / "ground_truth.csv",
        intrinsics=intrinsics,
        target_radius_m=target_radius_m,
        view_margin_px=view_margin_px,
    )
    ground_truth_kinematics = _ground_truth_kinematics_by_frame(ground_truth)
    rgb_paths = sorted((sequence_dir / "rgb").glob("*.png"))
    if max_frames is not None:
        rgb_paths = rgb_paths[: int(max_frames)]
    if not rgb_paths:
        raise ValueError(f"No RGB frames found in {sequence_dir / 'rgb'}")
    fps = float(json.loads((sequence_dir / "sequence_meta.json").read_text())["fps"])
    tracker = MultiTargetVisionTracker(
        intrinsics,
        config=config,
        fine_tracker=fine_tracker,
    )

    first_image = _read_rgb(rgb_paths[0])
    initial_kinematics = (
        ground_truth_kinematics.get(0)
        if kinematics_source == "ground_truth"
        else None
    )
    _synchronize_device(timing_device)
    start = time.perf_counter()
    results = [
        tracker.initialize(
            first_image,
            boxes,
            priority_biases=biases,
            target_kinematics=initial_kinematics,
            timestamp_s=0.0,
        )
    ]
    _synchronize_device(timing_device)
    initialize_time_s = time.perf_counter() - start
    update_times_s: list[float] = []
    for frame_index, rgb_path in enumerate(rgb_paths[1:], start=1):
        image = _read_rgb(rgb_path)
        frame_kinematics = (
            ground_truth_kinematics.get(frame_index)
            if kinematics_source == "ground_truth"
            else None
        )
        _synchronize_device(timing_device)
        start = time.perf_counter()
        results.append(
            tracker.update(
                image,
                dt_s=1.0 / fps,
                timestamp_s=frame_index / fps,
                target_kinematics=frame_kinematics,
            )
        )
        _synchronize_device(timing_device)
        update_times_s.append(time.perf_counter() - start)
    return results, ground_truth, update_times_s, initialize_time_s, fps


def _synchronize_device(device: str | None) -> None:
    if device is None or not str(device).startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(device)


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in summary.items():
        row[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_meta = json.loads(
        (dataset_dir / "dataset_meta.json").read_text(encoding="utf-8")
    )
    manifest = _read_manifest(dataset_dir)
    if args.max_sequences is not None:
        manifest = manifest[: int(args.max_sequences)]

    fine_tracker = None
    if not args.disable_fine_tracking:
        fine_tracker = OSTrackFrameTracker(
            repo_path=args.ostrack_repo,
            checkpoint=args.ostrack_checkpoint,
            config=args.ostrack_config,
            device=args.device,
            debug=int(args.ostrack_debug),
        )
        print("[multi-eval] loading OSTrack checkpoint before timing")
        fine_tracker.prepare()
    config = _build_config(args, dataset_meta)
    target_radius_m = float(dataset_meta.get("target_radius_m", 0.35))
    view_margin_px = float(dataset_meta.get("target_view_margin_px", 0.0))
    aggregate: _MetricsAccumulator | None = None
    speed_range = _SpeedRangeAccumulator(
        bin_width_mps=float(args.speed_bin_width_mps),
        min_observations=int(args.speed_min_observations),
        min_tracking_availability=float(args.speed_min_tracking_availability),
        min_success_iou_0_5=float(args.speed_min_success_iou_0_5),
        max_angular_p95_deg=float(args.speed_max_angular_p95_deg),
        position_range_adjustment_m=float(args.position_range_adjustment_m),
    )
    sequence_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    for index, row in enumerate(manifest):
        sequence_dir = dataset_dir / row["seq_dir"]
        results, ground_truth, update_times, init_time, fps = _evaluate_sequence(
            sequence_dir=sequence_dir,
            config=config,
            fine_tracker=fine_tracker,
            max_frames=args.max_frames,
            timing_device=str(args.device) if fine_tracker is not None else None,
            kinematics_source=str(args.kinematics_source),
            target_radius_m=target_radius_m,
            view_margin_px=view_margin_px,
        )
        if aggregate is None:
            aggregate = _MetricsAccumulator(
                fps=fps,
                prediction_horizon_s=config.prediction_horizon_s,
                position_range_adjustment_m=float(args.position_range_adjustment_m),
            )
        elif not math.isclose(aggregate.fps, fps):
            raise ValueError("All sequences must have the same FPS")
        sequence_metrics = _MetricsAccumulator(
            fps=fps,
            prediction_horizon_s=config.prediction_horizon_s,
            position_range_adjustment_m=float(args.position_range_adjustment_m),
        )
        sequence_metrics.add_sequence(
            results=results,
            ground_truth=ground_truth,
            update_times_s=update_times,
            initialize_time_s=init_time,
        )
        aggregate.add_sequence(
            results=results,
            ground_truth=ground_truth,
            update_times_s=update_times,
            initialize_time_s=init_time,
        )
        speed_range.add_sequence(results=results, ground_truth=ground_truth)
        sequence_rows.append(
            {
                "sequence_name": sequence_dir.name,
                **_flatten_summary(sequence_metrics.summary()),
            }
        )
        refinement_expected = (
            config.fine_tracker_policy == "fixed"
            or len(results) > config.adaptive_max_interval_frames
        )
        if (
            fine_tracker is not None
            and index == 0
            and refinement_expected
            and not any(
                estimate.source == "ostrack"
                for result in results[1:]
                for estimate in result.targets
            )
        ):
            raise RuntimeError(
                f"OSTrack produced no accepted refinement in {sequence_dir.name}. "
                "Check the model output and --fine_tracker_min_score instead of "
                "treating this run as a hybrid evaluation."
            )
        if (index + 1) % int(args.progress_every) == 0 or index + 1 == len(manifest):
            elapsed = time.perf_counter() - run_start
            print(
                "[multi-eval]",
                f"sequences={index + 1}/{len(manifest)}",
                f"frames={aggregate.total_frames}",
                f"elapsed_s={elapsed:.1f}",
            )

    if aggregate is None:
        raise RuntimeError("No sequences were evaluated")
    speed_range_summary = speed_range.summary()
    summary = {
        "dataset_dir": str(dataset_dir),
        "fine_tracking": fine_tracker is not None,
        "device": str(args.device) if fine_tracker is not None else "cpu (OpenCV)",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kinematics_source": str(args.kinematics_source),
        "ground_truth_source": "target_pos_cam_reprojection",
        "position_estimate_source": "tracker_bearing_x_recorded_lidar_range",
        "position_range_adjustment_m": float(args.position_range_adjustment_m),
        "position_truth_source": "target_pos_cam",
        "position_error_frame": "opencv_camera_x_right_y_down_z_forward",
        "position_metrics_condition": "visible_ground_truth_and_lidar_valid",
        "ostrack": (
            {
                "repo": str(args.ostrack_repo.resolve()),
                "checkpoint": str(args.ostrack_checkpoint.resolve()),
                "config": str(args.ostrack_config),
            }
            if fine_tracker is not None
            else None
        ),
        "sequence_count": len(manifest),
        "wall_time_s": time.perf_counter() - run_start,
        "latency_scope": (
            "tracker initialize/update only; image decoding and checkpoint loading "
            "excluded; CUDA synchronized"
            if fine_tracker is not None
            else "tracker initialize/update only; image decoding excluded"
        ),
        "tracker_config": {
            "fb_max_error_px": config.fb_max_error_px,
            "ransac_reproj_threshold_px": config.ransac_reproj_threshold_px,
            "corner_redetect_interval_frames": config.corner_redetect_interval_frames,
            "max_missed_frames": config.max_missed_frames,
            "prediction_horizon_s": config.prediction_horizon_s,
            "target_width_m": config.target_width_m,
            "target_height_m": config.target_height_m,
            "kinematic_estimation_horizon_s": config.kinematic_estimation_horizon_s,
            "priority_confidence_weight": config.priority_confidence_weight,
            "priority_center_weight": config.priority_center_weight,
            "priority_area_weight": config.priority_area_weight,
            "priority_stability_weight": config.priority_stability_weight,
            "priority_distance_weight": config.priority_distance_weight,
            "priority_speed_weight": config.priority_speed_weight,
            "priority_distance_near_m": config.priority_distance_near_m,
            "priority_distance_far_m": config.priority_distance_far_m,
            "priority_speed_reference_mps": config.priority_speed_reference_mps,
            "priority_switch_margin": config.priority_switch_margin,
            "fine_tracker_interval_frames": config.fine_tracker_interval_frames,
            "fine_tracker_min_score": config.fine_tracker_min_score,
            "fine_tracker_policy": config.fine_tracker_policy,
            "adaptive_min_interval_frames": config.adaptive_min_interval_frames,
            "adaptive_max_interval_frames": config.adaptive_max_interval_frames,
            "adaptive_bad_frames": config.adaptive_bad_frames,
            "adaptive_warning_confidence": config.adaptive_warning_confidence,
            "adaptive_critical_confidence": config.adaptive_critical_confidence,
            "adaptive_risk_threshold": config.adaptive_risk_threshold,
            "adaptive_critical_risk_threshold": (
                config.adaptive_critical_risk_threshold
            ),
            "adaptive_risk_clear_threshold": config.adaptive_risk_clear_threshold,
            "adaptive_desired_inliers": config.adaptive_desired_inliers,
            "adaptive_innovation_critical": config.adaptive_innovation_critical,
            "adaptive_prediction_iou_threshold": (
                config.adaptive_prediction_iou_threshold
            ),
            "adaptive_failure_backoff_initial_frames": (
                config.adaptive_failure_backoff_initial_frames
            ),
            "adaptive_failure_backoff_max_frames": (
                config.adaptive_failure_backoff_max_frames
            ),
            "fine_max_center_jump_box_diagonals": (
                config.fine_max_center_jump_box_diagonals
            ),
            "fine_max_scale_change": config.fine_max_scale_change,
        },
        "speed_range_analysis": speed_range_summary,
        **aggregate.summary(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "per_sequence.csv", sequence_rows)
    _write_csv(out_dir / "summary.csv", [_flatten_summary(summary)])
    speed_rows = []
    for row in speed_range_summary["bins"]:
        csv_row = dict(row)
        csv_row["failure_reasons"] = ";".join(row["failure_reasons"])
        speed_rows.append(csv_row)
    _write_csv(out_dir / "speed_by_bin.csv", speed_rows)
    (out_dir / "speed_range.json").write_text(
        json.dumps(speed_range_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[multi-eval] complete",
        f"sequences={summary['sequence_count']}",
        f"frames={summary['frames']}",
        f"angular_mae_deg={summary['angular_mae_deg']:.4f}",
        f"position_3d_mae_m={summary['position_3d_mae_m']:.4f}",
        f"position_range_availability={summary['position_range_availability']:.4f}",
        f"mean_iou={summary['mean_iou']:.4f}",
        f"latency_mean_ms={summary['latency_mean_ms']:.3f}",
        f"deadline_20ms_miss_rate={summary['deadline_20ms_miss_rate']:.4f}",
        f"fine_attempts={summary['fine_attempt_count']}",
        f"effective_fps={summary['effective_fps']:.2f}",
        "max_supported_speed_mps="
        f"{speed_range_summary['maximum_continuous_supported_speed_mps']}",
        f"summary={out_dir / 'summary.json'}",
    )


if __name__ == "__main__":
    main()

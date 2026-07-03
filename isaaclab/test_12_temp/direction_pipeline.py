from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from camera_geometry import CameraIntrinsics, bbox_center_xy
from test12.ostrack_tracker import TrackedBox


@dataclass
class FrameEstimate:
    frame_index: int
    timestamp_s: float
    valid_measurement: bool
    mask_area: int
    track_time_s: float | None
    bbox_xyxy: tuple[int, int, int, int] | None
    u: float | None
    v: float | None
    depth_m: float | None
    depth_pixels: int
    yaw_rad: float | None
    pitch_rad: float | None
    yaw_rate_radps: float | None
    pitch_rate_radps: float | None
    pos_cam_m: tuple[float, float, float] | None
    vel_cam_mps: tuple[float, float, float] | None

    @property
    def speed_mps(self) -> float | None:
        if self.vel_cam_mps is None:
            return None
        return float(np.linalg.norm(np.asarray(self.vel_cam_mps, dtype=np.float64)))


def load_rgb_paths(rgb_dir: str | Path) -> list[Path]:
    rgb_dir = Path(rgb_dir)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    rgb_paths = [path for path in rgb_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}.")
    return sorted(path.resolve() for path in rgb_paths)


def _empty_estimate(frame_index: int, timestamp_s: float) -> FrameEstimate:
    return FrameEstimate(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        valid_measurement=False,
        mask_area=0,
        track_time_s=None,
        bbox_xyxy=None,
        u=None,
        v=None,
        depth_m=None,
        depth_pixels=0,
        yaw_rad=None,
        pitch_rad=None,
        yaw_rate_radps=None,
        pitch_rate_radps=None,
        pos_cam_m=None,
        vel_cam_mps=None,
    )


def _fill_angular_rates(estimates: list[FrameEstimate], fps: float) -> None:
    dt = 1.0 / float(fps)
    prev: FrameEstimate | None = None
    for estimate in estimates:
        if estimate.yaw_rad is None or estimate.pitch_rad is None:
            continue
        if prev is not None and prev.yaw_rad is not None and prev.pitch_rad is not None:
            dyaw = math.atan2(
                math.sin(estimate.yaw_rad - prev.yaw_rad),
                math.cos(estimate.yaw_rad - prev.yaw_rad),
            )
            dpitch = estimate.pitch_rad - prev.pitch_rad
            frame_gap = max(1, estimate.frame_index - prev.frame_index)
            rate_dt = dt * frame_gap
            estimate.yaw_rate_radps = dyaw / rate_dt
            estimate.pitch_rate_radps = dpitch / rate_dt
        prev = estimate


def tracked_boxes_to_map(tracks: Iterable[TrackedBox]) -> dict[int, TrackedBox]:
    return {int(track.frame_index): track for track in tracks}


def track_timing_stats(tracks: Iterable[TrackedBox]) -> dict[str, float]:
    times_s = [float(track.track_time_s) for track in tracks if track.track_time_s is not None]
    if not times_s:
        return {
            "total_track_time_s": 0.0,
            "mean_track_time_ms": 0.0,
            "median_track_time_ms": 0.0,
            "p95_track_time_ms": 0.0,
            "track_fps": 0.0,
        }

    times = np.asarray(times_s, dtype=np.float64)
    total = float(times.sum())
    return {
        "total_track_time_s": total,
        "mean_track_time_ms": float(times.mean() * 1000.0),
        "median_track_time_ms": float(np.median(times) * 1000.0),
        "p95_track_time_ms": float(np.percentile(times, 95.0) * 1000.0),
        "track_fps": float(len(times) / total) if total > 0.0 else 0.0,
    }


def estimate_target_direction_from_boxes(
    rgb_paths: list[Path],
    tracks_by_frame: Mapping[int, TrackedBox],
    intrinsics: CameraIntrinsics,
    fps: float = 30.0,
) -> list[FrameEstimate]:
    if not rgb_paths:
        raise ValueError("rgb_paths is empty.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    dt = 1.0 / float(fps)
    estimates: list[FrameEstimate] = []
    for frame_index, _rgb_path in enumerate(rgb_paths):
        timestamp_s = frame_index * dt
        track = tracks_by_frame.get(frame_index)
        if track is None or track.bbox_xyxy is None or not track.ok:
            estimates.append(_empty_estimate(frame_index, timestamp_s))
            continue

        bbox = track.bbox_xyxy
        u, v = bbox_center_xy(bbox)
        yaw, pitch = intrinsics.direction_from_pixel(u, v)
        estimates.append(
            FrameEstimate(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                valid_measurement=True,
                mask_area=track.bbox_area,
                track_time_s=track.track_time_s,
                bbox_xyxy=bbox,
                u=float(u),
                v=float(v),
                depth_m=None,
                depth_pixels=0,
                yaw_rad=float(yaw),
                pitch_rad=float(pitch),
                yaw_rate_radps=None,
                pitch_rate_radps=None,
                pos_cam_m=None,
                vel_cam_mps=None,
            )
        )

    _fill_angular_rates(estimates, fps=fps)
    return estimates


def write_overlays(
    rgb_paths: list[Path],
    tracks_by_frame: Mapping[int, TrackedBox],
    estimates: list[FrameEstimate],
    out_dir: str | Path,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to write overlays. "
            "Install a cv2 build compatible with the active NumPy version."
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    estimate_map = {estimate.frame_index: estimate for estimate in estimates}

    for frame_index, rgb_path in enumerate(rgb_paths):
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        track = tracks_by_frame.get(frame_index)
        estimate = estimate_map.get(frame_index)
        overlay = image.copy()
        if track is not None and track.bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.drawMarker(
                overlay,
                (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))),
                (0, 180, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )
        if estimate is not None and estimate.yaw_rad is not None and estimate.pitch_rad is not None:
            label = (
                f"yaw={math.degrees(estimate.yaw_rad):.2f}deg "
                f"pitch={math.degrees(estimate.pitch_rad):.2f}deg"
            )
        else:
            label = "no valid estimate"
        cv2.putText(
            overlay,
            label,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"{frame_index:06d}.jpg"), overlay)
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
ISAACLAB_DIR = THIS_DIR.parent
TEST_TEMP_DIR = ISAACLAB_DIR / "test_temp"
for import_dir in (TEST_TEMP_DIR, ISAACLAB_DIR, THIS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from camera_geometry import CameraIntrinsics, build_intrinsics, load_initial_box
from direction_pipeline import (
    FrameEstimate,
    estimate_target_direction_from_boxes,
    load_rgb_paths,
    track_timing_stats,
    tracked_boxes_to_map,
    write_overlays,
)
from test12.ostrack_tracker import OSTrackBoxTracker
from test_temp.speed_prediction_model import VelocityPredictorGRU

@dataclass(frozen=True)
class RobotMotion:
    lin_vel_mps: tuple[float, float, float]
    ang_vel_radps: tuple[float, float, float]
    dog_moving_flag: float


@dataclass(frozen=True)
class FramePose:
    camera_pos_w: tuple[float, float, float]
    camera_quat_wxyz: tuple[float, float, float, float]
    root_pos_w: tuple[float, float, float]
    root_quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class VelocityEstimate:
    frame_index: int
    valid: bool
    window_start_frame: int | None
    window_size: int
    model_velocity_mps: tuple[float, float, float] | None
    finite_diff_velocity_mps: tuple[float, float, float] | None

    @property
    def model_speed_mps(self) -> float | None:
        if self.model_velocity_mps is None:
            return None
        return float(np.linalg.norm(np.asarray(self.model_velocity_mps, dtype=np.float64)))

    @property
    def finite_diff_speed_mps(self) -> float | None:
        if self.finite_diff_velocity_mps is None:
            return None
        return float(np.linalg.norm(np.asarray(self.finite_diff_velocity_mps, dtype=np.float64)))


def _parse_vec3(row: dict[str, str], aliases: Iterable[tuple[str, str, str]]) -> tuple[float, float, float] | None:
    for x_key, y_key, z_key in aliases:
        try:
            return float(row[x_key]), float(row[y_key]), float(row[z_key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _parse_optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _matches_target_row(
    row: dict[str, str],
    target_object_id: int | None = None,
    target_color_name: str | None = None,
) -> bool:
    if target_object_id is not None and str(row.get("object_id", "")).strip() != "":
        row_object_id = _parse_optional_int(row.get("object_id"))
        if row_object_id != int(target_object_id):
            return False

    requested_color = str(target_color_name or "").strip().lower()
    row_color = str(row.get("target_color_name", "")).strip().lower()
    if requested_color and row_color and row_color != requested_color:
        return False

    return True


def _parse_float_cell(value: Any) -> float | None:
    text = str(value or "").strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _row_measurement_is_valid(row: dict[str, str]) -> bool:
    valid_keys = (
        "measurement_valid",
        "sensor_valid",
        "valid_sensor",
        "lidar_valid_mask",
        "lidar_valid",
        "raycast_valid",
        "valid_mask",
    )
    for key in valid_keys:
        if key not in row:
            continue
        text = str(row.get(key, "")).strip()
        if text == "":
            continue
        try:
            return bool(int(float(text)))
        except ValueError:
            return text.lower() in {"true", "yes", "y", "valid"}
    return True


def _parse_quat4(row: dict[str, str], aliases: Iterable[tuple[str, str, str, str]]) -> tuple[float, float, float, float] | None:
    for w_key, x_key, y_key, z_key in aliases:
        values = tuple(_parse_float_cell(row.get(key)) for key in (w_key, x_key, y_key, z_key))
        if all(value is not None for value in values):
            return tuple(float(value) for value in values)  # type: ignore[arg-type]
    return None


def _read_rows_by_frame(
    path: Path,
    target_object_id: int | None = None,
    target_color_name: str | None = None,
) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "frame_index" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a frame_index column.")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            if not _matches_target_row(row, target_object_id=target_object_id, target_color_name=target_color_name):
                continue
            try:
                frame_index = int(float(row["frame_index"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame_index in {path}: {row}") from exc
            rows[frame_index] = row
    return rows


def load_distance_csv(
    path: Path,
    target_object_id: int | None = None,
    target_color_name: str | None = None,
) -> dict[int, float]:
    rows = _read_rows_by_frame(path, target_object_id=target_object_id, target_color_name=target_color_name)
    aliases = (
        "lidar_distance",
        "sensor_distance_m",
        "sensor_range_m",
        "pred_distance",
        "distance_m",
        "range_m",
        "depth_m",
        "z_cam_m",
        "target_distance_m",
    )
    out: dict[int, float] = {}
    for frame_index, row in rows.items():
        if not _row_measurement_is_valid(row):
            continue
        for key in aliases:
            value = _parse_float_cell(row.get(key))
            if value is not None:
                out[frame_index] = float(value)
                break
    if not out:
        raise ValueError(f"{path} must contain one of: {', '.join(aliases)}")
    return out


def load_position_csv(
    path: Path,
    target_object_id: int | None = None,
    target_color_name: str | None = None,
) -> dict[int, tuple[float, float, float]]:
    rows = _read_rows_by_frame(path, target_object_id=target_object_id, target_color_name=target_color_name)
    aliases = (
        ("x_input_m", "y_input_m", "z_input_m"),
        ("refined_pos_base_x", "refined_pos_base_y", "refined_pos_base_z"),
        ("sensor_pos_base_x", "sensor_pos_base_y", "sensor_pos_base_z"),
        ("pred_pos_base_x", "pred_pos_base_y", "pred_pos_base_z"),
        ("pos_base_x", "pos_base_y", "pos_base_z"),
        ("refined_x", "refined_y", "refined_z"),
        ("x", "y", "z"),
    )
    out: dict[int, tuple[float, float, float]] = {}
    for frame_index, row in rows.items():
        if not _row_measurement_is_valid(row):
            continue
        vec = _parse_vec3(row, aliases)
        if vec is not None:
            out[frame_index] = tuple(float(v) for v in vec)
    if not out:
        raise ValueError(f"{path} does not contain a recognized base-frame 3D position column set.")
    return out


def load_robot_motion_csv(
    path: Path,
    target_object_id: int | None = None,
    target_color_name: str | None = None,
) -> dict[int, RobotMotion]:
    rows = _read_rows_by_frame(path, target_object_id=target_object_id, target_color_name=target_color_name)
    lin_aliases = (
        ("robot_lin_vel_x", "robot_lin_vel_y", "robot_lin_vel_z"),
        ("robot_vx_mps", "robot_vy_mps", "robot_vz_mps"),
        ("lin_vel_x", "lin_vel_y", "lin_vel_z"),
    )
    ang_aliases = (
        ("robot_ang_vel_x", "robot_ang_vel_y", "robot_ang_vel_z"),
        ("robot_wx_radps", "robot_wy_radps", "robot_wz_radps"),
        ("ang_vel_x", "ang_vel_y", "ang_vel_z"),
    )
    out: dict[int, RobotMotion] = {}
    for frame_index, row in rows.items():
        lin = _parse_vec3(row, lin_aliases) or (0.0, 0.0, 0.0)
        ang = _parse_vec3(row, ang_aliases) or (0.0, 0.0, 0.0)
        dog_flag = row.get("dog_moving_flag", "")
        if dog_flag == "":
            dog_flag = row.get("moving_flag", "")
        out[frame_index] = RobotMotion(
            lin_vel_mps=lin,
            ang_vel_radps=ang,
            dog_moving_flag=0.0 if dog_flag == "" else float(dog_flag),
        )
    return out

def _default_robot_motion() -> RobotMotion:
    return RobotMotion(
        lin_vel_mps=(0.0, 0.0, 0.0),
        ang_vel_radps=(0.0, 0.0, 0.0),
        dog_moving_flag=0.0,
    )


def _point_from_range(yaw_rad: float, pitch_rad: float, range_m: float) -> tuple[float, float, float]:
    distance = float(range_m)
    yaw = float(yaw_rad)
    pitch = float(pitch_rad)
    x = distance * math.sin(yaw) * math.cos(pitch)
    y = -distance * math.sin(pitch)
    z = distance * math.cos(yaw) * math.cos(pitch)
    return float(x), float(y), float(z)


def _estimate_depth_from_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    intrinsics: CameraIntrinsics,
    target_diameter_m: float,
    bbox_dimension: str,
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    width_px = max(1.0, float(x2 - x1))
    height_px = max(1.0, float(y2 - y1))
    z_from_width = intrinsics.fx * float(target_diameter_m) / width_px
    z_from_height = intrinsics.fy * float(target_diameter_m) / height_px
    if bbox_dimension == "width":
        return float(z_from_width)
    if bbox_dimension == "height":
        return float(z_from_height)
    if bbox_dimension == "min":
        return float(max(z_from_width, z_from_height))
    if bbox_dimension == "max":
        return float(min(z_from_width, z_from_height))
    return float(0.5 * (z_from_width + z_from_height))


def _quat_apply_np(quat_wxyz: tuple[float, float, float, float], vec_xyz: tuple[float, float, float]) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec_xyz, dtype=np.float64)
    q_vec = q[1:]
    t = 2.0 * np.cross(q_vec, v)
    return v + q[0] * t + np.cross(q_vec, t)


def _quat_conjugate_np(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (float(quat_wxyz[0]), -float(quat_wxyz[1]), -float(quat_wxyz[2]), -float(quat_wxyz[3]))


def _camera_point_to_yaw_base(point_cam: tuple[float, float, float], pose: FramePose) -> tuple[float, float, float]:
    target_pos_w = np.asarray(pose.camera_pos_w, dtype=np.float64) + _quat_apply_np(pose.camera_quat_wxyz, point_cam)
    rel_w = target_pos_w - np.asarray(pose.root_pos_w, dtype=np.float64)

    forward_w = _quat_apply_np(pose.root_quat_wxyz, (1.0, 0.0, 0.0))
    yaw = math.atan2(float(forward_w[1]), float(forward_w[0]))
    yaw_quat = (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))
    pos_base = _quat_apply_np(_quat_conjugate_np(yaw_quat), tuple(float(v) for v in rel_w.tolist()))
    return tuple(float(v) for v in pos_base.tolist())


def load_direction_override_csv(
    path: Path,
    target_object_id: int | None = None,
    target_color_name: str | None = None,
) -> dict[int, tuple[float, float]]:
    rows = _read_rows_by_frame(path, target_object_id=target_object_id, target_color_name=target_color_name)
    aliases = (
        ("refined_yaw", "refined_pitch"),
        ("refined_yaw_rad", "refined_pitch_rad"),
        ("sensor_yaw_rad", "sensor_pitch_rad"),
        ("yaw_cam_rad", "pitch_cam_rad"),
        ("target_yaw_rad", "target_pitch_rad"),
    )
    out: dict[int, tuple[float, float]] = {}
    for frame_index, row in rows.items():
        if not _row_measurement_is_valid(row):
            continue
        for yaw_key, pitch_key in aliases:
            yaw = _parse_float_cell(row.get(yaw_key))
            pitch = _parse_float_cell(row.get(pitch_key))
            if yaw is not None and pitch is not None:
                out[frame_index] = (float(yaw), float(pitch))
                break
    if not out:
        raise ValueError(f"{path} does not contain recognized yaw/pitch override columns.")
    return out


def load_frame_pose_csv(
    path: Path,
    target_object_id: int | None = None,
    target_color_name: str | None = None,
) -> dict[int, FramePose]:
    rows = _read_rows_by_frame(path, target_object_id=target_object_id, target_color_name=target_color_name)
    camera_pos_aliases = (
        ("camera_pos_w_x", "camera_pos_w_y", "camera_pos_w_z"),
        ("cam_pos_w_x", "cam_pos_w_y", "cam_pos_w_z"),
        ("camera_x_w", "camera_y_w", "camera_z_w"),
    )
    camera_quat_aliases = (
        ("camera_quat_w", "camera_quat_x", "camera_quat_y", "camera_quat_z"),
        ("cam_quat_w", "cam_quat_x", "cam_quat_y", "cam_quat_z"),
        ("camera_quat_wxyz_w", "camera_quat_wxyz_x", "camera_quat_wxyz_y", "camera_quat_wxyz_z"),
    )
    root_pos_aliases = (
        ("root_pos_w_x", "root_pos_w_y", "root_pos_w_z"),
        ("robot_pos_w_x", "robot_pos_w_y", "robot_pos_w_z"),
        ("base_pos_w_x", "base_pos_w_y", "base_pos_w_z"),
    )
    root_quat_aliases = (
        ("root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z"),
        ("robot_quat_w", "robot_quat_x", "robot_quat_y", "robot_quat_z"),
        ("base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z"),
    )
    out: dict[int, FramePose] = {}
    for frame_index, row in rows.items():
        if not _row_measurement_is_valid(row):
            continue
        camera_pos = _parse_vec3(row, camera_pos_aliases)
        camera_quat = _parse_quat4(row, camera_quat_aliases)
        root_pos = _parse_vec3(row, root_pos_aliases)
        root_quat = _parse_quat4(row, root_quat_aliases)
        if None in (camera_pos, camera_quat, root_pos, root_quat):
            continue
        out[frame_index] = FramePose(
            camera_pos_w=tuple(float(v) for v in camera_pos),
            camera_quat_wxyz=tuple(float(v) for v in camera_quat),
            root_pos_w=tuple(float(v) for v in root_pos),
            root_quat_wxyz=tuple(float(v) for v in root_quat),
        )
    if not out:
        raise ValueError(f"{path} does not contain recognized camera/root pose columns.")
    return out

def build_positions(
    estimates: list[FrameEstimate],
    intrinsics: CameraIntrinsics,
    distance_by_frame: dict[int, float] | None,
    position_by_frame: dict[int, tuple[float, float, float]] | None,
    constant_distance_m: float | None,
    distance_kind: str,
    target_diameter_m: float | None,
    bbox_dimension: str,
    direction_override_by_frame: dict[int, tuple[float, float]] | None = None,
    frame_pose_by_frame: dict[int, FramePose] | None = None,
    position_source_name: str = "position_csv",
    distance_source_name: str = "distance_csv",
) -> tuple[dict[int, tuple[float, float, float]], dict[int, float], dict[int, str]]:
    positions: dict[int, tuple[float, float, float]] = {}
    distances: dict[int, float] = {}
    sources: dict[int, str] = {}

    for estimate in estimates:
        frame_index = estimate.frame_index
        if position_by_frame is not None and frame_index in position_by_frame:
            positions[frame_index] = position_by_frame[frame_index]
            distances[frame_index] = float(np.linalg.norm(np.asarray(positions[frame_index])))
            sources[frame_index] = position_source_name
            continue

        yaw_pitch = None
        if direction_override_by_frame is not None:
            yaw_pitch = direction_override_by_frame.get(frame_index)
        if yaw_pitch is None and estimate.yaw_rad is not None and estimate.pitch_rad is not None:
            yaw_pitch = (float(estimate.yaw_rad), float(estimate.pitch_rad))
        if yaw_pitch is None:
            continue

        yaw_rad, pitch_rad = yaw_pitch
        value: float | None = None
        source = ""
        kind = distance_kind
        if distance_by_frame is not None and frame_index in distance_by_frame:
            value = distance_by_frame[frame_index]
            source = distance_source_name
        elif constant_distance_m is not None:
            value = float(constant_distance_m)
            source = "constant_distance"
        elif target_diameter_m is not None and estimate.bbox_xyxy is not None:
            value = _estimate_depth_from_bbox(
                bbox_xyxy=estimate.bbox_xyxy,
                intrinsics=intrinsics,
                target_diameter_m=float(target_diameter_m),
                bbox_dimension=bbox_dimension,
            )
            source = "bbox_target_diameter"
            kind = "optical_z"

        if value is None:
            continue

        if kind == "optical_z":
            if estimate.u is None or estimate.v is None:
                continue
            point_cam = tuple(float(v) for v in intrinsics.backproject(estimate.u, estimate.v, value).tolist())
            range_m = float(np.linalg.norm(np.asarray(point_cam, dtype=np.float64)))
        else:
            point_cam = _point_from_range(yaw_rad, pitch_rad, value)
            range_m = float(value)

        if frame_pose_by_frame is not None and frame_index in frame_pose_by_frame:
            point = _camera_point_to_yaw_base(point_cam, frame_pose_by_frame[frame_index])
            if source:
                source = f"{source}+frame_pose"
        else:
            point = point_cam

        positions[frame_index] = point
        distances[frame_index] = range_m
        sources[frame_index] = source

    return positions, distances, sources


def smooth_position_sequence(pos: torch.Tensor, window: int) -> torch.Tensor:
    if window < 1:
        raise ValueError(f"pos_smooth_window must be >= 1, got {window}")
    if window == 1:
        return pos.clone()
    smoothed = pos.clone()
    for t in range(1, pos.shape[1]):
        start = max(0, t + 1 - window)
        smoothed[:, t, :] = pos[:, start : t + 1, :].mean(dim=1)
    return smoothed


def _torch_load_checkpoint(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=device)


def load_velocity_model(model_path: Path, device: torch.device) -> tuple[VelocityPredictorGRU, int]:
    ckpt = _torch_load_checkpoint(model_path, device=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"{model_path} must be a checkpoint produced by test_temp/train_gru_velocity.py "
            "and contain model_state_dict."
        )

    cfg = ckpt.get("config", {})
    model = VelocityPredictorGRU(
        input_dim=int(cfg.get("input_dim", 13)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    input_norm = ckpt.get("input_norm")
    if input_norm is not None:
        mean = input_norm["mean"].to(device=device, dtype=torch.float32)
        std = input_norm["std"].to(device=device, dtype=torch.float32)
        model.set_input_norm_stats(mean=mean, std=std)

    preprocessing = ckpt.get("preprocessing", {})
    train_args = ckpt.get("train_args", {})
    pos_smooth_window = int(preprocessing.get("pos_smooth_window", train_args.get("pos_smooth_window", 1)))
    model.eval()
    return model, pos_smooth_window


def build_model_input(
    frame_indices: list[int],
    positions: dict[int, tuple[float, float, float]],
    robot_motion_by_frame: dict[int, RobotMotion],
    pos_smooth_window: int,
) -> torch.Tensor:
    pos = torch.tensor(
        [[positions[frame_index] for frame_index in frame_indices]],
        dtype=torch.float32,
    )
    robot_lin = torch.tensor(
        [[robot_motion_by_frame.get(frame_index, _default_robot_motion()).lin_vel_mps for frame_index in frame_indices]],
        dtype=torch.float32,
    )
    robot_ang = torch.tensor(
        [[robot_motion_by_frame.get(frame_index, _default_robot_motion()).ang_vel_radps for frame_index in frame_indices]],
        dtype=torch.float32,
    )
    dog_flag = torch.tensor(
        [
            [
                [robot_motion_by_frame.get(frame_index, _default_robot_motion()).dog_moving_flag]
                for frame_index in frame_indices
            ]
        ],
        dtype=torch.float32,
    )

    smoothed_pos = smooth_position_sequence(pos, pos_smooth_window)
    pos_abs = smoothed_pos[:, 1:, :]
    pos_diff = smoothed_pos[:, 1:, :] - smoothed_pos[:, :-1, :]
    return torch.cat(
        [
            pos_abs,
            pos_diff,
            robot_lin[:, 1:, :],
            robot_ang[:, 1:, :],
            dog_flag[:, 1:, :],
        ],
        dim=-1,
    ).contiguous()


def finite_diff_velocity(
    frame_indices: list[int],
    positions: dict[int, tuple[float, float, float]],
    fps: float,
) -> tuple[float, float, float] | None:
    if len(frame_indices) < 2:
        return None
    a = frame_indices[-2]
    b = frame_indices[-1]
    if a not in positions or b not in positions:
        return None
    dt = max(1, b - a) / float(fps)
    if dt <= 0.0:
        return None
    pa = np.asarray(positions[a], dtype=np.float64)
    pb = np.asarray(positions[b], dtype=np.float64)
    vel = (pb - pa) / dt
    return tuple(float(v) for v in vel.tolist())


@torch.no_grad()
def estimate_velocity_windows(
    estimates: list[FrameEstimate],
    positions: dict[int, tuple[float, float, float]],
    robot_motion_by_frame: dict[int, RobotMotion],
    model: VelocityPredictorGRU | None,
    device: torch.device,
    fps: float,
    sequence_length: int,
    pos_smooth_window: int,
) -> dict[int, VelocityEstimate]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2.")

    valid_direction_frames = [
        estimate.frame_index
        for estimate in estimates
        if estimate.valid_measurement and estimate.frame_index in positions
    ]
    valid_set = set(valid_direction_frames)
    out: dict[int, VelocityEstimate] = {}

    for estimate in estimates:
        frame_index = estimate.frame_index
        window = list(range(frame_index - sequence_length + 1, frame_index + 1))
        has_full_window = all(idx in valid_set for idx in window)
        fd_vel = finite_diff_velocity(window[-2:], positions, fps=fps) if has_full_window else None

        if not has_full_window:
            out[frame_index] = VelocityEstimate(
                frame_index=frame_index,
                valid=False,
                window_start_frame=None,
                window_size=0,
                model_velocity_mps=None,
                finite_diff_velocity_mps=fd_vel,
            )
            continue

        model_vel: tuple[float, float, float] | None = None
        if model is not None:
            x = build_model_input(
                frame_indices=window,
                positions=positions,
                robot_motion_by_frame=robot_motion_by_frame,
                pos_smooth_window=pos_smooth_window,
            ).to(device=device, dtype=torch.float32)
            pred = model(x)[0].detach().cpu().to(torch.float32).numpy()
            model_vel = tuple(float(v) for v in pred.tolist())

        out[frame_index] = VelocityEstimate(
            frame_index=frame_index,
            valid=model_vel is not None,
            window_start_frame=window[0],
            window_size=len(window),
            model_velocity_mps=model_vel,
            finite_diff_velocity_mps=fd_vel,
        )
    return out


def _fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.9g}"
    return value


def write_combined_csv(
    estimates: list[FrameEstimate],
    positions: dict[int, tuple[float, float, float]],
    distances: dict[int, float],
    distance_sources: dict[int, str],
    velocities: dict[int, VelocityEstimate],
    out_csv: Path,
) -> None:
    fields = [
        "frame_index",
        "timestamp_s",
        "valid_direction",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "u",
        "v",
        "yaw_rad",
        "pitch_rad",
        "yaw_deg",
        "pitch_deg",
        "yaw_rate_radps",
        "pitch_rate_radps",
        "track_time_s",
        "distance_m",
        "distance_source",
        "x_input_m",
        "y_input_m",
        "z_input_m",
        "valid_velocity",
        "velocity_window_start_frame",
        "velocity_window_size",
        "vx_model_mps",
        "vy_model_mps",
        "vz_model_mps",
        "speed_model_mps",
        "vx_finite_diff_mps",
        "vy_finite_diff_mps",
        "vz_finite_diff_mps",
        "speed_finite_diff_mps",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for estimate in estimates:
            bbox = estimate.bbox_xyxy or (None, None, None, None)
            pos = positions.get(estimate.frame_index)
            if pos is None:
                pos = (None, None, None)
            velocity = velocities.get(
                estimate.frame_index,
                VelocityEstimate(
                    frame_index=estimate.frame_index,
                    valid=False,
                    window_start_frame=None,
                    window_size=0,
                    model_velocity_mps=None,
                    finite_diff_velocity_mps=None,
                ),
            )
            model_vel = velocity.model_velocity_mps or (None, None, None)
            fd_vel = velocity.finite_diff_velocity_mps or (None, None, None)
            yaw_deg = None if estimate.yaw_rad is None else math.degrees(estimate.yaw_rad)
            pitch_deg = None if estimate.pitch_rad is None else math.degrees(estimate.pitch_rad)
            writer.writerow(
                {
                    "frame_index": estimate.frame_index,
                    "timestamp_s": _fmt(estimate.timestamp_s),
                    "valid_direction": int(estimate.valid_measurement),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "u": _fmt(estimate.u),
                    "v": _fmt(estimate.v),
                    "yaw_rad": _fmt(estimate.yaw_rad),
                    "pitch_rad": _fmt(estimate.pitch_rad),
                    "yaw_deg": _fmt(yaw_deg),
                    "pitch_deg": _fmt(pitch_deg),
                    "yaw_rate_radps": _fmt(estimate.yaw_rate_radps),
                    "pitch_rate_radps": _fmt(estimate.pitch_rate_radps),
                    "track_time_s": _fmt(estimate.track_time_s),
                    "distance_m": _fmt(distances.get(estimate.frame_index)),
                    "distance_source": distance_sources.get(estimate.frame_index, ""),
                    "x_input_m": _fmt(pos[0]),
                    "y_input_m": _fmt(pos[1]),
                    "z_input_m": _fmt(pos[2]),
                    "valid_velocity": int(velocity.valid),
                    "velocity_window_start_frame": "" if velocity.window_start_frame is None else velocity.window_start_frame,
                    "velocity_window_size": velocity.window_size,
                    "vx_model_mps": _fmt(model_vel[0]),
                    "vy_model_mps": _fmt(model_vel[1]),
                    "vz_model_mps": _fmt(model_vel[2]),
                    "speed_model_mps": _fmt(velocity.model_speed_mps),
                    "vx_finite_diff_mps": _fmt(fd_vel[0]),
                    "vy_finite_diff_mps": _fmt(fd_vel[1]),
                    "vz_finite_diff_mps": _fmt(fd_vel[2]),
                    "speed_finite_diff_mps": _fmt(velocity.finite_diff_speed_mps),
                }
            )



def _safe_name(value: Any, default: str = "target") -> str:
    text = str(value or default).strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    safe = "".join(chars).strip("_")
    return safe or default


@dataclass(frozen=True)
class TargetRun:
    seq_dir: Path
    sequence_name: str
    target_name: str
    initial_box: list[float] | None
    initial_box_file: Path | None
    target_object_id: int | None
    target_color_name: str
    target_color_rgb: str
    position_csv: Path | None
    robot_motion_csv: Path | None
    distance_csv: Path | None
    direction_csv: Path | None
    frame_pose_csv: Path | None
    gt_velocity_csv: Path | None
    input_features_source: str


@dataclass(frozen=True)
class SequenceRunResult:
    sequence_name: str
    target_name: str
    seq_dir: Path
    out_csv: Path
    overlay_dir: Path | None
    frames: int
    tracked_boxes: int
    valid_directions: int
    valid_positions: int
    valid_velocities: int
    total_track_time_s: float
    mean_track_time_ms: float
    track_fps: float
    pos_smooth_window: int
    target_object_id: int | None
    target_color_name: str
    input_features_source: str


def _read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_seq_dir(dataset_dir: Path, row: dict[str, str]) -> Path:
    seq_dir_text = str(row.get("seq_dir", "")).strip()
    if seq_dir_text:
        seq_dir = Path(seq_dir_text)
        if seq_dir.is_absolute():
            return seq_dir.resolve()
        if seq_dir.exists():
            return seq_dir.resolve()
        candidate = dataset_dir / "sequences" / seq_dir.name
        if candidate.exists():
            return candidate.resolve()
        candidate = dataset_dir / seq_dir
        if candidate.exists():
            return candidate.resolve()
    seq_id = str(row.get("sequence_id", "")).strip()
    env_id = str(row.get("env_id", "")).strip()
    if seq_id and env_id:
        name = f"seq_{int(float(seq_id)):06d}_env_{int(float(env_id)):03d}"
        return (dataset_dir / "sequences" / name).resolve()
    raise RuntimeError(f"Cannot resolve sequence directory from manifest row: {row}")


def _resolve_row_path(dataset_dir: Path, seq_dir: Path, value: str | None, default_name: str) -> Path:
    value_text = str(value or "").strip()
    if value_text:
        path = Path(value_text)
        if path.is_absolute():
            return path.resolve()
        if path.exists():
            return path.resolve()
        candidate = seq_dir / path.name
        if candidate.exists():
            return candidate.resolve()
        candidate = dataset_dir / path
        if candidate.exists():
            return candidate.resolve()
    return (seq_dir / default_name).resolve()


def _read_initial_box_file(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8").strip()
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError(f"{path} must contain exactly four numbers.")
    return values


def _read_initial_boxes_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_id = _parse_optional_int(row.get("object_id"))
            if object_id is None:
                continue
            try:
                box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "target_object_id": int(object_id),
                    "target_color_name": str(row.get("target_color_name", "")).strip(),
                    "target_color_rgb": str(row.get("target_color_rgb", "")).strip(),
                    "initial_box": box,
                }
            )
    return rows


def _target_suffix(object_id: int | None, color_name: str) -> str:
    safe_color = _safe_name(color_name, default="target")
    if object_id is None:
        return safe_color
    return f"obj_{int(object_id):02d}_{safe_color}"


def _matches_requested_target(args: argparse.Namespace, object_id: int | None, color_name: str) -> bool:
    if args.target_object_id is not None and object_id != int(args.target_object_id):
        return False
    if args.target_color_name is not None:
        requested = {str(name).strip().lower() for name in args.target_color_name}
        if str(color_name or "").strip().lower() not in requested:
            return False
    return True


def _infer_fps_from_dataset(dataset_dir: Path, fallback: float) -> float:
    meta_path = dataset_dir / "dataset_meta.json"
    if not meta_path.exists():
        return float(fallback)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return float(fallback)
    fps = meta.get("fps")
    if fps is None:
        frame_dt_s = meta.get("frame_dt_s")
        if frame_dt_s:
            fps = 1.0 / float(frame_dt_s)
    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        return float(fallback)
    return fps_value if fps_value > 0.0 else float(fallback)


def _collect_dataset_runs(args: argparse.Namespace) -> list[TargetRun]:
    dataset_dir = args.dataset_dir.resolve()
    rows = _read_manifest(dataset_dir)
    complete_rows = [row for row in rows if str(row.get("status", "")).strip().lower() == "complete"]
    runs: list[TargetRun] = []
    selected_sequences = 0

    for row in complete_rows:
        seq_dir = _resolve_seq_dir(dataset_dir, row)
        rgb_dir = seq_dir / "rgb"
        if not rgb_dir.exists():
            continue
        selected_sequences += 1
        sequence_name = seq_dir.name
        initial_boxes_csv = _resolve_row_path(dataset_dir, seq_dir, row.get("initial_boxes_csv"), "initial_boxes.csv")
        velocity_features_csv = _resolve_row_path(
            dataset_dir,
            seq_dir,
            row.get("velocity_features_csv"),
            "velocity_features.csv",
        )
        sensor_features_csv = seq_dir / str(args.sensor_features_name)
        input_features_csv = sensor_features_csv if sensor_features_csv.exists() else velocity_features_csv
        input_features_source = "sensor_features_csv" if sensor_features_csv.exists() else "velocity_features_csv"
        position_csv = input_features_csv if input_features_csv.exists() else None
        robot_motion_csv = input_features_csv if input_features_csv.exists() else None
        distance_csv = args.distance_csv if args.distance_csv is not None else (input_features_csv if input_features_csv.exists() else None)
        direction_csv = input_features_csv if input_features_csv.exists() else None
        frame_pose_csv = input_features_csv if input_features_csv.exists() else None
        gt_velocity_csv = velocity_features_csv if velocity_features_csv.exists() else None

        box_rows = _read_initial_boxes_csv(initial_boxes_csv)
        if box_rows:
            for box_row in box_rows:
                object_id = int(box_row["target_object_id"])
                color_name = str(box_row.get("target_color_name", ""))
                if not _matches_requested_target(args, object_id, color_name):
                    continue
                target_name = f"{sequence_name}_{_target_suffix(object_id, color_name)}"
                runs.append(
                    TargetRun(
                        seq_dir=seq_dir,
                        sequence_name=sequence_name,
                        target_name=target_name,
                        initial_box=[float(v) for v in box_row["initial_box"]],
                        initial_box_file=None,
                        target_object_id=object_id,
                        target_color_name=color_name,
                        target_color_rgb=str(box_row.get("target_color_rgb", "")),
                        position_csv=position_csv,
                        robot_motion_csv=robot_motion_csv,
                        distance_csv=distance_csv,
                        direction_csv=direction_csv,
                        frame_pose_csv=frame_pose_csv,
                        gt_velocity_csv=gt_velocity_csv,
                        input_features_source=input_features_source,
                    )
                )
        else:
            object_id = _parse_optional_int(row.get("target_object_id"))
            color_name = str(row.get("target_color_name", ""))
            if not _matches_requested_target(args, object_id, color_name):
                continue
            initial_box_file = _resolve_row_path(dataset_dir, seq_dir, row.get("initial_box_file"), "initial_box.txt")
            if not initial_box_file.exists():
                raise FileNotFoundError(f"initial_box.txt not found: {initial_box_file}")
            target_name = f"{sequence_name}_{_target_suffix(object_id, color_name)}"
            runs.append(
                TargetRun(
                    seq_dir=seq_dir,
                    sequence_name=sequence_name,
                    target_name=target_name,
                    initial_box=None,
                    initial_box_file=initial_box_file,
                    target_object_id=object_id,
                    target_color_name=color_name,
                    target_color_rgb=str(row.get("target_color_rgb", "")),
                    position_csv=position_csv,
                    robot_motion_csv=robot_motion_csv,
                    distance_csv=distance_csv,
                    direction_csv=direction_csv,
                    frame_pose_csv=frame_pose_csv,
                    gt_velocity_csv=gt_velocity_csv,
                    input_features_source=input_features_source,
                )
            )

        if args.max_sequences is not None and selected_sequences >= int(args.max_sequences):
            break

    print(
        "[test12+temp dataset] manifest",
        f"rows={len(rows)}",
        f"complete_sequences={len(complete_rows)}",
        f"selected_sequences={selected_sequences}",
        f"target_runs={len(runs)}",
    )
    if args.target_color_name is not None:
        print("[test12+temp dataset] color filter", f"target_color_name={args.target_color_name}")
    if args.target_object_id is not None:
        print("[test12+temp dataset] object filter", f"target_object_id={int(args.target_object_id)}")
    return runs


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine test12 direction tracking with test_temp GRU velocity prediction. "
            "Pass --dataset_dir to evaluate every collected sequence, or --rgb_dir for one sequence."
        )
    )
    parser.add_argument("--dataset_dir", type=Path, default=None, help="Collected dataset root containing manifest.csv.")
    parser.add_argument("--out_dir", type=Path, default=Path("test12_temp_outputs/dataset_eval"))
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--target_color_name", nargs="+", default=None)
    parser.add_argument("--target_object_id", type=int, default=None)
    parser.add_argument("--sensor_features_name", type=str, default="sensor_features.csv", help="Per-sequence sensor feature CSV name used before velocity_features.csv in dataset mode.")
    parser.add_argument("--force", action="store_true", help="Re-run existing per-sequence output CSV files in dataset mode.")

    parser.add_argument("--rgb_dir", type=Path, default=None, help="Directory of sorted RGB frames for single-sequence mode.")
    parser.add_argument("--out_csv", type=Path, default=Path("test12_temp_outputs/estimates.csv"))
    parser.add_argument("--overlay_dir", type=Path, default=None)
    parser.add_argument("--overlay", action="store_true", help="In dataset mode, write overlays under out_dir/overlays.")
    parser.add_argument("--fps", type=float, default=50.0)
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
    tracker.add_argument("--initial_box", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    tracker.add_argument("--initial_box_file", type=Path, default=None)
    tracker.add_argument("--ostrack_repo", required=True, type=Path)
    tracker.add_argument("--ostrack_checkpoint", required=True, type=Path)
    tracker.add_argument("--ostrack_config", type=str, default="vitb_256_mae_ce_32x4_ep300")
    tracker.add_argument("--device", type=str, default="cuda:0")
    tracker.add_argument("--debug", type=int, default=0)

    geometry = parser.add_argument_group("3D position source for velocity")
    geometry.add_argument(
        "--distance_csv",
        type=Path,
        default=None,
        help="Optional CSV with frame_index plus distance_m/range_m/depth_m/z_cam_m.",
    )
    geometry.add_argument("--position_csv", type=Path, default=None, help="Optional CSV with frame_index and base-frame 3D positions.")
    geometry.add_argument("--direction_csv", type=Path, default=None, help="Optional CSV with sensor/refined yaw/pitch columns.")
    geometry.add_argument("--frame_pose_csv", type=Path, default=None, help="Optional CSV with camera/root world pose for sensor distance -> base position reconstruction.")
    geometry.add_argument("--constant_distance_m", type=float, default=None)
    geometry.add_argument(
        "--distance_kind",
        choices=("range", "optical_z"),
        default="range",
        help="How to interpret --distance_csv/--constant_distance_m values.",
    )
    geometry.add_argument(
        "--target_diameter_m",
        type=float,
        default=None,
        help="Estimate optical depth from tracked bbox size and known target diameter.",
    )
    geometry.add_argument(
        "--bbox_dimension",
        choices=("width", "height", "mean", "min", "max"),
        default="mean",
        help="BBox dimension used with --target_diameter_m.",
    )

    velocity = parser.add_argument_group("test_temp velocity model")
    velocity.add_argument("--velocity_model_path", type=Path, default=None, help="Path to test_temp best_model.pt.")
    velocity.add_argument("--robot_motion_csv", type=Path, default=None)
    velocity.add_argument("--sequence_length", type=int, default=9)
    velocity.add_argument(
        "--pos_smooth_window",
        type=int,
        default=None,
        help="Override checkpoint pos_smooth_window. Defaults to checkpoint preprocessing.",
    )
    velocity.add_argument(
        "--allow_missing_velocity_inputs",
        action="store_true",
        help="Write direction-only rows instead of failing when no 3D position source is provided.",
    )
    return parser.parse_args()


def _run_sequence(
    args: argparse.Namespace,
    rgb_dir: Path,
    intrinsics_json: Path | None,
    initial_box: list[float] | None,
    initial_box_file: Path | None,
    out_csv: Path,
    overlay_dir: Path | None,
    tracker: OSTrackBoxTracker,
    model: VelocityPredictorGRU | None,
    device: torch.device,
    pos_smooth_window: int,
    fps: float,
    position_csv: Path | None,
    distance_csv: Path | None,
    robot_motion_csv: Path | None,
    direction_csv: Path | None = None,
    frame_pose_csv: Path | None = None,
    input_features_source: str = "position_csv",
    target_object_id: int | None = None,
    target_color_name: str | None = None,
) -> SequenceRunResult:
    if fps <= 0:
        raise ValueError("fps must be positive.")

    intrinsics_args = argparse.Namespace(
        intrinsics_json=intrinsics_json,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=args.width,
        height=args.height,
    )
    intrinsics = build_intrinsics(intrinsics_args)
    if initial_box is None:
        if initial_box_file is None:
            box_args = argparse.Namespace(initial_box=args.initial_box, initial_box_file=args.initial_box_file)
            initial_box = load_initial_box(box_args)
        else:
            initial_box = _read_initial_box_file(initial_box_file)

    rgb_paths = load_rgb_paths(rgb_dir)
    if args.max_frames is not None:
        rgb_paths = rgb_paths[: int(args.max_frames)]

    tracks = list(tracker.track(rgb_paths=rgb_paths, initial_box_xyxy=initial_box))
    tracks_by_frame = tracked_boxes_to_map(tracks)
    estimates = estimate_target_direction_from_boxes(
        rgb_paths=rgb_paths,
        tracks_by_frame=tracks_by_frame,
        intrinsics=intrinsics,
        fps=fps,
    )

    distance_by_frame = None
    if distance_csv is not None:
        try:
            distance_by_frame = load_distance_csv(
                distance_csv,
                target_object_id=target_object_id,
                target_color_name=target_color_name,
            )
        except ValueError:
            if distance_csv != position_csv:
                raise
    position_by_frame = None
    if position_csv is not None:
        try:
            position_by_frame = load_position_csv(
                position_csv,
                target_object_id=target_object_id,
                target_color_name=target_color_name,
            )
        except ValueError:
            if position_csv != distance_csv:
                raise
    direction_override_by_frame = None
    if direction_csv is not None:
        try:
            direction_override_by_frame = load_direction_override_csv(
                direction_csv,
                target_object_id=target_object_id,
                target_color_name=target_color_name,
            )
        except ValueError:
            if direction_csv not in (position_csv, distance_csv):
                raise
    frame_pose_by_frame = None
    if frame_pose_csv is not None:
        try:
            frame_pose_by_frame = load_frame_pose_csv(
                frame_pose_csv,
                target_object_id=target_object_id,
                target_color_name=target_color_name,
            )
        except ValueError:
            if frame_pose_csv not in (position_csv, distance_csv, direction_csv):
                raise

    positions, distances, distance_sources = build_positions(
        estimates=estimates,
        intrinsics=intrinsics,
        distance_by_frame=distance_by_frame,
        position_by_frame=position_by_frame,
        constant_distance_m=args.constant_distance_m,
        distance_kind=args.distance_kind,
        target_diameter_m=args.target_diameter_m,
        bbox_dimension=args.bbox_dimension,
        direction_override_by_frame=direction_override_by_frame,
        frame_pose_by_frame=frame_pose_by_frame,
        position_source_name=input_features_source,
        distance_source_name=input_features_source,
    )

    if args.velocity_model_path is not None and not positions and not args.allow_missing_velocity_inputs:
        raise RuntimeError(
            "Velocity prediction needs 3D target positions. Provide one of "
            "--position_csv, --distance_csv, --constant_distance_m, or --target_diameter_m. "
            "Use --allow_missing_velocity_inputs to write direction-only output."
        )

    robot_motion_by_frame = (
        load_robot_motion_csv(robot_motion_csv, target_object_id=target_object_id, target_color_name=target_color_name)
        if robot_motion_csv is not None
        else {}
    )
    velocities = estimate_velocity_windows(
        estimates=estimates,
        positions=positions,
        robot_motion_by_frame=robot_motion_by_frame,
        model=model if positions else None,
        device=device,
        fps=fps,
        sequence_length=int(args.sequence_length),
        pos_smooth_window=pos_smooth_window,
    )
    write_combined_csv(
        estimates=estimates,
        positions=positions,
        distances=distances,
        distance_sources=distance_sources,
        velocities=velocities,
        out_csv=out_csv,
    )

    if overlay_dir is not None:
        write_overlays(rgb_paths, tracks_by_frame, estimates, overlay_dir)

    valid_direction_count = sum(int(e.valid_measurement) for e in estimates)
    valid_position_count = len(positions)
    valid_velocity_count = sum(int(v.valid) for v in velocities.values())
    ok_track_count = sum(int(t.ok and t.bbox_xyxy is not None) for t in tracks)
    timing = track_timing_stats(tracks)
    return SequenceRunResult(
        sequence_name=rgb_dir.parent.name,
        target_name=out_csv.stem,
        seq_dir=rgb_dir.parent,
        out_csv=out_csv,
        overlay_dir=overlay_dir,
        frames=len(estimates),
        tracked_boxes=ok_track_count,
        valid_directions=valid_direction_count,
        valid_positions=valid_position_count,
        valid_velocities=valid_velocity_count,
        total_track_time_s=float(timing["total_track_time_s"]),
        mean_track_time_ms=float(timing["mean_track_time_ms"]),
        track_fps=float(timing["track_fps"]),
        pos_smooth_window=pos_smooth_window,
        target_object_id=target_object_id,
        target_color_name=str(target_color_name or ""),
        input_features_source=str(input_features_source),
    )


def _prepare_model(args: argparse.Namespace) -> tuple[VelocityPredictorGRU | None, torch.device, int]:
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    model: VelocityPredictorGRU | None = None
    pos_smooth_window = int(args.pos_smooth_window) if args.pos_smooth_window is not None else 1
    if args.velocity_model_path is not None:
        model, ckpt_pos_smooth_window = load_velocity_model(args.velocity_model_path, device=device)
        if args.pos_smooth_window is None:
            pos_smooth_window = ckpt_pos_smooth_window
    return model, device, pos_smooth_window



def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _parse_boolish(value: Any, default: bool = True) -> bool:
    text = str(value or "").strip()
    if text == "":
        return bool(default)
    try:
        return bool(int(float(text)))
    except ValueError:
        return text.lower() in {"true", "yes", "y"}


def _empty_metric_row() -> dict[str, Any]:
    return {
        "direction_eval_frames": "",
        "yaw_mae_deg": "",
        "pitch_mae_deg": "",
        "angular_mae_deg": "",
        "yaw_rmse_deg": "",
        "pitch_rmse_deg": "",
        "angular_rmse_deg": "",
        "velocity_eval_frames": "",
        "vx_mae_mps": "",
        "vy_mae_mps": "",
        "vz_mae_mps": "",
        "velocity_mae_mps": "",
        "velocity_rmse_mps": "",
    }


def _compute_direction_metrics(
    est_csv: Path,
    gt_csv: Path | None,
    target_object_id: int | None,
    target_color_name: str | None,
) -> dict[str, Any]:
    out = _empty_metric_row()
    if gt_csv is None or not gt_csv.exists() or not est_csv.exists():
        return out

    est_rows = _read_rows_by_frame(est_csv)
    gt_rows = _read_rows_by_frame(gt_csv, target_object_id=target_object_id, target_color_name=target_color_name)
    yaw_errs: list[float] = []
    pitch_errs: list[float] = []
    for frame_index in sorted(set(est_rows) & set(gt_rows)):
        est = est_rows[frame_index]
        if not _parse_boolish(est.get("valid_direction", est.get("valid_measurement", "1")), default=True):
            continue
        est_yaw = _parse_float(est.get("yaw_rad"))
        est_pitch = _parse_float(est.get("pitch_rad"))
        gt_yaw = _parse_float(gt_rows[frame_index].get("yaw_rad"))
        gt_pitch = _parse_float(gt_rows[frame_index].get("pitch_rad"))
        if None in (est_yaw, est_pitch, gt_yaw, gt_pitch):
            continue
        yaw_errs.append(math.atan2(math.sin(float(est_yaw) - float(gt_yaw)), math.cos(float(est_yaw) - float(gt_yaw))))
        pitch_errs.append(float(est_pitch) - float(gt_pitch))

    if not yaw_errs:
        return out
    yaw = np.asarray(yaw_errs, dtype=np.float64)
    pitch = np.asarray(pitch_errs, dtype=np.float64)
    angular = np.sqrt(yaw * yaw + pitch * pitch)
    out.update(
        {
            "direction_eval_frames": int(len(yaw)),
            "yaw_mae_deg": f"{math.degrees(float(np.mean(np.abs(yaw)))):.6f}",
            "pitch_mae_deg": f"{math.degrees(float(np.mean(np.abs(pitch)))):.6f}",
            "angular_mae_deg": f"{math.degrees(float(np.mean(angular))):.6f}",
            "yaw_rmse_deg": f"{math.degrees(float(np.sqrt(np.mean(yaw * yaw)))):.6f}",
            "pitch_rmse_deg": f"{math.degrees(float(np.sqrt(np.mean(pitch * pitch)))):.6f}",
            "angular_rmse_deg": f"{math.degrees(float(np.sqrt(np.mean(angular * angular)))):.6f}",
        }
    )
    return out


def _compute_velocity_metrics(
    est_csv: Path,
    gt_csv: Path | None,
    target_object_id: int | None,
    target_color_name: str | None,
) -> dict[str, Any]:
    out = _empty_metric_row()
    if gt_csv is None or not gt_csv.exists() or not est_csv.exists():
        return out

    est_rows = _read_rows_by_frame(est_csv)
    gt_rows = _read_rows_by_frame(gt_csv, target_object_id=target_object_id, target_color_name=target_color_name)
    errors: list[tuple[float, float, float]] = []
    for frame_index in sorted(set(est_rows) & set(gt_rows)):
        est = est_rows[frame_index]
        if not _parse_boolish(est.get("valid_velocity", "0"), default=False):
            continue
        pred = (
            _parse_float(est.get("vx_model_mps")),
            _parse_float(est.get("vy_model_mps")),
            _parse_float(est.get("vz_model_mps")),
        )
        gt = (
            _parse_float(gt_rows[frame_index].get("target_velocity_base_x")),
            _parse_float(gt_rows[frame_index].get("target_velocity_base_y")),
            _parse_float(gt_rows[frame_index].get("target_velocity_base_z")),
        )
        if any(v is None for v in pred + gt):
            continue
        errors.append(tuple(float(p) - float(g) for p, g in zip(pred, gt)))

    if not errors:
        return out
    err = np.asarray(errors, dtype=np.float64)
    norm = np.linalg.norm(err, axis=1)
    out.update(
        {
            "velocity_eval_frames": int(err.shape[0]),
            "vx_mae_mps": f"{float(np.mean(np.abs(err[:, 0]))):.6f}",
            "vy_mae_mps": f"{float(np.mean(np.abs(err[:, 1]))):.6f}",
            "vz_mae_mps": f"{float(np.mean(np.abs(err[:, 2]))):.6f}",
            "velocity_mae_mps": f"{float(np.mean(norm)):.6f}",
            "velocity_rmse_mps": f"{float(np.sqrt(np.mean(norm * norm))):.6f}",
        }
    )
    return out


def _compute_dataset_metrics(run: TargetRun, out_csv: Path) -> dict[str, Any]:
    metrics = _empty_metric_row()
    direction_metrics = _compute_direction_metrics(
        est_csv=out_csv,
        gt_csv=run.seq_dir / "ground_truth_direction.csv",
        target_object_id=run.target_object_id,
        target_color_name=run.target_color_name,
    )
    velocity_metrics = _compute_velocity_metrics(
        est_csv=out_csv,
        gt_csv=run.gt_velocity_csv,
        target_object_id=run.target_object_id,
        target_color_name=run.target_color_name,
    )
    metrics.update(direction_metrics)
    for key, value in velocity_metrics.items():
        if value != "":
            metrics[key] = value
    return metrics


def _summarize_existing_result(
    run: TargetRun,
    out_csv: Path,
    overlay_dir: Path | None,
    pos_smooth_window: int,
) -> SequenceRunResult:
    rows = _read_rows_by_frame(out_csv) if out_csv.exists() else {}
    track_times = [
        value
        for value in (_parse_float(row.get("track_time_s")) for row in rows.values())
        if value is not None
    ]
    total_time = float(sum(track_times))
    return SequenceRunResult(
        sequence_name=run.sequence_name,
        target_name=run.target_name,
        seq_dir=run.seq_dir,
        out_csv=out_csv,
        overlay_dir=overlay_dir,
        frames=len(rows),
        tracked_boxes=sum(int(_parse_boolish(row.get("valid_direction", "0"), default=False)) for row in rows.values()),
        valid_directions=sum(int(_parse_boolish(row.get("valid_direction", "0"), default=False)) for row in rows.values()),
        valid_positions=sum(int(_parse_float(row.get("x_input_m")) is not None) for row in rows.values()),
        valid_velocities=sum(int(_parse_boolish(row.get("valid_velocity", "0"), default=False)) for row in rows.values()),
        total_track_time_s=total_time,
        mean_track_time_ms=(total_time / len(track_times) * 1000.0) if track_times else 0.0,
        track_fps=(len(track_times) / total_time) if total_time > 0.0 else 0.0,
        pos_smooth_window=int(pos_smooth_window),
        target_object_id=run.target_object_id,
        target_color_name=run.target_color_name,
        input_features_source=run.input_features_source,
    )

def _summary_row(result: SequenceRunResult) -> dict[str, Any]:
    return {
        "sequence": result.sequence_name,
        "target_run": result.target_name,
        "seq_dir": str(result.seq_dir),
        "out_csv": str(result.out_csv),
        "overlay_dir": "" if result.overlay_dir is None else str(result.overlay_dir),
        "target_object_id": "" if result.target_object_id is None else int(result.target_object_id),
        "target_color_name": result.target_color_name,
        "input_features_source": result.input_features_source,
        "frames": int(result.frames),
        "tracked_boxes": int(result.tracked_boxes),
        "valid_directions": int(result.valid_directions),
        "valid_positions": int(result.valid_positions),
        "valid_velocities": int(result.valid_velocities),
        "total_track_time_s": f"{result.total_track_time_s:.6f}",
        "mean_track_time_ms": f"{result.mean_track_time_ms:.6f}",
        "track_fps": f"{result.track_fps:.6f}",
        "pos_smooth_window": int(result.pos_smooth_window),
    }



def _overall_summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def _num(row: dict[str, Any], key: str) -> float | None:
        return _parse_float(row.get(key))

    def _count(row: dict[str, Any], key: str) -> int:
        value = _num(row, key)
        return 0 if value is None else int(value)

    def _sum_count(key: str) -> int:
        return sum(_count(row, key) for row in rows)

    def _weighted_mean(key: str, weight_key: str) -> float | None:
        total_weight = 0.0
        total_value = 0.0
        for row in rows:
            value = _num(row, key)
            weight = _num(row, weight_key)
            if value is None or weight is None or weight <= 0.0:
                continue
            total_value += value * weight
            total_weight += weight
        if total_weight <= 0.0:
            return None
        return total_value / total_weight

    def _combined_rmse(key: str, weight_key: str) -> float | None:
        total_weight = 0.0
        total_sq = 0.0
        for row in rows:
            value = _num(row, key)
            weight = _num(row, weight_key)
            if value is None or weight is None or weight <= 0.0:
                continue
            total_sq += value * value * weight
            total_weight += weight
        if total_weight <= 0.0:
            return None
        return math.sqrt(total_sq / total_weight)

    def _fmt_metric(value: float | None) -> str:
        return "" if value is None else f"{value:.6f}"

    total_frames = _sum_count("frames")
    total_track_time_s = sum((_num(row, "total_track_time_s") or 0.0) for row in rows)
    out = {key: "" for key in rows[0].keys()}
    out.update(
        {
            "sequence": "ALL_WEIGHTED_BY_FRAMES",
            "target_run": "ALL",
            "seq_dir": "",
            "out_csv": "",
            "overlay_dir": "",
            "target_object_id": "ALL",
            "target_color_name": "ALL",
            "input_features_source": "mixed" if len({str(row.get("input_features_source", "")) for row in rows}) > 1 else rows[0].get("input_features_source", ""),
            "frames": total_frames,
            "tracked_boxes": _sum_count("tracked_boxes"),
            "valid_directions": _sum_count("valid_directions"),
            "valid_positions": _sum_count("valid_positions"),
            "valid_velocities": _sum_count("valid_velocities"),
            "total_track_time_s": f"{total_track_time_s:.6f}",
            "mean_track_time_ms": _fmt_metric(_weighted_mean("mean_track_time_ms", "frames")),
            "track_fps": f"{(total_frames / total_track_time_s) if total_track_time_s > 0.0 else 0.0:.6f}",
            "pos_smooth_window": "mixed" if len({str(row.get("pos_smooth_window", "")) for row in rows}) > 1 else rows[0].get("pos_smooth_window", ""),
            "direction_eval_frames": _sum_count("direction_eval_frames"),
            "velocity_eval_frames": _sum_count("velocity_eval_frames"),
        }
    )

    for key in ("yaw_mae_deg", "pitch_mae_deg", "angular_mae_deg"):
        out[key] = _fmt_metric(_weighted_mean(key, "direction_eval_frames"))
    for key in ("yaw_rmse_deg", "pitch_rmse_deg", "angular_rmse_deg"):
        out[key] = _fmt_metric(_combined_rmse(key, "direction_eval_frames"))
    for key in ("vx_mae_mps", "vy_mae_mps", "vz_mae_mps", "velocity_mae_mps"):
        out[key] = _fmt_metric(_weighted_mean(key, "velocity_eval_frames"))
    out["velocity_rmse_mps"] = _fmt_metric(_combined_rmse("velocity_rmse_mps", "velocity_eval_frames"))
    return out

def run_single(args: argparse.Namespace) -> None:
    if args.rgb_dir is None:
        raise ValueError("Provide --rgb_dir for single-sequence mode or --dataset_dir for dataset mode.")
    tracker = OSTrackBoxTracker(
        repo_path=args.ostrack_repo,
        checkpoint=args.ostrack_checkpoint,
        config=args.ostrack_config,
        device=args.device,
        debug=args.debug,
    )
    model, device, pos_smooth_window = _prepare_model(args)
    result = _run_sequence(
        args=args,
        rgb_dir=args.rgb_dir,
        intrinsics_json=args.intrinsics_json,
        initial_box=args.initial_box,
        initial_box_file=args.initial_box_file,
        out_csv=args.out_csv,
        overlay_dir=args.overlay_dir,
        tracker=tracker,
        model=model,
        device=device,
        pos_smooth_window=pos_smooth_window,
        fps=float(args.fps),
        position_csv=args.position_csv,
        distance_csv=args.distance_csv,
        robot_motion_csv=args.robot_motion_csv,
        direction_csv=args.direction_csv,
        frame_pose_csv=args.frame_pose_csv,
        input_features_source="manual_csv",
    )
    print(
        "[test12+temp]",
        f"frames={result.frames}",
        f"tracked_boxes={result.tracked_boxes}",
        f"directions={result.valid_directions}",
        f"positions={result.valid_positions}",
        f"model_velocities={result.valid_velocities}",
    )
    print(
        "[test12+temp] tracking",
        f"total={result.total_track_time_s:.3f}s",
        f"mean={result.mean_track_time_ms:.2f}ms",
        f"fps={result.track_fps:.2f}",
    )
    if args.velocity_model_path is None:
        print("[test12+temp] no --velocity_model_path was provided; model velocity columns are empty.")
    elif result.valid_positions > 0:
        print(f"[test12+temp] velocity pos_smooth_window={pos_smooth_window}")
    print(f"[test12+temp] wrote {args.out_csv}")
    if args.overlay_dir is not None:
        print(f"[test12+temp] wrote overlays to {args.overlay_dir}")


def run_dataset(args: argparse.Namespace) -> None:
    args.dataset_dir = args.dataset_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fps = _infer_fps_from_dataset(args.dataset_dir, fallback=float(args.fps))

    runs = _collect_dataset_runs(args)
    if not runs:
        raise RuntimeError(f"No matching complete target runs found in {args.dataset_dir / 'manifest.csv'}")

    tracker = OSTrackBoxTracker(
        repo_path=args.ostrack_repo,
        checkpoint=args.ostrack_checkpoint,
        config=args.ostrack_config,
        device=args.device,
        debug=args.debug,
    )
    model, device, pos_smooth_window = _prepare_model(args)

    summary_rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        out_csv = args.out_dir / "estimates" / f"{run.target_name}.csv"
        overlay_dir = args.out_dir / "overlays" / run.target_name if args.overlay else None
        print(
            f"[test12+temp dataset] {index + 1}/{len(runs)} {run.sequence_name}",
            f"object_id={run.target_object_id}",
            f"color={run.target_color_name}",
        )
        if out_csv.exists() and not args.force:
            print(f"[test12+temp dataset] reuse {out_csv}")
            result = _summarize_existing_result(
                run=run,
                out_csv=out_csv,
                overlay_dir=overlay_dir,
                pos_smooth_window=pos_smooth_window,
            )
            row = _summary_row(result)
            row.update(_compute_dataset_metrics(run, out_csv))
            summary_rows.append(row)
            continue
        result = _run_sequence(
            args=args,
            rgb_dir=run.seq_dir / "rgb",
            intrinsics_json=run.seq_dir / "intrinsics.json",
            initial_box=run.initial_box,
            initial_box_file=run.initial_box_file,
            out_csv=out_csv,
            overlay_dir=overlay_dir,
            tracker=tracker,
            model=model,
            device=device,
            pos_smooth_window=pos_smooth_window,
            fps=float(args.fps),
            position_csv=args.position_csv if args.position_csv is not None else run.position_csv,
            distance_csv=args.distance_csv if args.distance_csv is not None else run.distance_csv,
            robot_motion_csv=args.robot_motion_csv if args.robot_motion_csv is not None else run.robot_motion_csv,
            direction_csv=args.direction_csv if args.direction_csv is not None else run.direction_csv,
            frame_pose_csv=args.frame_pose_csv if args.frame_pose_csv is not None else run.frame_pose_csv,
            input_features_source="manual_position_csv" if args.position_csv is not None else run.input_features_source,
            target_object_id=run.target_object_id,
            target_color_name=run.target_color_name,
        )
        result = SequenceRunResult(
            **{
                **result.__dict__,
                "sequence_name": run.sequence_name,
                "target_name": run.target_name,
                "target_object_id": run.target_object_id,
                "target_color_name": run.target_color_name,
            }
        )
        row = _summary_row(result)
        row.update(_compute_dataset_metrics(run, out_csv))
        summary_rows.append(row)
        print(
            "[test12+temp dataset] result",
            f"frames={result.frames}",
            f"directions={result.valid_directions}",
            f"positions={result.valid_positions}",
            f"model_velocities={result.valid_velocities}",
            f"csv={out_csv}",
        )

    detail_row_count = len(summary_rows)
    overall_row = _overall_summary_row(summary_rows)
    if overall_row:
        summary_rows.append(overall_row)

    summary_csv = args.out_dir / "summary.csv"
    _write_summary_csv(summary_csv, summary_rows)
    total_frames = sum(int(row["frames"]) for row in summary_rows[:detail_row_count])
    total_velocities = sum(int(row["valid_velocities"]) for row in summary_rows[:detail_row_count])
    print(
        "[test12+temp dataset] done",
        f"runs={detail_row_count}/{len(runs)}",
        f"frames={total_frames}",
        f"model_velocities={total_velocities}",
        f"summary={summary_csv}",
    )


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.dataset_dir is not None:
        run_dataset(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
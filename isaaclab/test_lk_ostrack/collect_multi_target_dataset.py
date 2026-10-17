from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
STANDALONE_DIR = THIS_DIR.parent
SOURCE_DIR = STANDALONE_DIR.parent
ROBOT_USD_PATH = THIS_DIR / "lite3_with_turret.usd"

# Prefer this workspace's Isaac Lab package only when it is a complete source
# checkout.  Some experiment workspaces contain copied extension files and
# ``__pycache__`` entries but no ``isaaclab.app`` sources; prepending such a
# directory creates a namespace package that shadows the valid installation
# and makes ``from isaaclab.app import AppLauncher`` fail.  When this workspace
# is incomplete, use the source tree belonging to the launcher instead.
local_isaaclab_source = SOURCE_DIR / "isaaclab"
launcher_root = os.environ.get("ISAACLAB_PATH")
launcher_isaaclab_source = (
    Path(launcher_root) / "source" / "isaaclab" if launcher_root else None
)
isaaclab_source = next(
    (
        candidate
        for candidate in (local_isaaclab_source, launcher_isaaclab_source)
        if candidate is not None
        and (candidate / "isaaclab" / "app" / "__init__.py").is_file()
    ),
    None,
)
for import_path in (isaaclab_source, STANDALONE_DIR):
    if import_path is None:
        continue
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpy as np
import torch

from isaaclab.app import AppLauncher
from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.multi_target_tracker import (
    MultiTargetConfig,
    MultiTargetFrameResult,
    MultiTargetVisionTracker,
)
from test_lk_ostrack.ostrack_tracker import OSTrackFrameTracker
from test_lk_ostrack.rgb_visibility import rgb_target_visibility_mask


BUNDLED_OSTRACK_REPO = Path(__file__).resolve().parent / "ostrack" / "OSTrack-main"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Isaac Sim sequences with online LK/OSTrack bearing "
            "measurements and RayCaster ranges, optionally saving RGB frames."
        )
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_targets", type=int, default=3)
    parser.add_argument("--num_sequences", type=int, default=20)
    parser.add_argument("--sequence_length", type=int, default=120)
    parser.add_argument("--capture_stride", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument(
        "--save_images", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--disable_fine_tracking",
        action="store_true",
        help="Use LK/RANSAC/Kalman only. Without this flag, OSTrack is enabled when both paths are provided.",
    )
    parser.add_argument("--ostrack_repo", type=Path, default=BUNDLED_OSTRACK_REPO)
    parser.add_argument("--ostrack_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--ostrack_config", type=str, default="vitb_256_mae_ce_32x4_ep300"
    )
    parser.add_argument("--fine_tracker_interval_frames", type=int, default=3)
    parser.add_argument("--fine_tracker_min_score", type=float, default=0.20)
    parser.add_argument(
        "--fine_tracker_policy", choices=("fixed", "adaptive"), default="fixed"
    )
    parser.add_argument("--adaptive_min_interval_frames", type=int, default=2)
    parser.add_argument("--adaptive_max_interval_frames", type=int, default=12)
    parser.add_argument("--adaptive_bad_frames", type=int, default=2)
    parser.add_argument("--adaptive_warning_confidence", type=float, default=0.75)
    parser.add_argument("--adaptive_critical_confidence", type=float, default=0.55)
    parser.add_argument("--adaptive_risk_threshold", type=float, default=0.55)
    parser.add_argument("--adaptive_critical_risk_threshold", type=float, default=0.85)
    parser.add_argument("--adaptive_risk_clear_threshold", type=float, default=0.35)
    parser.add_argument("--adaptive_desired_inliers", type=int, default=9)
    parser.add_argument("--adaptive_innovation_critical", type=float, default=0.75)
    parser.add_argument("--adaptive_prediction_iou_threshold", type=float, default=0.55)
    parser.add_argument("--adaptive_failure_backoff_initial_frames", type=int, default=1)
    parser.add_argument("--adaptive_failure_backoff_max_frames", type=int, default=8)
    parser.add_argument("--fine_max_center_jump_box_diagonals", type=float, default=2.0)
    parser.add_argument("--fine_max_scale_change", type=float, default=2.5)
    parser.add_argument("--camera_width", type=int, default=1024)
    parser.add_argument("--camera_height", type=int, default=576)
    parser.add_argument("--camera_projection_type", type=str, default="perspective")
    parser.add_argument("--camera_focal_length", type=float, default=10.0)
    parser.add_argument("--camera_horizontal_aperture", type=float, default=5.76)
    parser.add_argument("--camera_vertical_aperture", type=float, default=3.24)
    parser.add_argument("--camera_clip_near", type=float, default=0.1)
    parser.add_argument("--camera_clip_far", type=float, default=1000000.0)
    parser.add_argument("--camera_f_stop", type=float, default=0.0)
    parser.add_argument("--dynamic_targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep_targets_in_view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_distance_min", type=float, default=80.0)
    parser.add_argument("--target_distance_max", type=float, default=150.0)
    parser.add_argument("--target_height_min", type=float, default=0.45)
    parser.add_argument("--target_height_max", type=float, default=3.0)
    parser.add_argument(
        "--target_height_constraint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Constrain target world height (disabled by default because the "
            "configured upward-looking camera may not intersect the default "
            "height range at long distance)."
        ),
    )
    parser.add_argument("--target_radius", type=float, default=0.35)
    parser.add_argument(
        "--target_features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--target_speed_min", type=float, default=0.6)
    parser.add_argument("--target_speed_max", type=float, default=5)
    parser.add_argument(
        "--target_motion_mode",
        choices=("random", "camera_lateral"),
        default="random",
        help=(
            "Target velocity sampling mode. camera_lateral uses the requested "
            "speed exactly along camera x with zero vertical/depth velocity, "
            "which is intended for controlled tracking-speed benchmarks."
        ),
    )
    parser.add_argument("--target_vertical_speed_min", type=float, default=-0.5)
    parser.add_argument("--target_vertical_speed_max", type=float, default=0.5)
    parser.add_argument("--target_depth_speed_scale", type=float, default=0.15)
    parser.add_argument("--raycast_min_range", type=float, default=65.0)
    parser.add_argument("--raycast_max_range", type=float, default=165.0)
    parser.add_argument("--raycast_h_fov_deg", type=float, default=40.0)
    parser.add_argument("--raycast_v_fov_deg", type=float, default=24.0)
    parser.add_argument("--raycast_h_res_deg", type=float, default=0.15)
    parser.add_argument("--raycast_v_res_deg", type=float, default=0.15)
    parser.add_argument("--raycast_primary_cone_deg", type=float, default=0.75)
    parser.add_argument("--raycast_fallback_cone_deg", type=float, default=1.5)
    parser.add_argument("--min_target_separation_px", type=float, default=64.0)
    parser.add_argument("--target_spawn_border_px", type=float, default=16.0)
    parser.add_argument("--turret_action_scale", type=float, default=0.04)
    parser.add_argument("--action_resample_steps", type=int, default=25)
    parser.add_argument("--locomotion_policy_path", type=Path, default=None)
    parser.add_argument(
        "--locomotion_zero_command",
        action="store_true",
        help=(
            "Set vx, vy, and yaw-rate commands to zero. If a locomotion policy "
            "is provided, it remains active to hold the standing pose."
        ),
    )
    parser.add_argument("--locomotion_command_resample_s", type=float, default=3.0)
    parser.add_argument("--max_static_base_speed_mps", type=float, default=0.15)
    parser.add_argument(
        "--max_static_base_angular_speed_radps", type=float, default=0.15
    )
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--debug_diagnostics", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    integer_positive = (
        "num_envs",
        "num_targets",
        "num_sequences",
        "sequence_length",
        "capture_stride",
        "max_steps",
        "camera_width",
        "camera_height",
        "action_resample_steps",
        "progress_every",
    )
    for name in integer_positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.num_targets < 2:
        raise ValueError("--num_targets must be at least 2")
    if args.sequence_length < 2:
        raise ValueError(
            "--sequence_length must be at least 2 because the bootstrap frame is not a measurement frame"
        )
    if args.target_distance_max <= args.target_distance_min:
        raise ValueError("Target distance range is invalid")
    if args.target_height_max <= args.target_height_min:
        raise ValueError("Target height range is invalid")
    if args.target_speed_max < args.target_speed_min or args.target_speed_min < 0.0:
        raise ValueError("Target speed range is invalid")
    if args.target_depth_speed_scale < 0.0:
        raise ValueError("--target_depth_speed_scale must be non-negative")
    if args.raycast_min_range < 0.0 or args.raycast_max_range <= args.raycast_min_range:
        raise ValueError("Raycast range must satisfy 0 <= min < max")
    if any(
        value <= 0.0
        for value in (
            args.raycast_h_fov_deg,
            args.raycast_v_fov_deg,
            args.raycast_h_res_deg,
            args.raycast_v_res_deg,
        )
    ):
        raise ValueError("Raycast FOV and angular resolutions must be positive")
    if (
        args.raycast_primary_cone_deg <= 0.0
        or args.raycast_fallback_cone_deg < args.raycast_primary_cone_deg
    ):
        raise ValueError(
            "Raycast association cones must satisfy 0 < primary <= fallback"
        )
    if args.target_radius <= 0.0:
        raise ValueError("--target_radius must be positive")
    if args.min_target_separation_px < 0.0 or args.target_spawn_border_px < 0.0:
        raise ValueError("Target separation and border must be non-negative")
    if args.turret_action_scale < 0.0:
        raise ValueError("--turret_action_scale must be non-negative")
    if args.locomotion_command_resample_s <= 0.0:
        raise ValueError("--locomotion_command_resample_s must be positive")
    if args.max_static_base_speed_mps <= 0.0:
        raise ValueError("--max_static_base_speed_mps must be positive")
    if args.max_static_base_angular_speed_radps <= 0.0:
        raise ValueError("--max_static_base_angular_speed_radps must be positive")
    if args.locomotion_policy_path is not None:
        policy_path = args.locomotion_policy_path.expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"Locomotion policy was not found: {policy_path}")
    if args.camera_focal_length <= 0.0:
        raise ValueError("--camera_focal_length must be positive")
    if (
        args.camera_horizontal_aperture <= 0.0
        or args.camera_vertical_aperture <= 0.0
    ):
        raise ValueError("Camera apertures must be positive")
    if (
        args.camera_clip_near <= 0.0
        or args.camera_clip_far <= args.camera_clip_near
    ):
        raise ValueError("Camera clipping range must satisfy 0 < near < far")
    if args.camera_f_stop < 0.0:
        raise ValueError("--camera_f_stop must be non-negative")
    if args.fine_tracker_interval_frames <= 0:
        raise ValueError("--fine_tracker_interval_frames must be positive")
    if not 0.0 <= args.fine_tracker_min_score <= 1.0:
        raise ValueError("--fine_tracker_min_score must be in [0, 1]")
    for name in ("ostrack_repo", "ostrack_checkpoint"):
        path = getattr(args, name)
        if path is not None and not path.expanduser().resolve().exists():
            raise FileNotFoundError(f"{name} was not found: {path}")


def _rgb_batch_to_uint8(images: torch.Tensor) -> np.ndarray:
    image_tensor = images.detach().to(device="cpu")
    if image_tensor.ndim != 4 or image_tensor.shape[-1] < 3:
        raise RuntimeError(f"Expected camera images [N,H,W,C], got {tuple(image_tensor.shape)}")
    image_tensor = image_tensor[..., :3]
    if image_tensor.dtype != torch.uint8:
        image_tensor = image_tensor.to(dtype=torch.float32)
        if image_tensor.numel() and float(image_tensor.max().item()) <= 1.5:
            image_tensor *= 255.0
        image_tensor = image_tensor.clamp(0.0, 255.0).round().to(torch.uint8)
    return np.ascontiguousarray(image_tensor.numpy())


def _camera_intrinsics_from_matrix(
    matrix: np.ndarray, width: int, height: int
) -> CameraIntrinsics:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite camera intrinsics [3,3], got {values.shape}")
    return CameraIntrinsics(
        fx=float(values[0, 0]),
        fy=float(values[1, 1]),
        cx=float(values[0, 2]),
        cy=float(values[1, 2]),
        width=int(width),
        height=int(height),
    )


def _build_vision_tracker(
    args: argparse.Namespace, intrinsics: CameraIntrinsics
) -> MultiTargetVisionTracker:
    fine_tracker = None
    if (
        not bool(args.disable_fine_tracking)
        and args.ostrack_repo is not None
        and args.ostrack_checkpoint is not None
    ):
        fine_tracker = OSTrackFrameTracker(
            repo_path=args.ostrack_repo.expanduser().resolve(),
            checkpoint=args.ostrack_checkpoint.expanduser().resolve(),
            config=str(args.ostrack_config),
            device=str(args.device),
        )
        # Load the network before the first sequence so the first measurement
        # is not delayed by checkpoint initialization.
        fine_tracker.prepare()

    config = MultiTargetConfig(
        target_width_m=2.0 * float(args.target_radius),
        target_height_m=2.0 * float(args.target_radius),
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
    return MultiTargetVisionTracker(
        intrinsics,
        config=config,
        fine_tracker=fine_tracker,
    )


def _result_to_directions(
    result: MultiTargetFrameResult, target_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tracker output to raycast angles and a visual-valid mask."""
    directions = np.zeros((target_count, 2), dtype=np.float32)
    valid = np.zeros((target_count,), dtype=bool)
    for estimate in result.targets:
        target_id = int(estimate.target_id)
        if not 0 <= target_id < target_count:
            continue
        yaw = float(estimate.yaw_rad)
        pitch = float(estimate.pitch_rad)
        if not np.isfinite((yaw, pitch)).all():
            continue
        directions[target_id] = (yaw, pitch)
        valid[target_id] = estimate.status in {"initialized", "tracking"}
    return directions, valid


def _measured_frame_for_env(
    batch_cpu: dict[str, np.ndarray],
    env_id: int,
    result: MultiTargetFrameResult,
    lidar_range_m: np.ndarray,
    lidar_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Merge visual/raycast measurements with simulation labels.

    The target pose/velocity fields remain labels for supervised learning.
    Direction, image geometry, visibility, and distance are replaced by the
    online perception measurements before they reach the text writer.
    """
    measurement = {
        name: np.array(value, copy=True)
        for name, value in _supervision_for_env(batch_cpu, env_id).items()
    }
    estimates = {int(item.target_id): item for item in result.targets}
    target_count = int(lidar_range_m.shape[0])
    direction = np.zeros((target_count, 2), dtype=np.float64)
    bbox = np.zeros((target_count, 4), dtype=np.float64)
    center = np.zeros((target_count, 2), dtype=np.float64)
    visible = np.zeros((target_count,), dtype=bool)
    fully_visible = np.zeros((target_count,), dtype=bool)
    for target_id in range(target_count):
        estimate = estimates.get(target_id)
        if estimate is None:
            continue
        bbox[target_id] = np.asarray(estimate.bbox_xyxy, dtype=np.float64)
        center[target_id] = np.asarray(estimate.center_xy, dtype=np.float64)
        direction[target_id] = (float(estimate.yaw_rad), float(estimate.pitch_rad))
        visual_valid = estimate.status in {"initialized", "tracking"}
        visible[target_id] = visual_valid
        fully_visible[target_id] = visual_valid

    raycast_range = np.asarray(lidar_range_m, dtype=np.float64)
    raycast_valid = np.asarray(lidar_valid, dtype=bool)
    measurement["bbox_xyxy"] = bbox
    measurement["bbox_xyxy_raw"] = bbox.copy()
    measurement["center_uv"] = center
    measurement["direction"] = direction
    measurement["visible"] = visible
    measurement["fully_visible"] = fully_visible
    measurement["lidar_range_m"] = raycast_range.copy()
    measurement["lidar_valid"] = raycast_valid.copy()
    # distance_m is a legacy scalar field; keep it finite for invalid rays and
    # use lidar_valid to distinguish an actual range from a missing measurement.
    measurement["distance_m"] = np.where(
        raycast_valid & np.isfinite(raycast_range), raycast_range, 0.0
    )
    return measurement


def _supervision_for_env(
    batch_cpu: dict[str, np.ndarray], env_id: int
) -> dict[str, np.ndarray]:
    """Extract simulation labels; online measurement fields are overwritten later."""
    names = (
        "bbox_xyxy",
        "bbox_xyxy_raw",
        "center_uv",
        "direction",
        "distance_m",
        "lidar_range_m",
        "lidar_valid",
        "target_pos_cam",
        "target_vel_cam",
        "target_pos_yaw",
        "target_vel_yaw",
        "camera_position_yaw",
        "rotation_yaw_from_camera",
        "base_linear_velocity_yaw",
        "base_angular_velocity_yaw",
        "camera_linear_velocity_cam",
        "camera_angular_velocity_cam",
        "visible",
        "fully_visible",
    )
    return {name: batch_cpu[name][env_id] for name in names}


def main(
    *,
    force_text_only: bool = False,
    force_static_robot: bool = False,
    primary_target_only: bool = False,
) -> None:
    args = parse_args()
    if force_text_only:
        args.save_images = False
    if force_static_robot:
        args.locomotion_zero_command = True
    if primary_target_only:
        args.dynamic_targets = True
    _validate_args(args)
    if primary_target_only and float(args.target_speed_min) <= 0.0:
        raise ValueError(
            "Primary-target velocity collection requires --target_speed_min > 0"
        )
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(
            "primary_target_velocity_dataset"
            if primary_target_only
            else "multi_target_tracking_dataset"
        )
        / timestamp
        if args.out_dir is None
        else args.out_dir.expanduser()
    ).resolve()
    if not ROBOT_USD_PATH.is_file():
        raise FileNotFoundError(f"Robot USD was not found: {ROBOT_USD_PATH}")
    if output_dir.exists():
        unexpected = [
            path for path in output_dir.iterdir() if path.name not in {"sequences", "failure.log"}
        ]
        sequences_dir = output_dir / "sequences"
        has_sequences = sequences_dir.is_dir() and any(sequences_dir.iterdir())
        if unexpected or has_sequences:
            raise FileExistsError(f"Output directory contains collection data: {output_dir}")
        failure_log = output_dir / "failure.log"
        if failure_log.exists():
            failure_log.unlink()
    (output_dir / "sequences").mkdir(parents=True, exist_ok=True)

    launcher = AppLauncher(
        headless=bool(args.headless),
        enable_cameras=True,
        device=str(args.device),
    )
    simulation_app = launcher.app
    env = None
    manifest_rows: list[dict[str, Any]] = []
    active_writers: list[Any | None] = [None for _ in range(int(args.num_envs))]
    vision_trackers: list[MultiTargetVisionTracker | None] = [
        None for _ in range(int(args.num_envs))
    ]
    completed_sequences = 0
    next_sequence_id = 0
    static_motion_aborts = 0
    primary_switch_aborts = 0
    rgb_visibility_resamples = 0
    rgb_visibility_aborts = 0

    try:
        import gymnasium as gym

        from test_lk_ostrack.isaacsim_dataset import (
            IsaacSimSequenceWriter,
            validate_collection_dataset,
            write_dataset_manifest,
        )
        from test_lk_ostrack.multi_target_collect_env import (
            MultiTargetCollectEnv,
            MultiTargetCollectEnvCfg,
        )

        env_cfg = MultiTargetCollectEnvCfg()
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.sim.device = str(args.device)
        env_cfg.sim.use_fabric = False
        env_cfg.usd_path = str(ROBOT_USD_PATH)
        env_cfg.debug_vis = not bool(args.headless)
        env_cfg.debug_diagnostics = bool(args.debug_diagnostics)
        env_cfg.enable_dynamic_target_raycast_debug = True
        env_cfg.render_target_sphere = True
        env_cfg.enable_vision_supervision = True
        env_cfg.vision_history_length = 1
        env_cfg.camera_width = int(args.camera_width)
        env_cfg.camera_height = int(args.camera_height)
        env_cfg.camera_projection_type = str(args.camera_projection_type)
        env_cfg.camera_focal_length = float(args.camera_focal_length)
        env_cfg.camera_horizontal_aperture = float(args.camera_horizontal_aperture)
        env_cfg.camera_vertical_aperture = float(args.camera_vertical_aperture)
        env_cfg.camera_clipping_range = (
            float(args.camera_clip_near),
            float(args.camera_clip_far),
        )
        env_cfg.camera_f_stop = float(args.camera_f_stop)
        env_cfg.tiled_camera.width = int(args.camera_width)
        env_cfg.tiled_camera.height = int(args.camera_height)
        env_cfg.tiled_camera.update_period = 0.0
        # RGB is always required by LK/OSTrack. ``save_images`` controls only
        # disk output; frames are converted per capture and released after the
        # tracker/writer pass, so CSV-only collection does not retain images.
        env_cfg.tiled_camera.data_types = ["rgb"]
        env_cfg.num_visual_targets = int(args.num_targets)
        env_cfg.target_radius_range = (
            float(args.target_distance_min),
            float(args.target_distance_max),
        )
        env_cfg.target_height_range = (
            float(args.target_height_min),
            float(args.target_height_max),
        )
        env_cfg.constrain_target_height = bool(args.target_height_constraint)
        env_cfg.target_sphere_radius = float(args.target_radius)
        env_cfg.add_optical_flow_features = bool(args.target_features)
        env_cfg.target_speed_range = (
            float(args.target_speed_min),
            float(args.target_speed_max),
        )
        env_cfg.target_motion_mode = str(args.target_motion_mode)
        env_cfg.target_vertical_speed_range = (
            float(args.target_vertical_speed_min),
            float(args.target_vertical_speed_max),
        )
        env_cfg.target_depth_speed_scale = float(args.target_depth_speed_scale)
        env_cfg.dynamic_target_raycast_min_range = float(args.raycast_min_range)
        env_cfg.dynamic_target_raycast_max_range = float(args.raycast_max_range)
        env_cfg.dynamic_target_raycast_horizontal_fov_deg = float(
            args.raycast_h_fov_deg
        )
        env_cfg.dynamic_target_raycast_vertical_fov_deg = float(
            args.raycast_v_fov_deg
        )
        env_cfg.dynamic_target_raycast_horizontal_resolution_deg = float(
            args.raycast_h_res_deg
        )
        env_cfg.dynamic_target_raycast_vertical_resolution_deg = float(
            args.raycast_v_res_deg
        )
        env_cfg.dynamic_target_raycast_primary_cone_deg = float(
            args.raycast_primary_cone_deg
        )
        env_cfg.dynamic_target_raycast_fallback_cone_deg = float(
            args.raycast_fallback_cone_deg
        )
        env_cfg.min_target_separation_px = float(args.min_target_separation_px)
        env_cfg.target_spawn_border_px = float(args.target_spawn_border_px)
        env_cfg.enable_dynamic_target = bool(args.dynamic_targets)
        env_cfg.keep_targets_in_view = bool(args.keep_targets_in_view)
        env_cfg.sample_target_in_image = True
        env_cfg.sample_target_in_camera_sector = False
        locomotion_policy_path = (
            args.locomotion_policy_path.expanduser().resolve()
            if args.locomotion_policy_path is not None
            else None
        )
        env_cfg.enable_locomotion_policy = locomotion_policy_path is not None
        env_cfg.locomotion_policy_path = (
            str(locomotion_policy_path) if locomotion_policy_path is not None else None
        )
        if args.locomotion_zero_command:
            env_cfg.locomotion_lin_vel_x_range = (0.0, 0.0)
            env_cfg.locomotion_lin_vel_y_range = (0.0, 0.0)
            env_cfg.locomotion_ang_vel_z_range = (0.0, 0.0)
        env_cfg.locomotion_command_resample_s = float(
            args.locomotion_command_resample_s
        )
        # Give the locomotion policy and camera one second to settle before
        # initializing a tracker or recording a sequence.
        env_cfg.aim_warmup_s = 1.0
        env_cfg.episode_length_s = 60.0

        task_id = "Standalone-TurretAim-MultiTargetCollect-v0"
        if task_id not in gym.registry:
            gym.register(id=task_id, entry_point=MultiTargetCollectEnv)
        env = gym.make(task_id, cfg=env_cfg, disable_env_checker=True)
        base_env: MultiTargetCollectEnv = env.unwrapped
        env.reset(seed=int(args.seed))

        capture_fps = 1.0 / (float(base_env.step_dt) * int(args.capture_stride))
        current_random_action = torch.zeros(
            (base_env.num_envs, 2), dtype=torch.float32, device=base_env.device
        )
        print(
            "[multi-collect]",
            f"output={output_dir}",
            f"envs={base_env.num_envs}",
            f"targets={args.num_targets}",
            f"sequence_length={args.sequence_length}",
            f"capture_fps={capture_fps:.3f}",
            f"storage={'rgb_csv' if args.save_images else 'csv_only'}",
            f"recording={'primary_only' if primary_target_only else 'all_targets'}",
            "robot="
            + (
                "policy_zero_command"
                if locomotion_policy_path is not None
                and args.locomotion_zero_command
                else "policy_configured_motion"
                if locomotion_policy_path is not None
                else "default_joint_targets"
            ),
        )

        for step in range(int(args.max_steps)):
            action = torch.zeros(
                (base_env.num_envs, 3), dtype=torch.float32, device=base_env.device
            )
            if step % int(args.action_resample_steps) == 0:
                current_random_action = (
                    torch.rand(
                        (base_env.num_envs, 2), device=base_env.device
                    )
                    * 2.0
                    - 1.0
                ) * float(args.turret_action_scale)
            action[:, :2] = current_random_action
            action[:, 2] = -1.0
            _, _, terminated, truncated, _ = env.step(action)
            # The camera must be refreshed even for CSV-only output because
            # the RGB frame is the online measurement input.
            base_env.refresh_vision_supervision()

            if step % int(args.capture_stride) != 0:
                continue
            batch = base_env.get_multi_target_collection_batch(
                include_rgb=True,
                include_raycast=False,
            )
            images = _rgb_batch_to_uint8(batch["images"])
            tensor_names = (
                "intrinsics",
                "bbox_xyxy",
                "bbox_xyxy_raw",
                "center_uv",
                "direction",
                "distance_m",
                "lidar_range_m",
                "lidar_valid",
                "target_pos_cam",
                "target_vel_cam",
                "target_pos_yaw",
                "target_vel_yaw",
                "camera_position_yaw",
                "rotation_yaw_from_camera",
                "base_linear_velocity_yaw",
                "base_angular_velocity_yaw",
                "camera_linear_velocity_cam",
                "camera_angular_velocity_cam",
                "visible",
                "fully_visible",
                "collection_ready",
                "separated",
                "initializable",
            )
            batch_cpu = {
                name: batch[name].detach().cpu().numpy() for name in tensor_names
            }
            done = (
                torch.as_tensor(terminated, device="cpu", dtype=torch.bool)
                | torch.as_tensor(truncated, device="cpu", dtype=torch.bool)
            ).numpy()

            visual_results: list[MultiTargetFrameResult | None] = [
                None for _ in range(base_env.num_envs)
            ]
            visual_directions = np.zeros(
                (base_env.num_envs, int(args.num_targets), 2), dtype=np.float32
            )
            visual_valid = np.zeros(
                (base_env.num_envs, int(args.num_targets)), dtype=bool
            )
            initialized_now = np.zeros((base_env.num_envs,), dtype=bool)
            capture_dt_s = 1.0 / capture_fps
            capture_timestamp_s = float(step) * float(base_env.step_dt)

            # Perception runs before any range query.  The only simulation
            # geometry used here is the first-frame bbox needed to bootstrap a
            # conventional online tracker; it is never written as a
            # direction measurement.
            for env_id in range(base_env.num_envs):
                writer = active_writers[env_id]
                if force_static_robot:
                    base_speed = float(
                        np.linalg.norm(batch_cpu["base_linear_velocity_yaw"][env_id])
                    )
                    base_angular_speed = float(
                        np.linalg.norm(batch_cpu["base_angular_velocity_yaw"][env_id])
                    )
                    robot_is_static = (
                        base_speed <= float(args.max_static_base_speed_mps)
                        and base_angular_speed
                        <= float(args.max_static_base_angular_speed_radps)
                    )
                    if not robot_is_static:
                        if writer is not None:
                            writer.abort()
                            active_writers[env_id] = None
                            static_motion_aborts += 1
                        tracker = vision_trackers[env_id]
                        if tracker is not None and tracker.initialized:
                            tracker.reset()
                        continue
                if writer is not None and bool(done[env_id]):
                    writer.abort()
                    active_writers[env_id] = None
                    writer = None
                if bool(done[env_id]) and vision_trackers[env_id] is not None:
                    vision_trackers[env_id].reset()

                active_count = sum(item is not None for item in active_writers)
                can_reserve = (
                    completed_sequences + active_count < int(args.num_sequences)
                )
                if (
                    writer is None
                    and can_reserve
                    and bool(batch_cpu["initializable"][env_id])
                ):
                    initial_boxes = batch_cpu["bbox_xyxy"][env_id]
                    rgb_target_visible = rgb_target_visibility_mask(
                        images[env_id], initial_boxes
                    )
                    if not bool(rgb_target_visible.all()):
                        if rgb_visibility_resamples == 0:
                            from PIL import Image
                            import json

                            Image.fromarray(images[env_id]).save(output_dir / "initialization_preview.png")
                            (output_dir / "initialization_preview.json").write_text(json.dumps({
                                "boxes_xyxy": initial_boxes.tolist(),
                                "rgb_target_visible": rgb_target_visible.tolist(),
                                "camera_position_yaw": batch_cpu["camera_position_yaw"][env_id].tolist(),
                                "target_pos_cam": batch_cpu["target_pos_cam"][env_id].tolist(),
                            }, indent=2))
                        tracker = vision_trackers[env_id]
                        if tracker is not None and tracker.initialized:
                            tracker.reset()
                        base_env.resample_collection_targets([env_id])
                        rgb_visibility_resamples += 1
                        continue
                    if vision_trackers[env_id] is None:
                        vision_trackers[env_id] = _build_vision_tracker(
                            args,
                            _camera_intrinsics_from_matrix(
                                batch_cpu["intrinsics"][env_id],
                                int(args.camera_width),
                                int(args.camera_height),
                            ),
                        )
                    if next_sequence_id == 0:
                        box_sizes = initial_boxes[:, 2:4] - initial_boxes[:, 0:2]
                        print(
                            "[multi-collect] initial target pixel size",
                            f"min={float(box_sizes.min()):.2f}px",
                            f"max={float(box_sizes.max()):.2f}px",
                        )
                        if float(box_sizes.min()) < 12.0:
                            print(
                                "[multi-collect] warning: some targets are below 12 px; "
                                "9-16 stable LK corners are unlikely"
                            )
                    result = vision_trackers[env_id].initialize(
                        images[env_id],
                        initial_boxes,
                        timestamp_s=capture_timestamp_s,
                    )
                    if primary_target_only and result.selected_target_id is None:
                        vision_trackers[env_id].reset()
                        base_env.resample_collection_targets([env_id])
                        continue
                    recorded_target_ids = (
                        (int(result.selected_target_id),)
                        if primary_target_only
                        else None
                    )
                    writer = IsaacSimSequenceWriter(
                        dataset_root=output_dir,
                        sequence_id=next_sequence_id,
                        env_id=env_id,
                        target_count=int(args.num_targets),
                        fps=capture_fps,
                        intrinsics=batch_cpu["intrinsics"][env_id],
                        initial_boxes_xyxy=initial_boxes,
                        expected_frames=int(args.sequence_length),
                        image_width=int(args.camera_width),
                        image_height=int(args.camera_height),
                        save_images=bool(args.save_images),
                        recorded_target_ids=recorded_target_ids,
                        extra_meta={
                            "start_sim_step": step,
                            "collection_warmup_s": float(env_cfg.aim_warmup_s),
                            "dynamic_targets": bool(args.dynamic_targets),
                            "keep_targets_in_view": bool(args.keep_targets_in_view),
                            "tracking_frame": "yaw_only",
                            "yaw_only_frame_definition": (
                                "capture-time robot base yaw; z is world-up"
                            ),
                            "direction_source": (
                                "lk+ostrack"
                                if (
                                    not args.disable_fine_tracking
                                    and args.ostrack_repo is not None
                                    and args.ostrack_checkpoint is not None
                                )
                                else "lk_ransac_kalman"
                            ),
                            "range_source": "raycaster_using_visual_direction",
                            "initialization_source": "simulation_bbox_only",
                            "measurement_gt_free_after_initialization": True,
                            "primary_target_only": bool(primary_target_only),
                            "primary_target_id": (
                                int(result.selected_target_id)
                                if primary_target_only
                                else None
                            ),
                            "robot_zero_velocity_command": bool(
                                args.locomotion_zero_command
                            ),
                            "max_static_base_speed_mps": (
                                float(args.max_static_base_speed_mps)
                                if force_static_robot
                                else None
                            ),
                            "max_static_base_angular_speed_radps": (
                                float(args.max_static_base_angular_speed_radps)
                                if force_static_robot
                                else None
                            ),
                            "rgb_storage": (
                                "png"
                                if args.save_images
                                else "in_memory_only_for_tracker"
                            ),
                        },
                    )
                    active_writers[env_id] = writer
                    next_sequence_id += 1
                    visual_results[env_id] = result
                    directions, valid = _result_to_directions(
                        result, int(args.num_targets)
                    )
                    visual_directions[env_id] = directions
                    visual_valid[env_id] = valid
                    initialized_now[env_id] = True
                elif writer is not None:
                    tracker = vision_trackers[env_id]
                    if tracker is None or not tracker.initialized:
                        raise RuntimeError(
                            f"Environment {env_id} has an active writer without an initialized vision tracker"
                        )
                    rgb_target_visible = rgb_target_visibility_mask(
                        images[env_id], batch_cpu["bbox_xyxy"][env_id]
                    )
                    required_target_ids = np.asarray(
                        writer.recorded_target_ids, dtype=np.int64
                    )
                    if not bool(rgb_target_visible[required_target_ids].all()):
                        writer.abort()
                        active_writers[env_id] = None
                        rgb_visibility_aborts += 1
                        tracker.reset()
                        base_env.resample_collection_targets([env_id])
                        continue
                    result = tracker.update(
                        images[env_id],
                        dt_s=capture_dt_s,
                        timestamp_s=capture_timestamp_s,
                    )
                    if primary_target_only and result.selected_target_id not in (
                        writer.recorded_target_ids
                    ):
                        writer.abort()
                        active_writers[env_id] = None
                        primary_switch_aborts += 1
                        tracker.reset()
                        base_env.resample_collection_targets([env_id])
                        continue
                    visual_results[env_id] = result
                    directions, valid = _result_to_directions(
                        result, int(args.num_targets)
                    )
                    visual_directions[env_id] = directions
                    visual_valid[env_id] = valid

            raycast_directions = torch.as_tensor(
                visual_directions, device=base_env.device, dtype=torch.float32
            )
            lidar_range_t, lidar_valid_t = base_env.get_multi_target_raycast_measurements(
                raycast_directions
            )
            visual_valid_t = torch.as_tensor(
                visual_valid, device=base_env.device, dtype=torch.bool
            )
            lidar_valid_t = lidar_valid_t & visual_valid_t
            lidar_range_t = torch.where(
                lidar_valid_t,
                lidar_range_t,
                torch.full_like(lidar_range_t, float("inf")),
            )
            batch_cpu["lidar_range_m"] = lidar_range_t.detach().cpu().numpy()
            batch_cpu["lidar_valid"] = lidar_valid_t.detach().cpu().numpy()

            for env_id in range(base_env.num_envs):
                writer = active_writers[env_id]
                result = visual_results[env_id]
                if writer is None or result is None or bool(initialized_now[env_id]):
                    continue
                lidar_range = batch_cpu["lidar_range_m"][env_id]
                lidar_valid = batch_cpu["lidar_valid"][env_id]
                try:
                    writer.add_frame(
                        images[env_id] if args.save_images else None,
                        _measured_frame_for_env(
                            batch_cpu,
                            env_id,
                            result,
                            lidar_range,
                            lidar_valid,
                        ),
                        timestamp_s=writer.frame_count / capture_fps,
                    )
                except Exception:
                    writer.abort()
                    active_writers[env_id] = None
                    raise

                if writer.complete:
                    manifest_rows.append(writer.finish())
                    active_writers[env_id] = None
                    completed_sequences += 1
                    if vision_trackers[env_id] is not None:
                        vision_trackers[env_id].reset()
                    base_env.resample_collection_targets([env_id])

            # Do not retain the GPU RGB tensor (or the temporary CPU batch)
            # between capture instants when capture_stride > 1.
            del batch
            del images

            if (step + 1) % int(args.progress_every) == 0:
                active_count = sum(item is not None for item in active_writers)
                print(
                    "[multi-collect]",
                    f"step={step + 1}",
                    f"complete={completed_sequences}/{args.num_sequences}",
                    f"active={active_count}",
                    f"static_aborts={static_motion_aborts}",
                    f"primary_switch_aborts={primary_switch_aborts}",
                    f"rgb_resamples={rgb_visibility_resamples}",
                    f"rgb_aborts={rgb_visibility_aborts}",
                )
                print(
                    "[multi-collect] initialization gates",
                    f"ready={int(batch_cpu['collection_ready'].sum())}/{base_env.num_envs}",
                    "fully_visible="
                    f"{int(batch_cpu['fully_visible'].all(axis=1).sum())}/{base_env.num_envs}",
                    f"separated={int(batch_cpu['separated'].sum())}/{base_env.num_envs}",
                    f"initializable={int(batch_cpu['initializable'].sum())}/{base_env.num_envs}",
                    "lidar_valid="
                    f"{int(batch_cpu['lidar_valid'].sum())}/"
                    f"{base_env.num_envs * int(args.num_targets)}",
                )
            if completed_sequences >= int(args.num_sequences):
                break

        for writer in active_writers:
            if writer is not None:
                writer.abort()
        if completed_sequences < int(args.num_sequences):
            raise RuntimeError(
                f"Collected {completed_sequences}/{args.num_sequences} sequences "
                f"within max_steps={args.max_steps}"
            )

        dataset_meta = {
            "created_at": datetime.now().isoformat(),
            "generator": (
                "test_lk_ostrack.collect_multi_target_dataset"
                if args.save_images
                else "test_lk_ostrack.collect_velocity_text_dataset"
            ),
            "seed": int(args.seed),
            "num_sequences": completed_sequences,
            "target_count": int(args.num_targets),
            "sequence_length": int(args.sequence_length),
            "capture_stride": int(args.capture_stride),
            "fps": capture_fps,
            "camera_width": int(args.camera_width),
            "camera_height": int(args.camera_height),
            "camera_projection_type": str(args.camera_projection_type),
            "camera_focal_length": float(args.camera_focal_length),
            "camera_horizontal_aperture": float(args.camera_horizontal_aperture),
            "camera_vertical_aperture": float(args.camera_vertical_aperture),
            "camera_clipping_range": [
                float(args.camera_clip_near),
                float(args.camera_clip_far),
            ],
            "camera_f_stop": float(args.camera_f_stop),
            "dynamic_targets": bool(args.dynamic_targets),
            "keep_targets_in_view": bool(args.keep_targets_in_view),
            "target_distance_range_m": [
                float(args.target_distance_min),
                float(args.target_distance_max),
            ],
            "target_height_constraint": bool(args.target_height_constraint),
            "target_ground_clearance_enforced": True,
            "target_view_margin_px": float(args.target_spawn_border_px),
            "target_speed_range_mps": [
                float(args.target_speed_min),
                float(args.target_speed_max),
            ],
            "target_motion_mode": str(args.target_motion_mode),
            "target_depth_speed_scale": float(args.target_depth_speed_scale),
            "measurement_source": "visual_bearing_multimesh_raycast_range",
            "direction_source": (
                "lk+ostrack"
                if (
                    not args.disable_fine_tracking
                    and args.ostrack_repo is not None
                    and args.ostrack_checkpoint is not None
                )
                else "lk_ransac_kalman"
            ),
            "range_source": "raycaster_using_visual_direction",
            "initialization_source": "simulation_bbox_only",
            "measurement_gt_free_after_initialization": True,
            "vision": {
                "tracker": "MultiTargetVisionTracker",
                "optical_flow": "lucas_kanade_forward_backward_ransac",
                "fine_tracker": (
                    "ostrack"
                    if (
                        not args.disable_fine_tracking
                        and args.ostrack_repo is not None
                        and args.ostrack_checkpoint is not None
                    )
                    else None
                ),
                "fine_tracker_interval_frames": int(
                    args.fine_tracker_interval_frames
                ),
                "fine_tracker_min_score": float(args.fine_tracker_min_score),
                "fine_tracker_policy": str(args.fine_tracker_policy),
                "adaptive_min_interval_frames": int(
                    args.adaptive_min_interval_frames
                ),
                "adaptive_max_interval_frames": int(
                    args.adaptive_max_interval_frames
                ),
                "adaptive_bad_frames": int(args.adaptive_bad_frames),
                "adaptive_warning_confidence": float(
                    args.adaptive_warning_confidence
                ),
                "adaptive_critical_confidence": float(
                    args.adaptive_critical_confidence
                ),
                "adaptive_risk_threshold": float(args.adaptive_risk_threshold),
                "adaptive_critical_risk_threshold": float(
                    args.adaptive_critical_risk_threshold
                ),
                "adaptive_risk_clear_threshold": float(
                    args.adaptive_risk_clear_threshold
                ),
                "adaptive_desired_inliers": int(args.adaptive_desired_inliers),
                "adaptive_innovation_critical": float(
                    args.adaptive_innovation_critical
                ),
                "adaptive_prediction_iou_threshold": float(
                    args.adaptive_prediction_iou_threshold
                ),
                "adaptive_failure_backoff_initial_frames": int(
                    args.adaptive_failure_backoff_initial_frames
                ),
                "adaptive_failure_backoff_max_frames": int(
                    args.adaptive_failure_backoff_max_frames
                ),
                "fine_max_center_jump_box_diagonals": float(
                    args.fine_max_center_jump_box_diagonals
                ),
                "fine_max_scale_change": float(args.fine_max_scale_change),
            },
            "measurement_association": "ray_mesh_id_then_angle_window",
            "tracking_frame": "yaw_only",
            "yaw_only_frame_definition": (
                "capture-time robot base yaw; z is world-up"
            ),
            "base_motion_fields": {
                "linear_velocity": "base_v*_yaw_mps",
                "angular_velocity": "base_w*_yaw_radps",
                "used_as_gru_input": True,
                "linear_velocity_used_as_gru_input": True,
                "angular_velocity_used_as_gru_input": False,
                "used_for_static_sequence_filter": True,
            },
            "raycast": {
                "mount_parent": env_cfg.dynamic_target_raycast_parent_prim_path_expr,
                "translation_m": list(env_cfg.dynamic_target_raycast_translation),
                "orientation_wxyz": list(
                    env_cfg.dynamic_target_raycast_orientation_wxyz
                ),
                "min_range_m": float(args.raycast_min_range),
                "max_range_m": float(args.raycast_max_range),
                "horizontal_fov_deg": float(args.raycast_h_fov_deg),
                "vertical_fov_deg": float(args.raycast_v_fov_deg),
                "horizontal_resolution_deg": float(args.raycast_h_res_deg),
                "vertical_resolution_deg": float(args.raycast_v_res_deg),
                "primary_association_cone_deg": float(
                    args.raycast_primary_cone_deg
                ),
                "fallback_association_cone_deg": float(
                    args.raycast_fallback_cone_deg
                ),
                "update_mesh_ids": True,
            },
            "target_radius_m": float(args.target_radius),
            "target_features": bool(args.target_features),
            "save_images": bool(args.save_images),
            "storage_format": "rgb_csv" if args.save_images else "csv_only",
            "recorded_targets": "primary_only" if primary_target_only else "all",
            "primary_target_sequence_policy": (
                "abort_on_target_switch" if primary_target_only else None
            ),
            "locomotion_policy_path": (
                str(locomotion_policy_path)
                if locomotion_policy_path is not None
                else None
            ),
            "locomotion_command_resample_s": float(
                args.locomotion_command_resample_s
            ),
            "collection_warmup_s": float(env_cfg.aim_warmup_s),
            "locomotion_zero_command": bool(args.locomotion_zero_command),
            "static_robot_filter": {
                "enabled": bool(force_static_robot),
                "max_base_speed_mps": float(args.max_static_base_speed_mps),
                "max_base_angular_speed_radps": float(
                    args.max_static_base_angular_speed_radps
                ),
                "aborted_sequences": static_motion_aborts,
            },
            "primary_switch_aborted_sequences": primary_switch_aborts,
            "rgb_visibility": {
                "validation": "target_bbox_chroma",
                "checked_before_initialization_and_every_recorded_frame": True,
                "minimum_chroma": 24,
                "minimum_colored_fraction": 0.01,
                "minimum_colored_pixels": 4,
                "initialization_resamples": rgb_visibility_resamples,
                "aborted_sequences": rgb_visibility_aborts,
            },
            "locomotion_command_ranges": {
                "vx_mps": list(env_cfg.locomotion_lin_vel_x_range),
                "vy_mps": list(env_cfg.locomotion_lin_vel_y_range),
                "yaw_rate_radps": list(env_cfg.locomotion_ang_vel_z_range),
            },
        }
        write_dataset_manifest(output_dir, manifest_rows, dataset_meta)
        validation = validate_collection_dataset(output_dir)
        print(
            "[multi-collect] complete",
            f"sequences={validation['sequences']}",
            f"frames={validation['frames']}",
            f"dataset={output_dir}",
        )
    except Exception:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "failure.log"
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[multi-collect] failed; traceback written to {failure_path}")
        raise
    finally:
        for writer in active_writers:
            if writer is not None:
                writer.abort()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

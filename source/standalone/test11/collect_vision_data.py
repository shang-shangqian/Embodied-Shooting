# collect_vision_data.py

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any
import os
import numpy as np
import torch

from isaaclab.app import AppLauncher

# Ensure we can import ../turret_aim_env.py when running this script directly.
THIS_DIR = Path(__file__).resolve().parent
TEST4_DIR = THIS_DIR.parent
if str(TEST4_DIR) not in sys.path:
    sys.path.insert(0, str(TEST4_DIR))


# TAQ v2 collector dependencies.  The LK/OSTrack package is kept outside
# test11, so add its parent rather than relying on the caller's CWD.
STANDALONE_DIR = THIS_DIR.parent
if str(STANDALONE_DIR) not in sys.path:
    sys.path.insert(0, str(STANDALONE_DIR))

from speed_prediction.time_aware_velocity_data import (  # noqa: E402
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION,
    TAQ_STREAM_FORMAT,
    TAQ_YAW_RATE_SEMANTICS,
    SOURCE_NAME_TO_ID,
    TrackerSource,
    build_time_aware_features,
    save_taq_velocity_chunk,
)
from speed_prediction.stage2_degradation import (  # noqa: E402
    DATA_SPLITS,
    DEGRADATION_PROFILES,
    DEGRADATION_SEVERITIES,
    Stage2Degrader,
)
from test_lk_ostrack.camera_geometry import CameraIntrinsics  # noqa: E402
from speed_prediction.sensor_time_contract import (
    capture_packet, TIME_CONTRACT_VERSION, PhysicalSamplingGate, EpisodeCollectionWarmup,
)
from test_lk_ostrack.multi_target_tracker import (  # noqa: E402
    MultiTargetConfig,
    MultiTargetVisionTracker,
    TargetEstimate,
)
from test_lk_ostrack.ostrack_tracker import OSTrackFrameTracker  # noqa: E402


def estimate_distance_from_lidar_and_direction(
    batch: dict,
    pred_yaw: "torch.Tensor",
    pred_pitch: "torch.Tensor",
    base_env: Any,
) -> "torch.Tensor":
    import torch

    # 1) 优先使用环境已经给出的 MultiMeshRayCaster 距离
    if "lidar_distance" in batch:
        pred_distance = batch["lidar_distance"]
        if pred_distance.ndim != 1:
            pred_distance = pred_distance.squeeze(-1)
        pred_distance = pred_distance.to(device=base_env.device, dtype=torch.float32)
        return torch.nan_to_num(pred_distance, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) 如果上游已经给了预测距离，也允许直接用
    if "pred_distance" in batch:
        pred_distance = batch["pred_distance"]
        if pred_distance.ndim != 1:
            pred_distance = pred_distance.squeeze(-1)
        pred_distance = pred_distance.to(device=base_env.device, dtype=torch.float32)
        return torch.nan_to_num(pred_distance, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) 最后才回退到 GT
    if "distance" in batch:
        pred_distance = batch["distance"]
        if pred_distance.ndim != 1:
            pred_distance = pred_distance.squeeze(-1)
        pred_distance = pred_distance.to(device=base_env.device, dtype=torch.float32)
        pred_distance = torch.nan_to_num(pred_distance, nan=0.0, posinf=0.0, neginf=0.0)
        print("[warning] No lidar_distance in batch, fallback to GT distance.")
        return pred_distance

    raise KeyError("No distance source found in batch.")


def _build_vision_model(device: str):
    """
    构建你的视觉模型实例。
    """
    from test4.vision_model.vision_cnn.vision_model1 import build_vision_model

    model = build_vision_model(
        use_proprio=True,
        proprio_dim=1,
    )
    return model.to(device)


def load_vision_model(model_path: str, device: str):
    """
    严格加载视觉模型。

    支持：
    1) torch.save(model.state_dict(), path)
    2) torch.save({"model_state_dict": ...}, path)
    3) torch.save({"state_dict": ...}, path)
    4) torch.save(model, path)

    与测试脚本保持一致：
    - 不做 256->257 兼容补丁
    - 不允许 strict=False
    - 只要结构不一致，立即报错
    """
    import torch

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vision model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    # 情况 1：直接保存了整个 nn.Module
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        model.eval()
        print(f"[vision] loaded full nn.Module from: {model_path}")
        return model

    # 其余情况：先构建模型，再严格加载 state_dict
    model = _build_vision_model(device=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise RuntimeError(
            f"Unsupported checkpoint type: {type(ckpt)}. "
            "Expected nn.Module or dict-like state_dict checkpoint."
        )

    # 去掉 DataParallel 的 "module." 前缀
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[new_k] = v

    # 严格加载：任何 key / shape 不一致都直接报错
    model.load_state_dict(cleaned_state_dict, strict=True)

    print(f"[vision] strictly loaded checkpoint from: {model_path}")
    model.eval()
    return model


def run_vision_model(
    vision_model,
    images: "torch.Tensor",
    dog_moving_flag: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    输入:
        images: [N, 3, H, W] 或 [N, 1, 3, H, W]
        dog_moving_flag: [N, 1]

    输出:
        pred_yaw:   [N]
        pred_pitch: [N]

    说明：
    - 这里只允许“单帧输入”
    - 如果上游给的是 [N, 1, 3, H, W]，会自动 squeeze 成 [N, 3, H, W]
    """
    import torch

    if images.ndim == 5:
        if images.shape[1] != 1 or images.shape[2] != 3:
            raise RuntimeError(
                f"Unexpected 5D image shape for single-frame vision model: {tuple(images.shape)}, "
                "expected [N, 1, 3, H, W]"
            )
        images = images[:, 0]  # -> [N, 3, H, W]

    if images.ndim != 4 or images.shape[1] != 3:
        raise RuntimeError(
            f"Unexpected image shape for single-frame vision model: {tuple(images.shape)}, "
            "expected [N, 3, H, W]"
        )

    if dog_moving_flag.ndim != 2 or dog_moving_flag.shape[0] != images.shape[0] or dog_moving_flag.shape[1] != 1:
        raise RuntimeError(
            "dog_moving_flag must have shape [N, 1], "
            f"got {tuple(dog_moving_flag.shape)}"
        )

    model_device = next(vision_model.parameters()).device
    images = images.to(device=model_device, dtype=torch.float32)
    dog_moving_flag = dog_moving_flag.to(device=model_device, dtype=torch.float32)

    with torch.no_grad():
        out = vision_model(images, proprio=dog_moving_flag)

        if not hasattr(out, "direction"):
            raise RuntimeError("vision_model(images, proprio=...) returned no '.direction' field")

        direction = out.direction

    if direction.ndim != 2 or direction.shape[1] < 2:
        raise RuntimeError(f"Vision direction shape must be [N, 2], got {tuple(direction.shape)}")

    pred_yaw = direction[:, 0]
    pred_pitch = direction[:, 1]

    pred_yaw = torch.nan_to_num(pred_yaw, nan=0.0, posinf=0.0, neginf=0.0)
    pred_pitch = torch.nan_to_num(pred_pitch, nan=0.0, posinf=0.0, neginf=0.0)

    return pred_yaw, pred_pitch



def init_lidar_interface():
    """
    在脚本运行的 Isaac Sim 进程里显式启用 physx sensor 扩展，再获取 LiDAR 接口。
    """
    import importlib
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()

    if not ext_manager.is_extension_enabled("isaacsim.sensors.physx"):
        ext_manager.set_extension_enabled_immediate("isaacsim.sensors.physx", True)

    mod = importlib.import_module("isaacsim.sensors.physx._range_sensor")
    return mod.acquire_lidar_sensor_interface()


def read_lidar_scan(
    lidar_iface,
    lidar_path: str,
):
    """
    读取 LiDAR 当前整幅扫描，并整理成 2D 栅格。

    返回:
        depth_2d:   [R, C]，单位 m
        azimuth_2d: [R, C]，单位 rad
        zenith_2d:  [R, C]，单位 rad
    """
    # 读取原始数据
    num_rows = int(lidar_iface.get_num_rows(lidar_path))
    num_cols = int(lidar_iface.get_num_cols(lidar_path))

    if num_rows <= 0 or num_cols <= 0:
        raise RuntimeError(
            f"Invalid lidar scan shape from {lidar_path}: rows={num_rows}, cols={num_cols}"
        )

    linear_depth = np.array(lidar_iface.get_linear_depth_data(lidar_path), dtype=np.float32)
    azimuth = np.array(lidar_iface.get_azimuth_data(lidar_path), dtype=np.float32)
    zenith = np.array(lidar_iface.get_zenith_data(lidar_path), dtype=np.float32)

    # 常见情况：
    # - depth 长度 = rows * cols
    # - azimuth 长度 = cols
    # - zenith 长度 = rows
    #
    # 这里做成尽量稳健的 reshape / broadcast
    if linear_depth.size != num_rows * num_cols:
        raise RuntimeError(
            f"Unexpected lidar depth buffer size: got {linear_depth.size}, "
            f"expected {num_rows * num_cols}"
        )

    depth_2d = linear_depth.reshape(num_rows, num_cols)

    if azimuth.size == num_cols:
        azimuth_2d = np.broadcast_to(azimuth.reshape(1, num_cols), (num_rows, num_cols))
    elif azimuth.size == num_rows * num_cols:
        azimuth_2d = azimuth.reshape(num_rows, num_cols)
    else:
        raise RuntimeError(
            f"Unexpected azimuth buffer size: got {azimuth.size}, "
            f"expected {num_cols} or {num_rows * num_cols}"
        )

    if zenith.size == num_rows:
        zenith_2d = np.broadcast_to(zenith.reshape(num_rows, 1), (num_rows, num_cols))
    elif zenith.size == num_rows * num_cols:
        zenith_2d = zenith.reshape(num_rows, num_cols)
    else:
        raise RuntimeError(
            f"Unexpected zenith buffer size: got {zenith.size}, "
            f"expected {num_rows} or {num_rows * num_cols}"
        )

    return depth_2d, azimuth_2d, zenith_2d



def _flush_chunk(
    out_dir: Path,
    chunk_id: int,
    sequences_list: list,
    target_velocity_list: list,
    meta: dict,
):
    import torch
    if len(sequences_list) == 0:
        return

    data_tensor = torch.stack(sequences_list, dim=0)          
    target_vel_tensor = torch.stack(target_velocity_list, dim=0)  

    save_path = out_dir / f"velocity_seq_chunk_{chunk_id:05d}.pt"

    payload = {
        "sequence_data": data_tensor.cpu().to(torch.float32),
        "target_velocity_base": target_vel_tensor.cpu().to(torch.float32),
        "meta": meta,
    }
    torch.save(payload, save_path)
    print(
        f"[collect] saved {save_path} | "
        f"data shape={tuple(data_tensor.shape)} | "
        f"target shape={tuple(target_vel_tensor.shape)}"
    )


def _save_debug_rgb_image(
    image_chw: "torch.Tensor",
    save_path: Path,
):
    """
    保存单张 RGB 图像。
    输入:
        image_chw: [3, H, W]
    兼容:
        - float [0, 1]
        - float [0, 255]
        - uint8 [0, 255]
    """
    import torch
    from PIL import Image

    img = image_chw.detach().cpu()

    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected image [3,H,W], got {tuple(img.shape)}")

    if img.dtype != torch.uint8:
        img = img.to(torch.float32)
        if float(img.max().item()) <= 1.5:
            img = img * 255.0
        img = img.clamp(0.0, 255.0).to(torch.uint8)

    img_hwc = img.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(img_hwc).save(save_path)


def _dump_camera_debug_info(
    base_env,
    single_frame_images: "torch.Tensor",
    save_path: Path,
    step: int,
):
    """
    仅保存当前 TiledCamera 输出的图像，并打印运行时相机位姿。
    不再创建独立 Replicator render product，
    也绝不调用 rep.orchestrator.step()。
    """
    _save_debug_rgb_image(single_frame_images[0], save_path)

    cam_pos_rt, cam_quat_rt = base_env._get_runtime_camera_pose_w()
    cam_pos_rt_0 = [float(v) for v in cam_pos_rt[0].detach().cpu().tolist()]
    cam_quat_rt_0 = [float(v) for v in cam_quat_rt[0].detach().cpu().tolist()]

    cam_pos_data_0 = None
    if (
        hasattr(base_env, "_vision_camera")
        and base_env._vision_camera is not None
        and hasattr(base_env._vision_camera, "data")
        and hasattr(base_env._vision_camera.data, "pos_w")
        and base_env._vision_camera.data.pos_w is not None
    ):
        cam_pos_data_0 = [
            float(v) for v in base_env._vision_camera.data.pos_w[0].detach().cpu().tolist()
        ]

    print(
        "[debug image capture]",
        f"step={step}",
        f"current={save_path}",
        f"runtime_cam_pos0={cam_pos_rt_0}",
        f"runtime_cam_quat0={cam_quat_rt_0}",
        f"sensor_cam_pos0={cam_pos_data_0}",
    )


def legacy_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--locomotion_policy",
        type=Path,
        required=True,
        help="TorchScript locomotion policy; paired conditions use the same command.",
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--dynamic_target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target_distance_min", type=float, default=80.0)
    parser.add_argument("--target_distance_max", type=float, default=150.0)
    parser.add_argument("--target_height_min", type=float, default=0.5)
    parser.add_argument("--target_height_max", type=float, default=3.0)
    parser.add_argument("--target_cam_yaw_min", type=float, default=-0.26)
    parser.add_argument("--target_cam_yaw_max", type=float, default=0.26)
    parser.add_argument("--target_cam_pitch_min", type=float, default=-0.14)
    parser.add_argument("--target_cam_pitch_max", type=float, default=0.14)
    parser.add_argument("--target_sphere_radius", type=float, default=0.4)
    parser.add_argument("--camera_projection_type", type=str, default="perspective")
    parser.add_argument("--camera_focal_length", type=float, default=10.0)
    parser.add_argument("--camera_horizontal_aperture", type=float, default=5.76)
    parser.add_argument("--camera_vertical_aperture", type=float, default=3.24)
    parser.add_argument("--camera_clip_near", type=float, default=0.1)
    parser.add_argument("--camera_clip_far", type=float, default=1000000.0)
    parser.add_argument("--camera_f_stop", type=float, default=0.0)
    parser.add_argument("--vision_distance_bounce_margin", type=float, default=0.2)
    parser.add_argument("--raycast_min_range", type=float, default=65.0)
    parser.add_argument("--raycast_max_range", type=float, default=165.0)
    parser.add_argument("--raycast_h_fov_deg", type=float, default=40.0)
    parser.add_argument("--raycast_v_fov_deg", type=float, default=24.0)
    parser.add_argument("--raycast_h_res_deg", type=float, default=0.15)
    parser.add_argument("--raycast_v_res_deg", type=float, default=0.15)
    parser.add_argument("--min_visible_radius_pix", type=float, default=1.0)
    parser.add_argument("--target_speed_min", type=float, default=0.0)
    parser.add_argument("--target_speed_max", type=float, default=5.0)
    parser.add_argument("--target_vz_min", type=float, default=-0.5)
    parser.add_argument("--target_vz_max", type=float, default=0.5)
    parser.add_argument("--target_max_speed_xy", type=float, default=5.0)
    parser.add_argument("--target_acc_min", type=float, default=0.0)
    parser.add_argument("--target_acc_max", type=float, default=0.0)
    parser.add_argument("--target_vz_acc_min", type=float, default=0.0)
    parser.add_argument("--target_vz_acc_max", type=float, default=0.0)
    parser.add_argument("--sequence_length", type=int, default=9)
    parser.add_argument("--sequence_stride", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--chunk_size", type=int, default=2048) # 每个文件保存多少个 9 帧序列
    parser.add_argument("--max_lidar_invalid_frames", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="velocity_dataset")
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--action_scale", type=float, default=0.2)
    parser.add_argument("--debug_diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_debug_frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--locomotion_policy", type=str, default=None)
    parser.add_argument(
        "--vision_model_path",
        type=str,
        default="/home/amx/Documents/isaacsim4.5.0/IsaacLab/source/standalone/test4/vision_model/vision_runs/vision_runs_cnn_only/04-04-11-12/vision_best.pth",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_img_dir = out_dir / "debug_frames"
    if args.save_debug_frames or args.debug_diagnostics:
        debug_img_dir.mkdir(parents=True, exist_ok=True)

    # Keep Kit's rendering/PhysX GPU aligned with env_cfg.sim.device.  Without
    # forwarding this argument AppLauncher silently defaults to cuda:0, even
    # when the collector was invoked with ``--device cuda:1``.
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=bool(args.enable_cameras),
        device=str(args.device),
    )
    sim_app = app_launcher.app
    env = None
    import omni.kit.app

    ext_mgr = omni.kit.app.get_app().get_extension_manager()

    # 兼容 Isaac Sim 4.5 常见名字
    for ext_name in [
        "isaacsim.sensors.physx",
        "omni.isaac.range_sensor",
    ]:
        try:
            if not ext_mgr.is_extension_enabled(ext_name):
                ext_mgr.set_extension_enabled_immediate(ext_name, True)
                print(f"[lidar ext] enabled: {ext_name}")
            else:
                print(f"[lidar ext] already enabled: {ext_name}")
        except Exception as e:
            print(f"[lidar ext] failed to enable {ext_name}: {e}")

    try:
        import gymnasium as gym
        # 确保导入 euler 转换和四元数旋转函数 
        from isaaclab.utils.math import euler_xyz_from_quat, quat_apply
        from turret_aim_env_collect import TurretAimCollectEnv, TurretAimCollectEnvCfg
    except ModuleNotFoundError as exc:
        raise

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)    
    
    # ------------------------------------------------------------
    # 加载视觉模型
    # ------------------------------------------------------------
    vision_model = load_vision_model(
        model_path=args.vision_model_path,
        device=str(args.device),
    )

    # LiDAR 接口

    env_cfg = TurretAimCollectEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(args.device)
    env_cfg.vision_history_length = 1

    # 关键：绕开 camera/xform view 的 Fabric 初始化路径
    env_cfg.sim.use_fabric = False

    env_cfg.training_stage = args.training_stage
    env_cfg.debug_vis = not args.headless
    env_cfg.debug_diagnostics = bool(args.debug_diagnostics)
    env_cfg.enable_dynamic_target = bool(args.dynamic_target)

    # 使用 MultiMeshRayCaster 作为动态目标距离源
    env_cfg.enable_dynamic_target_raycast_debug = True
    env_cfg.dynamic_target_raycast_debug_every = 20   # 少打印一点
    env_cfg.dynamic_target_raycast_min_range = float(args.raycast_min_range)
    env_cfg.dynamic_target_raycast_max_range = float(args.raycast_max_range)
    env_cfg.dynamic_target_raycast_horizontal_fov_deg = float(args.raycast_h_fov_deg)
    env_cfg.dynamic_target_raycast_vertical_fov_deg = float(args.raycast_v_fov_deg)
    env_cfg.dynamic_target_raycast_horizontal_resolution_deg = float(args.raycast_h_res_deg)
    env_cfg.dynamic_target_raycast_vertical_resolution_deg = float(args.raycast_v_res_deg)

    env_cfg.target_radius_range = (
        float(args.target_distance_min),
        float(args.target_distance_max),
    )
    env_cfg.target_height_range = (
        float(args.target_height_min),
        float(args.target_height_max),
    )
    env_cfg.target_cam_yaw_range = (
        float(args.target_cam_yaw_min),
        float(args.target_cam_yaw_max),
    )
    env_cfg.target_cam_pitch_range = (
        float(args.target_cam_pitch_min),
        float(args.target_cam_pitch_max),
    )
    env_cfg.vision_valid_distance_range = (
        float(args.target_distance_min),
        float(args.target_distance_max),
    )
    env_cfg.target_sphere_radius = float(args.target_sphere_radius)
    env_cfg.camera_projection_type = str(args.camera_projection_type)
    env_cfg.camera_focal_length = float(args.camera_focal_length)
    env_cfg.camera_horizontal_aperture = float(args.camera_horizontal_aperture)
    env_cfg.camera_vertical_aperture = float(args.camera_vertical_aperture)
    env_cfg.camera_clipping_range = (
        float(args.camera_clip_near),
        float(args.camera_clip_far),
    )
    env_cfg.camera_f_stop = float(args.camera_f_stop)
    env_cfg.vision_distance_bounce_margin = float(args.vision_distance_bounce_margin)
    env_cfg.min_visible_target_radius_pix = float(args.min_visible_radius_pix)

    env_cfg.target_speed_range = (float(args.target_speed_min), float(args.target_speed_max))
    env_cfg.target_vertical_speed_range = (float(args.target_vz_min), float(args.target_vz_max))
    env_cfg.target_max_speed_xy = float(args.target_max_speed_xy)
    env_cfg.target_acc_range = (float(args.target_acc_min), float(args.target_acc_max))
    env_cfg.target_vertical_acc_range = (
        float(args.target_vz_acc_min),
        float(args.target_vz_acc_max),
    )

    # ------------------------------------------------------------
    # locomotion 设置
    # 关键修改：
    # - 即使是“静态机器狗”场景，也允许启用 locomotion policy
    # - 只要提供了 locomotion_policy，就让腿部经过策略稳态控制
    # - 若希望机器狗静止，则把速度/角速度指令范围全部设为 0
    # ------------------------------------------------------------
    env_cfg.enable_locomotion_policy = args.locomotion_policy is not None
    env_cfg.locomotion_policy_path = args.locomotion_policy

    if env_cfg.enable_locomotion_policy:
        # 机器狗静态：仍然跑 locomotion policy，但命令全 0
        if args.training_stage == 1:
            env_cfg.locomotion_lin_vel_x_range = (0.0, 0.0)
            env_cfg.locomotion_lin_vel_y_range = (0.0, 0.0)
            env_cfg.locomotion_ang_vel_z_range = (0.0, 0.0)
            env_cfg.locomotion_command_resample_s = 1.0
            print("[collect] stage=1: enable locomotion policy with zero commands for stable standing.")
        else:
            # 动态机器狗：保持原来的随机指令范围
            print("[collect] stage!=1: enable locomotion policy with sampled locomotion commands.")
    else:
        print("[collect] locomotion policy disabled.")

    TASK_ID = "Standalone-TurretAim-VelocityCollect-v0"
    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point=TurretAimCollectEnv)

    env = gym.make(TASK_ID, cfg=env_cfg, disable_env_checker=True)
    base_env: Any = env.unwrapped


    obs, _ = env.reset()

    print("[collect] PhysX runtime lidar disabled; MultiMeshRayCaster enabled for target distance.")
    sequence_length = max(2, int(args.sequence_length))
    sequence_stride = max(1, int(args.sequence_stride))
    sequence_duration_s = (sequence_length - 1) * sequence_stride * float(base_env.step_dt)
    print(
        "[collect] sequence window:",
        f"frames={sequence_length}",
        f"stride={sequence_stride}",
        f"duration={sequence_duration_s:.3f}s",
    )
    print(
        "[collect] target acceleration:",
        f"xy={env_cfg.target_acc_range}",
        f"z={env_cfg.target_vertical_acc_range}",
    )
    sequence_buffers = [[] for _ in range(base_env.num_envs)]
    lidar_invalid_flag_buffers = [[] for _ in range(base_env.num_envs)]

    collected_sequences = []
    collected_target_velocities = []

    existing_chunk_ids = []
    for existing_path in out_dir.glob("velocity_seq_chunk_*.pt"):
        try:
            existing_chunk_ids.append(int(existing_path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    chunk_id = (max(existing_chunk_ids) + 1) if existing_chunk_ids else 0
    if existing_chunk_ids:
        print(
            "[collect warning] appending to an existing dataset directory; "
            "make sure older chunks use the same sequence_length and sequence_stride."
        )
    total_samples = 0

    meta = {
        "created_at": datetime.now().isoformat(),
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "sequence_duration_s": sequence_duration_s,
        "target_acc_range": env_cfg.target_acc_range,
        "target_vertical_acc_range": env_cfg.target_vertical_acc_range,
        "feature_dim": 26,
        "feature_definition": [
            "timestamp (1)",
            "gt_pos_base_X, gt_pos_base_Y, gt_pos_base_Z (3)",
            "pred_pos_base_X, pred_pos_base_Y, pred_pos_base_Z (3)",
            "refined_pos_base_X, refined_pos_base_Y, refined_pos_base_Z (3)",
            "robot_ang_vel_x, robot_ang_vel_y, robot_ang_vel_z (3)",
            "robot_lin_vel_x, robot_lin_vel_y, robot_lin_vel_z (3)",
            "dog_moving_flag (1)",
            "robot_roll, robot_pitch, robot_yaw (3)",
            "pred_yaw (1)",
            "pred_pitch (1)",
            "refined_yaw (1)",
            "refined_pitch (1)",
            "gt_yaw (1)",
            "gt_pitch (1)",
        ],
    }

    current_random_action = torch.zeros((base_env.num_envs, 2), device=base_env.device)

    # 轻量诊断：检查视觉输出是否长期“几乎不变”
    prev_pred_yaw = None
    prev_pred_pitch = None
    prev_img0 = None

    # 用于 valid_mask=False 时的缓存（仅在 done 时清空）
    last_valid_pred_yaw = torch.zeros(
        (base_env.num_envs,), device=base_env.device, dtype=torch.float32
    )
    last_valid_pred_pitch = torch.zeros(
        (base_env.num_envs,), device=base_env.device, dtype=torch.float32
    )
    has_last_valid_pred = torch.zeros(
        (base_env.num_envs,), device=base_env.device, dtype=torch.bool
    )

    for step in range(args.num_steps):
        action = torch.zeros((base_env.num_envs, 3), device=base_env.device, dtype=torch.float32)

        if float(args.action_scale) > 0.0 and step % 20 == 0:
            current_random_action = (
                (torch.rand((base_env.num_envs, 2), device=base_env.device) * 2.0 - 1.0)
                * float(args.action_scale)
            )
        action[:, :2] = current_random_action
        action[:, 2] = -1.0 # 不开火

        obs, rew, terminated, truncated, info = env.step(action)

        terminated_t = torch.as_tensor(terminated, device=base_env.device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=base_env.device, dtype=torch.bool)
        done_mask = terminated_t | truncated_t

        # 只在 env 真正 done/reset 时清空缓存，避免 warmup 期间反复清零导致自锁
        done_env_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            last_valid_pred_yaw[done_env_ids] = 0.0
            last_valid_pred_pitch[done_env_ids] = 0.0
            has_last_valid_pred[done_env_ids] = False

        base_env.refresh_vision_supervision()
        batch = base_env.get_vision_supervision_batch()

        # --------------------------------------------------------------
        # 先把 batch 里的字段取出来，再做检查
        # --------------------------------------------------------------
        images = batch["images"]         # [N,1,3,H,W] or [N,3,H,W]
        direction = batch["direction"]   # [N,2]
        velocity = batch["velocity"]     # [N,3]

        lidar_valid_mask = batch.get("lidar_valid_mask", None)
        if lidar_valid_mask is None:
            lidar_valid_mask = torch.ones((base_env.num_envs,), dtype=torch.bool, device=base_env.device)
        else:
            lidar_valid_mask = lidar_valid_mask.to(device=base_env.device, dtype=torch.bool)

        # valid_mask 只用于“样本门控/缓存门控”，不再用于主动 reset
        valid_mask = batch.get("valid_mask", None)
        if valid_mask is None:
            valid_mask = torch.ones((base_env.num_envs,), dtype=torch.bool, device=base_env.device)
        else:
            valid_mask = valid_mask.to(device=base_env.device, dtype=torch.bool)

        # --------------------------------------------------------------
        # 当前单帧图像：
        #   [N,1,3,H,W] -> [N,3,H,W]
        #   [N,3,H,W]   -> [N,3,H,W]
        # --------------------------------------------------------------
        if images.ndim == 5:
            if images.shape[1] != 1:
                raise ValueError(
                    f"Expected single-frame history [N,1,3,H,W], got {tuple(images.shape)}"
                )
            single_frame_images = images[:, -1]   # [N,3,H,W]
        elif images.ndim == 4:
            single_frame_images = images
        else:
            raise ValueError(
                f"Unexpected images shape: {tuple(images.shape)}, expected [N,1,3,H,W] or [N,3,H,W]"
            )
        # --------------------------------------------------------------
        # 调试：step=200 和 step=280 时，只保存当前真正喂给模型的图
        # 不再走独立 Replicator render product，
        # 防止 rep.orchestrator.step() 打乱 Isaac Lab 的相机刷新节奏。
        # --------------------------------------------------------------
        if bool(args.save_debug_frames) and (step + 1) in (200, 280):
            try:
                current_path = debug_img_dir / f"step_{step + 1:04d}_env0_current.png"
                _dump_camera_debug_info(
                    base_env=base_env,
                    single_frame_images=single_frame_images,
                    save_path=current_path,
                    step=step + 1,
                )
            except Exception as e:
                print(f"[debug image capture failed] step={step + 1} err={e}")

        # 基本检查
        if single_frame_images.ndim != 4 or single_frame_images.shape[1] != 3:
            raise ValueError(
                f"Expected single_frame_images [N,3,H,W], got {tuple(single_frame_images.shape)}"
            )
        if direction.ndim != 2 or direction.shape[-1] != 2:
            raise ValueError(f"Expected direction [N,2], got {tuple(direction.shape)}")
        if velocity.ndim != 2 or velocity.shape[-1] != 3:
            raise ValueError(f"Expected velocity [N,3], got {tuple(velocity.shape)}")

        # 只对“当前有效且未 done”的 env 刷新 last-valid 缓存
        effective_mask = valid_mask & (~done_mask)

        # ------------------------------------------------------------------
        # 1. 单帧图像 -> 单帧 pred_yaw / pred_pitch
        #    然后再由 sequence_buffers 连续累积成 9 帧序列
        # ------------------------------------------------------------------
        dog_moving_flag = batch.get("dog_moving_flag", None)
        if dog_moving_flag is None:
            robot_speed_xy = torch.norm(
                base_env.turret.data.root_lin_vel_b[:, :2], dim=-1, keepdim=True
            )
            dog_moving_flag = (robot_speed_xy > 0.05).to(torch.float32)
        else:
            if dog_moving_flag.ndim == 1:
                dog_moving_flag = dog_moving_flag.unsqueeze(-1)
            dog_moving_flag = dog_moving_flag.to(device=base_env.device, dtype=torch.float32)

        raw_pred_yaw, raw_pred_pitch = run_vision_model(
            vision_model=vision_model,
            images=single_frame_images,   # 关键：这里只喂单帧
            dog_moving_flag=dog_moving_flag,
        )

        raw_pred_yaw = raw_pred_yaw.to(device=base_env.device, dtype=torch.float32).view(-1)
        raw_pred_pitch = raw_pred_pitch.to(device=base_env.device, dtype=torch.float32).view(-1)
        raw_pred_yaw = torch.nan_to_num(raw_pred_yaw, nan=0.0, posinf=0.0, neginf=0.0)
        raw_pred_pitch = torch.nan_to_num(raw_pred_pitch, nan=0.0, posinf=0.0, neginf=0.0)

        # --------------------------------------------------------------
        # 严格模式：
        # 只有“当前帧有效且未 done”的 env，才允许用当前预测并更新缓存。
        # 当前帧无效时，虽然仍给 pred 一个数值以避免后续张量出错，
        # 但该帧绝不会进入 9 帧样本，因为后面的 strict_keep_mask 会直接清空 buffer。
        # --------------------------------------------------------------
        pred_yaw = raw_pred_yaw.clone()
        pred_pitch = raw_pred_pitch.clone()

        # 标记哪些 env 当前帧并不是“真实有效帧”
        invalid_current_frame_mask = ~effective_mask

        # --------------------------------------------------------------
        # 当前帧无效时，不再复用 last-valid 预测；
        # 直接把 pred_yaw / pred_pitch 置 0，
        # 只用于保证后续 raycast / 张量计算不炸。
        # 这些无效帧后面不会进入有效 9 帧样本。
        # --------------------------------------------------------------
        if torch.any(invalid_current_frame_mask):
            pred_yaw[invalid_current_frame_mask] = 0.0
            pred_pitch[invalid_current_frame_mask] = 0.0

        # 只有当前帧真的有效，才使用当前预测并更新缓存
        if torch.any(effective_mask):
            pred_yaw[effective_mask] = raw_pred_yaw[effective_mask]
            pred_pitch[effective_mask] = raw_pred_pitch[effective_mask]

            last_valid_pred_yaw[effective_mask] = raw_pred_yaw[effective_mask]
            last_valid_pred_pitch[effective_mask] = raw_pred_pitch[effective_mask]
            has_last_valid_pred[effective_mask] = True

        # --------------------------------------------------------------
        # 把“当前帧视觉角度（当前预测 + last-valid 回退）”送给 RayCaster，
        # 然后立刻在“当前帧”上重跑一次 raycast 筛选。
        #
        # 这样后面读到的 lidar_distance / lidar_valid_mask / refined_yaw / refined_pitch
        # 就都是“当前帧图像 + 当前帧雷达”的同步结果，而不是上一帧角度筛当前帧雷达。
        # --------------------------------------------------------------
        if hasattr(base_env, "set_dynamic_target_raycast_angles"):
            base_env.set_dynamic_target_raycast_angles(
                yaw=pred_yaw.detach().clone(),
                pitch=pred_pitch.detach().clone(),
            )

            # 关键：立刻在当前帧上重跑一次 RayCaster 角窗筛选
            if hasattr(base_env, "run_dynamic_target_raycast_for_current_frame"):
                base_env.run_dynamic_target_raycast_for_current_frame()
            elif hasattr(base_env, "_debug_raycast_dynamic_target"):
                # 兼容旧代码；如果你已经加了公开接口，正常不会走到这里
                base_env._debug_raycast_dynamic_target()

            # 重新读取当前帧 batch，拿到“当前帧同步”的 raycast 结果
            batch = base_env.get_vision_supervision_batch()

            lidar_valid_mask = batch.get("lidar_valid_mask", None)
            if lidar_valid_mask is None:
                lidar_valid_mask = torch.ones(
                    (base_env.num_envs,),
                    dtype=torch.bool,
                    device=base_env.device,
                )
            else:
                lidar_valid_mask = lidar_valid_mask.to(
                    device=base_env.device,
                    dtype=torch.bool,
                )

        # GT 只用于评估/记录，不能参与 RayCaster 角度设置
        gt_yaw = batch["direction"][:, 0].to(device=base_env.device, dtype=torch.float32).view(-1)
        gt_pitch = batch["direction"][:, 1].to(device=base_env.device, dtype=torch.float32).view(-1)

        # refined 角度：现在这里读到的是“当前帧视觉角度筛当前帧雷达”后的结果
        refined_yaw = batch.get("refined_yaw", None)
        refined_pitch = batch.get("refined_pitch", None)

        if refined_yaw is None:
            refined_yaw = pred_yaw.clone()
        else:
            refined_yaw = refined_yaw.to(device=base_env.device, dtype=torch.float32).view(-1)

        if refined_pitch is None:
            refined_pitch = pred_pitch.clone()
        else:
            refined_pitch = refined_pitch.to(device=base_env.device, dtype=torch.float32).view(-1)

        refined_yaw = torch.nan_to_num(refined_yaw, nan=0.0, posinf=0.0, neginf=0.0)
        refined_pitch = torch.nan_to_num(refined_pitch, nan=0.0, posinf=0.0, neginf=0.0)

        # 对于当前 raycast 无效的 env，refined 角度没有意义，回退到视觉角度
        refined_invalid_mask = ~lidar_valid_mask.bool()
        if torch.any(refined_invalid_mask):
            refined_yaw[refined_invalid_mask] = pred_yaw[refined_invalid_mask]
            refined_pitch[refined_invalid_mask] = pred_pitch[refined_invalid_mask]

        # ------------------------------------------------------------------
        # 2. 距离估计：
        #    - 如果 env 侧 LiDAR 可用，就走真实 LiDAR
        #    - 否则自动 fallback 到 batch["lidar_distance"] / batch["distance"]
        # ------------------------------------------------------------------
        pred_distance = estimate_distance_from_lidar_and_direction(
            batch=batch,
            pred_yaw=pred_yaw,
            pred_pitch=pred_pitch,
            base_env=base_env,
        )

        pred_distance = pred_distance.to(device=base_env.device, dtype=torch.float32)
        pred_distance = torch.nan_to_num(pred_distance, nan=0.0, posinf=0.0, neginf=0.0)
        pred_distance = pred_distance.clamp_min(1e-4)

        gt_distance = batch["distance"]
        if gt_distance.ndim != 1:
            gt_distance = gt_distance.squeeze(-1)
        gt_distance = gt_distance.to(device=base_env.device, dtype=torch.float32)
        gt_distance = torch.nan_to_num(gt_distance, nan=0.0, posinf=0.0, neginf=0.0)

        if bool(args.debug_diagnostics) and (step + 1) % 100 == 0:
            print(
                "[distance debug]",
                f"step={step+1}",
                f"gt_yaw[0]={float(gt_yaw[0].item()):.4f}",
                f"gt_pitch[0]={float(gt_pitch[0].item()):.4f}",
                f"pred_distance[0]={float(pred_distance[0].item()):.4f}",
                f"gt_distance[0]={float(gt_distance[0].item()):.4f}",
            )

            print(
                "[lidar source]",
                f"step={step+1}",
                f"valid_hits={int(lidar_valid_mask.sum().item())}/{base_env.num_envs}",
                f"dist0={float(pred_distance[0].item()):.4f}",
            )


        # ------------------------------------------------------------
        # 3. 用“视觉方向 + 雷达距离”构造相机坐标系下的预测位置
        #    ROS optical frame:
        #      x -> right
        #      y -> down
        #      z -> forward
        # ------------------------------------------------------------
        pred_x = pred_distance * torch.sin(pred_yaw) * torch.cos(pred_pitch)
        pred_y = -pred_distance * torch.sin(pred_pitch)
        pred_z = pred_distance * torch.cos(pred_yaw) * torch.cos(pred_pitch)
        pred_pos_cam = torch.stack([pred_x, pred_y, pred_z], dim=-1)  # [N, 3]

        # ------------------------------------------------------------
        # 4. 用“refined 方向 + 同一距离”构造 refined 预测位置
        # ------------------------------------------------------------
        refined_x = pred_distance * torch.sin(refined_yaw) * torch.cos(refined_pitch)
        refined_y = -pred_distance * torch.sin(refined_pitch)
        refined_z = pred_distance * torch.cos(refined_yaw) * torch.cos(refined_pitch)
        refined_pos_cam = torch.stack([refined_x, refined_y, refined_z], dim=-1)  # [N, 3]

        # ------------------------------------------------------------
        # 5. Camera -> World -> Base
        #    关键：优先使用 runtime 重建的相机位姿，避免 data pose 冻结
        # ------------------------------------------------------------
        if hasattr(base_env, "_get_runtime_camera_pose_w"):
            cam_pos_w, cam_quat_w = base_env._get_runtime_camera_pose_w()
        else:
            cam_pos_w = base_env._vision_camera.data.pos_w
            cam_quat_w = base_env._vision_camera.data.quat_w_ros

        pred_pos_w = cam_pos_w + quat_apply(cam_quat_w, pred_pos_cam)
        refined_pos_w = cam_pos_w + quat_apply(cam_quat_w, refined_pos_cam)

        root_pos_w = base_env.turret.data.root_pos_w
        root_quat_w = base_env.turret.data.root_quat_w

        if hasattr(base_env, "_rotate_world_to_yaw_only_local"):
            pred_pos_base = base_env._rotate_world_to_yaw_only_local(root_quat_w, pred_pos_w - root_pos_w)
            refined_pos_base = base_env._rotate_world_to_yaw_only_local(root_quat_w, refined_pos_w - root_pos_w)
        else:
            pred_pos_base = base_env._rotate_world_to_local(root_quat_w, pred_pos_w - root_pos_w)
            refined_pos_base = base_env._rotate_world_to_local(root_quat_w, refined_pos_w - root_pos_w)

        # ------------------------------------------------------------------
        # 2. 提取其他物理数据（统一到 yaw-only base frame）
        # ------------------------------------------------------------------
        gt_pos_base = batch["pos_base_gt"]          # [N, 3]，已是 yaw-only base
        target_velocity_base = batch["velocity_base"]  # [N, 3]，已是 yaw-only base

        root_quat_w = base_env.turret.data.root_quat_w

        # Isaac Lab 原生这两个是 body frame（真实躯干坐标系，含 roll/pitch 倾斜）
        root_lin_vel_b = base_env.turret.data.root_lin_vel_b   # [N, 3]
        root_ang_vel_b = base_env.turret.data.root_ang_vel_b   # [N, 3]

        # 先从 body frame 转回 world frame
        robot_lin_vel_w = quat_apply(root_quat_w, root_lin_vel_b)   # [N, 3]
        robot_ang_vel_w = quat_apply(root_quat_w, root_ang_vel_b)   # [N, 3]

        # 再统一转到 yaw-only base frame
        if hasattr(base_env, "_rotate_world_to_yaw_only_local"):
            robot_lin_vel = base_env._rotate_world_to_yaw_only_local(root_quat_w, robot_lin_vel_w)
            robot_ang_vel = base_env._rotate_world_to_yaw_only_local(root_quat_w, robot_ang_vel_w)
        else:
            # 兜底：若环境里还没切到 yaw-only，就退回完整 base frame
            robot_lin_vel = base_env._rotate_world_to_local(root_quat_w, robot_lin_vel_w)
            robot_ang_vel = base_env._rotate_world_to_local(root_quat_w, robot_ang_vel_w)

        # 姿态这里建议也与 yaw-only 定义保持一致：
        # roll / pitch 仍保留真实机体倾角；yaw 使用完整姿态提取出的航向角
        roll, pitch, yaw = euler_xyz_from_quat(root_quat_w)
        robot_euler = torch.stack([roll, pitch, yaw], dim=-1)  # [N, 3]

        timestamp = torch.full((base_env.num_envs, 1), step * base_env.step_dt, device=base_env.device)

        # 将 1 维的角度张量扩展为 [N, 1] 形状，以便能够与后面的张量拼接
        pred_yaw_col = pred_yaw.unsqueeze(-1)
        pred_pitch_col = pred_pitch.unsqueeze(-1)
        refined_yaw_col = refined_yaw.unsqueeze(-1)
        refined_pitch_col = refined_pitch.unsqueeze(-1)
        gt_yaw_col = gt_yaw.unsqueeze(-1)
        gt_pitch_col = gt_pitch.unsqueeze(-1)

        # 拼接单步 26 维特征向量 (全部统一为 Base 坐标系)
        current_features = torch.cat([
            timestamp,      # 1
            gt_pos_base,        # 3
            pred_pos_base,      # 3
            refined_pos_base,   # 3
            robot_ang_vel,      # 3
            robot_lin_vel,      # 3
            dog_moving_flag,    # 1
            robot_euler,        # 3
            pred_yaw_col,       # 1
            pred_pitch_col,     # 1
            refined_yaw_col,    # 1
            refined_pitch_col,  # 1
            gt_yaw_col,         # 1
            gt_pitch_col,       # 1
        ], dim=-1)

        # ------------------------------------------------------------------
        # 3. 连续 9 帧分段采样（非滑动窗口）
        #    - visual invalid / done：直接清空，重新开始
        #    - lidar invalid：默认不保存，避免用 GT 距离回填造成数据泄漏
        #    - 若显式设置 --max_lidar_invalid_frames，可放宽这个限制
        #    - 一旦满 9 帧，不管存不存，都立刻清空，开始下一段
        # ------------------------------------------------------------------
        keep_visual_mask = valid_mask.bool() & (~done_mask)

        keep_visual_mask_cpu = keep_visual_mask.cpu().numpy()
        lidar_valid_mask_cpu = lidar_valid_mask.bool().cpu().numpy()
        current_features_cpu = current_features.cpu().numpy()
        sample_this_step = (step % sequence_stride) == 0

        for env_idx in range(base_env.num_envs):
            # 视觉无效 / env done：这不是“连续帧”，直接清空重来
            if not bool(keep_visual_mask_cpu[env_idx]):
                sequence_buffers[env_idx].clear()
                lidar_invalid_flag_buffers[env_idx].clear()
                continue

            # 当前帧加入当前 9 帧段
            if not sample_this_step:
                continue

            sequence_buffers[env_idx].append(current_features_cpu[env_idx])
            lidar_invalid_flag_buffers[env_idx].append(not bool(lidar_valid_mask_cpu[env_idx]))

            # 只有凑满 9 帧，才判断是否存样本
            if len(sequence_buffers[env_idx]) == sequence_length:
                num_lidar_invalid = int(sum(lidar_invalid_flag_buffers[env_idx]))

                if num_lidar_invalid <= int(args.max_lidar_invalid_frames):
                    seq_data = torch.tensor(
                        np.array(sequence_buffers[env_idx]),
                        dtype=torch.float32,
                    )
                    collected_sequences.append(seq_data)

                    # 标签仍然用当前第 9 帧时刻目标在 base 坐标系下的真实速度
                    target_vel_label = target_velocity_base[env_idx].detach().cpu().to(torch.float32)
                    collected_target_velocities.append(target_vel_label)

                    total_samples += 1

                # 关键修改：
                # 不再使用滑动窗口。无论这一段是否通过筛选，
                # 当前 9 帧段处理完后都直接清空，重新积累下一段 9 帧。
                sequence_buffers[env_idx].clear()
                lidar_invalid_flag_buffers[env_idx].clear()

        # 轻量诊断：每 20 步打印一次预测变化量 + 图像变化量
        if bool(args.debug_diagnostics) and (step + 1) % 20 == 0:
            with torch.no_grad():
                if prev_pred_yaw is not None and prev_pred_pitch is not None:
                    dy = torch.abs(pred_yaw - prev_pred_yaw)
                    dp = torch.abs(pred_pitch - prev_pred_pitch)
                    yaw_delta_mean = float(dy.mean().item())
                    pitch_delta_mean = float(dp.mean().item())
                else:
                    yaw_delta_mean = float("nan")
                    pitch_delta_mean = float("nan")

                img0 = single_frame_images[0]
                if prev_img0 is not None:
                    img_delta = float(torch.mean(torch.abs(img0 - prev_img0)).item())
                else:
                    img_delta = float("nan")

                print(
                    "[vision drift diag]",
                    f"step={step+1}",
                    f"pred_yaw0={float(pred_yaw[0].item()):.4f}",
                    f"pred_pitch0={float(pred_pitch[0].item()):.4f}",
                    f"mean_abs_delta_yaw={yaw_delta_mean:.6f}",
                    f"mean_abs_delta_pitch={pitch_delta_mean:.6f}",
                    f"img0_mean_abs_delta={img_delta:.6f}",
                )

                prev_pred_yaw = pred_yaw.clone()
                prev_pred_pitch = pred_pitch.clone()
                prev_img0 = img0.clone()

        # 分块存盘
        if len(collected_sequences) >= args.chunk_size:
            _flush_chunk(
                out_dir=out_dir,
                chunk_id=chunk_id,
                sequences_list=collected_sequences,
                target_velocity_list=collected_target_velocities,
                meta=meta,
            )
            chunk_id += 1
            collected_sequences.clear()
            collected_target_velocities.clear()

        progress_every = int(args.progress_every)
        if progress_every > 0 and (step + 1) % progress_every == 0:
            valid_envs = int(keep_visual_mask.sum().item())
            lidar_valid_envs = int(lidar_valid_mask.sum().item())
            done_envs = int(done_mask.sum().item())
            print(
                "[collect progress]",
                f"step={step + 1}/{args.num_steps}",
                f"samples={total_samples}",
                f"pending={len(collected_sequences)}",
                f"chunks_next={chunk_id}",
                f"valid_envs={valid_envs}/{base_env.num_envs}",
                f"lidar_valid={lidar_valid_envs}/{base_env.num_envs}",
                f"done={done_envs}/{base_env.num_envs}",
            )

        # ... 略 ...

    # 循环结束，清理残留数据
    if len(collected_sequences) > 0:
        _flush_chunk(
            out_dir=out_dir,
            chunk_id=chunk_id,
            sequences_list=collected_sequences,
            target_velocity_list=collected_target_velocities,
            meta=meta,
        )
        
    if env is not None:
        env.close()
    sim_app.close()

    print(f"[collect] Collection Finished. Total {sequence_length}-Frame Sequences: {total_samples}")

@dataclass
class _TAQFrame:
    """One causal visual/range observation for a single simulated episode."""

    timestamp_s: float
    position_base_m: np.ndarray
    robot_linear_velocity_base_mps: np.ndarray
    robot_angular_velocity_base_rps: np.ndarray
    source_id: int
    quality: np.ndarray
    range_valid: bool
    target_relative_velocity_base_mps: np.ndarray
    state_reset: bool = False
    audit: dict[str, Any] = field(default_factory=dict)
    sensor_packet: dict[str, Any] | None = None


@dataclass
class _TAQSample:
    """One continuous, episode-local training window plus audit timestamps."""

    features: torch.Tensor
    source_ids: torch.Tensor
    target_relative_velocity_base_mps: torch.Tensor
    window_start_s: float
    window_end_s: float


@dataclass
class _RaycasterDistanceAudit:
    """Episode-local, GT-only diagnostic for validating the simulated range path.

    This data is deliberately not added to TAQ features and is never used to
    select samples.  It is saved only as metadata so a collection run can
    verify that the moving XForm target is actually followed by the RayCaster.
    """

    eligible_frames: int = 0
    hit_frames: int = 0
    absolute_errors_m: list[float] = field(default_factory=list)
    relative_errors: list[float] = field(default_factory=list)

    def record(self, raycast_distance_m: float, raycast_valid: bool, gt_distance_m: float) -> None:
        if not math.isfinite(gt_distance_m) or gt_distance_m <= 0.0:
            return
        self.eligible_frames += 1
        if not raycast_valid or not math.isfinite(raycast_distance_m) or raycast_distance_m <= 0.0:
            return
        absolute_error = abs(raycast_distance_m - gt_distance_m)
        self.hit_frames += 1
        self.absolute_errors_m.append(float(absolute_error))
        self.relative_errors.append(float(absolute_error / gt_distance_m))

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95)),
            "max": float(array.max()),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "gt_used_for_audit_only": True,
            "excluded_from_features_and_sample_selection": True,
            "eligible_tracker_frames": int(self.eligible_frames),
            "raycast_hit_frames": int(self.hit_frames),
            "raycast_hit_rate": float(self.hit_frames / max(self.eligible_frames, 1)),
            "absolute_distance_error_m": self._stats(self.absolute_errors_m),
            "relative_distance_error": self._stats(self.relative_errors),
        }


def _taq_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect clean or Stage-2 non-ideal TAQ v2 velocity data from "
            "LK/OSTrack and the visual-direction RayCaster."
        )
    )
    parser.add_argument("--scenario", choices=("ss", "sd"), required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--locomotion_policy",
        type=Path,
        default=None,
        help="TorchScript locomotion policy; paired conditions use the same command.",
    )
    parser.add_argument(
        "--validation_condition",
        choices=("none", "A", "B", "C", "D", "P0", "P1", "P2", "P3"),
        default="none",
        help=(
            "Explicit velocity-error validation condition: A/B use a fixed world camera, "
            "A/C use GT direction, and P2/P3 use GT range."
        ),
    )
    parser.add_argument("--pair_id", type=str, default=None)
    parser.add_argument("--locomotion_vx_mps", type=float, default=0.0)
    parser.add_argument("--locomotion_vy_mps", type=float, default=0.0)
    parser.add_argument("--locomotion_yaw_rate_radps", type=float, default=0.0)
    parser.add_argument(
        "--trajectory_seed", type=int, default=None,
        help="Seed identifying the target trajectory shared by paired conditions; defaults to --seed.",
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument(
        "--collection_run_id",
        type=str,
        default=None,
        help="Unique collection identifier. Defaults to a timestamped run ID.",
    )
    parser.add_argument("--split", choices=DATA_SPLITS, default="train")
    parser.add_argument("--capture_skip_probability", type=float, default=0.0,
                        help="Physical capture mode only: skip image/tracker sampling while physics continues.")
    parser.add_argument(
        "--sensor_time_contract", choices=("legacy_v0", "physical_capture_v1"),
        default="legacy_v0",
        help="Physical capture mode exports GT-free, timestamped sensor packets for paired delivery replay. Requires clean D complete streams.",
    )
    parser.add_argument(
        "--degradation_profile",
        choices=DEGRADATION_PROFILES,
        default="clean",
        help="Non-ideal sensor family; mixed samples 2-4 families per episode.",
    )
    parser.add_argument(
        "--severity",
        choices=DEGRADATION_SEVERITIES,
        default="medium",
        help=(
            "Fixed severity or a per-episode mild/medium/hard mix "
            "(50/35/15, or 40/40/20 for mixed)."
        ),
    )
    parser.add_argument(
        "--degradation_seed",
        type=int,
        default=None,
        help="Independent degradation seed (default: --seed + 10000).",
    )
    parser.add_argument(
        "--target_samples",
        type=int,
        default=0,
        help=(
            "Stop after at least this many pending+saved windows; complete-stream "
            "collection finishes active episodes first. 0 uses --num_steps only."
        ),
    )
    parser.add_argument(
        "--use_gt_visibility_gate_after_initialization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug compatibility option. Leave disabled for deployable Stage-2 selection.",
    )
    parser.add_argument(
        "--keep_measurement_discontinuities",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep windows across accepted OSTrack corrections (default: true for degraded data).",
    )
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument(
        "--episode_warmup_s", type=float, default=1.0,
        help=(
            "Simulation seconds excluded after EVERY episode reset. Physics and "
            "tracking continue, but no training windows, episode frames, or paired "
            "audit rows are saved during warmup. Set 0 for historical behavior."
        ),
    )
    parser.add_argument(
        "--camera_prerender_frames",
        type=int,
        default=5,
        help=(
            "Render this many camera frames immediately after simulator reset "
            "before collection starts, allowing RTX temporal buffers to converge."
        ),
    )
    parser.add_argument(
        "--debug_rgb_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for raw camera-frame diagnostics. No RGB frames "
            "are written unless this option is provided."
        ),
    )
    parser.add_argument(
        "--debug_rgb_steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Zero-based physics steps to save when --debug_rgb_dir is set; "
            "all environments are saved. If omitted, every step is saved."
        ),
    )
    parser.add_argument("--target_distance_min", type=float, default=80.0)
    parser.add_argument("--target_distance_max", type=float, default=150.0)
    parser.add_argument("--target_height_min", type=float, default=0.5)
    parser.add_argument("--target_height_max", type=float, default=3.0)
    parser.add_argument("--target_sphere_radius", type=float, default=0.4)
    parser.add_argument("--target_speed_min", type=float, default=0.0)
    parser.add_argument("--target_speed_max", type=float, default=5.0)
    parser.add_argument("--target_vz_min", type=float, default=-0.5)
    parser.add_argument("--target_vz_max", type=float, default=0.5)
    parser.add_argument("--target_acc_min", type=float, default=0.0)
    parser.add_argument("--target_acc_max", type=float, default=0.0)
    parser.add_argument("--target_vz_acc_min", type=float, default=0.0)
    parser.add_argument("--target_vz_acc_max", type=float, default=0.0)
    parser.add_argument("--camera_focal_length", type=float, default=10.0)
    parser.add_argument("--camera_horizontal_aperture", type=float, default=5.76)
    parser.add_argument("--camera_vertical_aperture", type=float, default=3.24)
    parser.add_argument(
        "--target_boundary_mode", choices=("fixed_reference_toi", "legacy_projection"),
        default="fixed_reference_toi",
        help="SD motion: fixed-world collision-time reflections, or historical camera-driven position projections.",
    )
    parser.add_argument(
        "--target_fov_margin_fraction", type=float, default=0.10,
        help="Fixed motion domain: fraction of image width/height reserved at EACH edge.",
    )
    parser.add_argument(
        "--min_timestamp_dt_ratio",
        type=float,
        default=0.75,
        help=(
            "Minimum degraded sensor dt as a fraction of nominal dt. "
            "The default keeps a 20 ms stream at or above 15 ms."
        ),
    )
    parser.add_argument("--window_s", type=float, default=0.48)
    parser.add_argument(
        "--min_window_gap_s",
        type=float,
        default=0.02,
        help="Required extra gap between consecutive saved windows in one episode.",
    )
    parser.add_argument("--supervision_burn_in_s", type=float, default=0.10)
    parser.add_argument(
        "--save_episode_stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save every accepted episode frame for true streaming evaluation.",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=3,
        help="Check for an eligible window every N tracked frames; saved windows still cannot overlap.",
    )
    parser.add_argument("--max_sequence_steps", type=int, default=32)
    parser.add_argument(
        "--max_windows_per_episode",
        type=int,
        default=4,
        help=(
            "Maximum number of non-overlapping windows saved from one episode. "
            "A small cap prevents one continuous trajectory from dominating the data set."
        ),
    )
    parser.add_argument("--min_range_valid_frames", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--min_bootstrap_box_px", type=float, default=12.0)
    parser.add_argument("--max_static_base_speed_mps", type=float, default=0.15)
    parser.add_argument("--max_static_base_angular_speed_radps", type=float, default=0.15)
    parser.add_argument("--disable_fine_tracking", action="store_true")
    parser.add_argument("--ostrack_repo", type=Path, default=None)
    parser.add_argument("--ostrack_checkpoint", type=Path, default=None)
    parser.add_argument("--ostrack_config", type=str, default="vitb_256_mae_ce_32x4_ep300")
    parser.add_argument("--fine_tracker_policy", choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument("--fine_tracker_interval_frames", type=int, default=3)
    parser.add_argument("--fine_tracker_min_score", type=float, default=0.20)
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
    parser.add_argument(
        "--measurement_rejection_innovation_threshold", type=float, default=0.80
    )
    parser.add_argument(
        "--measurement_rejection_iou_threshold", type=float, default=0.05
    )
    parser.add_argument("--adaptive_failure_backoff_initial_frames", type=int, default=1)
    parser.add_argument("--adaptive_failure_backoff_max_frames", type=int, default=8)
    parser.add_argument("--fine_max_center_jump_box_diagonals", type=float, default=0.50)
    parser.add_argument(
        "--fine_discontinuity_center_jump_box_diagonals", type=float, default=0.25
    )
    parser.add_argument("--fine_max_scale_change", type=float, default=1.75)
    args = parser.parse_args()
    if args.degradation_seed is None:
        args.degradation_seed = int(args.seed) + 10000
    if args.keep_measurement_discontinuities is None:
        args.keep_measurement_discontinuities = args.degradation_profile != "clean"
    if args.collection_run_id is None:
        args.collection_run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.trajectory_seed is None:
        args.trajectory_seed = int(args.seed)
    return args


def _validate_taq_args(args: argparse.Namespace) -> None:
    if args.num_envs < 1 or args.num_steps < 1 or args.chunk_size < 1:
        raise ValueError("--num_envs, --num_steps, and --chunk_size must be positive")
    if args.sensor_time_contract == TIME_CONTRACT_VERSION and (
        args.degradation_profile != "clean" or args.validation_condition != "D" or not args.save_episode_stream
    ):
        raise ValueError("physical_capture_v1 requires clean condition D and --save_episode_stream; timing degradation is explicit packet delivery replay")
    if not 0 <= args.capture_skip_probability < 1 or (
        args.capture_skip_probability and args.sensor_time_contract != TIME_CONTRACT_VERSION
    ):
        raise ValueError("Capture skipping requires physical_capture_v1 and probability in [0,1)")
    if args.debug_rgb_steps is not None and any(
        step < 0 for step in args.debug_rgb_steps
    ):
        raise ValueError("--debug_rgb_steps values must be non-negative")
    if args.debug_rgb_steps is not None and args.debug_rgb_dir is None:
        raise ValueError("--debug_rgb_steps requires --debug_rgb_dir")
    if args.camera_prerender_frames < 0:
        raise ValueError("--camera_prerender_frames must be non-negative")
    if not math.isfinite(args.episode_warmup_s) or args.episode_warmup_s < 0:
        raise ValueError("--episode_warmup_s must be finite and non-negative")
    if args.target_samples < 0:
        raise ValueError("--target_samples cannot be negative")
    if args.window_s <= 0.0 or args.sample_stride < 1 or args.max_sequence_steps < 2:
        raise ValueError("Invalid sequence-window configuration")
    if args.min_window_gap_s < 0.0 or args.supervision_burn_in_s < 0.0:
        raise ValueError("Window gap and supervision burn-in must be non-negative")
    if args.supervision_burn_in_s >= args.window_s:
        raise ValueError("--supervision_burn_in_s must be shorter than --window_s")
    if not 0.0 < args.min_timestamp_dt_ratio <= 1.0:
        raise ValueError("--min_timestamp_dt_ratio must be in (0, 1]")
    if args.max_windows_per_episode < 1:
        raise ValueError("--max_windows_per_episode must be positive")
    if args.min_range_valid_frames < 1:
        raise ValueError("--min_range_valid_frames must be positive")
    if args.target_distance_max <= args.target_distance_min or args.target_sphere_radius <= 0.0:
        raise ValueError("Invalid target geometry configuration")
    if args.target_speed_min < 0.0 or args.target_speed_max < args.target_speed_min:
        raise ValueError("Invalid target-speed range")
    if not math.isfinite(args.target_fov_margin_fraction) or not 0 <= args.target_fov_margin_fraction < .4:
        raise ValueError("--target_fov_margin_fraction must be in [0, 0.4)")
    if args.camera_focal_length <= 0.0 or args.camera_horizontal_aperture <= 0.0 or args.camera_vertical_aperture <= 0.0:
        raise ValueError("Camera calibration values must be positive")
    if args.max_static_base_speed_mps <= 0.0 or args.max_static_base_angular_speed_radps <= 0.0:
        raise ValueError("Static-base speed thresholds must be positive")
    if not (
        0.0
        < args.fine_discontinuity_center_jump_box_diagonals
        <= args.fine_max_center_jump_box_diagonals
    ):
        raise ValueError(
            "The fine-tracker discontinuity threshold must be positive and no "
            "greater than its rejection threshold"
        )
    if args.fine_max_scale_change <= 1.0:
        raise ValueError("--fine_max_scale_change must exceed 1")
    if args.measurement_rejection_innovation_threshold <= 0.0:
        raise ValueError(
            "--measurement_rejection_innovation_threshold must be positive"
        )
    if not 0.0 <= args.measurement_rejection_iou_threshold <= 1.0:
        raise ValueError("--measurement_rejection_iou_threshold must be in [0, 1]")
    for name in ("locomotion_vx_mps", "locomotion_vy_mps", "locomotion_yaw_rate_radps"):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"--{name} must be finite")
    if args.validation_condition != "none" and not args.pair_id:
        raise ValueError("--pair_id is required for a validation condition")
    if args.validation_condition != "none" and int(args.trajectory_seed) != int(args.seed):
        raise ValueError(
            "This collector has no external trajectory replay yet, so --trajectory_seed "
            "must equal --seed for paired validation"
        )
    if args.locomotion_policy is None:
        raise ValueError(
            "--locomotion_policy is required for every condition; paired A/B must "
            "execute the same robot physics and policy path as C/D"
        )
    if not args.locomotion_policy.expanduser().resolve().is_file():
        raise FileNotFoundError(f"--locomotion_policy was not found: {args.locomotion_policy}")
    if (args.ostrack_repo is None) != (args.ostrack_checkpoint is None):
        raise ValueError("Provide both --ostrack_repo and --ostrack_checkpoint, or neither for LK-only collection")
    for name in ("ostrack_repo", "ostrack_checkpoint"):
        value = getattr(args, name)
        if value is not None and not value.expanduser().resolve().exists():
            raise FileNotFoundError(f"--{name} was not found: {value}")


def _rgb_chw_batch_to_uint8(images: torch.Tensor) -> np.ndarray:
    """Convert current camera frames ``[N,3,H,W]`` into tracker RGB images."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected images [N,3,H,W], got {tuple(images.shape)}")
    value = images.detach().to(device="cpu", dtype=torch.float32)
    if value.numel() and float(value.max().item()) <= 1.5:
        value = value * 255.0
    return np.ascontiguousarray(value.clamp(0.0, 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy())


def _background_global_flow_px(
    previous_rgb: np.ndarray | None,
    current_rgb: np.ndarray,
    target_bbox_xyxy: tuple[float, float, float, float] | None,
) -> float | None:
    """Estimate robust background image displacement between two frames."""
    if previous_rgb is None or previous_rgb.shape != current_rgb.shape:
        return None
    import cv2

    previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.full(previous_gray.shape, 255, dtype=np.uint8)
    if target_bbox_xyxy is not None:
        x1, y1, x2, y2 = target_bbox_xyxy
        padding = 12
        x1_i = max(0, int(math.floor(x1)) - padding)
        y1_i = max(0, int(math.floor(y1)) - padding)
        x2_i = min(mask.shape[1], int(math.ceil(x2)) + padding)
        y2_i = min(mask.shape[0], int(math.ceil(y2)) + padding)
        mask[y1_i:y2_i, x1_i:x2_i] = 0
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=160,
        qualityLevel=0.01,
        minDistance=8,
        mask=mask,
    )
    if points is None or len(points) < 8:
        return None
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if next_points is None or status is None:
        return None
    good = status.reshape(-1).astype(bool)
    if int(good.sum()) < 8:
        return None
    displacement = next_points.reshape(-1, 2)[good] - points.reshape(-1, 2)[good]
    magnitudes = np.linalg.norm(displacement, axis=1)
    median = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - median)))
    inliers = magnitudes <= median + max(3.0 * 1.4826 * mad, 0.5)
    return float(np.median(magnitudes[inliers])) if np.any(inliers) else median


def _quaternion_wxyz_to_rpy(quaternion: torch.Tensor) -> list[float]:
    w, x, y, z = (float(value) for value in quaternion.detach().cpu().tolist())
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_sine)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


def _quaternion_distance_rad(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the shortest angular distance between two wxyz quaternions."""
    first_np = first.detach().cpu().numpy().astype(np.float64)
    second_np = second.detach().cpu().numpy().astype(np.float64)
    first_np /= max(float(np.linalg.norm(first_np)), 1e-12)
    second_np /= max(float(np.linalg.norm(second_np)), 1e-12)
    dot = float(np.clip(abs(np.dot(first_np, second_np)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _camera_velocity_from_pose(
    previous: tuple[np.ndarray, np.ndarray] | None,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate world-frame camera linear/angular velocity from two poses."""
    current_position = position.detach().cpu().numpy().astype(np.float64)
    current_quaternion = quaternion.detach().cpu().numpy().astype(np.float64)
    current_quaternion /= max(float(np.linalg.norm(current_quaternion)), 1e-12)
    if previous is None or dt_s <= 0.0:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    previous_position, previous_quaternion = previous
    previous_quaternion = previous_quaternion / max(
        float(np.linalg.norm(previous_quaternion)), 1e-12
    )
    if float(np.dot(previous_quaternion, current_quaternion)) < 0.0:
        current_quaternion = -current_quaternion
    linear_velocity = (current_position - previous_position) / float(dt_s)
    pw, px, py, pz = previous_quaternion
    cw, cx, cy, cz = current_quaternion
    # q_delta = q_current * conjugate(q_previous), so the axis is in world coordinates.
    delta = np.asarray(
        [
            cw * pw + cx * px + cy * py + cz * pz,
            -cw * px + cx * pw - cy * pz + cz * py,
            -cw * py + cx * pz + cy * pw - cz * px,
            -cw * pz - cx * py + cy * px + cz * pw,
        ],
        dtype=np.float64,
    )
    delta /= max(float(np.linalg.norm(delta)), 1e-12)
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm <= 1e-12:
        angular_velocity = np.zeros(3, dtype=np.float64)
    else:
        angle = 2.0 * math.atan2(vector_norm, float(np.clip(delta[0], -1.0, 1.0)))
        angular_velocity = delta[1:] * (angle / (vector_norm * float(dt_s)))
    return linear_velocity, angular_velocity


def _camera_intrinsics_from_image(args: argparse.Namespace, width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(width) * float(args.camera_focal_length) / float(args.camera_horizontal_aperture),
        fy=float(height) * float(args.camera_focal_length) / float(args.camera_vertical_aperture),
        cx=0.5 * float(width),
        cy=0.5 * float(height),
        width=width,
        height=height,
    )


def _bootstrap_box_from_gt(
    yaw_rad: float,
    pitch_rad: float,
    distance_m: float,
    intrinsics: CameraIntrinsics,
    target_radius_m: float,
    min_box_px: float,
) -> tuple[float, float, float, float] | None:
    """Produce a GT box only for tracker initialization in simulation.

    It is not persisted as a visual measurement and is never used after the
    tracker has been initialized.  This mirrors conventional detector-to-
    tracker hand-off while avoiding a detector dependency in this collector.
    """
    if not np.isfinite((yaw_rad, pitch_rad, distance_m)).all() or distance_m <= 1e-5:
        return None
    x_norm = float(np.tan(yaw_rad))
    y_norm = -float(np.tan(pitch_rad)) * float(np.sqrt(1.0 + x_norm * x_norm))
    center_x = intrinsics.cx + intrinsics.fx * x_norm
    center_y = intrinsics.cy + intrinsics.fy * y_norm
    radius_px = max(float(min_box_px) * 0.5, intrinsics.fx * float(target_radius_m) / float(distance_m))
    x1, y1 = center_x - radius_px, center_y - radius_px
    x2, y2 = center_x + radius_px, center_y + radius_px
    if x2 <= 1.0 or y2 <= 1.0 or x1 >= float(intrinsics.width) - 1.0 or y1 >= float(intrinsics.height) - 1.0:
        return None
    return (
        max(0.0, x1), max(0.0, y1),
        min(float(intrinsics.width), x2), min(float(intrinsics.height), y2),
    )


def _gt_validation_estimate(
    *,
    yaw_rad: float,
    pitch_rad: float,
    distance_m: float,
    bbox_xyxy: tuple[float, float, float, float],
    initial: bool,
) -> TargetEstimate:
    """Build an oracle direction observation without exposing GT velocity."""
    x1, y1, x2, y2 = bbox_xyxy
    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    source = "initial" if initial else "flow"
    return TargetEstimate(
        target_id=0,
        bbox_xyxy=bbox_xyxy,
        predicted_bbox_xyxy=bbox_xyxy,
        center_xy=center,
        predicted_center_xy=center,
        yaw_rad=float(yaw_rad),
        pitch_rad=float(pitch_rad),
        predicted_yaw_rad=float(yaw_rad),
        predicted_pitch_rad=float(pitch_rad),
        distance_m=max(float(distance_m), 0.0),
        speed_m_s=0.0,
        kinematics_source="validation_gt_projection",
        distance_priority_score=1.0,
        speed_priority_score=1.0,
        confidence=1.0,
        priority=1.0,
        status="active",
        source=source,
        corners=16,
        ransac_inliers=16,
        median_fb_error_px=0.0,
        missed_frames=0,
        selected=True,
        motion_confidence=1.0,
        point_retention=1.0,
        ransac_inlier_ratio=1.0,
        bbox_innovation=0.0,
        bbox_iou_with_prediction=1.0,
        fine_policy="validation_gt_projection",
        fine_action="none",
        fine_trigger_reason="oracle_condition",
    )


def _direction_angular_error_deg(
    predicted_yaw: float,
    predicted_pitch: float,
    gt_yaw: float,
    gt_pitch: float,
) -> float:
    def vector(yaw: float, pitch: float) -> np.ndarray:
        return np.asarray(
            [math.cos(pitch) * math.sin(yaw), -math.sin(pitch), math.cos(pitch) * math.cos(yaw)],
            dtype=np.float64,
        )

    dot = float(np.dot(vector(predicted_yaw, predicted_pitch), vector(gt_yaw, gt_pitch)))
    return float(math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0)))))


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _build_taq_tracker(args: argparse.Namespace, intrinsics: CameraIntrinsics) -> MultiTargetVisionTracker:
    fine_tracker = None
    if not args.disable_fine_tracking and args.ostrack_repo is not None:
        fine_tracker = OSTrackFrameTracker(
            repo_path=args.ostrack_repo.expanduser().resolve(),
            checkpoint=args.ostrack_checkpoint.expanduser().resolve(),
            config=str(args.ostrack_config),
            device=str(args.device),
        )
        fine_tracker.prepare()
    config = MultiTargetConfig(
        target_width_m=2.0 * float(args.target_sphere_radius),
        target_height_m=2.0 * float(args.target_sphere_radius),
        fine_tracker_interval_frames=int(args.fine_tracker_interval_frames),
        fine_tracker_min_score=float(args.fine_tracker_min_score),
        fine_tracker_policy=str(args.fine_tracker_policy),
        adaptive_min_interval_frames=int(args.adaptive_min_interval_frames),
        adaptive_max_interval_frames=int(args.adaptive_max_interval_frames),
        adaptive_bad_frames=int(args.adaptive_bad_frames),
        adaptive_warning_confidence=float(args.adaptive_warning_confidence),
        adaptive_critical_confidence=float(args.adaptive_critical_confidence),
        adaptive_risk_threshold=float(args.adaptive_risk_threshold),
        adaptive_critical_risk_threshold=float(args.adaptive_critical_risk_threshold),
        adaptive_risk_clear_threshold=float(args.adaptive_risk_clear_threshold),
        adaptive_desired_inliers=int(args.adaptive_desired_inliers),
        adaptive_innovation_critical=float(args.adaptive_innovation_critical),
        adaptive_prediction_iou_threshold=float(args.adaptive_prediction_iou_threshold),
        measurement_rejection_innovation_threshold=float(
            args.measurement_rejection_innovation_threshold
        ),
        measurement_rejection_iou_threshold=float(
            args.measurement_rejection_iou_threshold
        ),
        adaptive_failure_backoff_initial_frames=int(args.adaptive_failure_backoff_initial_frames),
        adaptive_failure_backoff_max_frames=int(args.adaptive_failure_backoff_max_frames),
        fine_max_center_jump_box_diagonals=float(args.fine_max_center_jump_box_diagonals),
        fine_discontinuity_center_jump_box_diagonals=float(
            args.fine_discontinuity_center_jump_box_diagonals
        ),
        fine_max_scale_change=float(args.fine_max_scale_change),
    )
    return MultiTargetVisionTracker(intrinsics, config=config, fine_tracker=fine_tracker)


def _estimate_quality(estimate: TargetEstimate, config: MultiTargetConfig | None) -> np.ndarray:
    """Normalize tracker diagnostics to the seven TAQ quality features."""
    fb_error = 0.0 if estimate.median_fb_error_px is None else float(estimate.median_fb_error_px)
    bbox_innovation = float(estimate.bbox_innovation)
    bbox_iou = float(estimate.bbox_iou_with_prediction)
    if estimate.fine_applied:
        x1, y1, x2, y2 = estimate.bbox_xyxy
        bbox_diagonal = max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)
        fine_center_ratio = float(estimate.fine_correction_center_px) / bbox_diagonal
        # On an OSTrack frame these slots must describe the correction that
        # produced the final box, not the preceding LK measurement.
        bbox_innovation = max(bbox_innovation, fine_center_ratio)
        bbox_iou = min(bbox_iou, float(estimate.fine_correction_iou))
    fb_scale = float(config.fb_max_error_px) if config is not None else 1.0
    return np.asarray(
        [
            np.clip(estimate.confidence, 0.0, 1.0),
            np.clip(estimate.motion_confidence, 0.0, 1.0),
            np.clip(estimate.point_retention, 0.0, 1.0),
            np.clip(estimate.ransac_inlier_ratio, 0.0, 1.0),
            np.clip(fb_error / max(fb_scale, 1e-6), 0.0, 1.0),
            np.clip(bbox_innovation, 0.0, 1.0),
            np.clip(bbox_iou, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def _source_id(source: str) -> int:
    return int(SOURCE_NAME_TO_ID.get(str(source).lower(), TrackerSource.INVALID))


def _pack_and_save_taq_samples(
    samples: list[_TAQSample],
    *,
    output_dir: Path,
    env_id: int,
    episode_id: int,
    chunk_size: int,
    metadata: dict[str, Any],
) -> int:
    """Save one episode to independently splittable TAQ v2 chunk files."""
    if not samples:
        return 0
    episode_dir = output_dir / "episodes" / f"env_{env_id:03d}" / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for chunk_index, start in enumerate(range(0, len(samples), chunk_size)):
        chunk = samples[start : start + chunk_size]
        max_steps = max(int(sample.features.shape[0]) for sample in chunk)
        feature_batch = torch.zeros((len(chunk), max_steps, 24), dtype=torch.float32)
        source_batch = torch.full((len(chunk), max_steps), int(TrackerSource.INVALID), dtype=torch.long)
        valid_batch = torch.zeros((len(chunk), max_steps), dtype=torch.bool)
        label_batch = torch.zeros((len(chunk), max_steps, 3), dtype=torch.float32)
        target_mask_batch = torch.zeros((len(chunk), max_steps), dtype=torch.bool)
        window_start_batch = torch.zeros(len(chunk), dtype=torch.float64)
        window_end_batch = torch.zeros(len(chunk), dtype=torch.float64)
        for row, sample in enumerate(chunk):
            length = int(sample.features.shape[0])
            feature_batch[row, :length] = sample.features
            source_batch[row, :length] = sample.source_ids
            valid_batch[row, :length] = True
            label_batch[row, :length] = sample.target_relative_velocity_base_mps
            elapsed = torch.cumsum(sample.features[:, 12].to(torch.float64), dim=0)
            target_mask_batch[row, :length] = (
                (elapsed >= float(metadata["supervision_burn_in_s"]))
                & torch.isfinite(sample.target_relative_velocity_base_mps).all(dim=-1)
            )
            window_start_batch[row] = sample.window_start_s
            window_end_batch[row] = sample.window_end_s
        chunk_metadata = {
            **metadata,
            "env_id": int(env_id),
            "episode_id": int(episode_id),
            "chunk_index": int(chunk_index),
            "sample_count": len(chunk),
        }
        save_taq_velocity_chunk(
            episode_dir / f"taq_velocity_chunk_{chunk_index:04d}.pt",
            sequence_features=feature_batch,
            sequence_source_ids=source_batch,
            sequence_valid_mask=valid_batch,
            target_velocity_base=label_batch,
            target_valid_mask=target_mask_batch,
            sample_window_start_s=window_start_batch,
            sample_window_end_s=window_end_batch,
            metadata=chunk_metadata,
        )
        saved += len(chunk)
    return saved


def _pack_and_save_taq_stream(
    frames: list[_TAQFrame],
    *,
    output_dir: Path,
    env_id: int,
    episode_id: int,
    metadata: dict[str, Any],
) -> int:
    """Save one complete chronological episode stream outside training chunks."""
    if not frames:
        return 0
    features, _ = build_time_aware_features(
        timestamps_s=[item.timestamp_s for item in frames],
        position_base_m=[item.position_base_m for item in frames],
        robot_linear_velocity_base_mps=[item.robot_linear_velocity_base_mps for item in frames],
        robot_angular_velocity_base_rps=[item.robot_angular_velocity_base_rps for item in frames],
        source_ids=[item.source_id for item in frames],
        quality=[item.quality for item in frames],
        range_valid_mask=[item.range_valid for item in frames],
    )
    timestamps = torch.tensor([item.timestamp_s for item in frames], dtype=torch.float64)
    sequence_valid = torch.ones(len(frames), dtype=torch.bool)
    labels = torch.stack(
        [torch.as_tensor(item.target_relative_velocity_base_mps, dtype=torch.float32) for item in frames]
    )
    elapsed = timestamps - timestamps[0]
    target_valid = (
        (elapsed >= float(metadata["supervision_burn_in_s"]))
        & torch.isfinite(labels).all(dim=-1)
    )
    reset_mask = torch.tensor([item.state_reset for item in frames], dtype=torch.bool)
    reset_mask[0] = True
    episode_dir = output_dir / "episodes" / f"env_{env_id:03d}" / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": TAQ_SCHEMA_VERSION,
            "stream_format": TAQ_STREAM_FORMAT,
            "label_semantics": TAQ_LABEL_SEMANTICS,
            "timestamps_s": timestamps,
            "sequence_features": features,
            "sequence_source_ids": torch.tensor([item.source_id for item in frames], dtype=torch.long),
            "sequence_valid_mask": sequence_valid,
            "target_velocity_base": labels,
            "target_valid_mask": target_valid,
            "state_reset_mask": reset_mask,
            "metadata": dict(metadata),
        },
        episode_dir / "taq_velocity_episode.taqstream",
    )
    episode_uid = str(metadata.get("episode_uid", episode_dir.as_posix()))
    audit_lines = []
    for frame_index, item in enumerate(frames):
        audit_row = {
            "episode_uid": episode_uid,
            "frame_index": frame_index,
            **item.audit,
            "state_reset": bool(item.state_reset),
            "state_reset_reason": (
                "tracking_unavailable" if item.source_id == int(TrackerSource.INVALID)
                else "initial_observation" if item.source_id == int(TrackerSource.INITIAL)
                else "stream_start" if frame_index == 0
                else "measurement_discontinuity" if item.state_reset
                else None
            ),
        }
        audit_lines.append(
            json.dumps(_json_safe(audit_row), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        )
    (episode_dir / "velocity_error_audit.jsonl").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )
    if frames[0].sensor_packet is not None:
        if any(item.sensor_packet is None for item in frames):
            raise RuntimeError("Incomplete sensor packet stream")
        torch.save(dict(version=TIME_CONTRACT_VERSION, episode_uid=episode_uid,
                        packets=[item.sensor_packet for item in frames]),
                   episode_dir / "timed_sensor_packets.pt")
    return len(frames)


def _write_paired_state_audit(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    env_id: int,
    episode_id: int,
    metadata: dict[str, Any],
) -> int:
    """Persist every paired physics frame independently of tracker availability."""
    if not rows:
        return 0
    episode_dir = output_dir / "episodes" / f"env_{env_id:03d}" / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_uid = str(metadata.get("episode_uid", episode_dir.as_posix()))
    trajectory_uid = str(metadata.get("trajectory_uid", ""))
    lines = []
    for row in rows:
        payload = {
            "episode_uid": episode_uid,
            "trajectory_uid": trajectory_uid,
            "condition": str(metadata.get("condition", "")),
            **row,
        }
        lines.append(
            json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    (episode_dir / "paired_state_audit.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return len(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(frame_observer=None, collection_finished=None) -> None:
    """Collect schema-v2 TAQ samples for the SS/SD velocity experiments."""
    args = _taq_args()
    _validate_taq_args(args)
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    # AppLauncher selects the active rendering and PhysX GPUs before the
    # environment is built, so setting only env_cfg.sim.device below is too
    # late to move an out-of-memory startup away from cuda:0.
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=bool(args.enable_cameras),
        device=str(args.device),
    )
    sim_app = app_launcher.app
    env = None
    completed = False
    try:
        import gymnasium as gym
        from isaaclab.utils.math import quat_apply
        from turret_aim_env_collect import TurretAimCollectEnv, TurretAimCollectEnvCfg

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        env_cfg = TurretAimCollectEnvCfg()
        env_cfg.seed = int(args.seed)
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.sim.device = str(args.device)
        env_cfg.sim.use_fabric = False
        env_cfg.vision_history_length = 1
        env_cfg.debug_vis = not args.headless
        env_cfg.debug_diagnostics = False
        # Paired validation uses the same deterministic command in every
        # condition; the scenario flag changes target motion only.
        env_cfg.training_stage = 1
        fixed_camera_validation = args.validation_condition in {"A", "B"}
        gt_direction_validation = args.validation_condition in {"A", "C", "P1", "P3"}
        gt_range_validation = args.validation_condition in {"P2", "P3"}
        env_cfg.validation_fixed_camera = fixed_camera_validation
        env_cfg.validation_reference_target_trajectory = args.validation_condition != "none"
        env_cfg.validation_trajectory_seed = int(args.trajectory_seed)
        if args.validation_condition != "none":
            env_cfg.sample_target_in_camera_sector = True
            env_cfg.target_yaw_range = tuple(env_cfg.target_cam_yaw_range)
        # Run identical robot physics and policy paths in every paired
        # condition.  A/B intervene only on the independent render proxy.
        env_cfg.enable_locomotion_policy = args.locomotion_policy is not None
        env_cfg.locomotion_policy_path = (
            None if args.locomotion_policy is None
            else str(args.locomotion_policy.expanduser().resolve())
        )
        env_cfg.locomotion_lin_vel_x_range = (float(args.locomotion_vx_mps),) * 2
        env_cfg.locomotion_lin_vel_y_range = (float(args.locomotion_vy_mps),) * 2
        env_cfg.locomotion_ang_vel_z_range = (float(args.locomotion_yaw_rate_radps),) * 2
        env_cfg.locomotion_command_resample_s = 1.0
        env_cfg.enable_dynamic_target = args.scenario == "sd"
        # The new SD domain is fixed in world coordinates; live camera motion
        # affects visibility but cannot relocate the target.
        env_cfg.constrain_dynamic_target_to_camera_fov = args.scenario == "sd"
        env_cfg.target_boundary_mode = str(args.target_boundary_mode)
        env_cfg.target_fov_margin_fraction = float(args.target_fov_margin_fraction)
        env_cfg.target_speed_range = (0.0, 0.0) if args.scenario == "ss" else (float(args.target_speed_min), float(args.target_speed_max))
        env_cfg.target_vertical_speed_range = (0.0, 0.0) if args.scenario == "ss" else (float(args.target_vz_min), float(args.target_vz_max))
        env_cfg.target_acc_range = (
            (0.0, 0.0)
            if args.scenario == "ss" or env_cfg.constrain_dynamic_target_to_camera_fov
            else (float(args.target_acc_min), float(args.target_acc_max))
        )
        env_cfg.target_vertical_acc_range = (
            (0.0, 0.0)
            if args.scenario == "ss" or env_cfg.constrain_dynamic_target_to_camera_fov
            else (float(args.target_vz_acc_min), float(args.target_vz_acc_max))
        )
        env_cfg.target_radius_range = (float(args.target_distance_min), float(args.target_distance_max))
        env_cfg.target_height_range = (float(args.target_height_min), float(args.target_height_max))
        env_cfg.target_sphere_radius = float(args.target_sphere_radius)
        env_cfg.camera_focal_length = float(args.camera_focal_length)
        env_cfg.camera_horizontal_aperture = float(args.camera_horizontal_aperture)
        env_cfg.camera_vertical_aperture = float(args.camera_vertical_aperture)
        env_cfg.enable_dynamic_target_raycast_debug = True
        env_cfg.terminate_on_target_out_of_image = (
            args.validation_condition == "none" and args.degradation_profile == "clean"
        )
        if args.episode_warmup_s >= float(env_cfg.episode_length_s):
            raise ValueError("--episode_warmup_s must be shorter than the episode")

        task_id = "Standalone-TurretAim-TAQVelocityCollect-v0"
        if task_id not in gym.registry:
            gym.register(id=task_id, entry_point=TurretAimCollectEnv)
        env = gym.make(task_id, cfg=env_cfg, disable_env_checker=True)
        base_env: Any = env.unwrapped
        collection_warmup = EpisodeCollectionWarmup(
            float(args.episode_warmup_s), float(base_env.step_dt)
        )
        if collection_warmup.steps >= base_env.max_episode_length - 2:
            raise ValueError("Episode collection warmup leaves no recordable physics steps")
        env.reset(seed=int(args.seed))
        if args.camera_prerender_frames > 0:
            print(
                "[taq-collect] warming RTX camera:",
                f"frames={args.camera_prerender_frames}",
                flush=True,
            )
            for _ in range(int(args.camera_prerender_frames)):
                base_env._sync_render_camera_before_read()

        degrader = Stage2Degrader(
            profile=str(args.degradation_profile),
            severity=str(args.severity),
            seed=int(args.degradation_seed),
            num_envs=int(base_env.num_envs),
            nominal_dt_s=float(base_env.step_dt),
            min_timestamp_dt_ratio=float(args.min_timestamp_dt_ratio),
        )
        trackers: list[MultiTargetVisionTracker | None] = [None] * base_env.num_envs
        tracker_sample_times = [None] * base_env.num_envs
        sampling_gates = [PhysicalSamplingGate(args.capture_skip_probability,
                          np.random.SeedSequence([args.seed, env_id, 991]))
                          for env_id in range(base_env.num_envs)]
        histories: list[deque[_TAQFrame]] = [deque() for _ in range(base_env.num_envs)]
        episode_stream_frames: list[list[_TAQFrame]] = [[] for _ in range(base_env.num_envs)]
        paired_state_audit_rows: list[list[dict[str, Any]]] = [
            [] for _ in range(base_env.num_envs)
        ]
        previous_experimental_camera_pose: list[
            tuple[np.ndarray, np.ndarray] | None
        ] = [None] * base_env.num_envs
        stream_reset_pending = [True] * base_env.num_envs
        episode_samples: list[list[_TAQSample]] = [[] for _ in range(base_env.num_envs)]
        episode_ids = [0] * base_env.num_envs
        frame_counts = [0] * base_env.num_envs
        episode_window_counts = [0] * base_env.num_envs
        last_window_end_s: list[float | None] = [None] * base_env.num_envs
        last_bboxes: list[tuple[float, float, float, float] | None] = [None] * base_env.num_envs
        previous_rgb_images: list[np.ndarray | None] = [None] * base_env.num_envs
        raycaster_audits = [_RaycasterDistanceAudit() for _ in range(base_env.num_envs)]
        warmup_discarded_frames = [0] * base_env.num_envs
        finished_after_target = [False] * base_env.num_envs
        for env_id in range(base_env.num_envs):
            degrader.reset_episode(env_id, episode_ids[env_id])
        total_samples = 0
        measurement_discontinuities = 0
        metadata = {
            "schema_version": TAQ_SCHEMA_VERSION,
            "label_semantics": TAQ_LABEL_SEMANTICS,
            "collection_run_id": str(args.collection_run_id),
            "created_at": datetime.now().isoformat(),
            "scenario": args.scenario,
            "split": str(args.split),
            "environment_seed": int(args.seed),
            "trajectory_seed": int(args.trajectory_seed),
            "condition": str(args.validation_condition),
            "pair_id": "" if args.pair_id is None else str(args.pair_id),
            "validation_fixed_camera": bool(fixed_camera_validation),
            "validation_camera_mode": (
                "fixed_world_proxy" if fixed_camera_validation else "torso_mounted_proxy"
            ),
            "validation_gt_direction": bool(gt_direction_validation),
            "validation_gt_range": bool(gt_range_validation),
            "validation_reference_target_trajectory": args.validation_condition != "none",
            "validation_trajectory_seed": int(args.trajectory_seed),
            "validation_target_sampling": (
                "fixed_reference_camera_frustum_with_safety_margin"
                if env_cfg.constrain_dynamic_target_to_camera_fov and args.target_boundary_mode == "fixed_reference_toi"
                else "common_fixed_and_mounted_camera_frusta"
                if env_cfg.constrain_dynamic_target_to_camera_fov
                and args.validation_condition != "none"
                else "live_mounted_camera_frustum"
                if env_cfg.constrain_dynamic_target_to_camera_fov
                else "nominal_reference_camera_sector"
                if args.validation_condition != "none"
                else "environment_default"
            ),
            "fixed_camera_method": (
                "cached_nominal_world_pose_proxy"
                if fixed_camera_validation else "none"
            ),
            "fixed_camera_pose_semantics": (
                "cached once in world coordinates; independent of live torso motion"
                if fixed_camera_validation else "torso-mounted"
            ),
            "target_speed_range_mps": [
                float(args.target_speed_min), float(args.target_speed_max)
            ],
            "target_speed_semantics": (
                "uniform 3-D speed magnitude; isotropic random direction"
                if args.scenario == "sd" else "static target"
            ),
            "target_vertical_speed_limit_applied": bool(
                args.scenario == "sd"
                and not env_cfg.constrain_dynamic_target_to_camera_fov
            ),
            "target_height_bounds_applied": not bool(
                env_cfg.constrain_dynamic_target_to_camera_fov
            ),
            "target_sphere_radius_m": float(env_cfg.target_sphere_radius),
            "dynamic_target_ground_constraint": (
                "world_z_0_plane_with_vertical_velocity_reflection"
                if env_cfg.constrain_dynamic_target_to_camera_fov else "legacy_height_bounds"
            ),
            "dynamic_target_ground_min_center_z_m": (
                float(env_cfg.target_sphere_radius) + float(env_cfg.target_ground_clearance_m)
                if env_cfg.constrain_dynamic_target_to_camera_fov else None
            ),
            "dynamic_target_camera_fov_constraint": (
                "fixed reference frustum; live camera does not modify target motion"
                if env_cfg.constrain_dynamic_target_to_camera_fov and args.target_boundary_mode == "fixed_reference_toi"
                else "intersection of fixed reference and live mounted camera frusta"
                if env_cfg.constrain_dynamic_target_to_camera_fov
                and args.validation_condition != "none"
                else "live mounted camera frustum"
                if env_cfg.constrain_dynamic_target_to_camera_fov
                else "none"
            ),
            "camera_prerender_frames": int(args.camera_prerender_frames),
            "target_boundary_mode": str(args.target_boundary_mode),
            "target_fov_margin_fraction_per_edge": float(args.target_fov_margin_fraction),
            "target_boundary_position_projection": (
                "reset_initialization_only" if args.target_boundary_mode == "fixed_reference_toi"
                else "every_step_legacy_projection"
            ),
            "target_velocity_label_time_semantics": "instantaneous_end_of_physics_step",
            "episode_warmup_s": float(args.episode_warmup_s),
            "episode_warmup_steps": collection_warmup.steps,
            "episode_warmup_effective_s": collection_warmup.effective_s,
            "episode_warmup_semantics": "physics_and_tracking_run_without_recording",
            "environment_episode_length_s": float(env_cfg.episode_length_s),
            "robot_root_intervention": "none",
            "coordinate_frame": "yaw_only_base",
            "sensor_time_contract": str(args.sensor_time_contract),
            "capture_skip_probability": float(args.capture_skip_probability),
            "sensor_delivery_semantics": (
                "zero-latency acquisition trace; delays/dropouts applied by paired packet replay"
                if args.sensor_time_contract == TIME_CONTRACT_VERSION
                else "synchronous simulation capture timestamps; no injected delay or dropout"
                if args.degradation_profile == "clean"
                else "legacy synthetic timestamps and value-only delay queues"
            ),
            "angular_velocity_z_semantics": TAQ_YAW_RATE_SEMANTICS,
            "target_id": 0,
            "range_source": (
                "ground_truth_validation" if gt_range_validation
                else "raycaster_using_tracking_direction"
            ),
            "raycaster_gt_audit": "episode_metadata_only_not_a_model_input",
            "initialization_source": "simulation_bbox_detector_proxy",
            "measurement_gt_free_after_initialization": not (
                gt_direction_validation or gt_range_validation
            ),
            "gt_visibility_gate_after_initialization": bool(
                args.use_gt_visibility_gate_after_initialization
            ),
            "keep_measurement_discontinuities": bool(args.keep_measurement_discontinuities),
            "degradation": degrader.config_metadata(),
            "window_s": float(args.window_s),
            "min_window_gap_s": float(args.min_window_gap_s),
            "supervision_burn_in_s": float(args.supervision_burn_in_s),
            "min_timestamp_dt_ratio": float(args.min_timestamp_dt_ratio),
            "sample_stride": int(args.sample_stride),
            "max_sequence_steps": int(args.max_sequence_steps),
            "window_sampling": "continuous_non_overlapping_capped_per_episode",
            "max_windows_per_episode": int(args.max_windows_per_episode),
            "fine_tracker_policy": str(args.fine_tracker_policy),
            "fine_tracking_enabled": not bool(args.disable_fine_tracking) and args.ostrack_repo is not None,
            "fine_max_center_jump_box_diagonals": float(args.fine_max_center_jump_box_diagonals),
            "fine_discontinuity_center_jump_box_diagonals": float(
                args.fine_discontinuity_center_jump_box_diagonals
            ),
            "fine_max_scale_change": float(args.fine_max_scale_change),
            "measurement_rejection_innovation_threshold": float(
                args.measurement_rejection_innovation_threshold
            ),
            "measurement_rejection_iou_threshold": float(
                args.measurement_rejection_iou_threshold
            ),
            "locomotion_policy": (
                None if args.locomotion_policy is None
                else str(args.locomotion_policy.expanduser().resolve())
            ),
            "locomotion_command": [
                float(args.locomotion_vx_mps), float(args.locomotion_vy_mps),
                float(args.locomotion_yaw_rate_radps),
            ],
            "max_static_base_speed_mps": float(args.max_static_base_speed_mps),
            "max_static_base_angular_speed_radps": float(args.max_static_base_angular_speed_radps),
            "static_base_filter_applied": args.validation_condition == "none",
            "camera_specific_out_of_view_reset": False
            if args.validation_condition != "none"
            else bool(env_cfg.terminate_on_target_out_of_image),
        }

        def flush_episode(env_id: int, *, complete: bool = False) -> int:
            trajectory_uid = (
                f"{args.pair_id}:env_{env_id:03d}:episode_{episode_ids[env_id]:06d}"
                if args.pair_id else ""
            )
            episode_metadata = {
                **metadata,
                "env_id": int(env_id),
                "episode_id": int(episode_ids[env_id]),
                "trajectory_uid": trajectory_uid,
                "episode_uid": (
                    f"{args.collection_run_id}:{args.split}:{args.scenario}:"
                    f"env_{env_id:03d}:episode_{episode_ids[env_id]:06d}"
                ),
                "degradation_episode": degrader.episode_metadata(env_id),
                "raycaster_vs_gt_audit": raycaster_audits[env_id].metadata(),
                "episode_warmup_discarded_frames": warmup_discarded_frames[env_id],
                "episode_complete": complete,
                "episode_first_recorded_elapsed_s": (
                    episode_stream_frames[env_id][0].audit["episode_elapsed_s"]
                    if episode_stream_frames[env_id] else None
                ),
            }
            if args.save_episode_stream:
                _pack_and_save_taq_stream(
                    episode_stream_frames[env_id],
                    output_dir=args.out_dir,
                    env_id=env_id,
                    episode_id=episode_ids[env_id],
                    metadata=episode_metadata,
                )
            _write_paired_state_audit(
                paired_state_audit_rows[env_id],
                output_dir=args.out_dir,
                env_id=env_id,
                episode_id=episode_ids[env_id],
                metadata=episode_metadata,
            )
            saved = _pack_and_save_taq_samples(
                episode_samples[env_id],
                output_dir=args.out_dir,
                env_id=env_id,
                episode_id=episode_ids[env_id],
                chunk_size=int(args.chunk_size),
                metadata=episode_metadata,
            )
            episode_samples[env_id].clear()
            episode_stream_frames[env_id].clear()
            paired_state_audit_rows[env_id].clear()
            return saved

        print(
            f"[taq-collect] scenario={args.scenario} split={args.split} "
            f"condition={args.validation_condition} pair_id={args.pair_id} "
            f"profile={args.degradation_profile} severity={args.severity} "
            f"environment_seed={args.seed} degradation_seed={args.degradation_seed} "
            f"episode_warmup_s={collection_warmup.effective_s} "
            f"min_timestamp_dt_ratio={args.min_timestamp_dt_ratio} output={args.out_dir}"
        )

        reached_target = False
        for step in range(int(args.num_steps)):
            action = torch.zeros((base_env.num_envs, 3), device=base_env.device, dtype=torch.float32)
            action[:, 2] = -1.0
            _, _, terminated, truncated, _ = env.step(action)
            done = (torch.as_tensor(terminated, device=base_env.device, dtype=torch.bool) | torch.as_tensor(truncated, device=base_env.device, dtype=torch.bool))
            base_env.refresh_vision_supervision()
            batch = base_env.get_vision_supervision_batch()
            images = batch["images"]
            current_images = images[:, -1] if images.ndim == 5 else images
            rgb_images = _rgb_chw_batch_to_uint8(current_images)
            valid_visual = batch["valid_mask"].to(
                device=base_env.device, dtype=torch.bool
            )
            if (
                args.debug_rgb_dir is not None
                and (
                    args.debug_rgb_steps is None
                    or step in args.debug_rgb_steps
                )
            ):
                from PIL import Image

                debug_rgb_dir = args.debug_rgb_dir.expanduser().resolve()
                debug_rgb_dir.mkdir(parents=True, exist_ok=True)
                for env_id, rgb_image in enumerate(rgb_images):
                    Image.fromarray(rgb_image).save(
                        debug_rgb_dir
                        / f"step_{step:06d}_env_{env_id:03d}.png"
                    )
            background_flow_px: list[float | None] = [None] * base_env.num_envs
            for env_id in range(base_env.num_envs):
                if bool(done[env_id]) or not bool(valid_visual[env_id]):
                    previous_rgb_images[env_id] = None
                    previous_experimental_camera_pose[env_id] = None
                    continue
                if args.validation_condition != "none":
                    background_flow_px[env_id] = _background_global_flow_px(
                        previous_rgb_images[env_id],
                        rgb_images[env_id],
                        last_bboxes[env_id],
                    )
                    previous_rgb_images[env_id] = rgb_images[env_id].copy()
            gt_direction = batch["direction"].to(device=base_env.device, dtype=torch.float32)
            gt_distance = batch["distance"].to(device=base_env.device, dtype=torch.float32).view(-1)

            directions = np.zeros((base_env.num_envs, 2), dtype=np.float32)
            estimates: list[TargetEstimate | None] = [None] * base_env.num_envs
            timestamps_s = np.zeros(base_env.num_envs, dtype=np.float64)
            sampled_images = [True] * base_env.num_envs
            tracker_input_dts = [None] * base_env.num_envs
            # Images/GT are read AFTER env.step, so clean capture begins at dt.
            use_capture_clock = args.degradation_profile == "clean" or args.sensor_time_contract == TIME_CONTRACT_VERSION
            nominal_now_s = float(step + use_capture_clock) * float(base_env.step_dt)
            recording_ready = [False] * base_env.num_envs
            for env_id in range(base_env.num_envs):
                if bool(done[env_id]):
                    total_samples += flush_episode(env_id, complete=True)
                    if reached_target and args.save_episode_stream:
                        finished_after_target[env_id] = True
                    histories[env_id].clear()
                    episode_ids[env_id] += 1
                    frame_counts[env_id] = 0
                    episode_window_counts[env_id] = 0
                    last_window_end_s[env_id] = None
                    stream_reset_pending[env_id] = True
                    last_bboxes[env_id] = None
                    previous_experimental_camera_pose[env_id] = None
                    raycaster_audits[env_id] = _RaycasterDistanceAudit()
                    warmup_discarded_frames[env_id] = 0
                    if trackers[env_id] is not None:
                        trackers[env_id].reset()
                    degrader.reset_episode(env_id, episode_ids[env_id])
                    tracker_sample_times[env_id] = None
                    continue
                if finished_after_target[env_id]:
                    continue
                recording_ready[env_id] = collection_warmup.ready(
                    int(base_env.episode_length_buf[env_id].item())
                )
                if not recording_ready[env_id]:
                    warmup_discarded_frames[env_id] += 1
                timestamp_s, actual_dt_s = degrader.next_timestamp(env_id, nominal_now_s)
                timestamps_s[env_id] = timestamp_s
                if args.sensor_time_contract == TIME_CONTRACT_VERSION:
                    sampled_images[env_id] = sampling_gates[env_id].take(step)
                    if not sampled_images[env_id]:
                        continue
                    if tracker_sample_times[env_id] is not None:
                        actual_dt_s = timestamp_s-tracker_sample_times[env_id]
                image_height, image_width = rgb_images[env_id].shape[:2]
                intrinsics = _camera_intrinsics_from_image(args, image_width, image_height)
                if gt_direction_validation:
                    if not bool(valid_visual[env_id]):
                        continue
                    gt_box = _bootstrap_box_from_gt(
                        float(gt_direction[env_id, 0].item()),
                        float(gt_direction[env_id, 1].item()),
                        float(gt_distance[env_id].item()),
                        intrinsics,
                        float(args.target_sphere_radius),
                        float(args.min_bootstrap_box_px),
                    )
                    if gt_box is None:
                        continue
                    estimate = _gt_validation_estimate(
                        yaw_rad=float(gt_direction[env_id, 0].item()),
                        pitch_rad=float(gt_direction[env_id, 1].item()),
                        distance_m=float(gt_distance[env_id].item()),
                        bbox_xyxy=gt_box,
                        initial=bool(stream_reset_pending[env_id]),
                    )
                    estimates[env_id] = estimate
                    last_bboxes[env_id] = gt_box
                    directions[env_id] = gt_direction[env_id].detach().cpu().numpy()
                    degrader.record_tracker_source(env_id, estimate.source)
                    continue
                tracker = trackers[env_id]
                if tracker is None or not tracker.initialized:
                    # GT is used only to emulate the detector proposal needed
                    # to bootstrap a tracker.  It is never fed to TAQ features.
                    if not bool(valid_visual[env_id]):
                        continue
                    box = _bootstrap_box_from_gt(
                        float(gt_direction[env_id, 0].item()), float(gt_direction[env_id, 1].item()),
                        float(gt_distance[env_id].item()), intrinsics, float(args.target_sphere_radius), float(args.min_bootstrap_box_px),
                    )
                    if box is None:
                        continue
                    box = degrader.perturb_bootstrap_box(
                        env_id,
                        box,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    if box is None:
                        continue
                    tracker_image = degrader.degrade_image(
                        env_id,
                        rgb_images[env_id],
                        target_bbox_xyxy=box,
                        allow_dropout=False,
                    )
                    tracker = _build_taq_tracker(args, intrinsics)
                    result = tracker.initialize(tracker_image, {0: box}, timestamp_s=timestamp_s)
                    trackers[env_id] = tracker
                else:
                    if args.use_gt_visibility_gate_after_initialization and not bool(valid_visual[env_id]):
                        continue
                    tracker_image = degrader.degrade_image(
                        env_id,
                        rgb_images[env_id],
                        target_bbox_xyxy=last_bboxes[env_id],
                        allow_dropout=True,
                    )
                    result = tracker.update(
                        tracker_image,
                        dt_s=actual_dt_s,
                        timestamp_s=timestamp_s,
                    )
                tracker_sample_times[env_id] = timestamp_s
                tracker_input_dts[env_id] = actual_dt_s
                estimate = result.selected_target
                if estimate is None or estimate.status == "lost":
                    degrader.record_tracker_source(env_id, "lost")
                    tracker.reset()
                    histories[env_id].clear()
                    frame_counts[env_id] = 0
                    last_bboxes[env_id] = None
                    stream_reset_pending[env_id] = True
                    continue
                estimates[env_id] = estimate
                last_bboxes[env_id] = tuple(float(value) for value in estimate.bbox_xyxy)
                degrader.record_tracker_source(env_id, estimate.source)
                if bool(getattr(estimate, "measurement_rejected", False)):
                    degrader.record_tracker_measurement_rejection(env_id)
                directions[env_id] = degrader.degrade_direction(
                    env_id,
                    estimate.yaw_rad,
                    estimate.pitch_rad,
                )

            base_env.set_dynamic_target_raycast_angles(
                torch.as_tensor(directions[:, 0], device=base_env.device),
                torch.as_tensor(directions[:, 1], device=base_env.device),
            )
            base_env.run_dynamic_target_raycast_for_current_frame()
            measured = base_env.get_vision_supervision_batch()
            raw_raycast_range_m = measured["lidar_distance"].to(
                device=base_env.device, dtype=torch.float32
            ).view(-1)
            raw_raycast_valid = measured["lidar_valid_mask"].to(
                device=base_env.device, dtype=torch.bool
            ).view(-1)
            gt_range_for_audit_m = measured["distance"].to(
                device=base_env.device, dtype=torch.float32
            ).view(-1)
            for env_id, estimate in enumerate(estimates):
                if estimate is None or not recording_ready[env_id]:
                    continue
                raycaster_audits[env_id].record(
                    float(raw_raycast_range_m[env_id].item()),
                    bool(raw_raycast_valid[env_id].item()),
                    float(gt_range_for_audit_m[env_id].item()),
                )
            if gt_range_validation:
                range_m = gt_range_for_audit_m.clone()
                range_valid = valid_visual.clone()
            else:
                range_m = raw_raycast_range_m
                range_valid = raw_raycast_valid
                degraded_ranges = np.zeros(base_env.num_envs, dtype=np.float32)
                degraded_range_valid = np.zeros(base_env.num_envs, dtype=np.bool_)
                for env_id in range(base_env.num_envs):
                    if bool(done[env_id]):
                        degraded_ranges[env_id] = np.nan
                        degraded_range_valid[env_id] = False
                        continue
                    degraded_range, degraded_valid = degrader.degrade_range(
                        env_id,
                        float(range_m[env_id].item()),
                        bool(range_valid[env_id].item()),
                    )
                    degraded_ranges[env_id] = degraded_range
                    degraded_range_valid[env_id] = degraded_valid
                range_m = torch.as_tensor(
                    degraded_ranges,
                    device=base_env.device,
                    dtype=torch.float32,
                )
                range_valid = torch.as_tensor(
                    degraded_range_valid,
                    device=base_env.device,
                    dtype=torch.bool,
                )

            yaw_t = torch.as_tensor(directions[:, 0], device=base_env.device)
            pitch_t = torch.as_tensor(directions[:, 1], device=base_env.device)
            safe_range = torch.where(range_valid, range_m, torch.zeros_like(range_m))
            position_cam = torch.stack((safe_range * torch.sin(yaw_t) * torch.cos(pitch_t), -safe_range * torch.sin(pitch_t), safe_range * torch.cos(yaw_t) * torch.cos(pitch_t)), dim=-1)
            cam_pos_w, cam_quat_w = base_env._get_runtime_camera_pose_w()
            mounted_cam_pos_w, mounted_cam_quat_w = base_env.get_mounted_camera_pose_w()
            reference_cam_pos_w, reference_cam_quat_w = (
                base_env._get_validation_reference_camera_pose_w()
            )
            target_in_experimental_camera = base_env._compute_target_projected_in_image_mask(
                cam_pos_w, cam_quat_w
            )
            target_in_mounted_camera = base_env._compute_target_projected_in_image_mask(
                mounted_cam_pos_w, mounted_cam_quat_w
            )
            root_pos_w = base_env.turret.data.root_pos_w
            root_quat_w = base_env.turret.data.root_quat_w
            position_w = cam_pos_w + quat_apply(cam_quat_w, position_cam)
            position_base = base_env._rotate_world_to_yaw_only_local(root_quat_w, position_w - root_pos_w)
            position_bias = torch.as_tensor(
                np.stack(
                    [degrader.position_bias_base_m(env_id) for env_id in range(base_env.num_envs)]
                ),
                device=base_env.device,
                dtype=torch.float32,
            )
            position_base = position_base + position_bias
            lin_world = quat_apply(root_quat_w, base_env.turret.data.root_lin_vel_b)
            ang_world = quat_apply(root_quat_w, base_env.turret.data.root_ang_vel_b)
            robot_linear_sim_raw = base_env._rotate_world_to_yaw_only_local(root_quat_w, lin_world)
            robot_angular_sim_raw = base_env._rotate_world_to_yaw_only_local(
                root_quat_w, ang_world
            ).clone()
            # The yaw-only coordinate frame follows extracted heading, whose
            # exact derivative differs from world angular-z under roll/pitch.
            # Store that heading rate in angular-z so the v2 physical-velocity
            # baseline can analytically undo the rotating-frame position term.
            robot_angular_sim_raw[:, 2] = base_env._get_yaw_rate_world(root_quat_w, ang_world)
            robot_linear_raw = robot_linear_sim_raw.clone()
            robot_angular_raw = robot_angular_sim_raw.clone()
            velocity_label = measured["target_relative_velocity_base_mps"].to(
                device=base_env.device, dtype=torch.float32
            )
            position_gt_base = measured["pos_base_gt"].to(
                device=base_env.device, dtype=torch.float32
            )
            if args.validation_condition != "none":
                # Paired validation keeps full commanded robot motion;
                # invalid/fallen states already abort in the environment.
                base_eligible = torch.ones(
                    base_env.num_envs, device=base_env.device, dtype=torch.bool
                )
            else:
                base_eligible = (
                    torch.linalg.vector_norm(robot_linear_raw, dim=-1)
                    <= float(args.max_static_base_speed_mps)
                ) & (
                    torch.linalg.vector_norm(robot_angular_raw, dim=-1)
                    <= float(args.max_static_base_angular_speed_radps)
                )

            current_pair_audit: list[dict[str, Any] | None] = [
                None for _ in range(base_env.num_envs)
            ]
            for env_id in range(base_env.num_envs):
                if bool(done[env_id]) or finished_after_target[env_id]:
                    continue
                camera_linear_velocity, camera_angular_velocity = _camera_velocity_from_pose(
                    previous_experimental_camera_pose[env_id],
                    cam_pos_w[env_id],
                    cam_quat_w[env_id],
                    float(base_env.step_dt),
                )
                current_position_np = (
                    cam_pos_w[env_id].detach().cpu().numpy().astype(np.float64)
                )
                current_quaternion_np = (
                    cam_quat_w[env_id].detach().cpu().numpy().astype(np.float64)
                )
                previous_experimental_camera_pose[env_id] = (
                    current_position_np,
                    current_quaternion_np,
                )
                estimate = estimates[env_id]
                state_audit = {
                    "physics_step_index": int(step),
                    "physics_timestamp_s": float(nominal_now_s),
                    "episode_elapsed_s": float(base_env.episode_length_buf[env_id].item()) * float(base_env.step_dt),
                    "target_velocity_world_mps": base_env._target_vel_w[env_id].detach().cpu().tolist(),
                    "target_boundary_collision_count": (
                        int(base_env._target_boundary_collision_count[env_id].item())
                        if hasattr(base_env, "_target_boundary_collision_count") else None
                    ),
                    "camera_mode": (
                        "fixed_world_proxy"
                        if fixed_camera_validation
                        else "torso_mounted_proxy"
                    ),
                    "target_in_experimental_camera": bool(
                        target_in_experimental_camera[env_id].item()
                    ),
                    "target_in_mounted_camera": bool(
                        target_in_mounted_camera[env_id].item()
                    ),
                    "tracker_available": estimate is not None,
                    "measurement_valid": bool(
                        estimate is not None
                        and _source_id(estimate.source) in (
                            int(TrackerSource.INITIAL), int(TrackerSource.FLOW),
                            int(TrackerSource.OSTRACK),
                        )
                        and bool(range_valid[env_id].item())
                        and bool(torch.isfinite(range_m[env_id]).item())
                    ),
                    "prediction_available": None,  # filled by offline checkpoint replay
                    "observation_invalid_reason": (
                        "out_of_view" if not bool(target_in_experimental_camera[env_id].item())
                        else "tracker_lost" if estimate is None
                        else "range_invalid" if not bool(range_valid[env_id].item())
                        else None
                    ),
                    "tracker_source": "lost" if estimate is None else str(estimate.source),
                    "target_position_world_m": base_env._target_pos_w[
                        env_id
                    ].detach().cpu().tolist(),
                    "camera_position_world_m": cam_pos_w[
                        env_id
                    ].detach().cpu().tolist(),
                    "camera_quaternion_world": cam_quat_w[
                        env_id
                    ].detach().cpu().tolist(),
                    "mounted_camera_position_world_m": mounted_cam_pos_w[
                        env_id
                    ].detach().cpu().tolist(),
                    "mounted_camera_quaternion_world": mounted_cam_quat_w[
                        env_id
                    ].detach().cpu().tolist(),
                    "camera_linear_velocity_world_mps": camera_linear_velocity.tolist(),
                    "camera_angular_velocity_world_rps": camera_angular_velocity.tolist(),
                    "camera_position_drift_m": float(
                        torch.linalg.vector_norm(
                            cam_pos_w[env_id] - reference_cam_pos_w[env_id]
                        ).item()
                    ),
                    "camera_orientation_drift_rad": _quaternion_distance_rad(
                        cam_quat_w[env_id], reference_cam_quat_w[env_id]
                    ),
                    "experimental_to_mounted_camera_position_m": float(
                        torch.linalg.vector_norm(
                            cam_pos_w[env_id] - mounted_cam_pos_w[env_id]
                        ).item()
                    ),
                    "experimental_to_mounted_camera_orientation_rad": (
                        _quaternion_distance_rad(
                            cam_quat_w[env_id], mounted_cam_quat_w[env_id]
                        )
                    ),
                    "base_position_world_m": root_pos_w[
                        env_id
                    ].detach().cpu().tolist(),
                    "base_quaternion_world": root_quat_w[
                        env_id
                    ].detach().cpu().tolist(),
                    "base_roll_pitch_yaw_rad": _quaternion_wxyz_to_rpy(
                        root_quat_w[env_id]
                    ),
                    "robot_linear_velocity_base_mps": robot_linear_raw[
                        env_id
                    ].detach().cpu().tolist(),
                    "robot_angular_velocity_base_rps": robot_angular_raw[
                        env_id
                    ].detach().cpu().tolist(),
                    "background_global_flow_px": background_flow_px[env_id],
                }
                current_pair_audit[env_id] = state_audit
                if recording_ready[env_id]:
                    paired_state_audit_rows[env_id].append(state_audit)
            degraded_linear: list[np.ndarray] = []
            degraded_angular: list[np.ndarray] = []
            for env_id in range(base_env.num_envs):
                if bool(done[env_id]):
                    degraded_linear.append(
                        robot_linear_raw[env_id].detach().cpu().numpy().astype(np.float32)
                    )
                    degraded_angular.append(
                        robot_angular_raw[env_id].detach().cpu().numpy().astype(np.float32)
                    )
                    continue
                linear_np, angular_np = degrader.degrade_proprioception(
                    env_id,
                    robot_linear_raw[env_id].detach().cpu().numpy(),
                    robot_angular_raw[env_id].detach().cpu().numpy(),
                )
                degraded_linear.append(linear_np)
                degraded_angular.append(angular_np)
            robot_linear = torch.as_tensor(
                np.stack(degraded_linear), device=base_env.device, dtype=torch.float32
            )
            robot_angular = torch.as_tensor(
                np.stack(degraded_angular), device=base_env.device, dtype=torch.float32
            )

            raw_sensor_packets = [None] * base_env.num_envs
            if args.sensor_time_contract == TIME_CONTRACT_VERSION:
                for env_id, estimate in enumerate(estimates):
                    if not recording_ready[env_id]:
                        continue
                    raw_sensor_packets[env_id] = capture_packet(
                        sample_id=step, sample_time_s=nominal_now_s,
                        yaw_pitch=directions[env_id], range_m=float(range_m[env_id].item()),
                        range_valid=bool(range_valid[env_id].item()),
                        source_id=int(TrackerSource.INVALID) if estimate is None else _source_id(estimate.source),
                        quality=np.zeros(7) if estimate is None else _estimate_quality(estimate, trackers[env_id].config),
                        linear=robot_linear[env_id].detach().cpu().numpy(),
                        angular=robot_angular[env_id].detach().cpu().numpy(),
                        camera_position_world=cam_pos_w[env_id].detach().cpu().numpy(),
                        camera_quaternion_world=cam_quat_w[env_id].detach().cpu().numpy(),
                        base_position_world=root_pos_w[env_id].detach().cpu().numpy(),
                        base_quaternion_world=root_quat_w[env_id].detach().cpu().numpy(),
                        tracker_reset=stream_reset_pending[env_id],
                    )
                    raw_sensor_packets[env_id]['image_sampled'] = sampled_images[env_id]
                    raw_sensor_packets[env_id]['range_sampled'] = sampled_images[env_id]
                    raw_sensor_packets[env_id]['tracker_input_dt_s'] = tracker_input_dts[env_id]
                    if not sampled_images[env_id]:
                        raw_sensor_packets[env_id]['range_valid'] = False
                        raw_sensor_packets[env_id]['range_m'] = None

            # Preserve camera-specific tracker loss/out-of-view frames in the
            # episode stream. They remain excluded from training windows, but
            # streaming evaluation can now measure the full failure duration.
            if args.save_episode_stream and args.validation_condition != "none":
                for env_id, estimate in enumerate(estimates):
                    if estimate is not None or not recording_ready[env_id]:
                        continue
                    state_audit = current_pair_audit[env_id]
                    if state_audit is None:
                        raise RuntimeError(
                            f"missing paired state audit for unavailable env {env_id}"
                        )
                    stream_reset_pending[env_id] = True
                    invalid_position = np.full(3, np.nan, dtype=np.float32)
                    unavailable_audit = {
                        **state_audit,
                        "image_timestamp_s": float(timestamps_s[env_id]),
                        "tracker_measurement_timestamp_s": float(timestamps_s[env_id]),
                        "tracker_result_timestamp_s": float(timestamps_s[env_id]),
                        "velocity_input_timestamp_s": float(timestamps_s[env_id]),
                        "label_timestamp_s": float(timestamps_s[env_id]),
                        "target_id": 0,
                        "bbox_pred_xyxy": None,
                        "bbox_gt_xyxy": None,
                        "predicted_center_px": None,
                        "gt_center_px": None,
                        "predicted_yaw_pitch": None,
                        "gt_yaw_pitch": gt_direction[
                            env_id
                        ].detach().cpu().tolist(),
                        "direction_angular_error_deg": None,
                        "bbox_iou_gt": None,
                        "range_measured_m": None,
                        "range_raycast_raw_m": None,
                        "range_gt_m": float(gt_range_for_audit_m[env_id].item()),
                        "range_valid": False,
                        "tracker_source": "lost",
                        "tracker_confidence": 0.0,
                        "motion_confidence": 0.0,
                        "point_retention": 0.0,
                        "ransac_inlier_ratio": 0.0,
                        "median_fb_error_px": None,
                        "bbox_innovation": None,
                        "bbox_iou_with_prediction": None,
                        "fine_correction_center_px": None,
                        "fine_correction_iou": None,
                        "fine_action": "none",
                        "measurement_discontinuity": False,
                        "measurement_rejected": False,
                        "position_observed_base_m": invalid_position.tolist(),
                        "position_gt_base_m": position_gt_base[
                            env_id
                        ].detach().cpu().tolist(),
                        "validation_condition": str(args.validation_condition),
                        "gt_direction_used_for_model_input": bool(
                            gt_direction_validation
                        ),
                        "gt_range_used_for_model_input": bool(gt_range_validation),
                    }
                    episode_stream_frames[env_id].append(
                        _TAQFrame(
                            timestamp_s=float(timestamps_s[env_id]),
                            position_base_m=invalid_position,
                            robot_linear_velocity_base_mps=robot_linear[
                                env_id
                            ].detach().cpu().numpy().astype(np.float32),
                            robot_angular_velocity_base_rps=robot_angular[
                                env_id
                            ].detach().cpu().numpy().astype(np.float32),
                            source_id=int(TrackerSource.INVALID),
                            quality=np.zeros(7, dtype=np.float32),
                            range_valid=False,
                            target_relative_velocity_base_mps=velocity_label[
                                env_id
                            ].detach().cpu().numpy().astype(np.float32),
                            state_reset=True,
                            audit=unavailable_audit,
                            sensor_packet=raw_sensor_packets[env_id],
                        )
                    )
                    if frame_observer is not None:
                        frame_observer(
                            episode_stream_frames[env_id][-1], env_id=env_id,
                            episode_id=episode_ids[env_id], metadata=metadata,
                        )

            for env_id, estimate in enumerate(estimates):
                if estimate is None or not recording_ready[env_id]:
                    continue
                if not bool(base_eligible[env_id].item()):
                    # Do not allow a disturbed/falling robot interval to leak
                    # into ordinary SS/SD training data. Paired validation does
                    # not use this speed filter.
                    histories[env_id].clear()
                    stream_reset_pending[env_id] = True
                    continue
                frame_counts[env_id] += 1
                if bool(getattr(estimate, "measurement_discontinuity", False)):
                    measurement_discontinuities += 1
                    if not args.keep_measurement_discontinuities:
                        histories[env_id].clear()
                        stream_reset_pending[env_id] = True
                valid_range = bool(range_valid[env_id].item()) and bool(torch.isfinite(range_m[env_id]).item())
                position_np = position_base[env_id].detach().cpu().numpy().astype(np.float32)
                if not valid_range:
                    position_np[:] = np.nan
                image_height, image_width = rgb_images[env_id].shape[:2]
                intrinsics = _camera_intrinsics_from_image(args, image_width, image_height)
                gt_box = _bootstrap_box_from_gt(
                    float(gt_direction[env_id, 0].item()),
                    float(gt_direction[env_id, 1].item()),
                    float(gt_range_for_audit_m[env_id].item()),
                    intrinsics,
                    float(args.target_sphere_radius),
                    float(args.min_bootstrap_box_px),
                )
                tracker_config = trackers[env_id].config if trackers[env_id] is not None else None
                estimate_box = tuple(float(value) for value in estimate.bbox_xyxy)
                state_audit = current_pair_audit[env_id]
                if state_audit is None:
                    raise RuntimeError(f"missing paired state audit for env {env_id}")
                audit = {
                    **state_audit,
                    "image_timestamp_s": float(timestamps_s[env_id]),
                    "tracker_measurement_timestamp_s": float(timestamps_s[env_id]),
                    "tracker_result_timestamp_s": float(timestamps_s[env_id]),
                    "velocity_input_timestamp_s": float(timestamps_s[env_id]),
                    "label_timestamp_s": float(timestamps_s[env_id]),
                    "target_id": int(estimate.target_id),
                    "bbox_pred_xyxy": list(estimate_box),
                    "bbox_gt_xyxy": None if gt_box is None else list(gt_box),
                    "predicted_center_px": list(estimate.center_xy),
                    "gt_center_px": None if gt_box is None else [
                        0.5 * (gt_box[0] + gt_box[2]), 0.5 * (gt_box[1] + gt_box[3])
                    ],
                    "predicted_yaw_pitch": [float(directions[env_id, 0]), float(directions[env_id, 1])],
                    "gt_yaw_pitch": [
                        float(gt_direction[env_id, 0].item()),
                        float(gt_direction[env_id, 1].item()),
                    ],
                    "direction_angular_error_deg": _direction_angular_error_deg(
                        float(directions[env_id, 0]), float(directions[env_id, 1]),
                        float(gt_direction[env_id, 0].item()), float(gt_direction[env_id, 1].item()),
                    ),
                    "bbox_iou_gt": None if gt_box is None else _bbox_iou(estimate_box, gt_box),
                    "range_measured_m": float(range_m[env_id].item()) if valid_range else None,
                    "range_raycast_raw_m": float(raw_raycast_range_m[env_id].item()),
                    "range_gt_m": float(gt_range_for_audit_m[env_id].item()),
                    "range_valid": bool(valid_range),
                    "tracker_source": str(estimate.source),
                    "tracker_confidence": float(estimate.confidence),
                    "motion_confidence": float(estimate.motion_confidence),
                    "point_retention": float(estimate.point_retention),
                    "ransac_inlier_ratio": float(estimate.ransac_inlier_ratio),
                    "median_fb_error_px": None if estimate.median_fb_error_px is None else float(estimate.median_fb_error_px),
                    "bbox_innovation": float(estimate.bbox_innovation),
                    "bbox_iou_with_prediction": float(estimate.bbox_iou_with_prediction),
                    "fine_correction_center_px": float(estimate.fine_correction_center_px),
                    "fine_correction_iou": float(estimate.fine_correction_iou),
                    "fine_action": str(estimate.fine_action),
                    "measurement_discontinuity": bool(estimate.measurement_discontinuity),
                    "measurement_rejected": bool(estimate.measurement_rejected),
                    "position_observed_base_m": position_np.tolist(),
                    "position_gt_base_m": position_gt_base[env_id].detach().cpu().tolist(),
                    "validation_condition": str(args.validation_condition),
                    "gt_direction_used_for_model_input": bool(gt_direction_validation),
                    "gt_range_used_for_model_input": bool(gt_range_validation),
                }
                frame = _TAQFrame(
                    timestamp_s=float(timestamps_s[env_id]),
                    position_base_m=position_np,
                    robot_linear_velocity_base_mps=robot_linear[env_id].detach().cpu().numpy().astype(np.float32),
                    robot_angular_velocity_base_rps=robot_angular[env_id].detach().cpu().numpy().astype(np.float32),
                    source_id=_source_id(estimate.source), quality=_estimate_quality(estimate, tracker_config),
                    range_valid=valid_range,
                    target_relative_velocity_base_mps=velocity_label[env_id].detach().cpu().numpy().astype(np.float32),
                    state_reset=stream_reset_pending[env_id] or _source_id(estimate.source) == int(TrackerSource.INITIAL),
                    audit=audit,
                    sensor_packet=raw_sensor_packets[env_id],
                )
                stream_reset_pending[env_id] = False
                if frame_observer is not None:
                    frame_observer(
                        frame, env_id=env_id, episode_id=episode_ids[env_id],
                        metadata=metadata,
                    )
                histories[env_id].append(frame)
                if args.save_episode_stream:
                    episode_stream_frames[env_id].append(frame)
                history = histories[env_id]
                while (
                    len(history) > 2
                    and history[1].timestamp_s
                    < float(timestamps_s[env_id]) - float(args.window_s)
                ):
                    history.popleft()
                if frame_counts[env_id] % int(args.sample_stride) != 0 or len(history) < 2:
                    continue
                if history[-1].timestamp_s - history[0].timestamp_s < 0.9 * float(args.window_s):
                    continue
                window = list(history)[-int(args.max_sequence_steps):]
                if episode_window_counts[env_id] >= int(args.max_windows_per_episode):
                    continue
                window_start_s = float(window[0].timestamp_s)
                window_end_s = float(window[-1].timestamp_s)
                previous_end_s = last_window_end_s[env_id]
                # Strict inequality also prevents adjacent windows from sharing
                # their boundary frame.  History resets never reset this value;
                # only a new simulator episode does.
                if previous_end_s is not None and window_start_s < previous_end_s + float(args.min_window_gap_s) - 1e-9:
                    continue
                if sum(item.range_valid for item in window) < int(args.min_range_valid_frames):
                    continue
                features, _ = build_time_aware_features(
                    timestamps_s=[item.timestamp_s for item in window],
                    position_base_m=[item.position_base_m for item in window],
                    robot_linear_velocity_base_mps=[item.robot_linear_velocity_base_mps for item in window],
                    robot_angular_velocity_base_rps=[item.robot_angular_velocity_base_rps for item in window],
                    source_ids=[item.source_id for item in window],
                    quality=[item.quality for item in window],
                    range_valid_mask=[item.range_valid for item in window],
                )
                episode_samples[env_id].append(
                    _TAQSample(
                        features=features,
                        source_ids=torch.tensor(
                            [item.source_id for item in window], dtype=torch.long
                        ),
                        target_relative_velocity_base_mps=torch.stack(
                            [torch.as_tensor(item.target_relative_velocity_base_mps, dtype=torch.float32) for item in window]
                        ),
                        window_start_s=window_start_s,
                        window_end_s=window_end_s,
                    )
                )
                episode_window_counts[env_id] += 1
                last_window_end_s[env_id] = window_end_s

            pending = sum(len(items) for items in episode_samples)
            if args.progress_every > 0 and (step + 1) % int(args.progress_every) == 0:
                active = sum(tracker is not None and tracker.initialized for tracker in trackers)
                print(
                    f"[taq-collect] step={step + 1}/{args.num_steps} "
                    f"saved={total_samples} pending={pending} "
                    f"active_trackers={active}/{base_env.num_envs} "
                    f"eligible_base={int(base_eligible.sum().item())}/{base_env.num_envs} "
                    f"valid_range={int(range_valid.sum().item())}/{base_env.num_envs} "
                    f"discontinuities={measurement_discontinuities} "
                    f"events={degrader.aggregate_counters()}"
                )
            if args.target_samples > 0 and total_samples + pending >= int(args.target_samples):
                reached_target = True
            if reached_target:
                if not args.save_episode_stream:
                    break
                # Keep every active episode intact when the window quota is
                # reached. Do not record a new episode in an already drained env.
                for env_id in range(base_env.num_envs):
                    if bool(done[env_id]):
                        finished_after_target[env_id] = True
                if all(finished_after_target):
                    break

        for env_id in range(base_env.num_envs):
            total_samples += flush_episode(env_id)
        print(
            f"[taq-collect] complete: saved_samples={total_samples} "
            f"target_reached={reached_target} discontinuities={measurement_discontinuities} "
            f"events={degrader.aggregate_counters()} output={args.out_dir}",
            flush=True,
        )
        completed = True
    finally:
        try:
            if collection_finished is not None:
                collection_finished(completed)
        finally:
            if env is not None:
                env.close()
            sim_app.close()


if __name__ == "__main__":
    main()

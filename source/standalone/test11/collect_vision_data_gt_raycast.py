# collect_vision_data.py

from __future__ import annotations

import argparse
from datetime import datetime
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
    加载视觉模型。
    兼容以下几种常见保存方式：
    1) torch.save(model.state_dict(), path)
    2) torch.save({"model_state_dict": ...}, path)
    3) torch.save({"state_dict": ...}, path)
    4) torch.save(model, path)   # 不太推荐，但有人这么存
    """
    import torch

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vision model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    # 情况 1：直接存了整个 nn.Module
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        model.eval()
        print(f"[vision] loaded full nn.Module from: {model_path}")
        return model

    # 其余情况：需要先构建模型，再加载参数
    model = _build_vision_model(device=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # 有些人会直接 torch.save(model.state_dict(), path)
            state_dict = ckpt
    else:
        raise RuntimeError(
            f"Unsupported checkpoint type: {type(ckpt)}. "
            "Expected nn.Module or dict-like state_dict checkpoint."
        )

    # 兼容 DataParallel 保存的 'module.' 前缀
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[new_k] = v

    # 兼容旧版权重：如果 direction_head 首层输入是 256，而当前模型是 257（多了 dog_moving_flag）
    expected_state_dict = model.state_dict()
    adapted_state_dict = {}
    skipped_keys = []

    for k, v in cleaned_state_dict.items():
        if k not in expected_state_dict:
            adapted_state_dict[k] = v
            continue

        expected = expected_state_dict[k]
        if expected.shape == v.shape:
            adapted_state_dict[k] = v
            continue

        if (
            k == "direction_head.net.0.weight"
            and expected.ndim == 2
            and v.ndim == 2
            and expected.shape[0] == v.shape[0]
            and expected.shape[1] == v.shape[1] + 1
        ):
            patched = expected.clone()
            patched[:, : v.shape[1]] = v
            patched[:, v.shape[1] :] = 0.0
            adapted_state_dict[k] = patched
            print(
                "[vision] adapted direction_head.net.0.weight "
                f"from {tuple(v.shape)} -> {tuple(expected.shape)} by zero-padding proprio column"
            )
            continue

        skipped_keys.append((k, tuple(v.shape), tuple(expected.shape)))

    if len(skipped_keys) > 0:
        print("[vision] skipped incompatible keys:")
        for key, got_shape, expect_shape in skipped_keys:
            print(f"  - {key}: checkpoint={got_shape}, model={expect_shape}")

    missing, unexpected = model.load_state_dict(adapted_state_dict, strict=False)
    print(f"[vision] loaded checkpoint from: {model_path}")
    print(f"[vision] missing keys: {missing}")
    print(f"[vision] unexpected keys: {unexpected}")

    model.eval()
    return model


def run_vision_model(
    vision_model,
    images: "torch.Tensor",
    dog_moving_flag: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    输入:
        images: [N, 3, H, W]，单帧图片 batch
        dog_moving_flag: [N, 1]，机器狗运动状态（0/1）
    输出:
        pred_yaw:   [N]
        pred_pitch: [N]

    说明：
    - 这里不走 TurretVisionNet.forward()
    - 直接使用 encoder + direction_head
    - 这样就适配“单张图片输入”的采集方式
    """
    import torch

    if images.ndim != 4 or images.shape[1] != 3:
        raise RuntimeError(f"Unexpected image shape for vision model: {tuple(images.shape)}")

    if dog_moving_flag.ndim != 2 or dog_moving_flag.shape[0] != images.shape[0] or dog_moving_flag.shape[1] != 1:
        raise RuntimeError(
            "dog_moving_flag must have shape [N, 1], "
            f"got {tuple(dog_moving_flag.shape)}"
        )

    with torch.no_grad():
        features = vision_model.encoder(images)              # [N, 256]
        dog_moving_flag = dog_moving_flag.to(device=features.device, dtype=features.dtype)
        fused_features = torch.cat([features, dog_moving_flag], dim=-1)  # [N, 257]
        direction = vision_model.direction_head(fused_features)           # [N, 2]

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


def estimate_distance_from_lidar_window_multi_env(
    lidar_iface,
    lidar_paths: list[str],
    pred_yaw: "torch.Tensor",
    pred_pitch: "torch.Tensor",
    yaw_half_window_rad: float = np.deg2rad(2.0),
    pitch_half_window_rad: float = np.deg2rad(2.0),
    min_points: int = 3,
):
    """
    多环境版本：每个 env 读取各自 lidar prim，基于对应的预测 yaw/pitch 估计距离。

    输入:
        lidar_paths: 长度为 N 的 prim path 列表
        pred_yaw:    [N]，rad，LiDAR 坐标系
        pred_pitch:  [N]，rad，LiDAR 坐标系

    返回:
        pred_distance: [N]，torch.float32，单位 m
    """
    import torch

    device = pred_yaw.device
    num_envs = pred_yaw.shape[0]

    if len(lidar_paths) != num_envs:
        raise RuntimeError(
            f"lidar_paths length ({len(lidar_paths)}) != batch size ({num_envs})"
        )

    pred_distance_list = []

    for i in range(num_envs):
        lidar_path_i = lidar_paths[i]

        try:
            depth_2d, azimuth_2d, zenith_2d = read_lidar_scan(lidar_iface, lidar_path_i)
        except Exception as exc:
            print(f"[lidar warning] failed to read lidar for env={i}, path={lidar_path_i}, err={exc}")
            pred_distance_list.append(0.0)
            continue

        # 如果后面发现上下方向反了，把这一行改成 -zenith_2d
        beam_pitch_2d = zenith_2d

        yaw_i = float(pred_yaw[i].detach().cpu().item())
        pitch_i = float(pred_pitch[i].detach().cpu().item())

        angle_mask = (
            (np.abs(azimuth_2d - yaw_i) <= yaw_half_window_rad)
            & (np.abs(beam_pitch_2d - pitch_i) <= pitch_half_window_rad)
        )

        valid_mask = (
            np.isfinite(depth_2d)
            & (depth_2d > 1e-4)
            & angle_mask
        )

        candidate_depths = depth_2d[valid_mask]

        if candidate_depths.size < min_points:
            relaxed_mask = (
                (np.abs(azimuth_2d - yaw_i) <= yaw_half_window_rad * 1.8)
                & (np.abs(beam_pitch_2d - pitch_i) <= pitch_half_window_rad * 1.8)
                & np.isfinite(depth_2d)
                & (depth_2d > 1e-4)
            )
            candidate_depths = depth_2d[relaxed_mask]

        if candidate_depths.size == 0:
            pred_distance_list.append(0.0)
            continue

        median_d = float(np.median(candidate_depths))
        cluster_mask = np.abs(candidate_depths - median_d) <= max(0.25, 0.08 * median_d)
        clustered = candidate_depths[cluster_mask]

        if clustered.size >= 1:
            dist = float(np.mean(clustered))
        else:
            dist = median_d

        pred_distance_list.append(dist)

    pred_distance = torch.tensor(pred_distance_list, dtype=torch.float32, device=device)
    pred_distance = torch.nan_to_num(pred_distance, nan=0.0, posinf=0.0, neginf=0.0)
    return pred_distance


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--dynamic_target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_speed_min", type=float, default=0.6)
    parser.add_argument("--target_speed_max", type=float, default=1.2)
    parser.add_argument("--target_vz_min", type=float, default=-0.3)
    parser.add_argument("--target_vz_max", type=float, default=0.3)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--chunk_size", type=int, default=2048) # 每个文件保存多少个 9 帧序列
    parser.add_argument("--out_dir", type=str, default="velocity_dataset")
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--action_scale", type=float, default=0.2)
    parser.add_argument("--locomotion_policy", type=str, default=None)
    parser.add_argument(
        "--vision_model_path",
        type=str,
        default="/home/amx/Documents/isaacsim4.5.0/IsaacLab/source/standalone/test4/vision_model/vision_runs/vision_runs_cnn_only/04-04-11-12/vision_best.pth",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=bool(args.enable_cameras))
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

    # 关键：绕开 camera/xform view 的 Fabric 初始化路径
    env_cfg.sim.use_fabric = False

    env_cfg.training_stage = args.training_stage
    env_cfg.debug_vis = not args.headless
    env_cfg.debug_diagnostics = False
    env_cfg.enable_dynamic_target = bool(args.dynamic_target)

    # 关键：彻底关闭旧 PhysX runtime lidar 路径
    env_cfg.enable_runtime_lidar = False
    # 使用 MultiMeshRayCaster 作为动态目标距离源
    env_cfg.enable_dynamic_target_raycast_debug = True
    env_cfg.dynamic_target_raycast_debug_every = 200   # 少打印一点
    env_cfg.dynamic_target_raycast_max_dist = 25.0
    env_cfg.runtime_lidar_rotation_rate_hz = 0.0

    env_cfg.target_speed_range = (float(args.target_speed_min), float(args.target_speed_max))
    env_cfg.target_vertical_speed_range = (float(args.target_vz_min), float(args.target_vz_max))

    if args.training_stage == 1:
        env_cfg.enable_locomotion_policy = False
    else:
        env_cfg.enable_locomotion_policy = args.locomotion_policy is not None
        env_cfg.locomotion_policy_path = args.locomotion_policy

    TASK_ID = "Standalone-TurretAim-VelocityCollect-v0"
    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point=TurretAimCollectEnv)

    env = gym.make(TASK_ID, cfg=env_cfg, disable_env_checker=True)
    base_env: Any = env.unwrapped

    obs, _ = env.reset()

    print("[collect] PhysX runtime lidar disabled; MultiMeshRayCaster enabled for target distance.")
    sequence_buffers = [[] for _ in range(base_env.num_envs)]
    
    collected_sequences = []
    collected_target_velocities = []

    chunk_id = 0
    total_samples = 0

    meta = {
        "created_at": datetime.now().isoformat(),
        "sequence_length": 9,
        "feature_dim": 21,
        "feature_definition": [
            "timestamp (1)",
            "gt_pos_base_X, gt_pos_base_Y, gt_pos_base_Z (3)",
            "pred_pos_base_X, pred_pos_base_Y, pred_pos_base_Z (3)",
            "robot_ang_vel_x, robot_ang_vel_y, robot_ang_vel_z (3)",
            "robot_lin_vel_x, robot_lin_vel_y, robot_lin_vel_z (3)",
            "dog_moving_flag (1)",
            "robot_roll, robot_pitch, robot_yaw (3)",
            "pred_yaw (1)",
            "pred_pitch (1)",
            "gt_yaw (1)",
            "gt_pitch (1)"
        ],
    }

    current_random_action = torch.zeros((base_env.num_envs, 2), device=base_env.device)
    _printed_gt_raycast_override = False

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

        base_env.refresh_vision_supervision()
        batch = base_env.get_vision_supervision_batch()

        lidar_valid_mask = batch.get("lidar_valid_mask", None)
        if lidar_valid_mask is None:
            lidar_valid_mask = torch.ones((base_env.num_envs,), dtype=torch.bool, device=base_env.device)
        else:
            lidar_valid_mask = lidar_valid_mask.to(device=base_env.device, dtype=torch.bool)
            

        # ------------------------------------------------------------------
        # 1. 运行视觉模块：输入最后一帧图像，输出 pred_yaw / pred_pitch
        # ------------------------------------------------------------------
        images = batch["images"][:, -1, ...]   # [N, 3, H, W]

        dog_moving_flag = batch.get("dog_moving_flag", None)
        if dog_moving_flag is None:
            robot_speed_xy = torch.norm(base_env.turret.data.root_lin_vel_b[:, :2], dim=-1, keepdim=True)
            dog_moving_flag = (robot_speed_xy > 0.05).to(torch.float32)
        else:
            if dog_moving_flag.ndim == 1:
                dog_moving_flag = dog_moving_flag.unsqueeze(-1)
            dog_moving_flag = dog_moving_flag.to(device=base_env.device, dtype=torch.float32)

        # 1. 获取模型预测的角度（保留用于日志和数据记录）
        pred_yaw, pred_pitch = run_vision_model(
            vision_model=vision_model,
            images=images,
            dog_moving_flag=dog_moving_flag,
        )

        # 2. 从环境 batch 中提取绝对真实的 GT 角度
        gt_yaw = batch["direction"][:, 0]
        gt_pitch = batch["direction"][:, 1]

        # 调试模式：让 MultiMeshRayCaster 始终沿 GT yaw / GT pitch 发射，
        # 用来验证 raycast 命中和距离返回链路是否正确。
        if hasattr(base_env, "set_dynamic_target_raycast_angles"):
            base_env.set_dynamic_target_raycast_angles(
                gt_yaw.detach(),
                gt_pitch.detach(),
            )

        if not _printed_gt_raycast_override:
            print("[collect] MultiMeshRayCaster direction override = GT yaw/pitch")
            _printed_gt_raycast_override = True

        if (step + 1) % 100 == 0:
            print(
                "[vision infer]",
                f"step={step+1}",
                f"image_shape={tuple(images.shape)}",
                f"dog_moving_flag[0]={float(dog_moving_flag[0, 0].item()):.1f}",
                f"pred_yaw[0]={float(pred_yaw[0].item()):.4f}",
                f"pred_pitch[0]={float(pred_pitch[0].item()):.4f}",
            )

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

        if (step + 1) % 100 == 0:
            lidar_ready = False
            if hasattr(base_env, "runtime_lidar_ready"):
                lidar_ready = bool(base_env.runtime_lidar_ready())

            print(
                "[distance debug]",
                f"step={step+1}",
                f"lidar_ready={lidar_ready}",
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
        # 4. Camera -> World -> Base
        # ------------------------------------------------------------
        cam_pos_w = base_env._vision_camera.data.pos_w
        cam_quat_w = base_env._vision_camera.data.quat_w_ros
        pred_pos_w = cam_pos_w + quat_apply(cam_quat_w, pred_pos_cam)

        root_pos_w = base_env.turret.data.root_pos_w
        root_quat_w = base_env.turret.data.root_quat_w
        pred_pos_base = base_env._rotate_world_to_local(root_quat_w, pred_pos_w - root_pos_w)

        # ------------------------------------------------------------------
        # 2. 提取其他物理数据
        # ------------------------------------------------------------------
        gt_pos_base = batch["pos_base_gt"]  # 仿真吐出的真实 Base 坐标系目标位置 [N, 3]
        target_velocity_base = batch["velocity_base"]
        robot_lin_vel = base_env.turret.data.root_lin_vel_b # [N, 3]
        robot_ang_vel = base_env.turret.data.root_ang_vel_b # [N, 3]

        # 计算机器狗欧拉角
        roll, pitch, yaw = euler_xyz_from_quat(base_env.turret.data.root_quat_w)
        robot_euler = torch.stack([roll, pitch, yaw], dim=-1) # [N, 3]

        timestamp = torch.full((base_env.num_envs, 1), step * base_env.step_dt, device=base_env.device)

        # 将 1 维的角度张量扩展为 [N, 1] 形状，以便能够与后面的张量拼接
        pred_yaw_col = pred_yaw.unsqueeze(-1)
        pred_pitch_col = pred_pitch.unsqueeze(-1)
        gt_yaw_col = gt_yaw.unsqueeze(-1)
        gt_pitch_col = gt_pitch.unsqueeze(-1)

        # 拼接单步 21 维特征向量 (全部统一为 Base 坐标系)
        current_features = torch.cat([
            timestamp,      # 1
            gt_pos_base,    # 3
            pred_pos_base,  # 3
            robot_ang_vel,  # 3
            robot_lin_vel,  # 3
            dog_moving_flag, # 1
            robot_euler,    # 3
            pred_yaw_col,   # 1  (新增: 预测偏航角)
            pred_pitch_col, # 1  (新增: 预测俯仰角)
            gt_yaw_col,     # 1  (新增: 真实偏航角)
            gt_pitch_col    # 1  (新增: 真实俯仰角)
        ], dim=-1)

        # ------------------------------------------------------------------
        # 3. 严格连续 9 帧队列管理
        # ------------------------------------------------------------------
        terminated_t = torch.as_tensor(terminated, device=base_env.device)
        truncated_t = torch.as_tensor(truncated, device=base_env.device)
        done_mask = (terminated_t | truncated_t).to(dtype=torch.bool)
        valid_mask = batch.get("valid_mask", None)

        # keep_mask 为 True 意味着：
        # 1) 没 reset / 没 done
        # 2) 当前视觉样本有效
        # 3) 当前 lidar / raycast 样本有效
        if valid_mask is not None:
            keep_mask = valid_mask.bool() & (~done_mask) & lidar_valid_mask.bool()
        else:
            keep_mask = (~done_mask) & lidar_valid_mask.bool()

        keep_mask_cpu = keep_mask.cpu().numpy()
        current_features_cpu = current_features.cpu().numpy()

        for env_idx in range(base_env.num_envs):
            # 只收：视觉有效 + 未done + raycast真正命中
            if not bool(keep_mask_cpu[env_idx]):
                sequence_buffers[env_idx].clear()
                continue

            sequence_buffers[env_idx].append(current_features_cpu[env_idx])

            # 满 9 帧就记一个样本
            if len(sequence_buffers[env_idx]) == 9:
                seq_data = torch.tensor(
                    np.array(sequence_buffers[env_idx]),
                    dtype=torch.float32,
                )
                collected_sequences.append(seq_data)

                # 标签仍然用当前时刻目标在 base 坐标系下的真实速度
                target_vel_label = target_velocity_base[env_idx].detach().cpu().to(torch.float32)
                collected_target_velocities.append(target_vel_label)

                total_samples += 1

                # 滑动窗口
                sequence_buffers[env_idx].pop(0)

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

    print(f"[collect] Collection Finished. Total 9-Frame Sequences: {total_samples}")

if __name__ == "__main__":
    main()
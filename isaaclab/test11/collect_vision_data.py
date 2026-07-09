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

# Add source/standalone so the sibling test4 package can be imported directly.
THIS_DIR = Path(__file__).resolve().parent
STANDALONE_DIR = THIS_DIR.parent
if str(STANDALONE_DIR) not in sys.path:
    sys.path.insert(0, str(STANDALONE_DIR))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--target_speed_min", type=float, default=0.6)
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
    parser.add_argument("--seed", type=int, default=65)
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

if __name__ == "__main__":
    main()

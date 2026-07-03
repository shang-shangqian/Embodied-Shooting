# collect_vision_data.py

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher


# Ensure we can import ../turret_aim_env.py when running this script directly.
THIS_DIR = Path(__file__).resolve().parent
TEST4_DIR = THIS_DIR.parent
if str(TEST4_DIR) not in sys.path:
    sys.path.insert(0, str(TEST4_DIR))


def _has_any_done(flag: Any) -> bool:
    if hasattr(flag, "any"):
        return bool(flag.any().item())
    return bool(flag)


def _flush_chunk(
    out_dir: Path,
    chunk_id: int,
    images_list: list,
    direction_list: list,
    velocity_list: list,
    state_list: list,
    dog_moving_flag_value: float,
    meta: dict,
):
    import torch

    if len(images_list) == 0:
        return

    images = torch.cat(images_list, dim=0)         # [M, T, 3, H, W]
    direction = torch.cat(direction_list, dim=0)   # [M, 2]
    velocity = torch.cat(velocity_list, dim=0)     # [M, 3]
    state = torch.cat(state_list, dim=0)           # [M, 10]
    dog_moving_flag = torch.full((images.shape[0], 1), float(dog_moving_flag_value), dtype=torch.float32)

    save_path = out_dir / f"vision_chunk_{chunk_id:05d}.pt"
    # images 当前通常是 float32 且范围 [0,1]
    # 存盘时转成 uint8，可大幅减小体积
    if images.dtype == torch.float32:
        images_to_save = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        images_to_save = images

    payload = {
        "images": images_to_save.cpu(),   # uint8
        "direction": direction.cpu().to(torch.float16),
        "velocity": velocity.cpu().to(torch.float16),
        "state": state.cpu().to(torch.float16),
        "dog_moving_flag": dog_moving_flag.cpu().to(torch.float16),
        "meta": {
            **meta,
            "images_dtype": "uint8",
            "direction_dtype": "float16",
            "velocity_dtype": "float16",
            "state_dtype": "float16",
            "dog_moving_flag_dtype": "float16",
            "images_range": "[0,255]",
        },
    }
    torch.save(payload, save_path)
    print(
        f"[collect] saved {save_path} | "
        f"images={tuple(images.shape)} direction={tuple(direction.shape)} "
        f"velocity={tuple(velocity.shape)} state={tuple(state.shape)} "
        f"dog_moving_flag={tuple(dog_moving_flag.shape)} value={dog_moving_flag_value}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Simulation device, e.g. cuda:0 / cuda:1 / cpu.",
    )
    parser.add_argument(
        "--enable_cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable camera rendering required by TiledCamera.",
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--dynamic_target", action="store_true")
    parser.add_argument("--target_speed_min", type=float, default=0.6)
    parser.add_argument("--target_speed_max", type=float, default=1.2)
    parser.add_argument("--target_vz_min", type=float, default=-0.3)
    parser.add_argument("--target_vz_max", type=float, default=0.3)

    parser.add_argument("--num_steps", type=int, default=5000, help="How many env steps to collect.")
    parser.add_argument("--chunk_size", type=int, default=2048, help="How many samples per saved chunk.")
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=4,
        help="Temporal stride between saved samples for each env. "
            "sample_stride=4 means non-overlapping 4-frame windows: [1,2,3,4] -> [5,6,7,8].",
    )
    parser.add_argument("--out_dir", type=str, default="vision_dataset")
    parser.add_argument("--seed", type=int, default=100)

    # 随机动作幅度，可先让炮塔和底盘都有点变化
    parser.add_argument("--action_scale", type=float, default=0.2)

    # Static and moving data both use locomotion policy; stage 1 uses zero commands.
    parser.add_argument("--locomotion_policy", type=str, default=None)

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 启动 Isaac Sim
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=bool(args.enable_cameras))
    sim_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch

        from turret_aim_env_collect import TurretAimCollectEnv, TurretAimCollectEnvCfg
    except ModuleNotFoundError as exc:
        if "omni." in str(exc) or "omni" in str(exc):
            raise ModuleNotFoundError(
                "Isaac Sim python modules were not found/initialized.\n"
                "Run via Isaac Lab launcher, e.g.:\n"
                "  ./isaaclab.sh -p source/standalone/testX/collect_vision_data.py --headless\n"
            ) from exc
        raise

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建环境配置
    env_cfg = TurretAimCollectEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(args.device)
    env_cfg.training_stage = args.training_stage
    env_cfg.debug_vis = not args.headless
    env_cfg.debug_diagnostics = True
    env_cfg.enable_dynamic_target = bool(args.dynamic_target)

    # 按你的要求：保留你希望的小球运动参数，不改距离/速度分布
    env_cfg.target_speed_range = (float(args.target_speed_min), float(args.target_speed_max))
    env_cfg.target_vertical_speed_range = (float(args.target_vz_min), float(args.target_vz_max))

    print(f"[collect] using sim device: {env_cfg.sim.device}")

    env_cfg.enable_locomotion_policy = args.locomotion_policy is not None
    env_cfg.locomotion_policy_path = args.locomotion_policy
    if not env_cfg.enable_locomotion_policy:
        raise ValueError("--locomotion_policy is required for both static and moving data collection.")

    if args.training_stage == 1:
        env_cfg.locomotion_lin_vel_x_range = (0.0, 0.0)
        env_cfg.locomotion_lin_vel_y_range = (0.0, 0.0)
        env_cfg.locomotion_ang_vel_z_range = (0.0, 0.0)
        print("[collect] Stage-1 enabled: locomotion policy with zero command.")
    else:
        print("[collect] Stage-2/3 enabled: locomotion policy with sampled motion commands.")

    # 机器狗运动状态标签：仅根据 training_stage 推断
    # stage=1 -> 静止(0), stage=2/3 -> 运动(1)
    dog_moving_flag_value = 0.0 if int(args.training_stage) == 1 else 1.0
    print(f"[collect] dog_moving_flag={dog_moving_flag_value} (from training_stage={args.training_stage})")

    TASK_ID = "Standalone-TurretAim-VisionCollectOnly-v0"
    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point=TurretAimCollectEnv)

    env = gym.make(TASK_ID, cfg=env_cfg, disable_env_checker=True)
    base_env: Any = env.unwrapped

    # 这里要求你在 turret_aim_env.py 里新增这个接口
    if not hasattr(base_env, "get_vision_supervision_batch"):
        raise AttributeError(
            "TurretAimEnv must implement `get_vision_supervision_batch()` before collecting vision data.\n"
            "Expected return dict:\n"
            "  {\n"
            "    'images': Tensor[num_envs, T, 3, 128, 128],\n"
            "    'direction': Tensor[num_envs, 2],\n"
            "    'velocity': Tensor[num_envs, 3],\n"
            "  }"
        )

    obs, _ = env.reset()

    images_buf: list[torch.Tensor] = []
    direction_buf: list[torch.Tensor] = []
    velocity_buf: list[torch.Tensor] = []
    state_buf: list[torch.Tensor] = []

    # per-env stride gate:
    # 只有 cooldown==0 的 env，当前 step 才允许把样本写入数据集
    # 当某个 env 本 step 被采样后，将其 cooldown 置为 sample_stride-1
    # 之后每步递减，直到再次变成 0
    sample_stride = max(1, int(args.sample_stride))
    save_cooldown = torch.zeros((base_env.num_envs,), dtype=torch.long, device=base_env.device)

    chunk_id = 0
    total_samples = 0

    meta = {
        "created_at": datetime.now().isoformat(),
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "training_stage": int(args.training_stage),
        "dog_moving_flag_value": dog_moving_flag_value,
        "dog_moving_flag_definition": "0=static, 1=moving",
        "dog_moving_flag_source": "inferred from --training_stage",
        "dynamic_target": bool(args.dynamic_target),
        "image_shape": [4, 3, 128, 128],  # 这里假设 T=4，后续可由环境真实返回覆盖
        "direction_definition": "[yaw_cam, pitch_cam]",
        "velocity_definition": "[vx_cam, vy_cam, vz_cam]",
        "state_definition": [
            "yaw",
            "pitch",
            "yaw_vel",
            "pitch_vel",
            "root_ang_vel_b_x",
            "root_ang_vel_b_y",
            "root_ang_vel_b_z",
            "projected_gravity_b_x",
            "projected_gravity_b_y",
            "projected_gravity_b_z",
        ],
        "state_time_alignment": "state corresponds to the last image frame in each T-frame window",
    }

    current_random_action = torch.zeros((base_env.num_envs, 2), device=base_env.device)

    for step in range(args.num_steps):
        action = torch.zeros((base_env.num_envs, 3), device=base_env.device, dtype=torch.float32)

        # 修复：每隔 20步（约0.4秒）换一次随机扫描方向，而不是每步都换
        if float(args.action_scale) > 0.0 and step % 20 == 0:
            current_random_action = (
                (torch.rand((base_env.num_envs, 2), device=base_env.device) * 2.0 - 1.0)
                * float(args.action_scale)
            )
        action[:, :2] = current_random_action

        # 采集阶段固定不开火
        action[:, 2] = -1.0

        obs, rew, terminated, truncated, info = env.step(action)

        # 显式刷新一次视觉缓存
        base_env.refresh_vision_supervision()

        # 从环境取视觉训练数据
        batch = base_env.get_vision_supervision_batch()

        images = batch["images"]         # [N, T, 3, H, W]
        direction = batch["direction"]   # [N, 2]
        velocity = batch["velocity"]     # [N, 3]

        # ------------------------------------------------------------------
        # collect state aligned with the LAST frame of the image window
        # state = [yaw, pitch, yaw_vel, pitch_vel, root_ang_vel_b(3), projected_gravity_b(3)]
        # ------------------------------------------------------------------
        joint_pos = base_env.turret.data.joint_pos
        joint_vel = base_env.turret.data.joint_vel

        yaw = joint_pos[:, int(base_env._yaw_dof_ids[0])].unsqueeze(-1)
        pitch = joint_pos[:, int(base_env._pitch_dof_ids[0])].unsqueeze(-1)
        yaw_vel = joint_vel[:, int(base_env._yaw_dof_ids[0])].unsqueeze(-1)
        pitch_vel = joint_vel[:, int(base_env._pitch_dof_ids[0])].unsqueeze(-1)

        root_ang_vel_b = base_env.turret.data.root_ang_vel_b           # [N, 3]
        projected_gravity_b = base_env.turret.data.projected_gravity_b # [N, 3]

        state = torch.cat(
            [
                yaw,
                pitch,
                yaw_vel,
                pitch_vel,
                root_ang_vel_b,
                projected_gravity_b,
            ],
            dim=-1,
        )  # [N, 10]
        # 基本检查
        if images.ndim != 5:
            raise ValueError(f"Expected images [N,T,3,H,W], got {tuple(images.shape)}")
        if direction.ndim != 2 or direction.shape[-1] != 2:
            raise ValueError(f"Expected direction [N,2], got {tuple(direction.shape)}")
        if velocity.ndim != 2 or velocity.shape[-1] != 3:
            raise ValueError(f"Expected velocity [N,3], got {tuple(velocity.shape)}")
        if state.ndim != 2 or state.shape[-1] != 10:
            raise ValueError(f"Expected state [N,10], got {tuple(state.shape)}")

        valid_mask = batch.get("valid_mask", None)

        # 过滤本 step 已经 done 的 env，避免把 reset/切换期样本混进数据集
        terminated_t = torch.as_tensor(terminated, device=base_env.device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=base_env.device, dtype=torch.bool)
        done_mask = terminated_t | truncated_t

        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            eligible_mask = valid_mask & (~done_mask)
        else:
            eligible_mask = ~done_mask

        # 只在 cooldown==0 时允许保存，避免滑动窗口逐帧重叠采样
        stride_ready_mask = save_cooldown == 0
        keep_mask = eligible_mask & stride_ready_mask

        if (step + 1) % 50 == 0:
            print(
                f"[debug] step={step+1} "
                f"valid_raw={int(valid_mask.sum().item()) if valid_mask is not None else -1}/"
                f"{int(valid_mask.numel()) if valid_mask is not None else base_env.num_envs} "
                f"done={int(done_mask.sum().item())}/{int(done_mask.numel())} "
                f"stride_ready={int(stride_ready_mask.sum().item())}/{int(stride_ready_mask.numel())} "
                f"kept={int(keep_mask.sum().item())}/{int(keep_mask.numel())}"
            )

        # 先按 keep_mask 取出本步真正要保存的样本
        images = images[keep_mask]
        direction = direction[keep_mask]
        velocity = velocity[keep_mask]
        state = state[keep_mask]

        if images.shape[0] > 0:
            images_buf.append(images.detach().cpu())
            direction_buf.append(direction.detach().cpu())
            velocity_buf.append(velocity.detach().cpu())
            state_buf.append(state.detach().cpu())
            total_samples += int(images.shape[0])

        # 更新 per-env cooldown：
        # 1) 已经结束的 env 清零（reset 后重新开始计）
        save_cooldown[done_mask] = 0

        # 2) 所有仍 >0 的 env 每步减 1
        positive_mask = save_cooldown > 0
        save_cooldown[positive_mask] -= 1

        # 3) 本步刚刚被保存的 env，重新置为 sample_stride-1
        if sample_stride > 1:
            save_cooldown[keep_mask] = sample_stride - 1

        # 分块写盘
        current_chunk_samples = sum(x.shape[0] for x in images_buf)
        if current_chunk_samples >= args.chunk_size:
            actual_image_shape = list(images_buf[0].shape[1:]) if len(images_buf) > 0 else [4, 3, 128, 128]
            meta["image_shape"] = actual_image_shape
            _flush_chunk(
                out_dir=out_dir,
                chunk_id=chunk_id,
                images_list=images_buf,
                direction_list=direction_buf,
                velocity_list=velocity_buf,
                state_list=state_buf,
                dog_moving_flag_value=dog_moving_flag_value,
                meta=meta,
            )
            chunk_id += 1
            images_buf.clear()
            direction_buf.clear()
            velocity_buf.clear()
            state_buf.clear()
        # 不要因为单个 env done 就全局 reset
        # IsaacLab 的 VecEnv 一般会在内部处理 done env 的 reset/回收
        # 这里不主动 env.reset()

        if (step + 1) % 100 == 0:
            num_done = int(done_mask.sum().item())
            print(f"[collect] step={step+1}/{args.num_steps} total_samples={total_samples} done_envs={num_done}")

    # flush tail
    actual_image_shape = list(images_buf[0].shape[1:]) if len(images_buf) > 0 else [4, 3, 128, 128]
    meta["image_shape"] = actual_image_shape
    _flush_chunk(
        out_dir=out_dir,
        chunk_id=chunk_id,
        images_list=images_buf,
        direction_list=direction_buf,
        velocity_list=velocity_buf,
        state_list=state_buf,
        dog_moving_flag_value=dog_moving_flag_value,
        meta=meta,
    )
    env.close()
    sim_app.close()
    print(f"[collect] done. total_samples={total_samples} out_dir={out_dir}")


if __name__ == "__main__":
    main()

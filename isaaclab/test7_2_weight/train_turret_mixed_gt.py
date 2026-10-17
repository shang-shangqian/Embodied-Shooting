import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Resolve sibling runtime packages for both direct-script and installed use.
_MODULE_DIR = Path(__file__).resolve().parent
for _import_dir in (_MODULE_DIR.parent, _MODULE_DIR):
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

import asym_central_value_gru_network_gt
from isaaclab.app import AppLauncher
from target_follow_motion_gt import install_target_follow_motion
from ppo_training_config import add_ppo_arguments, make_learning_rate_restore_agent, resolve_ppo_settings
import types


TRAIN_DIR = Path("runs") / "test7_2_weight"


def _parse_rank_env(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer environment variable {name}={value!r}") from exc


def _get_rank_info(distributed: bool) -> tuple[int, int, int]:
    """Return (local_rank, global_rank, world_size) from torchrun env vars."""
    if not distributed:
        return 0, 0, 1

    missing = [name for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE") if name not in os.environ]
    if missing:
        raise RuntimeError(
            "--distributed must be launched with torchrun/python -m torch.distributed.run. "
            f"Missing env vars: {missing}"
        )

    local_rank = _parse_rank_env("LOCAL_RANK")
    global_rank = _parse_rank_env("RANK")
    world_size = _parse_rank_env("WORLD_SIZE", 1)
    if world_size <= 1:
        raise RuntimeError("--distributed requires WORLD_SIZE > 1. Use torchrun --nproc_per_node=2 for dual GPU.")
    return local_rank, global_rank, world_size


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "run"


def _resolve_experiment_name(requested_name: str | None, distributed: bool, global_rank: int, world_size: int) -> str:
    """Use one shared run name across torchrun ranks."""
    start_time = time.time()
    if requested_name is not None and requested_name.strip() != "":
        return _sanitize_name(requested_name)

    if not distributed:
        return f"turret_aim_mixed_{datetime.now().strftime('%m-%d-%H-%M-%S')}"

    rendezvous_id = os.environ.get("TORCHELASTIC_RUN_ID", "").strip()
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    key = _sanitize_name(f"{rendezvous_id}_{master_addr}_{master_port}_{world_size}")
    name_file = TRAIN_DIR / ".distributed_run_names" / f"{key}.txt"

    if global_rank == 0:
        run_name = f"turret_aim_mixed_{datetime.now().strftime('%m-%d-%H-%M-%S')}"
        name_file.parent.mkdir(parents=True, exist_ok=True)
        name_file.write_text(run_name, encoding="utf-8")
        return run_name

    deadline = time.time() + 60.0
    while time.time() < deadline:
        if name_file.exists() and name_file.stat().st_mtime >= start_time - 5.0:
            run_name = name_file.read_text(encoding="utf-8").strip()
            if run_name:
                return _sanitize_name(run_name)
        time.sleep(0.1)

    # Last-resort fallback keeps ranks aligned if TORCHELASTIC_RUN_ID is stable.
    return f"turret_aim_mixed_dist_{key}"


def _pick_checkpoint_from_dir(run_dir: Path, which: str) -> Path:
    nn_dir = run_dir / "nn"
    search_dirs = [nn_dir, run_dir]

    all_pths: list[Path] = []
    for directory in search_dirs:
        if directory.is_dir():
            all_pths.extend(sorted(directory.glob("*.pth")))

    if len(all_pths) == 0:
        raise FileNotFoundError(
            f"No .pth checkpoints found under: {run_dir} (looked in '{nn_dir}' and the directory root)"
        )

    best_candidates = [
        path
        for path in all_pths
        if (
            path.name == "turret_aim.pth"
            or (path.parent.name == "nn" and not path.name.startswith("last_"))
        )
    ]
    last_candidates = [path for path in all_pths if path.name.startswith("last_")]

    def newest(paths: list[Path]) -> Path:
        return max(paths, key=lambda path: path.stat().st_mtime)

    which = which.lower().strip()
    if which == "best":
        if len(best_candidates) > 0:
            return newest(best_candidates)
        non_last = [path for path in all_pths if not path.name.startswith("last_")]
        return newest(non_last) if len(non_last) > 0 else newest(all_pths)
    if which == "last":
        if len(last_candidates) == 0:
            raise FileNotFoundError(
                f"No 'last_*.pth' checkpoints found under: {run_dir}. Available: {[path.name for path in all_pths]}"
            )
        return newest(last_candidates)
    if which == "latest":
        return newest(all_pths)
    raise ValueError("--which must be one of: best, last, latest")


def _resolve_checkpoint_path(path_str: str, which: str = "best") -> Path:
    ckpt_path = Path(path_str).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if ckpt_path.is_dir():
        picked = _pick_checkpoint_from_dir(ckpt_path, which=which)
        print(f"[train-mixed] Using checkpoint ({which}): {picked}")
        ckpt_path = picked

    if ckpt_path.suffix != ".pth":
        raise ValueError(f"Expected a .pth checkpoint file, got: {ckpt_path}")

    return ckpt_path


def _install_mixed_training_logic(
    base_env: Any,
    case_probs: tuple[float, float, float, float],
):
    import torch
    import types

    prob = torch.tensor(case_probs, device=base_env.device, dtype=torch.float32)
    if torch.any(prob < 0.0):
        raise ValueError("case probabilities must be >= 0")
    prob_sum = float(prob.sum().item())
    if prob_sum <= 0.0:
        raise ValueError("case probabilities sum must be > 0")
    prob = prob / prob.sum()

    # case id mapping
    # 0: robot_static + target_static
    # 1: robot_static + target_dynamic
    # 2: robot_dynamic + target_static
    # 3: robot_dynamic + target_dynamic
    base_env._mix_case_prob = prob
    base_env._mix_case_id = torch.zeros((base_env.num_envs,), dtype=torch.long, device=base_env.device)
    base_env._mix_robot_dynamic = torch.zeros((base_env.num_envs,), dtype=torch.bool, device=base_env.device)
    base_env._mix_target_dynamic = torch.zeros((base_env.num_envs,), dtype=torch.bool, device=base_env.device)
    base_env._mix_case_counts = torch.zeros((4,), dtype=torch.long, device=base_env.device)

    original_reset_idx = base_env._reset_idx
    original_pre_physics_step = base_env._pre_physics_step

    def _sample_case_ids(count: int) -> torch.Tensor:
        if count <= 0:
            return torch.zeros((0,), dtype=torch.long, device=base_env.device)
        return torch.multinomial(base_env._mix_case_prob, num_samples=count, replacement=True)

    def _apply_case_masks(env_ids_t: torch.Tensor):
        if env_ids_t.numel() == 0:
            return

        case_ids = base_env._mix_case_id[env_ids_t]
        robot_dynamic = case_ids >= 2
        target_dynamic = (case_ids % 2) == 1
        robot_static = ~robot_dynamic
        target_static = ~target_dynamic

        base_env._mix_robot_dynamic[env_ids_t] = robot_dynamic
        base_env._mix_target_dynamic[env_ids_t] = target_dynamic
        # Every moving target stays inside the robot/camera distance shell and
        # angular FOV. For a static robot the reset anchor remains fixed; for a
        # moving robot the constraint frame follows the current robot pose.
        relative_constraint = target_dynamic
        fixed_anchor_constraint = (~robot_dynamic) & target_dynamic
        base_env._target_motion_mask[env_ids_t] = target_dynamic
        base_env._target_relative_constraint_mask[env_ids_t] = relative_constraint
        base_env._target_constraint_fixed_anchor_mask[env_ids_t] = fixed_anchor_constraint
        base_env._target_out_of_view_termination_mask[env_ids_t] = ~robot_dynamic

        if torch.any(robot_static):
            static_ids = env_ids_t[robot_static]
            base_env._locomotion_commands[static_ids] = 0.0
            base_env._locomotion_command_time_left[static_ids] = base_env.cfg.locomotion_command_resample_s
            base_env._locomotion_last_actions[static_ids] = 0.0

            # 静止狗：仍然使用 locomotion policy，
            # 但把 command 固定为 0，不再随机重采样，不再强制写死腿部目标。
            base_env._locomotion_commands[static_ids] = 0.0
            base_env._locomotion_command_time_left[static_ids] = base_env.cfg.locomotion_command_resample_s

            # reset 时把上一帧动作清零，避免跨 episode 残留。
            # 注意：这里只在 reset 时清零，step 内不要再清零，
            # 否则就不是“零指令下跑 locomotion 模型”了。
            base_env._locomotion_last_actions[static_ids] = 0.0

        # All cases retain the reset-time warmup. Even with zero locomotion
        # command, SS must wait for the standing policy and articulation to
        # settle before aiming or firing.

        if torch.any(target_static):
            static_target_ids = env_ids_t[target_static]
            base_env._target_vel_w[static_target_ids] = 0.0
            if hasattr(base_env, "_target_acc_w"):
                base_env._target_acc_w[static_target_ids] = 0.0

        constrained_ids = env_ids_t[relative_constraint]
        if constrained_ids.numel() > 0:
            fixed_mask = base_env._target_constraint_fixed_anchor_mask[constrained_ids]
            fixed_ids = constrained_ids[fixed_mask]
            moving_anchor_ids = constrained_ids[~fixed_mask]
            if fixed_ids.numel() > 0:
                base_env._handle_target_out_of_bounds(fixed_ids, fixed_anchor=True)
            if moving_anchor_ids.numel() > 0:
                base_env._handle_target_out_of_bounds(moving_anchor_ids, fixed_anchor=False)
            base_env._episode_start_target_pos_w[constrained_ids] = base_env._target_pos_w[constrained_ids]

    def _resample_cases(env_ids_t: torch.Tensor):
        if env_ids_t.numel() == 0:
            return
        sampled = _sample_case_ids(int(env_ids_t.numel()))
        base_env._mix_case_id[env_ids_t] = sampled
        ones = torch.ones_like(sampled, dtype=torch.long)
        base_env._mix_case_counts.scatter_add_(0, sampled, ones)

    def patched_reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        result = original_reset_idx(env_ids_t)

        # mixed-case 逻辑
        _resample_cases(env_ids_t)
        _apply_case_masks(env_ids_t)

        # perception bridge 的时序缓存也在这里一起 reset
        if hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None:
            self._rl_perception_bridge.reset_envs(env_ids_t)

        return result

    def patched_pre_physics_step(self, actions):
        robot_static_mask = ~self._mix_robot_dynamic
        # 静止机器狗仍然走 locomotion policy，
        # 只是 command 固定为 0，且不允许被随机重采样。
        robot_static_mask = ~self._mix_robot_dynamic
        if torch.any(robot_static_mask):
            self._locomotion_commands[robot_static_mask] = 0.0
            self._locomotion_command_time_left[robot_static_mask] = self.cfg.locomotion_command_resample_s

            # 关键：
            # 不能设成 0，否则 original_pre_physics_step -> _update_locomotion_commands()
            # 会立刻把这些静止狗重采样成随机速度指令。
            self._locomotion_command_time_left[robot_static_mask] = self.cfg.locomotion_command_resample_s

            # Do not clear SS warmup here. Zero-command standing still needs
            # the same settling interval as SD before the turret is enabled.

        # 让原始逻辑正常执行：
        # _update_locomotion_commands()
        # _compute_locomotion_observations()
        # locomotion_policy(obs)
        # _locomotion_joint_targets = default + scale * action
        original_pre_physics_step(actions)

        # 原逻辑之后不要再把静止狗写回默认腿位，
        # 否则就等于绕过了 locomotion policy。

        # 静态目标速度强制为 0，避免目标漂移。
        target_static_mask = ~self._mix_target_dynamic
        if torch.any(target_static_mask):
            self._target_vel_w[target_static_mask] = 0.0
            if hasattr(self, "_target_acc_w"):
                self._target_acc_w[target_static_mask] = 0.0

    def get_mix_case_stats(self):
        counts = self._mix_case_counts.detach().cpu().to(dtype=torch.float32)
        total = float(counts.sum().item())
        if total <= 0.0:
            return {
                "ss": 0.0,
                "sd": 0.0,
                "ds": 0.0,
                "dd": 0.0,
                "total": 0.0,
            }
        return {
            "ss": float((counts[0] / total).item()),
            "sd": float((counts[1] / total).item()),
            "ds": float((counts[2] / total).item()),
            "dd": float((counts[3] / total).item()),
            "total": total,
        }

    base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)
    base_env._pre_physics_step = types.MethodType(patched_pre_physics_step, base_env)
    base_env.get_mix_case_stats = types.MethodType(get_mix_case_stats, base_env)
    # gym.make() has already performed the first environment reset before this
    # patch is installed. Sample and apply cases now so the first rollout is
    # not accidentally all-SS.
    initial_env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
    _resample_cases(initial_env_ids)
    _apply_case_masks(initial_env_ids)
    print(
        "[train-mixed] mixed-case logic installed | "
        "all cases use aim_warmup_s for no-aim/no-fire warmup"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--training_mode",
        choices=("gt", "full_chain"),
        default="gt",
        help="gt uses privileged target state; full_chain uses RGB tracker + raycast range + frozen velocity GRU.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Single-GPU device, e.g. cuda:0/cuda:1/cpu.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Run one training job across multiple GPUs with torchrun.",
    )
    parser.add_argument("--num_envs", type=int, default=None)
    add_ppo_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional: existing checkpoint file or run directory. Leave empty to train from scratch.",
    )
    parser.add_argument("--which", type=str, default="best", choices=["best", "last", "latest"])
    parser.add_argument("--max_epochs", type=int, default=15000)
    parser.add_argument("--max_shots_per_episode", type=int, default=0,
                        help="Actual-shot budget per episode; 0 disables it. The 10 s timeout still applies.")
    parser.add_argument("--reward_log_interval", type=int, default=10)
    parser.add_argument("--debug_diagnostics", action=argparse.BooleanOptionalAction, default=False,
                        help="Print environment camera/raycast diagnostics; disabled by default.")
    parser.add_argument("--save_best_after", type=int, default=None)
    parser.add_argument(
        "--locomotion_policy",
        type=str,
        required=True,
        help="Path to exported locomotion TorchScript policy (policy.pt).",
    )
    parser.add_argument(
        "--velocity_model",
        type=str,
        default=(
            "checkpoints/velocity/best_model.pt"
        ),
        help="Frozen velocity checkpoint; defaults to the full-episode-finetuned ordinary GRU.",
    )
    parser.add_argument("--velocity_backend", choices=("legacy_gru", "taq_gru"), default="legacy_gru")
    parser.add_argument(
        "--fine_tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OSTrack corrections in addition to LK tracking.",
    )
    parser.add_argument(
        "--ostrack_repo",
        type=str,
        default=str(_MODULE_DIR.parent / "test_lk_ostrack" / "ostrack" / "OSTrack-main"),
    )
    parser.add_argument(
        "--ostrack_checkpoint",
        type=str,
        default=(
            "checkpoints/OSTrack_ep0300.pth.tar"
        ),
    )
    parser.add_argument("--ostrack_config", type=str, default="vitb_256_mae_ce_32x4_ep300")
    parser.add_argument("--fine_tracker_policy", choices=("always", "interval", "adaptive"), default="adaptive")
    parser.add_argument("--fine_tracker_interval_frames", type=int, default=3)
    parser.add_argument("--camera_fx", type=float, default=None, help="Optional fx override; default reads Isaac camera K.")
    parser.add_argument("--camera_fy", type=float, default=None, help="Optional fy override; default reads Isaac camera K.")
    parser.add_argument("--camera_cx", type=float, default=None, help="Optional cx override; default reads Isaac camera K.")
    parser.add_argument("--camera_cy", type=float, default=None, help="Optional cy override; default reads Isaac camera K.")
    parser.add_argument("--min_velocity_observations", type=int, default=3)
    parser.add_argument("--min_velocity_history_s", type=float, default=0.10)
    parser.add_argument(
        "--min_bootstrap_box_px",
        type=float,
        default=8.0,
        help="Minimum width/height of the once-per-episode tracker initialization box.",
    )
    parser.add_argument(
        "--bridge_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print periodic full-chain perception readiness statistics.",
    )
    parser.add_argument("--target_speed_min", type=float, default=0.6, help="Minimum 3-D target speed magnitude (m/s).")
    parser.add_argument("--target_speed_max", type=float, default=5.0, help="Maximum 3-D target speed magnitude (m/s).")

    # 4-case mix probabilities: [SS, SD, DS, DD]
    parser.add_argument("--p_ss", type=float, default=0.3, help="P(robot static, target static)")
    parser.add_argument("--p_sd", type=float, default=0.3, help="P(robot static, target dynamic)")
    parser.add_argument("--p_ds", type=float, default=0.2, help="P(robot dynamic, target static)")
    parser.add_argument("--p_dd", type=float, default=0.2, help="P(robot dynamic, target dynamic)")
    parser.add_argument("--dog_vx_min", type=float, default=-1.5)
    parser.add_argument("--dog_vx_max", type=float, default=1.5)
    parser.add_argument("--dog_vy_min", type=float, default=-0.8)
    parser.add_argument("--dog_vy_max", type=float, default=0.8)
    parser.add_argument("--dog_wz_min", type=float, default=-0.8)
    parser.add_argument("--dog_wz_max", type=float, default=0.8)
    parser.add_argument("--dog_follow_speed_min", type=float, default=0.2)
    parser.add_argument("--dog_follow_speed_max", type=float, default=1.5)
    parser.add_argument("--dog_command_resample_min_s", type=float, default=0.75)
    parser.add_argument("--dog_command_resample_max_s", type=float, default=2.0)
    parser.add_argument("--dog_direction_noise_deg", type=float, default=20.0)
    parser.add_argument("--dog_lateral_noise_min", type=float, default=-0.2)
    parser.add_argument("--dog_lateral_noise_max", type=float, default=0.2)
    parser.add_argument("--dog_yaw_noise_min", type=float, default=-0.15)
    parser.add_argument("--dog_yaw_noise_max", type=float, default=0.15)
    parser.add_argument("--entropy_coef", type=float, default=0.003)
    parser.add_argument("--fire_intent_threshold", type=float, default=0.50)
    parser.add_argument("--fire_intent_hold_steps", type=int, default=1)
    parser.add_argument("--no_fire_timeout_penalty", type=float, default=-5.0)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional shared run directory name under runs/test7_2_weight/.",
    )

    args = parser.parse_args()
    ppo_settings = resolve_ppo_settings(args)

    full_chain = args.training_mode == "full_chain"

    if args.max_epochs <= 0:
        raise ValueError("--max_epochs must be > 0")
    if args.max_shots_per_episode < 0:
        raise ValueError("--max_shots_per_episode must be >= 0")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be > 0")
    default_case_mix = (0.3, 0.3, 0.2, 0.2)
    requested_case_mix = (float(args.p_ss), float(args.p_sd), float(args.p_ds), float(args.p_dd))
    if full_chain and requested_case_mix == default_case_mix:
        args.p_ss, args.p_sd, args.p_ds, args.p_dd = 0.4, 0.6, 0.0, 0.0
        print("[train-mixed][full_chain] default case mix changed to SS/SD = 0.4/0.6")
    if full_chain and (float(args.p_ds) > 0.0 or float(args.p_dd) > 0.0):
        raise ValueError(
            "The current velocity checkpoints support only SS/SD "
            "(static robot). Set --p_ds 0 --p_dd 0 and choose --p_ss/--p_sd."
        )
    if full_chain and args.fine_tracking and args.num_envs > 2:
        print(
            f"[train-mixed][full_chain] OSTrack enabled for {args.num_envs} envs with one model per env. "
            "There is no hard environment limit; reduce --num_envs if GPU memory runs out."
        )

    if full_chain:
        required_paths = [Path(args.velocity_model)]
        if args.fine_tracking:
            required_paths.extend((Path(args.ostrack_repo), Path(args.ostrack_checkpoint)))
        missing = [str(path) for path in required_paths if not path.expanduser().exists()]
        if missing:
            raise FileNotFoundError(f"Full-chain model path(s) not found: {missing}")

    local_rank, global_rank, world_size = _get_rank_info(bool(args.distributed))
    launcher_device = f"cuda:{local_rank}" if bool(args.distributed) else str(args.device)
    experiment_name = _resolve_experiment_name(
        requested_name=args.experiment_name,
        distributed=bool(args.distributed),
        global_rank=global_rank,
        world_size=world_size,
    )
    if full_chain and args.experiment_name is None:
        experiment_name = experiment_name.replace("turret_aim_mixed_", "turret_aim_full_chain_", 1)

    if bool(args.distributed):
        print(
            "[train-mixed][distributed] "
            f"rank={global_rank}/{world_size} local_rank={local_rank} device={launcher_device} "
            f"experiment={experiment_name}"
        )

    checkpoint_path = None
    if args.checkpoint is not None and str(args.checkpoint).strip() != "":
        checkpoint_path = _resolve_checkpoint_path(args.checkpoint, which=args.which)
        print(f"[train-mixed][{args.training_mode}] Resume from checkpoint: {checkpoint_path}")
    else:
        print(f"[train-mixed][{args.training_mode}] Train from scratch (no checkpoint).")

    app_launcher = AppLauncher(
        headless=args.headless,
        device=launcher_device,
        enable_cameras=full_chain,
    )
    sim_app = app_launcher.app

    try:
        import torch
        import gymnasium as gym

        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner

        from turret_aim_env_gt import TurretAimEnv, TurretAimEnvCfg
    except ModuleNotFoundError as exc:
        if "omni." in str(exc) or "omni" in str(exc):
            raise ModuleNotFoundError(
                "Isaac Sim modules were not found/initialized. Run this script via IsaacLab launcher."
            ) from exc
        raise

    if torch.cuda.is_available() and str(launcher_device).startswith("cuda"):
        torch.cuda.set_device(local_rank if bool(args.distributed) else torch.device(launcher_device).index or 0)

    env_cfg = TurretAimEnvCfg()
    env_cfg.max_shots_per_episode = args.max_shots_per_episode
    print(f"[train-mixed] episode_length_s={env_cfg.episode_length_s} "
          f"max_shots_per_episode={env_cfg.max_shots_per_episode} (0=unlimited)", flush=True)
    env_cfg.scene.num_envs = args.num_envs


    # 这里保留 use_fabric=False 也没问题
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "use_fabric"):
        env_cfg.sim.use_fabric = False
        print(f"[train-mixed][{args.training_mode}] env_cfg.sim.use_fabric = False")
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        env_cfg.sim.device = launcher_device
        print(f"[train-mixed] env_cfg.sim.device = {env_cfg.sim.device}")
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = 60 + int(global_rank)

    env_cfg.enable_locomotion_policy = True
    env_cfg.locomotion_policy_path = args.locomotion_policy
    env_cfg.locomotion_lin_vel_x_range = (float(args.dog_vx_min), float(args.dog_vx_max))
    env_cfg.locomotion_lin_vel_y_range = (float(args.dog_vy_min), float(args.dog_vy_max))
    env_cfg.locomotion_ang_vel_z_range = (float(args.dog_wz_min), float(args.dog_wz_max))
    env_cfg.enable_dynamic_target = True
    env_cfg.target_speed_range = (float(args.target_speed_min), float(args.target_speed_max))

    # Actor does not observe target acceleration, so both modes use constant
    # target velocity. This also matches the Stage1 ideal-v2 distribution.
    env_cfg.enable_air_drag = True
    env_cfg.target_acc_range = (0.0, 0.0)
    env_cfg.target_vertical_acc_range = (0.0, 0.0)
    env_cfg.fire_intent_threshold = float(args.fire_intent_threshold)
    env_cfg.fire_intent_hold_steps = int(args.fire_intent_hold_steps)
    env_cfg.r_no_fire_timeout_penalty = float(args.no_fire_timeout_penalty)

    env_cfg.debug_vis = not args.headless
    env_cfg.debug_diagnostics = bool(args.debug_diagnostics)
    # Headless full-chain training still needs a rendered target.
    env_cfg.render_target_sphere = full_chain or (not args.headless)
    env_cfg.enable_vision_supervision = full_chain
    env_cfg.enable_dynamic_target_raycast_debug = full_chain
    # In full-chain mode, sample range plus camera yaw/pitch directly. The
    # fixed target_height_range remains only as a camera-unavailable fallback.
    env_cfg.sample_target_in_camera_sector = full_chain
    if full_chain and env_cfg.dynamic_target_raycast_max_range < env_cfg.target_radius_range[1]:
        raise ValueError(
            "dynamic target raycaster range must cover the complete target sampling shell: "
            f"{env_cfg.dynamic_target_raycast_max_range} < {env_cfg.target_radius_range[1]}"
        )

    task_id = "Standalone-TurretAim-Mixed-Direct-v0"
    if task_id not in gym.registry:
        gym.register(id=task_id, entry_point=TurretAimEnv)

    env = gym.make(task_id, cfg=env_cfg, disable_env_checker=True)
    base_env: Any = env.unwrapped

    if full_chain:
        print("[train-mixed][full_chain] bootstrapping TiledCamera and raycaster")
        base_env.bootstrap_vision_camera()
    else:
        print("[train-mixed][GT-only] skip bootstrap_vision_camera()")

    _install_mixed_training_logic(
        base_env,
        case_probs=(args.p_ss, args.p_sd, args.p_ds, args.p_dd),
    )

    install_target_follow_motion(
        base_env,
        follow_speed=0.5,
        follow_speed_range=(float(args.dog_follow_speed_min), float(args.dog_follow_speed_max)),
        stop_speed_threshold=0.05,
        yaw_follow_gain=0.8,
        max_yaw_rate=max(abs(float(args.dog_wz_min)), abs(float(args.dog_wz_max))),
        blend=0.2,
        command_resample_s_range=(
            float(args.dog_command_resample_min_s),
            float(args.dog_command_resample_max_s),
        ),
        direction_noise_rad=math.radians(float(args.dog_direction_noise_deg)),
        lateral_noise_range=(float(args.dog_lateral_noise_min), float(args.dog_lateral_noise_max)),
        yaw_rate_noise_range=(float(args.dog_yaw_noise_min), float(args.dog_yaw_noise_max)),
        enable_debug=False,
    )

    rl_device = launcher_device if torch.cuda.is_available() and str(launcher_device).startswith("cuda") else "cpu"
    full_chain_bridge = None
    if full_chain:
        standalone_dir = Path(__file__).resolve().parents[1]
        if str(standalone_dir) not in sys.path:
            sys.path.insert(0, str(standalone_dir))
        from integrated_shooting_chain.isaac_training_bridge import (
            IsaacFullChainPerceptionBridge,
            IsaacTrainingBridgeConfig,
            install_training_perception_bridge,
        )

        bridge_config = IsaacTrainingBridgeConfig(
            velocity_checkpoint=Path(args.velocity_model).expanduser().resolve(),
            velocity_backend=args.velocity_backend,
            device=rl_device,
            camera_fx=args.camera_fx,
            camera_fy=args.camera_fy,
            camera_cx=args.camera_cx,
            camera_cy=args.camera_cy,
            target_radius_m=float(env_cfg.target_sphere_radius),
            min_bootstrap_box_px=float(args.min_bootstrap_box_px),
            min_velocity_observations=int(args.min_velocity_observations),
            min_velocity_history_s=float(args.min_velocity_history_s),
            enable_fine_tracking=bool(args.fine_tracking),
            ostrack_repo=Path(args.ostrack_repo).expanduser().resolve(),
            ostrack_checkpoint=Path(args.ostrack_checkpoint).expanduser().resolve(),
            ostrack_config=str(args.ostrack_config),
            fine_tracker_policy=str(args.fine_tracker_policy),
            fine_tracker_interval_frames=int(args.fine_tracker_interval_frames),
            debug=bool(args.bridge_debug),
        )
        full_chain_bridge = IsaacFullChainPerceptionBridge(base_env, bridge_config)
        install_training_perception_bridge(base_env, full_chain_bridge)
        print(
            f"[train-mixed][full_chain] RGB tracker + raycast range + {args.velocity_backend} installed; "
            "actor GT fallback is disabled."
        )
    else:
        # Do not mark the zero prediction buffers ready. GT-only obtains its
        # actor target state from the environment fallback path.
        base_env._gt_only_mode = True
        base_env._use_gt_fallback_for_missing_rl_pred = True
        base_env._rl_pred_ready[:] = False
        base_env._rl_bridge_training_ready[:] = False
        base_env._rl_bridge_velocity_ready[:] = False
        base_env._rl_pred_pos_base.zero_()
        base_env._rl_pred_vel_base.zero_()
        base_env._rl_pred_distance.zero_()
        base_env._rl_pred_yaw_base.zero_()
        base_env._rl_pred_pitch_base.zero_()
        print("[train-mixed][GT-only] perception bridge disabled; actor/critic use GT observations.")

    env = RlGamesVecEnvWrapper(
        base_env,
        rl_device,
        clip_obs=10.0,
        clip_actions=1.0,
        obs_groups={"obs": ["policy", "critic"], "states": ["critic"]},
        concate_obs_group=True,
    )

    class RewardLoggingObserver(IsaacAlgoObserver):
        def __init__(
            self,
            log_interval: int,
            env_ref: Any,
            experiment_name: str,
            ckpt_base_name: str,
            save_best_after: int,
            global_rank: int = 0,
        ):
            super().__init__()
            self._log_interval = max(1, log_interval)
            self._env_ref = env_ref
            self._experiment_name = experiment_name
            self._ckpt_base_name = ckpt_base_name
            self._save_best_after = max(0, int(save_best_after))
            self._is_primary = int(global_rank) == 0
            self._best_epoch_log_file = TRAIN_DIR / self._experiment_name / "best_checkpoint_epoch.txt"
            self._best_reward_seen = float("-inf")

        def _best_ckpt_path(self) -> Path:
            return TRAIN_DIR / self._experiment_name / "nn" / f"{self._ckpt_base_name}.pth"

        def _try_record_best_checkpoint_epoch(self, epoch_num: int) -> None:
            if not self._is_primary:
                return
            if epoch_num < self._save_best_after:
                return

            if not hasattr(self, "algo") or self.algo is None:
                return

            if self.algo.game_rewards.current_size == 0:
                return

            mean_rewards = self.algo.game_rewards.get_mean()
            curr_reward_raw = mean_rewards[0]
            curr_reward = float(curr_reward_raw.item() if hasattr(curr_reward_raw, "item") else curr_reward_raw)

            last_best_raw = self.algo.last_mean_rewards
            last_best = float(last_best_raw.item() if hasattr(last_best_raw, "item") else last_best_raw)

            if curr_reward > self._best_reward_seen:
                self._best_reward_seen = curr_reward

            if not (curr_reward > last_best):
                return

            best_ckpt = self._best_ckpt_path()
            self._best_epoch_log_file.parent.mkdir(parents=True, exist_ok=True)
            with self._best_epoch_log_file.open("w", encoding="utf-8") as f:
                f.write(f"epoch={int(epoch_num)} file={best_ckpt.name}\n")
                f.write(f"reward={curr_reward:.6f}\n")

            print(
                f"[train-mixed] new best | "
                f"epoch={int(epoch_num)} reward={curr_reward:.4f} file={best_ckpt.name}"
            )

        def after_print_stats(self, frame, epoch_num, total_time):
            self._try_record_best_checkpoint_epoch(int(epoch_num))
            super().after_print_stats(frame, epoch_num, total_time)

            if not self._is_primary:
                return

            perception_bridge = getattr(self._env_ref, "_full_chain_perception_bridge", None)
            perception = perception_bridge.get_stats() if perception_bridge is not None else None
            writer = getattr(self, "writer", None)
            if perception is not None and writer is not None:
                for key in ("visual_valid", "tracker_valid", "range_valid", "ready", "mean_gate"):
                    writer.add_scalar(
                        f"Perception/{key}",
                        float(perception.get(key, 0.0)),
                        int(frame),
                    )

            if epoch_num % self._log_interval != 0:
                return

            if self.algo is None or self.algo.game_rewards.current_size == 0:
                print(f"[train-mixed] epoch={epoch_num} mean_reward=n/a")
                return

            mean_reward = float(self.algo.game_rewards.get_mean()[0].item())
            mean_length = float(self.algo.game_lengths.get_mean().item())
            mix_stats = self._env_ref.get_mix_case_stats()

            lr = None
            if hasattr(self.algo, "optimizer") and self.algo.optimizer is not None:
                if len(self.algo.optimizer.param_groups) > 0:
                    lr = float(self.algo.optimizer.param_groups[0]["lr"])

            self._best_reward_seen = max(self._best_reward_seen, mean_reward)

            # ------------------------------------------------------------
            # 从环境 extras["log"] 里取自定义 episode 指标
            # 注意：这些值来自最近 reset 的 env batch，不是全历史平均。
            # ------------------------------------------------------------
            env_log = {}
            try:
                env_log = getattr(self._env_ref, "extras", {}).get("log", {})
            except Exception:
                env_log = {}

            def _log_float(key: str, default: float = 0.0) -> float:
                val = env_log.get(key, default)
                try:
                    if hasattr(val, "detach"):
                        return float(val.detach().mean().cpu().item())
                    if hasattr(val, "item"):
                        return float(val.item())
                    return float(val)
                except Exception:
                    return float(default)

            hit_rate = _log_float("Episode_Termination/hit_rate")
            timeout_rate = _log_float("Episode_Termination/time_out_rate")
            shot_limit_rate = _log_float("Episode_Termination/shot_budget_exhausted_rate")

            mean_fires = _log_float("Fire/mean_fires_per_episode")
            fired_episode_rate = _log_float("Fire/fired_episode_rate")
            one_fire_hit_rate = _log_float("Fire/one_fire_hit_rate")
            fire_min_dist_mean = _log_float("Fire/fire_min_dist_mean")
            fire_min_dist_min = _log_float("Fire/fire_min_dist_min")

            msg = (
                f"[train-mixed] epoch={int(epoch_num):05d} "
                f"reward={mean_reward:.4f} "
                f"ep_len={mean_length:.2f} "
                f"best={self._best_reward_seen:.4f} "
                f"hit={hit_rate:.3f} "
                f"timeout={timeout_rate:.3f} "
                f"shot_limit={shot_limit_rate:.3f} "
                f"fires={mean_fires:.2f} "
                f"fired_ep={fired_episode_rate:.3f} "
                f"one_fire_hit={one_fire_hit_rate:.3f} "
                f"fire_d_mean={fire_min_dist_mean:.3f} "
                f"fire_d_min={fire_min_dist_min:.3f} "
                f"mix=({mix_stats['ss']:.2f}/{mix_stats['sd']:.2f}/{mix_stats['ds']:.2f}/{mix_stats['dd']:.2f})"
            )

            if perception is not None:
                msg += (
                    f" visual_valid={perception.get('visual_valid', 0.0):.3f}"
                    f" perception_ready={perception.get('ready', 0.0):.3f}"
                    f" tracker_valid={perception.get('tracker_valid', 0.0):.3f}"
                    f" range_valid={perception.get('range_valid', 0.0):.3f}"
                    f" velocity_gate={perception.get('mean_gate', 0.0):.3f}"
                )

            if lr is not None:
                msg += f" lr={lr:.2e}"

            if int(epoch_num) < self._save_best_after:
                msg += f" best_save_wait={self._save_best_after - int(epoch_num)}"

            print(msg)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    rollout_batch_size = int(env.unwrapped.num_envs) * args.horizon_length
    print(
        f"[train-mixed] PPO rollout_batch={rollout_batch_size} "
        f"minibatch={args.minibatch_size} seq_length={args.seq_length} "
        f"mini_epochs={args.mini_epochs} learning_rate={args.learning_rate:.6g} "
        f"lr_schedule={args.lr_schedule} reset_learning_rate={args.reset_learning_rate}"
    )
    if global_rank == 0:
        settings_path = TRAIN_DIR / experiment_name / "training_settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps({"arguments": vars(args), "ppo": ppo_settings}, indent=2) + "\n")

    config = {
        "params": {
            "seed": 60 + int(global_rank),
            "algo": {
                "name": "a2c_continuous",
            },
            "model": {
                "name": "continuous_a2c_tanh",
            },
            "network": {
                "name": "asym_central_value_gru",
                "actor_target_dim": 9,
                "actor_robot_dim": 19,
                "critic_target_dim": 12,
                "critic_robot_dim": 19,
                "branch_hidden_dim": 64,
                "actor_rnn_hidden_dim": 128,
                "critic_rnn_hidden_dim": 128,
                "head_hidden_dim": 64,
                "fixed_sigma": True,
            },
            "load_checkpoint": checkpoint_path is not None,
            "load_path": str(checkpoint_path) if checkpoint_path is not None else "",
            "config": {
                "name": "turret_aim",
                "train_dir": str(TRAIN_DIR),
                "full_experiment_name": experiment_name,
                "env_name": "rlgpu",
                "has_central_value": False,
                "num_actors": env.unwrapped.num_envs,
                "env_config": {},
                "device": rl_device,
                "device_name": rl_device,
                "multi_gpu": bool(args.distributed),
                "ppo": True,
                "reward_shaper": {
                    "scale_value": 1.0,
                    "shift_value": 0.0,
                },
                "bounds_loss_coef": 0.0001,
                **ppo_settings,
                "gamma": 0.99,
                "tau": 0.95,
                "clip_value": True,
                "critic_coef": 2.0,
                "max_epochs": int(args.max_epochs),
                "score_to_win": 100000000,
                "normalize_advantage": True,
                "truncate_grads": True,
                "normalize_input": True,
                "normalize_value": True,
                "zero_rnn_on_done": True,
            },
        }
    }

    runner = Runner(
        RewardLoggingObserver(
            args.reward_log_interval,
            base_env,
            experiment_name,
            "turret_aim",
            args.save_best_after,
            global_rank,
        )
    )
    runner.load(config)
    from rl_games.algos_torch.a2c_continuous import A2CAgent

    training_agent = make_learning_rate_restore_agent(A2CAgent, reset_learning_rate=args.reset_learning_rate)
    runner.algo_factory.register_builder("a2c_continuous", lambda **kwargs: training_agent(**kwargs))

    run_args: dict[str, Any] = {
        "train": True,
        "play": False,
    }

    if checkpoint_path is not None:
        run_args["checkpoint"] = str(checkpoint_path)

    runner.run(run_args)

    sim_app.close()


if __name__ == "__main__":
    main()

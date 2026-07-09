import argparse
from datetime import datetime
from pathlib import Path
from typing import Any
import asym_central_value_gru_network
from isaaclab.app import AppLauncher
from target_follow_motion import install_target_follow_motion
import types
from rl_perception_bridge import install_rl_perception_bridge


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
    sd_fire_delay_s: float = 0.0,
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
    base_env._mix_sd_fire_delay_steps = max(0, int(round(float(sd_fire_delay_s) / float(base_env.step_dt))))
    base_env._mix_fire_delay_left_steps = torch.zeros(
        (base_env.num_envs,),
        dtype=torch.long,
        device=base_env.device,
    )

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

        sd_mask = case_ids == 1
        base_env._mix_fire_delay_left_steps[env_ids_t] = torch.where(
            sd_mask,
            torch.full_like(case_ids, int(base_env._mix_sd_fire_delay_steps)),
            torch.zeros_like(case_ids),
        )

        if torch.any(robot_static):
            static_ids = env_ids_t[robot_static]

            # 静止狗：仍然使用 locomotion policy，
            # 但把 command 固定为 0，不再随机重采样，不再强制写死腿部目标。
            base_env._aim_warmup_left_steps[static_ids] = 0
            base_env._locomotion_commands[static_ids] = 0.0
            base_env._locomotion_command_time_left[static_ids] = base_env.cfg.locomotion_command_resample_s

            # reset 时把上一帧动作清零，避免跨 episode 残留。
            # 注意：这里只在 reset 时清零，step 内不要再清零，
            # 否则就不是“零指令下跑 locomotion 模型”了。
            base_env._locomotion_last_actions[static_ids] = 0.0

        if torch.any(target_static):
            static_target_ids = env_ids_t[target_static]
            base_env._target_vel_w[static_target_ids] = 0.0
            if hasattr(base_env, "_target_acc_w"):
                base_env._target_acc_w[static_target_ids] = 0.0

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
        # 静止机器狗仍然走 locomotion policy，
        # 只是 command 固定为 0，且不允许被随机重采样。
        robot_static_mask = ~self._mix_robot_dynamic
        if torch.any(robot_static_mask):
            self._aim_warmup_left_steps[robot_static_mask] = 0
            self._locomotion_commands[robot_static_mask] = 0.0

            # 关键：
            # 不能设成 0，否则 original_pre_physics_step -> _update_locomotion_commands()
            # 会立刻把这些静止狗重采样成随机速度指令。
            self._locomotion_command_time_left[robot_static_mask] = self.cfg.locomotion_command_resample_s

        # 让原始逻辑正常执行：
        # _update_locomotion_commands()
        # _compute_locomotion_observations()
        # locomotion_policy(obs)
        # _locomotion_joint_targets = default + scale * action
        actions_for_step = actions
        sd_fire_delay_mask = self._mix_fire_delay_left_steps > 0
        if torch.any(sd_fire_delay_mask):
            actions_for_step = actions.clone()
            actions_for_step[sd_fire_delay_mask, 2] = -1.0

        original_pre_physics_step(actions_for_step)

        if torch.any(sd_fire_delay_mask):
            self._mix_fire_delay_left_steps[sd_fire_delay_mask] = torch.clamp(
                self._mix_fire_delay_left_steps[sd_fire_delay_mask] - 1,
                min=0,
            )

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
    print(
        "[train-mixed] mixed-case logic installed | "
        f"sd_fire_delay_s={float(sd_fire_delay_s):.3f} "
        f"steps={int(base_env._mix_sd_fire_delay_steps)}"
    )


def _parse_step_tuple(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    text = str(value).strip()
    if text == "":
        return ()

    steps: list[int] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if item == "":
            continue
        steps.append(int(item))
    return tuple(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional: existing checkpoint file or run directory. Leave empty to train from scratch.",
    )
    parser.add_argument("--which", type=str, default="best", choices=["best", "last", "latest"])
    parser.add_argument("--max_epochs", type=int, default=15000)
    parser.add_argument("--reward_log_interval", type=int, default=10)
    parser.add_argument("--save_best_after", type=int, default=11500)
    parser.add_argument(
        "--locomotion_policy",
        type=str,
        required=True,
        help="Path to exported locomotion TorchScript policy (policy.pt).",
    )
    parser.add_argument(
        "--vision_model",
        type=str,
        default="source/standalone/test4/vision_model/vision_runs/vision_runs_cnn_only/04-27-17-54/vision_best.pth",
        help="Path to vision model checkpoint used by rl_perception_bridge.",
    )
    parser.add_argument(
        "--velocity_model",
        type=str,
        default="",
        help="Path to velocity GRU checkpoint used by rl_perception_bridge. Required when bridge is enabled.",
    )
    parser.add_argument(
        "--velocity_model_def",
        type=str,
        default="source/standalone/test11/speed_prediction_model.py",
        help=(
            "Optional path to speed_prediction_model.py that defines VelocityPredictorGRU. "
            "Defaults to the renamed test11 directory."
        ),
    )
    parser.add_argument(
        "--enable_rl_perception_bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to install rl_perception_bridge into the environment.",
    )
    parser.add_argument(
        "--bridge_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable periodic debug print from rl_perception_bridge.",
    )
    parser.add_argument(
        "--bridge_use_gt_distance_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow bridge to fallback to GT distance when predicted/lidar distance is unavailable. Disabled by default for strict camera+raycast training.",
    )
    parser.add_argument(
        "--bridge_use_gt_velocity_until_buffer_ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow bridge to fallback to GT velocity until GRU sequence buffer is ready. Disabled by default for strict velocity-module training.",
    )
    parser.add_argument(
        "--bridge_velocity_sequence_stride",
        type=int,
        default=3,
        help="Env-step stride for velocity-GRU sequence samples. Default matches test11 collection.",
    )
    parser.add_argument(
        "--bridge_disable_env_gt_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable env-side GT fallback for missing RL predictions (strict module-only mode).",
    )
    parser.add_argument(
        "--camera_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable camera pose diagnostics, debug image dumps, and raycast pose comparison.",
    )
    parser.add_argument(
        "--camera_debug_steps",
        type=str,
        default="20,50,100,200",
        help="Comma-separated sim steps where camera debug images/metadata are dumped. Empty disables dumps.",
    )
    parser.add_argument(
        "--camera_debug_env_id",
        type=int,
        default=0,
        help="Environment id used for camera debug image dumps and pose checks.",
    )
    parser.add_argument(
        "--camera_debug_dir",
        type=str,
        default="debug_camera_frames_test7_1",
        help="Directory for camera debug PNG/TXT dumps.",
    )
    parser.add_argument(
        "--camera_debug_raycast_every",
        type=int,
        default=20,
        help="Print camera/raycast pose comparison every N raycast debug updates.",
    )
    parser.add_argument("--target_speed_min", type=float, default=0.6)
    parser.add_argument("--target_speed_max", type=float, default=5)
    parser.add_argument("--target_vz_min", type=float, default=-0.5)
    parser.add_argument("--target_vz_max", type=float, default=0.5)

    # 4-case mix probabilities: [SS, SD, DS, DD]
    parser.add_argument("--p_ss", type=float, default=0.5, help="P(robot static, target static)")
    parser.add_argument("--p_sd", type=float, default=0.5, help="P(robot static, target dynamic)")
    parser.add_argument("--p_ds", type=float, default=0, help="P(robot dynamic, target static)")
    parser.add_argument("--p_dd", type=float, default=0, help="P(robot dynamic, target dynamic)")
    parser.add_argument(
        "--sd_fire_delay_s",
        type=float,
        default=2.0,
        help="In SD episodes, let the target move for this many seconds before fire is allowed.",
    )

    args = parser.parse_args()

    if args.max_epochs <= 0:
        raise ValueError("--max_epochs must be > 0")

    checkpoint_path = None
    if args.checkpoint is not None and str(args.checkpoint).strip() != "":
        checkpoint_path = _resolve_checkpoint_path(args.checkpoint, which=args.which)
        print(f"[train-mixed] Resume from checkpoint: {checkpoint_path}")
    else:
        print("[train-mixed] Train from scratch (no checkpoint).")

    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=bool(args.enable_cameras),
    )
    sim_app = app_launcher.app

    try:
        import torch
        import gymnasium as gym

        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner

        from turret_aim_env import TurretAimEnv, TurretAimEnvCfg
    except ModuleNotFoundError as exc:
        if "omni." in str(exc) or "omni" in str(exc):
            raise ModuleNotFoundError(
                "Isaac Sim modules were not found/initialized. Run this script via IsaacLab launcher."
            ) from exc
        raise

    env_cfg = TurretAimEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    # -----------------------------
    # 显式启用 Fabric，加速并行环境的物理状态同步。
    # TiledCamera 不再读取自身 pose，因此不会进入旧的
    # XFormPrimView/usdrt.hierarchy 相机位姿路径。
    # -----------------------------
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "use_fabric"):
        env_cfg.sim.use_fabric = True
        print("[train-mixed] env_cfg.sim.use_fabric = True")

    # 为了覆盖四种情况，底层配置需要同时支持移动底盘 + 动态目标。
    env_cfg.training_stage = 2
    env_cfg.enable_locomotion_policy = True
    env_cfg.locomotion_policy_path = args.locomotion_policy
    env_cfg.enable_dynamic_target = True
    env_cfg.target_speed_range = (float(args.target_speed_min), float(args.target_speed_max))
    env_cfg.target_vertical_speed_range = (float(args.target_vz_min), float(args.target_vz_max))
    env_cfg.enable_air_drag = False
    env_cfg.target_acc_range = (0.0, 0.0)
    env_cfg.target_vertical_acc_range = (0.0, 0.0)
    env_cfg.debug_vis = not args.headless

    if args.enable_rl_perception_bridge:
        env_cfg.enable_vision_supervision = True
        env_cfg.vision_history_length = 1
        env_cfg.enable_dynamic_target_raycast_debug = True
        env_cfg.dynamic_target_raycast_debug_every = max(
            1,
            int(args.camera_debug_raycast_every) if args.camera_debug else 10**9,
        )
        print(
            "[train-mixed] perception bridge sensors enabled | "
            "camera=True raycast=True "
            f"raycast_debug={'on' if args.camera_debug else 'off'}"
        )

    if args.camera_debug:
        camera_debug_steps = _parse_step_tuple(args.camera_debug_steps)
        env_cfg.debug_diagnostics = True
        env_cfg.dump_debug_camera_steps = camera_debug_steps
        env_cfg.dump_debug_camera_env_id = int(args.camera_debug_env_id)
        env_cfg.dump_debug_camera_dir = str(args.camera_debug_dir)
        env_cfg.enable_dynamic_target_raycast_debug = True
        env_cfg.dynamic_target_raycast_debug_every = max(
            1,
            int(args.camera_debug_raycast_every),
        )
        print(
            "[train-mixed] camera debug enabled | "
            f"steps={camera_debug_steps} "
            f"env_id={env_cfg.dump_debug_camera_env_id} "
            f"dir={env_cfg.dump_debug_camera_dir} "
            f"raycast_every={env_cfg.dynamic_target_raycast_debug_every}"
        )

    task_id = "Standalone-TurretAim-Mixed-Direct-v0"
    if task_id not in gym.registry:
        gym.register(id=task_id, entry_point=TurretAimEnv)

    env = gym.make(task_id, cfg=env_cfg, disable_env_checker=True)
    base_env: Any = env.unwrapped

    # ---------------------------------------------------------
    # one-shot bootstrap for delayed TiledCamera initialization
    # 必须在 RL 第一次 env.reset() 之前完成
    # ---------------------------------------------------------
    if hasattr(base_env, "bootstrap_vision_camera"):
        base_env.bootstrap_vision_camera()

    _install_mixed_training_logic(
        base_env,
        case_probs=(args.p_ss, args.p_sd, args.p_ds, args.p_dd),
        sd_fire_delay_s=float(args.sd_fire_delay_s),
    )

    install_target_follow_motion(
        base_env,
        follow_speed=0.5,
        stop_speed_threshold=0.05,
        yaw_follow_gain=0.8,
        max_yaw_rate=0.4,
        blend=0.2,
        enable_debug=False,
    )

    # ---------------------------------------------------------
    # Install RL perception bridge
    # 放在 base_env 创建后、vec wrapper 之前。
    # bridge 会在每次组装 actor/critic obs 之前，把
    # vision + distance + velocity 的预测结果写回 env。
    # ---------------------------------------------------------
    if args.enable_rl_perception_bridge:
        if not args.vision_model.strip():
            raise ValueError(
                "--enable_rl_perception_bridge is True, but --vision_model is empty."
            )
        if not args.velocity_model.strip():
            raise ValueError(
                "--enable_rl_perception_bridge is True, but --velocity_model is empty. "
                "Pass the velocity GRU checkpoint trained with test11/train_gru_velocity.py."
            )

        bridge = install_rl_perception_bridge(
            base_env=base_env,
            vision_model_path=args.vision_model,
            velocity_model_path=args.velocity_model if args.velocity_model.strip() else None,
            velocity_model_def_path=args.velocity_model_def if args.velocity_model_def.strip() else None,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_gt_distance_fallback=args.bridge_use_gt_distance_fallback,
            use_gt_velocity_until_buffer_ready=args.bridge_use_gt_velocity_until_buffer_ready,
            velocity_sequence_stride=args.bridge_velocity_sequence_stride,
            debug=(args.bridge_debug or args.camera_debug),
        )

        # 可选：严格模式下，不允许 env 在“预测缺失”时自行回退到 GT
        # 这样 policy 看到的就是纯 bridge 写回结果；未 ready 时由 env 内部 obs 逻辑处理。
        if args.bridge_disable_env_gt_fallback:
            base_env._use_gt_fallback_for_missing_rl_pred = False
            print("[train_turret_mixed] disabled env GT fallback for missing RL predictions.")


        # 2) 不再把 bridge.step() 挂到 _get_observations()
        #    改为在 env.step(action) 完成后统一执行一次：
        #    - 避免在观测收集阶段强制触发视觉刷新/渲染同步
        #    - 保持 rl_perception_bridge.py 中“after env.step()”的设计语义
        orig_step = base_env.step

        def patched_step(self, actions):
            out = orig_step(actions)

            if not (hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None):
                return out

            # env.step() 完成后，基于当前 step 已渲染好的视觉缓存跑一次 perception bridge
            self._rl_perception_bridge.step()

            # 用 bridge 写回后的预测结果重新组装 obs，返回给策略
            obs = self._get_observations()

            if isinstance(out, tuple):
                if len(out) == 5:
                    _, rew, terminated, truncated, extras = out
                    return obs, rew, terminated, truncated, extras
                if len(out) == 4:
                    _, rew, dones, extras = out
                    return obs, rew, dones, extras

            raise RuntimeError(
                f"[train_turret_mixed] Unexpected env.step() return format: type={type(out)}"
            )

        base_env.step = types.MethodType(patched_step, base_env)

        print("[train_turret_mixed] rl_perception_bridge hooked into step() and _reset_idx.")


    rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = RlGamesVecEnvWrapper(
        base_env,
        rl_device,
        clip_obs=10.0,
        clip_actions=1.0,
        obs_groups={"obs": ["policy", "critic"], "states": ["critic"]},
        concate_obs_group=True,
    )

    experiment_name = f"turret_aim_mixed_{datetime.now().strftime('%m-%d-%H-%M-%S')}"

    class RewardLoggingObserver(IsaacAlgoObserver):
        def __init__(
            self,
            log_interval: int,
            env_ref: Any,
            experiment_name: str,
            ckpt_base_name: str,
            save_best_after: int,
        ):
            super().__init__()
            self._log_interval = max(1, log_interval)
            self._env_ref = env_ref
            self._experiment_name = experiment_name
            self._ckpt_base_name = ckpt_base_name
            self._save_best_after = max(0, int(save_best_after))
            self._best_epoch_log_file = Path("runs") / self._experiment_name / "best_checkpoint_epoch.txt"
            self._best_reward_seen = float("-inf")

        def _best_ckpt_path(self) -> Path:
            return Path("runs") / self._experiment_name / "nn" / f"{self._ckpt_base_name}.pth"

        def _try_record_best_checkpoint_epoch(self, epoch_num: int) -> None:
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
                f"fires={mean_fires:.2f} "
                f"fired_ep={fired_episode_rate:.3f} "
                f"one_fire_hit={one_fire_hit_rate:.3f} "
                f"fire_d_mean={fire_min_dist_mean:.3f} "
                f"fire_d_min={fire_min_dist_min:.3f} "
                f"mix=({mix_stats['ss']:.2f}/{mix_stats['sd']:.2f}/{mix_stats['ds']:.2f}/{mix_stats['dd']:.2f})"
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

    config = {
        "params": {
            "seed": 60,
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
                "full_experiment_name": experiment_name,
                "env_name": "rlgpu",
                "has_central_value": False,
                "num_actors": env.unwrapped.num_envs,
                "env_config": {},
                "device": rl_device,
                "device_name": rl_device,
                "ppo": True,
                "reward_shaper": {
                    "scale_value": 1.0,
                    "shift_value": 0.0,
                },
                "bounds_loss_coef": 0.0001,
                "horizon_length": 32,
                "minibatch_size": 2048,
                "mini_epochs": 4,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 5e-5,
                "clip_value": True,
                "entropy_coef": 0.0,
                "critic_coef": 2.0,
                "e_clip": 0.1,
                "max_epochs": int(args.max_epochs),
                "save_best_after": int(args.save_best_after),
                "save_frequency": 0,
                "score_to_win": 100000000,
                "lr_schedule": "adaptive",
                "kl_threshold": 0.01,
                "normalize_advantage": True,
                "grad_norm": 0.5,
                "truncate_grads": True,
                "normalize_input": True,
                "normalize_value": True,
                "seq_length": 8,
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
        )
    )
    runner.load(config)

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

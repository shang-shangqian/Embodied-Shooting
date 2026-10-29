from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import  asym_central_value_gru_network_gt  # noqa: F401
from isaaclab.app import AppLauncher
from target_follow_motion_gt import install_target_follow_motion
from yaw_control import checkpoint_contract, contracts_match, observation_contract
import types

from rl_perception_bridge_gt import install_rl_perception_bridge


def _install_tanh_deterministic_player_patch() -> None:
    """Use the tanh-squashed policy mean for deterministic RL-Games evaluation."""
    import torch

    from rl_games.algos_torch import players
    from rl_games.common.tr_helpers import unsqueeze_obs

    player_cls = players.PpoPlayerContinuous
    if getattr(player_cls, "_test7_2_tanh_deterministic_patch", False):
        return

    def get_action(self, obs, is_deterministic=False):
        if not self.has_batch_dimension:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)

        mu = res_dict["mus"]
        sampled_action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        current_action = torch.tanh(mu) if is_deterministic else sampled_action

        if not self.has_batch_dimension:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return players.rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        return current_action

    player_cls.get_action = get_action
    player_cls._test7_2_tanh_deterministic_patch = True


def _install_runtime_track_draw(
    base_env: Any,
    track_env_id: int,
    max_points: int,
    shot_history: int,
    ballistic_draw_stride: int,
    robot_color: tuple[float, float, float, float],
    target_color: tuple[float, float, float, float],
    bullet_color: tuple[float, float, float, float],
    robot_width: float,
    target_width: float,
    bullet_width: float,
    pause_between_episodes: bool,
    pause_seconds: float,
) -> bool:
    """在不修改环境源码的前提下，为 TurretAimEnv 动态挂接轨迹画线逻辑。"""
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    except ModuleNotFoundError:
        print("[eval] Debug draw extension unavailable. Skip trajectory lines.")
        return False

    if not hasattr(base_env, "_pre_physics_step") or not hasattr(base_env, "_reset_idx"):
        print("[eval] Env does not expose expected hooks. Skip trajectory lines.")
        return False

    draw = omni_debug_draw.acquire_debug_draw_interface()
    if draw is None:
        print("[eval] Failed to acquire debug draw interface. Skip trajectory lines.")
        return False

    base_env._track_draw_interface = draw
    base_env._track_env_id = int(track_env_id)
    base_env._track_max_points = max(2, int(max_points))
    base_env._track_shot_history = max(1, int(shot_history))
    base_env._track_ballistic_stride = max(1, int(ballistic_draw_stride))
    base_env._track_robot_color = [float(v) for v in robot_color]
    base_env._track_target_color = [float(v) for v in target_color]
    base_env._track_bullet_color = [float(v) for v in bullet_color]
    base_env._track_robot_width = float(robot_width)
    base_env._track_target_width = float(target_width)
    base_env._track_bullet_width = float(bullet_width)
    base_env._track_robot_points = []
    base_env._track_target_points = []
    base_env._track_bullet_polylines = []
    base_env._pause_between_episodes = bool(pause_between_episodes)
    base_env._pause_seconds = max(0.0, float(pause_seconds))

    # 新增：记录 tracked env 上一帧是否仍处于“占位目标阶段”
    base_env._track_prev_target_pending = False

    original_pre_physics_step = base_env._pre_physics_step
    original_reset_idx = base_env._reset_idx

    def _append_point(
        point_list: list[list[float]],
        point: list[float],
        keep: int,
        reset_jump_threshold: float = 1.0,   # 原来 3.0，改小一点更稳
    ):
        if len(point_list) > 0:
            last = point_list[-1]
            dx = float(point[0] - last[0])
            dy = float(point[1] - last[1])
            dz = float(point[2] - last[2])
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist > reset_jump_threshold:
                point_list.clear()

        point_list.append(point)
        if len(point_list) > keep:
            del point_list[: len(point_list) - keep]

    def _append_polyline(polyline: list[list[float]]):
        if len(polyline) < 2:
            return
        base_env._track_bullet_polylines.append(polyline)
        if len(base_env._track_bullet_polylines) > base_env._track_shot_history:
            del base_env._track_bullet_polylines[: len(base_env._track_bullet_polylines) - base_env._track_shot_history]

    def _build_bullet_polyline(env_id: int) -> list[list[float]]:
        muzzle_pos_w, _, muzzle_fwd_w = base_env._compute_turret_frame()

        if base_env._muzzle_body_id is not None:
            body_pos_w = base_env.turret.data.body_pos_w[:, base_env._muzzle_body_id]
        else:
            body_pos_w = base_env.turret.data.root_pos_w

        muzzle_vel_w = base_env._compute_muzzle_velocity_w(
            body_pos_w=body_pos_w,
            muzzle_pos_w=muzzle_pos_w,
        )

        p0 = muzzle_pos_w[env_id : env_id + 1]
        v0 = muzzle_vel_w[env_id : env_id + 1] + muzzle_fwd_w[env_id : env_id + 1] * float(base_env.cfg.muzzle_speed)

        polyline: list[list[float]] = []

        if not bool(base_env.cfg.enable_air_drag):
            times = base_env._ballistic_time_samples.to(device=p0.device, dtype=p0.dtype)
            gravity_w = base_env._gravity_w.to(device=p0.device, dtype=p0.dtype).view(1, 3)

            for i, t in enumerate(times):
                if (i % base_env._track_ballistic_stride) != 0 and i != len(times) - 1:
                    continue

                tt = t.view(1, 1)
                bullet_pos = p0 + v0 * tt + 0.5 * gravity_w * tt * tt
                polyline.append(bullet_pos.squeeze(0).detach().cpu().tolist())

            return polyline

        # 备用：如果以后手动打开空气阻力，则使用数值积分画轨迹。
        bullet_pos = p0.clone()
        bullet_vel = v0.clone()

        polyline.append(bullet_pos.squeeze(0).detach().cpu().tolist())

        gravity_acc = base_env._gravity_w.to(
            device=bullet_pos.device,
            dtype=bullet_pos.dtype,
        ).unsqueeze(0)

        for i, dt in enumerate(base_env._ballistic_dt_samples):
            dt = dt.to(device=bullet_pos.device, dtype=bullet_pos.dtype).view(1, 1)

            acc0 = gravity_acc + base_env._compute_air_drag_accel_w(bullet_vel)

            vel_mid = bullet_vel + 0.5 * acc0 * dt

            acc_mid = gravity_acc + base_env._compute_air_drag_accel_w(vel_mid)

            bullet_pos = bullet_pos + vel_mid * dt
            bullet_vel = bullet_vel + acc_mid * dt

            if (i % base_env._track_ballistic_stride) == 0:
                polyline.append(bullet_pos.squeeze(0).detach().cpu().tolist())

        end_point = bullet_pos.squeeze(0).detach().cpu().tolist()
        if polyline[-1] != end_point:
            polyline.append(end_point)

        return polyline

    def _polyline_to_segments(points: list[list[float]], color: list[float], width: float, starts, ends, colors, widths):
        if len(points) < 2:
            return
        for p0, p1 in zip(points[:-1], points[1:]):
            starts.append(p0)
            ends.append(p1)
            colors.append(color)
            widths.append(width)

    def _draw_all_tracks():
        env_id = base_env._track_env_id
        if env_id < 0 or env_id >= int(base_env.num_envs):
            return

        robot_pos = base_env.turret.data.root_pos_w[env_id, :3].detach().cpu().tolist()
        _append_point(base_env._track_robot_points, [float(v) for v in robot_pos], base_env._track_max_points)

        # -----------------------------
        # 关键修复：占位目标阶段不记录 target 轨迹
        # True -> False 的那个瞬间先 clear，再开始记录正式目标
        # -----------------------------
        target_pending = False
        if hasattr(base_env, "_pending_target_resample"):
            target_pending = bool(base_env._pending_target_resample[env_id].item())

        prev_pending = bool(getattr(base_env, "_track_prev_target_pending", False))

        if prev_pending and (not target_pending):
            # 刚从占位目标切到正式目标，清掉旧 target 轨迹，避免连线
            base_env._track_target_points.clear()

        base_env._track_prev_target_pending = target_pending

        if not target_pending:
            target_pos = base_env._target_pos_w[env_id, :3].detach().cpu().tolist()
            _append_point(
                base_env._track_target_points,
                [float(v) for v in target_pos],
                base_env._track_max_points,
            )

        fire_now = base_env._fire_command.bool()
        if bool(fire_now[env_id].item()):
            import torch

            is_hit_fire = False
            if hasattr(base_env, "_get_cached_ballistic_metrics"):
                _, ballistic_hit, _ = base_env._get_cached_ballistic_metrics()
                dist = torch.norm(base_env._target_pos_w - base_env.turret.data.root_pos_w[:, :3], dim=-1)
                if hasattr(base_env, "_aim_warmup_left_steps"):
                    aiming_enabled = base_env._aim_warmup_left_steps <= 0
                else:
                    aiming_enabled = torch.ones_like(fire_now, dtype=torch.bool)
                hit_now = ballistic_hit & (dist < float(base_env.cfg.max_hit_distance)) & aiming_enabled
                is_hit_fire = bool(hit_now[env_id].item())

            if is_hit_fire:
                base_env._track_bullet_polylines.clear()
                _append_polyline(_build_bullet_polyline(env_id))

        starts: list[list[float]] = []
        ends: list[list[float]] = []
        colors: list[list[float]] = []
        widths: list[float] = []

        _polyline_to_segments(
            base_env._track_robot_points,
            base_env._track_robot_color,
            base_env._track_robot_width,
            starts,
            ends,
            colors,
            widths,
        )
        _polyline_to_segments(
            base_env._track_target_points,
            base_env._track_target_color,
            base_env._track_target_width,
            starts,
            ends,
            colors,
            widths,
        )
        for line in base_env._track_bullet_polylines:
            _polyline_to_segments(
                line,
                base_env._track_bullet_color,
                base_env._track_bullet_width,
                starts,
                ends,
                colors,
                widths,
            )

        base_env._track_draw_interface.clear_lines()
        if len(starts) > 0:
            base_env._track_draw_interface.draw_lines(starts, ends, colors, widths)

    def patched_pre_physics_step(self, actions):
        original_pre_physics_step(actions)
        try:
            _draw_all_tracks()
        except Exception as exc:
            if not hasattr(self, "_track_draw_warned"):
                print(f"[eval] trajectory draw disabled due to runtime error: {exc}")
                self._track_draw_warned = True

    def patched_reset_idx(self, env_ids):
        should_pause = False
        hit_on_track_env = False
        try:
            if self._pause_between_episodes:
                import torch

                if env_ids is None:
                    env_ids_t = torch.arange(int(self.num_envs), device=self.device, dtype=torch.long)
                else:
                    env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

                contains_track_env = bool(torch.any(env_ids_t == self._track_env_id).item())
                if contains_track_env:
                    if hasattr(self, "_hit_flag"):
                        hit_on_track_env = bool(self._hit_flag[self._track_env_id].item())
                    if hasattr(self, "episode_length_buf"):
                        should_pause = bool(torch.any(self.episode_length_buf[env_ids_t] > 0).item())
                    else:
                        should_pause = bool(getattr(self, "common_step_counter", 0) > 0)
        except Exception:
            should_pause = False
            hit_on_track_env = False

        if should_pause and self._pause_seconds > 0.0:
            try:
                if hit_on_track_env:
                    self._track_bullet_polylines.clear()
                    _append_polyline(_build_bullet_polyline(self._track_env_id))
                _draw_all_tracks()
                if hasattr(self, "sim"):
                    self.sim.render()
            except Exception:
                pass
            print(f"[eval] Episode finished. Hold frame {self._pause_seconds:.2f}s before reset...")
            start_t = time.time()
            while (time.time() - start_t) < self._pause_seconds:
                try:
                    if hasattr(self, "sim"):
                        self.sim.render()
                except Exception:
                    pass
                time.sleep(1.0 / 60.0)

        result = original_reset_idx(env_ids)
        try:
            env_ids_t = env_ids
            if env_ids_t is None:
                reset_all = True
            else:
                import torch
                env_ids_t = torch.as_tensor(env_ids_t, device=self.device, dtype=torch.long)
                reset_all = bool(torch.any(env_ids_t == self._track_env_id).item())

            if reset_all:
                self._track_robot_points.clear()
                self._track_target_points.clear()
                self._track_bullet_polylines.clear()
                self._track_prev_target_pending = False
                self._track_draw_interface.clear_lines()
        except Exception:
            pass
        return result

    import types
    base_env._pre_physics_step = types.MethodType(patched_pre_physics_step, base_env)
    base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)
    print(f"[eval] Trajectory line drawing enabled (track env id: {base_env._track_env_id}).")
    return True


def _pick_checkpoint_from_dir(run_dir: Path, which: str) -> Path:
    nn_dir = run_dir / "nn"
    search_dirs = [nn_dir, run_dir]

    all_pths: list[Path] = []
    for directory in search_dirs:
        if directory.is_dir():
            all_pths.extend(sorted(directory.glob("*.pth")))

    if len(all_pths) == 0:
        raise FileNotFoundError(
            f"No .pth checkpoints found under: {run_dir} "
            f"(looked in '{nn_dir}' and the directory root)"
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
                f"No 'last_*.pth' checkpoints found under: {run_dir}. "
                f"Available: {[path.name for path in all_pths]}"
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
        print(f"[eval] Using checkpoint ({which}): {picked}")
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
    base_env._mix_episode_steps = torch.zeros((base_env.num_envs,), dtype=torch.long, device=base_env.device)
    base_env._mix_response_track_mask = torch.zeros((base_env.num_envs,), dtype=torch.bool, device=base_env.device)
    base_env._mix_response_case_id = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=base_env.device)
    base_env._mix_response_start_step = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=base_env.device)
    base_env._mix_first_fire_step = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=base_env.device)
    base_env._mix_first_hit_step = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=base_env.device)
    base_env._mix_response_samples_limit = 50
    base_env._mix_ss_completed_episodes = 0
    base_env._mix_ss_response_started_episodes = 0
    base_env._mix_ss_first_fire_response_times_s: list[float] = []
    base_env._mix_ss_first_hit_response_times_s: list[float] = []
    base_env._mix_ss_response_episode_samples: list[dict[str, float | int | None]] = []
    base_env._mix_sd_completed_episodes = 0
    base_env._mix_sd_response_started_episodes = 0
    base_env._mix_sd_first_fire_response_times_s: list[float] = []
    base_env._mix_sd_first_hit_response_times_s: list[float] = []
    base_env._mix_sd_response_episode_samples: list[dict[str, float | int | None]] = []
    base_env._mix_eval_case_episodes = torch.zeros((4,), dtype=torch.long, device=base_env.device)
    base_env._mix_eval_case_fires = torch.zeros((4,), dtype=torch.long, device=base_env.device)
    base_env._mix_eval_case_hits = torch.zeros((4,), dtype=torch.long, device=base_env.device)
    base_env._mix_eval_case_timeouts = torch.zeros((4,), dtype=torch.long, device=base_env.device)
    base_env._mix_eval_case_shot_budget_exhausted = torch.zeros((4,), dtype=torch.long, device=base_env.device)
    base_env._mix_eval_case_fallen = torch.zeros((4,), dtype=torch.long, device=base_env.device)
    base_env._mix_eval_case_invalid = torch.zeros((4,), dtype=torch.long, device=base_env.device)
    base_env._mix_eval_case_out_of_view = torch.zeros((4,), dtype=torch.long, device=base_env.device)

    original_reset_idx = base_env._reset_idx
    original_pre_physics_step = base_env._pre_physics_step
    original_get_dones = base_env._get_dones
    original_reset_eval_metrics = base_env.reset_eval_metrics

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
        # angular FOV. A moving robot uses its current pose as the constraint frame.
        relative_constraint = target_dynamic
        fixed_anchor_constraint = (~robot_dynamic) & target_dynamic
        base_env._target_motion_mask[env_ids_t] = target_dynamic
        base_env._target_relative_constraint_mask[env_ids_t] = relative_constraint
        base_env._target_constraint_fixed_anchor_mask[env_ids_t] = fixed_anchor_constraint
        base_env._target_out_of_view_termination_mask[env_ids_t] = ~robot_dynamic

        ss_mask = case_ids == 0
        sd_mask = case_ids == 1
        response_mask = ss_mask | sd_mask
        base_env._mix_episode_steps[env_ids_t] = 0
        base_env._mix_response_track_mask[env_ids_t] = response_mask
        base_env._mix_response_case_id[env_ids_t] = torch.where(
            response_mask,
            case_ids,
            torch.full_like(case_ids, -1),
        )
        base_env._mix_first_fire_step[env_ids_t] = -1
        base_env._mix_first_hit_step[env_ids_t] = -1

        static_response_start_steps = torch.full_like(case_ids, int(base_env._aim_warmup_steps))
        base_env._mix_response_start_step[env_ids_t] = torch.where(
            response_mask,
            static_response_start_steps,
            torch.full_like(case_ids, -1),
        )

        if torch.any(robot_static):
            static_ids = env_ids_t[robot_static]
            base_env._locomotion_commands[static_ids] = 0.0
            base_env._locomotion_command_time_left[static_ids] = base_env.cfg.locomotion_command_resample_s
            base_env._locomotion_last_actions[static_ids] = 0.0

            # 静止狗：仍然使用 locomotion policy，
            # 但把 command 固定为 0，不再随机重采样，不再强制写死腿部目标。
            base_env._locomotion_commands[static_ids] = 0.0
            base_env._locomotion_command_time_left[static_ids] = base_env.cfg.locomotion_command_resample_s

            # reset 时清零上一帧动作，避免跨 episode 残留。
            base_env._locomotion_last_actions[static_ids] = 0.0

        # SS and SD both retain the reset-time warmup. The zero-command
        # standing policy and articulation still need time to settle before
        # aiming, firing, and response-time measurement are enabled.

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

    def _record_completed_response_episodes(self, env_ids_t: torch.Tensor):
        if env_ids_t.numel() == 0:
            return

        completed_mask = self._mix_response_track_mask[env_ids_t]
        if hasattr(self, "episode_length_buf"):
            completed_mask = completed_mask & (self.episode_length_buf[env_ids_t] > 0)
        if not torch.any(completed_mask):
            return

        dt = float(self.step_dt)
        completed_ids = env_ids_t[completed_mask].detach().cpu().tolist()
        for env_id in completed_ids:
            case_id = int(self._mix_response_case_id[env_id].item())
            if case_id == 0:
                self._mix_ss_completed_episodes += 1
                completed_episode = int(self._mix_ss_completed_episodes)
                first_fire_values = self._mix_ss_first_fire_response_times_s
                first_hit_values = self._mix_ss_first_hit_response_times_s
                samples = self._mix_ss_response_episode_samples
            elif case_id == 1:
                self._mix_sd_completed_episodes += 1
                completed_episode = int(self._mix_sd_completed_episodes)
                first_fire_values = self._mix_sd_first_fire_response_times_s
                first_hit_values = self._mix_sd_first_hit_response_times_s
                samples = self._mix_sd_response_episode_samples
            else:
                continue

            start_step = int(self._mix_response_start_step[env_id].item())
            if start_step < 0:
                continue

            if case_id == 0:
                self._mix_ss_response_started_episodes += 1
            else:
                self._mix_sd_response_started_episodes += 1

            first_fire_s: float | None = None
            first_fire_step = int(self._mix_first_fire_step[env_id].item())
            if first_fire_step >= start_step:
                first_fire_s = max(0.0, float(first_fire_step - start_step) * dt)
                first_fire_values.append(first_fire_s)

            first_hit_s: float | None = None
            first_hit_step = int(self._mix_first_hit_step[env_id].item())
            if first_hit_step >= start_step:
                first_hit_s = max(0.0, float(first_hit_step - start_step) * dt)
                first_hit_values.append(first_hit_s)

            if len(samples) < int(self._mix_response_samples_limit):
                samples.append(
                    {
                        "episode": completed_episode,
                        "first_fire_s": first_fire_s,
                        "first_hit_s": first_hit_s,
                    }
                )

    def _record_completed_case_episodes(self, env_ids_t: torch.Tensor):
        if env_ids_t.numel() == 0 or not bool(getattr(self, "_collect_eval_details", False)):
            return

        completed_mask = torch.ones_like(env_ids_t, dtype=torch.bool)
        if hasattr(self, "episode_length_buf"):
            completed_mask &= self.episode_length_buf[env_ids_t] > 0
        completed_ids = env_ids_t[completed_mask]
        if completed_ids.numel() == 0:
            return

        case_ids = self._mix_case_id[completed_ids]
        ones = torch.ones_like(case_ids, dtype=torch.long)
        self._mix_eval_case_episodes.scatter_add_(0, case_ids, ones)
        self._mix_eval_case_fires.scatter_add_(
            0, case_ids, self._episode_fire_count[completed_ids].to(dtype=torch.long)
        )
        self._mix_eval_case_hits.scatter_add_(
            0, case_ids, self._hit_flag[completed_ids].to(dtype=torch.long)
        )
        self._mix_eval_case_timeouts.scatter_add_(
            0, case_ids, self.reset_time_outs[completed_ids].to(dtype=torch.long)
        )
        self._mix_eval_case_shot_budget_exhausted.scatter_add_(
            0, case_ids, self._shot_budget_exhausted[completed_ids].to(dtype=torch.long)
        )

        root_height = self.turret.data.root_pos_w[completed_ids, 2]
        upright_cos = -self.turret.data.projected_gravity_b[completed_ids, 2]
        fallen = (
            (root_height < float(self.cfg.failure_height_threshold))
            | (upright_cos < float(self.cfg.failure_tilt_cos_threshold))
        )
        invalid = self._get_invalid_state_mask()[completed_ids]
        out_of_view = self._target_out_of_view_terminated[completed_ids]
        self._mix_eval_case_fallen.scatter_add_(0, case_ids, fallen.to(dtype=torch.long))
        self._mix_eval_case_invalid.scatter_add_(0, case_ids, invalid.to(dtype=torch.long))
        self._mix_eval_case_out_of_view.scatter_add_(0, case_ids, out_of_view.to(dtype=torch.long))

    def reset_sd_response_metrics(self):
        self._mix_episode_steps.zero_()
        self._mix_response_track_mask.zero_()
        self._mix_response_case_id.fill_(-1)
        self._mix_response_start_step.fill_(-1)
        self._mix_first_fire_step.fill_(-1)
        self._mix_first_hit_step.fill_(-1)
        self._mix_ss_completed_episodes = 0
        self._mix_ss_response_started_episodes = 0
        self._mix_ss_first_fire_response_times_s.clear()
        self._mix_ss_first_hit_response_times_s.clear()
        self._mix_ss_response_episode_samples.clear()
        self._mix_sd_completed_episodes = 0
        self._mix_sd_response_started_episodes = 0
        self._mix_sd_first_fire_response_times_s.clear()
        self._mix_sd_first_hit_response_times_s.clear()
        self._mix_sd_response_episode_samples.clear()

    def reset_eval_metrics_with_case_stats(self):
        original_reset_eval_metrics()
        self._mix_eval_case_episodes.zero_()
        self._mix_eval_case_fires.zero_()
        self._mix_eval_case_hits.zero_()
        self._mix_eval_case_timeouts.zero_()
        self._mix_eval_case_shot_budget_exhausted.zero_()
        self._mix_eval_case_fallen.zero_()
        self._mix_eval_case_invalid.zero_()
        self._mix_eval_case_out_of_view.zero_()

    def get_sd_response_metrics(self):
        def _stats(values: list[float]) -> tuple[float, float, float]:
            if len(values) == 0:
                return 0.0, 0.0, 0.0
            return float(sum(values) / len(values)), float(min(values)), float(max(values))

        ss_fire_mean, ss_fire_min, ss_fire_max = _stats(self._mix_ss_first_fire_response_times_s)
        ss_hit_mean, ss_hit_min, ss_hit_max = _stats(self._mix_ss_first_hit_response_times_s)
        sd_fire_mean, sd_fire_min, sd_fire_max = _stats(self._mix_sd_first_fire_response_times_s)
        sd_hit_mean, sd_hit_min, sd_hit_max = _stats(self._mix_sd_first_hit_response_times_s)

        return {
            "ss_response_completed_episodes": float(self._mix_ss_completed_episodes),
            "ss_response_started_episodes": float(self._mix_ss_response_started_episodes),
            "ss_first_fire_response_count": float(len(self._mix_ss_first_fire_response_times_s)),
            "ss_first_fire_response_mean_s": ss_fire_mean,
            "ss_first_fire_response_min_s": ss_fire_min,
            "ss_first_fire_response_max_s": ss_fire_max,
            "ss_first_hit_response_count": float(len(self._mix_ss_first_hit_response_times_s)),
            "ss_first_hit_response_mean_s": ss_hit_mean,
            "ss_first_hit_response_min_s": ss_hit_min,
            "ss_first_hit_response_max_s": ss_hit_max,
            "ss_response_episode_samples": self._mix_ss_response_episode_samples,
            "sd_response_completed_episodes": float(self._mix_sd_completed_episodes),
            "sd_response_started_episodes": float(self._mix_sd_response_started_episodes),
            "sd_first_fire_response_count": float(len(self._mix_sd_first_fire_response_times_s)),
            "sd_first_fire_response_mean_s": sd_fire_mean,
            "sd_first_fire_response_min_s": sd_fire_min,
            "sd_first_fire_response_max_s": sd_fire_max,
            "sd_first_hit_response_count": float(len(self._mix_sd_first_hit_response_times_s)),
            "sd_first_hit_response_mean_s": sd_hit_mean,
            "sd_first_hit_response_min_s": sd_hit_min,
            "sd_first_hit_response_max_s": sd_hit_max,
            "sd_response_episode_samples": self._mix_sd_response_episode_samples,
        }

    def _response_open_mask_for_steps(self, step_values: torch.Tensor) -> torch.Tensor:
        return (
            self._mix_response_track_mask
            & (self._mix_response_start_step >= 0)
            & (step_values >= self._mix_response_start_step)
        )

    def _record_first_hit_steps(self, hit_now: torch.Tensor, step_values: torch.Tensor):
        if not torch.any(hit_now):
            return
        response_open_mask = _response_open_mask_for_steps(self, step_values)
        first_hit_mask = response_open_mask & (self._mix_first_hit_step < 0) & hit_now
        if torch.any(first_hit_mask):
            self._mix_first_hit_step[first_hit_mask] = step_values[first_hit_mask]

    def patched_reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        _record_completed_case_episodes(self, env_ids_t)
        _record_completed_response_episodes(self, env_ids_t)
        result = original_reset_idx(env_ids_t)

        _resample_cases(env_ids_t)
        _apply_case_masks(env_ids_t)

        # 如果以后误装了 bridge，也同步 reset buffer
        if hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None:
            self._rl_perception_bridge.reset_envs(env_ids_t)

        return result

    def patched_pre_physics_step(self, actions):
        current_steps = self._mix_episode_steps.clone()
        robot_static_mask = ~self._mix_robot_dynamic

        # 静止机器狗仍然走 locomotion policy，
        # 只是 command 固定为 0，且不允许被随机重采样。
        robot_static_mask = ~self._mix_robot_dynamic
        if torch.any(robot_static_mask):
            self._locomotion_commands[robot_static_mask] = 0.0

            # 关键：不能设成 0，否则 original_pre_physics_step 会重新采样随机速度指令
            self._locomotion_command_time_left[robot_static_mask] = self.cfg.locomotion_command_resample_s

            # Do not clear SS warmup here; keep evaluation aligned with
            # training and with the base environment's firing gate.

        # Match training: pass the policy action through unchanged. The base
        # environment already blocks aiming, firing, and fire-intent counting
        # until warmup finishes.
        original_pre_physics_step(actions)

        response_open_mask = _response_open_mask_for_steps(self, current_steps)
        if torch.any(response_open_mask):
            fire_now = self._fire_command.bool()
            first_fire_mask = response_open_mask & (self._mix_first_fire_step < 0) & fire_now
            if torch.any(first_fire_mask):
                self._mix_first_fire_step[first_fire_mask] = current_steps[first_fire_mask]

        self._mix_episode_steps += 1

        # 不要再把静止狗写回默认腿位，否则 eval 和训练分布不一致。

        target_static_mask = ~self._mix_target_dynamic
        if torch.any(target_static_mask):
            self._target_vel_w[target_static_mask] = 0.0
            if hasattr(self, "_target_acc_w"):
                self._target_acc_w[target_static_mask] = 0.0

    def patched_get_dones(self):
        result = original_get_dones()
        try:
            if hasattr(self, "_compute_current_step_hit"):
                hit_now = self._compute_current_step_hit()
                hit_step_values = torch.clamp(self._mix_episode_steps - 1, min=0)
                _record_first_hit_steps(self, hit_now, hit_step_values)
        except Exception as exc:
            if not hasattr(self, "_mix_first_hit_warned"):
                print(f"[eval] first_hit_response timing disabled due to runtime error: {exc}")
                self._mix_first_hit_warned = True
        return result

    def get_mix_case_stats(self):
        case_names = ("ss", "sd", "ds", "dd")
        episodes = self._mix_eval_case_episodes.detach().cpu()
        fires = self._mix_eval_case_fires.detach().cpu()
        hits = self._mix_eval_case_hits.detach().cpu()
        timeouts = self._mix_eval_case_timeouts.detach().cpu()
        shot_budget_exhausted = self._mix_eval_case_shot_budget_exhausted.detach().cpu()
        fallen = self._mix_eval_case_fallen.detach().cpu()
        invalid = self._mix_eval_case_invalid.detach().cpu()
        out_of_view = self._mix_eval_case_out_of_view.detach().cpu()
        total = int(episodes.sum().item())
        result: dict[str, float] = {"total": float(total)}
        for case_id, case_name in enumerate(case_names):
            case_episodes = int(episodes[case_id].item())
            case_fires = int(fires[case_id].item())
            case_hits = int(hits[case_id].item())
            denom = max(case_episodes, 1)
            result[case_name] = float(case_episodes / max(total, 1))
            result[f"{case_name}_episodes"] = float(case_episodes)
            result[f"{case_name}_fires"] = float(case_fires)
            result[f"{case_name}_hits"] = float(case_hits)
            result[f"{case_name}_hit_rate"] = float(case_hits / denom)
            result[f"{case_name}_accuracy_rate"] = float(case_hits / max(case_fires, 1))
            result[f"{case_name}_avg_fires_per_episode"] = float(case_fires / denom)
            result[f"{case_name}_timeouts"] = float(timeouts[case_id].item())
            result[f"{case_name}_shot_budget_exhausted"] = float(shot_budget_exhausted[case_id].item())
            result[f"{case_name}_shot_budget_exhausted_rate"] = float(shot_budget_exhausted[case_id].item() / denom)
            result[f"{case_name}_fallen"] = float(fallen[case_id].item())
            result[f"{case_name}_invalid"] = float(invalid[case_id].item())
            result[f"{case_name}_out_of_view"] = float(out_of_view[case_id].item())
        return result

    base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)
    base_env._pre_physics_step = types.MethodType(patched_pre_physics_step, base_env)
    base_env._get_dones = types.MethodType(patched_get_dones, base_env)
    base_env.reset_eval_metrics = types.MethodType(reset_eval_metrics_with_case_stats, base_env)
    base_env.reset_sd_response_metrics = types.MethodType(reset_sd_response_metrics, base_env)
    base_env.get_sd_response_metrics = types.MethodType(get_sd_response_metrics, base_env)
    base_env.get_mix_case_stats = types.MethodType(get_mix_case_stats, base_env)
    print(
        "[eval] mixed-case logic installed | "
        "all cases use aim_warmup_s for no-aim/no-fire warmup | "
        "first_hit_response uses post-physics hit timing"
    )


def _case_to_probs(case_name: str) -> tuple[float, float, float, float]:
    case_name = case_name.lower().strip()
    if case_name == "mixed":
        return (0.25, 0.25, 0.25, 0.25)
    if case_name == "ss":
        return (1.0, 0.0, 0.0, 0.0)
    if case_name == "sd":
        return (0.0, 1.0, 0.0, 0.0)
    if case_name == "ds":
        return (0.0, 0.0, 1.0, 0.0)
    if case_name == "dd":
        return (0.0, 0.0, 0.0, 1.0)
    raise ValueError("--case must be one of: ss, sd, ds, dd, mixed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0", help="Simulation, rendering and model device, e.g. cuda:1.")
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--debug_diagnostics", action=argparse.BooleanOptionalAction, default=False,
                        help="Print environment camera/raycast diagnostics; disabled by default.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pth checkpoint or a run directory.",
    )
    parser.add_argument("--which", type=str, default="best", choices=["best", "last", "latest"])

    parser.add_argument(
        "--locomotion_policy",
        type=str,
        required=True,
        help="Path to exported locomotion TorchScript policy (policy.pt).",
    )

    parser.add_argument(
        "--vision_model",
        type=str,
        default="source/standalone/test4/vision_model/vision_runs/vision_runs_cnn_only/04-10-00-24/vision_best.pth",
        help="Path to vision model checkpoint used by rl_perception_bridge.",
    )
    parser.add_argument(
        "--velocity_model",
        type=str,
        default="source/standalone/test6/speed_prediction/gru_runs/04-10-21-51/best_model.pt",
        help="Path to velocity GRU checkpoint used by rl_perception_bridge. Empty means no velocity model.",
    )
    parser.add_argument(
        "--full_chain_perception",
        action="store_true",
        help="Evaluate with test_lk_ostrack RGB tracking and test11 frozen velocity GRU.",
    )
    parser.add_argument(
        "--full_chain_velocity_checkpoint", "--gru_velocity_checkpoint", "--taq_velocity_checkpoint",
        dest="full_chain_velocity_checkpoint",
        type=str,
        default=(
            "source/standalone/test11/speed_prediction/legacy_gru/full_episode_finetune_runs/"
            "20261013_163052_278937/best_model.pt"
        ),
        help="Full-chain velocity checkpoint; defaults to the full-episode-finetuned ordinary GRU. Old TAQ option name remains an alias.",
    )
    parser.add_argument("--velocity_backend", choices=("legacy_gru", "taq_gru"), default="legacy_gru")
    parser.add_argument("--fine_tracking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ostrack_repo", type=str, default="source/standalone/test_lk_ostrack/ostrack/OSTrack-main")
    parser.add_argument(
        "--ostrack_checkpoint",
        type=str,
        default=(
            "source/standalone/test_lk_ostrack/ostrack/OSTrack-main/"
            "output/checkpoints/OSTrack_ep0300.pth.tar"
        ),
    )
    parser.add_argument("--ostrack_config", type=str, default="vitb_256_mae_ce_32x4_ep300")
    parser.add_argument("--fine_tracker_policy", choices=("always", "interval", "adaptive"), default="adaptive")
    parser.add_argument("--fine_tracker_interval_frames", type=int, default=3)
    parser.add_argument("--min_velocity_observations", type=int, default=3)
    parser.add_argument("--min_velocity_history_s", type=float, default=0.10)
    parser.add_argument("--min_bootstrap_box_px", type=float, default=8.0)
    parser.add_argument(
        "--enable_rl_perception_bridge",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        default=True,
        help="Allow bridge to fallback to GT distance when predicted/lidar distance is unavailable.",
    )
    parser.add_argument(
        "--bridge_use_gt_velocity_until_buffer_ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow bridge to fallback to GT velocity until GRU sequence buffer is ready.",
    )

    parser.add_argument(
        "--case",
        type=str,
        default="mixed",
        choices=["ss", "sd", "ds", "dd", "mixed"],
        help="Evaluation scenario.",
    )

    parser.add_argument("--target_speed_min", type=float, default=0.6, help="Minimum 3-D target speed magnitude (m/s).")
    parser.add_argument("--target_speed_max", type=float, default=5.0, help="Maximum 3-D target speed magnitude (m/s).")
    parser.add_argument("--fire_intent_threshold", type=float, default=0.50)
    parser.add_argument("--fire_intent_hold_steps", type=int, default=1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print_case_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_shots_per_episode", type=int, default=0,
                        help="Actual shots per single-target episode: 5 ends after the fifth shot; "
                             "a fifth-shot hit is success. The 10 s timeout remains; 0 disables the budget.")
    parser.add_argument("--static_salvo", action="store_true", help="SS continuous clearance of a fixed target pool; one episode is a complete pool.")
    parser.add_argument("--yaw_error_mode", choices=("joint_reachable", "wrapped_los"), default="wrapped_los",
                        help="Actor yaw input: wrapped LOS shortest turn (default), matching historical untagged checkpoints.")
    parser.add_argument("--salvo_targets", type=int, default=100)
    parser.add_argument("--salvo_timeout_s", type=float, default=600.0, help="Whole-pool simulation timeout including warmup.")
    parser.add_argument("--salvo_fire_cooldown_steps", type=int, default=None, help="Override the existing firing cooldown for a rate experiment.")
    parser.add_argument("--salvo_report", type=str, default=None, help="JSON result path; default is outputs/static_salvo/<timestamp>.json.")
    parser.add_argument("--salvo_switch_policy", choices=("both", "misses", "timeout"), default="both",
                        help="Default both: switch on five misses OR the attempt timeout. misses/timeout enable only one trigger.")
    parser.add_argument("--salvo_miss_limit", type=int, default=5,
                        help="Consecutive actual missed shots per attempt before switching; idle/cooldown steps do not count.")
    parser.add_argument("--salvo_stall_timeout_s", type=float, default=3.0, help="Attempt timeout in simulation seconds, used by both/timeout policies.")
    parser.add_argument("--salvo_retry_delay_s", type=float, default=1.0, help="Simulation seconds before a deferred target can be selected again.")
    parser.add_argument("--salvo_max_attempts", type=int, default=3, help="Maximum engagement attempts per target, including the first.")
    parser.add_argument("--salvo_diagnostics", type=str, default=None, help="Per-step firing diagnostic CSV; defaults beside the salvo report.")
    parser.add_argument("--salvo_log_interval_s", type=float, default=10.0, help="Progress print interval in wall-clock seconds.")
    parser.add_argument("--salvo_ablation", choices=list("ABCD"), default=None, help="Static salvo defaults to D: hold when unready and reset Actor GRU on target switch. A/B/C remain available for comparison.")
    parser.add_argument("--salvo_wait_for_velocity", action=argparse.BooleanOptionalAction, default=None,
                        help="Pause policy inference until the selected target's range/velocity chain is ready; defaults on for D. Disable to reproduce the original D experiment.")
    parser.add_argument("--salvo_return_home", action=argparse.BooleanOptionalAction, default=False,
                        help="Physically return yaw/pitch to zero before every target attempt; requires D and velocity waiting.")
    parser.add_argument("--salvo_home_angle_tol_deg", type=float, default=.5)
    parser.add_argument("--salvo_home_velocity_tol_rps", type=float, default=.05)
    parser.add_argument("--salvo_home_settle_frames", type=int, default=5)
    parser.add_argument("--salvo_home_timeout_s", type=float, default=8.,
                        help="End the episode with return_home_timeout if physical homing cannot settle in time.")
    parser.add_argument("--salvo_seed", type=int, default=61, help="Explicit environment/RL-Games/OpenCV seed for SS evaluation.")
    layout_args = parser.add_mutually_exclusive_group()
    layout_args.add_argument("--salvo_layout_in", help="Load a saved env-local target layout.")
    layout_args.add_argument("--salvo_layout_out", help="Save the first target layout to a new JSON file.")
    parser.add_argument("--record_video", action="store_true", help="Save episode MP4s with the robot RGB camera inset.")
    parser.add_argument("--video_episodes", type=int, default=10, help="Maximum complete episodes recorded from video_env_id.")
    parser.add_argument("--video_env_id", type=int, default=0)
    parser.add_argument("--video_hit_hold_s", type=float, default=1.5, help="Freeze the annotated hit frame for this many video seconds; does not step physics.")
    parser.add_argument("--video_dir", type=str, default=None, help="New output directory; defaults to source/standalone/integrated_shooting_chain/video/<timestamp>.")
    parser.add_argument("--draw_tracks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track_env_id", type=int, default=0)
    parser.add_argument("--track_max_points", type=int, default=600)
    parser.add_argument("--track_shot_history", type=int, default=25)
    parser.add_argument("--track_ballistic_stride", type=int, default=3)
    parser.add_argument("--robot_track_width", type=float, default=8.0)
    parser.add_argument("--target_track_width", type=float, default=8.0)
    parser.add_argument("--bullet_track_width", type=float, default=6.0)
    parser.add_argument("--pause_between_episodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pause_seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_shots_per_episode < 0:
        raise ValueError("--max_shots_per_episode must be >= 0")
    if args.static_salvo and args.max_shots_per_episode:
        raise ValueError("--max_shots_per_episode is for single-target episodes, not --static_salvo")
    if args.salvo_ablation is None:
        args.salvo_ablation = "D" if args.static_salvo else "A"
    if args.salvo_return_home:
        if not args.static_salvo or args.salvo_ablation != "D" or args.salvo_wait_for_velocity is False:
            raise ValueError("--salvo_return_home requires --static_salvo, group D and velocity waiting")
        if any(not math.isfinite(v) or v <= 0 for v in (
            args.salvo_home_angle_tol_deg, args.salvo_home_velocity_tol_rps, args.salvo_home_timeout_s
        )) or args.salvo_home_settle_frames < 1:
            raise ValueError("Return-home tolerances, settle frames and timeout must be positive and finite")
    if not args.static_salvo and (args.salvo_ablation != "A" or args.salvo_wait_for_velocity is not None or args.salvo_layout_in or args.salvo_layout_out):
        raise ValueError("Salvo ablation and layout options require --static_salvo")
    if args.static_salvo:
        if args.salvo_seed < 0 or args.salvo_seed >= 2**31:
            raise ValueError("salvo_seed must be in [0, 2**31)")
        if (args.salvo_layout_in or args.salvo_layout_out) and (args.num_envs != 1 or args.episodes != 1):
            raise ValueError("Fixed-layout experiments require --num_envs 1 --episodes 1")
        # This scenario exercises the requested multi-target visual tracking.
        args.full_chain_perception = True
        if args.case == "mixed":
            args.case = "ss"
        if args.case != "ss":
            raise ValueError("--static_salvo currently supports --case ss only.")
        if args.salvo_targets < 1 or not math.isfinite(args.salvo_timeout_s) or args.salvo_timeout_s <= 2.0:
            raise ValueError("salvo_targets must be positive and salvo_timeout_s must exceed the 2 s warmup.")
        if args.salvo_fire_cooldown_steps is not None and args.salvo_fire_cooldown_steps < 0:
            raise ValueError("salvo_fire_cooldown_steps must be nonnegative.")
        if any(not math.isfinite(v) or v <= 0 for v in (
            args.salvo_retry_delay_s, args.salvo_log_interval_s
        )) or args.salvo_max_attempts < 1 or args.salvo_miss_limit < 1:
            raise ValueError("Salvo retry/log intervals, max_attempts and miss_limit must be positive.")
        if args.salvo_switch_policy in ("both", "timeout") and (
            not math.isfinite(args.salvo_stall_timeout_s) or args.salvo_stall_timeout_s <= 0
        ):
            raise ValueError("Salvo stall timeout must be finite and positive in both/timeout modes.")

    # Existing evaluation stays GT-only by default. Full-chain mode turns on
    # the rendered camera and installs the LK/OSTrack + velocity bridge below.
    full_chain = bool(args.full_chain_perception)
    if args.record_video:
        if not full_chain:
            raise ValueError("--record_video requires --full_chain_perception for the robot-camera image.")
        if args.video_episodes < 1 or not 0 <= args.video_env_id < args.num_envs or args.video_hit_hold_s < 0:
            raise ValueError("--video_episodes must be positive, --video_env_id within num_envs, and --video_hit_hold_s nonnegative.")
    args.enable_cameras = full_chain or args.static_salvo
    args.enable_rl_perception_bridge = False
    if full_chain and args.case == "mixed":
        args.case = "sd"
    if full_chain and args.case not in ("ss", "sd"):
        raise ValueError("Full-chain velocity eval supports --case ss or --case sd only.")
    if full_chain and args.fine_tracking and args.num_envs > 2:
        print(
            f"[eval][full_chain] OSTrack enabled for {args.num_envs} envs with one shared network and independent tracking state. "
            "There is no hard environment limit; reduce --num_envs if GPU memory runs out."
        )
    if full_chain:
        required = [Path(args.full_chain_velocity_checkpoint)]
        if args.fine_tracking:
            required.extend((Path(args.ostrack_repo), Path(args.ostrack_checkpoint)))
        missing = [str(path) for path in required if not path.expanduser().exists()]
        if missing:
            raise FileNotFoundError(f"Full-chain checkpoint/path not found: {missing}")

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, which=args.which)
    print(f"[eval] checkpoint = {checkpoint_path}")

    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=full_chain,
        device=args.device,
        multi_gpu=False,
    )
    sim_app = app_launcher.app

    try:
        import torch
        import gymnasium as gym
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(args.device)

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

    if args.deterministic:
        _install_tanh_deterministic_player_patch()
        print("[eval] deterministic action = tanh(mu)")

    env_cfg = TurretAimEnvCfg()
    env_cfg.yaw_error_mode = args.yaw_error_mode
    aim_contract = observation_contract(args.yaw_error_mode, env_cfg.aim_error_obs_scale, env_cfg.aim_error_obs_clip)
    source_aim_contract = checkpoint_contract(checkpoint_path)
    aim_contract_matches = contracts_match(source_aim_contract, aim_contract)
    print(f"[aim-contract] checkpoint={source_aim_contract['version']} evaluation={aim_contract['version']} "
          f"matches={aim_contract_matches}", flush=True)
    if not aim_contract_matches:
        print("[aim-contract] WARNING: this checkpoint has not been adapted to the requested yaw input. "
              "This run is a contract-mismatch diagnostic; finetune before interpreting it as the new policy, "
              "or select --yaw_error_mode to match the checkpoint contract.", flush=True)
    env_cfg.max_shots_per_episode = args.max_shots_per_episode
    if not args.static_salvo:
        print(f"[eval] episode_length_s={env_cfg.episode_length_s} "
              f"max_shots_per_episode={env_cfg.max_shots_per_episode} (0=unlimited)", flush=True)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device
    if args.static_salvo:
        import cv2
        env_cfg.seed = args.salvo_seed
        cv2.setRNGSeed(args.salvo_seed)

    # -----------------------------
    # Camera-backed perception needs the non-Fabric sensor path used by train.
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "use_fabric"):
        env_cfg.sim.use_fabric = not (full_chain or args.static_salvo)
        print(f"[eval] env_cfg.sim.use_fabric = {env_cfg.sim.use_fabric}")

    # 与训练时保持一致：底层环境统一支持 moving base + dynamic target，
    # 再通过 case mask 在每个 episode 约束成 ss/sd/ds/dd。
    env_cfg.enable_locomotion_policy = True
    env_cfg.locomotion_policy_path = args.locomotion_policy
    env_cfg.enable_dynamic_target = True
    env_cfg.target_speed_range = (float(args.target_speed_min), float(args.target_speed_max))

    # ---------------------------------------------------------
    # GT-only 对比实验：
    # - 弹丸只考虑重力，不考虑空气阻力；
    # - 目标允许匀速运动，但不允许加速度，否则 actor 虽然吃 GT pos/vel，
    #   但没有 target_acc，训练目标会变成部分不可观测。
    # ---------------------------------------------------------
    env_cfg.target_acc_range = (0.0, 0.0)
    env_cfg.target_vertical_acc_range = (0.0, 0.0)
    env_cfg.fire_intent_threshold = float(args.fire_intent_threshold)
    env_cfg.fire_intent_hold_steps = int(args.fire_intent_hold_steps)

    env_cfg.debug_vis = not args.headless
    env_cfg.render_target_sphere = full_chain or args.static_salvo or (not args.headless)
    env_cfg.debug_diagnostics = bool(args.debug_diagnostics)

    # 排查时建议先关 warmup，或者缩短 warmup
    env_cfg.aim_warmup_s = 2.0
    env_cfg.enable_dynamic_target = True
    env_cfg.enable_air_drag = True

    env_cfg.enable_vision_supervision = full_chain or args.static_salvo
    env_cfg.enable_dynamic_target_raycast_debug = full_chain
    env_cfg.sample_target_in_camera_sector = full_chain or args.static_salvo
    if args.static_salvo:
        env_cfg.episode_length_s = float(args.salvo_timeout_s)
        if args.salvo_fire_cooldown_steps is not None:
            env_cfg.fire_cooldown_steps = int(args.salvo_fire_cooldown_steps)
    if args.record_video:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = int(args.video_env_id)
        env_cfg.viewer.eye = (3.5, 3.5, 2.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.4)
        env_cfg.viewer.resolution = (1280, 720)

    task_id = "Standalone-TurretAim-Mixed-Eval-v0"
    if task_id not in gym.registry:
        gym.register(id=task_id, entry_point=TurretAimEnv)

    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args.record_video else None, disable_env_checker=True)
    base_env: Any = env.unwrapped

    print("[eval] enable_air_drag =", bool(base_env.cfg.enable_air_drag))
    print("[eval] ballistic_max_time =", float(base_env.cfg.ballistic_max_time))
    print("[eval] ballistic_num_samples =", int(base_env.cfg.ballistic_num_samples))
    print("[eval] drag_coeff =", float(base_env._drag_accel_coeff.item()))
    # raise RuntimeError("[DEBUG STOP] 程序已运行到指定位置，手动停止")

    # ---------------------------------------------------------
    # Initialize the camera before installing a perception bridge.
    if full_chain or args.static_salvo:
        print("[eval] bootstrapping camera for perception or salvo frustum sampling")
        base_env.bootstrap_vision_camera()
    else:
        print("[eval][GT-only] skip bootstrap_vision_camera()")
    base_env._gt_only_mode = not full_chain
    base_env._use_gt_fallback_for_missing_rl_pred = not full_chain
    base_env._rl_pred_ready[:] = False
    base_env._rl_bridge_training_ready[:] = False
    base_env._rl_bridge_velocity_ready[:] = False

    _install_mixed_training_logic(
        base_env,
        case_probs=_case_to_probs(args.case),
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

    if args.static_salvo:
        standalone_dir = Path(__file__).resolve().parents[1]
        if str(standalone_dir) not in sys.path:
            sys.path.insert(0, str(standalone_dir))
        from static_salvo_eval import install_static_salvo
        from integrated_shooting_chain.salvo_ablation import install_salvo_ablation

        install_static_salvo(base_env, target_count=args.salvo_targets,
                             stall_timeout_s=(args.salvo_stall_timeout_s if args.salvo_switch_policy != "misses" else None),
                             retry_delay_s=args.salvo_retry_delay_s,
                             max_attempts=args.salvo_max_attempts,
                             miss_limit=args.salvo_miss_limit,
                             switch_policy=args.salvo_switch_policy,
                             layout_in=args.salvo_layout_in, layout_out=args.salvo_layout_out)
        if args.salvo_return_home:
            from integrated_shooting_chain.salvo_return_home import SalvoReturnHomeConfig, install_salvo_return_home
            install_salvo_return_home(base_env, SalvoReturnHomeConfig(
                angle_tolerance_deg=args.salvo_home_angle_tol_deg,
                velocity_tolerance_rps=args.salvo_home_velocity_tol_rps,
                settle_frames=args.salvo_home_settle_frames, timeout_s=args.salvo_home_timeout_s))
            print("[salvo] return-home enabled: rate-limited physical yaw/pitch -> 0 before each attempt; "
                  "Actor paused, no fire; homing counts toward the episode but not the attempt deadline.")
        ablation = install_salvo_ablation(base_env, args.salvo_ablation,
                                         wait_for_velocity=args.salvo_wait_for_velocity)
        print(f"[salvo] ablation={args.salvo_ablation} hold_when_unready={ablation.config.hold_when_unready} "
              f"reset_actor_on_switch={ablation.config.reset_actor_on_switch} "
              f"wait_for_velocity={ablation.config.pause_actor_when_unready} seed={args.salvo_seed}")
        print("[salvo] SS pool: one GT-box frame for all targets, then continuous multi-target RGB tracking.")
        print("[salvo] Select nearest visual target within the current round; failures wait for the next round. Bounded retries without GT rebootstrap. Other-target raycast occlusion is not modeled.")
        print(f"[salvo] switch_policy={args.salvo_switch_policy} miss_limit={args.salvo_miss_limit} "
              f"stall_timeout_s={base_env._static_salvo.engagement_config['stall_timeout_s']}")

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
            velocity_checkpoint=Path(args.full_chain_velocity_checkpoint).expanduser().resolve(),
            velocity_backend=args.velocity_backend,
            device=str(base_env.device),
            target_radius_m=float(env_cfg.target_sphere_radius),
            min_bootstrap_box_px=float(args.min_bootstrap_box_px),
            min_velocity_observations=int(args.min_velocity_observations),
            min_velocity_history_s=float(args.min_velocity_history_s),
            enable_fine_tracking=bool(args.fine_tracking),
            ostrack_repo=Path(args.ostrack_repo).expanduser().resolve() if args.fine_tracking else None,
            ostrack_checkpoint=Path(args.ostrack_checkpoint).expanduser().resolve() if args.fine_tracking else None,
            ostrack_config=str(args.ostrack_config),
            fine_tracker_policy=str(args.fine_tracker_policy),
            fine_tracker_interval_frames=int(args.fine_tracker_interval_frames),
            debug=bool(args.bridge_debug),
        )
        if args.static_salvo:
            from integrated_shooting_chain.static_salvo_perception import IsaacStaticSalvoPerceptionBridge

            bridge = IsaacStaticSalvoPerceptionBridge(base_env, bridge_config)
        else:
            bridge = IsaacFullChainPerceptionBridge(base_env, bridge_config)
        install_training_perception_bridge(base_env, bridge)
        print(f"[eval][full_chain] RGB tracker + raycast range + {args.velocity_backend} installed; GT actor fallback disabled.")

    # This legacy option is kept for compatibility with existing CLI callers.
    if args.enable_rl_perception_bridge:
        if not args.vision_model.strip():
            raise ValueError(
                "--enable_rl_perception_bridge is True, but --vision_model is empty."
            )

        bridge = install_rl_perception_bridge(
            base_env=base_env,
            vision_model_path=args.vision_model,
            velocity_model_path=args.velocity_model if args.velocity_model.strip() else None,
            device=str(base_env.device),
            use_gt_distance_fallback=args.bridge_use_gt_distance_fallback,
            use_gt_velocity_until_buffer_ready=args.bridge_use_gt_velocity_until_buffer_ready,
            debug=args.bridge_debug,
        )

        # reset 时同步清空 bridge 的时序缓存
        orig_reset_idx = base_env._reset_idx

        def patched_reset_idx(self, env_ids):
            out = orig_reset_idx(env_ids)
            if hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None:
                self._rl_perception_bridge.reset_envs(env_ids)
            return out

        base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)

        # 不再在 _get_observations() 中跑 bridge.step()
        # 改为在 env.step(action) 之后统一跑 perception bridge
        orig_step = base_env.step

        def patched_step(self, actions):
            out = orig_step(actions)

            if not (hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None):
                return out

            self._rl_perception_bridge.step()
            obs = self._get_observations()

            if isinstance(out, tuple):
                if len(out) == 5:
                    _, rew, terminated, truncated, extras = out
                    return obs, rew, terminated, truncated, extras
                if len(out) == 4:
                    _, rew, dones, extras = out
                    return obs, rew, dones, extras

            raise RuntimeError(
                f"[eval] Unexpected env.step() return format: type={type(out)}"
            )

        base_env.step = types.MethodType(patched_step, base_env)

        print("[eval] rl_perception_bridge hooked into step() and _reset_idx.")

    if env_cfg.debug_vis and bool(args.draw_tracks) and not args.record_video and not args.static_salvo:
        _install_runtime_track_draw(
            base_env=base_env,
            track_env_id=int(args.track_env_id),
            max_points=int(args.track_max_points),
            shot_history=int(args.track_shot_history),
            ballistic_draw_stride=int(args.track_ballistic_stride),
            robot_color=(0.1, 0.9, 0.1, 1.0),
            target_color=(1.0, 0.2, 0.2, 1.0),
            bullet_color=(1.0, 0.9, 0.1, 1.0),
            robot_width=float(args.robot_track_width),
            target_width=float(args.target_track_width),
            bullet_width=float(args.bullet_track_width),
            pause_between_episodes=bool(args.pause_between_episodes),
            pause_seconds=float(args.pause_seconds),
        )

    if hasattr(base_env, "reset_eval_metrics"):
        base_env.reset_eval_metrics()
    if hasattr(base_env, "reset_sd_response_metrics"):
        base_env.reset_sd_response_metrics()

    video_recorder = None
    if args.record_video:
        from integrated_shooting_chain.episode_video import EpisodeVideoRecorder, install_episode_video_recorder

        video_directory = args.video_dir or str(
            Path(__file__).resolve().parents[1] / "integrated_shooting_chain" / "video"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        video_recorder = EpisodeVideoRecorder(
            video_directory, env_id=args.video_env_id, episodes=args.video_episodes, step_dt=base_env.step_dt,
            hit_hold_s=args.video_hit_hold_s,
        )
        base_env.render()
        base_env.sim.render()
        install_episode_video_recorder(base_env, video_recorder)
        print(f"[eval][video] recording env {args.video_env_id}, up to {args.video_episodes} complete episodes")
        print(f"[eval][video] directory = {video_recorder.directory}")
        if args.num_envs > 1:
            print("[eval][video] --episodes counts all envs; increase it or use --num_envs 1 to obtain the requested videos.")

    rl_device = str(base_env.device)
    env = RlGamesVecEnvWrapper(
        base_env,
        rl_device,
        clip_obs=10.0,
        clip_actions=1.0,
        obs_groups={"obs": ["policy", "critic"], "states": ["critic"]},
        concate_obs_group=True,
    )

    experiment_name = f"turret_aim_eval_{datetime.now().strftime('%m-%d-%H-%M-%S')}"

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
    )

    config = {
        "params": {
            "seed": args.salvo_seed if args.static_salvo else 61,
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
            "load_checkpoint": True,
            "load_path": str(checkpoint_path),
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
                "minibatch_size": 1024,
                "mini_epochs": 2,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 5e-5,
                "clip_value": True,
                "entropy_coef": 0.0,
                "critic_coef": 2.0,
                "e_clip": 0.1,
                "max_epochs": 1,
                "save_best_after": 0,
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
                "player": {
                    "deterministic": bool(args.deterministic),
                    "games_num": int(args.episodes),
                    "print_stats": True,
                },
            },
        }
    }

    runner = Runner(IsaacAlgoObserver())
    runner.load(config)

    diagnostics = None
    report_path = None
    previous_sigint_handler = None
    if args.static_salvo:
        from integrated_shooting_chain.salvo_diagnostics import (
            SalvoDiagnostics, install_salvo_diagnostics, install_salvo_interrupt_handler,
        )
        from integrated_shooting_chain.salvo_engagement import required_player_steps

        if args.salvo_report:
            report_path = Path(args.salvo_report).expanduser()
        elif video_recorder is not None:
            report_path = video_recorder.directory / "salvo_report.json"
        else:
            report_path = Path("outputs/static_salvo") / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        diagnostics_path = Path(args.salvo_diagnostics).expanduser() if args.salvo_diagnostics else report_path.with_suffix(".steps.csv")
        diagnostics = SalvoDiagnostics(diagnostics_path, log_interval_s=args.salvo_log_interval_s)
        install_salvo_diagnostics(base_env, diagnostics)
        print(f"[salvo] step diagnostics = {diagnostics.path.resolve()}")
        original_create_player = runner.create_player

        def create_salvo_player():
            from integrated_shooting_chain.salvo_ablation import install_actor_switch_reset
            player = original_create_player()
            player.max_steps = max(player.max_steps, required_player_steps(
                args.salvo_timeout_s, base_env.step_dt, args.episodes))
            install_actor_switch_reset(player, ablation)
            return player

        runner.create_player = create_salvo_player
        previous_sigint_handler = install_salvo_interrupt_handler()

    run_args: dict[str, Any] = {
        "train": False,
        "play": True,
        "checkpoint": str(checkpoint_path),
    }

    print("[eval] starting play mode ...")
    try:
        runner.run(run_args)
    except KeyboardInterrupt:
        if not args.static_salvo:
            raise
        print("[salvo] interrupted; finalizing video, diagnostics and partial results ...", flush=True)
        state = base_env._static_salvo
        for env_id in range(base_env.num_envs):
            if not bool(state.finished[env_id]) and int(base_env.episode_length_buf[env_id]) > 0:
                total_s = int(base_env.episode_length_buf[env_id]) * base_env.step_dt
                state.finish(env_id, "interrupted", max(0.0, total_s - int(state.steps[env_id]) * base_env.step_dt))
    finally:
        if diagnostics is not None:
            diagnostics.close()
        if video_recorder is not None:
            video_recorder.close()
        if args.static_salvo:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            records = base_env._static_salvo.records
            report = {
                "scenario": "ss", "targets_per_episode": args.salvo_targets,
                "ablation_group": args.salvo_ablation,
                "hold_when_unready": ablation.config.hold_when_unready,
                "reset_actor_on_switch": ablation.config.reset_actor_on_switch,
                "wait_for_velocity": ablation.config.pause_actor_when_unready,
                "actor_wait_protocol": ("return_home_then_skip_inference_until_velocity_ready" if args.salvo_return_home else
                                        "skip_inference_hold_command_no_fire_until_velocity_ready"
                                        if ablation.config.pause_actor_when_unready else "inference_while_unready"),
                "return_home_enabled": args.salvo_return_home,
                "return_home_config": ({
                    "target_joint_angles_rad": [0., 0.],
                    "angle_tolerance_deg": args.salvo_home_angle_tol_deg,
                    "velocity_tolerance_rps": args.salvo_home_velocity_tol_rps,
                    "settle_frames": args.salvo_home_settle_frames,
                    "timeout_s": args.salvo_home_timeout_s,
                    "attempt_clock": "exclude_homing_only_keep_perception_wait_and_episode_clock",
                    "actor_reset": "once_after_home_each_attempt_including_first",
                } if args.salvo_return_home else None),
                "layout_in": args.salvo_layout_in, "layout_out": args.salvo_layout_out,
                "layout_sha256": getattr(base_env._static_salvo, "layout_sha256", None),
                "step_dt_s": float(base_env.step_dt),
                "fire_cooldown_steps": int(env_cfg.fire_cooldown_steps),
                "configured_max_firing_hz": 1.0 / ((int(env_cfg.fire_cooldown_steps) + 1) * base_env.step_dt),
                "full_chain_perception": full_chain,
                "selection": "dynamic_nearest_per_round_deferred_retries",
                "switch_policy": args.salvo_switch_policy,
                "miss_limit": args.salvo_miss_limit,
                "stall_timeout_s": base_env._static_salvo.engagement_config["stall_timeout_s"],
                "retry_delay_s": args.salvo_retry_delay_s,
                "max_attempts": args.salvo_max_attempts,
                "diagnostics_csv": str(diagnostics.path.resolve()),
                "diagnostics_schema_version": diagnostics.schema_version,
                "diagnostics_sampling": "decision_before_pre_physics_and_post_after_rewards_before_reset",
                "diagnostics_gt_usage": "evaluation_only_never_fed_to_tracker_or_actor",
                "aim_observation_contract": aim_contract,
                "checkpoint_aim_observation_contract": source_aim_contract,
                "aim_observation_contract_matches_checkpoint": aim_contract_matches,
                "decision_checkpoint": str(checkpoint_path.resolve()),
                "velocity_checkpoint": str(Path(args.full_chain_velocity_checkpoint).expanduser().resolve()),
                "velocity_backend": args.velocity_backend,
                "fine_tracking": bool(args.fine_tracking),
                "ostrack_checkpoint": str(Path(args.ostrack_checkpoint).expanduser().resolve()) if args.fine_tracking else None,
                "environment_seed": getattr(env_cfg, "seed", None),
                "rl_games_seed": config["params"]["seed"],
                "control_diagnostics_config": {
                    "muzzle_body_name_expr": env_cfg.muzzle_body_name_expr,
                    "muzzle_forward_axis": list(env_cfg.muzzle_forward_axis),
                    "muzzle_offset_b": list(env_cfg.muzzle_offset_b),
                    "max_delta_yaw_rad": float(env_cfg.max_delta_yaw),
                    "max_delta_pitch_rad": float(env_cfg.max_delta_pitch),
                    "aim_error_obs_scale": float(env_cfg.aim_error_obs_scale),
                    "aim_error_obs_clip": float(env_cfg.aim_error_obs_clip),
                },
                "gt_box_protocol": "one_frame_all_targets_per_episode_no_rebootstrap",
                "tracking": "LK_all_targets_OSTrack_selected_target_when_enabled",
                "hit_scoring": "selected_target_synchronous_ballistic_prediction",
                "other_target_raycast_occlusion": False,
                "episodes": records,
            }
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            print(f"[salvo] report = {report_path.resolve()}")
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)

    # Standard single-target hit/episode metrics have different semantics.
    if args.static_salvo:
        env.close()
        sim_app.close()
        return

    if hasattr(base_env, "get_eval_metrics"):
        metrics = base_env.get_eval_metrics()
        print("\n[eval] ===== Aggregated Metrics =====")
        print(f"[eval] episode_length_s={base_env.cfg.episode_length_s} "
              f"max_shots_per_episode={base_env.cfg.max_shots_per_episode} (0=unlimited)")
        print(f"[eval] episodes={int(metrics['episodes'])}")
        print(f"[eval] fires={int(metrics['fires'])}")
        print(f"[eval] hits={int(metrics['hits'])}")
        print(f"[eval] hit_rate={metrics['hit_rate']:.4f}")
        accuracy_denom = max(float(metrics["fires"]), 1.0)
        print(f"[eval] accuracy_rate={float(metrics['hits'])/accuracy_denom:.4f}")
        print(f"[eval] avg_fires_per_episode={metrics['avg_fires_per_episode']:.4f}")
        print(f"[eval] timeouts={int(metrics['timeouts'])} timeout_rate={metrics['timeout_rate']:.4f}")
        print(f"[eval] shot_budget_exhausted={int(metrics['shot_budget_exhausted'])} "
              f"shot_budget_exhausted_rate={metrics['shot_budget_exhausted_rate']:.4f}")
        print(f"[eval] fallen={int(metrics['fallen'])} fallen_rate={metrics['fallen_rate']:.4f}")
        print(f"[eval] invalid={int(metrics['invalid'])} invalid_rate={metrics['invalid_rate']:.4f}")
        print(
            f"[eval] target_spawn_count={int(metrics['target_spawn_count'])} "
            f"target_spawn_mean=({metrics['target_spawn_mean_x']:.3f}, {metrics['target_spawn_mean_y']:.3f}, {metrics['target_spawn_mean_z']:.3f})"
        )
        print(
            f"[eval] fire_min_dist_count={int(metrics['fire_min_dist_count'])} "
            f"fire_min_dist_mean={metrics['fire_min_dist_mean']:.4f} "
            f"fire_min_dist_min={metrics['fire_min_dist_min']:.4f} "
            f"fire_min_dist_max={metrics['fire_min_dist_max']:.4f}"
        )

        target_samples = metrics.get("target_spawn_samples", [])
        if isinstance(target_samples, list) and len(target_samples) > 0:
            print("[eval] target_spawn_samples(first up to 50):")
            for idx, pos in enumerate(target_samples):
                if isinstance(pos, list) and len(pos) >= 3:
                    print(f"[eval]   #{idx:02d}: ({float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f})")

        dist_samples = metrics.get("fire_min_dist_samples", [])
        if isinstance(dist_samples, list) and len(dist_samples) > 0:
            print("[eval] fire_min_dist_samples(first up to 50):")
            for idx, dist in enumerate(dist_samples):
                print(f"[eval]   #{idx:02d}: {float(dist):.4f}")

        episode_fire_counts = metrics.get("episode_fire_counts", [])
        episode_hits = metrics.get("episode_hits", [])
        if (
            isinstance(episode_fire_counts, list)
            and isinstance(episode_hits, list)
            and len(episode_fire_counts) > 0
            and len(episode_fire_counts) == len(episode_hits)
        ):
            print("[eval] per-episode fire/hit:")
            for idx, (fire_count, hit_flag) in enumerate(zip(episode_fire_counts, episode_hits)):
                hit_text = "HIT" if int(hit_flag) > 0 else "MISS"
                print(f"[eval]   ep#{idx + 1:02d}: fires={int(fire_count)} result={hit_text}")

    if args.print_case_stats and hasattr(base_env, "get_mix_case_stats"):
        case_stats = base_env.get_mix_case_stats()
        print("\n[eval] ===== Per-Case Metrics =====")
        print(f"[eval] completed_episodes={int(case_stats['total'])}")
        for case_name in ("ss", "sd", "ds", "dd"):
            print(
                f"[eval] {case_name.upper()} "
                f"episodes={int(case_stats[f'{case_name}_episodes'])} "
                f"episode_rate={case_stats[case_name]:.4f} "
                f"fires={int(case_stats[f'{case_name}_fires'])} "
                f"hits={int(case_stats[f'{case_name}_hits'])} "
                f"hit_rate={case_stats[f'{case_name}_hit_rate']:.4f} "
                f"accuracy_rate={case_stats[f'{case_name}_accuracy_rate']:.4f} "
                f"avg_fires={case_stats[f'{case_name}_avg_fires_per_episode']:.4f} "
                f"timeouts={int(case_stats[f'{case_name}_timeouts'])} "
                f"fallen={int(case_stats[f'{case_name}_fallen'])} "
                f"invalid={int(case_stats[f'{case_name}_invalid'])} "
                f"out_of_view={int(case_stats[f'{case_name}_out_of_view'])} "
                f"shot_budget_exhausted={int(case_stats[f'{case_name}_shot_budget_exhausted'])} "
                f"shot_budget_exhausted_rate={case_stats[f'{case_name}_shot_budget_exhausted_rate']:.4f}"
            )

    if hasattr(base_env, "get_sd_response_metrics"):
        response_metrics = base_env.get_sd_response_metrics()
        print("\n[eval] ===== SS Response Metrics =====")
        print(
            f"[eval] ss_completed_episodes={int(response_metrics['ss_response_completed_episodes'])} "
            f"ss_response_started_episodes={int(response_metrics['ss_response_started_episodes'])}"
        )
        print(
            f"[eval] first_fire_response_count={int(response_metrics['ss_first_fire_response_count'])} "
            f"mean_s={response_metrics['ss_first_fire_response_mean_s']:.4f} "
            f"min_s={response_metrics['ss_first_fire_response_min_s']:.4f} "
            f"max_s={response_metrics['ss_first_fire_response_max_s']:.4f}"
        )
        print(
            f"[eval] first_hit_response_count={int(response_metrics['ss_first_hit_response_count'])} "
            f"mean_s={response_metrics['ss_first_hit_response_mean_s']:.4f} "
            f"min_s={response_metrics['ss_first_hit_response_min_s']:.4f} "
            f"max_s={response_metrics['ss_first_hit_response_max_s']:.4f}"
        )

        ss_response_samples = response_metrics.get("ss_response_episode_samples", [])
        if isinstance(ss_response_samples, list) and len(ss_response_samples) > 0:
            print("[eval] ss_response_episode_samples(first up to 50):")
            for sample in ss_response_samples:
                if not isinstance(sample, dict):
                    continue
                fire_s = sample.get("first_fire_s")
                hit_s = sample.get("first_hit_s")
                fire_text = "NA" if fire_s is None else f"{float(fire_s):.4f}"
                hit_text = "NA" if hit_s is None else f"{float(hit_s):.4f}"
                print(
                    f"[eval]   ep#{int(sample.get('episode', 0)):02d}: "
                    f"first_fire_s={fire_text} first_hit_s={hit_text}"
                )

        print("\n[eval] ===== SD Response Metrics =====")
        print(
            f"[eval] sd_completed_episodes={int(response_metrics['sd_response_completed_episodes'])} "
            f"sd_response_started_episodes={int(response_metrics['sd_response_started_episodes'])}"
        )
        print(
            f"[eval] first_fire_response_count={int(response_metrics['sd_first_fire_response_count'])} "
            f"mean_s={response_metrics['sd_first_fire_response_mean_s']:.4f} "
            f"min_s={response_metrics['sd_first_fire_response_min_s']:.4f} "
            f"max_s={response_metrics['sd_first_fire_response_max_s']:.4f}"
        )
        print(
            f"[eval] first_hit_response_count={int(response_metrics['sd_first_hit_response_count'])} "
            f"mean_s={response_metrics['sd_first_hit_response_mean_s']:.4f} "
            f"min_s={response_metrics['sd_first_hit_response_min_s']:.4f} "
            f"max_s={response_metrics['sd_first_hit_response_max_s']:.4f}"
        )

        response_samples = response_metrics.get("sd_response_episode_samples", [])
        if isinstance(response_samples, list) and len(response_samples) > 0:
            print("[eval] sd_response_episode_samples(first up to 50):")
            for sample in response_samples:
                if not isinstance(sample, dict):
                    continue
                fire_s = sample.get("first_fire_s")
                hit_s = sample.get("first_hit_s")
                fire_text = "NA" if fire_s is None else f"{float(fire_s):.4f}"
                hit_text = "NA" if hit_s is None else f"{float(hit_s):.4f}"
                print(
                    f"[eval]   ep#{int(sample.get('episode', 0)):02d}: "
                    f"first_fire_s={fire_text} first_hit_s={hit_text}"
                )

    sim_app.close()


if __name__ == "__main__":
    main()

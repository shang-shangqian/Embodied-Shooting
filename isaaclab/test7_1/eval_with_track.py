from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import asym_central_value_gru_network  # noqa: F401
from isaaclab.app import AppLauncher
from target_follow_motion import install_target_follow_motion
import types

from rl_perception_bridge import install_rl_perception_bridge


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

        muzzle_vel_w = base_env._compute_muzzle_velocity_w(body_pos_w=body_pos_w, muzzle_pos_w=muzzle_pos_w)
        bullet_pos = muzzle_pos_w[env_id : env_id + 1].clone()
        bullet_vel = muzzle_vel_w[env_id : env_id + 1] + muzzle_fwd_w[env_id : env_id + 1] * base_env.cfg.muzzle_speed

        polyline: list[list[float]] = [bullet_pos.squeeze(0).detach().cpu().tolist()]
        for i, dt in enumerate(base_env._ballistic_dt_samples):
            dt = dt.view(1, 1)
            if base_env.cfg.enable_air_drag:
                speed = bullet_vel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                drag_acc = -base_env._drag_accel_coeff * speed * bullet_vel
            else:
                drag_acc = bullet_vel * 0.0
            total_acc = base_env._gravity_w.unsqueeze(0) + drag_acc
            bullet_vel = bullet_vel + total_acc * dt
            bullet_pos = bullet_pos + bullet_vel * dt
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
    sd_fire_delay_s: float = 0.0,
    freeze_sd_aim_during_delay: bool = True,
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
    base_env._mix_strict_response_no_hidden_cache = True
    base_env._mix_sd_fire_delay_steps = max(0, int(round(float(sd_fire_delay_s) / float(base_env.step_dt))))
    base_env._mix_freeze_sd_aim_during_delay = (
        bool(freeze_sd_aim_during_delay) or bool(base_env._mix_strict_response_no_hidden_cache)
    )
    base_env._mix_fire_delay_left_steps = torch.zeros(
        (base_env.num_envs,),
        dtype=torch.long,
        device=base_env.device,
    )
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

        ss_mask = case_ids == 0
        sd_mask = case_ids == 1
        response_mask = ss_mask | sd_mask
        base_env._mix_fire_delay_left_steps[env_ids_t] = torch.where(
            sd_mask,
            torch.full_like(case_ids, int(base_env._mix_sd_fire_delay_steps)),
            torch.zeros_like(case_ids),
        )
        base_env._mix_episode_steps[env_ids_t] = 0
        base_env._mix_response_track_mask[env_ids_t] = response_mask
        base_env._mix_response_case_id[env_ids_t] = torch.where(
            response_mask,
            case_ids,
            torch.full_like(case_ids, -1),
        )
        base_env._mix_first_fire_step[env_ids_t] = -1
        base_env._mix_first_hit_step[env_ids_t] = -1

        no_delay_sd = sd_mask & (base_env._mix_fire_delay_left_steps[env_ids_t] <= 0)
        base_env._mix_response_start_step[env_ids_t] = torch.where(
            ss_mask | no_delay_sd,
            torch.zeros_like(case_ids),
            torch.full_like(case_ids, -1),
        )

        if torch.any(robot_static):
            static_ids = env_ids_t[robot_static]
            base_env._aim_warmup_left_steps[static_ids] = 0
            base_env._locomotion_commands[static_ids] = 0.0
            base_env._locomotion_last_actions[static_ids] = 0.0
            base_env._locomotion_command_time_left[static_ids] = base_env.cfg.locomotion_command_resample_s

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

    def _clear_response_perception_state(
        self,
        env_ids_t: torch.Tensor,
        reset_vision_history: bool = True,
    ):
        # Strict response timing: no target predictions or velocity history may leak from pre-open frames.
        if env_ids_t.numel() == 0:
            return

        env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long)

        if hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None:
            self._rl_perception_bridge.reset_envs(env_ids_t)

        for name in (
            "_rl_pred_pos_base",
            "_rl_pred_vel_base",
            "_rl_pred_distance",
            "_rl_pred_yaw_base",
            "_rl_pred_pitch_base",
        ):
            if hasattr(self, name):
                getattr(self, name)[env_ids_t] = 0.0

        for name in (
            "_rl_pred_ready",
            "_rl_bridge_training_ready",
            "_rl_bridge_velocity_ready",
        ):
            if hasattr(self, name):
                getattr(self, name)[env_ids_t] = False

        if reset_vision_history:
            for name in (
                "_vision_rgb_history",
                "_vision_direction_gt",
                "_vision_velocity_gt",
                "_vision_velocity_base_gt",
                "_vision_distance_gt",
                "_vision_pos_base_gt",
                "_raycast_hit_distance",
                "_raycast_refined_yaw",
                "_raycast_refined_pitch",
            ):
                if hasattr(self, name):
                    getattr(self, name)[env_ids_t] = 0.0

            for name in (
                "_vision_frame_valid_history",
                "_vision_valid_mask",
                "_raycast_hit_valid",
            ):
                if hasattr(self, name):
                    getattr(self, name)[env_ids_t] = False

            if hasattr(self, "_vision_history_count"):
                self._vision_history_count[env_ids_t] = 0

    def _mask_pre_response_perception_state(self):
        pre_response_mask = self._mix_response_track_mask & (self._mix_response_start_step < 0)
        if not torch.any(pre_response_mask):
            return
        pre_response_ids = torch.nonzero(pre_response_mask, as_tuple=False).squeeze(-1)
        _clear_response_perception_state(self, pre_response_ids, reset_vision_history=True)

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
            "freeze_sd_aim_during_delay": bool(self._mix_freeze_sd_aim_during_delay),
            "strict_response_no_hidden_cache": bool(self._mix_strict_response_no_hidden_cache),
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

    def patched_reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        _record_completed_response_episodes(self, env_ids_t)
        result = original_reset_idx(env_ids_t)
        _resample_cases(env_ids_t)
        _apply_case_masks(env_ids_t)
        if hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None:
            self._rl_perception_bridge.reset_envs(env_ids_t)
        return result

    def patched_pre_physics_step(self, actions):
        current_steps = self._mix_episode_steps.clone()

        robot_static_mask = ~self._mix_robot_dynamic
        if torch.any(robot_static_mask):
            self._aim_warmup_left_steps[robot_static_mask] = 0
            self._locomotion_commands[robot_static_mask] = 0.0
            self._locomotion_command_time_left[robot_static_mask] = self.cfg.locomotion_command_resample_s

        actions_for_step = actions
        sd_fire_delay_mask = self._mix_fire_delay_left_steps > 0
        if torch.any(sd_fire_delay_mask):
            actions_for_step = actions.clone()
            if self._mix_freeze_sd_aim_during_delay:
                actions_for_step[sd_fire_delay_mask, 0:2] = 0.0
            actions_for_step[sd_fire_delay_mask, 2] = -1.0

        original_pre_physics_step(actions_for_step)

        response_open_mask = (
            self._mix_response_track_mask
            & (self._mix_response_start_step >= 0)
            & (current_steps >= self._mix_response_start_step)
        )
        if torch.any(response_open_mask):
            fire_now = self._fire_command.bool()
            first_fire_mask = response_open_mask & (self._mix_first_fire_step < 0) & fire_now
            if torch.any(first_fire_mask):
                self._mix_first_fire_step[first_fire_mask] = current_steps[first_fire_mask]

            if hasattr(self, "_compute_current_step_hit"):
                hit_now = self._compute_current_step_hit()
                first_hit_mask = response_open_mask & (self._mix_first_hit_step < 0) & hit_now
                if torch.any(first_hit_mask):
                    self._mix_first_hit_step[first_hit_mask] = current_steps[first_hit_mask]

        if torch.any(sd_fire_delay_mask):
            self._mix_fire_delay_left_steps[sd_fire_delay_mask] = torch.clamp(
                self._mix_fire_delay_left_steps[sd_fire_delay_mask] - 1,
                min=0,
            )
            delay_finished_mask = (
                sd_fire_delay_mask
                & self._mix_response_track_mask
                & (self._mix_fire_delay_left_steps <= 0)
                & (self._mix_response_start_step < 0)
            )
            if torch.any(delay_finished_mask):
                self._mix_response_start_step[delay_finished_mask] = current_steps[delay_finished_mask] + 1
                delay_finished_ids = torch.nonzero(delay_finished_mask, as_tuple=False).squeeze(-1)
                _clear_response_perception_state(self, delay_finished_ids, reset_vision_history=True)

        self._mix_episode_steps += 1

        target_static_mask = ~self._mix_target_dynamic
        if torch.any(target_static_mask):
            self._target_vel_w[target_static_mask] = 0.0
            if hasattr(self, "_target_acc_w"):
                self._target_acc_w[target_static_mask] = 0.0

    def get_mix_case_stats(self):
        counts = self._mix_case_counts.detach().cpu().to(dtype=torch.float32)
        total = float(counts.sum().item())
        if total <= 0.0:
            return {"ss": 0.0, "sd": 0.0, "ds": 0.0, "dd": 0.0, "total": 0.0}
        return {
            "ss": float((counts[0] / total).item()),
            "sd": float((counts[1] / total).item()),
            "ds": float((counts[2] / total).item()),
            "dd": float((counts[3] / total).item()),
            "total": total,
        }

    base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)
    base_env._pre_physics_step = types.MethodType(patched_pre_physics_step, base_env)
    base_env._mix_clear_response_perception_state = types.MethodType(_clear_response_perception_state, base_env)
    base_env._mix_mask_pre_response_perception_state = types.MethodType(_mask_pre_response_perception_state, base_env)
    base_env.reset_sd_response_metrics = types.MethodType(reset_sd_response_metrics, base_env)
    base_env.get_sd_response_metrics = types.MethodType(get_sd_response_metrics, base_env)
    base_env.get_mix_case_stats = types.MethodType(get_mix_case_stats, base_env)
    print(
        "[eval] mixed-case logic installed | "
        f"sd_fire_delay_s={float(sd_fire_delay_s):.3f} "
        f"steps={int(base_env._mix_sd_fire_delay_steps)} "
        f"freeze_sd_aim_during_delay={bool(base_env._mix_freeze_sd_aim_during_delay)} "
        f"strict_response_no_hidden_cache={bool(base_env._mix_strict_response_no_hidden_cache)}"
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
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_envs", type=int, default=16)

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
        default="source/standalone/test4/vision_model/vision_runs/vision_runs_cnn_only/04-27-17-54/vision_best.pth",
        help="Path to vision model checkpoint used by rl_perception_bridge.",
    )
    parser.add_argument(
        "--velocity_model",
        type=str,
        default="source/standalone/test11/speed_prediction/gru_runs/04-28-13-59/best_model.pt",
        help="Path to velocity GRU checkpoint used by rl_perception_bridge. Required when bridge is enabled.",
    )
    parser.add_argument(
        "--velocity_model_def",
        type=str,
        default="source/standalone/test11/speed_prediction/speed_prediction_model.py",
        help="Path to speed_prediction_model.py used to define the velocity GRU. Optional if it is next to --velocity_model.",
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
        help="Allow bridge to fallback to GT distance when predicted/lidar distance is unavailable. Disabled by default for strict camera+raycast eval.",
    )
    parser.add_argument(
        "--bridge_use_gt_velocity_until_buffer_ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow bridge to fallback to GT velocity until GRU sequence buffer is ready. Disabled by default for strict velocity-module eval.",
    )
    parser.add_argument(
        "--bridge_velocity_sequence_stride",
        type=int,
        default=3,
        help="Env-step stride for velocity-GRU sequence samples. Default matches test_temp collection.",
    )

    parser.add_argument(
        "--case",
        type=str,
        default="mixed",
        choices=["ss", "sd", "ds", "dd", "mixed"],
        help="Evaluation scenario.",
    )

    parser.add_argument("--target_speed_min", type=float, default=0.6)
    parser.add_argument("--target_speed_max", type=float, default=5)
    parser.add_argument("--target_vz_min", type=float, default=-0.5)
    parser.add_argument("--target_vz_max", type=float, default=0.5)
    parser.add_argument(
        "--sd_fire_delay_s",
        type=float,
        default=2.0,
        help="In SD episodes, let the target move for this many seconds before fire is allowed.",
    )
    parser.add_argument(
        "--sd_freeze_aim_during_delay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In SD episodes, freeze yaw/pitch during sd_fire_delay_s. Strict response timing keeps this enabled.",
    )

    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--print_case_stats", action="store_true", default=True)
    parser.add_argument("--episodes", type=int, default=10)
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

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, which=args.which)
    print(f"[eval] checkpoint = {checkpoint_path}")

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
    env_cfg.scene.num_envs = int(args.num_envs)

    # -----------------------------
    # 关键：关闭 Fabric
    # 避免 XFormPrimView 在 camera 初始化时走 usdrt.hierarchy 分支
    # -----------------------------
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "use_fabric"):
        env_cfg.sim.use_fabric = False
        print("[eval] env_cfg.sim.use_fabric = False")

    # 与训练时保持一致：底层环境统一支持 moving base + dynamic target，
    # 再通过 case mask 在每个 episode 约束成 ss/sd/ds/dd。
    env_cfg.training_stage = 2
    env_cfg.enable_locomotion_policy = True
    env_cfg.locomotion_policy_path = args.locomotion_policy
    env_cfg.enable_dynamic_target = True
    env_cfg.target_speed_range = (float(args.target_speed_min), float(args.target_speed_max))
    env_cfg.target_vertical_speed_range = (float(args.target_vz_min), float(args.target_vz_max))
    env_cfg.enable_air_drag = True
    env_cfg.target_acc_range = (0.0, 0.0)
    env_cfg.target_vertical_acc_range = (0.0, 0.0)
    env_cfg.debug_vis = not args.headless
    env_cfg.debug_diagnostics = False

    if args.enable_rl_perception_bridge:
        env_cfg.enable_vision_supervision = True
        env_cfg.vision_history_length = 1
        env_cfg.enable_dynamic_target_raycast_debug = True
        print("[eval] perception bridge sensors enabled | camera=True raycast=True")

    task_id = "Standalone-TurretAim-Mixed-Eval-v0"
    if task_id not in gym.registry:
        gym.register(id=task_id, entry_point=TurretAimEnv)

    env = gym.make(task_id, cfg=env_cfg, disable_env_checker=True)
    base_env: Any = env.unwrapped

    # ---------------------------------------------------------
    # one-shot bootstrap for delayed TiledCamera initialization
    # 必须在 RL perception bridge 安装之前完成
    # ---------------------------------------------------------
    if hasattr(base_env, "bootstrap_vision_camera"):
        base_env.bootstrap_vision_camera()
        print("[eval] bootstrap_vision_camera() done.")

    _install_mixed_training_logic(
        base_env,
        case_probs=_case_to_probs(args.case),
        sd_fire_delay_s=float(args.sd_fire_delay_s),
        freeze_sd_aim_during_delay=bool(args.sd_freeze_aim_during_delay),
    )

    install_target_follow_motion(
        base_env,
        follow_speed=0.35,
        stop_speed_threshold=0.05,
        yaw_follow_gain=0.8,
        max_yaw_rate=0.35,
        blend=0.7,
        pos_gain=0.75,
        vel_gain=0.25,
        close_distance=1.5,
        enable_debug=False,
    )

    # ---------------------------------------------------------
    # Install RL perception bridge for eval
    # ---------------------------------------------------------
    if args.enable_rl_perception_bridge:
        if not args.vision_model.strip():
            raise ValueError(
                "--enable_rl_perception_bridge is True, but --vision_model is empty."
            )
        if not args.velocity_model.strip():
            raise ValueError(
                "--enable_rl_perception_bridge is True, but --velocity_model is empty. "
                "Pass the velocity GRU checkpoint trained with test_temp/train_gru_velocity.py."
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
            debug=args.bridge_debug,
        )

        if hasattr(base_env, "_use_gt_fallback_for_missing_rl_pred"):
            base_env._use_gt_fallback_for_missing_rl_pred = False
            print("[eval] disabled env GT fallback for strict module-only response timing.")

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
            if hasattr(self, "_mix_mask_pre_response_perception_state"):
                self._mix_mask_pre_response_perception_state()
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

    if args.print_case_stats:
        print(f"[eval] case = {args.case}")

    if env_cfg.debug_vis and bool(args.draw_tracks):
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

    rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
            "seed": 61,
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
                "minibatch_size": 128,
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

    run_args: dict[str, Any] = {
        "train": False,
        "play": True,
        "checkpoint": str(checkpoint_path),
    }

    print("[eval] starting play mode ...")
    runner.run(run_args)

    if hasattr(base_env, "get_eval_metrics"):
        metrics = base_env.get_eval_metrics()
        print("\n[eval] ===== Aggregated Metrics =====")
        print(f"[eval] episodes={int(metrics['episodes'])}")
        print(f"[eval] fires={int(metrics['fires'])}")
        print(f"[eval] hits={int(metrics['hits'])}")
        print(f"[eval] hit_rate={metrics['hit_rate']:.4f}")
        accuracy_denom = max(float(metrics["fires"]), 1.0)
        print(f"[eval] accuracy_rate={float(metrics['hits'])/accuracy_denom:.4f}")
        print(f"[eval] avg_fires_per_episode={metrics['avg_fires_per_episode']:.4f}")
        print(f"[eval] timeouts={int(metrics['timeouts'])} timeout_rate={metrics['timeout_rate']:.4f}")
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
        print(f"[eval] freeze_sd_aim_during_delay={bool(response_metrics['freeze_sd_aim_during_delay'])}")
        print(f"[eval] strict_response_no_hidden_cache={bool(response_metrics['strict_response_no_hidden_cache'])}")
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

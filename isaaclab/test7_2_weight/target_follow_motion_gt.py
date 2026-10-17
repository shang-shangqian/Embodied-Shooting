from __future__ import annotations

import types
from typing import Any

import torch


def install_target_follow_motion(
    base_env: Any,
    follow_speed: float = 0.35,
    follow_speed_range: tuple[float, float] | None = None,
    stop_speed_threshold: float = 0.05,
    yaw_follow_gain: float = 0.8,
    max_yaw_rate: float = 0.35,
    blend: float = 0.7,
    pos_gain: float = 0.75,
    vel_gain: float = 0.25,
    close_distance: float = 1.5,
    command_resample_s_range: tuple[float, float] | None = None,
    direction_noise_rad: float = 0.0,
    lateral_noise_range: tuple[float, float] = (0.0, 0.0),
    yaw_rate_noise_range: tuple[float, float] = (0.0, 0.0),
    enable_debug: bool = False,
):
    """Patch the env locomotion command with randomized target-follow motion.

    Dynamic robot cases receive commands that are mostly pointed toward the
    target, with per-env randomized speed, direction, lateral, and yaw-rate
    perturbations. Static robot cases are forced to zero command.
    """

    if not hasattr(base_env, "_target_vel_w"):
        raise AttributeError("base_env has no attribute '_target_vel_w'")
    if not hasattr(base_env, "_target_pos_w"):
        raise AttributeError("base_env has no attribute '_target_pos_w'")
    if not hasattr(base_env, "_locomotion_commands"):
        raise AttributeError("base_env has no attribute '_locomotion_commands'")
    if not hasattr(base_env, "_rotate_world_to_local"):
        raise AttributeError("base_env has no method '_rotate_world_to_local'")
    if not hasattr(base_env, "turret") or not hasattr(base_env.turret, "data"):
        raise AttributeError("base_env.turret.data is required")
    if not hasattr(base_env, "_update_locomotion_commands"):
        raise AttributeError("base_env has no method '_update_locomotion_commands'")

    def _ordered_pair(values: tuple[float, float], name: str) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        low = float(values[0])
        high = float(values[1])
        return (low, high) if low <= high else (high, low)

    follow_speed_min, follow_speed_max = _ordered_pair(
        follow_speed_range if follow_speed_range is not None else (follow_speed, follow_speed),
        "follow_speed_range",
    )
    if follow_speed_min < 0.0:
        raise ValueError("follow_speed_range is a speed magnitude and must be >= 0")

    if command_resample_s_range is None:
        command_resample_s_range = (
            float(base_env.cfg.locomotion_command_resample_s),
            float(base_env.cfg.locomotion_command_resample_s),
        )
    command_resample_min_s, command_resample_max_s = _ordered_pair(
        command_resample_s_range,
        "command_resample_s_range",
    )
    command_resample_min_s = max(0.0, command_resample_min_s)
    command_resample_max_s = max(command_resample_min_s, command_resample_max_s)

    lateral_noise_min, lateral_noise_max = _ordered_pair(lateral_noise_range, "lateral_noise_range")
    yaw_noise_min, yaw_noise_max = _ordered_pair(yaw_rate_noise_range, "yaw_rate_noise_range")
    direction_noise_rad = abs(float(direction_noise_rad))

    original_update_locomotion_commands = base_env._update_locomotion_commands
    original_reset_idx = base_env._reset_idx

    base_env._target_follow_enabled = True
    base_env._target_follow_prev_cmd = torch.zeros(
        (base_env.num_envs, 3),
        dtype=torch.float32,
        device=base_env.device,
    )
    base_env._target_follow_speed = torch.full(
        (base_env.num_envs,),
        float(follow_speed),
        dtype=torch.float32,
        device=base_env.device,
    )
    base_env._target_follow_dir_noise = torch.zeros(
        (base_env.num_envs,),
        dtype=torch.float32,
        device=base_env.device,
    )
    base_env._target_follow_lateral_noise = torch.zeros(
        (base_env.num_envs,),
        dtype=torch.float32,
        device=base_env.device,
    )
    base_env._target_follow_yaw_noise = torch.zeros(
        (base_env.num_envs,),
        dtype=torch.float32,
        device=base_env.device,
    )
    base_env._target_follow_param_time_left = torch.zeros(
        (base_env.num_envs,),
        dtype=torch.float32,
        device=base_env.device,
    )

    def reset_target_follow_state(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)

        if env_ids_t.numel() == 0:
            return

        self._target_follow_prev_cmd[env_ids_t] = 0.0
        self._target_follow_speed[env_ids_t] = 0.0
        self._target_follow_dir_noise[env_ids_t] = 0.0
        self._target_follow_lateral_noise[env_ids_t] = 0.0
        self._target_follow_yaw_noise[env_ids_t] = 0.0
        self._target_follow_param_time_left[env_ids_t] = 0.0
        self._locomotion_commands[env_ids_t] = 0.0
        self._locomotion_command_time_left[env_ids_t] = 0.0

    def patched_reset_idx(self, env_ids):
        result = original_reset_idx(env_ids)
        self.reset_target_follow_state(env_ids)
        return result

    def _sample_uniform(self, low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
        if high <= low:
            return torch.full(shape, float(low), dtype=torch.float32, device=self.device)
        return torch.empty(shape, dtype=torch.float32, device=self.device).uniform_(float(low), float(high))

    def _resample_follow_params(self, env_ids_t: torch.Tensor) -> None:
        if env_ids_t.numel() == 0:
            return

        count = int(env_ids_t.numel())
        self._target_follow_speed[env_ids_t] = _sample_uniform(
            self, follow_speed_min, follow_speed_max, (count,)
        )
        self._target_follow_dir_noise[env_ids_t] = _sample_uniform(
            self, -direction_noise_rad, direction_noise_rad, (count,)
        )
        self._target_follow_lateral_noise[env_ids_t] = _sample_uniform(
            self, lateral_noise_min, lateral_noise_max, (count,)
        )
        self._target_follow_yaw_noise[env_ids_t] = _sample_uniform(
            self, yaw_noise_min, yaw_noise_max, (count,)
        )
        self._target_follow_param_time_left[env_ids_t] = _sample_uniform(
            self, command_resample_min_s, command_resample_max_s, (count,)
        )

    def patched_update_locomotion_commands(self):
        if self._locomotion_policy is None:
            original_update_locomotion_commands()
            return

        if hasattr(self, "_mix_robot_dynamic"):
            robot_dynamic_mask = self._mix_robot_dynamic.to(dtype=torch.bool)
        else:
            robot_dynamic_mask = torch.ones(
                (self.num_envs,),
                dtype=torch.bool,
                device=self.device,
            )

        if torch.any(robot_dynamic_mask):
            self._target_follow_param_time_left[robot_dynamic_mask] -= float(self.step_dt)
            resample_mask = robot_dynamic_mask & (self._target_follow_param_time_left <= 0.0)
            if torch.any(resample_mask):
                resample_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
                _resample_follow_params(self, resample_ids)

        static_robot_mask = ~robot_dynamic_mask
        if torch.any(static_robot_mask):
            self._target_follow_param_time_left[static_robot_mask] = 0.0
            self._target_follow_speed[static_robot_mask] = 0.0
            self._target_follow_dir_noise[static_robot_mask] = 0.0
            self._target_follow_lateral_noise[static_robot_mask] = 0.0
            self._target_follow_yaw_noise[static_robot_mask] = 0.0

        root_pos_w = self.turret.data.root_pos_w
        root_quat_w = self.turret.data.root_quat_w

        rel_pos_w = self._target_pos_w - root_pos_w
        rel_pos_base = self._rotate_world_to_local(root_quat_w, rel_pos_w)
        rel_xy_base = rel_pos_base[:, :2]
        dist_xy = torch.norm(rel_xy_base, dim=-1, keepdim=True)
        pos_dir_xy = rel_xy_base / dist_xy.clamp_min(1e-6)

        target_vel_xy_w = torch.zeros_like(self._target_vel_w)
        target_vel_xy_w[:, :2] = self._target_vel_w[:, :2]
        target_vel_base = self._rotate_world_to_local(root_quat_w, target_vel_xy_w)
        vel_xy_base = target_vel_base[:, :2]
        vel_norm = torch.norm(vel_xy_base, dim=-1, keepdim=True)
        vel_dir_xy = vel_xy_base / vel_norm.clamp_min(1e-6)

        moving_mask = vel_norm > float(stop_speed_threshold)
        desired_dir_xy = float(pos_gain) * pos_dir_xy + float(vel_gain) * vel_dir_xy
        desired_dir_xy = torch.where(moving_mask, desired_dir_xy, pos_dir_xy)
        desired_dir_xy = desired_dir_xy / torch.norm(desired_dir_xy, dim=-1, keepdim=True).clamp_min(1e-6)

        noise_angle = self._target_follow_dir_noise
        noise_cos = torch.cos(noise_angle)
        noise_sin = torch.sin(noise_angle)
        desired_dir_xy = torch.stack(
            [
                desired_dir_xy[:, 0] * noise_cos - desired_dir_xy[:, 1] * noise_sin,
                desired_dir_xy[:, 0] * noise_sin + desired_dir_xy[:, 1] * noise_cos,
            ],
            dim=-1,
        )
        desired_dir_xy = desired_dir_xy / torch.norm(desired_dir_xy, dim=-1, keepdim=True).clamp_min(1e-6)

        speed_scale = (dist_xy / max(float(close_distance), 1e-6)).clamp(0.25, 1.0)
        desired_speed = self._target_follow_speed.unsqueeze(-1) * speed_scale
        desired_xy = desired_dir_xy * desired_speed

        lateral_dir_xy = torch.stack([-desired_dir_xy[:, 1], desired_dir_xy[:, 0]], dim=-1)
        desired_xy = desired_xy + lateral_dir_xy * self._target_follow_lateral_noise.unsqueeze(-1)

        desired_heading = torch.atan2(rel_xy_base[:, 1], rel_xy_base[:, 0])
        desired_wz = float(yaw_follow_gain) * desired_heading
        desired_wz = torch.clamp(desired_wz, min=-float(max_yaw_rate), max=float(max_yaw_rate))
        desired_wz = desired_wz + self._target_follow_yaw_noise

        desired_cmd = torch.zeros_like(self._locomotion_commands)
        desired_cmd[:, 0] = desired_xy[:, 0]
        desired_cmd[:, 1] = desired_xy[:, 1]
        desired_cmd[:, 2] = desired_wz

        vx_min, vx_max = self.cfg.locomotion_lin_vel_x_range
        vy_min, vy_max = self.cfg.locomotion_lin_vel_y_range
        wz_min, wz_max = self.cfg.locomotion_ang_vel_z_range

        desired_cmd[:, 0] = torch.clamp(desired_cmd[:, 0], float(vx_min), float(vx_max))
        desired_cmd[:, 1] = torch.clamp(desired_cmd[:, 1], float(vy_min), float(vy_max))
        desired_cmd[:, 2] = torch.clamp(desired_cmd[:, 2], float(wz_min), float(wz_max))

        alpha = float(blend)
        robot_dynamic_mask_f = robot_dynamic_mask.unsqueeze(-1).to(dtype=desired_cmd.dtype)

        self._target_follow_prev_cmd = torch.where(
            robot_dynamic_mask_f.bool(),
            self._target_follow_prev_cmd,
            torch.zeros_like(self._target_follow_prev_cmd),
        )

        if alpha > 0.0:
            desired_cmd = alpha * self._target_follow_prev_cmd + (1.0 - alpha) * desired_cmd

        desired_cmd = torch.where(
            robot_dynamic_mask_f.bool(),
            desired_cmd,
            torch.zeros_like(desired_cmd),
        )

        self._locomotion_commands.copy_(desired_cmd)
        self._target_follow_prev_cmd.copy_(desired_cmd)
        self._locomotion_command_time_left[:] = self.cfg.locomotion_command_resample_s

        if enable_debug and (int(self.common_step_counter) % 50 == 0):
            env0_rel = rel_xy_base[0].detach().cpu().tolist()
            env0_vel = vel_xy_base[0].detach().cpu().tolist()
            env0_cmd = desired_cmd[0].detach().cpu().tolist()
            env0_dist = float(dist_xy[0].item())
            env0_speed = float(self._target_follow_speed[0].detach().cpu().item())
            env0_dir_noise = float(self._target_follow_dir_noise[0].detach().cpu().item())
            env0_lat_noise = float(self._target_follow_lateral_noise[0].detach().cpu().item())
            env0_yaw_noise = float(self._target_follow_yaw_noise[0].detach().cpu().item())
            print(
                "[target_follow_motion]",
                f"step={int(self.common_step_counter)}",
                f"rel_xy_base={env0_rel}",
                f"vel_xy_base={env0_vel}",
                f"dist_xy={env0_dist:.3f}",
                f"speed={env0_speed:.3f}",
                f"dir_noise={env0_dir_noise:.3f}",
                f"lat_noise={env0_lat_noise:.3f}",
                f"yaw_noise={env0_yaw_noise:.3f}",
                f"cmd={env0_cmd}",
            )

    base_env._update_locomotion_commands = types.MethodType(
        patched_update_locomotion_commands, base_env
    )
    base_env.reset_target_follow_state = types.MethodType(reset_target_follow_state, base_env)
    base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)

    print(
        "[target_follow_motion] installed | "
        f"follow_speed={follow_speed}, "
        f"follow_speed_range=({follow_speed_min}, {follow_speed_max}), "
        f"stop_speed_threshold={stop_speed_threshold}, "
        f"yaw_follow_gain={yaw_follow_gain}, "
        f"max_yaw_rate={max_yaw_rate}, "
        f"blend={blend}, "
        f"pos_gain={pos_gain}, "
        f"vel_gain={vel_gain}, "
        f"close_distance={close_distance}, "
        f"command_resample_s_range=({command_resample_min_s}, {command_resample_max_s}), "
        f"direction_noise_rad={direction_noise_rad}, "
        f"lateral_noise_range=({lateral_noise_min}, {lateral_noise_max}), "
        f"yaw_rate_noise_range=({yaw_noise_min}, {yaw_noise_max})"
    )

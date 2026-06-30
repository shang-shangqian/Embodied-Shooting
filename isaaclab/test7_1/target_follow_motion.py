from __future__ import annotations

import types
from typing import Any

import torch


def install_target_follow_motion(
    base_env: Any,
    follow_speed: float = 0.35,
    stop_speed_threshold: float = 0.05,
    yaw_follow_gain: float = 0.8,
    max_yaw_rate: float = 0.35,
    blend: float = 0.7,
    pos_gain: float = 0.75,
    vel_gain: float = 0.25,
    close_distance: float = 1.5,
    enable_debug: bool = False,
):
    """
    给现有 TurretAimEnv 动态挂接“机器狗追踪目标”的 locomotion command 逻辑。

    相比旧版：
    1) 不再只跟随目标速度方向
    2) 同时考虑：
       - 靠近目标的位置方向
       - 跟随目标的速度方向
    3) 对命令做更强平滑，减小突变，降低摔倒风险

    生成 command: [vx, vy, wz]

    参数：
    - follow_speed: 期望平面移动速度大小（m/s）
    - stop_speed_threshold: 目标平面速度低于该值时，速度方向项视为无效
    - yaw_follow_gain: 朝向跟随增益
    - max_yaw_rate: 最大角速度命令
    - blend: 命令平滑系数，越大越平滑
    - pos_gain: 位置方向项权重
    - vel_gain: 速度方向项权重
    - close_distance: 距离目标足够近时，降低前进速度，避免冲过头
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

    original_update_locomotion_commands = base_env._update_locomotion_commands

    base_env._target_follow_enabled = True
    base_env._target_follow_prev_cmd = torch.zeros(
        (base_env.num_envs, 3),
        dtype=torch.float32,
        device=base_env.device,
    )

    def patched_update_locomotion_commands(self):
        # 若没启用 locomotion policy，则保持原行为
        if self._locomotion_policy is None:
            original_update_locomotion_commands()
            return

        root_pos_w = self.turret.data.root_pos_w          # [N, 3]
        root_quat_w = self.turret.data.root_quat_w        # [N, 4]

        # ------------------------------------------------------------
        # 1) 目标相对位置（world -> base）
        # ------------------------------------------------------------
        rel_pos_w = self._target_pos_w - root_pos_w       # [N, 3]
        rel_pos_base = self._rotate_world_to_local(root_quat_w, rel_pos_w)

        rel_xy_base = rel_pos_base[:, :2]                 # [N, 2]
        dist_xy = torch.norm(rel_xy_base, dim=-1, keepdim=True)  # [N, 1]
        pos_dir_xy = rel_xy_base / dist_xy.clamp_min(1e-6)

        # ------------------------------------------------------------
        # 2) 目标速度方向（world -> base）
        # ------------------------------------------------------------
        target_vel_w = self._target_vel_w
        target_vel_xy_w = torch.zeros_like(target_vel_w)
        target_vel_xy_w[:, :2] = target_vel_w[:, :2]

        target_vel_base = self._rotate_world_to_local(root_quat_w, target_vel_xy_w)
        vel_xy_base = target_vel_base[:, :2]
        vel_norm = torch.norm(vel_xy_base, dim=-1, keepdim=True)
        vel_dir_xy = vel_xy_base / vel_norm.clamp_min(1e-6)

        moving_mask = vel_norm > float(stop_speed_threshold)

        # ------------------------------------------------------------
        # 3) 位置项 + 速度项 融合
        #    优先靠近目标，同时保留一点运动趋势跟随
        # ------------------------------------------------------------
        desired_dir_xy = float(pos_gain) * pos_dir_xy + float(vel_gain) * vel_dir_xy

        # 如果目标速度太小，则只使用位置项
        desired_dir_xy = torch.where(
            moving_mask,
            desired_dir_xy,
            pos_dir_xy,
        )

        desired_dir_norm = torch.norm(desired_dir_xy, dim=-1, keepdim=True)
        desired_dir_xy = desired_dir_xy / desired_dir_norm.clamp_min(1e-6)

        # ------------------------------------------------------------
        # 4) 距离调速：离得近就减速，减少冲过头和大幅修正
        # ------------------------------------------------------------
        # 远处 -> 用满 follow_speed
        # 近处 -> 线性减速，但保留一点最小移动能力
        speed_scale = (dist_xy / max(float(close_distance), 1e-6)).clamp(0.25, 1.0)
        desired_speed = float(follow_speed) * speed_scale

        desired_xy = desired_dir_xy * desired_speed

        # ------------------------------------------------------------
        # 5) yaw 跟随：朝向优先对准目标相对位置方向，而不是目标速度方向
        # ------------------------------------------------------------
        desired_heading = torch.atan2(rel_xy_base[:, 1], rel_xy_base[:, 0])
        desired_wz = float(yaw_follow_gain) * desired_heading
        desired_wz = torch.clamp(
            desired_wz,
            min=-float(max_yaw_rate),
            max=float(max_yaw_rate),
        )

        desired_cmd = torch.zeros_like(self._locomotion_commands)
        desired_cmd[:, 0] = desired_xy[:, 0]
        desired_cmd[:, 1] = desired_xy[:, 1]
        desired_cmd[:, 2] = desired_wz

        # ------------------------------------------------------------
        # 6) 限幅到 locomotion policy 训练范围
        # ------------------------------------------------------------
        vx_min, vx_max = self.cfg.locomotion_lin_vel_x_range
        vy_min, vy_max = self.cfg.locomotion_lin_vel_y_range
        wz_min, wz_max = self.cfg.locomotion_ang_vel_z_range

        desired_cmd[:, 0] = torch.clamp(desired_cmd[:, 0], vx_min, vx_max)
        desired_cmd[:, 1] = torch.clamp(desired_cmd[:, 1], vy_min, vy_max)
        desired_cmd[:, 2] = torch.clamp(desired_cmd[:, 2], wz_min, wz_max)

        # ------------------------------------------------------------
        # 7) 一阶平滑，降低命令突变
        # ------------------------------------------------------------
        alpha = float(blend)

        if hasattr(self, "_mix_robot_dynamic"):
            robot_dynamic_mask = self._mix_robot_dynamic.to(dtype=torch.bool)
        else:
            robot_dynamic_mask = torch.ones(
                (self.num_envs,),
                dtype=torch.bool,
                device=self.device,
            )

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

        # 保持 command timer 有效，避免原随机采样逻辑重新介入
        self._locomotion_command_time_left[:] = self.cfg.locomotion_command_resample_s

        if enable_debug and (int(self.common_step_counter) % 50 == 0):
            env0_rel = rel_xy_base[0].detach().cpu().tolist()
            env0_vel = vel_xy_base[0].detach().cpu().tolist()
            env0_cmd = desired_cmd[0].detach().cpu().tolist()
            env0_dist = float(dist_xy[0].item())
            print(
                "[target_follow_motion]",
                f"step={int(self.common_step_counter)}",
                f"rel_xy_base={env0_rel}",
                f"vel_xy_base={env0_vel}",
                f"dist_xy={env0_dist:.3f}",
                f"cmd={env0_cmd}",
            )

    base_env._update_locomotion_commands = types.MethodType(
        patched_update_locomotion_commands, base_env
    )

    print(
        "[target_follow_motion] installed | "
        f"follow_speed={follow_speed}, "
        f"stop_speed_threshold={stop_speed_threshold}, "
        f"yaw_follow_gain={yaw_follow_gain}, "
        f"max_yaw_rate={max_yaw_rate}, "
        f"blend={blend}, "
        f"pos_gain={pos_gain}, "
        f"vel_gain={vel_gain}, "
        f"close_distance={close_distance}"
    )

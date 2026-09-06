from __future__ import annotations

import torch

try:
    from .turret_aim_env import TurretAimEnv, TurretAimEnvCfg
except ImportError:
    from turret_aim_env import TurretAimEnv, TurretAimEnvCfg


class TurretAimCollectEnvCfg(TurretAimEnvCfg):
    """Collection-only config.

    Keeps the same scene/asset/sensor setup, but is intended for vision data collection.
    """


class TurretAimCollectEnv(TurretAimEnv):
    """Collection-only environment.

    Goals:
    - keep warmup after reset so robot/camera can stabilize
    - only collect samples when target is truly inside image
    - reset env when target leaves image
    """

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # warmup 期间不因为目标暂时不在画面里而 reset
        warmup_done = self._aim_warmup_left_steps <= 0
        collect_warmup_done = self._vision_collect_warmup_left <= 0

        invalid_state = self._get_invalid_state_mask()

        # 只有在两层 warmup 都结束后，才因为目标不在图像里而终止
        target_out_of_image = (
            (~self._compute_target_projected_in_image_mask())
            & warmup_done
            & collect_warmup_done
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = invalid_state | target_out_of_image

        if self.cfg.debug_diagnostics and (int(self.common_step_counter) % 50 == 0):
            print(
                "[collect done debug]",
                f"step={int(self.common_step_counter)}",
                f"invalid_state={int(invalid_state.sum().item())}/{self.num_envs}",
                f"target_out_of_image={int(target_out_of_image.sum().item())}/{self.num_envs}",
                f"time_out={int(time_out.sum().item())}/{self.num_envs}",
                f"terminated={int(terminated.sum().item())}/{self.num_envs}",
            )

        return terminated, time_out

    def _refresh_vision_supervision(self) -> None:
        """Update image history + GT labels for the current sim step.

        Collection env uses the strict rule:
        current frame is valid only if target is truly inside image.
        """
        if (not self.cfg.enable_vision_supervision) or (self._vision_camera is None):
            raise RuntimeError("Vision supervision is disabled or camera is not available.")

        # ------------------------------------------------------------
        # 关键修复：
        # collect 子类虽然重写了 _refresh_vision_supervision，
        # 但如果 TiledCamera 绑定的是 proxy cameras，
        # 那么每次读图前都必须先把 runtime physics pose 同步到 proxy。
        # ------------------------------------------------------------
        self._sync_render_camera_before_read()
        rgb = self._vision_camera.data.output["rgb"]
        if rgb is None:
            raise RuntimeError("vision_camera returned no rgb output.")

        if rgb.ndim != 4:
            raise RuntimeError(f"Unexpected rgb shape: {tuple(rgb.shape)}")
        if rgb.shape[-1] < 3:
            raise RuntimeError(f"RGB channel dim must be >=3, got {rgb.shape[-1]}")
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]

        # Normalize only when needed to avoid double-normalizing float images.
        if rgb.dtype == torch.uint8:
            rgb = rgb.to(dtype=torch.float32)
            if self.cfg.vision_normalize_rgb:
                rgb = rgb / 255.0
        else:
            rgb = rgb.to(dtype=torch.float32)
            if self.cfg.vision_normalize_rgb:
                rgb_max = float(rgb.max().item()) if rgb.numel() > 0 else 0.0
                if rgb_max > 1.5:
                    rgb = rgb / 255.0

        # HWC -> CHW
        rgb = rgb.permute(0, 3, 1, 2).contiguous()  # [N, 3, H, W]

        # update frame history
        self._vision_rgb_history[:, :-1] = self._vision_rgb_history[:, 1:].clone()
        self._vision_rgb_history[:, -1] = rgb

        # ------------------------------------------------------------
        # 标签仍然使用 runtime pose
        # ------------------------------------------------------------
        cam_pos_w_rt, cam_quat_w_rt = self._get_runtime_camera_pose_w()

        self._vision_direction_gt[:] = self._compute_camera_direction_labels()
        self._vision_velocity_gt[:] = self._compute_camera_velocity_labels()
        self._vision_velocity_base_gt[:] = self._compute_base_relative_velocity_labels()

        cam_pos_finite = torch.isfinite(cam_pos_w_rt).all(dim=-1)
        cam_quat_finite = torch.isfinite(cam_quat_w_rt).all(dim=-1)
        rgb_finite = torch.isfinite(rgb).all(dim=(1, 2, 3))
        label_finite = (
            torch.isfinite(self._vision_direction_gt).all(dim=-1)
            & torch.isfinite(self._vision_velocity_gt).all(dim=-1)
            & torch.isfinite(self._vision_velocity_base_gt).all(dim=-1)
        )

        policy_warmup_done = self._aim_warmup_left_steps <= 0
        collect_warmup_done = self._vision_collect_warmup_left <= 0

        target_in_view = self._compute_target_projected_in_image_mask()

        current_frame_valid = (
            cam_pos_finite
            & cam_quat_finite
            & rgb_finite
            & label_finite
            & policy_warmup_done
            & collect_warmup_done
            & target_in_view
        )

        if self.cfg.debug_diagnostics and (int(self.common_step_counter) % 50 == 0):
            sensor_pos0 = None
            sensor_quat0 = None
            if (
                hasattr(self._vision_camera, "data")
                and hasattr(self._vision_camera.data, "pos_w")
                and self._vision_camera.data.pos_w is not None
            ):
                sensor_pos0 = [float(v) for v in self._vision_camera.data.pos_w[0].detach().cpu().tolist()]
            if (
                hasattr(self._vision_camera, "data")
                and hasattr(self._vision_camera.data, "quat_w")
                and self._vision_camera.data.quat_w is not None
            ):
                sensor_quat0 = [float(v) for v in self._vision_camera.data.quat_w[0].detach().cpu().tolist()]

            print(
                "[collect vision debug]",
                f"step={int(self.common_step_counter)}",
                f"cam_pos_finite={int(cam_pos_finite.sum().item())}/{self.num_envs}",
                f"cam_quat_finite={int(cam_quat_finite.sum().item())}/{self.num_envs}",
                f"rgb_finite={int(rgb_finite.sum().item())}/{self.num_envs}",
                f"label_finite={int(label_finite.sum().item())}/{self.num_envs}",
                f"policy_warmup_done={int(policy_warmup_done.sum().item())}/{self.num_envs}",
                f"collect_warmup_done={int(collect_warmup_done.sum().item())}/{self.num_envs}",
                f"target_in_view={int(target_in_view.sum().item())}/{self.num_envs}",
                f"valid={int(current_frame_valid.sum().item())}/{self.num_envs}",
            )
            print(
                "[camera pose dbg][collect only compare]",
                f"step={int(self.common_step_counter)}",
                f"sensor_cam_pos0={sensor_pos0}",
                f"sensor_cam_quat0={sensor_quat0}",
                f"runtime_cam_pos0={[float(v) for v in cam_pos_w_rt[0].detach().cpu().tolist()]}",
                f"runtime_cam_quat0={[float(v) for v in cam_quat_w_rt[0].detach().cpu().tolist()]}",
            )

        # update per-frame valid history
        self._vision_frame_valid_history[:, :-1] = self._vision_frame_valid_history[:, 1:].clone()
        self._vision_frame_valid_history[:, -1] = current_frame_valid

        self._vision_history_count += 1
        self._vision_history_count.clamp_(max=self._vision_history_length)

        self._vision_valid_mask[:] = (
            (self._vision_history_count >= self._vision_history_length)
            & self._vision_frame_valid_history.all(dim=-1)
        )

        # 距离 GT（runtime camera pose -> target）
        to_target_cam_w = self._target_pos_w - cam_pos_w_rt
        self._vision_distance_gt[:] = torch.norm(to_target_cam_w, dim=-1)

        # Base 坐标系下的目标真值位置
        root_pos_w = self.turret.data.root_pos_w
        root_quat_w = self.turret.data.root_quat_w
        to_target_root_w = self._target_pos_w - root_pos_w
        self._vision_pos_base_gt[:] = self._rotate_world_to_yaw_only_local(root_quat_w, to_target_root_w)

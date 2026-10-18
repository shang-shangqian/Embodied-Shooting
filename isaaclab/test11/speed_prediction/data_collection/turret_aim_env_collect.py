from __future__ import annotations

import torch
from isaaclab.utils.math import sample_uniform

from .turret_aim_env import TurretAimEnv, TurretAimEnvCfg


class TurretAimCollectEnvCfg(TurretAimEnvCfg):
    """Collection-only config.

    Keeps the same scene/asset/sensor setup, but is intended for vision data collection.
    """

    # Stage-2 collection must be allowed to observe tracking loss and recovery.
    # Paired validation always disables camera-specific out-of-view resets.
    terminate_on_target_out_of_image: bool = True
    # Validation-only counterfactual camera.  The robot keeps its real physics
    # and proprioception, while the independent render proxy stays at the
    # nominal initial world pose.  This isolates the visual camera-motion path.
    validation_fixed_camera: bool = False
    # Keep target dynamics independent of the current robot/camera pose so
    # fixed-camera and mounted-camera validation runs can replay one path.
    validation_reference_target_trajectory: bool = False
    validation_trajectory_seed: int = 0
    # SD target geometry. fixed_reference_toi uses a world-fixed reference
    # frustum; legacy_projection follows live/paired camera views.
    constrain_dynamic_target_to_camera_fov: bool = False
    # Keep the whole sphere above the world ground plane at z=0.  This is
    # independent of the legacy target_height_range, whose upper bound is
    # intentionally not used for camera-constrained 3-D target motion.
    target_ground_clearance_m: float = 0.001
    target_boundary_mode: str = "legacy_projection"
    # Fraction of image width/height inset at EACH edge of the fixed frustum.
    target_fov_margin_fraction: float = 0.10


class TurretAimCollectEnv(TurretAimEnv):
    """Collection-only environment.

    Goals:
    - keep warmup after reset so robot/camera can stabilize
    - preserve tracker loss/out-of-view events during paired validation
    - keep the robot and target physics identical across camera conditions
    """

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    def _uses_fixed_reference_target_motion(self) -> bool:
        return bool(self.cfg.constrain_dynamic_target_to_camera_fov) and (
            getattr(self.cfg, "target_boundary_mode", "legacy_projection") == "fixed_reference_toi"
        )

    def _fixed_target_motion_geometry(self):
        """Cache world-fixed inward planes, independent of live camera wobble."""
        cached = getattr(self, "_fixed_target_motion_geometry_cache", None)
        if cached is not None:
            return cached
        camera, quaternion = self._get_validation_reference_camera_pose_w()
        k = self._vision_camera.data.intrinsic_matrices
        fx, fy, cx, cy = k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]
        margin = float(self.cfg.target_fov_margin_fraction)
        if not 0 <= margin < .4:
            raise ValueError("target_fov_margin_fraction must be in [0, 0.4)")
        left = margin * float(self.cfg.camera_width)
        right = (1.-margin) * float(self.cfg.camera_width) - 1.
        top = margin * float(self.cfg.camera_height)
        bottom = (1.-margin) * float(self.cfg.camera_height) - 1.
        zero = torch.zeros_like(fx)
        local = torch.stack((
            torch.stack((zero, zero, torch.ones_like(fx)), -1),
            torch.stack((fx, zero, cx-left), -1),
            torch.stack((-fx, zero, right-cx), -1),
            torch.stack((zero, fy, cy-top), -1),
            torch.stack((zero, -fy, bottom-cy), -1),
        ), dim=1)
        local = local / torch.linalg.vector_norm(local, dim=-1, keepdim=True)
        normals = torch.stack([
            self._rotate_local_to_world(quaternion, local[:, i]) for i in range(5)
        ], dim=1).to(torch.float64)
        normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        center = camera.to(torch.float64)
        radius = float(self.cfg.target_sphere_radius)
        offsets = -(normals * center[:, None, :]).sum(-1) - radius
        offsets[:, 0] -= float(self.cfg.camera_clipping_range[0])
        ground_normal = torch.zeros_like(center)
        ground_normal[:, 2] = 1.
        normals = torch.cat((normals, ground_normal[:, None, :]), dim=1)
        ground_offset = torch.full_like(center[:, :1], -radius-float(self.cfg.target_ground_clearance_m))
        offsets = torch.cat((offsets, ground_offset), dim=1)
        self._fixed_target_motion_geometry_cache = (normals, offsets, center)
        return self._fixed_target_motion_geometry_cache

    def _initialize_fixed_target_motion(self, env_ids_t: torch.Tensor) -> None:
        """Reset-time placement only; never called by the integration step."""
        if env_ids_t.numel() == 0 or self._vision_camera is None:
            return
        from test11.speed_prediction.common.target_motion import initialize_in_domain
        normals, offsets, center = self._fixed_target_motion_geometry()
        self._target_pos_w[env_ids_t] = initialize_in_domain(
            self._target_pos_w[env_ids_t], normals[env_ids_t], offsets[env_ids_t],
            center[env_ids_t], *self.cfg.target_radius_range,
        )
        if not hasattr(self, "_target_boundary_collision_count"):
            self._target_boundary_collision_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_boundary_collision_count[env_ids_t] = 0

    def _advance_dynamic_target(self, dt_s: float) -> None:
        if not self._uses_fixed_reference_target_motion():
            return super()._advance_dynamic_target(dt_s)
        from test11.speed_prediction.common.target_motion import advance_reflecting_targets
        normals, offsets, center = self._fixed_target_motion_geometry()
        position, velocity, counts = advance_reflecting_targets(
            self._target_pos_w, self._target_vel_w, dt_s, normals, offsets, center,
            *self.cfg.target_radius_range,
        )
        self._target_pos_w.copy_(position)
        self._target_vel_w.copy_(velocity)
        self._target_boundary_collision_count = counts

    def _get_validation_reference_camera_pose_w_usd(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached nominal pose used by the world-fixed proxy camera."""
        cached_pose = getattr(self, "_validation_fixed_camera_pose_cache_usd", None)
        if cached_pose is not None:
            return cached_pose[0].clone(), cached_pose[1].clone()
        self._ensure_camera_extrinsics_from_torso()
        root_state = self.turret.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        root_state[:, 0] = self.scene.env_origins[:, 0] + self.cfg.base_init_pos[0]
        root_state[:, 1] = self.scene.env_origins[:, 1] + self.cfg.base_init_pos[1]
        root_state[:, 2] = self.scene.env_origins[:, 2] + self.cfg.base_init_pos[2]
        root_pos_w = root_state[:, :3]
        root_quat_w = root_state[:, 3:7]
        cam_local_pos = self._camera_local_pos_in_torso
        cam_local_quat = self._camera_local_quat_in_torso
        cam_pos_w = root_pos_w + self._rotate_local_to_world(root_quat_w, cam_local_pos)
        cam_quat_w_usd = self._quat_mul(root_quat_w, cam_local_quat)
        cam_quat_w_usd = cam_quat_w_usd / torch.norm(
            cam_quat_w_usd, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        self._validation_fixed_camera_pose_cache_usd = (
            cam_pos_w.detach().clone(), cam_quat_w_usd.detach().clone()
        )
        return cam_pos_w, cam_quat_w_usd

    def _camera_pose_usd_to_ros(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_w_usd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_usd_to_ros = torch.zeros_like(cam_quat_w_usd)
        q_usd_to_ros[:, 1] = 1.0
        cam_quat_w = self._quat_mul(cam_quat_w_usd, q_usd_to_ros)
        cam_quat_w = cam_quat_w / torch.norm(
            cam_quat_w, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        return cam_pos_w, cam_quat_w

    def _get_validation_reference_camera_pose_w(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the nominal fixed camera pose in the ROS optical convention."""
        return self._camera_pose_usd_to_ros(
            *self._get_validation_reference_camera_pose_w_usd()
        )

    def _get_mounted_camera_pose_w_usd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the physical torso-mounted camera pose, bypassing the intervention."""
        return super()._get_runtime_camera_pose_w_usd()

    def get_mounted_camera_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the physical torso-mounted camera pose in ROS optical coordinates."""
        return self._camera_pose_usd_to_ros(*self._get_mounted_camera_pose_w_usd())

    def _get_runtime_camera_pose_w_usd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the camera pose actually used by rendering and geometry."""
        if self.cfg.validation_fixed_camera:
            return self._get_validation_reference_camera_pose_w_usd()
        return self._get_mounted_camera_pose_w_usd()

    def _reflect_target_into_camera_frustum(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_w: torch.Tensor,
    ) -> None:
        """Keep the entire target sphere in the image and reflect it at image edges."""
        intrinsics = self._vision_camera.data.intrinsic_matrices
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        zero = torch.zeros_like(fx)
        radius = float(self.cfg.target_sphere_radius)
        width = float(self.cfg.camera_width)
        height = float(self.cfg.camera_height)
        near = float(self.cfg.camera_clipping_range[0])

        # Each plane is n dot p + offset >= 0 in ROS optical coordinates.
        planes = (
            (torch.stack((zero, zero, torch.ones_like(fx)), dim=-1),
             torch.full_like(fx, -(near + radius))),
            (torch.stack((fx, zero, cx), dim=-1), -fx * radius),
            (torch.stack((-fx, zero, width - 1.0 - cx), dim=-1), -fx * radius),
            (torch.stack((zero, fy, cy), dim=-1), -fy * radius),
            (torch.stack((zero, -fy, height - 1.0 - cy), dim=-1), -fy * radius),
        )

        # Repeated projections handle corners where two image edges meet.
        for _ in range(3):
            local_pos = self._rotate_world_to_local(
                cam_quat_w, self._target_pos_w - cam_pos_w
            )
            local_velocity = self._rotate_world_to_local(
                cam_quat_w, self._target_vel_w
            )
            for normal, offset in planes:
                norm_sq = torch.sum(normal.square(), dim=-1).clamp_min(1e-12)
                signed = torch.sum(normal * local_pos, dim=-1) + offset
                outside = signed < 0.0
                correction = torch.clamp(-signed, min=0.0) / norm_sq
                local_pos = local_pos + correction.unsqueeze(-1) * normal

                normal_rate = torch.sum(normal * local_velocity, dim=-1)
                reflection = torch.where(
                    outside,
                    2.0 * torch.clamp(-normal_rate, min=0.0) / norm_sq,
                    torch.zeros_like(normal_rate),
                )
                local_velocity = local_velocity + reflection.unsqueeze(-1) * normal

            self._target_pos_w = cam_pos_w + self._rotate_local_to_world(
                cam_quat_w, local_pos
            )
            self._target_vel_w = self._rotate_local_to_world(
                cam_quat_w, local_velocity
            )

    def _constrain_dynamic_target_to_camera_views(self) -> None:
        """Constrain SD targets by camera views, range, and the ground plane."""
        if self._vision_camera is None:
            return

        if self.cfg.validation_reference_target_trajectory:
            range_cam_pos_w, _ = self._get_validation_reference_camera_pose_w()
            fixed_pos_w, fixed_quat_w = self._get_validation_reference_camera_pose_w()
            mounted_pos_w, mounted_quat_w = self.get_mounted_camera_pose_w()
            views = (
                (fixed_pos_w, fixed_quat_w),
                (mounted_pos_w, mounted_quat_w),
            )
        else:
            range_cam_pos_w, range_cam_quat_w = self._get_runtime_camera_pose_w()
            views = ((range_cam_pos_w, range_cam_quat_w),)

        r_min = float(self.cfg.target_radius_range[0])
        r_max = float(self.cfg.target_radius_range[1])
        # Alternate range, frustum and ground projections. Ground is applied
        # last so even temporarily incompatible camera views (e.g. at reset)
        # cannot push the sphere under the floor.
        for _ in range(3):
            relative_w = self._target_pos_w - range_cam_pos_w
            distance = torch.norm(relative_w, dim=-1).clamp_min(1e-6)
            unit_w = relative_w / distance.unsqueeze(-1)
            clamped_distance = distance.clamp(min=r_min, max=r_max)
            outside_range = (distance < r_min) | (distance > r_max)
            self._target_pos_w = (
                range_cam_pos_w + unit_w * clamped_distance.unsqueeze(-1)
            )
            radial_speed = torch.sum(self._target_vel_w * unit_w, dim=-1)
            moving_outward = ((distance > r_max) & (radial_speed > 0.0)) | (
                (distance < r_min) & (radial_speed < 0.0)
            )
            reflected_radial = torch.where(
                outside_range & moving_outward,
                2.0 * radial_speed,
                torch.zeros_like(radial_speed),
            )
            self._target_vel_w = (
                self._target_vel_w - reflected_radial.unsqueeze(-1) * unit_w
            )

            for cam_pos_w, cam_quat_w in views:
                self._reflect_target_into_camera_frustum(
                    cam_pos_w, cam_quat_w
                )
            self._enforce_target_above_ground()

    def _handle_target_out_of_bounds(self) -> None:
        if bool(getattr(self.cfg, "constrain_dynamic_target_to_camera_fov", False)):
            self._constrain_dynamic_target_to_camera_views()
            return
        super()._handle_target_out_of_bounds()

    def _enforce_target_above_ground(
        self, env_ids_t: torch.Tensor | None = None
    ) -> None:
        if bool(getattr(self.cfg, "constrain_dynamic_target_to_camera_fov", False)):
            if env_ids_t is None:
                env_ids_t = torch.arange(self.num_envs, device=self.device)
            if env_ids_t.numel() == 0:
                return
            min_center_z = (
                float(self.cfg.target_sphere_radius)
                + float(self.cfg.target_ground_clearance_m)
            )
            z = self._target_pos_w[env_ids_t, 2]
            hit_ground = z < min_center_z
            self._target_pos_w[env_ids_t, 2] = z.clamp_min(min_center_z)
            vertical_velocity = self._target_vel_w[env_ids_t, 2]
            self._target_vel_w[env_ids_t, 2] = torch.where(
                hit_ground & (vertical_velocity < 0.0),
                -vertical_velocity,
                vertical_velocity,
            )
            return
        super()._enforce_target_above_ground(env_ids_t)

    def _get_target_out_of_camera_sector_mask(self) -> torch.Tensor:
        if (
            self.cfg.enable_dynamic_target
            and bool(getattr(self.cfg, "constrain_dynamic_target_to_camera_fov", False))
        ):
            cam_pos_w, cam_quat_w = self._get_runtime_camera_pose_w()
            return ~self._compute_target_projected_in_image_mask(
                cam_pos_w, cam_quat_w
            )
        return super()._get_target_out_of_camera_sector_mask()

    def _compute_camera_velocity_labels(
        self,
        cam_pos_w: torch.Tensor | None = None,
        cam_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use zero camera velocity for the fixed-camera counterfactual."""
        if not self.cfg.validation_fixed_camera:
            return super()._compute_camera_velocity_labels(cam_pos_w, cam_quat_w)
        if cam_pos_w is None or cam_quat_w is None:
            cam_pos_w, cam_quat_w = self._get_runtime_camera_pose_w()
        return self._rotate_world_to_local(cam_quat_w, self._target_vel_w)

    def _sample_targets_in_camera_sector(self, env_ids_t: torch.Tensor) -> None:
        if not self.cfg.validation_reference_target_trajectory:
            super()._sample_targets_in_camera_sector(env_ids_t)
            if self._uses_fixed_reference_target_motion():
                self._initialize_fixed_target_motion(env_ids_t)
            return
        if env_ids_t.numel() == 0:
            return
        cam_pos_w_all, cam_quat_w_all = self._get_validation_reference_camera_pose_w()
        cam_pos_w = cam_pos_w_all[env_ids_t]
        cam_quat_w = cam_quat_w_all[env_ids_t]
        env_count = int(env_ids_t.numel())
        distance = sample_uniform(
            self.cfg.target_radius_range[0],
            self.cfg.target_radius_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)
        yaw = sample_uniform(
            self.cfg.target_cam_yaw_range[0],
            self.cfg.target_cam_yaw_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)
        pitch = sample_uniform(
            self.cfg.target_cam_pitch_range[0],
            self.cfg.target_cam_pitch_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)
        target_cam = torch.stack(
            (
                distance * torch.sin(yaw) * torch.cos(pitch),
                -distance * torch.sin(pitch),
                distance * torch.cos(yaw) * torch.cos(pitch),
            ),
            dim=-1,
        )
        self._target_pos_w[env_ids_t] = cam_pos_w + self._rotate_local_to_world(
            cam_quat_w, target_cam
        )

    def _validation_generator(
        self,
        env_id: int,
        episode_index: int,
        channel: int,
    ) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        seed = (
            int(self.cfg.validation_trajectory_seed) * 1_000_003
            + int(env_id) * 10_007
            + int(episode_index) * 101
            + int(channel)
        )
        generator.manual_seed(seed)
        return generator

    def _validation_uniform(
        self,
        low: float,
        high: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return float(low) + (float(high) - float(low)) * torch.rand(
            (), device=self.device, generator=generator
        )

    def _resample_targets(
        self,
        env_ids_t: torch.Tensor,
        sample_position: bool,
        sample_velocity: bool,
    ) -> None:
        if not self.cfg.validation_reference_target_trajectory:
            super()._resample_targets(env_ids_t, sample_position, sample_velocity)
            if sample_position and self._uses_fixed_reference_target_motion():
                self._initialize_fixed_target_motion(env_ids_t)
            return
        if env_ids_t.numel() == 0:
            return
        if not hasattr(self, "_validation_target_episode_index"):
            self._validation_target_episode_index = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        reference_pos, reference_quat = self._get_validation_reference_camera_pose_w()
        for env_id_tensor in env_ids_t:
            env_id = int(env_id_tensor.item())
            episode_index = int(self._validation_target_episode_index[env_id].item())
            if sample_position:
                position_generator = self._validation_generator(env_id, episode_index, 0)
                distance = self._validation_uniform(
                    self.cfg.target_radius_range[0],
                    self.cfg.target_radius_range[1],
                    position_generator,
                )
                yaw = self._validation_uniform(
                    self.cfg.target_cam_yaw_range[0],
                    self.cfg.target_cam_yaw_range[1],
                    position_generator,
                )
                pitch = self._validation_uniform(
                    self.cfg.target_cam_pitch_range[0],
                    self.cfg.target_cam_pitch_range[1],
                    position_generator,
                )
                target_cam = torch.stack(
                    (
                        distance * torch.sin(yaw) * torch.cos(pitch),
                        -distance * torch.sin(pitch),
                        distance * torch.cos(yaw) * torch.cos(pitch),
                    )
                ).unsqueeze(0)
                target_world = reference_pos[env_id:env_id + 1] + self._rotate_local_to_world(
                    reference_quat[env_id:env_id + 1], target_cam
                )
                self._target_pos_w[env_id] = target_world[0]
                self._pending_target_resample[env_id] = False
            if sample_velocity:
                motion_generator = self._validation_generator(env_id, episode_index, 1)
                if self.cfg.enable_dynamic_target:
                    speed = self._validation_uniform(
                        self.cfg.target_speed_range[0],
                        self.cfg.target_speed_range[1],
                        motion_generator,
                    )
                    azimuth = self._validation_uniform(
                        -torch.pi, torch.pi, motion_generator
                    )
                    vertical_direction = self._validation_uniform(
                        -1.0, 1.0, motion_generator
                    )
                    horizontal_direction = torch.sqrt(
                        torch.clamp(1.0 - vertical_direction.square(), min=0.0)
                    )
                    self._target_vel_w[env_id] = speed * torch.stack(
                        (
                            horizontal_direction * torch.cos(azimuth),
                            horizontal_direction * torch.sin(azimuth),
                            vertical_direction,
                        )
                    )
                    acceleration = self._validation_uniform(
                        self.cfg.target_acc_range[0],
                        self.cfg.target_acc_range[1],
                        motion_generator,
                    )
                    acceleration_heading = self._validation_uniform(
                        -torch.pi, torch.pi, motion_generator
                    )
                    vertical_acceleration = self._validation_uniform(
                        self.cfg.target_vertical_acc_range[0],
                        self.cfg.target_vertical_acc_range[1],
                        motion_generator,
                    )
                    self._target_acc_w[env_id] = torch.stack(
                        (
                            acceleration * torch.cos(acceleration_heading),
                            acceleration * torch.sin(acceleration_heading),
                            vertical_acceleration,
                        )
                    )
                else:
                    self._target_vel_w[env_id] = 0.0
                    self._target_acc_w[env_id] = 0.0
            self._validation_target_episode_index[env_id] += 1
        if sample_position:
            self._enforce_target_above_ground(env_ids_t)
        if (
            self.cfg.enable_dynamic_target
            and bool(getattr(self.cfg, "constrain_dynamic_target_to_camera_fov", False))
        ):
            if self._uses_fixed_reference_target_motion():
                if sample_position:
                    self._initialize_fixed_target_motion(env_ids_t)
            else:
                self._constrain_dynamic_target_to_camera_views()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # warmup 期间不因为目标暂时不在画面里而 reset
        warmup_done = self._aim_warmup_left_steps <= 0
        collect_warmup_done = self._vision_collect_warmup_left <= 0

        invalid_state = self._get_invalid_state_mask()
        if self.cfg.validation_reference_target_trajectory and bool(torch.any(invalid_state)):
            invalid_ids = torch.nonzero(invalid_state, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "paired validation aborted because robot invalid-state resets would "
                f"desynchronize trajectories; env_ids={invalid_ids[:16]}"
            )

        # 只有在两层 warmup 都结束后，才因为目标不在图像里而终止
        target_out_of_image = (
            (~self._compute_target_projected_in_image_mask())
            & warmup_done
            & collect_warmup_done
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if (
            self.cfg.validation_reference_target_trajectory
            or not self.cfg.terminate_on_target_out_of_image
        ):
            target_out_of_image = torch.zeros_like(target_out_of_image)

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
        self._vision_target_relative_velocity_base_gt[:] = self._compute_target_relative_velocity_base_mps()
        self._vision_position_derivative_base_gt[:] = self._compute_target_position_derivative_base_mps()
        self._vision_velocity_base_gt[:] = self._vision_position_derivative_base_gt

        cam_pos_finite = torch.isfinite(cam_pos_w_rt).all(dim=-1)
        cam_quat_finite = torch.isfinite(cam_quat_w_rt).all(dim=-1)
        rgb_finite = torch.isfinite(rgb).all(dim=(1, 2, 3))
        label_finite = (
            torch.isfinite(self._vision_direction_gt).all(dim=-1)
            & torch.isfinite(self._vision_velocity_gt).all(dim=-1)
            & torch.isfinite(self._vision_target_relative_velocity_base_gt).all(dim=-1)
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

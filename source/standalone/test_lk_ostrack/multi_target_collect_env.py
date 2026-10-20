from __future__ import annotations

from collections.abc import Sequence
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

try:
    from test11.turret_aim_env_collect import (
        TurretAimCollectEnv,
        TurretAimCollectEnvCfg,
    )
except ModuleNotFoundError:
    from turret_aim_env_collect import TurretAimCollectEnv, TurretAimCollectEnvCfg


@configclass
class MultiTargetCollectEnvCfg(TurretAimCollectEnvCfg):
    """Isaac Sim collection scene containing several targets per camera."""

    num_visual_targets: int = 3
    min_target_separation_px: float = 48.0
    target_spawn_border_px: float = 12.0
    constrain_target_height: bool = True
    keep_targets_in_view: bool = True
    sample_target_in_image: bool = True
    add_optical_flow_features: bool = True
    target_core_radius_ratio: float = 0.82
    target_feature_offset_ratio: float = 0.78
    target_feature_radius_ratio: float = 0.16
    target_motion_mode: str = "random"
    target_depth_speed_scale: float = 0.15
    spawn_fallback_camera: bool = True
    # Keep the collector camera identical to test7_1/turret_aim_env.py.
    camera_width: int = 1024
    camera_height: int = 576
    camera_projection_type: str = "perspective"
    camera_focal_length: float = 10.0
    camera_horizontal_aperture: float = 5.76
    camera_vertical_aperture: float = 3.24
    camera_clipping_range: tuple[float, float] = (0.1, 1000000.0)
    camera_f_stop: float = 0.0
    camera_focus_distance: float = 400.0
    target_colors: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.1, 0.1),
        (0.1, 0.35, 1.0),
        (0.1, 0.9, 0.2),
        (1.0, 0.75, 0.1),
        (0.85, 0.15, 0.9),
        (0.1, 0.9, 0.9),
    )


class MultiTargetCollectEnv(TurretAimCollectEnv):
    """Extend the robot collection environment with multiple rendered targets."""

    cfg: MultiTargetCollectEnvCfg

    def __init__(self, cfg: MultiTargetCollectEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        if int(cfg.num_visual_targets) < 2:
            raise ValueError("num_visual_targets must be at least 2")
        if float(cfg.min_target_separation_px) < 0.0:
            raise ValueError("min_target_separation_px must be non-negative")
        if float(cfg.target_depth_speed_scale) < 0.0:
            raise ValueError("target_depth_speed_scale must be non-negative")
        if str(cfg.target_motion_mode) not in {"random", "camera_lateral"}:
            raise ValueError(
                "target_motion_mode must be 'random' or 'camera_lateral'"
            )
        if float(cfg.camera_focal_length) <= 0.0:
            raise ValueError("camera_focal_length must be positive")
        if float(cfg.camera_horizontal_aperture) <= 0.0:
            raise ValueError("camera_horizontal_aperture must be positive")
        if float(cfg.camera_vertical_aperture) <= 0.0:
            raise ValueError("camera_vertical_aperture must be positive")
        clip_near, clip_far = cfg.camera_clipping_range
        if float(clip_near) <= 0.0 or float(clip_far) <= float(clip_near):
            raise ValueError("camera_clipping_range must satisfy 0 < near < far")
        if float(cfg.camera_f_stop) < 0.0:
            raise ValueError("camera_f_stop must be non-negative")
        if float(cfg.camera_focus_distance) <= 0.0:
            raise ValueError("camera_focus_distance must be positive")
        super().__init__(cfg, render_mode, **kwargs)

        # Reuse the collector warmup for target resampling as well. The base
        # environment defaults to two vision-only steps, which is too short
        # for a locomotion policy to settle between recorded sequences.
        self._vision_collect_warmup_steps_after_reset = max(
            2, int(round(float(self.cfg.aim_warmup_s) / float(self.step_dt)))
        )

        target_count = int(self.cfg.num_visual_targets)
        self._multi_target_pos_w = torch.zeros(
            (self.num_envs, target_count, 3), dtype=torch.float32, device=self.device
        )
        self._multi_target_vel_w = torch.zeros_like(self._multi_target_pos_w)
        self._multi_target_pos_w[:, 0] = self._target_pos_w
        self._multi_target_vel_w[:, 0] = self._target_vel_w
        self._extra_target_prim_paths = [
            [
                f"/World/envs/env_{env_id}/TargetSphere_{target_id:02d}"
                for target_id in range(1, target_count)
            ]
            for env_id in range(self.num_envs)
        ]

    def _spawn_extra_scene_prims_before_clone(self) -> None:
        self._ensure_vision_camera_prim()
        target_count = int(self.cfg.num_visual_targets)
        colors = tuple(self.cfg.target_colors)
        if not colors:
            raise ValueError("target_colors must not be empty")
        core_radius = self._target_core_radius()
        primary_path = "/World/envs/env_0/TargetSphere"
        primary_sphere = self._ensure_primary_target_sphere(primary_path, core_radius)

        if self.cfg.add_optical_flow_features:
            primary_sphere.GetRadiusAttr().Set(core_radius)
            self._spawn_optical_flow_features(primary_path)

        for target_id in range(1, target_count):
            color = colors[target_id % len(colors)]
            sphere_cfg = sim_utils.SphereCfg(
                radius=core_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                rigid_props=None,
                collision_props=None,
            )
            target_path = f"/World/envs/env_0/TargetSphere_{target_id:02d}"
            sim_utils.spawn_sphere(
                prim_path=target_path,
                cfg=sphere_cfg,
                translation=(0.0, 0.0, 2.0 + 0.1 * target_id),
            )
            if self.cfg.add_optical_flow_features:
                self._spawn_optical_flow_features(target_path)

    def _ensure_primary_target_sphere(self, target_path: str, radius: float):
        """Reuse the base target sphere, or create it if the base config skipped it."""
        from pxr import UsdGeom

        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(target_path)
        if not root_prim.IsValid():
            sphere_cfg = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.cfg.target_sphere_color
                ),
                rigid_props=None,
                collision_props=None,
            )
            sim_utils.spawn_sphere(
                prim_path=target_path,
                cfg=sphere_cfg,
                translation=(0.0, 0.0, 2.0),
            )
            print(
                f"[multi-collect] created missing primary target sphere path={target_path}"
            )

        # Isaac Lab shape spawners use an Xform root and place the actual
        # UsdGeom shape under ``geometry/mesh``.
        mesh_path = f"{target_path}/geometry/mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Sphere):
            mesh_type = mesh_prim.GetTypeName() if mesh_prim.IsValid() else "missing"
            raise RuntimeError(
                f"Primary target sphere mesh is invalid: path={mesh_path}, type={mesh_type}"
            )
        return UsdGeom.Sphere(mesh_prim)

    def _ensure_vision_camera_prim(self) -> None:
        """Ensure the camera exists and apply the test7_1 imaging parameters."""
        if not self.cfg.spawn_fallback_camera:
            return
        from pxr import Gf, Sdf, UsdGeom

        camera_path = self.cfg.camera_prim_path_expr.replace("env_.*", "env_0")
        if any(token in camera_path for token in ("*", "[", "]")):
            raise ValueError(
                "Cannot derive the env_0 camera path from "
                f"camera_prim_path_expr={self.cfg.camera_prim_path_expr!r}"
            )
        stage = sim_utils.get_current_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        created_fallback = not (
            camera_prim.IsValid() and camera_prim.IsA(UsdGeom.Camera)
        )

        if created_fallback:
            # The failed remote reference leaves an untyped ``over`` prim at
            # this path. A stronger local Camera definition preserves the
            # mount while removing the network dependency.
            camera = UsdGeom.Camera.Define(stage, camera_path)
        else:
            camera = UsdGeom.Camera(camera_prim)

        # Apply these values even when the source camera resolves so every
        # collection run uses the same intrinsics as test7_1.
        camera.CreateProjectionAttr().Set(str(self.cfg.camera_projection_type))
        camera.CreateFocalLengthAttr().Set(float(self.cfg.camera_focal_length))
        camera.CreateFocusDistanceAttr().Set(float(self.cfg.camera_focus_distance))
        camera.CreateFStopAttr().Set(float(self.cfg.camera_f_stop))
        camera.CreateHorizontalApertureAttr().Set(
            float(self.cfg.camera_horizontal_aperture)
        )
        camera.CreateVerticalApertureAttr().Set(
            float(self.cfg.camera_vertical_aperture)
        )
        camera.CreateClippingRangeAttr().Set(
            Gf.Vec2f(
                float(self.cfg.camera_clipping_range[0]),
                float(self.cfg.camera_clipping_range[1]),
            )
        )

        if created_fallback:
            prim = camera.GetPrim()
            prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set(
                "pinhole"
            )
            prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
                Gf.Vec3d(0.0, 0.0, 0.0)
            )
            prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd).Set(
                Gf.Quatd(0.70710678, Gf.Vec3d(0.70710678, 0.0, 0.0))
            )
            prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(
                Gf.Vec3d(1.0, 1.0, 1.0)
            )
            prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
                ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
            )
        print(
            "[multi-collect] configured camera from test7_1",
            f"path={camera_path}",
            f"created_fallback={created_fallback}",
            f"focal_length={self.cfg.camera_focal_length}",
            f"aperture=({self.cfg.camera_horizontal_aperture}, "
            f"{self.cfg.camera_vertical_aperture})",
        )

    def _target_core_radius(self) -> float:
        target_radius = float(self.cfg.target_sphere_radius)
        if not self.cfg.add_optical_flow_features:
            return target_radius
        core_ratio = float(self.cfg.target_core_radius_ratio)
        offset_ratio = float(self.cfg.target_feature_offset_ratio)
        feature_ratio = float(self.cfg.target_feature_radius_ratio)
        if not 0.0 < core_ratio <= 1.0:
            raise ValueError("target_core_radius_ratio must be in (0, 1]")
        if offset_ratio < 0.0 or feature_ratio <= 0.0:
            raise ValueError("Target feature offset/radius ratios must be positive")
        if offset_ratio + feature_ratio > 1.0:
            raise ValueError("Optical-flow features must remain inside target_sphere_radius")
        return target_radius * core_ratio

    def _spawn_optical_flow_features(self, target_path: str) -> None:
        """Add visible high-contrast points without exceeding the GT target radius."""
        target_radius = float(self.cfg.target_sphere_radius)
        feature_radius = target_radius * float(self.cfg.target_feature_radius_ratio)
        feature_offset = target_radius * float(self.cfg.target_feature_offset_ratio)
        colors = ((0.015, 0.015, 0.015), (1.0, 1.0, 1.0))
        directions: list[tuple[float, float, float]] = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    norm = math.sqrt(float(x * x + y * y + z * z))
                    if norm > 0.0:
                        directions.append((x / norm, y / norm, z / norm))

        for feature_id, direction in enumerate(directions):
            feature_cfg = sim_utils.SphereCfg(
                radius=feature_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=colors[feature_id % len(colors)],
                    roughness=0.9,
                ),
                rigid_props=None,
                collision_props=None,
            )
            sim_utils.spawn_sphere(
                prim_path=f"{target_path}/OpticalFlowFeature_{feature_id:02d}",
                cfg=feature_cfg,
                translation=tuple(value * feature_offset for value in direction),
            )

    def _sample_placeholder_targets(self, env_ids_t: torch.Tensor) -> None:
        super()._sample_placeholder_targets(env_ids_t)
        if env_ids_t.numel() == 0:
            return
        self._multi_target_pos_w[env_ids_t, 0] = self._target_pos_w[env_ids_t]
        self._multi_target_vel_w[env_ids_t] = 0.0
        for target_id in range(1, int(self.cfg.num_visual_targets)):
            offset = torch.tensor(
                [0.0, 0.3 * target_id, 0.1 * (target_id % 2)],
                dtype=torch.float32,
                device=self.device,
            )
            self._multi_target_pos_w[env_ids_t, target_id] = (
                self._target_pos_w[env_ids_t] + offset
            )

    def _sample_targets_in_image(self, env_ids_t: torch.Tensor) -> None:
        """Sample separated target centers directly in each camera image."""
        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized")
        if env_ids_t.numel() == 0:
            return

        env_count = int(env_ids_t.numel())
        target_count = int(self.cfg.num_visual_targets)
        runtime_camera_pos, runtime_camera_quat = self._get_runtime_camera_pose_w()
        camera_pos = runtime_camera_pos[env_ids_t]
        camera_quat = runtime_camera_quat[env_ids_t]
        intrinsics = self._vision_camera.data.intrinsic_matrices[env_ids_t]
        fx = intrinsics[:, 0, 0].clamp_min(1e-6)
        fy = intrinsics[:, 1, 1].clamp_min(1e-6)
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        image_width = float(self.cfg.camera_width)
        image_height = float(self.cfg.camera_height)
        sphere_radius = float(self.cfg.target_sphere_radius)
        border = float(self.cfg.target_spawn_border_px)
        min_separation = float(self.cfg.min_target_separation_px)

        sampled_uv = torch.zeros((env_count, target_count, 2), device=self.device)
        sampled_pos = torch.zeros((env_count, target_count, 3), device=self.device)
        # ``--no-target_height_constraint`` removes the configured height
        # interval, but it must not allow a rendered sphere to intersect the
        # ground.  Otherwise its mathematical projection can be fully inside
        # the image while the RGB camera sees only the occluding ground plane.
        ground_clearance_height = (
            self.scene.env_origins[env_ids_t, 2] + sphere_radius + 0.02
        )
        if self.cfg.constrain_target_height:
            z_min = self.scene.env_origins[env_ids_t, 2] + max(
                float(self.cfg.target_height_range[0]), sphere_radius + 0.02
            )
            z_max = self.scene.env_origins[env_ids_t, 2] + float(
                self.cfg.target_height_range[1]
            )
            # Express the camera basis in world coordinates.  At long range the
            # allowed world-height interval occupies only a very thin strip in
            # the image.  Sampling v over the whole image and rejecting it is
            # therefore both slow and unreliable.  Instead, sample the desired
            # world height and solve directly for the camera-frame y coordinate.
            axis_x = torch.zeros((env_count, 3), device=self.device)
            axis_y = torch.zeros((env_count, 3), device=self.device)
            axis_z = torch.zeros((env_count, 3), device=self.device)
            axis_x[:, 0] = 1.0
            axis_y[:, 1] = 1.0
            axis_z[:, 2] = 1.0
            camera_x_world = quat_apply(camera_quat, axis_x)
            camera_y_world = quat_apply(camera_quat, axis_y)
            camera_z_world = quat_apply(camera_quat, axis_z)
            camera_y_world_z = camera_y_world[:, 2]

        for target_id in range(target_count):
            pending = torch.ones((env_count,), dtype=torch.bool, device=self.device)
            # At long range, the configured world-height interval occupies only
            # a thin horizontal band in the image, so allow enough retries.
            for _ in range(512):
                if not torch.any(pending):
                    break
                depth = self._uniform(
                    float(self.cfg.target_radius_range[0]),
                    float(self.cfg.target_radius_range[1]),
                    (env_count,),
                )
                radius_x = fx * sphere_radius / depth
                radius_y = fy * sphere_radius / depth
                x_low = radius_x + border
                x_high = image_width - radius_x - border
                y_low = radius_y + border
                y_high = image_height - radius_y - border
                bounds_valid = (x_high > x_low + 1.0) & (y_high > y_low + 1.0)
                u = x_low + torch.rand((env_count,), device=self.device) * (x_high - x_low).clamp_min(1.0)
                x_over_z = (u - cx) / fx
                x_cam = x_over_z * depth
                if self.cfg.constrain_target_height:
                    desired_height = z_min + torch.rand(
                        (env_count,), device=self.device
                    ) * (z_max - z_min)
                    solvable = camera_y_world_z.abs() > 1e-5
                    solved_y_cam = (
                        desired_height
                        - camera_pos[:, 2]
                        - camera_x_world[:, 2] * x_cam
                        - camera_z_world[:, 2] * depth
                    ) / torch.where(
                        solvable,
                        camera_y_world_z,
                        torch.ones_like(camera_y_world_z),
                    )
                    fallback_v = y_low + torch.rand(
                        (env_count,), device=self.device
                    ) * (y_high - y_low).clamp_min(1.0)
                    fallback_y_cam = (fallback_v - cy) / fy * depth
                    y_cam = torch.where(solvable, solved_y_cam, fallback_y_cam)
                    v = fy * y_cam / depth + cy
                    bounds_valid &= (v >= y_low) & (v <= y_high)
                else:
                    v = y_low + torch.rand(
                        (env_count,), device=self.device
                    ) * (y_high - y_low).clamp_min(1.0)
                    y_cam = (v - cy) / fy * depth

                target_cam = torch.stack(
                    [x_cam, y_cam, depth], dim=-1
                )
                target_world = camera_pos + quat_apply(camera_quat, target_cam)
                height_valid = target_world[:, 2] >= ground_clearance_height
                if self.cfg.constrain_target_height:
                    height_valid &= (target_world[:, 2] >= z_min) & (
                        target_world[:, 2] <= z_max
                    )
                valid = pending & bounds_valid & height_valid
                if target_id > 0:
                    previous_uv = sampled_uv[:, :target_id]
                    distance = torch.norm(
                        torch.stack([u, v], dim=-1).unsqueeze(1) - previous_uv,
                        dim=-1,
                    )
                    valid &= torch.all(distance >= min_separation, dim=1)
                if torch.any(valid):
                    sampled_uv[valid, target_id, 0] = u[valid]
                    sampled_uv[valid, target_id, 1] = v[valid]
                    sampled_pos[valid, target_id] = target_world[valid]
                    pending[valid] = False

            if torch.any(pending):
                failed_envs = env_ids_t[pending].detach().cpu().tolist()
                raise RuntimeError(
                    "Could not sample targets satisfying the image, separation, "
                    f"and physical ground/height constraints for envs {failed_envs} at "
                    f"target_id={target_id}. The current camera pose may make "
                    "the requested image area incompatible with the target "
                    "distance and ground plane; change the camera, distance, "
                    "or fixed height range."
                )

        self._multi_target_pos_w[env_ids_t] = sampled_pos
        self._target_pos_w[env_ids_t] = sampled_pos[:, 0]

    def _maybe_finalize_pending_target_resample(self) -> None:
        pending_ids = torch.nonzero(
            self._pending_target_resample & (self._vision_collect_warmup_left <= 0),
            as_tuple=False,
        ).squeeze(-1)
        if pending_ids.numel() == 0:
            return
        if self.cfg.sample_target_in_image:
            self._sample_targets_in_image(pending_ids)
            self._sample_multi_target_velocities(pending_ids)
            self._pending_target_resample[pending_ids] = False
            self._episode_start_target_pos_w[pending_ids] = self._target_pos_w[pending_ids]
            self._update_rendered_target_spheres(pending_ids)
            return
        super()._maybe_finalize_pending_target_resample()
        self._sample_multi_target_velocities(pending_ids)
        self._update_rendered_target_spheres(pending_ids)

    def _resample_targets(
        self,
        env_ids_t: torch.Tensor,
        sample_position: bool,
        sample_velocity: bool,
    ) -> None:
        if (
            sample_position
            and self.cfg.sample_target_in_image
            and self._vision_camera is not None
        ):
            # Reset first uses conservative placeholders. Once both warmups
            # finish, _maybe_finalize_pending_target_resample() samples all
            # targets from the stable runtime camera pose in one operation.
            super()._resample_targets(
                env_ids_t, sample_position=False, sample_velocity=False
            )
            self._sample_placeholder_targets(env_ids_t)
            self._target_pos_w[env_ids_t] = self._multi_target_pos_w[env_ids_t, 0]
            self._target_vel_w[env_ids_t] = 0.0
            self._target_acc_w[env_ids_t] = 0.0
            self._multi_target_vel_w[env_ids_t] = 0.0
            self._pending_target_resample[env_ids_t] = True
            return

        super()._resample_targets(env_ids_t, sample_position, sample_velocity)
        if env_ids_t.numel() == 0:
            return
        self._multi_target_pos_w[env_ids_t, 0] = self._target_pos_w[env_ids_t]
        self._multi_target_vel_w[env_ids_t] = 0.0
        self._multi_target_vel_w[env_ids_t, 0] = self._target_vel_w[env_ids_t]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        previous_positions = self._multi_target_pos_w.clone()
        super()._pre_physics_step(actions)
        self._multi_target_pos_w[:, 0] = self._target_pos_w
        self._multi_target_vel_w[:, 0] = self._target_vel_w

        if self.cfg.enable_dynamic_target:
            self._multi_target_pos_w[:, 1:] += (
                self._multi_target_vel_w[:, 1:] * float(self.step_dt)
            )
            # Ground clearance is a physical constraint, independent of the
            # optional camera-frustum retention policy.
            ground_clearance_height = (
                self.scene.env_origins[:, 2].unsqueeze(1)
                + float(self.cfg.target_sphere_radius)
                + 0.02
            )
            invalid = self._multi_target_pos_w[..., 2] < ground_clearance_height
            if self.cfg.constrain_target_height:
                minimum_height = self.scene.env_origins[:, 2].unsqueeze(1) + max(
                    float(self.cfg.target_height_range[0]),
                    float(self.cfg.target_sphere_radius) + 0.02,
                )
                maximum_height = self.scene.env_origins[:, 2].unsqueeze(1) + float(
                    self.cfg.target_height_range[1]
                )
                invalid |= (
                    (self._multi_target_pos_w[..., 2] < minimum_height)
                    | (self._multi_target_pos_w[..., 2] > maximum_height)
                )
            if self.cfg.keep_targets_in_view and self._vision_camera is not None:
                projection = self._project_multi_targets()
                invalid |= ~projection["fully_visible"]
                distance = projection["distance_m"]
                invalid |= (
                    (distance < float(self.cfg.target_radius_range[0]))
                    | (distance > float(self.cfg.target_radius_range[1]))
                )
            if torch.any(invalid):
                self._multi_target_pos_w[invalid] = previous_positions[invalid]
                self._multi_target_vel_w[invalid] *= -1.0

        self._target_pos_w.copy_(self._multi_target_pos_w[:, 0])
        self._target_vel_w.copy_(self._multi_target_vel_w[:, 0])
        self._update_rendered_target_spheres()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Collection sequences may intentionally contain partial target exits.
        terminated = self._get_invalid_state_mask()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def resample_collection_targets(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Start a new target layout without resetting the robot or camera."""
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if env_ids_t.numel() == 0:
            return
        self._sample_targets_in_image(env_ids_t)
        self._sample_multi_target_velocities(env_ids_t)
        self._pending_target_resample[env_ids_t] = False
        self._vision_history_count[env_ids_t] = 0
        self._vision_frame_valid_history[env_ids_t] = False
        self._vision_valid_mask[env_ids_t] = False
        self._vision_collect_warmup_left[env_ids_t] = (
            self._vision_collect_warmup_steps_after_reset
        )
        self._update_rendered_target_spheres(env_ids_t)

    def advance_text_collection_state(self) -> None:
        """Advance collection warmup without reading or copying an RGB frame."""
        self._vision_collect_warmup_left = torch.clamp(
            self._vision_collect_warmup_left - 1, min=0
        )

    def _multi_target_raycast_measurements(
        self, target_directions_cam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Associate one shared ray scan with every projected target direction."""
        target_count = int(self.cfg.num_visual_targets)
        ranges = torch.full(
            (self.num_envs, target_count),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        valid = torch.zeros(
            (self.num_envs, target_count), dtype=torch.bool, device=self.device
        )
        if (
            not self.cfg.enable_dynamic_target_raycast_debug
            or self._dynamic_target_raycast_sensor is None
        ):
            return ranges, valid
        if target_directions_cam.shape != (self.num_envs, target_count, 2):
            raise ValueError(
                "target_directions_cam must have shape "
                f"({self.num_envs}, {target_count}, 2), got "
                f"{tuple(target_directions_cam.shape)}"
            )

        had_external_angles = hasattr(self, "_external_raycast_yaw") and hasattr(
            self, "_external_raycast_pitch"
        )
        previous_yaw = (
            self._external_raycast_yaw.clone() if had_external_angles else None
        )
        previous_pitch = (
            self._external_raycast_pitch.clone() if had_external_angles else None
        )
        had_external_mesh_id = hasattr(self, "_external_raycast_target_mesh_id")
        previous_mesh_id = (
            self._external_raycast_target_mesh_id
            if had_external_mesh_id
            else None
        )
        try:
            mesh_slots = getattr(
                self, "_dynamic_target_raycast_mesh_slots", tuple(range(target_count))
            )
            if len(mesh_slots) != target_count:
                raise RuntimeError(
                    "Raycast target mesh-slot mapping does not match target count: "
                    f"slots={mesh_slots}, target_count={target_count}"
                )
            for target_id in range(target_count):
                self.set_dynamic_target_raycast_angles(
                    target_directions_cam[:, target_id, 0],
                    target_directions_cam[:, target_id, 1],
                    target_mesh_id=int(mesh_slots[target_id]),
                )
                self._debug_raycast_dynamic_target(update_sensor=target_id == 0)
                ranges[:, target_id] = self._raycast_hit_distance
                valid[:, target_id] = self._raycast_hit_valid
        finally:
            if had_external_angles:
                self._external_raycast_yaw = previous_yaw
                self._external_raycast_pitch = previous_pitch
            else:
                self.__dict__.pop("_external_raycast_yaw", None)
                self.__dict__.pop("_external_raycast_pitch", None)
            if had_external_mesh_id:
                self._external_raycast_target_mesh_id = previous_mesh_id
            else:
                self.__dict__.pop("_external_raycast_target_mesh_id", None)
        return ranges, valid

    def get_multi_target_raycast_measurements(
        self, target_directions_cam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raycast ranges using externally supplied camera-frame directions.

        The caller is expected to obtain ``target_directions_cam`` from a
        perception module.  This public wrapper keeps the collection script
        from reaching into the environment's private raycast implementation.
        """

        return self._multi_target_raycast_measurements(target_directions_cam)

    def get_multi_target_collection_batch(
        self,
        *,
        include_rgb: bool = True,
        include_raycast: bool = False,
        raycast_directions_cam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return labels, camera data, and optionally RGB/raycast measurements.

        ``include_raycast=False`` is used by the online collector for its
        first pass: RGB is read, LK/OSTrack estimates the directions, and a
        second explicit call then obtains the RayCaster ranges.  When
        ``include_raycast=True``, external directions are mandatory so this
        collection API cannot silently fall back to projected simulation GT.
        """
        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized")
        rgb = None
        if include_rgb:
            rgb = self._vision_camera.data.output["rgb"]
            if rgb is None:
                raise RuntimeError("Vision camera returned no RGB output")
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]

        camera_pos_w, camera_quat_w = self._get_runtime_camera_pose_w()
        camera_linear_velocity_w, camera_angular_velocity_w = self._camera_motion_world(
            camera_pos_w
        )
        projection = self._project_multi_targets()
        if include_raycast:
            if raycast_directions_cam is None:
                raise ValueError(
                    "raycast_directions_cam is required when include_raycast=True"
                )
            lidar_range_m, lidar_valid = self._multi_target_raycast_measurements(
                raycast_directions_cam
            )
        else:
            lidar_range_m = torch.full(
                (self.num_envs, int(self.cfg.num_visual_targets)),
                float("inf"),
                dtype=torch.float32,
                device=self.device,
            )
            lidar_valid = torch.zeros_like(lidar_range_m, dtype=torch.bool)
        camera_linear_velocity_cam = self._rotate_multi_world_to_camera(
            camera_quat_w,
            camera_linear_velocity_w.unsqueeze(1).expand(
                -1, self._multi_target_pos_w.shape[1], -1
            ),
        )
        camera_angular_velocity_cam = self._rotate_multi_world_to_camera(
            camera_quat_w,
            camera_angular_velocity_w.unsqueeze(1).expand(
                -1, self._multi_target_pos_w.shape[1], -1
            ),
        )
        yaw_fields = self._multi_target_yaw_only_fields(
            camera_pos_w=camera_pos_w,
            camera_quat_w=camera_quat_w,
        )
        rgb_finite = (
            torch.isfinite(rgb.float()).all(dim=(1, 2, 3))
            if rgb is not None
            else torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        )
        camera_finite = (
            torch.isfinite(camera_pos_w).all(dim=-1)
            & torch.isfinite(camera_quat_w).all(dim=-1)
            & torch.isfinite(camera_linear_velocity_w).all(dim=-1)
            & torch.isfinite(camera_angular_velocity_w).all(dim=-1)
        )
        warmup_done = (
            (self._aim_warmup_left_steps <= 0)
            & (self._vision_collect_warmup_left <= 0)
        )
        separated = self._target_separation_mask(projection["center_uv"])
        # When the final warm-up count is consumed after rendering, the target
        # resample is intentionally deferred until the next pre-physics step.
        # Do not expose the temporary 8--10 m placeholder boxes as valid
        # initialization data during that one-step gap.
        collection_ready = (
            rgb_finite
            & camera_finite
            & warmup_done
            & ~self._pending_target_resample
        )
        initializable = (
            collection_ready
            & projection["fully_visible"].all(dim=1)
            & separated
        )
        batch = {
            "intrinsics": self._vision_camera.data.intrinsic_matrices,
            "bbox_xyxy": projection["bbox_xyxy"],
            "bbox_xyxy_raw": projection["bbox_xyxy_raw"],
            "center_uv": projection["center_uv"],
            "direction": projection["direction"],
            "distance_m": projection["distance_m"],
            "lidar_range_m": lidar_range_m,
            "lidar_valid": lidar_valid,
            "target_pos_w": self._multi_target_pos_w,
            "target_vel_w": self._multi_target_vel_w,
            "target_pos_cam": projection["target_pos_cam"],
            "target_vel_cam": self._multi_target_velocity_cam(
                camera_pos_w=camera_pos_w,
                camera_quat_w=camera_quat_w,
                camera_linear_velocity_w=camera_linear_velocity_w,
                camera_angular_velocity_w=camera_angular_velocity_w,
            ),
            **yaw_fields,
            "camera_linear_velocity_cam": camera_linear_velocity_cam,
            "camera_angular_velocity_cam": camera_angular_velocity_cam,
            "visible": projection["visible"],
            "fully_visible": projection["fully_visible"],
            "collection_ready": collection_ready,
            "separated": separated,
            "initializable": initializable,
        }
        if rgb is not None:
            batch["images"] = rgb
        return batch

    def _project_multi_targets(self) -> dict[str, torch.Tensor]:
        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized")
        camera_pos, camera_quat = self._get_runtime_camera_pose_w()
        offsets_world = self._multi_target_pos_w - camera_pos.unsqueeze(1)
        target_cam = self._rotate_multi_world_to_camera(camera_quat, offsets_world)
        x, y, z = target_cam.unbind(dim=-1)
        z_safe = z.clamp_min(1e-6)

        intrinsics = self._vision_camera.data.intrinsic_matrices
        fx = intrinsics[:, 0, 0].unsqueeze(1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)
        u = fx * x / z_safe + cx
        v = fy * y / z_safe + cy
        radius_x = fx * float(self.cfg.target_sphere_radius) / z_safe
        radius_y = fy * float(self.cfg.target_sphere_radius) / z_safe
        raw_bbox = torch.stack(
            [u - radius_x, v - radius_y, u + radius_x, v + radius_y], dim=-1
        )
        bbox = raw_bbox.clone()
        bbox[..., 0::2].clamp_(0.0, float(self.cfg.camera_width))
        bbox[..., 1::2].clamp_(0.0, float(self.cfg.camera_height))
        finite = (
            torch.isfinite(target_cam).all(dim=-1)
            & torch.isfinite(raw_bbox).all(dim=-1)
        )
        in_front = z > 1e-4
        intersects = (
            (raw_bbox[..., 2] > 0.0)
            & (raw_bbox[..., 3] > 0.0)
            & (raw_bbox[..., 0] < float(self.cfg.camera_width))
            & (raw_bbox[..., 1] < float(self.cfg.camera_height))
        )
        size_ok = (radius_x >= 1.0) & (radius_y >= 1.0)
        view_margin = max(0.0, float(self.cfg.target_spawn_border_px))
        fully_inside = (
            (raw_bbox[..., 0] >= view_margin)
            & (raw_bbox[..., 1] >= view_margin)
            & (raw_bbox[..., 2] < float(self.cfg.camera_width) - view_margin)
            & (raw_bbox[..., 3] < float(self.cfg.camera_height) - view_margin)
        )
        visible = finite & in_front & intersects & size_ok
        fully_visible = visible & fully_inside
        yaw = torch.atan2(x, z_safe)
        pitch = torch.atan2(-y, torch.sqrt(x * x + z * z).clamp_min(1e-6))
        return {
            "bbox_xyxy": bbox,
            "bbox_xyxy_raw": raw_bbox,
            "center_uv": torch.stack([u, v], dim=-1),
            "direction": torch.stack([yaw, pitch], dim=-1),
            "distance_m": torch.norm(target_cam, dim=-1),
            "target_pos_cam": target_cam,
            "visible": visible,
            "fully_visible": fully_visible,
        }

    def _multi_target_yaw_only_fields(
        self,
        *,
        camera_pos_w: torch.Tensor,
        camera_quat_w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return capture-time yaw-only geometry for every target."""

        target_count = int(self._multi_target_pos_w.shape[1])
        root_pos_w = self.turret.data.root_pos_w
        root_quat_w = self.turret.data.root_quat_w
        yaw_quat_w = self._get_yaw_only_quat(root_quat_w)
        target_rel_world = self._multi_target_pos_w - root_pos_w.unsqueeze(1)
        yaw_quat_targets = yaw_quat_w.unsqueeze(1).expand(
            -1, target_count, -1
        ).reshape(-1, 4)
        target_pos_yaw = self._rotate_world_to_local(
            yaw_quat_targets, target_rel_world.reshape(-1, 3)
        ).reshape(self.num_envs, target_count, 3)

        root_linear_world = quat_apply(root_quat_w, self.turret.data.root_lin_vel_b)
        root_angular_world = quat_apply(root_quat_w, self.turret.data.root_ang_vel_b)
        yaw_rate_world = self._get_yaw_rate_world(root_quat_w, root_angular_world)
        yaw_angular_world = torch.zeros_like(root_angular_world)
        yaw_angular_world[:, 2] = yaw_rate_world
        # This is d/dt of the target position expressed about the root origin
        # in the instantaneous yaw-only frame.
        base_point_velocity_world = root_linear_world.unsqueeze(1) + torch.cross(
            yaw_angular_world.unsqueeze(1).expand_as(target_rel_world),
            target_rel_world,
            dim=-1,
        )
        target_vel_yaw = self._rotate_world_to_local(
            yaw_quat_targets,
            (self._multi_target_vel_w - base_point_velocity_world).reshape(-1, 3),
        ).reshape(self.num_envs, target_count, 3)

        # Keep robot base motion for the GRU context and static-sequence filtering.
        base_linear_velocity_yaw = self._rotate_world_to_local(
            yaw_quat_w, root_linear_world
        )
        base_angular_velocity_yaw = self._rotate_world_to_local(
            yaw_quat_w, yaw_angular_world
        )

        camera_position_yaw_single = self._rotate_world_to_local(
            yaw_quat_w, camera_pos_w - root_pos_w
        )
        camera_position_yaw = camera_position_yaw_single.unsqueeze(1).expand(
            -1, target_count, -1
        )

        camera_basis = torch.eye(3, device=self.device, dtype=camera_pos_w.dtype)
        camera_basis = camera_basis.unsqueeze(0).expand(self.num_envs, -1, -1)
        basis_world = self._rotate_local_to_world(
            camera_quat_w.unsqueeze(1)
            .expand(-1, 3, -1)
            .reshape(-1, 4),
            camera_basis.reshape(-1, 3),
        ).reshape(self.num_envs, 3, 3)
        rotation_rows = self._rotate_world_to_local(
            yaw_quat_w.unsqueeze(1)
            .expand(-1, 3, -1)
            .reshape(-1, 4),
            basis_world.reshape(-1, 3),
        ).reshape(self.num_envs, 3, 3)
        rotation_yaw_from_camera_single = rotation_rows.transpose(1, 2)
        rotation_yaw_from_camera = rotation_yaw_from_camera_single.unsqueeze(1).expand(
            -1, target_count, -1, -1
        )
        return {
            "target_pos_yaw": target_pos_yaw,
            "target_vel_yaw": target_vel_yaw,
            "camera_position_yaw": camera_position_yaw,
            "rotation_yaw_from_camera": rotation_yaw_from_camera,
            "base_linear_velocity_yaw": base_linear_velocity_yaw,
            "base_angular_velocity_yaw": base_angular_velocity_yaw,
        }

    def _camera_motion_world(
        self, camera_pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return camera linear/angular velocity in the world tracking frame."""
        root_quat = self.turret.data.root_quat_w
        root_linear_world = quat_apply(root_quat, self.turret.data.root_lin_vel_b)
        root_angular_world = quat_apply(root_quat, self.turret.data.root_ang_vel_b)
        body_id = getattr(self, "_torso_body_id", None)
        body_linear = getattr(self.turret.data, "body_lin_vel_w", None)
        body_angular = getattr(self.turret.data, "body_ang_vel_w", None)
        body_position = getattr(self.turret.data, "body_pos_w", None)
        if (
            body_id is not None
            and body_linear is not None
            and body_angular is not None
            and body_position is not None
        ):
            linear_world = body_linear[:, body_id]
            angular_world = body_angular[:, body_id]
            origin_w = body_position[:, body_id]
        else:
            linear_world = root_linear_world
            angular_world = root_angular_world
            origin_w = self.turret.data.root_pos_w
        camera_offset_w = camera_pos_w - origin_w
        camera_linear_world = linear_world + torch.cross(
            angular_world, camera_offset_w, dim=-1
        )
        return camera_linear_world, angular_world

    def _multi_target_velocity_cam(
        self,
        *,
        camera_pos_w: torch.Tensor | None = None,
        camera_quat_w: torch.Tensor | None = None,
        camera_linear_velocity_w: torch.Tensor | None = None,
        camera_angular_velocity_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized")
        if camera_pos_w is None or camera_quat_w is None:
            camera_pos_w, camera_quat_w = self._get_runtime_camera_pose_w()
        if camera_linear_velocity_w is None or camera_angular_velocity_w is None:
            camera_linear_velocity_w, camera_angular_velocity_w = self._camera_motion_world(
                camera_pos_w
            )
        relative_world = self._multi_target_vel_w - camera_linear_velocity_w.unsqueeze(1)
        relative_camera = self._rotate_multi_world_to_camera(camera_quat_w, relative_world)
        target_camera = self._rotate_multi_world_to_camera(
            camera_quat_w, self._multi_target_pos_w - camera_pos_w.unsqueeze(1)
        )
        angular_camera = self._rotate_multi_world_to_camera(
            camera_quat_w,
            camera_angular_velocity_w.unsqueeze(1).expand_as(self._multi_target_pos_w),
        )
        return relative_camera - torch.cross(angular_camera, target_camera, dim=-1)

    def _sample_multi_target_velocities(self, env_ids_t: torch.Tensor) -> None:
        if env_ids_t.numel() == 0:
            return
        if not self.cfg.enable_dynamic_target:
            self._multi_target_vel_w[env_ids_t] = 0.0
            self._target_vel_w[env_ids_t] = 0.0
            return
        env_count = int(env_ids_t.numel())
        target_count = int(self.cfg.num_visual_targets)
        speed = self._uniform(
            float(self.cfg.target_speed_range[0]),
            float(self.cfg.target_speed_range[1]),
            (env_count, target_count),
        )
        if str(self.cfg.target_motion_mode) == "camera_lateral":
            # Controlled tracking-speed experiments need the requested speed
            # to be the actual apparent-motion speed, rather than a magnitude
            # that is subsequently reduced by a random heading/depth split.
            direction = torch.where(
                torch.rand((env_count, target_count), device=self.device) < 0.5,
                -torch.ones_like(speed),
                torch.ones_like(speed),
            )
            velocity_cam = torch.stack(
                [direction * speed, torch.zeros_like(speed), torch.zeros_like(speed)],
                dim=-1,
            )
        else:
            heading = self._uniform(-math.pi, math.pi, (env_count, target_count))
            vertical = self._uniform(
                float(self.cfg.target_vertical_speed_range[0]),
                float(self.cfg.target_vertical_speed_range[1]),
                (env_count, target_count),
            )
            # ROS optical frame: +x right, +y down, +z forward. Depth motion is
            # intentionally configurable because it changes apparent target size.
            depth_speed = (
                speed * torch.sin(heading) * float(self.cfg.target_depth_speed_scale)
            )
            velocity_cam = torch.stack(
                [speed * torch.cos(heading), vertical, depth_speed], dim=-1
            )
        _, runtime_camera_quat = self._get_runtime_camera_pose_w()
        camera_quat = runtime_camera_quat[env_ids_t]
        expanded_quat = camera_quat.unsqueeze(1).expand(-1, target_count, -1)
        velocity_world = quat_apply(
            expanded_quat.reshape(-1, 4), velocity_cam.reshape(-1, 3)
        ).reshape(env_count, target_count, 3)
        self._multi_target_vel_w[env_ids_t] = velocity_world
        self._target_vel_w[env_ids_t] = velocity_world[:, 0]

    def _update_rendered_target_spheres(
        self, env_ids_t: torch.Tensor | None = None
    ) -> None:
        super()._update_rendered_target_spheres(env_ids_t)
        if not hasattr(self, "_multi_target_pos_w"):
            return
        if env_ids_t is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Gf, Usd, UsdGeom

        stage = get_current_stage()
        positions = self._multi_target_pos_w[env_ids_t].detach().cpu()
        origins = self.scene.env_origins[env_ids_t].detach().cpu()
        env_ids_list = [int(value) for value in env_ids_t.detach().cpu().tolist()]
        for local_index, env_id in enumerate(env_ids_list):
            for target_id in range(1, int(self.cfg.num_visual_targets)):
                prim = stage.GetPrimAtPath(
                    self._extra_target_prim_paths[env_id][target_id - 1]
                )
                if not prim.IsValid():
                    continue
                xform = UsdGeom.Xformable(prim)
                translate_op = next(
                    (
                        op
                        for op in xform.GetOrderedXformOps()
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                    ),
                    None,
                )
                if translate_op is None:
                    translate_op = xform.AddTranslateOp(
                        UsdGeom.XformOp.PrecisionDouble, "", False
                    )
                local_position = positions[local_index, target_id] - origins[local_index]
                translate_op.Set(
                    Gf.Vec3d(*(float(value) for value in local_position)),
                    Usd.TimeCode.Default(),
                )

    def _target_separation_mask(self, centers: torch.Tensor) -> torch.Tensor:
        target_count = int(centers.shape[1])
        if target_count < 2:
            return torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        distance = torch.cdist(centers, centers)
        identity = torch.eye(target_count, dtype=torch.bool, device=self.device).unsqueeze(0)
        distance = torch.where(identity, torch.full_like(distance, float("inf")), distance)
        return distance.amin(dim=(1, 2)) >= float(self.cfg.min_target_separation_px)

    def _rotate_multi_world_to_camera(
        self, camera_quat: torch.Tensor, vectors_world: torch.Tensor
    ) -> torch.Tensor:
        target_count = int(vectors_world.shape[1])
        expanded_quat = camera_quat.unsqueeze(1).expand(-1, target_count, -1)
        conjugate = expanded_quat.reshape(-1, 4).clone()
        conjugate[:, 1:] *= -1.0
        rotated = quat_apply(conjugate, vectors_world.reshape(-1, 3))
        return rotated.reshape(self.num_envs, target_count, 3)

    def _uniform(
        self, low: float, high: float, shape: tuple[int, ...]
    ) -> torch.Tensor:
        if high < low:
            raise ValueError(f"Invalid uniform range: ({low}, {high})")
        return low + torch.rand(shape, device=self.device) * (high - low)


__all__ = ["MultiTargetCollectEnv", "MultiTargetCollectEnvCfg"]

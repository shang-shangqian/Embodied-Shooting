"""Online vision -> velocity bridge for PPO training in ``test7_2_weight``.

The bridge deliberately stops before the decision policy.  RL-Games remains
the only owner of the Actor/critic forward pass and recurrent state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import types
from typing import Any, Iterable

import numpy as np
import torch

from .model_loaders import load_velocity_estimator
from .shooting_chain import tracker_quality_features


@dataclass(frozen=True)
class IsaacTrainingBridgeConfig:
    velocity_checkpoint: str | Path
    velocity_backend: str = "legacy_gru"
    device: str = "cuda:0"
    camera_fx: float | None = None
    camera_fy: float | None = None
    camera_cx: float | None = None
    camera_cy: float | None = None
    target_radius_m: float = 0.35
    # A 0.70 m diameter target projects to roughly 8--16 px over the current
    # 80--150 m training shell. Keep the farthest target box close to its true
    # footprint while still enforcing a usable minimum crop.
    min_bootstrap_box_px: float = 8.0
    min_velocity_observations: int = 3
    min_velocity_history_s: float = 0.10
    history_window_s: float = 0.48
    max_history_steps: int = 32
    enable_fine_tracking: bool = True
    ostrack_repo: str | Path | None = None
    ostrack_checkpoint: str | Path | None = None
    ostrack_config: str = "vitb_256_mae_ce_32x4_ep300"
    fine_tracker_policy: str = "adaptive"
    fine_tracker_interval_frames: int = 3
    debug: bool = False

    def __post_init__(self) -> None:
        if self.velocity_backend not in {"legacy_gru", "taq_gru"}:
            raise ValueError("velocity_backend must be legacy_gru or taq_gru")
        for name, value in (("camera_fx", self.camera_fx), ("camera_fy", self.camera_fy)):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when overridden")
        if self.target_radius_m <= 0.0 or self.min_bootstrap_box_px <= 0.0:
            raise ValueError("target radius and bootstrap box size must be positive")
        if self.min_velocity_observations < 2 or self.min_velocity_history_s <= 0.0:
            raise ValueError("velocity warmup requires at least two observations and positive history")
        if self.enable_fine_tracking and (
            self.ostrack_repo is None or self.ostrack_checkpoint is None
        ):
            raise ValueError(
                "OSTrack repo/checkpoint are required when fine tracking is enabled"
            )


def _images_to_rgb_uint8(images_nchw: torch.Tensor) -> np.ndarray:
    if images_nchw.ndim != 4 or images_nchw.shape[1] != 3:
        raise RuntimeError(f"Expected RGB batch [N,3,H,W], got {tuple(images_nchw.shape)}")
    images = images_nchw.detach().to(device="cpu", dtype=torch.float32)
    if images.numel() and float(images.max().item()) <= 1.5:
        images = images * 255.0
    return np.ascontiguousarray(
        images.clamp(0.0, 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    )


def _bootstrap_box(
    yaw_rad: float,
    pitch_rad: float,
    distance_m: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    target_radius_m: float,
    min_box_px: float,
) -> tuple[float, float, float, float] | None:
    """Simulation detector proxy used only for the once-per-episode bootstrap."""
    if not np.isfinite((yaw_rad, pitch_rad, distance_m)).all() or distance_m <= 1.0e-5:
        return None
    x_norm = math.tan(float(yaw_rad))
    y_norm = -math.tan(float(pitch_rad)) * math.sqrt(1.0 + x_norm * x_norm)
    center_x = float(cx) + float(fx) * x_norm
    center_y = float(cy) + float(fy) * y_norm
    radius_px = max(0.5 * float(min_box_px), float(fx) * float(target_radius_m) / distance_m)
    x1, y1 = center_x - radius_px, center_y - radius_px
    x2, y2 = center_x + radius_px, center_y + radius_px
    if x2 <= 1.0 or y2 <= 1.0 or x1 >= width - 1.0 or y1 >= height - 1.0:
        return None
    return (
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    )


def _exact_yaw_rate(
    base_env: Any,
    root_quat_w: torch.Tensor,
    angular_velocity_world: torch.Tensor,
) -> torch.Tensor:
    basis_x = torch.zeros_like(angular_velocity_world)
    basis_x[:, 0] = 1.0
    forward_world = base_env._rotate_local_to_world(root_quat_w, basis_x)
    forward_dot_world = torch.cross(angular_velocity_world, forward_world, dim=-1)
    horizontal_squared = (
        forward_world[:, 0].square() + forward_world[:, 1].square()
    ).clamp_min(1.0e-8)
    return (
        forward_world[:, 0] * forward_dot_world[:, 1]
        - forward_world[:, 1] * forward_dot_world[:, 0]
    ) / horizontal_squared


class IsaacFullChainPerceptionBridge:
    """Maintain independent LK/OSTrack and velocity state for every Isaac environment."""

    def __init__(self, base_env: Any, config: IsaacTrainingBridgeConfig) -> None:
        self.base_env = base_env
        self.config = config
        self.device = torch.device(config.device)
        self.num_envs = int(base_env.num_envs)
        self.velocity_estimator = load_velocity_estimator(
            config.velocity_checkpoint,
            backend=config.velocity_backend,
            device=self.device,
            history_window_s=config.history_window_s,
            max_history_steps=config.max_history_steps,
        )
        self._trackers: list[Any | None] = [None] * self.num_envs
        # GT is permitted only for the one-time episode bootstrap. Once that
        # bootstrap has been consumed, tracker loss cannot silently reacquire
        # the target from privileged state; recovery requires a new episode or
        # a future real detector/re-identification module.
        self._bootstrap_consumed = [False] * self.num_envs
        self._valid_observations = [0] * self.num_envs
        self._first_valid_timestamp: list[float | None] = [None] * self.num_envs
        self._last_timestamp: list[float | None] = [None] * self.num_envs
        self._step_count = 0
        self._last_stats: dict[str, float] = {}

    def _make_tracker(
        self,
        width: int,
        height: int,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> Any:
        from test_lk_ostrack.camera_geometry import CameraIntrinsics
        from test_lk_ostrack.multi_target_tracker import (
            MultiTargetConfig,
            MultiTargetVisionTracker,
        )

        intrinsics = CameraIntrinsics(
            fx=float(fx),
            fy=float(fy),
            cx=cx,
            cy=cy,
            width=int(width),
            height=int(height),
        )
        fine_tracker = None
        if self.config.enable_fine_tracking:
            from test_lk_ostrack.ostrack_tracker import OSTrackFrameTracker

            fine_tracker = OSTrackFrameTracker(
                repo_path=Path(self.config.ostrack_repo).expanduser().resolve(),
                checkpoint=Path(self.config.ostrack_checkpoint).expanduser().resolve(),
                config=self.config.ostrack_config,
                device=str(self.config.device),
            )
            fine_tracker.prepare()
        tracker_config = MultiTargetConfig(
            target_width_m=2.0 * float(self.config.target_radius_m),
            target_height_m=2.0 * float(self.config.target_radius_m),
            fine_tracker_policy=str(self.config.fine_tracker_policy),
            fine_tracker_interval_frames=int(self.config.fine_tracker_interval_frames),
        )
        return MultiTargetVisionTracker(
            intrinsics, config=tracker_config, fine_tracker=fine_tracker
        )

    def reset_envs(self, env_ids: torch.Tensor | Iterable[int] | None = None) -> None:
        if env_ids is None:
            ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            ids = [int(value) for value in env_ids.detach().cpu().flatten().tolist()]
        else:
            ids = [int(value) for value in env_ids]
        for env_id in ids:
            if not 0 <= env_id < self.num_envs:
                continue
            tracker = self._trackers[env_id]
            if tracker is not None:
                tracker.reset()
            self.velocity_estimator.reset(env_id)
            self._bootstrap_consumed[env_id] = False
            self._valid_observations[env_id] = 0
            self._first_valid_timestamp[env_id] = None
            self._last_timestamp[env_id] = None
        if ids:
            ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.base_env.device)
            self.base_env._rl_pred_ready[ids_t] = False
            self.base_env._rl_bridge_training_ready[ids_t] = False
            self.base_env._rl_bridge_velocity_ready[ids_t] = False

    def _invalidate_env(self, env_id: int, *, reset_tracker: bool = False,
                        reset_velocity: bool = False) -> None:
        if reset_tracker and self._trackers[env_id] is not None:
            self._trackers[env_id].reset()
        if reset_velocity or (reset_tracker and self.config.velocity_backend != "legacy_gru"):
            self.velocity_estimator.reset(env_id)
            self._valid_observations[env_id] = 0
            self._first_valid_timestamp[env_id] = None
            self._last_timestamp[env_id] = None

    def _track_frame(self, env_id, image, timestamp, camera, bootstrap):
        """Track one RGB frame; specialized evaluation bridges may seed many IDs."""
        height, width = image.shape[:2]
        tracker = self._trackers[env_id]
        if tracker is None:
            tracker = self._make_tracker(width, height, **camera)
            self._trackers[env_id] = tracker
        if not tracker.initialized:
            if self._bootstrap_consumed[env_id] or not bool(bootstrap["valid_mask"][env_id].item()):
                return None
            box = _bootstrap_box(
                float(bootstrap["direction"][env_id, 0].item()),
                float(bootstrap["direction"][env_id, 1].item()),
                float(bootstrap["distance"][env_id].item()),
                **camera, width=width, height=height,
                target_radius_m=self.config.target_radius_m,
                min_box_px=self.config.min_bootstrap_box_px,
            )
            if box is None:
                return None
            self._bootstrap_consumed[env_id] = True
            result = tracker.initialize(image, {0: box}, timestamp_s=timestamp)
        else:
            previous = self._last_timestamp[env_id]
            dt_s = float(self.base_env.step_dt) if previous is None else timestamp - previous
            if dt_s <= 0.0:
                self._invalidate_env(env_id, reset_tracker=True)
                return None
            result = tracker.update(image, dt_s=dt_s, timestamp_s=timestamp)
        return result.selected_target

    def _bootstrap_inputs(self, batch):
        return dict(
            valid_mask=batch["valid_mask"].to(device=self.base_env.device, dtype=torch.bool),
            direction=batch["direction"].to(device=self.base_env.device, dtype=torch.float32),
            distance=batch["distance"].to(device=self.base_env.device, dtype=torch.float32).view(-1),
        )

    @torch.no_grad()
    def step(self) -> dict[str, Any]:
        env = self.base_env
        env._ensure_vision_supervision_ready()
        batch = env.get_vision_supervision_batch()
        images_hist = batch["images"]
        current_images = images_hist[:, -1] if images_hist.ndim == 5 else images_hist
        rgb_images = _images_to_rgb_uint8(current_images)
        # valid_mask/direction/distance are privileged simulation values.  The
        # mask and labels are used only as a detector proxy while a tracker is
        # uninitialized; an active tracker is updated from RGB without a GT
        # visibility gate.
        visual_valid = batch["valid_mask"].to(device=env.device, dtype=torch.bool)
        bootstrap = self._bootstrap_inputs(batch)

        height, width = rgb_images.shape[1:3]
        intrinsic_matrices = env._vision_camera.data.intrinsic_matrices.detach().cpu().numpy()
        if intrinsic_matrices.shape[0] != self.num_envs:
            raise RuntimeError(
                "Camera intrinsic batch does not match number of environments: "
                f"{intrinsic_matrices.shape[0]} != {self.num_envs}"
            )
        timestamp = float(env.common_step_counter) * float(env.step_dt)
        directions = np.zeros((self.num_envs, 2), dtype=np.float32)
        estimates: list[Any | None] = [None] * self.num_envs
        policy_warmup_left = getattr(env, "_aim_warmup_left_steps", None)

        for env_id in range(self.num_envs):
            if (self.config.velocity_backend == "legacy_gru" and policy_warmup_left is not None
                    and int(policy_warmup_left[env_id].item()) > 0):
                continue
            try:
                camera_matrix = intrinsic_matrices[env_id]
                fx = (
                    float(camera_matrix[0, 0])
                    if self.config.camera_fx is None
                    else float(self.config.camera_fx)
                )
                fy = (
                    float(camera_matrix[1, 1])
                    if self.config.camera_fy is None
                    else float(self.config.camera_fy)
                )
                cx = (
                    float(camera_matrix[0, 2])
                    if self.config.camera_cx is None
                    else float(self.config.camera_cx)
                )
                cy = (
                    float(camera_matrix[1, 2])
                    if self.config.camera_cy is None
                    else float(self.config.camera_cy)
                )
                estimate = self._track_frame(
                    env_id, rgb_images[env_id], timestamp,
                    dict(fx=fx, fy=fy, cx=cx, cy=cy),
                    bootstrap,
                )
                if estimate is None or str(estimate.status) == "lost":
                    self._invalidate_env(env_id, reset_tracker=True)
                    continue
                estimates[env_id] = estimate
                directions[env_id, 0] = float(estimate.yaw_rad)
                directions[env_id, 1] = float(estimate.pitch_rad)
                self._last_timestamp[env_id] = timestamp
            except Exception as exc:
                self._invalidate_env(env_id, reset_tracker=True)
                if self.config.debug:
                    print(f"[full-chain][tracker] env={env_id} error={exc}")

        env.set_dynamic_target_raycast_angles(
            torch.as_tensor(directions[:, 0], device=env.device),
            torch.as_tensor(directions[:, 1], device=env.device),
        )
        env.run_dynamic_target_raycast_for_current_frame()
        measured = env.get_vision_supervision_batch()
        range_m = measured["lidar_distance"].to(device=env.device, dtype=torch.float32).view(-1)
        range_valid = measured["lidar_valid_mask"].to(device=env.device, dtype=torch.bool).view(-1)

        yaw_t = torch.as_tensor(directions[:, 0], device=env.device, dtype=torch.float32)
        pitch_t = torch.as_tensor(directions[:, 1], device=env.device, dtype=torch.float32)
        safe_range = torch.where(range_valid, range_m, torch.zeros_like(range_m))
        position_cam = torch.stack(
            (
                safe_range * torch.sin(yaw_t) * torch.cos(pitch_t),
                -safe_range * torch.sin(pitch_t),
                safe_range * torch.cos(yaw_t) * torch.cos(pitch_t),
            ),
            dim=-1,
        )
        cam_pos_w, cam_quat_w = env._get_runtime_camera_pose_w()
        root_pos_w = env.turret.data.root_pos_w
        root_quat_w = env.turret.data.root_quat_w
        target_pos_w = cam_pos_w + env._rotate_local_to_world(cam_quat_w, position_cam)
        position_base = env._rotate_world_to_yaw_only_local(
            root_quat_w, target_pos_w - root_pos_w
        )

        linear_world = env._rotate_local_to_world(
            root_quat_w, env.turret.data.root_lin_vel_b
        )
        angular_world = env._rotate_local_to_world(
            root_quat_w, env.turret.data.root_ang_vel_b
        )
        robot_linear = env._rotate_world_to_yaw_only_local(root_quat_w, linear_world)
        robot_angular = env._rotate_world_to_yaw_only_local(root_quat_w, angular_world).clone()
        yaw_rate = _exact_yaw_rate(env, root_quat_w, angular_world)
        robot_angular[:, 2] = yaw_rate

        pred_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=env.device)
        pred_vel = torch.zeros_like(pred_pos)
        ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=env.device)
        gates = torch.zeros((self.num_envs,), dtype=torch.float32, device=env.device)

        for env_id, estimate in enumerate(estimates):
            if estimate is None and self.config.velocity_backend != "legacy_gru":
                continue
            valid_range = estimate is not None and bool(range_valid[env_id].item()) and bool(
                torch.isfinite(range_m[env_id]).item()
            )
            position_np = position_base[env_id].detach().cpu().numpy().astype(np.float32)
            if not valid_range:
                position_np[:] = np.nan
            quality = (tracker_quality_features(estimate, self._trackers[env_id].config)
                       if estimate is not None else (0.0,) * 7)
            source = str(estimate.source) if estimate is not None else "lost"
            try:
                velocity_estimate = self.velocity_estimator.update(
                    env_id,
                    types.SimpleNamespace(
                        timestamp_s=timestamp,
                        position_base_m=position_np,
                        robot_linear_velocity_base_mps=robot_linear[env_id].detach().cpu().tolist(),
                        robot_angular_velocity_base_rps=robot_angular[env_id].detach().cpu().tolist(),
                        yaw_rate_rps=float(yaw_rate[env_id].item()),
                        source=source,
                        quality=quality,
                        range_valid=valid_range,
                    ),
                )
            except Exception as exc:
                self._invalidate_env(env_id, reset_tracker=True, reset_velocity=True)
                if self.config.debug:
                    print(f"[full-chain][velocity] env={env_id} error={exc}")
                continue

            direct = valid_range and source in {"initial", "flow", "ostrack"}
            if direct or (valid_range and self.config.velocity_backend != "legacy_gru"):
                self._valid_observations[env_id] += 1
                if self._first_valid_timestamp[env_id] is None:
                    self._first_valid_timestamp[env_id] = timestamp
            if velocity_estimate is None:
                continue
            gates[env_id] = float(velocity_estimate.reliability_gate)
            history_start = self._first_valid_timestamp[env_id]
            history_s = 0.0 if history_start is None else timestamp - history_start
            velocity_value = velocity_estimate.velocity_mean_mps.to(
                device=env.device, dtype=torch.float32
            )
            velocity_ready = (
                valid_range
                and self._valid_observations[env_id]
                >= int(self.config.min_velocity_observations)
                and history_s >= float(self.config.min_velocity_history_s)
                and bool(torch.isfinite(velocity_value).all().item())
            )
            if velocity_ready:
                pred_pos[env_id] = position_base[env_id]
                pred_vel[env_id] = velocity_value
                ready[env_id] = True

        env._rl_pred_ready[:] = False
        env._rl_bridge_training_ready[:] = False
        env._rl_bridge_velocity_ready[:] = False
        ready_ids = torch.nonzero(ready, as_tuple=False).squeeze(-1)
        if ready_ids.numel() > 0:
            distance = torch.linalg.vector_norm(pred_pos, dim=-1)
            yaw = torch.atan2(pred_pos[:, 1], pred_pos[:, 0])
            pitch = torch.atan2(
                -pred_pos[:, 2],
                torch.linalg.vector_norm(pred_pos[:, :2], dim=-1).clamp_min(1.0e-6),
            )
            env.set_rl_predicted_target_state(
                pred_pos,
                pred_vel,
                distance,
                yaw,
                pitch,
                env_ids=ready_ids,
            )
            env._rl_bridge_training_ready[ready_ids] = True
            env._rl_bridge_velocity_ready[ready_ids] = True

        self._step_count += 1
        self._last_stats = {
            "visual_valid": float(visual_valid.float().mean().item()),
            "tracker_valid": float(sum(item is not None for item in estimates) / max(self.num_envs, 1)),
            "range_valid": float(range_valid.float().mean().item()),
            "ready": float(ready.float().mean().item()),
            "mean_gate": float(gates.mean().item()),
        }
        if self.config.debug and self._step_count % 50 == 0:
            print(f"[full-chain] step={self._step_count} stats={self._last_stats}")
        return {"ready": ready, "position": pred_pos, "velocity": pred_vel, **self._last_stats}

    def get_stats(self) -> dict[str, float]:
        return dict(self._last_stats)


def install_training_perception_bridge(
    base_env: Any,
    bridge: IsaacFullChainPerceptionBridge,
) -> IsaacFullChainPerceptionBridge:
    """Patch reset/step so every returned rollout observation uses current perception."""
    if getattr(base_env, "_full_chain_bridge_installed", False):
        raise RuntimeError("A full-chain perception bridge is already installed")

    base_env._gt_only_mode = False
    base_env._use_gt_fallback_for_missing_rl_pred = False
    bridge.reset_envs()

    original_reset_idx = base_env._reset_idx

    def patched_reset_idx(self, env_ids):
        result = original_reset_idx(env_ids)
        bridge.reset_envs(env_ids)
        return result

    base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)
    original_step = base_env.step

    def patched_step(self, actions):
        output = original_step(actions)
        bridge.step()
        observations = self._get_observations()
        if not isinstance(output, tuple):
            raise RuntimeError("Unexpected Isaac environment step return type")
        if len(output) == 5:
            _, reward, terminated, truncated, extras = output
            return observations, reward, terminated, truncated, extras
        if len(output) == 4:
            _, reward, done, extras = output
            return observations, reward, done, extras
        raise RuntimeError(f"Unexpected Isaac environment step tuple length: {len(output)}")

    base_env.step = types.MethodType(patched_step, base_env)
    base_env._full_chain_perception_bridge = bridge
    base_env._full_chain_bridge_installed = True
    return bridge


__all__ = [
    "IsaacFullChainPerceptionBridge",
    "IsaacTrainingBridgeConfig",
    "install_training_perception_bridge",
]

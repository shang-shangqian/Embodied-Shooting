from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import math
import os

import numpy as np
import torch

from .camera_geometry import bbox_clip
from .batched_ostrack import SharedBatchedOSTrack
from .ostrack_tracker import OSTrackBoxTracker


def _xyxy_to_xywh(box_xyxy: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def _xywh_to_xyxy(box_xywh: Iterable[float]) -> list[float]:
    x, y, width, height = [float(value) for value in box_xywh]
    return [x, y, x + max(1.0, width), y + max(1.0, height)]


def _image_to_rgb_uint8(image_chw: torch.Tensor) -> np.ndarray:
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise RuntimeError(f"Expected image [3,H,W], got {tuple(image_chw.shape)}")

    image = image_chw.detach().to(device="cpu", dtype=torch.float32)
    if image.numel() == 0:
        raise RuntimeError("Received an empty RGB image")
    if float(image.max().item()) <= 1.5:
        image = image * 255.0
    image = image.clamp(0.0, 255.0).round().to(dtype=torch.uint8)
    return np.ascontiguousarray(image.permute(1, 2, 0).numpy())


def _images_to_rgb_uint8(images_nchw: torch.Tensor) -> np.ndarray:
    if images_nchw.ndim != 4 or images_nchw.shape[1] != 3:
        raise RuntimeError(f"Expected images [N,3,H,W], got {tuple(images_nchw.shape)}")

    images = images_nchw.detach().to(device="cpu", dtype=torch.float32)
    if images.numel() == 0:
        raise RuntimeError("Received an empty RGB batch")
    if float(images.max().item()) <= 1.5:
        images = images * 255.0
    images = images.clamp(0.0, 255.0).round().to(dtype=torch.uint8)
    return np.ascontiguousarray(images.permute(0, 2, 3, 1).numpy())


def _pixel_to_camera_angles(u: float, v: float, intrinsic: torch.Tensor) -> tuple[float, float]:
    fx = max(float(intrinsic[0, 0].item()), 1e-6)
    fy = max(float(intrinsic[1, 1].item()), 1e-6)
    cx = float(intrinsic[0, 2].item())
    cy = float(intrinsic[1, 2].item())
    x_norm = (float(u) - cx) / fx
    y_norm = (float(v) - cy) / fy
    yaw = math.atan2(x_norm, 1.0)
    pitch = math.atan2(-y_norm, math.sqrt(x_norm * x_norm + 1.0))
    return yaw, pitch


TRACKER_FAILURE_REASON_NAMES = {
    0: "valid",
    1: "not_candidate",
    2: "initialization_unavailable",
    3: "initialization_error",
    4: "missing_output",
    5: "score_below_threshold",
    6: "tracking_exception",
    7: "global_recovery_pending",
}


class OnlineOSTrackVision:
    """Run independent OSTrack state per environment with serial or batched inference.

    The tracker is initialized from a provided pixel box when one is supplied.
    Otherwise the first valid frame of every episode uses the simulator target
    projection only for initialization. Subsequent predictions use only RGB.
    """

    def __init__(
        self,
        *,
        base_env: Any,
        repo_path: str | Path,
        checkpoint: str | Path,
        config: str = "vitb_256_mae_ce_32x4_ep300",
        device: str = "cuda:0",
        debug: int = 0,
        max_envs: int = 0,
        initial_box_xyxy: Iterable[float] | None = None,
        init_box_scale: float = 1.6,
        min_box_size_px: float = 8.0,
        reset_after_lost: int = 3,
        min_score: float = 0.20,
        enable_recovery: bool = False,
        recovery_min_score: float = 0.35,
        recovery_interval_steps: int = 5,
        recovery_grid_rows: int = 3,
        recovery_grid_cols: int = 4,
        recovery_overlap: float = 0.25,
        execution: str = "serial",
        batch_size: int = 0,
    ) -> None:
        self.base_env = base_env
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.config = str(config)
        self.device = str(device)
        self.debug = int(debug)
        self.max_envs = int(max_envs)
        self.init_box_scale = float(init_box_scale)
        self.min_box_size_px = float(min_box_size_px)
        self.reset_after_lost = max(1, int(reset_after_lost))
        self.min_score = float(min_score)
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        self.enable_recovery = bool(enable_recovery)
        self.recovery_min_score = float(recovery_min_score)
        self.recovery_interval_steps = int(recovery_interval_steps)
        self.recovery_grid_rows = int(recovery_grid_rows)
        self.recovery_grid_cols = int(recovery_grid_cols)
        self.recovery_overlap = float(recovery_overlap)
        if not 0.0 <= self.recovery_min_score <= 1.0:
            raise ValueError("recovery_min_score must be in [0, 1]")
        if self.enable_recovery and self.recovery_min_score < self.min_score:
            raise ValueError("recovery_min_score must be at least min_score")
        if self.recovery_interval_steps <= 0:
            raise ValueError("recovery_interval_steps must be positive")
        if self.recovery_grid_rows <= 0 or self.recovery_grid_cols <= 0:
            raise ValueError("recovery grid dimensions must be positive")
        if not 0.0 <= self.recovery_overlap < 1.0:
            raise ValueError("recovery_overlap must be in [0, 1)")
        self.execution = str(execution).strip().lower()
        self.batch_size = int(batch_size)
        if self.execution not in {"serial", "batched"}:
            raise ValueError("execution must be one of: serial, batched")
        if self.batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if self.execution == "batched" and self.debug != 0:
            raise ValueError("Batched OSTrack requires debug=0; use serial mode for Visdom debug")
        if self.enable_recovery and self.execution != "batched":
            raise ValueError("OSTrack global recovery requires execution='batched'")
        self.initial_box_xyxy = None if initial_box_xyxy is None else tuple(float(v) for v in initial_box_xyxy)
        if self.initial_box_xyxy is not None and len(self.initial_box_xyxy) != 4:
            raise ValueError("initial_box_xyxy must contain exactly four values: x1 y1 x2 y2")

        if not self.repo_path.exists():
            raise FileNotFoundError(f"OSTrack repo not found: {self.repo_path}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"OSTrack checkpoint not found: {self.checkpoint}")

        num_envs = int(base_env.num_envs)
        self._trackers: list[Any | None] = [None for _ in range(num_envs)]
        self._lost_counts = [0 for _ in range(num_envs)]
        self._can_initialize = [True for _ in range(num_envs)]
        self._recovery_active = [False for _ in range(num_envs)]
        self._predict_step = 0
        self._last_recovery_attempt_step = [
            -self.recovery_interval_steps for _ in range(num_envs)
        ]
        self._warned_track_error = False
        self._warned_recovery_error = False
        self._adapter: OSTrackBoxTracker | None = None
        self._batched_backend: SharedBatchedOSTrack | None = None
        self._last_diagnostics: dict[str, torch.Tensor] | None = None

    def _get_adapter(self) -> OSTrackBoxTracker:
        if self._adapter is None:
            self._adapter = OSTrackBoxTracker(
                repo_path=self.repo_path,
                checkpoint=self.checkpoint,
                config=self.config,
                device=self.device,
                debug=self.debug,
            )
        return self._adapter

    def _build_tracker(self) -> Any:
        adapter = self._get_adapter()
        old_cwd = Path.cwd()
        os.chdir(str(adapter.repo_path))
        try:
            return adapter._build_tracker()
        finally:
            os.chdir(str(old_cwd))

    def _get_batched_backend(self) -> SharedBatchedOSTrack:
        if self._batched_backend is None:
            adapter = self._get_adapter()
            self._batched_backend = SharedBatchedOSTrack(
                adapter=adapter,
                num_envs=int(self.base_env.num_envs),
                max_batch_size=self.batch_size,
            )
            batch_label = "all" if self.batch_size <= 0 else str(self.batch_size)
            print(
                "[vision:ostrack] shared batched backend initialized; "
                f"model_copies=1 max_batch_size={batch_label}"
            )
        return self._batched_backend

    def _call_in_repo(self, fn):
        old_cwd = Path.cwd()
        os.chdir(str(self.repo_path))
        try:
            return fn()
        finally:
            os.chdir(str(old_cwd))

    def reset_envs(self, env_ids: torch.Tensor | Iterable[int] | None = None) -> None:
        if env_ids is None:
            ids = list(range(len(self._trackers)))
        elif isinstance(env_ids, torch.Tensor):
            ids = [int(value) for value in env_ids.detach().cpu().flatten().tolist()]
        else:
            ids = [int(value) for value in env_ids]

        if self._batched_backend is not None:
            self._batched_backend.reset_envs(ids)

        for env_id in ids:
            if 0 <= env_id < len(self._trackers):
                self._trackers[env_id] = None
                self._lost_counts[env_id] = 0
                self._can_initialize[env_id] = True
                self._recovery_active[env_id] = False
                self._last_recovery_attempt_step[env_id] = (
                    self._predict_step - self.recovery_interval_steps
                )

    def _empty_diagnostics(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        device_t = torch.device(device)
        num_envs = len(self._trackers)
        return {
            "tracker_score": torch.full(
                (num_envs,), float("nan"), dtype=torch.float32, device=device_t
            ),
            "tracker_failure_reason": torch.full(
                (num_envs,), 1, dtype=torch.long, device=device_t
            ),
            "tracker_initialized_this_frame": torch.zeros(
                (num_envs,), dtype=torch.bool, device=device_t
            ),
            "tracker_lost_count": torch.zeros(
                (num_envs,), dtype=torch.long, device=device_t
            ),
            "tracker_recovery_attempted": torch.zeros(
                (num_envs,), dtype=torch.bool, device=device_t
            ),
            "tracker_recovery_succeeded": torch.zeros(
                (num_envs,), dtype=torch.bool, device=device_t
            ),
            "tracker_recovery_score": torch.full(
                (num_envs,), float("nan"), dtype=torch.float32, device=device_t
            ),
            "tracker_recovery_pending": torch.zeros(
                (num_envs,), dtype=torch.bool, device=device_t
            ),
        }

    def _publish_diagnostics(self, diagnostics: dict[str, torch.Tensor]) -> None:
        device = diagnostics["tracker_score"].device
        diagnostics["tracker_lost_count"] = torch.as_tensor(
            self._lost_counts,
            dtype=torch.long,
            device=device,
        )
        diagnostics["tracker_recovery_pending"] = torch.as_tensor(
            self._recovery_active,
            dtype=torch.bool,
            device=device,
        )
        self._last_diagnostics = {
            name: value.detach().clone() for name, value in diagnostics.items()
        }

    def get_last_diagnostics(
        self, device: torch.device | str | None = None
    ) -> dict[str, torch.Tensor]:
        """Return tracker scores and failure causes from the latest predict call."""
        if self._last_diagnostics is None:
            return self._empty_diagnostics(device or "cpu")
        return {
            name: value.to(device=device) if device is not None else value.clone()
            for name, value in self._last_diagnostics.items()
        }

    @torch.no_grad()
    def global_search(
        self,
        images: torch.Tensor,
        search_mask: torch.Tensor,
        *,
        grid_rows: int = 3,
        grid_cols: int = 4,
        overlap: float = 0.25,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Use the preserved initial template to search overlapping full-image tiles."""
        if self.execution != "batched":
            raise RuntimeError("Template global search currently requires batched OSTrack")
        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(f"Expected images [N,3,H,W], got {tuple(images.shape)}")
        if int(grid_rows) <= 0 or int(grid_cols) <= 0:
            raise ValueError("global-search grid dimensions must be positive")
        if not 0.0 <= float(overlap) < 1.0:
            raise ValueError("global-search overlap must be in [0, 1)")

        num_envs = int(self.base_env.num_envs)
        device = images.device
        boxes = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        scores = torch.full((num_envs,), float("nan"), dtype=torch.float32, device=device)
        valid = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        yaw = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        pitch = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        active_ids = torch.nonzero(
            search_mask.to(device=device, dtype=torch.bool).view(-1), as_tuple=False
        ).squeeze(-1)
        if active_ids.numel() == 0:
            return yaw, pitch, valid, boxes, scores

        backend = self._get_batched_backend()
        max_envs = num_envs if self.max_envs <= 0 else min(num_envs, self.max_envs)
        images_rgb = _images_to_rgb_uint8(images[:max_envs])
        rows = int(grid_rows)
        cols = int(grid_cols)
        overlap_value = float(overlap)
        items: list[tuple[int, np.ndarray, list[list[float]]]] = []
        for env_id_tensor in active_ids:
            env_id = int(env_id_tensor.item())
            if env_id >= max_envs or not backend.is_initialized(env_id):
                continue
            image_rgb = images_rgb[env_id]
            height, width = image_rgb.shape[:2]
            tile_width = float(width) / (1.0 + max(cols - 1, 0) * (1.0 - overlap_value))
            tile_height = float(height) / (1.0 + max(rows - 1, 0) * (1.0 - overlap_value))
            side = max(tile_width, tile_height)
            anchor_side = max(2.0, side / float(backend.params.search_factor))
            centers_x = np.linspace(0.5 * tile_width, width - 0.5 * tile_width, cols)
            centers_y = np.linspace(0.5 * tile_height, height - 0.5 * tile_height, rows)
            anchors = [
                [
                    float(center_x - 0.5 * anchor_side),
                    float(center_y - 0.5 * anchor_side),
                    float(anchor_side),
                    float(anchor_side),
                ]
                for center_y in centers_y
                for center_x in centers_x
            ]
            items.append((env_id, image_rgb, anchors))

        candidates, _ = self._call_in_repo(lambda: backend.global_search_batch(items))
        intrinsic = self.base_env._vision_camera.data.intrinsic_matrices.to(
            device=device, dtype=torch.float32
        )
        for env_id, env_candidates in candidates.items():
            if not env_candidates:
                continue
            bbox_xywh, score = env_candidates[0]
            height, width = images_rgb[env_id].shape[:2]
            bbox = bbox_clip(_xywh_to_xyxy(bbox_xywh), width, height)
            boxes[env_id] = torch.tensor(bbox, dtype=torch.float32, device=device)
            scores[env_id] = float(score)
            center_u = 0.5 * float(bbox[0] + bbox[2])
            center_v = 0.5 * float(bbox[1] + bbox[3])
            yaw_value, pitch_value = _pixel_to_camera_angles(
                center_u, center_v, intrinsic[env_id]
            )
            yaw[env_id] = yaw_value
            pitch[env_id] = pitch_value
            valid[env_id] = True
        return yaw, pitch, valid, boxes, scores

    @torch.no_grad()
    def reinitialize_from_boxes(
        self,
        images: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Restart local OSTrack states from externally confirmed candidate boxes."""
        num_envs = int(self.base_env.num_envs)
        result = torch.zeros(num_envs, dtype=torch.bool, device=images.device)
        ids = torch.nonzero(
            valid_mask.to(device=images.device, dtype=torch.bool).view(-1), as_tuple=False
        ).squeeze(-1)
        if ids.numel() == 0:
            return result
        if self.execution != "batched":
            raise RuntimeError("External recovery reinitialization currently requires batched OSTrack")
        backend = self._get_batched_backend()
        images_rgb = _images_to_rgb_uint8(images)
        for env_id_tensor in ids:
            env_id = int(env_id_tensor.item())
            height, width = images_rgb[env_id].shape[:2]
            bbox = bbox_clip(
                boxes_xyxy[env_id].detach().cpu().tolist(), width, height
            )
            try:
                backend.initialize_env(
                    env_id, images_rgb[env_id], _xyxy_to_xywh(bbox)
                )
                self._lost_counts[env_id] = 0
                self._can_initialize[env_id] = False
                self._recovery_active[env_id] = False
                result[env_id] = True
            except Exception:
                result[env_id] = False
        return result

    @torch.no_grad()
    def recover_lost_tracks(
        self,
        images: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Recover persistently lost batched tracks without consulting GT boxes."""
        num_envs = int(self.base_env.num_envs)
        device = images.device
        result = {
            "yaw": torch.zeros((num_envs,), dtype=torch.float32, device=device),
            "pitch": torch.zeros((num_envs,), dtype=torch.float32, device=device),
            "boxes": torch.zeros((num_envs, 4), dtype=torch.float32, device=device),
            "scores": torch.full(
                (num_envs,), float("nan"), dtype=torch.float32, device=device
            ),
            "attempted_mask": torch.zeros(
                (num_envs,), dtype=torch.bool, device=device
            ),
            "recovered_mask": torch.zeros(
                (num_envs,), dtype=torch.bool, device=device
            ),
        }
        if not self.enable_recovery:
            return result
        if self.execution != "batched":
            raise RuntimeError("OSTrack global recovery requires batched execution")
        if images.ndim != 4 or images.shape[0] != num_envs:
            raise RuntimeError("recovery images must have shape [num_envs,3,H,W]")

        candidates = candidate_mask.to(device=device, dtype=torch.bool).view(-1)
        if candidates.numel() != num_envs:
            raise RuntimeError("recovery candidate_mask must contain one value per environment")

        backend = self._batched_backend
        if backend is None:
            return result

        due = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        for env_id in range(num_envs):
            if not bool(candidates[env_id].item()):
                continue
            if not self._recovery_active[env_id]:
                continue
            if (
                self._predict_step - self._last_recovery_attempt_step[env_id]
                < self.recovery_interval_steps
            ):
                continue
            if not backend.is_initialized(env_id):
                continue
            if not backend.has_reference_template(env_id):
                continue
            due[env_id] = True
            self._last_recovery_attempt_step[env_id] = self._predict_step

        if not torch.any(due):
            return result

        result["attempted_mask"].copy_(due)
        try:
            yaw, pitch, search_valid, boxes, scores = self.global_search(
                images,
                due,
                grid_rows=self.recovery_grid_rows,
                grid_cols=self.recovery_grid_cols,
                overlap=self.recovery_overlap,
            )
        except Exception as exc:
            diagnostics = self.get_last_diagnostics(device=device)
            diagnostics["tracker_recovery_attempted"].copy_(due)
            self._publish_diagnostics(diagnostics)
            if not self._warned_recovery_error:
                print(
                    "[vision:ostrack] global recovery failed; first error: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._warned_recovery_error = True
            return result
        accepted = (
            search_valid
            & torch.isfinite(scores)
            & (scores >= self.recovery_min_score)
        )
        reinitialized = self.reinitialize_from_boxes(images, boxes, accepted)
        recovered = accepted & reinitialized

        result["yaw"].copy_(yaw)
        result["pitch"].copy_(pitch)
        result["boxes"].copy_(boxes)
        result["scores"].copy_(scores)
        result["recovered_mask"].copy_(recovered)

        diagnostics = self.get_last_diagnostics(device=device)
        diagnostics["tracker_recovery_attempted"].copy_(due)
        diagnostics["tracker_recovery_succeeded"].copy_(recovered)
        diagnostics["tracker_recovery_score"].copy_(scores)
        diagnostics["tracker_score"][recovered] = scores[recovered]
        diagnostics["tracker_failure_reason"][recovered] = 0
        diagnostics["tracker_initialized_this_frame"][recovered] = True
        self._publish_diagnostics(diagnostics)
        return result

    def _project_initial_boxes(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = int(self.base_env.num_envs)
        boxes = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        valid = torch.zeros((num_envs,), dtype=torch.bool, device=device)

        camera = getattr(self.base_env, "_vision_camera", None)
        if camera is None:
            return boxes, valid

        if self.initial_box_xyxy is not None:
            box = torch.tensor(self.initial_box_xyxy, dtype=torch.float32, device=device)
            boxes[:] = box
            valid[:] = True
            return boxes, valid

        cam_pos_w, cam_quat_w = self.base_env._get_runtime_camera_pose_w()
        target_pos_w = self.base_env._target_pos_w
        cam_pos_w = cam_pos_w.to(device=device, dtype=torch.float32)
        cam_quat_w = cam_quat_w.to(device=device, dtype=torch.float32)
        target_pos_w = target_pos_w.to(device=device, dtype=torch.float32)
        target_cam = self.base_env._rotate_world_to_local(cam_quat_w, target_pos_w - cam_pos_w)
        target_cam = target_cam.to(device=device, dtype=torch.float32)

        intrinsic = camera.data.intrinsic_matrices.to(device=device, dtype=torch.float32)
        fx = intrinsic[:, 0, 0]
        fy = intrinsic[:, 1, 1]
        cx = intrinsic[:, 0, 2]
        cy = intrinsic[:, 1, 2]
        x, y, z = target_cam.unbind(dim=-1)
        z_safe = z.clamp_min(1e-6)
        u = fx * x / z_safe + cx
        v = fy * y / z_safe + cy

        target_radius = float(getattr(self.base_env.cfg, "target_sphere_radius", 0.35))
        radius_px = torch.minimum(fx * target_radius / z_safe, fy * target_radius / z_safe)
        half = torch.clamp(
            radius_px * self.init_box_scale,
            min=max(1.0, 0.5 * self.min_box_size_px),
        )

        image_height = int(getattr(self.base_env.cfg, "camera_height", 0))
        image_width = int(getattr(self.base_env.cfg, "camera_width", 0))
        if image_height <= 0 or image_width <= 0:
            image_height = int(camera.data.output["rgb"].shape[1])
            image_width = int(camera.data.output["rgb"].shape[2])

        boxes[:, 0] = torch.clamp(u - half, 0.0, float(image_width - 1))
        boxes[:, 1] = torch.clamp(v - half, 0.0, float(image_height - 1))
        boxes[:, 2] = torch.clamp(u + half, 1.0, float(image_width))
        boxes[:, 3] = torch.clamp(v + half, 1.0, float(image_height))
        valid = (
            torch.isfinite(boxes).all(dim=-1)
            & torch.isfinite(z)
            & torch.isfinite(radius_px)
            & (z > 1e-4)
            & (u >= 0.0)
            & (u < float(image_width))
            & (v >= 0.0)
            & (v < float(image_height))
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        return boxes, valid


    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        candidate_mask: torch.Tensor,
        search_factor_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._predict_step += 1
        if self.execution == "batched":
            return self._predict_batched(images, candidate_mask, search_factor_scale)
        return self._predict_serial(images, candidate_mask, search_factor_scale)

    @torch.no_grad()
    def _predict_batched(
        self,
        images: torch.Tensor,
        candidate_mask: torch.Tensor,
        search_factor_scale: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(f"Expected images [N,3,H,W], got {tuple(images.shape)}")

        num_envs = int(self.base_env.num_envs)
        if images.shape[0] != num_envs or candidate_mask.numel() != num_envs:
            raise RuntimeError("OSTrack batch size does not match the environment count")

        output_device = images.device
        yaw = torch.zeros((num_envs,), dtype=torch.float32, device=output_device)
        pitch = torch.zeros((num_envs,), dtype=torch.float32, device=output_device)
        valid = torch.zeros((num_envs,), dtype=torch.bool, device=output_device)
        boxes = torch.zeros((num_envs, 4), dtype=torch.float32, device=output_device)
        diagnostics = self._empty_diagnostics(output_device)

        max_envs = num_envs if self.max_envs <= 0 else min(num_envs, self.max_envs)
        candidate_cpu = (
            candidate_mask.detach().to(device="cpu", dtype=torch.bool).view(-1).numpy()
        )
        if isinstance(search_factor_scale, torch.Tensor):
            search_scale_cpu = search_factor_scale.detach().to(
                device="cpu", dtype=torch.float32
            ).view(-1)
            if search_scale_cpu.numel() != num_envs:
                raise ValueError("search_factor_scale tensor must contain one value per environment")
        else:
            search_scale_cpu = torch.full(
                (num_envs,), float(search_factor_scale), dtype=torch.float32
            )
        if not torch.isfinite(search_scale_cpu).all() or bool((search_scale_cpu <= 0.0).any()):
            raise ValueError("search_factor_scale values must be finite and positive")
        active_ids = [
            env_id for env_id in range(max_envs) if bool(candidate_cpu[env_id])
        ]
        if not active_ids:
            self._publish_diagnostics(diagnostics)
            return yaw, pitch, valid, boxes

        backend = self._get_batched_backend()
        images_rgb = _images_to_rgb_uint8(images[:max_envs])
        init_boxes, init_valid = self._project_initial_boxes(output_device)
        init_boxes_cpu = init_boxes.detach().cpu()
        init_valid_cpu = init_valid.detach().cpu()
        intrinsic_cpu = self.base_env._vision_camera.data.intrinsic_matrices.to(
            device="cpu", dtype=torch.float32
        )

        def commit_bbox(
            env_id: int,
            bbox: Iterable[float],
            *,
            score: float | None = None,
            initialized: bool = False,
        ) -> None:
            bbox_values = [float(value) for value in bbox]
            boxes[env_id] = torch.tensor(
                bbox_values,
                dtype=torch.float32,
                device=output_device,
            )
            center_u = 0.5 * (bbox_values[0] + bbox_values[2])
            center_v = 0.5 * (bbox_values[1] + bbox_values[3])
            yaw_value, pitch_value = _pixel_to_camera_angles(
                center_u,
                center_v,
                intrinsic_cpu[env_id],
            )
            yaw[env_id] = yaw_value
            pitch[env_id] = pitch_value
            valid[env_id] = True
            diagnostics["tracker_score"][env_id] = (
                float("nan") if score is None else float(score)
            )
            diagnostics["tracker_failure_reason"][env_id] = 0
            diagnostics["tracker_initialized_this_frame"][env_id] = initialized

        pending: list[tuple[int, np.ndarray, float]] = []
        for env_id in active_ids:
            image_rgb = images_rgb[env_id]
            height, width = image_rgb.shape[:2]
            if backend.is_initialized(env_id):
                pending.append((env_id, image_rgb, float(search_scale_cpu[env_id].item())))
                continue
            if not self._can_initialize[env_id] or not bool(init_valid_cpu[env_id].item()):
                diagnostics["tracker_failure_reason"][env_id] = 2
                continue

            bbox = bbox_clip(init_boxes_cpu[env_id].tolist(), width, height)
            try:
                backend.initialize_env(
                    env_id,
                    image_rgb,
                    _xyxy_to_xywh(bbox),
                )
                self._lost_counts[env_id] = 0
                self._can_initialize[env_id] = False
                commit_bbox(env_id, bbox, initialized=True)
            except Exception as exc:
                backend.reset_envs([env_id])
                self._lost_counts[env_id] = 0
                # No template was successfully created, so the episode's one
                # permitted GT initialization may be retried on the next frame.
                self._can_initialize[env_id] = True
                diagnostics["tracker_failure_reason"][env_id] = 3
                if not self._warned_track_error:
                    print(
                        "[vision:ostrack] batched initialization failed; first error: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._warned_track_error = True

        tracked_boxes, track_errors = self._call_in_repo(
            lambda: backend.track_batch(pending)
        )
        pending_ids = {env_id for env_id, _, _ in pending}

        for env_id, exc in track_errors.items():
            # Keep both local state and the immutable reference template. The
            # recovery path can still search globally after transient failures.
            self._lost_counts[env_id] += 1
            if self.enable_recovery and self._lost_counts[env_id] >= self.reset_after_lost:
                self._recovery_active[env_id] = True
            diagnostics["tracker_failure_reason"][env_id] = 6
            if not self._warned_track_error:
                print(
                    "[vision:ostrack] batched tracking failed; first error: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._warned_track_error = True

        for env_id, bbox_xywh in tracked_boxes.items():
            image_rgb = images_rgb[env_id]
            height, width = image_rgb.shape[:2]
            bbox = bbox_clip(_xywh_to_xyxy(bbox_xywh), width, height)
            score = backend.response_peak(env_id)
            diagnostics["tracker_score"][env_id] = (
                float("nan") if score is None else float(score)
            )
            score_rejected = self.min_score > 0.0 and (
                score is None
                or not np.isfinite(score)
                or float(score) < self.min_score
            )
            if score_rejected:
                diagnostics["tracker_failure_reason"][env_id] = 5
                self._lost_counts[env_id] += 1
                if self.enable_recovery and self._lost_counts[env_id] >= self.reset_after_lost:
                    self._recovery_active[env_id] = True
                continue
            if self._recovery_active[env_id]:
                diagnostics["tracker_failure_reason"][env_id] = 7
                continue
            self._lost_counts[env_id] = 0
            commit_bbox(env_id, bbox, score=score)

        missing_ids = pending_ids.difference(tracked_boxes).difference(track_errors)
        for env_id in missing_ids:
            diagnostics["tracker_failure_reason"][env_id] = 4
            self._lost_counts[env_id] += 1
            if self.enable_recovery and self._lost_counts[env_id] >= self.reset_after_lost:
                self._recovery_active[env_id] = True

        self._publish_diagnostics(diagnostics)
        return yaw, pitch, valid, boxes


    @torch.no_grad()
    def _predict_serial(
        self,
        images: torch.Tensor,
        candidate_mask: torch.Tensor,
        search_factor_scale: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(f"Expected images [N,3,H,W], got {tuple(images.shape)}")

        num_envs = int(self.base_env.num_envs)
        if images.shape[0] != num_envs or candidate_mask.numel() != num_envs:
            raise RuntimeError("OSTrack batch size does not match the environment count")

        output_device = images.device
        yaw = torch.zeros((num_envs,), dtype=torch.float32, device=output_device)
        pitch = torch.zeros((num_envs,), dtype=torch.float32, device=output_device)
        valid = torch.zeros((num_envs,), dtype=torch.bool, device=output_device)
        boxes = torch.zeros((num_envs, 4), dtype=torch.float32, device=output_device)
        diagnostics = self._empty_diagnostics(output_device)
        init_boxes, init_valid = self._project_initial_boxes(output_device)
        intrinsic = self.base_env._vision_camera.data.intrinsic_matrices.to(
            device=output_device, dtype=torch.float32
        )
        if isinstance(search_factor_scale, torch.Tensor):
            search_scales = search_factor_scale.to(
                device=output_device, dtype=torch.float32
            ).view(-1)
            if search_scales.numel() != num_envs:
                raise ValueError("search_factor_scale tensor must contain one value per environment")
        else:
            search_scales = torch.full(
                (num_envs,), float(search_factor_scale), dtype=torch.float32, device=output_device
            )

        max_envs = num_envs if self.max_envs <= 0 else min(num_envs, self.max_envs)
        for env_id in range(max_envs):
            if not bool(candidate_mask[env_id].item()):
                continue

            image_rgb = _image_to_rgb_uint8(images[env_id])
            height, width = image_rgb.shape[:2]
            try:
                if self._trackers[env_id] is None:
                    if not self._can_initialize[env_id] or not bool(init_valid[env_id].item()):
                        diagnostics["tracker_failure_reason"][env_id] = 2
                        continue
                    bbox = bbox_clip(init_boxes[env_id].detach().cpu().tolist(), width, height)
                    tracker = self._build_tracker()
                    self._call_in_repo(
                        lambda: tracker.initialize(image_rgb, {"init_bbox": _xyxy_to_xywh(bbox)})
                    )
                    self._trackers[env_id] = tracker
                    self._lost_counts[env_id] = 0
                    self._can_initialize[env_id] = False
                    diagnostics["tracker_failure_reason"][env_id] = 0
                    diagnostics["tracker_initialized_this_frame"][env_id] = True
                else:
                    tracker = self._trackers[env_id]
                    original_search_factor = float(tracker.params.search_factor)
                    tracker.params.search_factor = (
                        original_search_factor * float(search_scales[env_id].item())
                    )
                    try:
                        tracked = self._call_in_repo(lambda: tracker.track(image_rgb, {}))
                    finally:
                        tracker.params.search_factor = original_search_factor
                    bbox_xywh = tracked.get("target_bbox")
                    score = tracked.get("best_score", tracked.get("target_score"))
                    diagnostics["tracker_score"][env_id] = (
                        float("nan") if score is None else float(score)
                    )
                    # The unmodified official serial tracker does not return a
                    # score. Enforce the threshold whenever a score is present.
                    score_rejected = self.min_score > 0.0 and score is not None and (
                        not np.isfinite(score) or float(score) < self.min_score
                    )
                    if bbox_xywh is None or score_rejected:
                        diagnostics["tracker_failure_reason"][env_id] = (
                            4 if bbox_xywh is None else 5
                        )
                        self._lost_counts[env_id] += 1
                        if self._lost_counts[env_id] >= self.reset_after_lost:
                            self._trackers[env_id] = None
                        continue
                    bbox = bbox_clip(_xywh_to_xyxy(bbox_xywh), width, height)
                    self._lost_counts[env_id] = 0
            except Exception as exc:
                self._trackers[env_id] = None
                self._lost_counts[env_id] = 0
                self._can_initialize[env_id] = False
                diagnostics["tracker_failure_reason"][env_id] = 6
                if not self._warned_track_error:
                    print(f"[vision:ostrack] tracking failed; first error: {type(exc).__name__}: {exc}")
                    self._warned_track_error = True
                continue

            boxes[env_id] = torch.tensor(bbox, dtype=torch.float32, device=output_device)
            center_u = 0.5 * float(bbox[0] + bbox[2])
            center_v = 0.5 * float(bbox[1] + bbox[3])
            yaw_value, pitch_value = _pixel_to_camera_angles(center_u, center_v, intrinsic[env_id])
            yaw[env_id] = yaw_value
            pitch[env_id] = pitch_value
            valid[env_id] = True

        self._publish_diagnostics(diagnostics)
        return yaw, pitch, valid, boxes


__all__ = ["OnlineOSTrackVision"]

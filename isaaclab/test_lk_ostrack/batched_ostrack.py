from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import os

import numpy as np
import torch

from .ostrack_tracker import OSTrackBoxTracker


@dataclass
class _OSTrackEnvState:
    bbox_xywh: list[float] | None = None
    template_tensor: torch.Tensor | None = None
    ce_template_mask: torch.Tensor | None = None
    reference_template_tensor: torch.Tensor | None = None
    reference_ce_template_mask: torch.Tensor | None = None
    frame_id: int = 0
    response_peak: float | None = None

    @property
    def initialized(self) -> bool:
        return self.bbox_xywh is not None and self.template_tensor is not None


class SharedBatchedOSTrack:
    """Share one official OSTrack network across independent environment states."""

    def __init__(
        self,
        *,
        adapter: OSTrackBoxTracker,
        num_envs: int,
        max_batch_size: int = 0,
    ) -> None:
        if int(adapter.debug) != 0:
            raise ValueError(
                "Batched OSTrack does not support the official Visdom debug path; "
                "use --ostrack_debug 0 or --ostrack_execution serial."
            )

        self.repo_path = Path(adapter.repo_path)
        self.num_envs = int(num_envs)
        self.max_batch_size = int(max_batch_size)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.max_batch_size < 0:
            raise ValueError("max_batch_size must be non-negative")

        old_cwd = Path.cwd()
        os.chdir(str(self.repo_path))
        try:
            prototype = adapter._build_tracker()
            from lib.train.data.processing_utils import sample_target
            from lib.utils.box_ops import clip_box
            from lib.utils.ce_utils import generate_mask_cond
        finally:
            os.chdir(str(old_cwd))

        required = (
            "network",
            "preprocessor",
            "params",
            "cfg",
            "output_window",
            "transform_bbox_to_crop",
        )
        missing = [name for name in required if not hasattr(prototype, name)]
        if missing:
            raise RuntimeError(
                "The installed OSTrack tracker is incompatible with batched inference; "
                f"missing attributes: {missing}"
            )

        self._prototype = prototype
        self.network = prototype.network
        self.network.eval()
        self.preprocessor = prototype.preprocessor
        self.params = prototype.params
        self.cfg = prototype.cfg
        self.output_window = prototype.output_window
        self.device = next(self.network.parameters()).device
        self._sample_target = sample_target
        self._clip_box = clip_box
        self._generate_mask_cond = generate_mask_cond
        self._states = [_OSTrackEnvState() for _ in range(self.num_envs)]

    def is_initialized(self, env_id: int) -> bool:
        self._check_env_id(env_id)
        return self._states[env_id].initialized

    def has_reference_template(self, env_id: int) -> bool:
        self._check_env_id(env_id)
        return self._states[env_id].reference_template_tensor is not None

    def response_peak(self, env_id: int) -> float | None:
        """Return the latest Hann-windowed score-map peak for one environment."""
        self._check_env_id(env_id)
        return self._states[env_id].response_peak

    def reset_envs(self, env_ids: Iterable[int] | None = None) -> None:
        ids = range(self.num_envs) if env_ids is None else env_ids
        for env_id_raw in ids:
            env_id = int(env_id_raw)
            if 0 <= env_id < self.num_envs:
                self._states[env_id] = _OSTrackEnvState()

    def initialize_env(
        self,
        env_id: int,
        image_rgb: np.ndarray,
        bbox_xywh: Iterable[float],
    ) -> None:
        self._check_env_id(env_id)
        bbox = [float(value) for value in bbox_xywh]
        if len(bbox) != 4:
            raise ValueError("bbox_xywh must contain four values")

        patch, resize_factor, attention_mask = self._sample_target(
            image_rgb,
            bbox,
            self.params.template_factor,
            output_sz=self.params.template_size,
        )
        template_tensor, _ = self._preprocess_batch([patch], [attention_mask])

        ce_template_mask = None
        if bool(self.cfg.MODEL.BACKBONE.CE_LOC):
            template_bbox = self._prototype.transform_bbox_to_crop(
                bbox,
                resize_factor,
                template_tensor.device,
            ).squeeze(1)
            ce_template_mask = self._generate_mask_cond(
                self.cfg,
                1,
                template_tensor.device,
                template_bbox,
            )

        previous_state = self._states[env_id]
        reference_template = previous_state.reference_template_tensor
        reference_ce_mask = previous_state.reference_ce_template_mask
        if reference_template is None:
            reference_template = template_tensor.detach().clone()
            reference_ce_mask = (
                None if ce_template_mask is None else ce_template_mask.detach().clone()
            )
        self._states[env_id] = _OSTrackEnvState(
            bbox_xywh=bbox,
            template_tensor=template_tensor.detach(),
            ce_template_mask=(
                None if ce_template_mask is None else ce_template_mask.detach()
            ),
            reference_template_tensor=reference_template,
            reference_ce_template_mask=reference_ce_mask,
            frame_id=0,
            response_peak=None,
        )

    def track_batch(
        self,
        items: list[tuple[int, np.ndarray] | tuple[int, np.ndarray, float]],
    ) -> tuple[dict[int, list[float]], dict[int, Exception]]:
        """Track active environments and return xywh boxes and per-env errors."""

        outputs: dict[int, list[float]] = {}
        errors: dict[int, Exception] = {}
        prepared: list[dict[str, Any]] = []

        for item_raw in items:
            if len(item_raw) == 2:
                env_id_raw, image_rgb = item_raw
                search_factor_scale = 1.0
            else:
                env_id_raw, image_rgb, search_factor_scale = item_raw
            env_id = int(env_id_raw)
            try:
                self._check_env_id(env_id)
                state = self._states[env_id]
                if not state.initialized:
                    raise RuntimeError(f"env {env_id} is not initialized")

                height, width = image_rgb.shape[:2]
                patch, resize_factor, attention_mask = self._sample_target(
                    image_rgb,
                    state.bbox_xywh,
                    float(self.params.search_factor) * float(search_factor_scale),
                    output_sz=self.params.search_size,
                )
                state.frame_id += 1
                prepared.append(
                    {
                        "env_id": env_id,
                        "height": int(height),
                        "width": int(width),
                        "patch": patch,
                        "attention_mask": attention_mask,
                        "resize_factor": float(resize_factor),
                    }
                )
            except Exception as exc:
                errors[env_id] = exc

        if not prepared:
            return outputs, errors

        batch_size = self.max_batch_size if self.max_batch_size > 0 else len(prepared)
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            chunk_ids = [int(item["env_id"]) for item in chunk]
            try:
                search_tensor, _ = self._preprocess_batch(
                    [item["patch"] for item in chunk],
                    [item["attention_mask"] for item in chunk],
                )
                template_tensor = torch.cat(
                    [
                        self._states[env_id].template_tensor
                        for env_id in chunk_ids
                    ],
                    dim=0,
                )
                ce_template_mask = self._stack_ce_masks(chunk_ids)

                with torch.no_grad():
                    out_dict = self.network.forward(
                        template=template_tensor,
                        search=search_tensor,
                        ce_template_mask=ce_template_mask,
                    )
                    response = self.output_window * out_dict["score_map"]
                    response_peaks = response.reshape(len(chunk), -1).amax(dim=1)
                    pred_boxes = self.network.box_head.cal_bbox(
                        response,
                        out_dict["size_map"],
                        out_dict["offset_map"],
                    )
                    pred_boxes = pred_boxes.reshape(len(chunk), -1, 4).mean(dim=1)
                    resize_factors = torch.tensor(
                        [item["resize_factor"] for item in chunk],
                        device=pred_boxes.device,
                        dtype=pred_boxes.dtype,
                    ).unsqueeze(-1)
                    pred_boxes = (
                        pred_boxes * float(self.params.search_size) / resize_factors
                    ).detach().cpu()
                    response_peaks = response_peaks.detach().cpu()

                for batch_index, item in enumerate(chunk):
                    env_id = int(item["env_id"])
                    state = self._states[env_id]
                    mapped = self._map_box_back(
                        state.bbox_xywh,
                        pred_boxes[batch_index].tolist(),
                        float(item["resize_factor"]),
                    )
                    clipped = self._clip_box(
                        mapped,
                        int(item["height"]),
                        int(item["width"]),
                        margin=10,
                    )
                    state.bbox_xywh = [float(value) for value in clipped]
                    state.response_peak = float(response_peaks[batch_index].item())
                    outputs[env_id] = list(state.bbox_xywh)
            except Exception as exc:
                for env_id in chunk_ids:
                    errors[env_id] = exc

        return outputs, errors

    def global_search_batch(
        self,
        items: list[tuple[int, np.ndarray, list[list[float]]]],
    ) -> tuple[dict[int, list[tuple[list[float], float]]], dict[int, Exception]]:
        """Search arbitrary full-image anchors using the preserved initial template.

        Each anchor is an xywh pseudo-box whose center and scale define one OSTrack
        search crop. The persistent local-tracker box is not modified.
        """

        outputs: dict[int, list[tuple[list[float], float]]] = {}
        errors: dict[int, Exception] = {}
        prepared: list[dict[str, Any]] = []

        for env_id_raw, image_rgb, anchors_xywh in items:
            env_id = int(env_id_raw)
            try:
                self._check_env_id(env_id)
                state = self._states[env_id]
                if state.reference_template_tensor is None:
                    raise RuntimeError(f"env {env_id} has no preserved reference template")
                height, width = image_rgb.shape[:2]
                for anchor_index, anchor_raw in enumerate(anchors_xywh):
                    anchor = [float(value) for value in anchor_raw]
                    if len(anchor) != 4:
                        raise ValueError("global-search anchor must contain four xywh values")
                    patch, resize_factor, attention_mask = self._sample_target(
                        image_rgb,
                        anchor,
                        self.params.search_factor,
                        output_sz=self.params.search_size,
                    )
                    prepared.append(
                        {
                            "env_id": env_id,
                            "anchor_index": int(anchor_index),
                            "anchor_xywh": anchor,
                            "height": int(height),
                            "width": int(width),
                            "patch": patch,
                            "attention_mask": attention_mask,
                            "resize_factor": float(resize_factor),
                        }
                    )
            except Exception as exc:
                errors[env_id] = exc

        if not prepared:
            return outputs, errors

        batch_size = self.max_batch_size if self.max_batch_size > 0 else len(prepared)
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            chunk_ids = [int(item["env_id"]) for item in chunk]
            try:
                search_tensor, _ = self._preprocess_batch(
                    [item["patch"] for item in chunk],
                    [item["attention_mask"] for item in chunk],
                )
                template_tensor = torch.cat(
                    [
                        self._states[env_id].reference_template_tensor
                        for env_id in chunk_ids
                    ],
                    dim=0,
                )
                ce_template_mask = self._stack_reference_ce_masks(chunk_ids)
                with torch.no_grad():
                    out_dict = self.network.forward(
                        template=template_tensor,
                        search=search_tensor,
                        ce_template_mask=ce_template_mask,
                    )
                    response = self.output_window * out_dict["score_map"]
                    response_peaks = response.reshape(len(chunk), -1).amax(dim=1)
                    pred_boxes = self.network.box_head.cal_bbox(
                        response,
                        out_dict["size_map"],
                        out_dict["offset_map"],
                    )
                    pred_boxes = pred_boxes.reshape(len(chunk), -1, 4).mean(dim=1)
                    resize_factors = torch.tensor(
                        [item["resize_factor"] for item in chunk],
                        device=pred_boxes.device,
                        dtype=pred_boxes.dtype,
                    ).unsqueeze(-1)
                    pred_boxes = (
                        pred_boxes * float(self.params.search_size) / resize_factors
                    ).detach().cpu()
                    response_peaks = response_peaks.detach().cpu()

                for batch_index, item in enumerate(chunk):
                    env_id = int(item["env_id"])
                    mapped = self._map_box_back(
                        item["anchor_xywh"],
                        pred_boxes[batch_index].tolist(),
                        float(item["resize_factor"]),
                    )
                    clipped = self._clip_box(
                        mapped,
                        int(item["height"]),
                        int(item["width"]),
                        margin=1,
                    )
                    outputs.setdefault(env_id, []).append(
                        (
                            [float(value) for value in clipped],
                            float(response_peaks[batch_index].item()),
                        )
                    )
            except Exception as exc:
                for env_id in chunk_ids:
                    errors[env_id] = exc

        for candidates in outputs.values():
            candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        return outputs, errors

    def _preprocess_batch(
        self,
        patches: list[np.ndarray],
        attention_masks: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_array = np.ascontiguousarray(np.stack(patches, axis=0))
        mask_array = np.ascontiguousarray(np.stack(attention_masks, axis=0))

        image_tensor = torch.from_numpy(patch_array).to(
            device=self.device,
            dtype=torch.float32,
        )
        image_tensor = image_tensor.permute(0, 3, 1, 2).contiguous()
        mean = self.preprocessor.mean.to(device=self.device, dtype=torch.float32)
        std = self.preprocessor.std.to(device=self.device, dtype=torch.float32)
        image_tensor = ((image_tensor / 255.0) - mean) / std

        mask_tensor = torch.from_numpy(mask_array).to(
            device=self.device,
            dtype=torch.bool,
        )
        return image_tensor, mask_tensor

    def _stack_ce_masks(self, env_ids: list[int]) -> torch.Tensor | None:
        masks = [self._states[env_id].ce_template_mask for env_id in env_ids]
        if all(mask is None for mask in masks):
            return None
        if any(mask is None for mask in masks):
            raise RuntimeError("Mixed CE template-mask states cannot share one batch")
        return torch.cat([mask for mask in masks if mask is not None], dim=0)

    def _stack_reference_ce_masks(self, env_ids: list[int]) -> torch.Tensor | None:
        masks = [self._states[env_id].reference_ce_template_mask for env_id in env_ids]
        if all(mask is None for mask in masks):
            return None
        if any(mask is None for mask in masks):
            raise RuntimeError("Mixed reference CE template-mask states cannot share one batch")
        return torch.cat([mask for mask in masks if mask is not None], dim=0)

    def _map_box_back(
        self,
        previous_box_xywh: list[float] | None,
        pred_box_cxcywh: list[float],
        resize_factor: float,
    ) -> list[float]:
        if previous_box_xywh is None:
            raise RuntimeError("Cannot map a box without previous tracker state")
        x_prev, y_prev, width_prev, height_prev = previous_box_xywh
        cx_prev = x_prev + 0.5 * width_prev
        cy_prev = y_prev + 0.5 * height_prev
        cx, cy, width, height = pred_box_cxcywh
        half_side = 0.5 * float(self.params.search_size) / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [
            cx_real - 0.5 * width,
            cy_real - 0.5 * height,
            width,
            height,
        ]

    def _check_env_id(self, env_id: int) -> None:
        if not 0 <= int(env_id) < self.num_envs:
            raise IndexError(f"env_id out of range: {env_id}")


__all__ = ["SharedBatchedOSTrack"]

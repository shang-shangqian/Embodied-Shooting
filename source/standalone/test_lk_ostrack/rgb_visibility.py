from __future__ import annotations

import numpy as np


def rgb_target_visibility_mask(
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    *,
    min_chroma: int = 24,
    min_colored_fraction: float = 0.01,
    min_colored_pixels: int = 4,
) -> np.ndarray:
    """Return whether each target box contains enough rendered color pixels.

    The collection targets use saturated per-target colors while the robot,
    ground, and sky in this scene are nearly achromatic.  This render-space
    check catches targets hidden by the ground even when their mathematical
    projection is inside the image.
    """
    image = np.asarray(image_rgb)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected RGB image [H,W,C], got {image.shape}")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected target boxes [N,4], got {boxes.shape}")
    if min_chroma < 0:
        raise ValueError("min_chroma must be non-negative")
    if not 0.0 <= min_colored_fraction <= 1.0:
        raise ValueError("min_colored_fraction must be in [0, 1]")
    if min_colored_pixels <= 0:
        raise ValueError("min_colored_pixels must be positive")

    height, width = image.shape[:2]
    rgb = image[..., :3].astype(np.int16, copy=False)
    visible = np.zeros((boxes.shape[0],), dtype=bool)
    for target_id, (x1, y1, x2, y2) in enumerate(boxes):
        if not np.isfinite((x1, y1, x2, y2)).all():
            continue
        left = int(np.clip(np.floor(x1), 0, width))
        top = int(np.clip(np.floor(y1), 0, height))
        right = int(np.clip(np.ceil(x2), 0, width))
        bottom = int(np.clip(np.ceil(y2), 0, height))
        if right <= left or bottom <= top:
            continue
        patch = rgb[top:bottom, left:right]
        chroma = patch.max(axis=2) - patch.min(axis=2)
        required_pixels = max(
            int(min_colored_pixels),
            int(np.ceil(float(chroma.size) * float(min_colored_fraction))),
        )
        visible[target_id] = (
            int(np.count_nonzero(chroma >= min_chroma)) >= required_pixels
        )
    return visible

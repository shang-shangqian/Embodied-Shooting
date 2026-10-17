from __future__ import annotations

import numpy as np
import pytest

from test_lk_ostrack.rgb_visibility import rgb_target_visibility_mask


def test_rgb_target_visibility_detects_colored_targets_in_boxes() -> None:
    image = np.full((40, 60, 3), 128, dtype=np.uint8)
    image[5:15, 5:15] = (240, 20, 20)
    image[20:30, 35:45] = (20, 60, 240)
    boxes = np.asarray([[4, 4, 16, 16], [34, 19, 46, 31]], dtype=np.float32)

    assert rgb_target_visibility_mask(image, boxes).tolist() == [True, True]


def test_rgb_target_visibility_rejects_occluded_and_invalid_boxes() -> None:
    image = np.full((30, 40, 3), 90, dtype=np.uint8)
    image[2:4, 2:4] = (255, 0, 0)
    boxes = np.asarray(
        [
            [0, 0, 10, 10],
            [15, 10, 25, 20],
            [np.nan, 0, 2, 2],
            [50, 50, 60, 60],
        ],
        dtype=np.float32,
    )

    assert rgb_target_visibility_mask(image, boxes).tolist() == [True, False, False, False]


@pytest.mark.parametrize(
    ("image", "boxes"),
    [
        (np.zeros((10, 10), dtype=np.uint8), np.zeros((1, 4))),
        (np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((4,))),
    ],
)
def test_rgb_target_visibility_validates_shapes(
    image: np.ndarray, boxes: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        rgb_target_visibility_mask(image, boxes)

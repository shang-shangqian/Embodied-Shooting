from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for an OpenCV-style camera frame.

    Coordinate convention:
    - camera x: right
    - camera y: down
    - camera z: forward
    - yaw > 0 means target is to the right
    - pitch > 0 means target is above image center
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraIntrinsics":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "K" in data:
            k = np.asarray(data["K"], dtype=np.float64)
            if k.shape == (9,):
                k = k.reshape(3, 3)
            if k.shape != (3, 3):
                raise ValueError("K must be a 3x3 matrix or a flat list of 9 values.")
            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])
        else:
            fx = float(data["fx"])
            fy = float(data["fy"])
            cx = float(data["cx"])
            cy = float(data["cy"])

        width = data.get("width", None)
        height = data.get("height", None)
        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=None if width is None else int(width),
            height=None if height is None else int(height),
        )

    @classmethod
    def from_args(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int | None = None,
        height: int | None = None,
    ) -> "CameraIntrinsics":
        return cls(float(fx), float(fy), float(cx), float(cy), width, height)

    def pixel_to_normalized(self, u: float, v: float) -> tuple[float, float]:
        return (float(u) - self.cx) / self.fx, (float(v) - self.cy) / self.fy

    def direction_from_pixel(self, u: float, v: float) -> tuple[float, float]:
        x_norm, y_norm = self.pixel_to_normalized(u, v)
        yaw = math.atan2(x_norm, 1.0)
        pitch = math.atan2(-y_norm, math.sqrt(x_norm * x_norm + 1.0))
        return yaw, pitch

    def backproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        z = float(depth_m)
        x_norm, y_norm = self.pixel_to_normalized(u, v)
        return np.asarray([x_norm * z, y_norm * z, z], dtype=np.float64)

    def project(self, point_cam: Iterable[float]) -> tuple[float, float]:
        point = np.asarray(list(point_cam), dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"Expected point shape (3,), got {point.shape}.")
        z = max(float(point[2]), 1e-9)
        u = self.fx * float(point[0]) / z + self.cx
        v = self.fy * float(point[1]) / z + self.cy
        return u, v

    def direction_from_point(self, point_cam: Iterable[float]) -> tuple[float, float]:
        point = np.asarray(list(point_cam), dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"Expected point shape (3,), got {point.shape}.")
        x, y, z = float(point[0]), float(point[1]), max(float(point[2]), 1e-9)
        yaw = math.atan2(x, z)
        pitch = math.atan2(-y, math.sqrt(x * x + z * z))
        return yaw, pitch


def bbox_center_xy(bbox_xyxy: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def bbox_clip(
    bbox_xyxy: Iterable[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1_i = int(max(0, min(width - 1, math.floor(x1))))
    y1_i = int(max(0, min(height - 1, math.floor(y1))))
    x2_i = int(max(0, min(width, math.ceil(x2))))
    y2_i = int(max(0, min(height, math.ceil(y2))))
    if x2_i <= x1_i:
        x2_i = min(width, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(height, y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i
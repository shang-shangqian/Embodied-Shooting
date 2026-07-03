from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

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
        values = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        for name, value in values.items():
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"Camera intrinsic {name} must be finite, got {value!r}.")
        if float(fx) <= 0.0 or float(fy) <= 0.0:
            raise ValueError(f"Camera focal lengths must be positive, got fx={fx}, fy={fy}.")
        width_i = None if width is None else int(width)
        height_i = None if height is None else int(height)
        if width_i is not None and width_i <= 0:
            raise ValueError(f"Camera width must be positive, got {width}.")
        if height_i is not None and height_i <= 0:
            raise ValueError(f"Camera height must be positive, got {height}.")
        return cls(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=width_i,
            height=height_i,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraIntrinsics":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("intrinsics"), dict):
            data = data["intrinsics"]

        if isinstance(data, list):
            fx, fy, cx, cy = _intrinsics_from_matrix(data, path)
            return cls.from_args(fx=fx, fy=fy, cx=cx, cy=cy)

        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object or a 3x3 intrinsic matrix.")

        matrix = _first_present(data, ("intrinsic_matrix", "camera_matrix", "K"))
        if matrix is not None:
            fx, fy, cx, cy = _intrinsics_from_matrix(matrix, path)
        else:
            fx = _first_present(data, ("fx", "focal_x", "focal_length_x", "camera_fx"))
            fy = _first_present(data, ("fy", "focal_y", "focal_length_y", "camera_fy"))
            cx = _first_present(data, ("cx", "principal_x", "principal_point_x", "camera_cx"))
            cy = _first_present(data, ("cy", "principal_y", "principal_point_y", "camera_cy"))
            missing = [
                name
                for name, value in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy))
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"{path} must contain fx/fy/cx/cy or a 3x3 intrinsic matrix. "
                    f"Missing: {', '.join(missing)}"
                )

        width = _first_present(data, ("width", "image_width", "camera_width", "w"))
        height = _first_present(data, ("height", "image_height", "camera_height", "h"))
        return cls.from_args(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=None if width is None else int(float(width)),
            height=None if height is None else int(float(height)),
        )

    def direction_from_pixel(self, u: float, v: float) -> tuple[float, float]:
        x_norm = (float(u) - self.cx) / self.fx
        y_norm = (float(v) - self.cy) / self.fy
        yaw = math.atan2(x_norm, 1.0)
        pitch = math.atan2(-y_norm, math.sqrt(x_norm * x_norm + 1.0))
        return float(yaw), float(pitch)

    def backproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        z = float(depth_m)
        x = (float(u) - self.cx) / self.fx * z
        y = (float(v) - self.cy) / self.fy * z
        return np.asarray([x, y, z], dtype=np.float64)


def _first_present(data: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _intrinsics_from_matrix(value: Any, path: Path) -> tuple[float, float, float, float]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3):
        raise ValueError(f"{path} intrinsic matrix must be 3x3 or a flat list of 9 numbers.")
    return float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])


def build_intrinsics(args: Any) -> CameraIntrinsics:
    if args.intrinsics_json is not None:
        return CameraIntrinsics.from_json(args.intrinsics_json)

    missing = [name for name in ("fx", "fy", "cx", "cy") if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "Provide --intrinsics_json or all of --fx --fy --cx --cy. "
            f"Missing: {', '.join(missing)}"
        )
    return CameraIntrinsics.from_args(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=args.width,
        height=args.height,
    )


def load_initial_box(args: Any) -> list[float]:
    if args.initial_box is not None:
        return [float(v) for v in args.initial_box]
    if args.initial_box_file is None:
        raise ValueError("--initial_box or --initial_box_file is required.")
    text = Path(args.initial_box_file).read_text(encoding="utf-8").strip()
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError(f"{args.initial_box_file} must contain exactly four numbers.")
    return values

def bbox_center_xy(box_xyxy: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def bbox_clip(box_xyxy: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got width={width}, height={height}.")
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = min(max(left, 0.0), float(width - 1))
    top = min(max(top, 0.0), float(height - 1))
    right = min(max(right, left + 1.0), float(width))
    bottom = min(max(bottom, top + 1.0), float(height))
    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))
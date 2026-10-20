from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from collections.abc import Iterable
from typing import Any, Mapping

import numpy as np
from PIL import Image


GROUND_TRUTH_FIELDS = [
    "frame_index",
    "timestamp_s",
    "target_id",
    "visible",
    "fully_visible",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_raw_x1",
    "bbox_raw_y1",
    "bbox_raw_x2",
    "bbox_raw_y2",
    "center_u",
    "center_v",
    "yaw_rad",
    "pitch_rad",
    "distance_m",
    "lidar_range_m",
    "lidar_valid",
    "x_cam_m",
    "y_cam_m",
    "z_cam_m",
    "vx_cam_mps",
    "vy_cam_mps",
    "vz_cam_mps",
    "x_yaw_m",
    "y_yaw_m",
    "z_yaw_m",
    "vx_yaw_mps",
    "vy_yaw_mps",
    "vz_yaw_mps",
    "camera_position_yaw_x_m",
    "camera_position_yaw_y_m",
    "camera_position_yaw_z_m",
    "rotation_yaw_from_camera_0_0",
    "rotation_yaw_from_camera_0_1",
    "rotation_yaw_from_camera_0_2",
    "rotation_yaw_from_camera_1_0",
    "rotation_yaw_from_camera_1_1",
    "rotation_yaw_from_camera_1_2",
    "rotation_yaw_from_camera_2_0",
    "rotation_yaw_from_camera_2_1",
    "rotation_yaw_from_camera_2_2",
    "base_vx_yaw_mps",
    "base_vy_yaw_mps",
    "base_vz_yaw_mps",
    "base_wx_yaw_radps",
    "base_wy_yaw_radps",
    "base_wz_yaw_radps",
    "camera_vx_cam_mps",
    "camera_vy_cam_mps",
    "camera_vz_cam_mps",
    "camera_wx_cam_radps",
    "camera_wy_cam_radps",
    "camera_wz_cam_radps",
]


class IsaacSimSequenceWriter:
    """Write one fixed-length multi-target sequence, optionally without RGB."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        sequence_id: int,
        env_id: int,
        target_count: int,
        fps: float,
        intrinsics: np.ndarray,
        initial_boxes_xyxy: np.ndarray,
        expected_frames: int,
        image_width: int | None = None,
        image_height: int | None = None,
        save_images: bool = True,
        recorded_target_ids: Iterable[int] | None = None,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> None:
        if target_count < 2:
            raise ValueError("target_count must be at least 2")
        if fps <= 0.0 or expected_frames <= 0:
            raise ValueError("fps and expected_frames must be positive")
        matrix = np.asarray(intrinsics, dtype=np.float64)
        boxes = np.asarray(initial_boxes_xyxy, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected intrinsics [3,3], got {matrix.shape}")
        if boxes.shape != (target_count, 4):
            raise ValueError(
                f"Expected initial boxes [{target_count},4], got {boxes.shape}"
            )
        if not np.isfinite(matrix).all() or not np.isfinite(boxes).all():
            raise ValueError("Intrinsics and initial boxes must be finite")

        self.dataset_root = Path(dataset_root).resolve()
        self.sequence_id = int(sequence_id)
        self.env_id = int(env_id)
        self.target_count = int(target_count)
        if recorded_target_ids is None:
            recorded_ids = tuple(range(self.target_count))
        else:
            recorded_ids = tuple(int(target_id) for target_id in recorded_target_ids)
            if not recorded_ids:
                raise ValueError("recorded_target_ids must not be empty")
            if len(set(recorded_ids)) != len(recorded_ids):
                raise ValueError("recorded_target_ids must not contain duplicates")
            invalid_ids = [
                target_id
                for target_id in recorded_ids
                if not 0 <= target_id < self.target_count
            ]
            if invalid_ids:
                raise ValueError(
                    "recorded_target_ids must be within the scene target range; "
                    f"invalid IDs: {invalid_ids}"
                )
        self.recorded_target_ids = recorded_ids
        self.recorded_target_count = len(recorded_ids)
        self.fps = float(fps)
        self.expected_frames = int(expected_frames)
        self.save_images = bool(save_images)
        self.sequence_name = f"seq_{self.sequence_id:06d}_env_{self.env_id:03d}"
        self.sequence_dir = self.dataset_root / "sequences" / self.sequence_name
        self.rgb_dir = self.sequence_dir / "rgb"
        if self.save_images:
            self.rgb_dir.mkdir(parents=True, exist_ok=False)
        else:
            self.sequence_dir.mkdir(parents=True, exist_ok=False)
        self._rows: list[dict[str, Any]] = []
        self._frame_count = 0
        self._finished = False
        self._extra_meta = dict(extra_meta or {})

        intrinsics_payload = {
            "K": matrix.tolist(),
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
        }
        if image_width is not None:
            intrinsics_payload["width"] = int(image_width)
        if image_height is not None:
            intrinsics_payload["height"] = int(image_height)
        self._write_json(self.sequence_dir / "intrinsics.json", intrinsics_payload)
        initial_targets = {
            "targets": [
                {
                    "id": target_id,
                    "bbox": [float(value) for value in boxes[target_id]],
                    "priority": 0.0,
                }
                for target_id in range(self.target_count)
            ]
        }
        self._write_json(self.sequence_dir / "initial_targets.json", initial_targets)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def complete(self) -> bool:
        return self._frame_count >= self.expected_frames

    def add_frame(
        self,
        image_rgb: np.ndarray | None,
        ground_truth: Mapping[str, np.ndarray],
        *,
        timestamp_s: float,
    ) -> None:
        if self._finished:
            raise RuntimeError("Cannot add frames after finish()")
        if self.complete:
            raise RuntimeError("Sequence already contains expected_frames")
        if not np.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")

        arrays = self._validate_ground_truth(ground_truth)
        if self.save_images:
            if image_rgb is None:
                raise ValueError("image_rgb is required when save_images=True")
            image = np.asarray(image_rgb)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("image_rgb must be uint8 [H,W,3]")
            Image.fromarray(image, mode="RGB").save(
                self.rgb_dir / f"{self._frame_count:06d}.png"
            )
        elif image_rgb is not None:
            raise ValueError("image_rgb must be None when save_images=False")
        for target_id in self.recorded_target_ids:
            bbox = arrays["bbox_xyxy"][target_id]
            raw_bbox = arrays["bbox_xyxy_raw"][target_id]
            center = arrays["center_uv"][target_id]
            direction = arrays["direction"][target_id]
            lidar_range = arrays["lidar_range_m"][target_id]
            pos_cam = arrays["target_pos_cam"][target_id]
            vel_cam = arrays["target_vel_cam"][target_id]
            pos_yaw = arrays["target_pos_yaw"][target_id]
            vel_yaw = arrays["target_vel_yaw"][target_id]
            camera_position_yaw = arrays["camera_position_yaw"][target_id]
            rotation_yaw_from_camera = arrays["rotation_yaw_from_camera"][target_id]
            base_linear_velocity_yaw = arrays["base_linear_velocity_yaw"]
            base_angular_velocity_yaw = arrays["base_angular_velocity_yaw"]
            camera_linear_velocity = arrays["camera_linear_velocity_cam"][target_id]
            camera_angular_velocity = arrays["camera_angular_velocity_cam"][target_id]
            self._rows.append(
                {
                    "frame_index": self._frame_count,
                    "timestamp_s": float(timestamp_s),
                    "target_id": target_id,
                    "visible": int(bool(arrays["visible"][target_id])),
                    "fully_visible": int(bool(arrays["fully_visible"][target_id])),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "bbox_raw_x1": raw_bbox[0],
                    "bbox_raw_y1": raw_bbox[1],
                    "bbox_raw_x2": raw_bbox[2],
                    "bbox_raw_y2": raw_bbox[3],
                    "center_u": center[0],
                    "center_v": center[1],
                    "yaw_rad": direction[0],
                    "pitch_rad": direction[1],
                    "distance_m": arrays["distance_m"][target_id],
                    "lidar_range_m": lidar_range,
                    "lidar_valid": int(bool(arrays["lidar_valid"][target_id])),
                    "x_cam_m": pos_cam[0],
                    "y_cam_m": pos_cam[1],
                    "z_cam_m": pos_cam[2],
                    "vx_cam_mps": vel_cam[0],
                    "vy_cam_mps": vel_cam[1],
                    "vz_cam_mps": vel_cam[2],
                    "x_yaw_m": pos_yaw[0],
                    "y_yaw_m": pos_yaw[1],
                    "z_yaw_m": pos_yaw[2],
                    "vx_yaw_mps": vel_yaw[0],
                    "vy_yaw_mps": vel_yaw[1],
                    "vz_yaw_mps": vel_yaw[2],
                    "camera_position_yaw_x_m": camera_position_yaw[0],
                    "camera_position_yaw_y_m": camera_position_yaw[1],
                    "camera_position_yaw_z_m": camera_position_yaw[2],
                    **{
                        f"rotation_yaw_from_camera_{row}_{column}": rotation_yaw_from_camera[row, column]
                        for row in range(3)
                        for column in range(3)
                    },
                    "base_vx_yaw_mps": base_linear_velocity_yaw[0],
                    "base_vy_yaw_mps": base_linear_velocity_yaw[1],
                    "base_vz_yaw_mps": base_linear_velocity_yaw[2],
                    "base_wx_yaw_radps": base_angular_velocity_yaw[0],
                    "base_wy_yaw_radps": base_angular_velocity_yaw[1],
                    "base_wz_yaw_radps": base_angular_velocity_yaw[2],
                    "camera_vx_cam_mps": camera_linear_velocity[0],
                    "camera_vy_cam_mps": camera_linear_velocity[1],
                    "camera_vz_cam_mps": camera_linear_velocity[2],
                    "camera_wx_cam_radps": camera_angular_velocity[0],
                    "camera_wy_cam_radps": camera_angular_velocity[1],
                    "camera_wz_cam_radps": camera_angular_velocity[2],
                }
            )
        self._frame_count += 1

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("finish() was already called")
        if not self.complete:
            raise RuntimeError(
                f"Sequence has {self._frame_count}/{self.expected_frames} frames"
            )
        with (self.sequence_dir / "ground_truth.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=GROUND_TRUTH_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)

        meta = {
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "env_id": self.env_id,
            "target_count": self.target_count,
            "recorded_target_count": self.recorded_target_count,
            "recorded_target_ids": list(self.recorded_target_ids),
            "frame_count": self._frame_count,
            "fps": self.fps,
            "duration_s": self._frame_count / self.fps,
            "save_images": self.save_images,
            "storage_format": "rgb_csv" if self.save_images else "csv_only",
            **self._extra_meta,
        }
        self._write_json(self.sequence_dir / "sequence_meta.json", meta)
        self._finished = True
        relative_dir = self.sequence_dir.relative_to(self.dataset_root)
        return {
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "env_id": self.env_id,
            "seq_dir": str(relative_dir),
            "rgb_dir": str(relative_dir / "rgb") if self.save_images else "",
            "intrinsics_json": str(relative_dir / "intrinsics.json"),
            "initial_targets_json": str(relative_dir / "initial_targets.json"),
            "ground_truth_csv": str(relative_dir / "ground_truth.csv"),
            "frame_count": self._frame_count,
            "target_count": self.target_count,
            "recorded_target_count": self.recorded_target_count,
            "fps": self.fps,
            "save_images": int(self.save_images),
            "tracking_frame": str(self._extra_meta.get("tracking_frame", "camera")),
            "complete": 1,
        }

    def abort(self) -> None:
        if not self._finished and self.sequence_dir.exists():
            shutil.rmtree(self.sequence_dir)
        self._finished = True

    def _validate_ground_truth(
        self, ground_truth: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        expected_shapes = {
            "bbox_xyxy": (self.target_count, 4),
            "bbox_xyxy_raw": (self.target_count, 4),
            "center_uv": (self.target_count, 2),
            "direction": (self.target_count, 2),
            "distance_m": (self.target_count,),
            "lidar_range_m": (self.target_count,),
            "lidar_valid": (self.target_count,),
            "target_pos_cam": (self.target_count, 3),
            "target_vel_cam": (self.target_count, 3),
            "target_pos_yaw": (self.target_count, 3),
            "target_vel_yaw": (self.target_count, 3),
            "camera_position_yaw": (self.target_count, 3),
            "rotation_yaw_from_camera": (self.target_count, 3, 3),
            "base_linear_velocity_yaw": (3,),
            "base_angular_velocity_yaw": (3,),
            "camera_linear_velocity_cam": (self.target_count, 3),
            "camera_angular_velocity_cam": (self.target_count, 3),
            "visible": (self.target_count,),
            "fully_visible": (self.target_count,),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, expected_shape in expected_shapes.items():
            if name not in ground_truth:
                if name == "target_pos_yaw":
                    arrays[name] = np.asarray(ground_truth["target_pos_cam"]).copy()
                elif name == "target_vel_yaw":
                    arrays[name] = np.asarray(ground_truth["target_vel_cam"]).copy()
                elif name == "camera_position_yaw":
                    arrays[name] = np.zeros(expected_shape, dtype=np.float64)
                elif name == "rotation_yaw_from_camera":
                    arrays[name] = np.broadcast_to(
                        np.eye(3, dtype=np.float64), expected_shape
                    ).copy()
                elif name in {"base_linear_velocity_yaw", "base_angular_velocity_yaw"}:
                    # Legacy datasets did not record base motion. Keep them
                    # readable, but static filtering requires regenerated data.
                    arrays[name] = np.zeros(expected_shape, dtype=np.float64)
                elif name == "lidar_range_m":
                    arrays[name] = np.full(expected_shape, np.inf, dtype=np.float64)
                elif name == "lidar_valid":
                    arrays[name] = np.zeros(expected_shape, dtype=bool)
                else:
                    raise KeyError(f"Missing ground-truth field: {name}")
                array = arrays[name]
            else:
                array = np.asarray(ground_truth[name])
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {array.shape}"
                )
            if name == "lidar_range_m":
                lidar_valid = np.asarray(
                    ground_truth.get(
                        "lidar_valid", np.zeros(expected_shape, dtype=bool)
                    ),
                    dtype=bool,
                )
                if np.any(lidar_valid & (~np.isfinite(array) | (array <= 0.0))):
                    raise ValueError(
                        "lidar_range_m must be finite and positive when lidar_valid"
                    )
            elif name not in {"visible", "fully_visible", "lidar_valid"} and not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
            arrays[name] = array
        return arrays

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def write_dataset_manifest(
    dataset_root: str | Path,
    rows: list[Mapping[str, Any]],
    dataset_meta: Mapping[str, Any],
) -> None:
    root = Path(dataset_root)
    if not rows:
        raise ValueError("Cannot write an empty dataset manifest")
    fields = list(rows[0].keys())
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "dataset_meta.json").write_text(
        json.dumps(dict(dataset_meta), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_collection_dataset(dataset_root: str | Path) -> dict[str, int]:
    root = Path(dataset_root)
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        manifest_rows = list(csv.DictReader(file))
    if not manifest_rows:
        raise ValueError("Dataset manifest is empty")

    total_frames = 0
    total_targets = 0
    for manifest_row in manifest_rows:
        sequence_dir = root / manifest_row["seq_dir"]
        target_count = int(manifest_row["target_count"])
        frame_count = int(manifest_row["frame_count"])
        required = [
            sequence_dir / "intrinsics.json",
            sequence_dir / "initial_targets.json",
            sequence_dir / "ground_truth.csv",
            sequence_dir / "sequence_meta.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Sequence is incomplete: {missing}")
        save_images = bool(int(manifest_row.get("save_images", "1")))
        if save_images:
            images = sorted((sequence_dir / "rgb").glob("*.png"))
            if len(images) != frame_count:
                raise ValueError(
                    f"{sequence_dir}: images={len(images)} expected={frame_count}"
                )
        elif (sequence_dir / "rgb").exists():
            raise ValueError(f"{sequence_dir}: csv-only sequence must not contain rgb/")
        with (sequence_dir / "ground_truth.csv").open(
            "r", newline="", encoding="utf-8"
        ) as file:
            gt_rows = list(csv.DictReader(file))
        recorded_target_count = int(
            manifest_row.get("recorded_target_count", target_count)
        )
        if not 1 <= recorded_target_count <= target_count:
            raise ValueError(
                f"{sequence_dir}: recorded_target_count={recorded_target_count} "
                f"must be within [1, {target_count}]"
            )
        expected_rows = frame_count * recorded_target_count
        if len(gt_rows) != expected_rows:
            raise ValueError(
                f"{sequence_dir}: GT rows={len(gt_rows)} expected={expected_rows}"
            )
        frame_target_ids: dict[int, set[int]] = {}
        for row in gt_rows:
            frame_index = int(row["frame_index"])
            target_id = int(row["target_id"])
            if not 0 <= target_id < target_count:
                raise ValueError(
                    f"{sequence_dir}: target_id={target_id} is outside the scene"
                )
            frame_target_ids.setdefault(frame_index, set()).add(target_id)
        if set(frame_target_ids) != set(range(frame_count)):
            raise ValueError(f"{sequence_dir}: frame indices are incomplete")
        if any(
            len(target_ids) != recorded_target_count
            for target_ids in frame_target_ids.values()
        ):
            raise ValueError(
                f"{sequence_dir}: recorded target IDs are inconsistent per frame"
            )
        stable_target_ids = set.union(*frame_target_ids.values())
        if len(stable_target_ids) != recorded_target_count:
            raise ValueError(
                f"{sequence_dir}: recorded target IDs changed within the sequence"
            )
        initial = json.loads(
            (sequence_dir / "initial_targets.json").read_text(encoding="utf-8")
        )
        if len(initial.get("targets", [])) != target_count:
            raise ValueError(f"{sequence_dir}: initial target count is inconsistent")
        total_frames += frame_count
        total_targets += target_count
    return {
        "sequences": len(manifest_rows),
        "frames": total_frames,
        "targets_per_sequence_sum": total_targets,
    }


__all__ = [
    "GROUND_TRUTH_FIELDS",
    "IsaacSimSequenceWriter",
    "validate_collection_dataset",
    "write_dataset_manifest",
]

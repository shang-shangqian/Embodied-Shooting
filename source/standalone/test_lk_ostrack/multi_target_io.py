from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .multi_target_tracker import BBox, MultiTargetFrameResult


def load_initial_targets(path: str | Path) -> tuple[dict[int, BBox], dict[int, float]]:
    """Load target IDs, xyxy boxes, and optional priority biases.

    JSON accepts ``{"targets": [{"id": 3, "bbox": [...], "priority": 0.2}]}``,
    a mapping from ID to box, or a list of boxes. Text accepts one target per
    line as ``id x1 y1 x2 y2 [priority]``.
    """

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Initial target file not found: {input_path}")
    if input_path.suffix.lower() == ".json":
        return _parse_json_targets(json.loads(input_path.read_text(encoding="utf-8")))
    return _parse_text_targets(input_path.read_text(encoding="utf-8").splitlines())


def write_multi_target_csv(
    results: Iterable[MultiTargetFrameResult], out_csv: str | Path
) -> None:
    output_path = Path(out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index",
        "timestamp_s",
        "target_id",
        "selected",
        "status",
        "source",
        "confidence",
        "priority",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_u",
        "center_v",
        "predicted_center_u",
        "predicted_center_v",
        "yaw_rad",
        "pitch_rad",
        "predicted_yaw_rad",
        "predicted_pitch_rad",
        "distance_m",
        "speed_m_s",
        "kinematics_source",
        "distance_priority_score",
        "speed_priority_score",
        "yaw_deg",
        "pitch_deg",
        "predicted_yaw_deg",
        "predicted_pitch_deg",
        "corners",
        "ransac_inliers",
        "median_fb_error_px",
        "missed_frames",
        "fine_score",
        "motion_confidence",
        "point_retention",
        "ransac_inlier_ratio",
        "bbox_innovation",
        "bbox_iou_with_prediction",
        "fine_policy",
        "fine_action",
        "fine_trigger_reason",
        "fine_risk",
        "fine_bad_frame_streak",
        "frames_since_fine_attempt",
        "frames_since_fine_success",
        "fine_attempted",
        "fine_applied",
        "fine_latency_ms",
        "fine_failure_count",
        "fine_correction_center_px",
        "fine_correction_iou",
        "measurement_discontinuity",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for target in result.targets:
                writer.writerow(
                    {
                        "frame_index": result.frame_index,
                        "timestamp_s": result.timestamp_s,
                        "target_id": target.target_id,
                        "selected": int(target.selected),
                        "status": target.status,
                        "source": target.source,
                        "confidence": target.confidence,
                        "priority": target.priority,
                        "bbox_x1": target.bbox_xyxy[0],
                        "bbox_y1": target.bbox_xyxy[1],
                        "bbox_x2": target.bbox_xyxy[2],
                        "bbox_y2": target.bbox_xyxy[3],
                        "center_u": target.center_xy[0],
                        "center_v": target.center_xy[1],
                        "predicted_center_u": target.predicted_center_xy[0],
                        "predicted_center_v": target.predicted_center_xy[1],
                        "yaw_rad": target.yaw_rad,
                        "pitch_rad": target.pitch_rad,
                        "predicted_yaw_rad": target.predicted_yaw_rad,
                        "predicted_pitch_rad": target.predicted_pitch_rad,
                        "distance_m": target.distance_m,
                        "speed_m_s": target.speed_m_s,
                        "kinematics_source": target.kinematics_source,
                        "distance_priority_score": target.distance_priority_score,
                        "speed_priority_score": target.speed_priority_score,
                        "yaw_deg": math.degrees(target.yaw_rad),
                        "pitch_deg": math.degrees(target.pitch_rad),
                        "predicted_yaw_deg": math.degrees(target.predicted_yaw_rad),
                        "predicted_pitch_deg": math.degrees(target.predicted_pitch_rad),
                        "corners": target.corners,
                        "ransac_inliers": target.ransac_inliers,
                        "median_fb_error_px": target.median_fb_error_px,
                        "missed_frames": target.missed_frames,
                        "fine_score": target.fine_score,
                        "motion_confidence": target.motion_confidence,
                        "point_retention": target.point_retention,
                        "ransac_inlier_ratio": target.ransac_inlier_ratio,
                        "bbox_innovation": target.bbox_innovation,
                        "bbox_iou_with_prediction": target.bbox_iou_with_prediction,
                        "fine_policy": target.fine_policy,
                        "fine_action": target.fine_action,
                        "fine_trigger_reason": target.fine_trigger_reason,
                        "fine_risk": target.fine_risk,
                        "fine_bad_frame_streak": target.fine_bad_frame_streak,
                        "frames_since_fine_attempt": target.frames_since_fine_attempt,
                        "frames_since_fine_success": target.frames_since_fine_success,
                        "fine_attempted": int(target.fine_attempted),
                        "fine_applied": int(target.fine_applied),
                        "fine_latency_ms": target.fine_latency_ms,
                        "fine_failure_count": target.fine_failure_count,
                        "fine_correction_center_px": target.fine_correction_center_px,
                        "fine_correction_iou": target.fine_correction_iou,
                        "measurement_discontinuity": int(target.measurement_discontinuity),
                    }
                )


def write_multi_target_overlays(
    rgb_paths: Iterable[str | Path],
    results: Iterable[MultiTargetFrameResult],
    out_dir: str | Path,
) -> None:
    paths = [Path(path) for path in rgb_paths]
    frame_results = list(results)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(paths) != len(frame_results):
        raise ValueError("RGB path and result counts do not match")

    for rgb_path, result in zip(paths, frame_results):
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
        for target in result.targets:
            color = (0, 60, 255) if target.selected else _target_color(target.target_id)
            x1, y1, x2, y2 = (int(round(value)) for value in target.bbox_xyxy)
            thickness = 3 if target.selected else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            center = tuple(int(round(value)) for value in target.center_xy)
            predicted = tuple(int(round(value)) for value in target.predicted_center_xy)
            cv2.circle(image, center, 3, color, -1)
            cv2.arrowedLine(image, center, predicted, color, 2, tipLength=0.35)
            label = (
                f"ID {target.target_id} {target.status} "
                f"D={target.distance_m:.1f}m V={target.speed_m_s:.1f}m/s "
                f"C={target.confidence:.2f} P={target.priority:.2f}"
            )
            cv2.putText(
                image,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        selected = result.selected_target
        if selected is not None:
            summary = (
                f"PRIMARY ID {selected.target_id} "
                f"yaw={math.degrees(selected.predicted_yaw_rad):+.2f} "
                f"pitch={math.degrees(selected.predicted_pitch_rad):+.2f}"
            )
        else:
            summary = "NO VALID TARGET"
        cv2.putText(
            image,
            summary,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 60, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / f"{result.frame_index:06d}.jpg"), image)


def _parse_json_targets(data: object) -> tuple[dict[int, BBox], dict[int, float]]:
    if isinstance(data, dict) and "targets" in data:
        entries = data["targets"]
    elif isinstance(data, dict):
        entries = [{"id": key, "bbox": value} for key, value in data.items()]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Initial target JSON must be an object or list")
    if not isinstance(entries, list):
        raise ValueError("The JSON 'targets' field must be a list")

    boxes: dict[int, BBox] = {}
    biases: dict[int, float] = {}
    for default_id, entry in enumerate(entries):
        if isinstance(entry, dict):
            target_id = int(entry.get("id", default_id))
            bbox_raw = entry.get("bbox", entry.get("bbox_xyxy"))
            bias = float(entry.get("priority", entry.get("priority_bias", 0.0)))
        else:
            target_id = default_id
            bbox_raw = entry
            bias = 0.0
        if target_id in boxes:
            raise ValueError(f"Duplicate target ID: {target_id}")
        if not math.isfinite(bias):
            raise ValueError(f"Target {target_id} priority must be finite")
        boxes[target_id] = _parse_box(target_id, bbox_raw)
        biases[target_id] = bias
    return boxes, biases


def _parse_text_targets(lines: Iterable[str]) -> tuple[dict[int, BBox], dict[int, float]]:
    boxes: dict[int, BBox] = {}
    biases: dict[int, float] = {}
    next_id = 0
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        values = [float(value) for value in line.replace(",", " ").split()]
        if len(values) == 4:
            target_id = next_id
            box_values = values
            bias = 0.0
        elif len(values) in {5, 6}:
            target_id = int(values[0])
            box_values = values[1:5]
            bias = 0.0 if len(values) == 5 else values[5]
        else:
            raise ValueError(
                f"Line {line_number} must contain x1 y1 x2 y2 or id x1 y1 x2 y2 [priority]"
            )
        if target_id in boxes:
            raise ValueError(f"Duplicate target ID {target_id} on line {line_number}")
        boxes[target_id] = _parse_box(target_id, box_values)
        biases[target_id] = float(bias)
        next_id = max(next_id, target_id + 1)
    if not boxes:
        raise ValueError("Initial target file contains no boxes")
    return boxes, biases


def _parse_box(target_id: int, values: object) -> BBox:
    if values is None:
        raise ValueError(f"Target {target_id} has no bbox")
    bbox = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"Target {target_id} bbox must contain four finite numbers")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"Target {target_id} bbox must satisfy x2>x1 and y2>y1")
    return bbox


def _target_color(target_id: int) -> tuple[int, int, int]:
    hue = int((int(target_id) * 47) % 180)
    hsv = np.uint8([[[hue, 210, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


__all__ = [
    "load_initial_targets",
    "write_multi_target_csv",
    "write_multi_target_overlays",
]

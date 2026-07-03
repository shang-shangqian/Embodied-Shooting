from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate estimated yaw/pitch against ground-truth direction CSV."
    )
    parser.add_argument("--est_csv", required=True, type=Path)
    parser.add_argument("--gt_csv", required=True, type=Path)
    parser.add_argument(
        "--frame_col",
        type=str,
        default="frame_index",
        help="Frame id column shared by both CSV files.",
    )
    parser.add_argument("--est_yaw_col", type=str, default="yaw_rad")
    parser.add_argument("--est_pitch_col", type=str, default="pitch_rad")
    parser.add_argument("--gt_yaw_col", type=str, default="yaw_rad")
    parser.add_argument("--gt_pitch_col", type=str, default="pitch_rad")
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Set when yaw/pitch columns are in degrees instead of radians.",
    )
    return parser.parse_args()


def _read_direction_csv(
    path: Path,
    frame_col: str,
    yaw_col: str,
    pitch_col: str,
    degrees: bool,
) -> dict[int, tuple[float, float]]:
    rows: dict[int, tuple[float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(float(row[frame_col]))
                yaw = float(row[yaw_col])
                pitch = float(row[pitch_col])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(yaw) and math.isfinite(pitch)):
                continue
            if degrees:
                yaw = math.radians(yaw)
                pitch = math.radians(pitch)
            rows[frame] = (yaw, pitch)
    return rows


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def main() -> None:
    args = parse_args()
    est = _read_direction_csv(
        args.est_csv,
        args.frame_col,
        args.est_yaw_col,
        args.est_pitch_col,
        args.degrees,
    )
    gt = _read_direction_csv(
        args.gt_csv,
        args.frame_col,
        args.gt_yaw_col,
        args.gt_pitch_col,
        args.degrees,
    )

    frames = sorted(set(est.keys()) & set(gt.keys()))
    if not frames:
        raise RuntimeError("No overlapping valid frames between est_csv and gt_csv.")

    est_arr = np.asarray([est[frame] for frame in frames], dtype=np.float64)
    gt_arr = np.asarray([gt[frame] for frame in frames], dtype=np.float64)
    yaw_err = _angle_diff(est_arr[:, 0], gt_arr[:, 0])
    pitch_err = est_arr[:, 1] - gt_arr[:, 1]
    total_err = np.sqrt(yaw_err * yaw_err + pitch_err * pitch_err)

    def deg(value: float) -> float:
        return math.degrees(float(value))

    print(f"frames: {len(frames)}")
    print(f"yaw_mae_deg: {deg(np.mean(np.abs(yaw_err))):.4f}")
    print(f"pitch_mae_deg: {deg(np.mean(np.abs(pitch_err))):.4f}")
    print(f"angular_mae_deg: {deg(np.mean(total_err)):.4f}")
    print(f"yaw_rmse_deg: {deg(np.sqrt(np.mean(yaw_err * yaw_err))):.4f}")
    print(f"pitch_rmse_deg: {deg(np.sqrt(np.mean(pitch_err * pitch_err))):.4f}")
    print(f"angular_rmse_deg: {deg(np.sqrt(np.mean(total_err * total_err))):.4f}")


if __name__ == "__main__":
    main()
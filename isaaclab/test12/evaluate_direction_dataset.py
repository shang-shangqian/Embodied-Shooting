from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test10.evaluate_direction import _angle_diff, _read_direction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSTrack direction estimation on every complete sequence in a collected dataset."
    )
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("test12_outputs/direction_dataset_eval"))
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--ostrack_repo", required=True, type=Path)
    parser.add_argument("--ostrack_checkpoint", required=True, type=Path)
    parser.add_argument("--ostrack_config", type=str, default="vitb_256_mae_ce_32x4_ep300")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of complete scene sequences to evaluate. Multi-object sequences may create several runs.",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--target_color_name",
        nargs="+",
        default=None,
        help="Only evaluate targets whose color name matches these names, e.g. red. Omitted means all colors.",
    )
    parser.add_argument(
        "--target_object_id",
        type=int,
        default=None,
        help="Only evaluate targets whose object_id matches this id.",
    )
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument(
        "--num_sample_images",
        type=int,
        default=3,
        help="Number of boxed sample images to write per sampled target run. Set 0 to disable.",
    )
    parser.add_argument(
        "--max_sample_image_targets",
        type=int,
        default=5,
        help="Maximum target runs for sample image export. Set 0 or negative to export samples for all target runs.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run OSTrack even if an estimate CSV already exists.")
    return parser.parse_args()


def _read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "") or "<empty>")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _resolve_seq_dir(dataset_dir: Path, row: dict[str, str]) -> Path:
    seq_dir = Path(row.get("seq_dir", ""))
    if seq_dir.is_absolute():
        return seq_dir.resolve()
    if seq_dir.exists():
        return seq_dir.resolve()
    return (dataset_dir / "sequences" / seq_dir.name).resolve()


def _resolve_row_path(dataset_dir: Path, seq_dir: Path, value: str | None, default_name: str) -> Path:
    value_text = str(value or "").strip()
    if value_text:
        path = Path(value_text)
        if path.is_absolute():
            return path.resolve()
        if path.exists():
            return path.resolve()
        candidate = seq_dir / path.name
        if candidate.exists():
            return candidate.resolve()
        candidate = dataset_dir / path
        if candidate.exists():
            return candidate.resolve()
    return (seq_dir / default_name).resolve()


def _parse_optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_name(value: Any) -> str:
    text = str(value or "target").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    safe = "".join(chars).strip("_")
    return safe or "target"


def _target_suffix(item: dict[str, Any]) -> str:
    object_id = item.get("target_object_id")
    color_name = _safe_name(item.get("target_color_name", "target"))
    if object_id is None or str(object_id) == "":
        return color_name
    return f"obj_{int(object_id):02d}_{color_name}"


def _read_initial_box_file(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8").strip()
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError(f"{path} must contain exactly four numbers.")
    return values


def _write_initial_box_file(path: Path, box: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(f"{float(v):.6f}" for v in box) + "\n", encoding="utf-8")


def _read_initial_boxes_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_id = _parse_optional_int(row.get("object_id"))
            if object_id is None:
                continue
            try:
                box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "target_object_id": int(object_id),
                    "target_color_name": str(row.get("target_color_name", "")).strip(),
                    "target_color_rgb": str(row.get("target_color_rgb", "")).strip(),
                    "initial_box": box,
                }
            )
    return rows


def _matches_requested_target(spec: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.target_color_name is not None:
        requested = {str(name).strip().lower() for name in args.target_color_name}
        if str(spec.get("target_color_name", "")).strip().lower() not in requested:
            return False
    if args.target_object_id is not None:
        object_id = _parse_optional_int(spec.get("target_object_id"))
        if object_id != int(args.target_object_id):
            return False
    return True


def _target_specs_for_sequence(
    dataset_dir: Path,
    row: dict[str, str],
    seq_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    gt_csv = _resolve_row_path(dataset_dir, seq_dir, row.get("gt_direction_csv"), "ground_truth_direction.csv")
    initial_boxes_csv = _resolve_row_path(dataset_dir, seq_dir, row.get("initial_boxes_csv"), "initial_boxes.csv")
    is_multi_object = row.get("sequence_format") == "multi_object_scene" or initial_boxes_csv.exists()

    if is_multi_object:
        initial_boxes = _read_initial_boxes_csv(initial_boxes_csv)
        if not initial_boxes:
            raise RuntimeError(f"No target boxes found in {initial_boxes_csv}")
        specs: list[dict[str, Any]] = []
        for box_row in initial_boxes:
            spec = {
                "sequence_format": "multi_object_scene",
                "target_object_id": int(box_row["target_object_id"]),
                "target_color_name": str(box_row.get("target_color_name", "")),
                "target_color_rgb": str(box_row.get("target_color_rgb", "")),
                "initial_box": [float(v) for v in box_row["initial_box"]],
                "initial_box_file": None,
                "gt_csv": gt_csv,
                "objects_per_scene": row.get("objects_per_scene", ""),
            }
            if _matches_requested_target(spec, args):
                specs.append(spec)
        return specs

    object_id = _parse_optional_int(row.get("target_object_id"))
    spec = {
        "sequence_format": row.get("sequence_format", "single_object_sequence") or "single_object_sequence",
        "target_object_id": object_id,
        "target_color_name": row.get("target_color_name", ""),
        "target_color_rgb": row.get("target_color_rgb", ""),
        "initial_box": None,
        "initial_box_file": (seq_dir / "initial_box.txt").resolve(),
        "gt_csv": gt_csv,
        "objects_per_scene": row.get("objects_per_scene", ""),
    }
    if not _matches_requested_target(spec, args):
        return []
    return [spec]


def _filter_ground_truth_csv(src_csv: Path, out_csv: Path, item: dict[str, Any]) -> Path:
    with src_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_object_id = "object_id" in fieldnames
        has_color_name = "target_color_name" in fieldnames
        if not has_object_id and not has_color_name:
            return src_csv

        rows: list[dict[str, str]] = []
        target_object_id = _parse_optional_int(item.get("target_object_id"))
        target_color_name = str(item.get("target_color_name", "")).strip().lower()
        for row in reader:
            if has_object_id and target_object_id is not None:
                row_object_id = _parse_optional_int(row.get("object_id"))
                if row_object_id != target_object_id:
                    continue
            if has_color_name and target_color_name:
                if str(row.get("target_color_name", "")).strip().lower() != target_color_name:
                    continue
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No GT rows matched object_id={item.get('target_object_id')} "
            f"color={item.get('target_color_name')} in {src_csv}"
        )
    _write_csv(out_csv, rows)
    return out_csv


def _evaluate_pair(est_csv: Path, gt_csv: Path) -> dict[str, float]:
    est = _read_direction_csv(est_csv, "frame_index", "yaw_rad", "pitch_rad", False)
    gt = _read_direction_csv(gt_csv, "frame_index", "yaw_rad", "pitch_rad", False)
    frames = sorted(set(est.keys()) & set(gt.keys()))
    if not frames:
        raise RuntimeError(f"No overlapping valid frames: est={est_csv} gt={gt_csv}")

    est_arr = np.asarray([est[frame] for frame in frames], dtype=np.float64)
    gt_arr = np.asarray([gt[frame] for frame in frames], dtype=np.float64)
    yaw_err = _angle_diff(est_arr[:, 0], gt_arr[:, 0])
    pitch_err = est_arr[:, 1] - gt_arr[:, 1]
    total_err = np.sqrt(yaw_err * yaw_err + pitch_err * pitch_err)

    def deg(value: float) -> float:
        return math.degrees(float(value))

    return {
        "frames": float(len(frames)),
        "yaw_mae_deg": deg(np.mean(np.abs(yaw_err))),
        "pitch_mae_deg": deg(np.mean(np.abs(pitch_err))),
        "angular_mae_deg": deg(np.mean(total_err)),
        "yaw_rmse_deg": deg(np.sqrt(np.mean(yaw_err * yaw_err))),
        "pitch_rmse_deg": deg(np.sqrt(np.mean(pitch_err * pitch_err))),
        "angular_rmse_deg": deg(np.sqrt(np.mean(total_err * total_err))),
    }


def _read_timing_csv(est_csv: Path) -> dict[str, float]:
    times_s: list[float] = []
    with est_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                value = float(row.get("track_time_s", "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                times_s.append(value)
    if not times_s:
        return {
            "total_track_time_s": 0.0,
            "mean_track_time_ms": 0.0,
            "median_track_time_ms": 0.0,
            "p95_track_time_ms": 0.0,
            "track_fps": 0.0,
        }
    times = np.asarray(times_s, dtype=np.float64)
    total = float(times.sum())
    return {
        "total_track_time_s": total,
        "mean_track_time_ms": float(times.mean() * 1000.0),
        "median_track_time_ms": float(np.median(times) * 1000.0),
        "p95_track_time_ms": float(np.percentile(times, 95.0) * 1000.0),
        "track_fps": float(len(times) / total) if total > 0.0 else 0.0,
    }


def _run_estimate(
    args: argparse.Namespace,
    seq_dir: Path,
    initial_box_file: Path,
    est_csv: Path,
    overlay_dir: Path | None,
) -> None:
    standalone_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(standalone_root) if not old_pythonpath else f"{standalone_root}{os.pathsep}{old_pythonpath}"

    cmd = [
        sys.executable,
        "-m",
        "test12.estimate_video",
        "--rgb_dir",
        str(seq_dir / "rgb"),
        "--initial_box_file",
        str(initial_box_file),
        "--intrinsics_json",
        str(seq_dir / "intrinsics.json"),
        "--fps",
        str(float(args.fps)),
        "--out_csv",
        str(est_csv),
        "--ostrack_repo",
        str(args.ostrack_repo),
        "--ostrack_checkpoint",
        str(args.ostrack_checkpoint),
        "--ostrack_config",
        str(args.ostrack_config),
        "--device",
        str(args.device),
    ]
    if args.max_frames is not None:
        cmd.extend(["--max_frames", str(int(args.max_frames))])
    if overlay_dir is not None:
        cmd.extend(["--overlay_dir", str(overlay_dir)])

    subprocess.run(cmd, check=True, env=env)

def _load_rgb_paths(rgb_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path.resolve() for path in rgb_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _read_estimate_boxes(est_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with est_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_index = int(float(row["frame_index"]))
                bbox = [float(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in bbox):
                continue
            valid_text = str(row.get("valid_measurement", "1")).strip()
            try:
                valid = bool(int(float(valid_text)))
            except ValueError:
                valid = True
            if not valid:
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            rows.append({"frame_index": frame_index, "bbox": bbox})
    return rows


def _select_evenly_spaced_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    sample_count = min(int(count), len(rows))
    indices = np.linspace(0, len(rows) - 1, sample_count).round().astype(int).tolist()
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        selected.append(rows[index])
    return selected


def _write_sample_images(
    rgb_dir: Path,
    est_csv: Path,
    out_dir: Path,
    target_label: str,
    count: int,
) -> list[Path]:
    rgb_paths = _load_rgb_paths(rgb_dir)
    if not rgb_paths:
        print(f"[test12_dataset_eval] sample_images skipped: no RGB frames in {rgb_dir}")
        return []
    box_rows = _select_evenly_spaced_rows(_read_estimate_boxes(est_csv), count)
    if not box_rows:
        print(f"[test12_dataset_eval] sample_images skipped: no valid boxes in {est_csv}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        import cv2

        for item in box_rows:
            frame_index = int(item["frame_index"])
            if frame_index < 0 or frame_index >= len(rgb_paths):
                continue
            image = cv2.imread(str(rgb_paths[frame_index]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 80), 3)
            label = f"{target_label} f={frame_index}"
            cv2.putText(image, label, (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
            out_path = out_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), image)
            written.append(out_path)
        if written:
            return written
    except ImportError:
        pass

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[test12_dataset_eval] sample_images skipped: install opencv-python or pillow to draw boxes")
        return written

    for item in box_rows:
        frame_index = int(item["frame_index"])
        if frame_index < 0 or frame_index >= len(rgb_paths):
            continue
        try:
            image = Image.open(rgb_paths[frame_index]).convert("RGB")
        except OSError:
            continue
        width, height = image.size
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        draw = ImageDraw.Draw(image)
        for offset in range(3):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=(40, 255, 120))
        draw.text((max(0, x1), max(0, y1 - 16)), f"{target_label} f={frame_index}", fill=(40, 255, 120))
        out_path = out_dir / f"frame_{frame_index:06d}.jpg"
        image.save(out_path, quality=95)
        written.append(out_path)
    return written

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.ostrack_repo = args.ostrack_repo.resolve()
    args.ostrack_checkpoint = args.ostrack_checkpoint.resolve()
    config_path = Path(args.ostrack_config)
    if config_path.exists():
        args.ostrack_config = str(config_path.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_manifest(args.dataset_dir)
    print(
        "[test12_dataset_eval] manifest",
        f"rows={len(manifest_rows)}",
        f"status_counts={_count_by(manifest_rows, 'status')}",
        f"format_counts={_count_by(manifest_rows, 'sequence_format')}",
    )

    complete_rows = [row for row in manifest_rows if row.get("status") == "complete"]
    eval_items: list[dict[str, Any]] = []
    selected_sequence_count = 0
    for row in complete_rows:
        seq_dir = _resolve_seq_dir(args.dataset_dir, row)
        specs = _target_specs_for_sequence(args.dataset_dir, row, seq_dir, args)
        if not specs:
            continue
        selected_sequence_count += 1
        for spec in specs:
            eval_items.append({"row": row, "seq_dir": seq_dir, **spec})
        if args.max_sequences is not None and selected_sequence_count >= int(args.max_sequences):
            break

    print(
        "[test12_dataset_eval] selected",
        f"complete_sequences={len(complete_rows)}",
        f"selected_sequences={selected_sequence_count}",
        f"target_runs={len(eval_items)}",
        f"target_color_counts={_count_by(eval_items, 'target_color_name')}",
    )
    if args.target_color_name is not None:
        print("[test12_dataset_eval] color filter", f"target_color_name={args.target_color_name}")
    if args.target_object_id is not None:
        print("[test12_dataset_eval] object filter", f"target_object_id={int(args.target_object_id)}")
    if args.max_sequences is not None:
        print("[test12_dataset_eval] max_sequences", f"max_sequences={int(args.max_sequences)}")
    if not eval_items:
        raise RuntimeError(f"No matching complete targets found in {args.dataset_dir / 'manifest.csv'}")

    result_rows: list[dict[str, Any]] = []
    sample_image_targets_written = 0
    for index, item in enumerate(eval_items):
        seq_dir = Path(item["seq_dir"])
        seq_name = seq_dir.name
        suffix = _target_suffix(item)
        target_name = f"{seq_name}_{suffix}"
        est_csv = args.out_dir / "estimates" / f"{target_name}_est.csv"
        overlay_dir = args.out_dir / "overlays" / target_name if args.overlay else None
        gt_eval_csv = args.out_dir / "ground_truth" / f"{target_name}_gt.csv"

        initial_box_file = item.get("initial_box_file")
        if initial_box_file is None:
            initial_box_file = args.out_dir / "initial_boxes" / f"{target_name}.txt"
            _write_initial_box_file(initial_box_file, [float(v) for v in item["initial_box"]])
        else:
            initial_box_file = Path(initial_box_file)
            if not initial_box_file.exists():
                raise FileNotFoundError(f"initial_box.txt not found: {initial_box_file}")

        gt_csv = _filter_ground_truth_csv(Path(item["gt_csv"]), gt_eval_csv, item)

        print(
            f"[test12_dataset_eval] {index + 1}/{len(eval_items)} {seq_name}",
            f"object_id={item.get('target_object_id')}",
            f"color={item.get('target_color_name')}",
        )
        if args.force or not est_csv.exists():
            _run_estimate(
                args,
                seq_dir=seq_dir,
                initial_box_file=initial_box_file,
                est_csv=est_csv,
                overlay_dir=overlay_dir,
            )
        else:
            print(f"[test12_dataset_eval] reuse {est_csv}")

        sample_image_dir = ""
        if int(args.num_sample_images) > 0:
            target_limit = int(args.max_sample_image_targets)
            if target_limit <= 0 or sample_image_targets_written < target_limit:
                sample_dir = args.out_dir / "sample_images" / target_name
                sample_label = f"{item.get('target_color_name') or 'target'}:{item.get('target_object_id')}"
                sample_paths = _write_sample_images(
                    rgb_dir=seq_dir / "rgb",
                    est_csv=est_csv,
                    out_dir=sample_dir,
                    target_label=sample_label,
                    count=int(args.num_sample_images),
                )
                if sample_paths:
                    sample_image_targets_written += 1
                    sample_image_dir = str(sample_dir)
                    print(
                        "[test12_dataset_eval] sample_images",
                        f"target={target_name}",
                        f"count={len(sample_paths)}",
                        f"dir={sample_dir}",
                    )

        metrics = _evaluate_pair(est_csv, gt_csv)
        timing = _read_timing_csv(est_csv)
        result_rows.append(
            {
                "sequence": seq_name,
                "target_run": target_name,
                "seq_dir": str(seq_dir),
                "est_csv": str(est_csv),
                "gt_csv": str(gt_csv),
                "sample_image_dir": sample_image_dir,
                "sequence_format": str(item.get("sequence_format", "")),
                "target_color_name": str(item.get("target_color_name", "")),
                "target_color_rgb": str(item.get("target_color_rgb", "")),
                "target_object_id": "" if item.get("target_object_id") is None else str(item.get("target_object_id")),
                "objects_per_scene": str(item.get("objects_per_scene", "")),
                **{key: f"{value:.6f}" for key, value in metrics.items()},
                **{key: f"{value:.6f}" for key, value in timing.items()},
            }
        )

    total_frames = sum(float(row["frames"]) for row in result_rows)
    total_track_time_s = sum(float(row["total_track_time_s"]) for row in result_rows)
    summary = {
        "sequence": "ALL_WEIGHTED_BY_FRAMES",
        "target_run": "",
        "seq_dir": "",
        "est_csv": "",
        "gt_csv": "",
        "sample_image_dir": "",
        "sequence_format": "",
        "target_color_name": ",".join(args.target_color_name) if args.target_color_name else "ALL",
        "target_color_rgb": "",
        "target_object_id": str(args.target_object_id) if args.target_object_id is not None else "ALL",
        "objects_per_scene": "",
        "frames": f"{total_frames:.0f}",
        "total_track_time_s": f"{total_track_time_s:.6f}",
        "track_fps": f"{(total_frames / total_track_time_s) if total_track_time_s > 0.0 else 0.0:.6f}",
    }
    for key in (
        "yaw_mae_deg",
        "pitch_mae_deg",
        "angular_mae_deg",
        "yaw_rmse_deg",
        "pitch_rmse_deg",
        "angular_rmse_deg",
    ):
        weighted = sum(float(row[key]) * float(row["frames"]) for row in result_rows) / max(total_frames, 1.0)
        summary[key] = f"{weighted:.6f}"
    for key in (
        "mean_track_time_ms",
        "median_track_time_ms",
        "p95_track_time_ms",
    ):
        weighted = sum(float(row[key]) * float(row["frames"]) for row in result_rows) / max(total_frames, 1.0)
        summary[key] = f"{weighted:.6f}"

    result_rows.append(summary)
    summary_csv = args.out_dir / "summary.csv"
    _write_csv(summary_csv, result_rows)
    print(f"[test12_dataset_eval] wrote {summary_csv}")
    print(
        "[test12_dataset_eval] overall",
        f"target_runs={len(eval_items)}",
        f"frames={summary['frames']}",
        f"track_fps={summary['track_fps']}",
        f"mean_track_time_ms={summary['mean_track_time_ms']}",
        f"angular_mae_deg={summary['angular_mae_deg']}",
        f"angular_rmse_deg={summary['angular_rmse_deg']}",
    )


if __name__ == "__main__":
    main()
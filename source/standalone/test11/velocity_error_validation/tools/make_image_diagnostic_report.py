"""Build compact A/B/C/D contact sheets from saved diagnostic RGB frames."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance

try:
    import cv2
except ModuleNotFoundError:  # Contact sheets and pixel diffs remain available.
    cv2 = None  # type: ignore[assignment]


CONDITIONS = ("A", "B", "C", "D")
FRAME_PATTERN = re.compile(r"step_(\d+)_env_(\d+)\.png$")


def _global_flow(previous_rgb: np.ndarray, current_rgb: np.ndarray) -> dict[str, float | int] | None:
    """Reproduce the collector's unmasked LK background-flow diagnostic."""
    if cv2 is None:
        return None
    previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=160,
        qualityLevel=0.01,
        minDistance=8,
    )
    if points is None or len(points) < 8:
        return None
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if next_points is None or status is None:
        return None
    good = status.reshape(-1).astype(bool)
    if int(good.sum()) < 8:
        return None
    displacement = next_points.reshape(-1, 2)[good] - points.reshape(-1, 2)[good]
    magnitudes = np.linalg.norm(displacement, axis=1)
    median = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - median)))
    inliers = magnitudes <= median + max(3.0 * 1.4826 * mad, 0.5)
    robust_median = float(np.median(magnitudes[inliers])) if np.any(inliers) else median
    return {
        "flow_px": robust_median,
        "tracked_points": int(good.sum()),
        "inlier_points": int(inliers.sum()),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--thumbnail-width", type=int, default=256)
    return parser.parse_args()


def _labelled_thumbnail(
    path: Path,
    label: str,
    *,
    width: int,
    difference_from: Path | None = None,
) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if difference_from is not None:
        previous = Image.open(difference_from).convert("RGB")
        image = ImageEnhance.Contrast(ImageChops.difference(previous, image)).enhance(4.0)
    height = max(1, round(image.height * width / image.width))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (width, height + 24), "white")
    tile.paste(image, (0, 24))
    ImageDraw.Draw(tile).text((5, 5), label, fill="black")
    return tile


def _save_grid(rows: list[list[Image.Image]], path: Path) -> None:
    if not rows or not rows[0]:
        return
    cell_width = max(image.width for row in rows for image in row)
    cell_height = max(image.height for row in rows for image in row)
    columns = max(len(row) for row in rows)
    canvas = Image.new(
        "RGB", (columns * cell_width, len(rows) * cell_height), "#dddddd"
    )
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            canvas.paste(image, (column_index * cell_width, row_index * cell_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = _args()
    root = args.input_root.expanduser().resolve()
    report_dir = root / "diagnostic_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[tuple[str, str, int], dict[int, Path]] = defaultdict(dict)
    for path in sorted(root.glob("seed_*/[ABCD]/rgb_frames/step_*_env_*.png")):
        match = FRAME_PATTERN.match(path.name)
        if match is None:
            continue
        step, env_id = (int(value) for value in match.groups())
        seed = path.parents[2].name
        condition = path.parents[1].name
        frames[(seed, condition, env_id)][step] = path
    if not frames:
        raise RuntimeError(f"no diagnostic RGB frames found under {root}")

    summary: dict[str, object] = {
        "input_root": str(root),
        "opencv_available": cv2 is not None,
        "groups": {},
    }
    seeds = sorted({key[0] for key in frames})
    env_ids = sorted({key[2] for key in frames})
    for seed in seeds:
        for env_id in env_ids:
            available = sorted(
                set.intersection(
                    *(set(frames.get((seed, condition, env_id), {})) for condition in CONDITIONS)
                )
            )
            if not available:
                continue
            desired = [0, 1, 2, 3, 4, 5, 10, 20, 40, available[-1]]
            selected = list(dict.fromkeys(step for step in desired if step in available))
            raw_rows: list[list[Image.Image]] = []
            difference_rows: list[list[Image.Image]] = []
            for condition in CONDITIONS:
                condition_frames = frames[(seed, condition, env_id)]
                raw_rows.append(
                    [
                        _labelled_thumbnail(
                            condition_frames[step],
                            f"{condition} step {step}",
                            width=args.thumbnail_width,
                        )
                        for step in selected
                    ]
                )
                transitions = [
                    (first, second)
                    for first, second in zip(available, available[1:])
                    if second <= 5
                ]
                difference_rows.append(
                    [
                        _labelled_thumbnail(
                            condition_frames[second],
                            f"{condition} {first}->{second} |diff| x4",
                            width=args.thumbnail_width,
                            difference_from=condition_frames[first],
                        )
                        for first, second in transitions
                    ]
                )
            _save_grid(
                raw_rows,
                report_dir / f"{seed}_env_{env_id:03d}_ABCD_frames.png",
            )
            _save_grid(
                difference_rows,
                report_dir / f"{seed}_env_{env_id:03d}_startup_differences.png",
            )

    group_summary = summary["groups"]
    assert isinstance(group_summary, dict)
    for key, step_paths in sorted(frames.items()):
        seed, condition, env_id = key
        steps = sorted(step_paths)
        consecutive_mad: list[float] = []
        consecutive_flow: list[dict[str, float | int] | None] = []
        for first, second in zip(steps, steps[1:]):
            if second != first + 1:
                continue
            previous_uint8 = np.asarray(Image.open(step_paths[first]).convert("RGB"))
            current_uint8 = np.asarray(Image.open(step_paths[second]).convert("RGB"))
            previous = previous_uint8.astype(np.float32)
            current = current_uint8.astype(np.float32)
            consecutive_mad.append(float(np.mean(np.abs(current - previous))))
            consecutive_flow.append(_global_flow(previous_uint8, current_uint8))
        sample = Image.open(step_paths[steps[0]])
        finite_flow = [
            float(item["flow_px"])
            for item in consecutive_flow
            if item is not None and math.isfinite(float(item["flow_px"]))
        ]
        group_summary[f"{seed}/{condition}/env_{env_id:03d}"] = {
            "image_count": len(steps),
            "first_step": steps[0],
            "last_step": steps[-1],
            "resolution": list(sample.size),
            "consecutive_pixel_mad_mean": (
                float(np.mean(consecutive_mad)) if consecutive_mad else None
            ),
            "consecutive_pixel_mad_p95": (
                float(np.quantile(consecutive_mad, 0.95)) if consecutive_mad else None
            ),
            "startup_pixel_mad": consecutive_mad[:5],
            "finite_global_flow_count": len(finite_flow),
            "global_flow_median_px": (
                float(np.median(finite_flow)) if finite_flow else None
            ),
            "global_flow_p95_px": (
                float(np.quantile(finite_flow, 0.95)) if finite_flow else None
            ),
            "startup_global_flow": consecutive_flow[:5],
        }

    cross_condition: dict[str, object] = {}
    for seed in seeds:
        for env_id in env_ids:
            for first_condition, second_condition in (("A", "B"), ("C", "D")):
                first = frames.get((seed, first_condition, env_id), {})
                second = frames.get((seed, second_condition, env_id), {})
                shared_steps = sorted(set(first) & set(second))
                differences = []
                for step in shared_steps:
                    first_image = np.asarray(Image.open(first[step]).convert("RGB"), dtype=np.float32)
                    second_image = np.asarray(Image.open(second[step]).convert("RGB"), dtype=np.float32)
                    differences.append(float(np.mean(np.abs(first_image - second_image))))
                cross_condition[
                    f"{seed}/{first_condition}-{second_condition}/env_{env_id:03d}"
                ] = {
                    "shared_steps": len(shared_steps),
                    "pixel_mad_mean": float(np.mean(differences)) if differences else None,
                    "pixel_mad_max": float(np.max(differences)) if differences else None,
                }
    summary["cross_condition"] = cross_condition
    (report_dir / "image_diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[image-diagnostic-report] wrote {report_dir}")


if __name__ == "__main__":
    main()

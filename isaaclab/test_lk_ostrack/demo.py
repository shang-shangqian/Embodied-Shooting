"""Generate a deterministic two-target sequence and run the real CLI pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/lk_ostrack_demo"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--ostrack_checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 2 <= args.frames <= 120:
        parser.error("--frames must be between 2 and 120")
    root = args.out_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f"Output directory is not empty: {root}; choose a new --out_dir")
    rgb = root / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(65)
    patches = [rng.integers(25, 255, (48, 56, 3), dtype=np.uint8) for _ in range(2)]
    for i in range(args.frames):
        image = np.full((360, 640, 3), 16, dtype=np.uint8)
        x1, x2 = 50 + 2 * i, 490 - i
        image[80:128, x1:x1 + 56] = patches[0]
        image[230:278, x2:x2 + 56] = patches[1]
        if not cv2.imwrite(str(rgb / f"{i:06d}.png"), image):
            raise OSError("Could not write demo image")
    (root / "initial_targets.json").write_text(json.dumps({"targets": [
        {"id": 0, "bbox": [50, 80, 106, 128]},
        {"id": 1, "bbox": [490, 230, 546, 278]},
    ]}, indent=2))
    (root / "intrinsics.json").write_text(json.dumps({
        "fx": 600, "fy": 600, "cx": 320, "cy": 180, "width": 640, "height": 360,
    }, indent=2))
    command = [sys.executable, "-m", "test_lk_ostrack.estimate_video",
               "--rgb_dir", str(rgb), "--initial_boxes_file", str(root / "initial_targets.json"),
               "--intrinsics_json", str(root / "intrinsics.json"), "--fps", "30",
               "--out_csv", str(root / "tracks.csv"), "--overlay_dir", str(root / "overlay")]
    if args.ostrack_checkpoint:
        command += ["--ostrack_checkpoint", str(args.ostrack_checkpoint.expanduser().resolve()),
                    "--device", args.device]
    else:
        command += ["--disable_fine_tracking"]
    subprocess.run(command, check=True)
    print(f"Demo complete: {root / 'tracks.csv'}\nOverlays: {root / 'overlay'}")


if __name__ == "__main__":
    main()

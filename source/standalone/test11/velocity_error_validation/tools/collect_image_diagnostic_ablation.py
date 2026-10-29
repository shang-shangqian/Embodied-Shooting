"""Collect a small A/B/C/D image-diagnostic ablation data set.

This launcher deliberately does not run the formal paired quality gate.  Its
purpose is to retain raw RGB frames so rendering, startup transients, and
camera-motion effects can be inspected before another full collection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


CONDITIONS = ("A", "B", "C", "D")
CONDITION_DESCRIPTIONS = {
    "A": "GT direction + fixed-world proxy camera",
    "B": "LK/OSTrack direction + fixed-world proxy camera",
    "C": "GT direction + torso-mounted proxy camera",
    "D": "LK/OSTrack direction + torso-mounted proxy camera",
}


def _args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repo_root
            / "source/standalone/test11/velocity_error_validation/outputs"
            / "image_diagnostic_small"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[72])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--collector",
        type=Path,
        default=repo_root / "source/standalone/test11/collect_vision_data.py",
    )
    parser.add_argument(
        "--locomotion-policy",
        type=Path,
        default=(
            repo_root
            / "logs/rsl_rl/deeprobotics_lite3_rtx_flat"
            / "2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt"
        ),
    )
    parser.add_argument(
        "--ostrack-repo",
        type=Path,
        default=(
            repo_root
            / "source/standalone/test_lk_ostrack/ostrack/OSTrack-main"
        ),
    )
    parser.add_argument(
        "--ostrack-checkpoint",
        type=Path,
        default=(
            repo_root
            / "source/standalone/test_lk_ostrack/ostrack/OSTrack-main"
            / "output/checkpoints/OSTrack_ep0300.pth.tar"
        ),
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow a non-empty output root (individual files may be overwritten).",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Do not build contact sheets after collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.num_envs < 1 or args.num_steps < 2 or not args.seeds:
        raise ValueError("num-envs >= 1, num-steps >= 2, and at least one seed are required")

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.allow_existing:
        raise FileExistsError(
            f"refusing non-empty diagnostic output root: {output_root}; "
            "choose another path or pass --allow-existing"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    required_paths = {
        "python": args.python,
        "collector": args.collector,
        "locomotion policy": args.locomotion_policy,
        "OSTrack repository": args.ostrack_repo,
        "OSTrack checkpoint": args.ostrack_checkpoint,
    }
    for label, path in required_paths.items():
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    manifest_path = output_root / "image_diagnostic_manifest.json"
    manifest: dict[str, object] = {
        "schema": "velocity_error_image_diagnostic_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conditions": CONDITION_DESCRIPTIONS,
        "seeds": list(args.seeds),
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "camera_prerender_frames": 5,
        "jobs": [],
    }

    for seed in args.seeds:
        pair_id = f"image_debug_seed_{seed:06d}"
        for condition in CONDITIONS:
            condition_dir = output_root / f"seed_{seed:06d}" / condition
            rgb_dir = condition_dir / "rgb_frames"
            command = [
                str(args.python.expanduser().resolve()),
                str(args.collector.expanduser().resolve()),
                "--headless",
                "--save_episode_stream",
                "--split",
                "test",
                "--scenario",
                "sd",
                "--degradation_profile",
                "clean",
                "--num_envs",
                str(args.num_envs),
                "--num_steps",
                str(args.num_steps),
                "--device",
                str(args.device),
                "--locomotion_policy",
                str(args.locomotion_policy.expanduser().resolve()),
                "--ostrack_repo",
                str(args.ostrack_repo.expanduser().resolve()),
                "--ostrack_checkpoint",
                str(args.ostrack_checkpoint.expanduser().resolve()),
                "--seed",
                str(seed),
                "--out_dir",
                str(condition_dir),
                "--validation_condition",
                condition,
                "--pair_id",
                pair_id,
                "--trajectory_seed",
                str(seed),
                "--collection_run_id",
                f"{pair_id}_{condition}",
                "--debug_rgb_dir",
                str(rgb_dir),
                "--progress_every",
                "10",
                "--camera_prerender_frames",
                "5",
            ]
            job = {
                "seed": int(seed),
                "condition": condition,
                "description": CONDITION_DESCRIPTIONS[condition],
                "output_dir": str(condition_dir),
                "rgb_dir": str(rgb_dir),
                "command": command,
                "status": "running",
            }
            jobs = manifest["jobs"]
            assert isinstance(jobs, list)
            jobs.append(job)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[image-diagnostic] seed={seed} condition={condition}", flush=True)
            subprocess.run(command, cwd=Path(__file__).resolve().parents[5], check=True)
            image_count = len(list(rgb_dir.glob("step_*_env_*.png")))
            expected_images = int(args.num_envs) * int(args.num_steps)
            if image_count != expected_images:
                raise RuntimeError(
                    f"condition {condition} saved {image_count} RGB frames; "
                    f"expected {expected_images}"
                )
            job["image_count"] = image_count
            job["status"] = "complete"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    if not args.skip_report:
        report_script = Path(__file__).with_name("make_image_diagnostic_report.py")
        subprocess.run(
            [str(args.python.expanduser().resolve()), str(report_script), str(output_root)],
            cwd=Path(__file__).resolve().parents[5],
            check=True,
        )
    print(f"[image-diagnostic] complete: {output_root}")


if __name__ == "__main__":
    main()

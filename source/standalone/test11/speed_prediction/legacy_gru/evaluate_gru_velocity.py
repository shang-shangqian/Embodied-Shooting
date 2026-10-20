"""Evaluate a TAQ-v2-trained ordinary GRU on held-out windows or full episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.legacy_gru"

from .online_gru_velocity import load_legacy_checkpoint
from .taq_dataset import chunk_identity, collate_velocity_windows, discover_chunks
from .train_gru_velocity import WeightedSmoothL1Loss, build_concat_dataset, evaluate
from .train_gru_velocity_episodes import (
    EpisodeDataset, collate_episodes, discover_episodes, run_epoch,
)


def manifest_files(path: Path, split: str) -> list[Path]:
    """Honor the saved split and reject cross-split path or episode leakage."""
    payload = json.loads(path.read_text())
    groups = payload["files"]
    if split not in groups or not groups[split]:
        raise ValueError(f"Manifest contains no {split!r} files")
    owners: dict[str, str] = {}
    seen_paths: set[Path] = set()
    selected = []
    for name in ("train", "val", "test"):
        for raw in groups.get(name, []):
            file = Path(raw).expanduser()
            file = (file if file.is_absolute() else path.parent / file).resolve()
            _, _, uid = chunk_identity(file)
            if file in seen_paths or (uid in owners and owners[uid] != name):
                raise ValueError(f"Duplicate file or cross-split episode: {file}")
            seen_paths.add(file)
            owners[uid] = name
            if name == split:
                selected.append(file)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--manifest", type=Path, help="Saved data_manifest.json; defaults to its test split")
    data.add_argument("--data_dirs", type=Path, nargs="+", help="Directories containing only held-out data")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--context", choices=("full", "window"), default="full")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_json", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers nonnegative")
    if args.manifest is None and args.split != "test":
        parser.error("--split applies only to --manifest")
    return args


def main() -> None:
    args = parse_args()
    model, checkpoint = load_legacy_checkpoint(args.checkpoint.expanduser())
    smooth_window = int(checkpoint["preprocessing"]["pos_smooth_window"])
    if args.manifest is not None:
        files = manifest_files(args.manifest.expanduser().resolve(), args.split)
    else:
        discover = discover_episodes if args.context == "full" else discover_chunks
        files = discover(args.data_dirs)
    if args.context == "full" and any(path.suffix != ".taqstream" for path in files):
        raise ValueError("Full context needs complete .taqstream files; use --context window for chunks")
    if args.context == "window" and any(path.suffix == ".taqstream" for path in files):
        raise ValueError("Window context requires chunk files, not complete streams")
    requested = args.device
    device = torch.device("cpu" if requested.startswith("cuda") and not torch.cuda.is_available() else requested)
    model.to(device).eval()
    training = checkpoint.get("train_args", {})
    loss = WeightedSmoothL1Loss(
        weights=tuple(float(training.get("loss_w" + axis, 1.0)) for axis in "xyz"),
        beta=float(training.get("smooth_l1_beta", 0.05)),
    ).to(device)
    if args.context == "full":
        dataset = EpisodeDataset(files, smooth_window)
        collate = collate_episodes
    else:
        dataset = build_concat_dataset(files, smooth_window)
        collate = collate_velocity_windows
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    metrics = (run_epoch if args.context == "full" else evaluate)(model, loader, device, loss)
    result = dict(checkpoint=str(args.checkpoint.expanduser().resolve()), context=args.context,
                  split=args.split if args.manifest else "explicit_held_out_directories",
                  file_count=len(files), **metrics)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")


if __name__ == "__main__":
    main()

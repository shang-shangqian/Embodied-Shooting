from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_vision1 import VisionChunkDataset
from vision_model1 import build_vision_model


def _resolve_checkpoint(model_path_arg: str, which: str) -> Path:
    model_path = Path(model_path_arg).expanduser()
    if model_path.is_dir():
        filename = "vision_best.pth" if which == "best" else "vision_last.pth"
        direct_path = model_path / filename
        if direct_path.is_file():
            model_path = direct_path
        else:
            candidates = [p for p in model_path.glob(f"*/{filename}") if p.is_file()]
            if len(candidates) > 0:
                model_path = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                model_path = direct_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    return model_path


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Be tolerant of checkpoints saved from DataParallel.
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate test9 direction-only vision model.")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint file or run directory.")
    parser.add_argument(
        "--which",
        type=str,
        default="best",
        choices=["best", "last"],
        help="When model_path is a directory, choose vision_best.pth or vision_last.pth.",
    )
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="+",
        default=None,
        help="One or more dataset directories containing vision_chunk_*.pt.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Legacy single dataset directory argument. Prefer --dataset_dirs.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dirs = args.dataset_dirs
    if dataset_dirs is None:
        if args.dataset_dir is None:
            raise ValueError("Pass --dataset_dirs <dir...> or --dataset_dir <dir>.")
        dataset_dirs = [args.dataset_dir]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] device={device}")
    print(f"[test] dataset_dirs={dataset_dirs}")

    test_dataset = VisionChunkDataset(
        root_dirs=dataset_dirs,
        normalize_images=True,
        max_files_per_dir=args.max_files,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    checkpoint_path = _resolve_checkpoint(args.model_path, args.which)
    print(f"[test] loading checkpoint: {checkpoint_path}")

    model = build_vision_model(use_proprio=True, proprio_dim=1).to(device)
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.eval()

    total_samples = 0
    loss_sum = 0.0
    yaw_error_rad_sum = 0.0
    pitch_error_rad_sum = 0.0
    yaw_sq_error_sum = 0.0
    pitch_sq_error_sum = 0.0
    print_samples: list[dict[str, float]] = []

    print("[test] starting evaluation...")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["images"].to(device, non_blocking=True)
            gt_direction = batch["direction"].to(device, non_blocking=True)
            dog_moving_flag = batch["dog_moving_flag"].to(device, non_blocking=True)

            out = model(images, proprio=dog_moving_flag)
            pred_direction = out.direction

            batch_samples = int(images.shape[0])
            total_samples += batch_samples

            loss = F.smooth_l1_loss(pred_direction, gt_direction, reduction="sum")
            loss_sum += float(loss.item())

            yaw_diff = pred_direction[:, 0] - gt_direction[:, 0]
            yaw_err = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
            pitch_err = torch.abs(pred_direction[:, 1] - gt_direction[:, 1])

            yaw_error_rad_sum += float(yaw_err.sum().item())
            pitch_error_rad_sum += float(pitch_err.sum().item())
            yaw_sq_error_sum += float((yaw_err * yaw_err).sum().item())
            pitch_sq_error_sum += float((pitch_err * pitch_err).sum().item())

            if batch_idx == 0 and batch_samples > 0:
                sample_indices = random.sample(range(batch_samples), min(5, batch_samples))
                for i in sample_indices:
                    print_samples.append(
                        {
                            "flag": float(dog_moving_flag[i, 0].item()),
                            "gt_yaw": float(gt_direction[i, 0].item() * (180.0 / math.pi)),
                            "pred_yaw": float(pred_direction[i, 0].item() * (180.0 / math.pi)),
                            "gt_pitch": float(gt_direction[i, 1].item() * (180.0 / math.pi)),
                            "pred_pitch": float(pred_direction[i, 1].item() * (180.0 / math.pi)),
                        }
                    )

    if total_samples == 0:
        raise RuntimeError("No samples were evaluated.")

    rad_to_deg = 180.0 / math.pi
    avg_loss = loss_sum / total_samples
    avg_yaw_err_deg = (yaw_error_rad_sum / total_samples) * rad_to_deg
    avg_pitch_err_deg = (pitch_error_rad_sum / total_samples) * rad_to_deg
    rmse_yaw_deg = math.sqrt(yaw_sq_error_sum / total_samples) * rad_to_deg
    rmse_pitch_deg = math.sqrt(pitch_sq_error_sum / total_samples) * rad_to_deg

    print("\n" + "=" * 60)
    print("Direction-only evaluation")
    print("=" * 60)
    print(f"samples:        {total_samples}")
    print(f"smooth_l1_loss: {avg_loss:.6f}")
    print(f"yaw MAE:        {avg_yaw_err_deg:.3f} deg")
    print(f"pitch MAE:      {avg_pitch_err_deg:.3f} deg")
    print(f"yaw RMSE:       {rmse_yaw_deg:.3f} deg")
    print(f"pitch RMSE:     {rmse_pitch_deg:.3f} deg")
    print("=" * 60)

    print("\nSample predictions:")
    for idx, sample in enumerate(print_samples, start=1):
        yaw_abs = abs(sample["gt_yaw"] - sample["pred_yaw"])
        pitch_abs = abs(sample["gt_pitch"] - sample["pred_pitch"])
        print(
            f"{idx:02d} flag={sample['flag']:.0f} | "
            f"yaw gt={sample['gt_yaw']:7.3f}, pred={sample['pred_yaw']:7.3f}, err={yaw_abs:6.3f} deg | "
            f"pitch gt={sample['gt_pitch']:7.3f}, pred={sample['pred_pitch']:7.3f}, err={pitch_abs:6.3f} deg"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vision_model1 import build_vision_model, compute_vision_losses


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VisionChunkDataset(Dataset):
    """Read vision dataset chunks for direction-only training.

    Supported image formats:
    - images: [N, 3, H, W]
    - images: [N, T, 3, H, W]

    Required labels used by this script:
    - direction: [N, 2]
    - dog_moving_flag: [N, 1]

    Optional fields can still exist in the chunk, but are ignored here.
    """

    def __init__(
        self,
        root_dirs: str | Path | list[str] | list[Path],
        normalize_images: bool = True,
        max_files_per_dir: int | None = None,
    ):
        if isinstance(root_dirs, (str, Path)):
            self.root_dirs = [Path(root_dirs).expanduser()]
        else:
            self.root_dirs = [Path(p).expanduser() for p in root_dirs]

        for root_dir in self.root_dirs:
            if not root_dir.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        self.files: list[Path] = []
        for root_dir in self.root_dirs:
            files = sorted(root_dir.glob("vision_chunk_*.pt"))
            if len(files) == 0:
                raise FileNotFoundError(f"No vision_chunk_*.pt found under: {root_dir}")
            if max_files_per_dir is not None:
                files = files[:max_files_per_dir]
            self.files.extend(files)

        if len(self.files) == 0:
            raise FileNotFoundError("No dataset files found from the provided root_dirs.")

        self.normalize_images = normalize_images
        self.samples: list[tuple[int, int]] = []
        self.file_lengths: list[int] = []
        self._cache: dict[int, dict] = {}

        for file_idx, file_path in enumerate(self.files):
            payload = torch.load(file_path, map_location="cpu")

            if "images" not in payload:
                raise KeyError(f"{file_path} missing required field: images")
            if "direction" not in payload:
                raise KeyError(f"{file_path} missing required field: direction")
            if "dog_moving_flag" not in payload:
                raise KeyError(f"{file_path} missing required field: dog_moving_flag")

            images = payload["images"]
            direction = payload["direction"]
            dog_moving_flag = payload["dog_moving_flag"]

            if images.ndim not in (4, 5):
                raise ValueError(f"{file_path} images shape invalid: {tuple(images.shape)}")
            if direction.ndim != 2 or direction.shape[-1] != 2:
                raise ValueError(f"{file_path} direction shape invalid: {tuple(direction.shape)}")
            if dog_moving_flag.ndim != 2 or dog_moving_flag.shape[-1] != 1:
                raise ValueError(f"{file_path} dog_moving_flag shape invalid: {tuple(dog_moving_flag.shape)}")

            n = images.shape[0]
            if direction.shape[0] != n or dog_moving_flag.shape[0] != n:
                raise ValueError(f"{file_path} sample count mismatch")

            self.file_lengths.append(n)
            for sample_idx in range(n):
                self.samples.append((file_idx, sample_idx))

        print(f"[dataset] found {len(self.files)} chunk files from {len(self.root_dirs)} directories")
        print(f"[dataset] total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_file(self, file_idx: int) -> dict:
        if file_idx not in self._cache:
            self._cache.clear()
            self._cache[file_idx] = torch.load(self.files[file_idx], map_location="cpu")
        return self._cache[file_idx]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, sample_idx = self.samples[index]
        payload = self._load_file(file_idx)

        images = payload["images"][sample_idx].float()                     # [3,H,W] or [T,3,H,W]
        direction = payload["direction"][sample_idx].float()               # [2]
        dog_moving_flag = payload["dog_moving_flag"][sample_idx].float()   # [1]

        if self.normalize_images and images.numel() > 0 and images.max() > 1.5:
            images = images / 255.0

        return {
            "images": images,
            "direction": direction,
            "dog_moving_flag": dog_moving_flag,
        }


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def split_indices_by_chunk(samples: list[tuple[int, int]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """按 chunk(文件) 划分训练集和验证集，防止数据泄漏。"""
    unique_chunks = list(set([file_idx for file_idx, _ in samples]))
    rng = random.Random(seed)
    rng.shuffle(unique_chunks)

    val_chunk_count = max(1, int(len(unique_chunks) * val_ratio))
    val_chunks = set(unique_chunks[:val_chunk_count])

    train_indices = []
    val_indices = []
    for idx, (file_idx, _) in enumerate(samples):
        if file_idx in val_chunks:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    return train_indices, val_indices


class SubsetChunkSampler(torch.utils.data.Sampler):
    """按 chunk 连续读取，减少频繁切换文件。"""

    def __init__(self, subset: torch.utils.data.Subset, shuffle: bool = True):
        self.subset = subset
        self.shuffle = shuffle

        dataset = cast(VisionChunkDataset, self.subset.dataset)
        self.chunk_to_subset_idx: dict[int, list[int]] = {}
        for subset_idx, dataset_idx in enumerate(self.subset.indices):
            file_idx, _ = dataset.samples[dataset_idx]
            self.chunk_to_subset_idx.setdefault(file_idx, []).append(subset_idx)

    def __iter__(self):
        chunk_order = list(self.chunk_to_subset_idx.keys())
        if self.shuffle:
            random.shuffle(chunk_order)

        final_indices: list[int] = []
        for chunk_idx in chunk_order:
            indices = self.chunk_to_subset_idx[chunk_idx].copy()
            if self.shuffle:
                random.shuffle(indices)
            final_indices.extend(indices)
        return iter(final_indices)

    def __len__(self) -> int:
        return len(self.subset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()

    loss_meter = RunningAverage()
    dir_meter = RunningAverage()
    yaw_mae_meter = RunningAverage()
    pitch_mae_meter = RunningAverage()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            gt_direction = batch["direction"].to(device, non_blocking=True)
            dog_moving_flag = batch["dog_moving_flag"].to(device, non_blocking=True)

            out = model(images, proprio=dog_moving_flag)

            dummy_gt_velocity = torch.zeros((gt_direction.shape[0], 3), device=device, dtype=gt_direction.dtype)
            loss_dict = compute_vision_losses(
                pred_direction=out.direction,
                pred_velocity=out.velocity,
                gt_direction=gt_direction,
                gt_velocity=dummy_gt_velocity,
                direction_weight=1.0,
                velocity_weight=0.0,
            )

            yaw_diff = out.direction[:, 0] - gt_direction[:, 0]
            yaw_err = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
            pitch_err = torch.abs(out.direction[:, 1] - gt_direction[:, 1])

            batch_size = images.shape[0]
            loss_meter.update(float(loss_dict["loss"].item()), batch_size)
            dir_meter.update(float(loss_dict["direction_loss"].item()), batch_size)
            yaw_mae_meter.update(float(yaw_err.mean().item()), batch_size)
            pitch_mae_meter.update(float(pitch_err.mean().item()), batch_size)

    return {
        "loss": loss_meter.avg,
        "direction_loss": dir_meter.avg,
        "yaw_mae_rad": yaw_mae_meter.avg,
        "pitch_mae_rad": pitch_mae_meter.avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Train vision model for direction only.")
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset directories.",
    )
    parser.add_argument("--out_dir", type=str, default="vision_runs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_out_dir = Path(args.out_dir).expanduser()
    timestamp = datetime.now().strftime("%m-%d-%H-%M")
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = VisionChunkDataset(
        root_dirs=args.dataset_dirs,
        normalize_images=True,
        max_files_per_dir=args.max_files,
    )

    train_indices, val_indices = split_indices_by_chunk(dataset.samples, args.val_ratio, args.seed)
    if len(train_indices) == 0:
        raise ValueError("No training samples after split. Reduce --val_ratio or collect more data.")
    if len(val_indices) == 0:
        print("[train] WARNING: validation set is empty. Falling back to using training split as validation.")
        val_indices = train_indices[:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_sampler = SubsetChunkSampler(train_subset, shuffle=True)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_sampler = SubsetChunkSampler(val_subset, shuffle=False)
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_vision_model(use_proprio=True, proprio_dim=1)
    if torch.cuda.device_count() > 1:
        print(f"[train] detected {torch.cuda.device_count()} GPUs, enabling nn.DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_path = out_dir / "vision_best.pth"
    last_path = out_dir / "vision_last.pth"

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    print(f"[train] device={device}")
    print(f"[train] train_samples={len(train_subset)} val_samples={len(val_subset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_meter = RunningAverage()
        train_dir_meter = RunningAverage()
        train_yaw_mae_meter = RunningAverage()
        train_pitch_mae_meter = RunningAverage()

        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            gt_direction = batch["direction"].to(device, non_blocking=True)
            dog_moving_flag = batch["dog_moving_flag"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = model(images, proprio=dog_moving_flag)
            dummy_gt_velocity = torch.zeros((gt_direction.shape[0], 3), device=device, dtype=gt_direction.dtype)
            loss_dict = compute_vision_losses(
                pred_direction=out.direction,
                pred_velocity=out.velocity,
                gt_direction=gt_direction,
                gt_velocity=dummy_gt_velocity,
                direction_weight=1.0,
                velocity_weight=0.0,
            )

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            yaw_diff = out.direction[:, 0] - gt_direction[:, 0]
            yaw_err = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
            pitch_err = torch.abs(out.direction[:, 1] - gt_direction[:, 1])

            batch_size = images.shape[0]
            train_loss_meter.update(float(loss.item()), batch_size)
            train_dir_meter.update(float(loss_dict["direction_loss"].item()), batch_size)
            train_yaw_mae_meter.update(float(yaw_err.mean().item()), batch_size)
            train_pitch_mae_meter.update(float(pitch_err.mean().item()), batch_size)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                print(f"    [Epoch {epoch} | Batch {batch_idx + 1}/{num_batches}] Loss: {loss.item():.4f}")

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)

        rad_to_deg = 180.0 / torch.pi
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss_meter.avg:.6f} "
            f"train_dir={train_dir_meter.avg:.6f} "
            f"train_yaw={train_yaw_mae_meter.avg * float(rad_to_deg):.3f}deg "
            f"train_pitch={train_pitch_mae_meter.avg * float(rad_to_deg):.3f}deg "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_dir={val_metrics['direction_loss']:.6f} "
            f"val_yaw={val_metrics['yaw_mae_rad'] * float(rad_to_deg):.3f}deg "
            f"val_pitch={val_metrics['pitch_mae_rad'] * float(rad_to_deg):.3f}deg"
        )

        train_loss_history.append(train_loss_meter.avg)
        val_loss_history.append(val_metrics["loss"])

        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": vars(args),
            "task": "direction_only",
        }

        torch.save(checkpoint, last_path)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_path)
            print(f"[train] saved best checkpoint: {best_path}")

    try:
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(train_loss_history) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss_history, label="train_loss")
        plt.plot(epochs, val_loss_history, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Direction Loss")
        plt.title("Direction-Only Training/Validation Loss")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend()
        plot_path = out_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[train] saved loss plot: {plot_path}")
    except Exception as exc:
        print(f"[train] WARNING: failed to save loss plot: {exc}")

    print(f"[train] done. best_val_loss={best_val_loss:.6f}")
    print(f"[train] best checkpoint: {best_path}")
    print(f"[train] last checkpoint: {last_path}")


if __name__ == "__main__":
    main()

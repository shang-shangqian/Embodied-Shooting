# train_vision.py

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

from vision_model import build_vision_model, compute_vision_losses


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VisionChunkDataset(Dataset):
    """
    Reads dataset chunks saved by collect_vision_data.py.

    Expected file format (.pt):
    {
        "images": Tensor [N, T, 3, H, W],
        "direction": Tensor [N, 2],
        "velocity": Tensor [N, 3],
        "state": Tensor [N, 10],
        "meta": {...}
    }
    """

    def __init__(
        self,
        root_dir: str | Path,
        normalize_images: bool = True,
        max_files: int | None = None,
    ):
        self.root_dir = Path(root_dir).expanduser()
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.files = sorted(self.root_dir.glob("vision_chunk_*.pt"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No vision_chunk_*.pt found under: {self.root_dir}")

        if max_files is not None:
            self.files = self.files[:max_files]

        self.normalize_images = normalize_images

        self.samples: list[tuple[int, int]] = []
        self.file_lengths: list[int] = []
        self._cache: dict[int, dict] = {}

        for file_idx, file_path in enumerate(self.files):
            payload = torch.load(file_path, map_location="cpu")
            images = payload["images"]
            direction = payload["direction"]
            velocity = payload["velocity"]
            state = payload["state"]

            if images.ndim != 5:
                raise ValueError(f"{file_path} images shape invalid: {tuple(images.shape)}")
            if direction.ndim != 2 or direction.shape[-1] != 2:
                raise ValueError(f"{file_path} direction shape invalid: {tuple(direction.shape)}")
            if velocity.ndim != 2 or velocity.shape[-1] != 3:
                raise ValueError(f"{file_path} velocity shape invalid: {tuple(velocity.shape)}")
            if state.ndim != 2 or state.shape[-1] != 10:
                raise ValueError(f"{file_path} state shape invalid: {tuple(state.shape)}")

            n = images.shape[0]
            if direction.shape[0] != n or velocity.shape[0] != n or state.shape[0] != n:
                raise ValueError(f"{file_path} sample count mismatch")

            self.file_lengths.append(n)
            for sample_idx in range(n):
                self.samples.append((file_idx, sample_idx))

        print(f"[dataset] found {len(self.files)} chunk files")
        print(f"[dataset] total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_file(self, file_idx: int) -> dict:
        # 如果当前需要的文件不在缓存中
        if file_idx not in self._cache:
            # 清空缓存中所有其他的 chunk，释放内存！
            self._cache.clear() 
            # 加载新文件
            self._cache[file_idx] = torch.load(self.files[file_idx], map_location="cpu")
        return self._cache[file_idx]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, sample_idx = self.samples[index]
        payload = self._load_file(file_idx)

        images = payload["images"][sample_idx].float()       # [T, 3, H, W]
        direction = payload["direction"][sample_idx].float() # [2]
        velocity = payload["velocity"][sample_idx].float()   # [3]
        state = payload["state"][sample_idx].float()[4:]         # [6] (root_ang_vel_b + projected_gravity_b)

        if self.normalize_images:
            if images.max() > 1.5:
                images = images / 255.0

        return {
            "images": images,
            "direction": direction,
            "velocity": velocity,
            "state": state,
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
    """按 Chunk(文件) 划分训练集和验证集，防止时序数据泄漏"""
    # 提取所有不重复的 chunk 索引
    unique_chunks = list(set([file_idx for file_idx, _ in samples]))
    rng = random.Random(seed)
    rng.shuffle(unique_chunks)

    # 划分 chunk
    val_chunk_count = max(1, int(len(unique_chunks) * val_ratio))
    val_chunks = set(unique_chunks[:val_chunk_count])
    
    train_indices = []
    val_indices = []
    
    # 遍历所有样本，根据其所属的 chunk 归入对应的数据集
    for idx, (file_idx, _) in enumerate(samples):
        if file_idx in val_chunks:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
            
    return train_indices, val_indices

class SubsetChunkSampler(torch.utils.data.Sampler):
    """
    按数据块（Chunk）分组读取的采样器。
    保证同一个 Chunk 的数据被连续读取，避免硬盘频繁切换和缓存疯狂重载。
    """
    def __init__(self, subset: torch.utils.data.Subset, shuffle: bool = True):
        self.subset = subset
        self.shuffle = shuffle

        dataset = cast(VisionChunkDataset, self.subset.dataset)
        
        # 将 subset 中的索引按其对应的 chunk 文件分组
        self.chunk_to_subset_idx = {}
        for subset_idx, dataset_idx in enumerate(self.subset.indices):
            file_idx, _ = dataset.samples[dataset_idx]
            if file_idx not in self.chunk_to_subset_idx:
                self.chunk_to_subset_idx[file_idx] = []
            self.chunk_to_subset_idx[file_idx].append(subset_idx)
            
    def __iter__(self):
        # 1. 随机打乱 Chunk 的读取顺序
        chunk_order = list(self.chunk_to_subset_idx.keys())
        if self.shuffle:
            random.shuffle(chunk_order)
            
        final_indices = []
        for chunk_idx in chunk_order:
            indices = self.chunk_to_subset_idx[chunk_idx].copy()
            # 2. 随机打乱该 Chunk 内部的数据顺序
            if self.shuffle:
                random.shuffle(indices)
            final_indices.extend(indices)
            
        return iter(final_indices)
        
    def __len__(self) -> int:
        return len(self.subset)

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    loss_meter = RunningAverage()
    dir_meter = RunningAverage()
    vel_meter = RunningAverage()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            gt_direction = batch["direction"].to(device, non_blocking=True)
            gt_velocity = batch["velocity"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)

            out = model(images, proprio=state)
            loss_dict = compute_vision_losses(
                pred_direction=out.direction,
                pred_velocity=out.velocity,
                gt_direction=gt_direction,
                gt_velocity=gt_velocity,
                direction_weight=1.0,  # 方向权重保持不变
                velocity_weight=3.0,   # 【新增】强迫网络重视速度！
            )

            batch_size = images.shape[0]
            loss_meter.update(float(loss_dict["loss"].item()), batch_size)
            dir_meter.update(float(loss_dict["direction_loss"].item()), batch_size)
            vel_meter.update(float(loss_dict["velocity_loss"].item()), batch_size)

    return {
        "loss": loss_meter.avg,
        "direction_loss": dir_meter.avg,
        "velocity_loss": vel_meter.avg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="vision_runs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_out_dir = Path(args.out_dir).expanduser()
    timestamp = datetime.now().strftime("%m-%d-%H-%M")
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = VisionChunkDataset(
        root_dir=args.dataset_dir,
        normalize_images=True,
        max_files=args.max_files,
    )

    # 使用新的按 chunk 划分函数
    train_indices, val_indices = split_indices_by_chunk(dataset.samples, args.val_ratio, args.seed)
    if len(train_indices) == 0:
        raise ValueError("No training samples after split. Reduce --val_ratio or collect more data.")
    if len(val_indices) == 0:
        print("[train] WARNING: validation set is empty. Falling back to using training split as validation.")
        val_indices = train_indices[:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    # 使用自定义的 Chunk 采样器
    train_sampler = SubsetChunkSampler(train_subset, shuffle=True)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # 使用 sampler 时，不能再写 shuffle=True
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

    model = build_vision_model(use_proprio=True, proprio_dim=6)
    
    # 自动检测并使用多卡 (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"[train] 🚀 激活算力怪兽！检测到 {torch.cuda.device_count()} 张 GPU，启用 nn.DataParallel。")
        model = nn.DataParallel(model)
        
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

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
        train_vel_meter = RunningAverage()

        # 【添加进度追踪】
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)             # [B,T,3,H,W]
            gt_direction = batch["direction"].to(device, non_blocking=True)    # [B,2]
            gt_velocity = batch["velocity"].to(device, non_blocking=True)      # [B,3]
            state = batch["state"].to(device, non_blocking=True)               # [B,6]

            optimizer.zero_grad(set_to_none=True)

            out = model(images, proprio=state)
            loss_dict = compute_vision_losses(
                pred_direction=out.direction,
                pred_velocity=out.velocity,
                gt_direction=gt_direction,
                gt_velocity=gt_velocity,
                direction_weight=1.0,  # 方向权重保持不变
                velocity_weight=3.0,   # 【新增】强迫网络重视速度！
            )

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = images.shape[0]
            train_loss_meter.update(float(loss_dict["loss"].item()), batch_size)
            train_dir_meter.update(float(loss_dict["direction_loss"].item()), batch_size)
            train_vel_meter.update(float(loss_dict["velocity_loss"].item()), batch_size)
            
            # 【每跑 20 个 Batch 打印一次进度】
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                print(f"    [Epoch {epoch} | Batch {batch_idx + 1}/{num_batches}] Loss: {loss.item():.4f}")

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss_meter.avg:.6f} "
            f"train_dir={train_dir_meter.avg:.6f} "
            f"train_vel={train_vel_meter.avg:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_dir={val_metrics['direction_loss']:.6f} "
            f"val_vel={val_metrics['velocity_loss']:.6f}"
        )

        train_loss_history.append(train_loss_meter.avg)
        val_loss_history.append(val_metrics["loss"])

        # 剥离 DataParallel 的 "module." 外壳，确保保存的权重可以被单卡无缝加载
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": vars(args),
        }

        torch.save(checkpoint, last_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_path)
            print(f"[train] saved best checkpoint: {best_path}")

    # Save loss curves
    try:
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(train_loss_history) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss_history, label="train_loss")
        plt.plot(epochs, val_loss_history, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training/Validation Loss")
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
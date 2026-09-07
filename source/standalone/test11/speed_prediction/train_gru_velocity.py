import argparse
from datetime import datetime
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from speed_prediction_model import VelocityPredictorGRU


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pt_files(data_dirs: List[str]) -> List[Path]:
    files: List[Path] = []
    for data_dir in data_dirs:
        p = Path(data_dir).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Data directory not found: {p}")
        files.extend(sorted(p.glob("*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files found in: {data_dirs}")
    return files


def split_train_val_files(files: List[Path], train_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    files = files[:]
    rnd = random.Random(seed)
    rnd.shuffle(files)

    n = len(files)
    if n < 2:
        raise RuntimeError("Need at least 2 .pt files to split train/val.")

    n_train = max(1, int(n * train_ratio))
    if n_train >= n:
        n_train = n - 1

    train_files = files[:n_train]
    val_files = files[n_train:]

    if len(val_files) == 0:
        val_files = train_files[-1:]
        train_files = train_files[:-1]

    return train_files, val_files


def smooth_position_sequence(pos: torch.Tensor, window: int) -> torch.Tensor:
    if window < 1:
        raise ValueError(f"pos_smooth_window must be >= 1, got {window}")
    if window == 1:
        return pos.clone()

    smoothed = pos.clone()
    for t in range(1, pos.shape[1]):
        start = max(0, t + 1 - window)
        smoothed[:, t, :] = pos[:, start : t + 1, :].mean(dim=1)
    return smoothed


class VelocitySequenceDataset(Dataset):
    """
    当前数据:
      raw_seq: [N, T, 26]

    返回:
      x: [N, T-1, 13]
      y: [N, 3]

    输入组成:
      [
        refined_pos_abs(3),
        refined_pos_diff(3),
        robot_lin_vel(3),
        robot_ang_vel(3),
        dog_moving_flag(1),
      ]
    """

    def __init__(self, file_path: str | Path, pos_smooth_window: int = 3):
        super().__init__()
        self.file_path = str(file_path)
        self.pos_smooth_window = int(pos_smooth_window)
        payload = torch.load(self.file_path, map_location="cpu", weights_only=False)

        if "sequence_data" not in payload or "target_velocity_base" not in payload:
            raise KeyError(f"{self.file_path} missing required keys: sequence_data / target_velocity_base")

        raw_seq = payload["sequence_data"].to(torch.float32)
        target = payload["target_velocity_base"].to(torch.float32)

        if raw_seq.ndim != 3:
            raise RuntimeError(f"Expected sequence_data shape [N, T, F], got {tuple(raw_seq.shape)}")
        if raw_seq.shape[1] < 2:
            raise RuntimeError(f"Sequence length must be >= 2, got {tuple(raw_seq.shape)}")
        if raw_seq.shape[2] < 26:
            raise RuntimeError(f"Feature dim must be >= 26, got {tuple(raw_seq.shape)}")
        if target.ndim != 2 or target.shape[1] != 3:
            raise RuntimeError(f"Expected target_velocity_base shape [N, 3], got {tuple(target.shape)}")
        if target.shape[0] != raw_seq.shape[0]:
            raise RuntimeError(
                f"Sample count mismatch: sequence_data has {raw_seq.shape[0]} samples, "
                f"target_velocity_base has {target.shape[0]} samples"
            )

        # 26维特征定义：
        # 0: timestamp
        # 1:4   gt_pos_base
        # 4:7   pred_pos_base
        # 7:10  refined_pos_base
        # 10:13 robot_ang_vel
        # 13:16 robot_lin_vel
        # 16:17 dog_moving_flag
        # 17:20 robot_roll, robot_pitch, robot_yaw
        # 20:21 pred_yaw
        # 21:22 pred_pitch
        # 22:23 refined_yaw
        # 23:24 refined_pitch
        # 24:25 gt_yaw
        # 25:26 gt_pitch

        refined_pos = raw_seq[:, :, 7:10]        # [N, T, 3]
        robot_ang_vel = raw_seq[:, :, 10:13]     # [N, T, 3]
        robot_lin_vel = raw_seq[:, :, 13:16]     # [N, T, 3]
        dog_flag = raw_seq[:, :, 16:17]          # [N, T, 1]

        # ---------------------------------------------------------
        # 方案 A：先平滑位置，再做差分
        # 目的：抑制 refined_pos 的高频抖动，尤其是 z 轴抖动
        # 说明：
        #   smoothed_pos[:, t] 只使用当前及过去帧，不使用未来帧，避免信息泄漏
        # ---------------------------------------------------------
        # window=1 disables smoothing; window=3 matches the previous preprocessing.
        smoothed_pos = smooth_position_sequence(refined_pos, self.pos_smooth_window)

        # t = 1：两帧均值
        # t >= 2：三帧因果滑动平均
        # 用平滑后的坐标构造绝对位置和差分
        refined_pos_abs = smoothed_pos[:, 1:, :]                            # [N, T-1, 3]
        refined_pos_diff = smoothed_pos[:, 1:, :] - smoothed_pos[:, :-1, :]  # [N, T-1, 3]

        x = torch.cat(
            [
                refined_pos_abs,               # 3
                refined_pos_diff,              # 3
                robot_lin_vel[:, 1:, :],       # 3
                robot_ang_vel[:, 1:, :],       # 3
                dog_flag[:, 1:, :],            # 1
            ],
            dim=-1,
        )  # [N, T-1, 13]

        if x.shape[-1] != 13:
            raise RuntimeError(f"Constructed input dim must be 13, got {x.shape[-1]}")

        self.x = x.contiguous()
        self.y = target.contiguous()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, weights=(1.0, 1.0, 1.0), beta=1.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.beta = float(beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise RuntimeError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
        if pred.ndim != 2 or pred.shape[1] != 3:
            raise RuntimeError(f"Expected [B, 3], got {tuple(pred.shape)}")

        diff = pred - target
        abs_diff = diff.abs()

        if self.beta < 1e-8:
            per_dim_loss = abs_diff
        else:
            per_dim_loss = torch.where(
                abs_diff < self.beta,
                0.5 * (diff ** 2) / self.beta,
                abs_diff - 0.5 * self.beta,
            )

        weighted = per_dim_loss * self.weights.view(1, 3)
        return weighted.mean()


def build_concat_dataset(files: List[Path], pos_smooth_window: int = 3) -> Dataset:
    datasets = [VelocitySequenceDataset(f, pos_smooth_window=pos_smooth_window) for f in files]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def compute_input_norm_stats(dataset: Dataset, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    只基于训练集统计输入归一化参数。
    返回:
        mean: [13]
        std:  [13]
    """
    if isinstance(dataset, ConcatDataset):
        x_list = []
        for ds in dataset.datasets:
            if not hasattr(ds, "x"):
                raise RuntimeError("Expected each sub-dataset to have attribute 'x'")
            x_list.append(ds.x)
        x_all = torch.cat(x_list, dim=0)  # [N, T, 13]
    else:
        if not hasattr(dataset, "x"):
            raise RuntimeError("Expected dataset to have attribute 'x'")
        x_all = dataset.x

    if x_all.ndim != 3 or x_all.shape[-1] != 13:
        raise RuntimeError(f"Expected training inputs shape [N, T, 13], got {tuple(x_all.shape)}")

    mean = x_all.mean(dim=(0, 1))                # [13]
    std = x_all.std(dim=(0, 1), unbiased=False) # [13]
    std = std.clamp_min(eps)

    return mean, std


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module):
    model.eval()
    total_loss = 0.0
    total_count = 0
    mae_sum = torch.zeros(3, device=device)
    mse_sum = torch.zeros(3, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_count += bs

        err = pred - y
        mae_sum += err.abs().sum(dim=0)
        mse_sum += (err ** 2).sum(dim=0)

    mean_loss = total_loss / max(total_count, 1)
    mae = mae_sum / max(total_count, 1)
    rmse = torch.sqrt(mse_sum / max(total_count, 1))

    return {
        "loss": float(mean_loss),
        "mae_x": float(mae[0].item()),
        "mae_y": float(mae[1].item()),
        "mae_z": float(mae[2].item()),
        "rmse_x": float(rmse[0].item()),
        "rmse_y": float(rmse[1].item()),
        "rmse_z": float(rmse[2].item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train/Val only GRU for target velocity prediction")
    parser.add_argument("--data_dirs", type=str, nargs="+", required=True, help="One or more dataset directories")
    parser.add_argument("--save_dir", type=str, default="gru_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Remaining files are used for val")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_wx", type=float, default=1.0)
    parser.add_argument("--loss_wy", type=float, default=1.0)
    parser.add_argument("--loss_wz", type=float, default=1.0)
    parser.add_argument("--smooth_l1_beta", type=float, default=0.05)
    parser.add_argument(
        "--pos_smooth_window",
        type=int,
        default=1,
        help="Causal moving-average window for refined_pos. Use 1 to disable smoothing.",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    timestamp = datetime.now().strftime("%m-%d-%H-%M")
    save_root = Path(args.save_dir).expanduser()
    save_dir = save_root / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    if "cuda" in str(device) and not torch.cuda.is_available():
        print("[warning] CUDA not available, fallback to CPU.")
        device = torch.device("cpu")

    all_files = list_pt_files(args.data_dirs)
    train_files, val_files = split_train_val_files(all_files, args.train_ratio, args.seed)

    print(f"[split] total files={len(all_files)} train={len(train_files)} val={len(val_files)}")
    print("[note] This script does NOT create/use a test split. External test data should be evaluated separately.")

    if args.pos_smooth_window < 1:
        raise ValueError(f"--pos_smooth_window must be >= 1, got {args.pos_smooth_window}")

    print(f"[preprocess] pos_smooth_window={args.pos_smooth_window}")

    train_dataset = build_concat_dataset(train_files, pos_smooth_window=args.pos_smooth_window)
    val_dataset = build_concat_dataset(val_files, pos_smooth_window=args.pos_smooth_window)

    print(f"[dataset] train samples={len(train_dataset)} val samples={len(val_dataset)}")

    # 只用训练集统计归一化参数
    input_mean, input_std = compute_input_norm_stats(train_dataset)
    print("[input norm] mean =", [round(v, 6) for v in input_mean.tolist()])
    print("[input norm] std  =", [round(v, 6) for v in input_std.tolist()])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=("cuda" in str(device)),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in str(device)),
        drop_last=False,
    )

    model = VelocityPredictorGRU(
        input_dim=13,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model.set_input_norm_stats(
        mean=input_mean.to(device=device),
        std=input_std.to(device=device),
    )

    loss_fn = WeightedSmoothL1Loss(
        weights=(args.loss_wx, args.loss_wy, args.loss_wz),
        beta=args.smooth_l1_beta,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val = math.inf
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            bs = x.shape[0]
            running_loss += loss.item() * bs
            seen += bs

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(val_metrics["loss"])

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mae=({val_metrics['mae_x']:.4f}, {val_metrics['mae_y']:.4f}, {val_metrics['mae_z']:.4f}) "
            f"lr={lr_now:.2e}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "lr": lr_now,
        }
        history.append(epoch_record)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0

            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_dim": 13,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
                "input_norm": {
                    "mean": input_mean.cpu(),
                    "std": input_std.cpu(),
                },
                "preprocessing": {
                    "pos_smooth_window": int(args.pos_smooth_window),
                },
                "train_args": vars(args),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "train_files": [str(p) for p in train_files],
                "val_files": [str(p) for p in val_files],
            }
            torch.save(ckpt, save_dir / "best_model.pt")
            print(f"[save] best model updated at epoch {epoch}")
        else:
            no_improve += 1

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if no_improve >= args.early_stop_patience:
            print(f"[early stop] no improvement for {no_improve} epochs")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "preprocessing": {
            "pos_smooth_window": int(args.pos_smooth_window),
        },
        "note": "No internal test split was used. Please evaluate with external test data separately.",
    }
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()

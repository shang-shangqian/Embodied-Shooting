import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.legacy_gru"

from .speed_prediction_model import VelocityPredictorGRU


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
    """Dataset for velocity sequence chunks saved by collect_vision_data.py.

    Input chunk:
      sequence_data: [N, T, 26]
      target_velocity_base: [N, 3]

    Model input:
      x: [N, T-1, 13]
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

        # Feature indices:
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
        refined_pos = raw_seq[:, :, 7:10]
        robot_ang_vel = raw_seq[:, :, 10:13]
        robot_lin_vel = raw_seq[:, :, 13:16]
        dog_flag = raw_seq[:, :, 16:17]

        # Keep preprocessing exactly aligned with train_gru_velocity.py.
        smoothed_pos = smooth_position_sequence(refined_pos, self.pos_smooth_window)
        pos_abs = smoothed_pos[:, 1:, :]
        pos_diff = smoothed_pos[:, 1:, :] - smoothed_pos[:, :-1, :]

        self.x = torch.cat(
            [
                pos_abs,
                pos_diff,
                robot_lin_vel[:, 1:, :],
                robot_ang_vel[:, 1:, :],
                dog_flag[:, 1:, :],
            ],
            dim=-1,
        ).contiguous()
        self.y = target.contiguous()
        self.meta = torch.cat(
            [
                raw_seq[:, -1, 1:4],       # gt_pos_base at the label frame
                raw_seq[:, -1, 24:25],     # gt_yaw at the label frame
                raw_seq[:, -1, 25:26],     # gt_pitch at the label frame
            ],
            dim=-1,
        ).contiguous()

        if self.x.shape[-1] != 13:
            raise RuntimeError(f"Constructed input dim must be 13, got {self.x.shape[-1]}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.meta[idx]


def list_pt_files_in_dir(test_dir: str | Path) -> List[Path]:
    p = Path(test_dir).expanduser()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Test directory not found: {p}")
    files = sorted(p.glob("*.pt"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt files found in test directory: {p}")
    return files


def build_dataset_from_dir(test_dir: str | Path, pos_smooth_window: int = 3) -> Tuple[Dataset, List[Path]]:
    files = list_pt_files_in_dir(test_dir)
    datasets = [VelocitySequenceDataset(f, pos_smooth_window=pos_smooth_window) for f in files]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return dataset, files


@torch.no_grad()
def tensor_percentiles(values: torch.Tensor, qs=(0.5, 0.9, 0.95)) -> Dict[str, float]:
    if values.numel() == 0:
        return {f"p{int(q * 100)}": float("nan") for q in qs} | {"max": float("nan")}
    values = values.to(dtype=torch.float32)
    result = {f"p{int(q * 100)}": float(torch.quantile(values, q).item()) for q in qs}
    result["max"] = float(values.max().item())
    return result


def summarize_error_subset(err_abs: torch.Tensor, err_norm: torch.Tensor, mask: torch.Tensor) -> Dict:
    mask = mask.bool()
    count = int(mask.sum().item())
    if count == 0:
        return {
            "sample_count": 0,
            "mae_x": float("nan"),
            "mae_y": float("nan"),
            "mae_z": float("nan"),
            "rmse_x": float("nan"),
            "rmse_y": float("nan"),
            "rmse_z": float("nan"),
            "err_norm_p50": float("nan"),
            "err_norm_p90": float("nan"),
            "err_norm_p95": float("nan"),
            "err_norm_max": float("nan"),
        }

    sub_abs = err_abs[mask]
    sub_norm = err_norm[mask]
    mae = sub_abs.mean(dim=0)
    rmse = torch.sqrt((sub_abs ** 2).mean(dim=0))
    percentiles = tensor_percentiles(sub_norm)

    return {
        "sample_count": count,
        "mae_x": float(mae[0].item()),
        "mae_y": float(mae[1].item()),
        "mae_z": float(mae[2].item()),
        "rmse_x": float(rmse[0].item()),
        "rmse_y": float(rmse[1].item()),
        "rmse_z": float(rmse[2].item()),
        "err_norm_p50": percentiles["p50"],
        "err_norm_p90": percentiles["p90"],
        "err_norm_p95": percentiles["p95"],
        "err_norm_max": percentiles["max"],
    }


def make_bin_groups(
    values: torch.Tensor,
    bins: List[float],
    labels: List[str],
    err_abs: torch.Tensor,
    err_norm: torch.Tensor,
) -> List[Dict]:
    groups = []
    for i, label in enumerate(labels):
        lower = bins[i]
        upper = bins[i + 1]
        if i == 0:
            mask = (values >= lower) & (values < upper)
        else:
            mask = (values >= lower) & (values < upper)
        record = {"label": label, "range": [float(lower), float(upper)]}
        record.update(summarize_error_subset(err_abs, err_norm, mask))
        groups.append(record)
    return groups


def evaluate_dataset(model, loader, device, edge_yaw_abs_max: float, edge_pitch_abs_max: float):
    model.eval()
    loss_fn = nn.SmoothL1Loss(reduction="mean")

    total_loss = 0.0
    total_count = 0
    mae_sum = torch.zeros(3, device=device)
    mse_sum = torch.zeros(3, device=device)

    err_abs_chunks = []
    err_norm_chunks = []
    speed_chunks = []
    distance_chunks = []
    yaw_chunks = []
    pitch_chunks = []
    edge_score_chunks = []

    edge_yaw_abs_max = max(float(edge_yaw_abs_max), 1e-6)
    edge_pitch_abs_max = max(float(edge_pitch_abs_max), 1e-6)

    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_count += bs

        err = pred - y
        err_abs = err.abs()
        err_norm = torch.norm(err, dim=-1)
        speed = torch.norm(y, dim=-1)
        distance = torch.norm(meta[:, 0:3], dim=-1)
        yaw = meta[:, 3]
        pitch = meta[:, 4]
        edge_score = torch.maximum(yaw.abs() / edge_yaw_abs_max, pitch.abs() / edge_pitch_abs_max)

        mae_sum += err_abs.sum(dim=0)
        mse_sum += (err ** 2).sum(dim=0)

        err_abs_chunks.append(err_abs.detach().cpu())
        err_norm_chunks.append(err_norm.detach().cpu())
        speed_chunks.append(speed.detach().cpu())
        distance_chunks.append(distance.detach().cpu())
        yaw_chunks.append(yaw.detach().cpu())
        pitch_chunks.append(pitch.detach().cpu())
        edge_score_chunks.append(edge_score.detach().cpu())

    mae = mae_sum / max(total_count, 1)
    rmse = torch.sqrt(mse_sum / max(total_count, 1))

    err_abs_all = torch.cat(err_abs_chunks, dim=0) if err_abs_chunks else torch.empty((0, 3))
    err_norm_all = torch.cat(err_norm_chunks, dim=0) if err_norm_chunks else torch.empty((0,))
    speed_all = torch.cat(speed_chunks, dim=0) if speed_chunks else torch.empty((0,))
    distance_all = torch.cat(distance_chunks, dim=0) if distance_chunks else torch.empty((0,))
    yaw_all = torch.cat(yaw_chunks, dim=0) if yaw_chunks else torch.empty((0,))
    pitch_all = torch.cat(pitch_chunks, dim=0) if pitch_chunks else torch.empty((0,))
    edge_score_all = torch.cat(edge_score_chunks, dim=0) if edge_score_chunks else torch.empty((0,))

    err_norm_percentiles = tensor_percentiles(err_norm_all)
    err_abs_x_percentiles = tensor_percentiles(err_abs_all[:, 0]) if err_abs_all.numel() else tensor_percentiles(torch.empty(0))
    err_abs_y_percentiles = tensor_percentiles(err_abs_all[:, 1]) if err_abs_all.numel() else tensor_percentiles(torch.empty(0))
    err_abs_z_percentiles = tensor_percentiles(err_abs_all[:, 2]) if err_abs_all.numel() else tensor_percentiles(torch.empty(0))

    speed_bins = [0.0, 1.0, 2.0, 3.0, 5.0, float("inf")]
    speed_labels = ["0-1", "1-2", "2-3", "3-5", "5+"]
    distance_bins = [0.0, 80.0, 100.0, 120.0, 150.0, float("inf")]
    distance_labels = ["0-80", "80-100", "100-120", "120-150", "150+"]
    yaw_bins = [
        float("-inf"),
        -0.75 * edge_yaw_abs_max,
        -0.35 * edge_yaw_abs_max,
        0.35 * edge_yaw_abs_max,
        0.75 * edge_yaw_abs_max,
        float("inf"),
    ]
    yaw_labels = ["left_edge", "left_mid", "center", "right_mid", "right_edge"]
    pitch_bins = [
        float("-inf"),
        -0.75 * edge_pitch_abs_max,
        -0.35 * edge_pitch_abs_max,
        0.35 * edge_pitch_abs_max,
        0.75 * edge_pitch_abs_max,
        float("inf"),
    ]
    pitch_labels = ["lower_edge", "lower_mid", "center", "upper_mid", "upper_edge"]
    edge_bins = [0.0, 0.4, 0.75, float("inf")]
    edge_labels = ["center", "middle", "edge"]

    return {
        "sample_count": int(total_count),
        "loss": float(total_loss / max(total_count, 1)),
        "mae_x": float(mae[0].item()),
        "mae_y": float(mae[1].item()),
        "mae_z": float(mae[2].item()),
        "rmse_x": float(rmse[0].item()),
        "rmse_y": float(rmse[1].item()),
        "rmse_z": float(rmse[2].item()),
        "err_norm_p50": err_norm_percentiles["p50"],
        "err_norm_p90": err_norm_percentiles["p90"],
        "err_norm_p95": err_norm_percentiles["p95"],
        "err_norm_max": err_norm_percentiles["max"],
        "abs_err_x_p95": err_abs_x_percentiles["p95"],
        "abs_err_x_max": err_abs_x_percentiles["max"],
        "abs_err_y_p95": err_abs_y_percentiles["p95"],
        "abs_err_y_max": err_abs_y_percentiles["max"],
        "abs_err_z_p95": err_abs_z_percentiles["p95"],
        "abs_err_z_max": err_abs_z_percentiles["max"],
        "groups": {
            "speed": make_bin_groups(speed_all, speed_bins, speed_labels, err_abs_all, err_norm_all),
            "distance": make_bin_groups(distance_all, distance_bins, distance_labels, err_abs_all, err_norm_all),
            "yaw": make_bin_groups(yaw_all, yaw_bins, yaw_labels, err_abs_all, err_norm_all),
            "pitch": make_bin_groups(pitch_all, pitch_bins, pitch_labels, err_abs_all, err_norm_all),
            "edge_score": make_bin_groups(edge_score_all, edge_bins, edge_labels, err_abs_all, err_norm_all),
        },
    }


def print_result_block(name: str, test_dir: Path, num_files: int, result: Dict):
    print("=" * 90)
    print(f"[{name}]")
    print(f"dir       : {test_dir}")
    print(f"files     : {num_files}")
    print(f"samples   : {result['sample_count']}")
    print(f"loss      : {result['loss']:.6f}")
    print(
        f"MAE       : "
        f"({result['mae_x']:.4f}, {result['mae_y']:.4f}, {result['mae_z']:.4f})"
    )
    print(
        f"RMSE      : "
        f"({result['rmse_x']:.4f}, {result['rmse_y']:.4f}, {result['rmse_z']:.4f})"
    )
    print(
        f"ERR_NORM  : "
        f"p50={result['err_norm_p50']:.4f} "
        f"p90={result['err_norm_p90']:.4f} "
        f"p95={result['err_norm_p95']:.4f} "
        f"max={result['err_norm_max']:.4f}"
    )
    print(
        f"ABS_P95   : "
        f"({result['abs_err_x_p95']:.4f}, {result['abs_err_y_p95']:.4f}, {result['abs_err_z_p95']:.4f})"
    )
    print(
        f"ABS_MAX   : "
        f"({result['abs_err_x_max']:.4f}, {result['abs_err_y_max']:.4f}, {result['abs_err_z_max']:.4f})"
    )
    for group_name, groups in result.get("groups", {}).items():
        print(f"\n[{name}] by {group_name}")
        print("  label          samples  MAE(x,y,z)                 RMSE(x,y,z)                p50    p90    p95    max")
        for group in groups:
            print(
                f"  {group['label']:<13} "
                f"{group['sample_count']:>7}  "
                f"({group['mae_x']:.4f}, {group['mae_y']:.4f}, {group['mae_z']:.4f})  "
                f"({group['rmse_x']:.4f}, {group['rmse_y']:.4f}, {group['rmse_z']:.4f})  "
                f"{group['err_norm_p50']:.4f} "
                f"{group['err_norm_p90']:.4f} "
                f"{group['err_norm_p95']:.4f} "
                f"{group['err_norm_max']:.4f}"
            )
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRU velocity model on provided scenario dirs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pt")

    parser.add_argument("--dd_dir", type=str, default=None, help="Optional dir: dog_dynamic_target_dynamic")
    parser.add_argument("--ds_dir", type=str, default=None, help="Optional dir: dog_dynamic_target_static")
    parser.add_argument("--sd_dir", type=str, default=None, help="Optional dir: dog_static_target_dynamic")
    parser.add_argument("--ss_dir", type=str, default=None, help="Optional dir: dog_static_target_static")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pos_smooth_window",
        type=int,
        default=None,
        help="Causal moving-average window for refined_pos. Defaults to checkpoint preprocessing.",
    )
    parser.add_argument(
        "--edge_yaw_abs_max",
        type=float,
        default=0.26,
        help="Yaw half-range used to estimate image-edge score.",
    )
    parser.add_argument(
        "--edge_pitch_abs_max",
        type=float,
        default=0.14,
        help="Pitch half-range used to estimate image-edge score.",
    )
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save metrics json")
    args = parser.parse_args()

    scenario_args = {
        "dd": args.dd_dir,
        "ds": args.ds_dir,
        "sd": args.sd_dir,
        "ss": args.ss_dir,
    }
    eval_dirs = {
        name: Path(path).expanduser()
        for name, path in scenario_args.items()
        if path is not None
    }
    if not eval_dirs:
        raise ValueError("Pass at least one dataset directory, for example --sd_dir <dir> --ss_dir <dir>.")

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    if "cuda" in str(device) and not torch.cuda.is_available():
        print("[warning] CUDA not available, fallback to CPU.")
        device = torch.device("cpu")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    train_args = ckpt.get("train_args", {})
    preprocessing = ckpt.get("preprocessing", {})
    ckpt_pos_smooth_window = preprocessing.get(
        "pos_smooth_window",
        train_args.get("pos_smooth_window", 3),
    )
    pos_smooth_window = int(args.pos_smooth_window if args.pos_smooth_window is not None else ckpt_pos_smooth_window)
    if pos_smooth_window < 1:
        raise ValueError(f"--pos_smooth_window must be >= 1, got {pos_smooth_window}")
    print(f"[preprocess] pos_smooth_window={pos_smooth_window}")

    model = VelocityPredictorGRU(
        input_dim=int(cfg.get("input_dim", 13)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    input_norm = ckpt.get("input_norm", None)
    if input_norm is not None:
        mean = input_norm["mean"].to(device=device, dtype=torch.float32)
        std = input_norm["std"].to(device=device, dtype=torch.float32)
        model.set_input_norm_stats(mean=mean, std=std)
        print("[input norm] loaded from checkpoint")
        print("[input norm] mean =", [round(v, 6) for v in mean.detach().cpu().tolist()])
        print("[input norm] std  =", [round(v, 6) for v in std.detach().cpu().tolist()])
    else:
        print("[warning] No input_norm found in checkpoint. Using identity normalization.")

    all_results = {}

    for name, test_dir in eval_dirs.items():
        dataset, files = build_dataset_from_dir(test_dir, pos_smooth_window=pos_smooth_window)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=("cuda" in str(device)),
            drop_last=False,
        )

        result = evaluate_dataset(
            model,
            loader,
            device,
            edge_yaw_abs_max=args.edge_yaw_abs_max,
            edge_pitch_abs_max=args.edge_pitch_abs_max,
        )
        result["num_files"] = len(files)
        result["test_dir"] = str(test_dir)
        result["pos_smooth_window"] = pos_smooth_window
        result["edge_yaw_abs_max"] = float(args.edge_yaw_abs_max)
        result["edge_pitch_abs_max"] = float(args.edge_pitch_abs_max)
        result["files"] = [str(f) for f in files]

        all_results[name] = result
        print_result_block(name, test_dir, len(files), result)

    print("\n" + "#" * 90)
    print("Evaluation summary")
    print("#" * 90)
    for name in eval_dirs.keys():
        r = all_results[name]
        print(
            f"{name} | "
            f"samples={r['sample_count']} | "
            f"loss={r['loss']:.6f} | "
            f"MAE=({r['mae_x']:.4f}, {r['mae_y']:.4f}, {r['mae_z']:.4f}) | "
            f"RMSE=({r['rmse_x']:.4f}, {r['rmse_y']:.4f}, {r['rmse_z']:.4f}) | "
            f"ERR_NORM p95={r['err_norm_p95']:.4f} max={r['err_norm_max']:.4f}"
        )
    print("#" * 90)

    if args.save_json:
        out = Path(args.save_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[save] metrics json -> {out}")


if __name__ == "__main__":
    main()

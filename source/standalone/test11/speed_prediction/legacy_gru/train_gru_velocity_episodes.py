"""Train the unchanged legacy GRU using complete TAQ-v2 episode streams."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.legacy_gru"

from ..common.time_aware_velocity_data import TAQ_LABEL_SEMANTICS, TAQ_STREAM_FORMAT, load_torch_payload
from .speed_prediction_model import VelocityPredictorGRU
from .online_gru_velocity import load_legacy_checkpoint
from .taq_dataset import causal_smooth, chunk_identity, split_episode_files
from .train_gru_velocity import WeightedSmoothL1Loss, set_seed


def discover_episodes(data_dirs):
    files = set()
    for directory in data_dirs:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Data directory not found: {root}")
        files.update(root.rglob("*.taqstream"))
    if not files:
        raise FileNotFoundError("No .taqstream episodes found; window chunks are not complete episodes")
    return sorted(files)


def read_episode(path, smooth_window):
    _, scenario, uid = chunk_identity(path)
    payload = load_torch_payload(path)
    metadata = payload["metadata"]
    if payload.get("stream_format") != TAQ_STREAM_FORMAT:
        raise ValueError(f"{path}: unsupported episode stream format")
    if metadata.get("episode_complete") is not True:
        raise ValueError(f"{path}: complete-episode flag is missing or false")
    features = torch.as_tensor(payload["sequence_features"], dtype=torch.float32)
    labels = torch.as_tensor(payload["target_velocity_base"], dtype=torch.float32)
    valid = torch.as_tensor(payload["sequence_valid_mask"], dtype=torch.bool)
    supervised = torch.as_tensor(payload["target_valid_mask"], dtype=torch.bool)
    sources = torch.as_tensor(payload["sequence_source_ids"], dtype=torch.long)
    timestamps = torch.as_tensor(payload["timestamps_s"], dtype=torch.float64)
    n = len(features)
    if features.shape != (n, 24) or labels.shape != (n, 3):
        raise ValueError(f"{path}: expected [T,24] features and [T,3] labels")
    if any(value.shape != (n,) for value in (valid, supervised, sources, timestamps)):
        raise ValueError(f"{path}: inconsistent time/mask/source shapes")
    if not torch.equal(valid, torch.arange(n) < int(valid.sum())):
        raise ValueError(f"{path}: valid frames must form a contiguous prefix")
    if torch.any(supervised & ~valid):
        raise ValueError(f"{path}: supervised padding frames")
    n = int(valid.sum())
    features, labels, supervised = features[:n], labels[:n], supervised[:n]
    sources, timestamps = sources[:n], timestamps[:n]
    if not torch.isfinite(timestamps).all() or torch.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError(f"{path}: timestamps must be finite and strictly increasing")
    if not torch.isfinite(features).all() or not torch.isfinite(labels[supervised]).all():
        raise ValueError(f"{path}: nonfinite inputs or supervised labels")
    if torch.any((sources < 0) | (sources > 4)):
        raise ValueError(f"{path}: unsupported tracker source")

    # Match OnlineLegacyGRU(initialization_policy='first_valid', context='full').
    # Only initialize from a direct position/range observation. Thereafter retain
    # measured/predicted/zero-filled positions exactly as stored, without GT repair
    # or tracker-triggered hidden-state resets. Episode boundaries alone reset GRU.
    direct = (sources <= 2) & (features[:, 22] > .5) & (features[:, 23] > .5)
    available = torch.nonzero(direct, as_tuple=False).flatten()
    if not len(available):
        raise ValueError(f"{path}: no direct observation to initialize the legacy GRU")
    start = int(available[0])
    current = features[start:]
    if len(current) < 2 or not supervised[start + 1:].any():
        raise ValueError(f"{path}: no supervised predictions after initialization")
    position = causal_smooth(current[:, :3], smooth_window)
    moving = (current[1:, 6:8].norm(dim=-1) > .05).float()[:, None]
    x = torch.cat((position[1:], position[1:] - position[:-1],
                   current[1:, 6:12], moving), dim=-1)
    return {
        "x": x, "y": labels[start + 1:], "supervised": supervised[start + 1:],
        "direct": direct[start + 1:],
        "record": {"path": str(path), "episode_uid": uid, "scenario": scenario,
                   "recorded_frames": n, "input_frames": len(x),
                   "initialization_discarded_frames": start,
                   "supervised_frames": int(supervised[start + 1:].sum())},
    }


class EpisodeDataset(Dataset):
    def __init__(self, files, smooth_window=1):
        self.episodes = [read_episode(path, smooth_window) for path in files]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        return self.episodes[index]

    def counts(self):
        return {"episodes": len(self), **{
            key: sum(episode["record"][key] for episode in self.episodes)
            for key in ("recorded_frames", "input_frames", "supervised_frames")
        }}


def collate_episodes(batch):
    return {key: pad_sequence([episode[key] for episode in batch], batch_first=True)
            for key in ("x", "y", "supervised", "direct")}


def input_norm(dataset):
    # Only training episodes, only actual post-initialization input frames.
    values = torch.cat([episode["x"] for episode in dataset.episodes]).double()
    return values.mean(0).float(), values.std(0, unbiased=False).clamp_min(1e-6).float()


def prepare_initialization(args):
    """Resolve architecture/preprocessing and preserve all checkpoint buffers."""
    model = checkpoint = None
    info = {"mode": "from_scratch", "normalization_source": "training_episodes",
            "optimizer_state": "new_AdamW"}
    if args.init_checkpoint:
        path = Path(args.init_checkpoint).expanduser().resolve(strict=True)
        model, checkpoint = load_legacy_checkpoint(path)
        preprocessing = checkpoint.get("preprocessing", {})
        smoothing = preprocessing.get("pos_smooth_window")
        if not isinstance(smoothing, int) or smoothing < 1:
            raise ValueError("Initialization checkpoint must declare positive pos_smooth_window")
        if preprocessing.get("taq_moving_flag") != "horizontal_base_speed_gt_0.05_mps":
            raise ValueError("Initialization checkpoint has incompatible motion-flag preprocessing")
        if (not torch.isfinite(model.input_mean).all() or not torch.isfinite(model.input_std).all()
                or torch.any(model.input_std <= 0)):
            raise ValueError("Initialization checkpoint has invalid normalization statistics")
        info.update(mode="fine_tune", normalization_source="init_checkpoint",
                    checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    defaults = {"hidden_dim": 256, "num_layers": 1, "dropout": 0., "pos_smooth_window": 1}
    for name, default in defaults.items():
        inherited = (checkpoint["preprocessing"][name] if name == "pos_smooth_window"
                     else checkpoint["config"][name]) if checkpoint else default
        requested = getattr(args, name)
        if checkpoint and requested is not None and requested != inherited:
            raise ValueError(f"--{name}={requested} differs from initialization checkpoint ({inherited})")
        setattr(args, name, inherited if requested is None else requested)
    return model, info


def predict_episode_frames(model, x):
    output, _ = model.gru(model.normalize_input(x))
    return model.head(output)


class FrameMetrics:
    def __init__(self):
        self.loss_sum = 0.
        self.errors = []
        self.direct = []

    def add(self, prediction, target, direct, loss):
        error = (prediction - target).detach().cpu()
        self.errors.append(error)
        self.direct.append(direct.detach().cpu())
        self.loss_sum += float(loss.detach()) * len(error)

    def result(self):
        error = torch.cat(self.errors)
        norms = error.norm(dim=-1)
        observed = torch.cat(self.direct)
        result = {"loss": self.loss_sum / len(error), "frame_count": len(error),
                  "err_norm_mean": float(norms.mean()), "err_norm_max": float(norms.max()),
                  "axis_mae": error.abs().mean(0).tolist(),
                  "axis_rmse": error.square().mean(0).sqrt().tolist()}
        for percentile in (50, 75, 80, 85, 90, 95, 99):
            result[f"err_norm_p{percentile}"] = float(torch.quantile(norms, percentile / 100))
        for name, mask in (("direct_observation", observed), ("non_direct_observation", ~observed)):
            result[name] = {"frame_count": int(mask.sum())}
            if mask.any():
                result[name].update(mean=float(norms[mask].mean()),
                                    p95=float(torch.quantile(norms[mask], .95)))
        return result


def run_epoch(model, loader, device, loss_fn, optimizer=None, grad_clip=1.):
    training = optimizer is not None
    model.train(training)
    metrics = FrameMetrics()
    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["supervised"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Full backpropagation through each episode; no temporal cuts,
            # hidden-state resets or optimizer steps inside an episode.
            prediction = predict_episode_frames(model, x)[mask]
            target = y[mask]
            loss = loss_fn(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite full-episode loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            metrics.add(prediction, target, batch["direct"][batch["supervised"]], loss)
    return metrics.result()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--save_dir", default="source/standalone/test11/speed_prediction/legacy_gru/full_episode_runs")
    parser.add_argument("--train_ratio", type=float, default=.8)
    parser.add_argument("--test_ratio", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32, help="Complete episodes per batch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--smooth_l1_beta", type=float, default=.05)
    parser.add_argument("--init_checkpoint", help="Initialize weights and normalization; start a fresh optimizer")
    parser.add_argument("--pos_smooth_window", type=int, default=None,
                        help="Defaults to checkpoint preprocessing, or 1 when training from scratch")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    initialized_model, initialization = prepare_initialization(args)
    if min(args.batch_size, args.epochs, args.hidden_dim, args.num_layers,
           args.pos_smooth_window, args.early_stop_patience) < 1:
        parser.error("Batch size, epochs, architecture, smoothing and patience must be positive")
    if args.lr <= 0 or args.grad_clip <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        parser.error("Invalid optimizer or worker settings")
    files = discover_episodes(args.data_dirs)
    splits, split_info = split_episode_files(files, args.train_ratio, args.test_ratio, args.seed)
    # Different streams with the same uid would duplicate trajectories in a split.
    identities = [chunk_identity(path)[2] for path in files]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate episode identities; specify one copy of each trajectory")
    datasets = {name: EpisodeDataset(paths, args.pos_smooth_window)
                for name, paths in splits.items() if paths}
    if initialized_model is None:
        mean, std = input_norm(datasets["train"])
    else:
        mean = initialized_model.input_mean.detach().cpu().flatten().clone()
        std = initialized_model.input_std.detach().cpu().flatten().clone()
    counts = {name: dataset.counts() for name, dataset in datasets.items()}
    save_dir = Path(args.save_dir).expanduser() / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    save_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "context": "full_episode", "backpropagation": "full_episode_bptt",
        "initialization_policy": "first_valid", "hidden_reset": "episode_boundary_only",
        "supervision": "all_stored_target_valid_frames_after_first_position_pair",
        "metrics": "pooled_per_frame_L2_velocity_error_mps",
        "selection_metric": "validation_err_norm_mean", "selection_direction": "min",
        "input_contract": "legacy_13d_position_position_increment_proprio_motion_flag",
        "evaluation_split": "episode_disjoint",
        "initialization": initialization,
    }
    manifest = {**split_info, "seed": args.seed, "train_ratio": args.train_ratio,
                "test_ratio": args.test_ratio, "counts": counts, "protocol": protocol,
                "files": {name: [str(path) for path in paths] for name, paths in splits.items()},
                "episodes": {name: [e["record"] for e in ds.episodes] for name, ds in datasets.items()}}
    write_json(save_dir / "data_manifest.json", manifest)
    write_json(save_dir / "input_norm.json", {"mean": mean.tolist(), "std": std.tolist(),
               "enabled": bool(initialized_model.norm_enabled) if initialized_model is not None else True,
               "source": initialization["normalization_source"]})
    write_json(save_dir / "run_config.json", {"args": vars(args), "protocol": protocol})
    print(f"[dataset] {counts}", flush=True)
    print(f"[initialization] {initialization}", flush=True)
    print(f"[output] {save_dir.resolve()}", flush=True)
    if args.prepare_only:
        print("[prepare-only] Checked complete episodes and saved splits/normalization; no training started.")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate isaac_lab_232 or explicitly select --device cpu")
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train",
                               num_workers=args.num_workers, pin_memory=device.type == "cuda",
                               collate_fn=collate_episodes)
               for name, ds in datasets.items()}
    config = dict(input_dim=13, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    if initialized_model is None:
        model = VelocityPredictorGRU(**config).to(device)
        model.set_input_norm_stats(mean, std)
    else:
        # No set_input_norm_stats here: it would overwrite inherited buffers or
        # enable normalization for a checkpoint that deliberately disabled it.
        model = initialized_model.to(device)
    loss_fn = WeightedSmoothL1Loss(beta=args.smooth_l1_beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=4)
    history = []
    best = math.inf
    best_epoch = 0

    def save_best(epoch, validation):
        torch.save({"model_state_dict": model.state_dict(), "config": config,
                    "input_norm": {"mean": mean, "std": std},
                    "preprocessing": {"pos_smooth_window": args.pos_smooth_window,
                                      "taq_moving_flag": "horizontal_base_speed_gt_0.05_mps",
                                      "output_frame": "each_supervised_frame",
                                      "initialization_policy": "first_valid"},
                    "data_schema": split_info["schema"], "label_semantics": TAQ_LABEL_SEMANTICS,
                    "train_args": vars(args), "training_protocol": protocol,
                    "initialization": initialization,
                    "best_epoch": epoch, "best_val_loss": validation["loss"],
                    "best_val_err_norm_mean": validation["err_norm_mean"], "best_val_metrics": validation,
                    "train_files": manifest["files"]["train"],
                    "val_files": manifest["files"]["val"],
                    "test_files": manifest["files"]["test"]}, save_dir / "best_model.pt")

    initial_validation = None
    if initialized_model is not None:
        initial_validation = run_epoch(model, loaders["val"], device, loss_fn)
        best = initial_validation["err_norm_mean"]
        write_json(save_dir / "initial_val_metrics.json", initial_validation)
        save_best(0, initial_validation)
        scheduler.step(best)
        print(f"[initial validation] mean={best:.5f} "
              f"p95={initial_validation['err_norm_p95']:.5f} "
              f"axis_mae={initial_validation['axis_mae']}", flush=True)
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, loaders["train"], device, loss_fn, optimizer, args.grad_clip)
        validation = run_epoch(model, loaders["val"], device, loss_fn)
        score = validation["err_norm_mean"]
        scheduler.step(score)
        history.append({"epoch": epoch, "train": training, "val": validation,
                        "lr": optimizer.param_groups[0]["lr"]})
        write_json(save_dir / "history.json", history)
        print(f"[epoch {epoch:03d}] train_mean={training['err_norm_mean']:.5f} "
              f"val_mean={score:.5f} val_p95={validation['err_norm_p95']:.5f} "
              f"val_axis_mae={validation['axis_mae']}", flush=True)
        if score < best:
            best, best_epoch = score, epoch
            save_best(epoch, validation)
        if epoch - best_epoch >= args.early_stop_patience:
            print(f"[early stop] no improvement for {args.early_stop_patience} epochs", flush=True)
            break

    test_metrics = None
    if "test" in loaders:
        best_checkpoint = load_torch_payload(save_dir / "best_model.pt")
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        test_metrics = run_epoch(model, loaders["test"], device, loss_fn)
        write_json(save_dir / "test_metrics.json", test_metrics)
        print(f"[held-out test] {test_metrics}", flush=True)
    summary = {"best_epoch": best_epoch, "best_val_err_norm_mean": best,
               "counts": counts, "test_metrics": test_metrics, "protocol": protocol,
               "initial_val_metrics": initial_validation}
    write_json(save_dir / "summary.json", summary)
    print(f"[done] {save_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

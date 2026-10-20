"""Train the schema-v2 Time-Aware Quality-Gated GRU velocity model."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.taq_gru"

from ..common.time_aware_velocity_data import (
    DEFAULT_BASELINE_VERSION,
    SUPPORTED_BASELINE_VERSIONS,
    DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
    DEFAULT_NORMALIZED_FEATURE_CLIP,
    ROBUST_INPUT_TRANSFORM_VERSION,
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION,
    build_taq_dataset,
    collate_taq_velocity_batch,
    compute_episode_stratified_weights,
    compute_taq_feature_norm_stats,
    taq_scenario_window_counts,
)
from .time_aware_velocity_model import TimeAwareQualityGatedGRU, count_trainable_parameters
from .taq_velocity_training import (
    evaluate_taq_model,
    json_ready_metrics,
    masked_taq_velocity_loss,
    move_taq_batch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_top_p95_checkpoints(
    *,
    payload: dict,
    macro_p95: float,
    epoch: int,
    output_dir: Path,
    records: list[tuple[float, int, Path]],
    keep: int = 3,
) -> list[tuple[float, int, Path]]:
    """Persist the current epoch, then retain the actual lowest-P95 files."""
    if keep < 1:
        raise ValueError("keep must be positive")
    current_path = output_dir / f"p95_epoch_{epoch:03d}.pt"
    torch.save(payload, current_path)
    ranked = [*records, (float(macro_p95), int(epoch), current_path)]
    ranked.sort(key=lambda item: (item[0], item[1]))
    for _, _, stale_path in ranked[keep:]:
        if stale_path.is_file():
            stale_path.unlink()
    return ranked[:keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TAQ-GRU on episode-separated taq_velocity_v2 chunks."
    )
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--val_dirs", nargs="+", required=True)
    parser.add_argument("--save_dir", type=Path, default=Path("taq_gru_runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples_per_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--huber_beta", type=float, default=0.1)
    parser.add_argument("--uncertainty_weight", type=float, default=0.10)
    parser.add_argument("--speed_norm_weight", type=float, default=0.05)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_dt_s", type=float, default=0.25)
    parser.add_argument("--baseline_window_s", type=float, default=0.48)
    parser.add_argument("--baseline_version", choices=SUPPORTED_BASELINE_VERSIONS, default=DEFAULT_BASELINE_VERSION)
    parser.add_argument(
        "--finite_difference_clip_mps",
        type=float,
        default=DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
        help="Vector-norm cap for measured finite-difference velocity.",
    )
    parser.add_argument(
        "--normalized_feature_clip",
        type=float,
        default=DEFAULT_NORMALIZED_FEATURE_CLIP,
        help="Absolute cap applied after continuous-feature normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("--batch_size and --epochs must be positive")
    if args.huber_beta <= 0.0 or args.early_stop_patience < 1:
        raise ValueError("Invalid loss or early-stop configuration")
    if args.finite_difference_clip_mps <= 0.0 or args.normalized_feature_clip <= 0.0:
        raise ValueError("Robust feature clip values must be positive")
    train_paths = {str(Path(path).resolve()) for path in args.train_dirs}
    val_paths = {str(Path(path).resolve()) for path in args.val_dirs}
    if train_paths & val_paths:
        raise ValueError("--train_dirs and --val_dirs must be trajectory-disjoint")
    set_seed(args.seed)

    device = torch.device(args.device if "cuda" not in args.device or torch.cuda.is_available() else "cpu")
    train_dataset = build_taq_dataset(args.train_dirs)
    val_dataset = build_taq_dataset(args.val_dirs)
    train_scenarios = taq_scenario_window_counts(train_dataset)
    val_scenarios = taq_scenario_window_counts(val_dataset)
    for split_name, counts in (("train", train_scenarios), ("validation", val_scenarios)):
        missing = [name for name in ("ss", "sd") if counts[name] == 0]
        if missing:
            raise ValueError(f"{split_name} dataset is missing required scenarios: {missing}")
    sample_weights = compute_episode_stratified_weights(train_dataset)
    input_mean, input_std = compute_taq_feature_norm_stats(
        train_dataset,
        finite_difference_clip_mps=args.finite_difference_clip_mps,
        sample_weights=sample_weights,
    )
    print(f"[data] schema={TAQ_SCHEMA_VERSION} train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    print("[norm] mean=", [round(float(value), 6) for value in input_mean])
    print("[norm] std =", [round(float(value), 6) for value in input_std])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_taq_velocity_batch,
    }
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=args.samples_per_epoch if args.samples_per_epoch > 0 else len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = TimeAwareQualityGatedGRU(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_dt_s=args.max_dt_s,
        baseline_window_s=args.baseline_window_s,
        baseline_version=args.baseline_version,
        input_transform_version=ROBUST_INPUT_TRANSFORM_VERSION,
        finite_difference_clip_mps=args.finite_difference_clip_mps,
        normalized_feature_clip=args.normalized_feature_clip,
    ).to(device)
    model.set_input_norm_stats(input_mean.to(device), input_std.to(device))
    print(
        f"[model] trainable_parameters={count_trainable_parameters(model)} "
        f"baseline={model.baseline_version} "
        f"input_transform={model.input_transform_version} "
        f"fd_clip={model.finite_difference_clip_mps:g}m/s "
        f"normalized_clip=+/-{model.normalized_feature_clip:g}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    output_dir = args.save_dir.expanduser() / datetime.now().strftime("%m-%d-%H-%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": TAQ_SCHEMA_VERSION,
                "label_semantics": TAQ_LABEL_SEMANTICS,
                "baseline_version": model.baseline_version,
                "seed": args.seed,
                "train_dirs": [str(Path(path).expanduser().resolve()) for path in args.train_dirs],
                "val_dirs": [str(Path(path).expanduser().resolve()) for path in args.val_dirs],
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "samples_per_epoch": args.samples_per_epoch or len(train_dataset),
                "sampler": "scenario_speed_bin_episode_window",
                "scenario_weights": {"ss": 0.40, "sd": 0.60},
                "sd_speed_bin_weights": [0.30, 0.35, 0.35],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    best_macro_p95 = float("inf")
    best_tie_key: tuple[float, float, float, float, int] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, float | int]] = []
    top_p95: list[tuple[float, int, Path]] = []

    def checkpoint_payload(epoch: int, metrics: dict[str, float]) -> dict:
        return {
            "schema_version": TAQ_SCHEMA_VERSION,
            "label_semantics": TAQ_LABEL_SEMANTICS,
            "baseline_version": model.baseline_version,
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "input_norm": {"mean": input_mean.cpu(), "std": input_std.cpu()},
            "train_args": vars(args),
            "best_epoch": epoch,
            "best_val_metrics": metrics,
        }

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        target_windows = 0
        for raw_batch in train_loader:
            batch = move_taq_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            mean, log_sigma = model(batch["features"], batch["source_ids"], batch["valid_mask"])
            loss, _ = masked_taq_velocity_loss(
                mean,
                log_sigma,
                batch["target"],
                batch["target_valid_mask"],
                huber_beta=args.huber_beta,
                uncertainty_weight=args.uncertainty_weight,
                speed_norm_weight=args.speed_norm_weight,
            )
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            count = int(batch["target_valid_mask"].any(dim=1).sum().item())
            loss_sum += float(loss.item()) * count
            target_windows += count

        train_loss = loss_sum / max(target_windows, 1)
        val_metrics = evaluate_taq_model(
            model,
            val_loader,
            device,
            huber_beta=args.huber_beta,
            uncertainty_weight=args.uncertainty_weight,
            speed_norm_weight=args.speed_norm_weight,
        )
        scheduler.step(val_metrics["loss"])
        required_group_metrics = ("err_norm_p95_ss", "err_norm_p95_sd")
        missing_metrics = [name for name in required_group_metrics if name not in val_metrics]
        if missing_metrics:
            raise RuntimeError(f"Validation did not produce required P95 groups: {missing_metrics}")
        p95_ss = val_metrics["err_norm_p95_ss"]
        p95_sd = val_metrics["err_norm_p95_sd"]
        macro_p95 = 0.40 * p95_ss + 0.60 * p95_sd
        tie_key = (macro_p95, max(p95_ss, p95_sd), val_metrics["err_norm_p99"], val_metrics["err_norm_mean"], epoch)
        record = {"epoch": epoch, "train_loss": train_loss, **json_ready_metrics(val_metrics), "lr": optimizer.param_groups[0]["lr"], "val_p95_ss": p95_ss, "val_p95_sd": p95_sd, "val_macro_p95": macro_p95}
        history.append(record)
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
            f"mae=({val_metrics['mae_x']:.4f}, {val_metrics['mae_y']:.4f}, {val_metrics['mae_z']:.4f}) "
            f"p95_ss={p95_ss:.4f} p95_sd={p95_sd:.4f} macro_p95={macro_p95:.4f} lr={record['lr']:.2e}"
        )

        payload = checkpoint_payload(epoch, {**json_ready_metrics(val_metrics), "val_macro_p95": macro_p95})
        torch.save(payload, output_dir / "last.pt")
        previous_records = history[:-1]
        if val_metrics["loss"] <= min((float(item["loss"]) for item in previous_records), default=float("inf")):
            torch.save(payload, output_dir / "best_val_loss.pt")
        is_better_p95 = best_tie_key is None
        if best_tie_key is not None:
            delta = macro_p95 - best_tie_key[0]
            is_better_p95 = delta < -0.002 or (
                abs(delta) < 0.002 and tie_key[1:] < best_tie_key[1:]
            )
        if is_better_p95:
            best_tie_key = tie_key
            best_macro_p95 = macro_p95
            best_epoch = epoch
            no_improvement = 0
            torch.save(payload, output_dir / "best_macro_p95.pt")
            torch.save(payload, output_dir / "best_model.pt")
        else:
            no_improvement += 1
        top_p95 = update_top_p95_checkpoints(
            payload=payload,
            macro_p95=macro_p95,
            epoch=epoch,
            output_dir=output_dir,
            records=top_p95,
        )
        best_ss = min((float(item["val_p95_ss"]) for item in previous_records), default=float("inf"))
        best_sd = min((float(item["val_p95_sd"]) for item in previous_records), default=float("inf"))
        if p95_ss <= best_ss:
            torch.save(payload, output_dir / "best_ss_p95.pt")
        if p95_sd <= best_sd:
            torch.save(payload, output_dir / "best_sd_p95.pt")
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if no_improvement >= args.early_stop_patience:
            print(f"[early-stop] no validation improvement for {no_improvement} epochs")
            break

    summary = {
        "schema_version": TAQ_SCHEMA_VERSION,
        "label_semantics": TAQ_LABEL_SEMANTICS,
        "baseline_version": model.baseline_version,
        "best_epoch": best_epoch,
        "best_macro_p95": best_macro_p95,
        "best_val_loss": min((float(item["loss"]) for item in history), default=float("inf")),
        "model_parameters": count_trainable_parameters(model),
        "note": "Train and validation directories must be split by trajectory/episode, not random windows.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] best_epoch={best_epoch} best_macro_p95={best_macro_p95:.6f} out={output_dir}")


if __name__ == "__main__":
    main()

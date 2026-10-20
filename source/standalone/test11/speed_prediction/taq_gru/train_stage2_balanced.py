"""Train a robust TAQ-GRU on a controlled ideal/Stage-2 data mixture.

Sampling is hierarchical rather than proportional to the number of files:

1. ``ideal_fraction`` controls ideal versus Stage-2 collection mass.
2. ``ideal_sd_share`` controls SS versus SD inside the ideal domain; Stage-2
   SS and SD receive equal mass.
3. Profiles receive equal mass inside each Stage-2 scenario.

Normalization uses the same mixture weights.  Validation is reported
separately for ideal and Stage-2 data and also for every scenario/profile, so
a large group cannot hide a regression in a smaller group.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.taq_gru"

from .taq_velocity_training import (
    evaluate_taq_model,
    json_ready_metrics,
    masked_taq_velocity_loss,
    move_taq_batch,
)
from ..common.time_aware_velocity_data import (
    DEFAULT_BASELINE_VERSION,
    SUPPORTED_BASELINE_VERSIONS,
    CONTINUOUS_FEATURE_DIM,
    DEFAULT_FINITE_DIFFERENCE_CLIP_MPS,
    DEFAULT_NORMALIZED_FEATURE_CLIP,
    ROBUST_INPUT_TRANSFORM_VERSION,
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION,
    build_taq_dataset,
    collate_taq_velocity_batch,
    compute_taq_feature_norm_stats,
)
from .time_aware_velocity_model import TimeAwareQualityGatedGRU, count_trainable_parameters


@dataclass(frozen=True)
class DataGroup:
    key: str
    domain: str
    scenario: str
    profile: str
    dataset: Dataset
    target_mass: float
    paths: tuple[Path, ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_env_spec(spec: str) -> tuple[int, ...]:
    """Parse comma-separated environment IDs and inclusive ranges."""
    values: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", maxsplit=1)
            start, end = int(raw_start), int(raw_end)
            if start < 0 or end < start:
                raise ValueError(f"Invalid environment range {part!r}")
            values.update(range(start, end + 1))
        else:
            value = int(part)
            if value < 0:
                raise ValueError("Environment IDs must be non-negative")
            values.add(value)
    if not values:
        raise ValueError("Environment specification cannot be empty")
    return tuple(sorted(values))


def _env_paths(root: Path, env_ids: Iterable[int]) -> tuple[Path, ...]:
    paths = tuple(root / "episodes" / f"env_{env_id:03d}" for env_id in env_ids)
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing episode environment directories: " + ", ".join(map(str, missing)))
    return paths


def _make_group(
    *,
    key: str,
    domain: str,
    scenario: str,
    profile: str,
    paths: tuple[Path, ...],
    target_mass: float,
) -> DataGroup:
    dataset = build_taq_dataset(paths)
    if len(dataset) < 1:
        raise ValueError(f"Data group {key!r} is empty")
    return DataGroup(
        key=key,
        domain=domain,
        scenario=scenario,
        profile=profile,
        dataset=dataset,
        target_mass=float(target_mass),
        paths=paths,
    )


def discover_groups(
    *,
    ideal_ss_root: Path,
    ideal_sd_root: Path,
    stage2_root: Path,
    ideal_env_ids: tuple[int, ...],
    stage2_env_ids: tuple[int, ...],
    ideal_fraction: float,
    ideal_sd_share: float = 0.5,
) -> list[DataGroup]:
    """Discover ideal and Stage-2 groups with hierarchical target masses."""
    if not 0.0 < ideal_fraction < 1.0:
        raise ValueError("ideal_fraction must be strictly between 0 and 1")
    if not 0.0 < ideal_sd_share < 1.0:
        raise ValueError("ideal_sd_share must be strictly between 0 and 1")
    groups = [
        _make_group(
            key="ideal/ss",
            domain="ideal",
            scenario="ss",
            profile="ideal",
            paths=_env_paths(ideal_ss_root, ideal_env_ids),
            target_mass=ideal_fraction * (1.0 - ideal_sd_share),
        ),
        _make_group(
            key="ideal/sd",
            domain="ideal",
            scenario="sd",
            profile="ideal",
            paths=_env_paths(ideal_sd_root, ideal_env_ids),
            target_mass=ideal_fraction * ideal_sd_share,
        ),
    ]

    stage2_mass = 1.0 - ideal_fraction
    for scenario in ("ss", "sd"):
        scenario_root = stage2_root / scenario
        if not scenario_root.is_dir():
            raise FileNotFoundError(f"Missing Stage-2 scenario directory: {scenario_root}")
        profile_roots = sorted(
            path for path in scenario_root.iterdir() if path.is_dir() and (path / "episodes").is_dir()
        )
        if not profile_roots:
            raise FileNotFoundError(f"No Stage-2 profiles found below: {scenario_root}")
        profile_mass = stage2_mass / 2.0 / len(profile_roots)
        for profile_root in profile_roots:
            groups.append(
                _make_group(
                    key=f"stage2/{scenario}/{profile_root.name}",
                    domain="stage2",
                    scenario=scenario,
                    profile=profile_root.name,
                    paths=_env_paths(profile_root, stage2_env_ids),
                    target_mass=profile_mass,
                )
            )
    mass_sum = sum(group.target_mass for group in groups)
    if abs(mass_sum - 1.0) > 1e-6:
        raise RuntimeError(f"Sampling masses sum to {mass_sum}, expected 1")
    return groups


def build_balanced_sampler(
    groups: list[DataGroup],
    *,
    samples_per_epoch: int,
    seed: int,
) -> tuple[ConcatDataset, WeightedRandomSampler]:
    if samples_per_epoch < 1:
        raise ValueError("samples_per_epoch must be positive")
    dataset = ConcatDataset([group.dataset for group in groups])
    weights = torch.cat(
        [
            torch.full(
                (len(group.dataset),),
                group.target_mass / len(group.dataset),
                dtype=torch.float64,
            )
            for group in groups
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )
    return dataset, sampler


def compute_balanced_norm_stats(
    groups: list[DataGroup],
    *,
    finite_difference_clip_mps: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine per-group moments according to the training sampler masses."""
    mean = torch.zeros(CONTINUOUS_FEATURE_DIM, dtype=torch.float64)
    second_moment = torch.zeros_like(mean)
    for group in groups:
        group_mean, group_std = compute_taq_feature_norm_stats(
            group.dataset,
            eps=eps,
            finite_difference_clip_mps=finite_difference_clip_mps,
        )
        group_mean = group_mean.to(torch.float64)
        group_std = group_std.to(torch.float64)
        mean += group.target_mass * group_mean
        second_moment += group.target_mass * (group_std.square() + group_mean.square())
    variance = (second_moment - mean.square()).clamp_min(0.0)
    return mean.to(torch.float32), variance.sqrt().clamp_min(eps).to(torch.float32)


def _domain_macro_metrics(
    metrics_by_group: dict[str, dict[str, float]],
    groups: list[DataGroup],
    domain: str,
) -> dict[str, float]:
    selected = [group for group in groups if group.domain == domain]
    domain_mass = sum(group.target_mass for group in selected)
    if not selected or domain_mass <= 0.0:
        raise ValueError(f"No validation groups for domain {domain!r}")
    metric_names = metrics_by_group[selected[0].key].keys()
    return {
        name: sum(
            group.target_mass / domain_mass * metrics_by_group[group.key][name]
            for group in selected
        )
        for name in metric_names
        if name != "valid_steps"
    }


@torch.no_grad()
def evaluate_validation_groups(
    model: torch.nn.Module,
    groups: list[DataGroup],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    huber_beta: float,
    uncertainty_weight: float,
    speed_norm_weight: float,
    ideal_fraction: float,
) -> dict[str, Any]:
    metrics_by_group: dict[str, dict[str, float]] = {}
    for group in groups:
        loader = DataLoader(
            group.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_taq_velocity_batch,
        )
        metrics_by_group[group.key] = json_ready_metrics(
            evaluate_taq_model(
                model,
                loader,
                device,
                huber_beta=huber_beta,
                uncertainty_weight=uncertainty_weight,
                speed_norm_weight=speed_norm_weight,
            )
        )
    ideal = _domain_macro_metrics(metrics_by_group, groups, "ideal")
    stage2 = _domain_macro_metrics(metrics_by_group, groups, "stage2")
    composite_loss = ideal_fraction * ideal["loss"] + (1.0 - ideal_fraction) * stage2["loss"]
    worst_group_key = max(metrics_by_group, key=lambda key: metrics_by_group[key]["err_norm_p95"])
    return {
        "composite_loss": composite_loss,
        "ideal_macro": ideal,
        "stage2_macro": stage2,
        "worst_group": worst_group_key,
        "worst_group_p95": metrics_by_group[worst_group_key]["err_norm_p95"],
        "groups": metrics_by_group,
    }


def _group_manifest(groups: list[DataGroup]) -> list[dict[str, Any]]:
    return [
        {
            "key": group.key,
            "domain": group.domain,
            "scenario": group.scenario,
            "profile": group.profile,
            "sample_count": len(group.dataset),
            "target_mass": group.target_mass,
            "paths": [str(path) for path in group.paths],
        }
        for group in groups
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train robust TAQ-GRU with hierarchical ideal/Stage-2 balanced sampling."
    )
    parser.add_argument("--ideal_ss_root", type=Path, required=True)
    parser.add_argument("--ideal_sd_root", type=Path, required=True)
    parser.add_argument("--stage2_root", type=Path, required=True)
    parser.add_argument("--ideal_train_envs", default="0-12")
    parser.add_argument("--ideal_val_envs", default="13-14")
    parser.add_argument("--stage2_train_envs", default="0")
    parser.add_argument("--stage2_val_envs", default="1")
    parser.add_argument("--ideal_fraction", type=float, default=0.45)
    parser.add_argument(
        "--ideal_sd_share",
        type=float,
        default=0.5,
        help="Fraction of the ideal sampling mass assigned to SD; the remainder goes to SS.",
    )
    parser.add_argument("--samples_per_epoch", type=int, default=8192)
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("source/standalone/test11/speed_prediction/gru_runs/stage2_balanced"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=15)
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
    )
    parser.add_argument(
        "--normalized_feature_clip",
        type=float,
        default=DEFAULT_NORMALIZED_FEATURE_CLIP,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.samples_per_epoch < 1:
        raise ValueError("batch_size, epochs and samples_per_epoch must be positive")
    if args.num_workers < 0 or args.early_stop_patience < 1:
        raise ValueError("num_workers must be non-negative and patience must be positive")
    if args.huber_beta <= 0.0 or args.finite_difference_clip_mps <= 0.0:
        raise ValueError("huber_beta and finite_difference_clip_mps must be positive")
    if args.normalized_feature_clip <= 0.0:
        raise ValueError("normalized_feature_clip must be positive")

    ideal_train_envs = parse_env_spec(args.ideal_train_envs)
    ideal_val_envs = parse_env_spec(args.ideal_val_envs)
    stage2_train_envs = parse_env_spec(args.stage2_train_envs)
    stage2_val_envs = parse_env_spec(args.stage2_val_envs)
    if set(ideal_train_envs) & set(ideal_val_envs):
        raise ValueError("Ideal train and validation environment IDs overlap")
    if set(stage2_train_envs) & set(stage2_val_envs):
        raise ValueError("Stage-2 train and validation environment IDs overlap")

    set_seed(args.seed)
    train_groups = discover_groups(
        ideal_ss_root=args.ideal_ss_root.expanduser(),
        ideal_sd_root=args.ideal_sd_root.expanduser(),
        stage2_root=args.stage2_root.expanduser(),
        ideal_env_ids=ideal_train_envs,
        stage2_env_ids=stage2_train_envs,
        ideal_fraction=args.ideal_fraction,
        ideal_sd_share=args.ideal_sd_share,
    )
    val_groups = discover_groups(
        ideal_ss_root=args.ideal_ss_root.expanduser(),
        ideal_sd_root=args.ideal_sd_root.expanduser(),
        stage2_root=args.stage2_root.expanduser(),
        ideal_env_ids=ideal_val_envs,
        stage2_env_ids=stage2_val_envs,
        ideal_fraction=args.ideal_fraction,
        ideal_sd_share=args.ideal_sd_share,
    )
    train_dataset, train_sampler = build_balanced_sampler(
        train_groups,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )
    input_mean, input_std = compute_balanced_norm_stats(
        train_groups,
        finite_difference_clip_mps=args.finite_difference_clip_mps,
    )

    requested_device = args.device
    device = torch.device(
        requested_device
        if "cuda" not in requested_device or torch.cuda.is_available()
        else "cpu"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_taq_velocity_batch,
    )
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
        f"[data] schema={TAQ_SCHEMA_VERSION} raw_train_samples={len(train_dataset)} "
        f"sampled_per_epoch={args.samples_per_epoch} ideal_fraction={args.ideal_fraction:.3f} "
        f"ideal_sd_share={args.ideal_sd_share:.3f}"
    )
    for group in train_groups:
        print(
            f"[train-group] {group.key:24s} samples={len(group.dataset):6d} "
            f"target_mass={group.target_mass:.6f} expected_per_epoch={group.target_mass * args.samples_per_epoch:.1f}"
        )
    for group in val_groups:
        print(f"[val-group]   {group.key:24s} samples={len(group.dataset):6d}")
    print("[norm] mean=", [round(float(value), 6) for value in input_mean])
    print("[norm] std =", [round(float(value), 6) for value in input_std])
    print(
        f"[model] parameters={count_trainable_parameters(model)} "
        f"baseline={model.baseline_version} "
        f"input_transform={model.input_transform_version} "
        f"fd_clip={model.finite_difference_clip_mps:g}m/s "
        f"normalized_clip=+/-{model.normalized_feature_clip:g} device={device}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    output_dir = args.save_dir.expanduser() / datetime.now().strftime("%m-%d-%H-%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": TAQ_SCHEMA_VERSION,
        "label_semantics": TAQ_LABEL_SEMANTICS,
        "baseline_version": model.baseline_version,
        "training_recipe": "stage2_hierarchical_balanced_v1",
        "ideal_fraction": args.ideal_fraction,
        "ideal_sd_share": args.ideal_sd_share,
        "samples_per_epoch": args.samples_per_epoch,
        "ideal_train_envs": ideal_train_envs,
        "ideal_val_envs": ideal_val_envs,
        "stage2_train_envs": stage2_train_envs,
        "stage2_val_envs": stage2_val_envs,
        "train_groups": _group_manifest(train_groups),
        "val_groups": _group_manifest(val_groups),
    }
    (output_dir / "data_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        target_steps = 0
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
            count = int(batch["target_valid_mask"].sum().item())
            loss_sum += float(loss.item()) * count
            target_steps += count
        train_loss = loss_sum / max(target_steps, 1)

        validation = evaluate_validation_groups(
            model,
            val_groups,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            huber_beta=args.huber_beta,
            uncertainty_weight=args.uncertainty_weight,
            speed_norm_weight=args.speed_norm_weight,
            ideal_fraction=args.ideal_fraction,
        )
        composite_loss = float(validation["composite_loss"])
        scheduler.step(composite_loss)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **validation,
        }
        history.append(record)
        ideal = validation["ideal_macro"]
        stage2 = validation["stage2_macro"]
        print(
            f"[epoch {epoch:03d}] train={train_loss:.6f} composite={composite_loss:.6f} "
            f"ideal_p95={ideal['err_norm_p95']:.4f} stage2_p95={stage2['err_norm_p95']:.4f} "
            f"stage2_mae=({stage2['mae_x']:.4f},{stage2['mae_y']:.4f},{stage2['mae_z']:.4f}) "
            f"worst={validation['worst_group']}:{validation['worst_group_p95']:.4f} "
            f"lr={record['lr']:.2e}"
        )

        if composite_loss < best_loss:
            best_loss = composite_loss
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                {
                    "schema_version": TAQ_SCHEMA_VERSION,
                    "label_semantics": TAQ_LABEL_SEMANTICS,
                    "baseline_version": model.baseline_version,
                    "training_recipe": "stage2_hierarchical_balanced_v1",
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config,
                    "input_norm": {"mean": input_mean.cpu(), "std": input_std.cpu()},
                    "train_args": vars(args),
                    "data_manifest": run_manifest,
                    "best_epoch": best_epoch,
                    "best_val_metrics": validation,
                },
                output_dir / "best_model.pt",
            )
        else:
            no_improvement += 1
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if no_improvement >= args.early_stop_patience:
            print(f"[early-stop] no composite validation improvement for {no_improvement} epochs")
            break

    summary = {
        "schema_version": TAQ_SCHEMA_VERSION,
        "label_semantics": TAQ_LABEL_SEMANTICS,
        "baseline_version": model.baseline_version,
        "training_recipe": "stage2_hierarchical_balanced_v1",
        "best_epoch": best_epoch,
        "best_composite_val_loss": best_loss,
        "model_parameters": count_trainable_parameters(model),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] best_epoch={best_epoch} best_composite={best_loss:.6f} out={output_dir}")


if __name__ == "__main__":
    main()

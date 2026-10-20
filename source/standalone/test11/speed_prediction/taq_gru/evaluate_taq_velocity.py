"""Evaluate a TAQ-GRU checkpoint on schema-v2 velocity chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.taq_gru"

from ..common.time_aware_velocity_data import (
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION,
    build_taq_dataset,
    collate_taq_velocity_batch,
    load_torch_payload,
    taq_model_config_from_checkpoint,
)
from .time_aware_velocity_model import TimeAwareQualityGatedGRU, count_trainable_parameters
from .taq_velocity_training import evaluate_taq_model, json_ready_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TAQ-GRU velocity checkpoint")
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    payload = load_torch_payload(args.checkpoint)
    if payload.get("schema_version") != TAQ_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema={payload.get('schema_version')!r}; expected {TAQ_SCHEMA_VERSION!r}"
        )
    if payload.get("label_semantics") != TAQ_LABEL_SEMANTICS:
        raise ValueError("Checkpoint has missing or incompatible TAQ v2 label semantics")
    model_config = taq_model_config_from_checkpoint(payload)
    model = TimeAwareQualityGatedGRU(**model_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    norm = payload.get("input_norm")
    if norm is None:
        raise KeyError("TAQ checkpoint is missing input_norm")
    model.set_input_norm_stats(norm["mean"], norm["std"])

    device = torch.device(args.device if "cuda" not in args.device or torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = build_taq_dataset(args.data_dirs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_taq_velocity_batch,
    )
    train_args = payload.get("train_args", {})
    metrics = evaluate_taq_model(
        model,
        loader,
        device,
        huber_beta=float(train_args.get("huber_beta", 0.1)),
        uncertainty_weight=float(train_args.get("uncertainty_weight", 0.10)),
        speed_norm_weight=float(train_args.get("speed_norm_weight", 0.05)),
    )
    result = {
        "schema_version": TAQ_SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "sample_count": len(dataset),
        "model_parameters": count_trainable_parameters(model),
        **json_ready_metrics(metrics),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

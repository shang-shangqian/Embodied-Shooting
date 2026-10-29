"""Train unchanged TAQ-v2 on complete episodes with full BPTT and held-out scoring."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time
from zoneinfo import ZoneInfo

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.taq_gru"

from ..common.time_aware_velocity_data import (
    DEFAULT_BASELINE_VERSION, SUPPORTED_BASELINE_VERSIONS,
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION, TAQ_YAW_RATE_SEMANTICS, load_torch_payload,
    robust_finite_difference,
    taq_model_config_from_checkpoint,
)
from ..legacy_gru.train_gru_velocity_episodes import FrameMetrics
from .evaluate_taq_velocity_streaming import _segmented_baseline, _stream_tensors
from .taq_velocity_training import masked_taq_velocity_loss
from .time_aware_velocity_model import TimeAwareQualityGatedGRU


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_episode(path, model):
    payload = load_torch_payload(path)
    metadata = payload.get("metadata", {})
    if metadata.get("episode_complete") is not True:
        raise ValueError(f"Incomplete episode: {path}")
    if metadata.get("angular_velocity_z_semantics") != TAQ_YAW_RATE_SEMANTICS:
        raise ValueError(f"Incompatible yaw-rate semantics: {path}")
    if (metadata.get("validation_gt_direction") or metadata.get("validation_gt_range")
            or metadata.get("gt_visibility_gate_after_initialization")
            or metadata.get("measurement_gt_free_after_initialization") is not True
            or metadata.get("condition", metadata.get("validation_condition")) in {"A", "C", "P1", "P2", "P3"}
            or metadata.get("range_source") == "ground_truth_validation"):
        raise ValueError(f"GT/oracle measurements are forbidden: {path}")
    x, sources, valid, target, supervised, reset, timestamps = _stream_tensors(payload, Path(path))
    if not valid.all():
        raise ValueError(f"Complete episode stream must not contain padding: {path}")
    if not torch.isfinite(x).all() or not torch.isfinite(target).all():
        raise ValueError(f"Nonfinite features or labels: {path}")
    if torch.any((sources < 0) | (sources > 4)):
        raise ValueError(f"Unsupported source ID: {path}")
    direct = (sources <= 2) & (x[:, 22] > .5) & (x[:, 23] > .5)
    available = direct.nonzero().flatten()
    if not len(available):
        raise ValueError(f"No direct observation for initialization: {path}")
    start = int(available[0])
    x, sources, valid, target, supervised, reset, direct = (
        v[start:].clone() for v in (x, sources, valid, target, supervised, reset, direct))
    reset[0] = True
    # Score exactly the same time steps as legacy first-valid initialization:
    # its first velocity prediction needs two positions.
    supervised[0] = False
    if not supervised.any():
        raise ValueError(f"No supervised frames: {path}")
    baseline = _segmented_baseline(x, valid, reset, model=model, window_s=model.baseline_window_s)
    return dict(features=x, source_ids=sources, valid_mask=valid, target=target,
                target_valid_mask=supervised, reset_mask=reset, direct=direct,
                baseline=baseline, record=dict(path=str(Path(path).resolve()),
                episode_uid=metadata["episode_uid"], scenario=metadata["scenario"],
                recorded_frames=len(timestamps), input_frames=len(x),
                supervised_frames=int(supervised.sum()), resets=int(reset.sum())))


class EpisodeDataset(Dataset):
    def __init__(self, paths, model):
        self.episodes = [read_episode(p, model) for p in paths]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        return self.episodes[index]

    def counts(self):
        return dict(episodes=len(self), **{key: sum(e["record"][key] for e in self.episodes)
                    for key in ("recorded_frames", "input_frames", "supervised_frames", "resets")})


def collate_episodes(episodes):
    return {key: pad_sequence([e[key] for e in episodes], batch_first=True)
            for key in ("features", "source_ids", "valid_mask", "target",
                        "target_valid_mask", "reset_mask", "direct", "baseline")}


def input_norm(dataset, model):
    values = torch.cat([e["features"] for e in dataset.episodes]).double()
    clipped, _ = robust_finite_difference(values, clip_mps=model.finite_difference_clip_mps)
    values[:, 3:6] = clipped
    return values.mean(0).float(), values.std(0, unbiased=False).clamp_min(1e-6).float()


def forward_episode(model, batch):
    """Batch independent encoders/heads; keep the original causal GRU recurrence.

    No hidden detaches or optimizer updates occur within an episode. Stored
    tracker resets clear the hidden state, matching TAQ streaming evaluation.
    """
    x = batch["features"]
    b, t, f = x.shape
    encoded, _, dt = model._encode_step(x.reshape(b*t, f), batch["source_ids"].reshape(-1))
    encoded = encoded.reshape(b, t, -1)
    dt = dt.reshape(b, t)
    tau = torch.nn.functional.softplus(model.log_time_constant).to(x.dtype) + 1e-3
    decay = torch.exp(-dt.unsqueeze(-1) / tau)
    hidden = model.initial_hidden(b, device=x.device, dtype=x.dtype)
    states = []
    for index in range(t):
        hidden = torch.where(batch["reset_mask"][:, index, None], torch.zeros_like(hidden), hidden)
        candidate = model.gru_cell(encoded[:, index], hidden * decay[:, index])
        hidden = torch.where(batch["valid_mask"][:, index, None], candidate, hidden)
        states.append(hidden)
    head = model.head(torch.stack(states, dim=1))
    return (batch["baseline"] + model.velocity_delta_head(head),
            model.log_sigma_head(head).clamp(-5, 2))


def run_epoch(model, loader, device, optimizer=None, baseline_only=False):
    model.train(optimizer is not None)
    metrics = FrameMetrics()
    with torch.set_grad_enabled(optimizer is not None):
        for raw in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if baseline_only:
                mean, sigma = batch["baseline"], torch.zeros_like(batch["baseline"])
            else:
                mean, sigma = forward_episode(model, batch)
            mask = batch["target_valid_mask"]
            loss, _ = masked_taq_velocity_loss(mean, sigma, batch["target"], mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite complete-episode loss")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            metrics.add(mean[mask], batch["target"][mask], batch["direct"][mask], loss)
    return metrics.result()


def load_reference_splits(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    paths = {split: [Path(p).resolve() for p in manifest["files"][split]]
             for split in ("train", "val", "test")}
    if any(not value for value in paths.values()):
        raise ValueError("Reference manifest requires nonempty train/val/test")
    seen_paths, seen_uids = set(), set()
    for split, files in paths.items():
        for path in files:
            md = load_torch_payload(path)["metadata"]
            uid = md["episode_uid"]
            if path in seen_paths or uid in seen_uids:
                raise ValueError(f"Duplicate or cross-split episode: {path}")
            seen_paths.add(path)
            seen_uids.add(uid)
    return paths


def load_taq_checkpoint(path):
    payload = load_torch_payload(path)
    if (payload.get("schema_version") != TAQ_SCHEMA_VERSION
            or payload.get("label_semantics") != TAQ_LABEL_SEMANTICS):
        raise ValueError("Incompatible TAQ reference checkpoint")
    config = taq_model_config_from_checkpoint(payload)
    model = TimeAwareQualityGatedGRU(**config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_manifest", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path,
                        default=Path(__file__).resolve().parent / "full_episode_runs")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--old_taq_checkpoint", type=Path)
    parser.add_argument("--baseline_version", choices=SUPPORTED_BASELINE_VERSIONS, default=DEFAULT_BASELINE_VERSION)
    parser.add_argument("--baseline_window_s", type=float, default=0.48)
    args = parser.parse_args()
    if min(args.epochs, args.early_stop_patience, args.batch_size) < 1 or args.lr <= 0:
        parser.error("Training dimensions and learning rate must be positive")
    if args.baseline_window_s <= 0:
        parser.error("baseline_window_s must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = TimeAwareQualityGatedGRU(baseline_version=args.baseline_version,
                                    baseline_window_s=args.baseline_window_s)
    paths = load_reference_splits(args.reference_manifest)
    datasets = {split: EpisodeDataset(files, model) for split, files in paths.items()}
    mean, std = input_norm(datasets["train"], model)
    model.set_input_norm_stats(mean, std)
    counts = {split: ds.counts() for split, ds in datasets.items()}
    output = args.save_dir / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(context="full_episode", baseline_version=model.baseline_version,
                    baseline_window_s=model.baseline_window_s, backpropagation="full_episode_bptt",
                    initialization="first_valid_direct_observation", hidden_reset="stored_TAQ_state_reset_mask",
                    selection_metric="validation_err_norm_mean", normalization="training_episodes_only",
                    supervision="stored_target_valid_mask_excluding_first_position_frame",
                    sampling="uniform_episode_shuffle", initialization_weights="random_from_scratch",
                    loss="TAQ_default_Huber_0.1_plus_0.1_NLL_plus_0.05_speed_norm",
                    test_usage="after_validation_selection_only")
    write_json(output / "data_manifest.json", dict(reference_manifest=str(args.reference_manifest.resolve()),
        schema_version=TAQ_SCHEMA_VERSION, baseline_version=model.baseline_version, counts=counts, protocol=protocol,
        files={k: [str(p) for p in v] for k, v in paths.items()}))
    write_json(output / "run_config.json", dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                              protocol=protocol, model_config=model.config))
    print(f"[output] {output.resolve()}", flush=True)
    print(f"[dataset] {counts}", flush=True)
    print(f"[baseline] version={model.baseline_version} window_s={model.baseline_window_s}", flush=True)
    if args.prepare_only:
        print("[prepare-only] Complete episodes, GT guardrails and split isolation checked.", flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; request access to the GPU outside the sandbox")
    model.to(device)
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split=="train",
                                collate_fn=collate_episodes, num_workers=0,
                                pin_memory=device.type=="cuda") for split,ds in datasets.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=4)
    baseline_val = run_epoch(model, loaders["val"], device, baseline_only=True)
    write_json(output / "baseline_val_metrics.json", baseline_val)
    print(f"[baseline val] mean={baseline_val['err_norm_mean']:.5f} p95={baseline_val['err_norm_p95']:.5f}", flush=True)
    history, best, best_epoch = [], float("inf"), 0
    started = time.monotonic()
    for epoch in range(1, args.epochs+1):
        training = run_epoch(model, loaders["train"], device, optimizer)
        validation = run_epoch(model, loaders["val"], device)
        score = validation["err_norm_mean"]
        scheduler.step(score)
        history.append(dict(epoch=epoch, train=training, val=validation,
                            lr=optimizer.param_groups[0]["lr"], elapsed_s=time.monotonic()-started))
        write_json(output / "history.json", history)
        payload = dict(schema_version=TAQ_SCHEMA_VERSION, label_semantics=TAQ_LABEL_SEMANTICS,
            baseline_version=model.baseline_version, model_state_dict=model.state_dict(), model_config=model.config,
            input_norm=dict(mean=mean, std=std), train_args=vars(args), training_protocol=protocol,
            best_epoch=epoch, best_val_metrics=validation)
        torch.save(payload, output / "last.pt")
        if score < best:
            best, best_epoch = score, epoch
            torch.save(payload, output / "best_model.pt")
        print(f"[epoch {epoch:03d}] train_mean={training['err_norm_mean']:.5f} val_mean={score:.5f} "
              f"val_p95={validation['err_norm_p95']:.5f} best_epoch={best_epoch} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if epoch-best_epoch >= args.early_stop_patience:
            print("[early-stop] Validation mean did not improve.", flush=True)
            break
    model.load_state_dict(load_torch_payload(output/"best_model.pt")["model_state_dict"])
    test = run_epoch(model, loaders["test"], device)
    baseline_test = run_epoch(model, loaders["test"], device, baseline_only=True)
    write_json(output / "test_metrics.json", test)
    comparison = {"retrained_taq":test, "physical_baseline":baseline_test}
    if args.old_taq_checkpoint:
        old = load_taq_checkpoint(args.old_taq_checkpoint).to(device)
        old_ds = EpisodeDataset(paths["test"], old)
        comparison["old_taq"] = run_epoch(old, DataLoader(old_ds, batch_size=args.batch_size,
            collate_fn=collate_episodes), device)
    write_json(output / "comparison.json", comparison)
    write_json(output / "summary.json", dict(best_epoch=best_epoch, best_val_err_norm_mean=best,
        counts=counts, protocol=protocol, test_metrics=test, elapsed_s=time.monotonic()-started))
    print(f"[held-out test] {json.dumps(comparison, ensure_ascii=False)}", flush=True)
    print(f"[done] {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

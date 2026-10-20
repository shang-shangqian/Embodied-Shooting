"""Compare initialization policies on identical saved visual episode streams."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "speed_prediction.legacy_gru"

from ..common.time_aware_velocity_data import (
    FeatureIndex, TAQ_LABEL_SEMANTICS, TAQ_SCHEMA_VERSION, TAQ_STREAM_FORMAT, load_torch_payload,
)
from .evaluate_vision_gru import DEFAULT_CHECKPOINT, VisionGRUMetrics, velocity_metrics
from .online_gru_velocity import load_legacy_checkpoint


def replay_stream(observer, path, episode_id):
    payload = load_torch_payload(path)
    if (payload.get("schema_version") != TAQ_SCHEMA_VERSION
            or payload.get("stream_format") != TAQ_STREAM_FORMAT
            or payload.get("label_semantics") != TAQ_LABEL_SEMANTICS):
        raise ValueError(f"Incompatible visual stream: {path}")
    features = payload["sequence_features"].cpu().numpy()
    targets = payload["target_velocity_base"].cpu().numpy()
    timestamps = payload["timestamps_s"].cpu().numpy()
    sources = payload["sequence_source_ids"].cpu().numpy()
    if (features.shape != (len(timestamps), 24) or targets.shape != (len(timestamps), 3)
            or sources.shape != timestamps.shape):
        raise ValueError(f"Invalid stream shape: {path}")
    audit_path = path.parent / "velocity_error_audit.jsonl"
    audits = ([json.loads(line) for line in audit_path.read_text().splitlines()]
              if audit_path.exists() else [{} for _ in timestamps])
    if len(audits) != len(timestamps):
        raise ValueError(f"Stream/audit length mismatch: {path}")
    for index, timestamp in enumerate(timestamps):
        observer.observe(SimpleNamespace(
            timestamp_s=float(timestamp), position_base_m=features[index, :3],
            robot_linear_velocity_base_mps=features[index, 6:9],
            robot_angular_velocity_base_rps=features[index, 9:12],
            source_id=int(sources[index]), range_valid=bool(features[index, FeatureIndex.RANGE_VALID_MASK]),
            target_relative_velocity_base_mps=targets[index], audit=audits[index],
        ), env_id=int(payload.get("metadata", {}).get("env_id", 0)),
            episode_id=episode_id, metadata=payload.get("metadata", {}))


def matched_metrics(output, summaries):
    policy_rows = {}
    for policy in summaries:
        with (output / policy / "frame_metrics.csv").open() as handle:
            policy_rows[policy] = {
                (row["env_id"], row["episode_id"], row["timestamp_s"], row["context"]): row
                for row in csv.DictReader(handle) if row["scored"] == "True"
            }
    common = policy_rows["first_valid"].keys() & policy_rows["zero_filled"].keys()
    timing = {(str(row["env_id"]), str(row["episode_id"])): row["first_direct_s"]
              for row in summaries["first_valid"]["episode_timing"]}
    metrics = []
    for context in ("rolling", "full"):
        for group in ("all", "startup_0p5s", "after_startup"):
            keys = []
            for key in sorted(common):
                if key[3] != context:
                    continue
                age = float(key[2]) - timing[key[:2]]
                if group == "all" or (group == "startup_0p5s" and age < .5) or (group == "after_startup" and age >= .5):
                    keys.append(key)
            for policy, rows in policy_rows.items():
                errors = [[float(rows[key][f"prediction_{axis}"]) - float(rows[key][f"target_{axis}"])
                           for axis in ("vx", "vy", "vz")] for key in keys]
                metrics.append(dict(policy=policy, context=context, group=group, **velocity_metrics(errors)))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    streams = sorted(args.data_root.rglob("*.taqstream"))
    if not streams:
        raise FileNotFoundError(f"No episode streams in {args.data_root}")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, checkpoint = load_legacy_checkpoint(args.checkpoint)
    summaries = {}
    for policy in ("zero_filled", "first_valid"):
        output = args.out_dir / policy
        output.mkdir()
        settings = SimpleNamespace(
            context="both", device=args.device, window_s=.48, max_history_frames=32,
            measure_latency=False, burn_in_s=.1, progress_every=0, initialization_policy=policy,
            scenario="saved_stream", profile="saved_stream", condition="saved_stream", seed=None,
            locomotion_vx_mps=0., locomotion_vy_mps=0., locomotion_yaw_rate_radps=0.,
        )
        observer = VisionGRUMetrics(model, checkpoint, settings, output)
        completed = False
        try:
            for episode_id, stream in enumerate(streams):
                replay_stream(observer, stream, episode_id)
            completed = True
        finally:
            summaries[policy] = observer.finish(completed, args.checkpoint)
    metrics = matched_metrics(args.out_dir, summaries)
    comparison = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=summaries["first_valid"]["checkpoint_sha256"],
        device=args.device, streams=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                    for path in streams],
        protocol="Same observations, checkpoint and scored timestamps; exclude only frames without predictions in either policy; no added post-direct burn-in",
        matched_metrics=metrics,
    )
    (args.out_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    lines = ["# GRU 初始化同流对照", "", comparison["protocol"], "",
             "误差单位 m/s。首次有效帧仅建立历史，因此 first_valid 比旧策略少一个可评分帧；下表使用共同评分帧。",
             "", "| 策略 | 上下文 | 分组 | 共同帧数 | Mean | P95 | P99 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in metrics:
        if row["n"]:
            lines.append(f"| {row['policy']} | {row['context']} | {row['group']} | {row['n']} | "
                         f"{row['mean']:.4f} | {row['p95']:.4f} | {row['p99']:.4f} |")
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Comparison: {args.out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()

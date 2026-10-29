"""Compare frozen GRUs on identical causal raw-input histories and test frames."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization_components import ExperimentalGRU, WIDTHS, fit_baseline
from train_flow_ablations import gru_segment, write_csv
from speed_prediction.common.time_aware_velocity_data import (
    TAQ_LABEL_SEMANTICS, load_torch_payload, weighted_velocity_baseline,
)
from speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU
from speed_prediction.legacy_gru.taq_dataset import causal_smooth, chunk_identity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_inputs(features, smooth_window):
    position = causal_smooth(features[:, :3], smooth_window)
    moving = (features[1:, 6:8].norm(dim=-1) > 0.05).float()[:, None]
    return torch.cat((position[1:], position[1:] - position[:-1],
                      features[1:, 6:12], moving), dim=-1)


def history_tasks(records, context, width):
    tasks = []
    for record_index, record in enumerate(records):
        if context == "full_episode":
            tasks.append((record_index, 0, len(record["x"])))
            continue
        timestamps = record["time"].numpy()
        for endpoint in torch.nonzero(record["score"]).flatten().tolist():
            start = int(np.searchsorted(timestamps, timestamps[endpoint] - width - 1e-9))
            if endpoint - start < 1:
                raise ValueError("A scored endpoint has no preceding position for the legacy GRU")
            tasks.append((record_index, start, endpoint + 1))
    return tasks


def prepare_batch(records, tasks):
    sequences = []
    sources = []
    times = []
    for record_index, start, end in tasks:
        record = records[record_index]
        features = record["x"][start:end].clone()
        features[0, 3:6] = 0
        features[0, 12] = 0
        sequences.append(features)
        sources.append(record["source"][start:end])
        times.append(record["time"][start:end].numpy())
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    features = pad_sequence(sequences, batch_first=True)
    valid = torch.arange(features.shape[1])[None] < lengths[:, None]
    old_baseline = weighted_velocity_baseline(features, valid, window_s=0.48)
    legacy_extras = []
    aligned_extras = []
    aligned_baselines = []
    for sequence_index, (sequence, timestamps) in enumerate(zip(sequences, times)):
        reset = np.zeros(len(sequence), dtype=bool)
        reset[0] = True
        raw_extras = []
        physical_extras = []
        physical_baselines = []
        for width in WIDTHS:
            _, extra = fit_baseline(sequence.numpy(), timestamps, reset, width,
                                    aligned=False, fd_weight=True)
            raw_extras.append(extra)
            baseline, extra = fit_baseline(sequence.numpy(), timestamps, reset, width,
                                           aligned=True, fd_weight=False)
            physical_extras.append(extra)
            physical_baselines.append(baseline)
        raw_extras[-1][:, :3] = old_baseline[sequence_index, :len(sequence)].numpy() / 10
        legacy_extras.append(torch.from_numpy(np.concatenate(raw_extras, axis=1)))
        aligned_extras.append(torch.from_numpy(np.concatenate(physical_extras, axis=1)))
        aligned_baselines.append(torch.from_numpy(np.stack(physical_baselines, axis=1)))
    reset = torch.zeros_like(valid)
    reset[:, 0] = True
    return dict(x=features, source=pad_sequence(sources, batch_first=True), valid=valid,
                reset=reset, lengths=lengths, old=old_baseline,
                legacy_extra=pad_sequence(legacy_extras, batch_first=True),
                aligned_extra=pad_sequence(aligned_extras, batch_first=True),
                aligned=pad_sequence(aligned_baselines, batch_first=True)), sequences


@torch.inference_mode()
def predict_batch(legacy, candidates, batch, sequences, smooth_window, device):
    inputs = [legacy_inputs(sequence, smooth_window) for sequence in sequences]
    inputs = pad_sequence(inputs, batch_first=True).to(device)
    output, _ = legacy.gru(legacy.normalize_input(inputs))
    legacy_output = legacy.head(output).cpu().numpy()
    predictions = {"legacy_gru": []}
    for sequence_index, sequence in enumerate(sequences):
        prediction = np.full((len(sequence), 3), np.nan, dtype=np.float32)
        prediction[1:] = legacy_output[sequence_index, :len(sequence) - 1]
        predictions["legacy_gru"].append(prediction)
    for name, (model, checkpoint) in candidates.items():
        aligned = checkpoint["variant_config"].get("aligned", False)
        model_batch = {key: batch[key].to(device) for key in ("x", "source", "valid", "reset")}
        model_batch["extra"] = batch["aligned_extra" if aligned else "legacy_extra"].to(device)
        model_batch["base"] = (batch["aligned"][:, :, checkpoint["scale"]]
                               if aligned else batch["old"]).to(device)
        prediction, _, _ = gru_segment(model, model_batch, 0, batch["x"].shape[1])
        prediction = prediction.cpu().numpy()
        predictions[name] = [prediction[index, :len(sequence)]
                             for index, sequence in enumerate(sequences)]
    return predictions


def error_statistics(errors):
    return dict(n=len(errors), mean=float(np.mean(errors)),
                p50=float(np.quantile(errors, 0.5)), p95=float(np.quantile(errors, 0.95)),
                p99=float(np.quantile(errors, 0.99)), max=float(np.max(errors)),
                over_0p5=float(np.mean(errors > 0.5)), over_1=float(np.mean(errors > 1)),
                over_2=float(np.mean(errors > 2)))


def summarize(records, predictions, context):
    groups = defaultdict(list)
    episodes = []
    for model_name, saved in predictions.items():
        for record, prediction in zip(records, saved):
            error = np.linalg.norm(prediction - record["target"].numpy(), axis=1)
            if not np.isfinite(error[record["score"].numpy()]).all():
                raise ValueError(f"Missing prediction on a shared scored frame: {model_name}")
            for filter_name in ("unfiltered", "center_above_ground"):
                mask = record["score"].numpy().copy()
                if filter_name == "center_above_ground":
                    mask &= record["ground"].numpy()
                if not mask.any():
                    continue
                selected = error[mask]
                episodes.append(dict(model=model_name, context=context, uid=record["uid"],
                                     scenario=record["scenario"], profile=record["profile"],
                                     filter=filter_name, **error_statistics(selected)))
                for profile in ("all", record["profile"]):
                    groups[(model_name, record["scenario"], profile, filter_name)].append(selected)
    rows = [dict(model=key[0], context=context, scenario=key[1], profile=key[2],
                 filter=key[3], **error_statistics(np.concatenate(values)))
            for key, values in groups.items()]
    reference = {(row["scenario"], row["profile"], row["filter"]): row["n"]
                 for row in rows if row["model"] == "legacy_gru"}
    for row in rows:
        if row["n"] != reference[(row["scenario"], row["profile"], row["filter"])]:
            raise ValueError("Compared frame counts differ")
    return rows, episodes


def select_candidates(runs, variants):
    selected = {}
    for variant in variants:
        options = []
        for path in sorted(runs.glob(f"{variant}_seed*/best_model.pt")):
            checkpoint = load_torch_payload(path)
            if checkpoint["variant"] != variant:
                raise ValueError("Checkpoint variant mismatch")
            options.append((float(checkpoint["metric"]), path, checkpoint))
        if not options:
            raise FileNotFoundError(f"No trained checkpoints for {variant}")
        _, path, checkpoint = min(options, key=lambda option: option[0])
        selected[variant] = (path, checkpoint)
    return selected


def write_report(path, rows, protocol):
    lines = ["# 冻结普通 GRU 与新版 TAQ-GRU：同流、同帧、同历史比较", "",
             "未重新训练；普通 GRU 来自 20261009_152810_116165。"
             "TAQ 候选各自仅按保存的验证宏 P95 选种子，不按本次测试选模。", "",
             "## 统一协议", "",
             "- rolling_0p48：每个原评分时刻重放过去至多 0.48 s 的观测，"
             "每次冷启动；所有基线和质量拟合均在该历史内部重新计算。",
             "- full_episode：同样的原始输入，状态仅在 episode 开始重置。"
             "丢失、预测和 tracker 重初始化不清除任何模型状态。",
             "- 所有模型沿用原始测量/预测/零填充位置；不额外传播缺测位置。"
             "窗口首帧 dt/FD 清零，观测年龄和质量保持原保存值。",
             "- 普通 GRU 用原训练适配：位置、米制位置增量、机器人线/角速度、"
             "运动标志；第一位置帧仅用于后续差分。TAQ 使用其原 24 维输入及 source。",
             "- 所有归一化和权重冻结；相同 score/ground 掩码及物理相对平移速度标签。"
             "Mean/P95/P99 为 pooled-frame 三维速度误差范数，单位 m/s。",
             "- 时间窗口使用原保存传感器时间；没有在本次实验中修正 mixed 时间语义。", ""]
    for context in ("rolling_0p48", "full_episode"):
        lines += [f"## {context}", "",
                  "| 模型 | 场景 | profile | 筛选 | 帧数 | Mean | P95 | P99 | >2m/s |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["context"] == context:
                lines.append(f"| {row['model']} | {row['scenario']} | {row['profile']} | "
                             f"{row['filter']} | {row['n']} | {row['mean']:.4f} | "
                             f"{row['p95']:.4f} | {row['p99']:.4f} | {row['over_2']:.2%} |")
        lines.append("")
    lines += ["## 解释边界", "",
              "这是现有冻结权重的受控迁移/推理比较，不是同训练数据、参数量和预算的架构消融。"
              "普通 GRU 在 clean 窗口末帧训练；新 TAQ 在含 mixed 的流上训练。",
              "full_episode 超出普通 GRU 原短窗训练上下文。统一的 episode-only reset 和 raw "
              "缺测策略也改变了部分 TAQ 候选原生部署设置，尤其 flow_combined_scratch "
              "原本使用 propagated inputs；该候选结果是 raw-input 控制，不是原生部署复现。",
              "全部指定候选均报告，不根据测试结果替换正式模型。没有训练种子重训、"
              "显著性检验或行走/机动泛化结论。", "",
              f"推理设备：{protocol['device']}；耗时：{protocol['elapsed_s']:.1f} s。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--variants", nargs="+", default=["flow_original", "flow_quality", "flow_combined_scratch"])
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    legacy_checkpoint = load_torch_payload(args.legacy_checkpoint)
    if legacy_checkpoint.get("label_semantics") != TAQ_LABEL_SEMANTICS:
        raise ValueError("Legacy checkpoint lacks the current physical velocity label semantics")
    selections = select_candidates(args.runs, args.variants)
    data_path = args.runs / "prepared_flows.pt"
    data = load_torch_payload(data_path)
    records = data["test"]
    test_uids = {record["uid"] for record in records}
    test_seeds = {record["seed"] for record in records}
    for split in ("train", "val"):
        if test_seeds & {record["seed"] for record in data[split]}:
            raise ValueError("TAQ training/validation seeds overlap test seeds")
    legacy_uids = set()
    for path in legacy_checkpoint["train_files"] + legacy_checkpoint["val_files"]:
        _, _, episode_uid = chunk_identity(Path(path))
        legacy_uids.add(episode_uid)
    if legacy_uids & test_uids:
        raise ValueError("Legacy training/validation episodes overlap test episodes")
    for record in records:
        if not record["valid"].all() or not torch.isfinite(record["x"]).all():
            raise ValueError("Expected finite unpadded episode inputs")
        if record["score"][0]:
            raise ValueError("Episode first frame cannot be scored by the legacy adapter")
    paths = [args.legacy_checkpoint, data_path, Path(__file__),
             *[path for path, _ in selections.values()]]
    frozen_hashes = {str(path.resolve()): digest(path) for path in paths}
    legacy = VelocityPredictorGRU(**legacy_checkpoint["config"]).to(device).eval()
    legacy.load_state_dict(legacy_checkpoint["model_state_dict"], strict=True)
    for name in ("mean", "std"):
        torch.testing.assert_close(getattr(legacy, f"input_{name}").flatten().cpu(),
                                   legacy_checkpoint["input_norm"][name].flatten(), rtol=0, atol=0)
    candidates = {}
    for name, (path, checkpoint) in selections.items():
        configuration = checkpoint["variant_config"]
        model = ExperimentalGRU(checkpoint["original_config"],
                                quality=configuration.get("quality", False),
                                split_gate=configuration.get("split_gate", False))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        candidates[name] = (model.to(device).eval(), checkpoint)
        print(f"Frozen {name}: {path.name}, seed={checkpoint['seed']}, "
              f"validation_macro_p95={checkpoint['metric']:.4f}", flush=True)
    args.out.mkdir(parents=True)
    metrics = []
    episode_metrics = []
    for context in ("rolling_0p48", "full_episode"):
        tasks = history_tasks(records, context, 0.48)
        saved = {name: [np.full((len(record["x"]), 3), np.nan, dtype=np.float32)
                        for record in records] for name in ("legacy_gru", *candidates)}
        batch_size = min(args.batch_size, 8) if context == "full_episode" else args.batch_size
        for offset in range(0, len(tasks), batch_size):
            selected = tasks[offset:offset + batch_size]
            batch, sequences = prepare_batch(records, selected)
            predictions = predict_batch(legacy, candidates, batch, sequences,
                                        int(legacy_checkpoint["preprocessing"]["pos_smooth_window"]), device)
            for name, outputs in predictions.items():
                for (record_index, start, end), prediction in zip(selected, outputs):
                    if context == "full_episode":
                        saved[name][record_index][:] = prediction
                    else:
                        saved[name][record_index][end - 1] = prediction[-1]
            if offset % (batch_size * 20) == 0 or offset + batch_size >= len(tasks):
                print(f"{context}: {min(offset + batch_size, len(tasks))}/{len(tasks)}", flush=True)
        rows, episodes = summarize(records, saved, context)
        metrics.extend(rows)
        episode_metrics.extend(episodes)
        torch.save(dict(predictions=saved, uids=[record["uid"] for record in records],
                        score=[record["score"] for record in records],
                        ground=[record["ground"] for record in records]),
                   args.out / f"{context}_predictions.pt")
    for path, expected in frozen_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    protocol = dict(device=str(device), torch_version=str(torch.__version__),
                    frozen_sha256=frozen_hashes, test_episodes=len(records),
                    scored_frames=sum(int(record["score"].sum()) for record in records),
                    legacy_test_episode_overlap=0, taq_trainval_test_seed_overlap=0,
                    legacy_config=legacy_checkpoint["config"],
                    selected={name: dict(checkpoint=str(path.resolve()), seed=checkpoint["seed"],
                                        epoch=checkpoint["epoch"], validation_macro_p95=checkpoint["metric"],
                                        native_configuration=checkpoint["variant_config"])
                              for name, (path, checkpoint) in selections.items()},
                    context_window_s=0.48, input_policy="shared raw saved observations; no extra missing propagation",
                    reset_policy="episode-only for all models", exact_shared_score_masks=True,
                    retrained=False, architecture_only_comparison=False,
                    elapsed_s=time.perf_counter() - started)
    write_csv(args.out / "metrics.csv", metrics)
    write_csv(args.out / "episode_metrics.csv", episode_metrics)
    (args.out / "summary.json").write_text(json.dumps(dict(protocol=protocol, metrics=metrics),
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.out / "REPORT.md", metrics, protocol)
    print(f"Report saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()

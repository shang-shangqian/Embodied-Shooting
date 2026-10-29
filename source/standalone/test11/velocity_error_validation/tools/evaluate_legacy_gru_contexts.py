"""Frozen legacy 13-input GRU compatibility evaluation on the current test set.

Do not reinterpret TAQ finite-difference velocity as legacy position increments.
Every reported primary error uses the current physical relative-velocity labels.
The optional derivative-to-physical conversion is an explicit label hypothesis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from summarize_learned_filter_contexts import load_windows, sha, statistics, write_csv
from speed_prediction.time_aware_velocity_data import load_torch_payload

SPEED_DIR = Path(__file__).resolve().parents[2] / 'speed_prediction'
sys.path.insert(0, str(SPEED_DIR))
from speed_prediction_model import VelocityPredictorGRU
from test_gru_velocity import smooth_position_sequence


def legacy_features(features, smooth_window):
    """Keep the legacy training order and drop the first position-only frame."""
    features = torch.as_tensor(features, dtype=torch.float32)
    position = smooth_position_sequence(features[None, :, :3], smooth_window)[0]
    increment = position[1:] - position[:-1]
    moving = (torch.linalg.vector_norm(features[1:, 6:8], dim=-1) > .05).float()[:, None]
    inputs = torch.cat((position[1:], increment, features[1:, 6:9],
                        features[1:, 9:12], moving), dim=-1)
    if inputs.shape != (len(features) - 1, 13) or not torch.isfinite(inputs).all():
        raise ValueError('Invalid legacy inputs')
    return inputs


@torch.inference_mode()
def sequence_predictions(model, features, smooth_window, device, batch_size=64):
    results = []
    check_error = 0.
    # Prefix predictions are equivalent to the native last-frame model API.
    for offset in range(0, len(features), batch_size):
        original = features[offset:offset + batch_size]
        sequences = [legacy_features(f, smooth_window) for f in original]
        n = max(len(x) for x in sequences)
        batch = torch.zeros(len(sequences), n, 13, device=device)
        for i, x in enumerate(sequences):
            batch[i, :len(x)] = x.to(device)
        output, _ = model.gru(model.normalize_input(batch))
        prediction = model.head(output).cpu().numpy()
        for i, x in enumerate(sequences):
            # First raw frame has no position increment and is not scored.
            p = np.full((len(original[i]), 3), np.nan, np.float32)
            p[1:] = prediction[i, :len(x)]
            results.append(p)
            if offset == 0 and i < 3 and len(x):
                for end in sorted({min(5, len(x)), max(1, len(x) // 2), len(x)}):
                    native = model(x[None, :end].to(device)).cpu().numpy()[0]
                    check_error = max(check_error, float(np.max(abs(native - p[end]))))
        print(f'Predicted sequences: {min(offset + batch_size, len(features))}/{len(features)}', flush=True)
    if check_error > 1e-4:
        raise ValueError(f'Prefix/native API consistency failed: {check_error}')
    return results, check_error


@torch.inference_mode()
def check_continuous_state(model, raw, full_prediction, smooth_window, device):
    x = legacy_features(raw, smooth_window).to(device)
    state = None
    parts = []
    for start in range(0, len(x), 100):
        output, state = model.gru(model.normalize_input(x[None, start:start + 100]), state)
        parts.append(model.head(output)[0].cpu().numpy())
    delta = float(np.max(abs(np.concatenate(parts) - full_prediction[1:])))
    if delta > 1e-4:
        raise ValueError(f'Continuous-state segment consistency failed: {delta}')
    return delta


def rotation_term(features):
    f = np.asarray(features)
    return np.column_stack((-f[:, 11] * f[:, 1], f[:, 11] * f[:, 0], np.zeros(len(f))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--learned-report', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite report: {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    # Keep batch/prefix consistency checks in full float32 precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cp = load_torch_payload(args.checkpoint)
    hashes = {str(p.resolve()): sha(p) for p in
              (args.checkpoint, args.runs / 'prepared_flows.pt', args.learned_report / 'summary.json')}
    records = load_torch_payload(args.runs / 'prepared_flows.pt')['test']
    smooth_window = int(cp['preprocessing']['pos_smooth_window'])
    device = torch.device(args.device)
    model = VelocityPredictorGRU(**cp['config']).to(device)
    model.load_state_dict(cp['model_state_dict'], strict=True)
    # State buffers already contain the frozen training normalization.
    for name in ('mean', 'std'):
        torch.testing.assert_close(getattr(model, f'input_{name}').flatten().cpu(),
                                   cp['input_norm'][name].flatten(), rtol=0, atol=0)
    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    full_features = [r['x'] for r in records]
    full_prediction, full_api_check = sequence_predictions(model, full_features, smooth_window, device)
    stream_checks = []
    for scenario in ('ss', 'sd'):
        idx = next(i for i, r in enumerate(records) if r['scenario'] == scenario and len(r['x']) > 100)
        stream_checks.append(check_continuous_state(model, records[idx]['x'], full_prediction[idx],
                                                    smooth_window, device))
    windows, window_manifest = load_windows(records)
    chunks = {path: load_torch_payload(path) for path in {w['path'] for w in windows}}
    short_features = [chunks[w['path']]['sequence_features'][w['sample'], :len(w['indices'])]
                      for w in windows]
    short_prediction, short_api_check = sequence_predictions(model, short_features, smooth_window, device)
    by_uid = {r['uid']: (r, p) for r, p in zip(records, full_prediction)}
    groups = defaultdict(list)
    episodes = defaultdict(set)
    details = []
    saved_full, saved_short = [], []

    def collect(context, scenario, uid, prediction, features, target, score, ground):
        for variant, pred in (('raw_output', prediction),
                              ('derivative_label_hypothesis', prediction + rotation_term(features))):
            error = np.linalg.norm(pred - target, axis=1)
            for filt in ('center_above_ground', 'unfiltered'):
                mask = np.asarray(score, bool).copy()
                if filt == 'center_above_ground':
                    mask &= np.asarray(ground, bool)
                if not np.isfinite(error[mask]).all():
                    raise ValueError('Nonfinite prediction on a scored frame')
                if not mask.any():
                    continue
                for scene in (scenario, 'pooled'):
                    key = (variant, context, filt, scene)
                    groups[key].append(error[mask])
                    episodes[key].add(uid)
                if variant == 'raw_output':
                    details.append(dict(context=context, scenario=scenario, uid=uid,
                                        filter=filt, **statistics(error[mask])))

    for r, p in zip(records, full_prediction):
        collect('full_episodes', r['scenario'], r['uid'], p, r['x'].numpy(),
                r['target'].numpy(), r['score'].numpy(), r['ground'].numpy())
        saved_full.append(dict(uid=r['uid'], prediction=p,
                               error=np.linalg.norm(p - r['target'].numpy(), axis=1),
                               score=r['score'].numpy(), ground=r['ground'].numpy()))
    for w, x, p in zip(windows, short_features, short_prediction):
        full_record, full_p = by_uid[w['uid']]
        collect('short_windows', w['scenario'], w['uid'], p, x.numpy(), w['target'],
                w['score'], w['ground'])
        collect('full_episodes_on_short_window_frames', w['scenario'], w['uid'],
                full_p[w['indices']], full_record['x'].numpy()[w['indices']],
                w['target'], w['score'], w['ground'])
        last_score = np.zeros(len(x), bool)
        last_score[-1] = w['score'][-1]
        collect('short_window_endpoints_only', w['scenario'], w['uid'], p, x.numpy(),
                w['target'], last_score, w['ground'])
        saved_short.append(dict(uid=w['uid'], path=w['path'], sample=w['sample'],
                                indices=w['indices'], prediction=p,
                                error=np.linalg.norm(p - w['target'], axis=1),
                                score=w['score'], ground=w['ground']))
    rows = []
    for (variant, context, filt, scenario), values in groups.items():
        rows.append(dict(variant=variant, context=context, filter=filt, scenario=scenario,
                         scored_episodes=len(episodes[(variant, context, filt, scenario)]),
                         **statistics(np.concatenate(values))))
    learned = json.loads((args.learned_report / 'summary.json').read_text())['metrics']
    comparison = []
    for row in rows:
        if row['variant'] != 'raw_output' or row['context'] == 'short_window_endpoints_only':
            continue
        reference = next(r for r in learned if all(row[k] == r[k] for k in ('context','filter','scenario')))
        assert row['n'] == reference['n'], 'Comparison frame counts differ'
        for metric in ('mae','p50','p75','p80','p85','p90','p95'):
            comparison.append(dict(context=row['context'], filter=row['filter'], scenario=row['scenario'],
                                   metric=metric, legacy_gru=row[metric], learned_filter=reference[metric],
                                   legacy_minus_learned=row[metric] - reference[metric]))
    # Prove that all compared masks/index lists, not just counts, match.
    previous_windows = load_torch_payload(args.learned_report / 'short_window_predictions.pt')
    assert len(previous_windows) == len(saved_short)
    for a, b in zip(saved_short, previous_windows):
        assert a['uid'] == b['uid'] and a['path'] == b['path'] and a['sample'] == b['sample']
        for k in ('indices','score','ground'):
            np.testing.assert_array_equal(a[k], b[k])
    previous_full = load_torch_payload(args.runs / 'learned_filter_seed41/test_predictions.pt')
    assert len(previous_full) == len(saved_full)
    for a, b in zip(saved_full, previous_full):
        assert a['uid'] == b['uid']
        np.testing.assert_array_equal(a['score'], b['score'])
    for path, digest in hashes.items():
        assert sha(Path(path)) == digest, f'Input changed: {path}'
    protocol = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                    frozen_inputs=hashes, model_config=cp['config'], preprocessing=cp['preprocessing'],
                    parameters=sum(p.numel() for p in model.parameters()), device=str(device),
                    torch_version=torch.__version__, source_sha256=sha(Path(__file__)),
                    tf32_enabled=False,
                    model_source_sha256=sha(SPEED_DIR / 'speed_prediction_model.py'),
                    adapter='observed TAQ position, plain per-frame position increment in metres (no division by dt), supplied yaw-frame body linear/angular input, dog flag=(horizontal linear speed>0.05); drop first raw frame',
                    invalid_position_policy='use positions exactly as saved; nonfinite positions were zero-filled by TAQ collector; no learned-filter propagation or additional clipping',
                    angular_z='current TAQ heading derivative retained; original legacy collector angular-z was yaw-frame gyro-z; interface differs under roll/pitch',
                    primary_output='raw network output scored directly against physical relative translation velocity in current yaw-only body axes',
                    label_semantics_limit='legacy checkpoint has no recorded label semantics; original training chunks are absent; raw-output results are a current-interface compatibility test, not proof of original label equivalence',
                    derivative_label_diagnostic='explicit alternative hypothesis only: add heading_rate cross observed_position, with no GT target position input; not a validated output conversion',
                    short_policy='same 216 existing nominal 0.48 s windows as learned-filter report; state reset per window; prefix output at each common scored frame, 0.10 s burn-in',
                    full_policy='continuous causal GRU state for entire episode; no observation-triggered reset; original common score/ground masks',
                    native_test_note='legacy model was trained on last-frame outputs; prefix-wise and complete-stream evaluation extends its original testing protocol; endpoint-only diagnostics included',
                    score_definition='pooled-frame Euclidean 3-D error, m/s; MAE=mean norm; center_above_ground=current target center world z>=0',
                    gt_usage='GT target velocity/height and labels for scoring only; no audit input to model',
                    retrained=False, prefix_native_max_abs_mps=max(full_api_check, short_api_check),
                    continuous_segment_max_abs_mps=max(stream_checks), exact_comparison_masks_verified=True,
                    elapsed_s=time.perf_counter() - started,
                    cuda_peak_allocated_mb=(torch.cuda.max_memory_allocated(device) / 2**20
                                            if device.type == 'cuda' else None))
    write_csv(args.out / 'summary.csv', rows)
    write_csv(args.out / 'sequence_metrics.csv', details)
    write_csv(args.out / 'comparison_with_learned_filter.csv', comparison)
    (args.out / 'summary.json').write_text(json.dumps(dict(protocol=protocol, metrics=rows), indent=2))
    (args.out / 'window_manifest.json').write_text(json.dumps(window_manifest, indent=2))
    torch.save(saved_full, args.out / 'full_episode_predictions.pt')
    torch.save(saved_short, args.out / 'short_window_predictions.pt')
    lines = ['# 旧 GRU 04-28-13-59：短窗口与完整 episodes', '',
             '冻结模型：`speed_prediction/gru_runs/04-28-13-59/best_model.pt`。'
             '独立测试集与学习滤波器相同：`optimization_20261007/test`，D 条件，SS/SD × clean/mixed。未重新训练。', '',
             'MAE 为三维相对速度误差的欧氏范数均值，分位数按评分帧汇总，单位 m/s。'
             '短窗口和完整 episodes 的帧索引、评分掩码与刚才学习滤波器报告逐项核验一致。', '',
             '## 输入和输出兼容性', '',
             '旧模型输入 13 维：位置 3 + 未除 dt 的位置增量 3 + 机器人线速度 3 + 角速度 3 + 运动标志 1。'
             f'采用 checkpoint 原始归一化，位置平滑窗口={smooth_window}。每段第一帧仅供计算后续位置差分，不喂给 GRU。'
             '缺测位置沿用保存的零填充值；没有添加新滤波、裁剪或物理基线。运动标志由当前水平线速度 >0.05 m/s 推导。', '',
             '短窗口：同一批 216 个 nominal 0.48 s 保存窗口，每窗重置隐藏状态，逐前缀输出并评分（0.10 s burn-in）。'
             '完整 episodes：整段连续保留 GRU 状态，仅新 episode 重置。额外给出完整流在短窗同帧上的结果。', '',
             '**兼容性边界：**该 checkpoint 来自旧视觉数据，未记录明确的速度标签语义，且原训练 chunk 当前不存在。'
             '主表按当前物理相对平移速度标签评估原始网络输出，因此是当前接口兼容性结果。'
             '当前输入 angular-z 是 heading 导数，旧采集代码使用 yaw-frame gyro-z，二者在存在 roll/pitch 时有差异。'
             '原模型训练/测试只读窗口末帧，本次逐前缀与完整流输出是对原测试协议的扩展。', '']
    names = dict(short_windows='短窗口', full_episodes='完整 episodes',
                 full_episodes_on_short_window_frames='完整 episodes（短窗同帧）',
                 short_window_endpoints_only='短窗口（仅末帧）')
    for filt, title in (('center_above_ground', '地上评分（目标中心 z≥0）'),
                        ('unfiltered', '未筛地面评分')):
        lines += [f'## {title}', '', '| 评估方式 | 场景 | 帧数 | MAE | P50 | P75 | P80 | P85 | P90 | P95 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for context in names:
            for scenario in ('ss','sd','pooled'):
                row = next(r for r in rows if r['variant'] == 'raw_output'
                           and r['context'] == context and r['filter'] == filt and r['scenario'] == scenario)
                lines.append(f"| {names[context]} | {scenario.upper()} | {row['n']:,} | "
                             + ' | '.join(f"{row[k]:.4f}" for k in ('mae','p50','p75','p80','p85','p90','p95')) + ' |')
        lines += ['']
    lines += ['## 速度语义假设诊断', '',
              '若假定旧输出是旋转坐标位置导数，可加 `heading_rate × observed_position` 再比较物理速度。'
              '这是未确认的假设，没有把 GT 位置送入模型；不能按测试指标优劣选择一种解释作为正式标签。', '',
              '| 评估方式 | 场景 | 原始输出 MAE | 假设换算 MAE | 原始输出 P95 | 假设换算 P95 |',
              '|---|---|---:|---:|---:|---:|']
    for context in ('short_windows', 'full_episodes'):
        for scenario in ('ss','sd'):
            a = next(r for r in rows if r['variant'] == 'raw_output' and r['context'] == context
                     and r['scenario'] == scenario and r['filter'] == 'center_above_ground')
            b = next(r for r in rows if r['variant'] == 'derivative_label_hypothesis' and r['context'] == context
                     and r['scenario'] == scenario and r['filter'] == 'center_above_ground')
            lines.append(f"| {names[context]} | {scenario.upper()} | {a['mae']:.4f} | {b['mae']:.4f} | {a['p95']:.4f} | {b['p95']:.4f} |")
    lines += ['', f"CUDA:0 推理；逐前缀与原模型 forward 最大差 {protocol['prefix_native_max_abs_mps']:.9g} m/s，"
              f"分段携带状态与整段推理最大差 {protocol['continuous_segment_max_abs_mps']:.9g} m/s。", '',
              '[精确数据和协议](summary.json) · [CSV](summary.csv) · '
              '[与学习滤波器比较](comparison_with_learned_filter.csv)', '']
    (args.out / 'report.md').write_text('\n'.join(lines))
    (args.out / 'COMPLETE').write_text('Frozen legacy-GRU compatibility evaluation complete.\n')
    print(json.dumps(dict(protocol=protocol, metrics=[r for r in rows
                     if r['variant'] == 'raw_output' and r['filter'] == 'center_above_ground']), indent=2), flush=True)


if __name__ == '__main__':
    main()

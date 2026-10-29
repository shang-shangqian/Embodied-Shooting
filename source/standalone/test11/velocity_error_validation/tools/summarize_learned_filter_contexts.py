"""Evaluate frozen learned CV on saved short windows and full test episodes.

Short windows rebuild all propagated inputs and multiscale history locally.
Ground truth is used only to join labels and select scoring frames.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from optimization_components import WIDTHS, fit_baseline, propagated_inputs, rotate
from train_flow_ablations import evaluate, make_model
from speed_prediction.time_aware_velocity_data import load_torch_payload


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    result = dict(n=int(len(values)))
    if len(values):
        result['mae'] = float(values.mean())
        result.update({f'p{q}': float(np.quantile(values, q / 100))
                       for q in (50, 75, 80, 85, 90, 95)})
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def local_inputs(features, sources, timestamps):
    x, resets = propagated_inputs(features, sources, timestamps)
    extras = []
    for width in WIDTHS:
        baseline, extra = fit_baseline(x, timestamps, resets, width,
                                       aligned=True, fd_weight=False)
        # Match prepare_record's missing-observation feature propagation.
        for i in range(1, len(x)):
            if x[i, 23] < .5:
                baseline[i] = rotate(
                    baseline[i - 1] + x[i - 1, 6:9],
                    np.array(-x[i, 11] * (timestamps[i] - timestamps[i - 1])),
                ) - x[i, 6:9]
        extra[:, :3] = baseline / 10
        extras.append(extra)
    return x, np.concatenate(extras, axis=1)


def load_windows(records):
    windows = []
    manifest = []
    for r in records:
        stream_path = Path(r['path'])
        times = r['time'].numpy().astype(np.float64)
        occupied = set()
        for path in sorted(stream_path.parent.glob('taq_velocity_chunk_*.pt')):
            c = load_torch_payload(path)
            manifest.append(dict(path=str(path.resolve()), sha256=sha(path)))
            if c['metadata']['split'] != 'test' or c['metadata']['condition'] != 'D':
                raise ValueError('Only real-input independent test windows allowed')
            if abs(c['metadata']['window_s'] - .48) > 1e-9:
                raise ValueError('Unexpected saved window width')
            for j in range(len(c['sequence_features'])):
                valid = c['sequence_valid_mask'][j].numpy().astype(bool)
                n = int(valid.sum())
                if not valid[:n].all() or valid[n:].any():
                    raise ValueError('Non-prefix valid mask')
                x = c['sequence_features'][j, :n].numpy().copy()
                source = c['sequence_source_ids'][j, :n].numpy()
                timestamps = (float(c['sample_window_start_s'][j])
                              + np.cumsum(x[:, 12].astype(np.float64)))
                idx = np.searchsorted(times, timestamps).clip(0, len(times) - 1)
                left = np.maximum(0, idx - 1)
                idx = np.where(abs(times[left] - timestamps) < abs(times[idx] - timestamps), left, idx)
                delta = float(np.max(abs(times[idx] - timestamps)))
                if delta > 5e-5 or not np.all(np.diff(idx) == 1):
                    raise ValueError(f'Window/stream timestamp mismatch: {path} {delta}')
                if abs(timestamps[-1] - float(c['sample_window_end_s'][j])) > 5e-5:
                    raise ValueError('Window endpoint mismatch')
                if occupied.intersection(idx.tolist()):
                    raise ValueError('Saved windows overlap')
                occupied.update(idx.tolist())
                if not np.array_equal(source, r['source'].numpy()[idx]):
                    raise ValueError('Window/stream source mismatch')
                target = c['target_velocity_base'][j, :n].numpy()
                if not np.array_equal(target, r['target'].numpy()[idx]):
                    raise ValueError('Window/stream target mismatch')
                if np.max(abs(x[:, :3] - r['x'].numpy()[idx, :3])) > 1e-6:
                    raise ValueError('Window/stream observed position mismatch')
                if x[0, 12] != 0 or np.any(x[0, 3:6] != 0):
                    raise ValueError('Saved window must reset dt and finite difference')
                local_x, extra = local_inputs(x, source, times[idx])
                score = (c['target_valid_mask'][j, :n].numpy().astype(bool)
                         & r['score'].numpy()[idx])
                windows.append(dict(uid=r['uid'], scenario=r['scenario'], profile=r['profile'],
                                    path=str(path), sample=j, indices=idx,
                                    x=local_x, extra=extra, source=source, target=target,
                                    score=score, ground=r['ground'].numpy()[idx],
                                    span_s=float(times[idx[-1]] - times[idx[0]]),
                                    age_s=times[idx] - times[0], join_max_error_s=delta))
    return windows, manifest


@torch.inference_mode()
def predict_windows(model, windows, device, batch_size=128):
    saved = []
    for start in range(0, len(windows), batch_size):
        rows = windows[start:start + batch_size]
        length = max(len(r['x']) for r in rows)
        x = torch.zeros(len(rows), length, 24, device=device)
        extra = torch.zeros_like(x)
        source = torch.zeros(len(rows), length, dtype=torch.long, device=device)
        valid = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
        for j, row in enumerate(rows):
            n = len(row['x'])
            x[j, :n] = torch.tensor(row['x'], device=device)
            extra[j, :n] = torch.tensor(row['extra'], device=device)
            source[j, :n] = torch.tensor(row['source'], device=device)
            valid[j, :n] = True
        prediction, _, _ = model.forward_segment(x, source, extra, valid, state=None)
        prediction = prediction.cpu().numpy()
        for j, row in enumerate(rows):
            p = prediction[j, :len(row['x'])]
            saved.append(dict(uid=row['uid'], path=row['path'], sample=row['sample'],
                              indices=row['indices'], prediction=p,
                              error=np.linalg.norm(p - row['target'], axis=1),
                              score=row['score'], ground=row['ground']))
        print(f'Short windows: {min(start + batch_size, len(windows))}/{len(windows)}', flush=True)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite report: {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.perf_counter()
    selection = json.loads((args.runs / 'evaluation/frozen_selection.json').read_text())
    if selection['run'] != 'learned_filter_seed41':
        raise ValueError('Unexpected frozen model selection')
    cp_path = args.runs / selection['run'] / 'best_model.pt'
    data_path = args.runs / 'prepared_flows.pt'
    saved_path = args.runs / selection['run'] / 'test_predictions.pt'
    hashes = {str(p.resolve()): sha(p) for p in (cp_path, data_path, saved_path)}
    cp = load_torch_payload(cp_path)
    records = load_torch_payload(data_path)['test']
    full_saved = load_torch_payload(saved_path)
    for r, p in zip(records, full_saved):
        if r['uid'] != p['uid'] or len(r['x']) != len(p['prediction']):
            raise ValueError('Saved full-episode prediction mismatch')
        np.testing.assert_array_equal(r['score'].numpy(), p['score'])
        np.testing.assert_allclose(p['error'], np.linalg.norm(
            p['prediction'] - r['target'].numpy(), axis=1), rtol=0, atol=1e-6)
    assert len(records) == len(full_saved)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    model = make_model(cp['variant_config'], device)
    model.load_state_dict(cp['state_dict'], strict=True)
    model.eval()
    _, _, _, reproduced = evaluate(model, records, cp['variant_config'], device,
                                   scale=cp['scale'], return_rows=True)
    reproduction_error = max(float(np.max(abs(a['prediction'] - b['prediction'])))
                             for a, b in zip(full_saved, reproduced))
    if reproduction_error > 1e-4:
        raise ValueError(f'Full-episode reproduction failed: {reproduction_error}')
    windows, manifest = load_windows(records)
    short_saved = predict_windows(model, windows, device)
    full_by_uid = {r['uid']: (r, s) for r, s in zip(records, full_saved)}
    errors = defaultdict(list)
    ages = defaultdict(list)
    counts = defaultdict(set)
    window_rows = []
    for r, s in zip(records, full_saved):
        for filt in ('center_above_ground', 'unfiltered'):
            mask = r['score'].numpy().copy()
            if filt == 'center_above_ground':
                mask &= r['ground'].numpy()
            if mask.any():
                for scenario in (r['scenario'], 'pooled'):
                    key = ('full_episodes', filt, scenario)
                    errors[key].append(s['error'][mask])
                    ages[key].append(r['time'].numpy()[mask] - float(r['time'][0]))
                    counts[key].add(r['uid'])
    for w, s in zip(windows, short_saved):
        _, full = full_by_uid[w['uid']]
        for filt in ('center_above_ground', 'unfiltered'):
            mask = w['score'].copy()
            if filt == 'center_above_ground':
                mask &= w['ground']
            if not mask.any():
                continue
            window_rows.append(dict(uid=w['uid'], scenario=w['scenario'], profile=w['profile'],
                                    filter=filt, path=w['path'], sample=w['sample'],
                                    span_s=w['span_s'], **statistics(s['error'][mask])))
            for scenario in (w['scenario'], 'pooled'):
                for context, e in (('short_windows', s['error']),
                                   ('full_episodes_on_short_window_frames', full['error'][w['indices']])):
                    key = (context, filt, scenario)
                    errors[key].append(e[mask])
                    ages[key].append(w['age_s'][mask])
                    counts[key].add(w['uid'])
    rows = []
    for context in ('short_windows', 'full_episodes', 'full_episodes_on_short_window_frames'):
        for filt in ('center_above_ground', 'unfiltered'):
            for scenario in ('ss', 'sd', 'pooled'):
                key = (context, filt, scenario)
                rows.append(dict(context=context, filter=filt, scenario=scenario,
                                 scored_episodes=len(counts[key]),
                                 mean_episode_age_s=float(np.concatenate(ages[key]).mean()),
                                 **statistics(np.concatenate(errors[key]))))
    # Reproduction against the previously published independent-test summary.
    prior = list(csv.DictReader((args.runs / selection['run'] / 'test_summary.csv').open()))
    for row in rows:
        if row['context'] != 'full_episodes' or row['scenario'] == 'pooled':
            continue
        old = next(r for r in prior if r['scenario'] == row['scenario']
                   and r['filter'] == row['filter'] and r['group'] == 'all')
        assert row['n'] == int(old['n'])
        assert abs(row['mae'] - float(old['mean'])) < 1e-6
        assert abs(row['p95'] - float(old['p95'])) < 1e-6
    for path, digest in hashes.items():
        assert sha(Path(path)) == digest, f'Input changed: {path}'
    protocol = dict(model=selection['run'], checkpoint_sha256=hashes[str(cp_path.resolve())],
                    source_sha256=sha(Path(__file__)), frozen_inputs=hashes,
                    device=str(device), torch_version=torch.__version__,
                    metric='mean and pooled-frame quantiles of Euclidean 3-D relative-velocity error, m/s',
                    mae_definition='mean ||v_pred-v_gt||_2; not per-axis L1 or speed-magnitude MAE',
                    nominal_short_window_s=.48, saved_windows=len(windows),
                    actual_short_span_min_s=min(w['span_s'] for w in windows),
                    actual_short_span_max_s=max(w['span_s'] for w in windows),
                    short_policy='existing saved non-overlapping capped windows; reset state, covariance, odometry and all multiscale feature history per window',
                    short_score='saved target_valid_mask (0.10 s local burn-in) intersect original full-flow score; ground filter only changes scoring',
                    full_policy='carry state across complete episode; original supervision/first-direct score; reset only on new episode',
                    ground_policy='current target center world z>=0; audit used for scoring only',
                    inference_inputs='measured position, source/quality, robot motion and within-window causal history; no GT audit',
                    max_timestamp_join_error_s=max(w['join_max_error_s'] for w in windows),
                    full_episode_reproduction_max_abs_mps=reproduction_error,
                    window_coverage_note='windows limited to four per episode and require valid range; short and full rows have different frame coverage; same-frame control provided',
                    elapsed_s=time.perf_counter() - start,
                    cuda_peak_allocated_mb=(torch.cuda.max_memory_allocated(device) / 2**20
                                            if device.type == 'cuda' else None))
    write_csv(args.out / 'summary.csv', rows)
    write_csv(args.out / 'window_metrics.csv', window_rows)
    (args.out / 'summary.json').write_text(json.dumps(dict(protocol=protocol, metrics=rows), indent=2))
    (args.out / 'window_manifest.json').write_text(json.dumps(manifest, indent=2))
    torch.save(short_saved, args.out / 'short_window_predictions.pt')
    lines = ['# 最新学习滤波器：短窗口与完整 episodes 指标', '',
             f"模型：`{selection['run']}`，独立测试集 `optimization_20261007/test`，D 条件（机身相机、真实跟踪与测距），clean/mixed 合并。模型参数未调整。", '',
             '误差为三维相对速度向量误差的欧氏范数，单位 m/s；MAE 为该误差的逐帧均值，各分位数汇总评分帧计算，不是 episode 指标的平均。SS 是静态目标，SD 是动态目标。', '',
             '短窗口使用已保存的 nominal 0.48 s 窗口，实际首末时间跨度 '
             f"{protocol['actual_short_span_min_s']:.4f}–{protocol['actual_short_span_max_s']:.4f} s。每个窗口重置滤波状态、协方差、里程计和所有历史特征；保留窗口内 0.10 s burn-in。完整 episodes 保留连续状态，沿用已有评分掩码。", '',
             f"保存窗口共 {len(windows)} 个；每个 episode 最多四个，并要求有效测距，因此两种口径覆盖帧不同。下方包含完整流在短窗口同帧上的对照。", '']
    for filt, title in (('center_above_ground', '地上评分（目标中心 z≥0）'),
                        ('unfiltered', '未筛地面评分')):
        lines += [f'## {title}', '', '| 评估方式 | 场景 | 帧数 | MAE | P50 | P75 | P80 | P85 | P90 | P95 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        names = dict(short_windows='短窗口', full_episodes='完整 episodes',
                     full_episodes_on_short_window_frames='完整 episodes（短窗口同帧）')
        for row in rows:
            if row['filter'] != filt:
                continue
            lines.append(f"| {names[row['context']]} | {row['scenario'].upper()} | {row['n']:,} | "
                         + ' | '.join(f"{row[k]:.4f}" for k in ('mae','p50','p75','p80','p85','p90','p95')) + ' |')
        lines += ['']
    lines += ['完整流重算与原保存预测最大逐分量差：'
              f'{reproduction_error:.9g} m/s；所有原有全流 Mean/P95 与帧数均已复核一致。', '',
              '[精确数据](summary.csv) · [统计协议](summary.json) · [逐窗口指标](window_metrics.csv)', '']
    (args.out / 'report.md').write_text('\n'.join(lines))
    (args.out / 'COMPLETE').write_text('Frozen learned-filter context evaluation complete.\n')
    print(json.dumps(dict(protocol=protocol, metrics=rows), indent=2), flush=True)


if __name__ == '__main__':
    main()

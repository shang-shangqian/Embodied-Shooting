"""E1: controlled inference-context contrasts and causal rolling end points.

No training or changes to production inference. GT audit positions are used
only for scoring filters. H/B/F select prefix hidden state, prefix baseline,
and original first-step dt/FD respectively; all other stream features stay
fixed. Saved-window inference is a separate anchor because saved age and
missing-frame histories can differ from the stream.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
from analyze_error_sources import (  # noqa: E402
    _first_measurement_timestamp, _load_model, _segmented_baseline, _stream_arrays,
)
from validation_io import SOURCE_NAMES, write_csv  # noqa: E402
from speed_prediction.time_aware_velocity_data import (  # noqa: E402
    FeatureIndex, TAQ_BASELINE_VERSION, load_torch_payload, weighted_velocity_baseline,
)


BITS = list(itertools.product((0, 1), repeat=3))
FILTERS = ("unfiltered", "center_above_ground")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_step(features: torch.Tensor, retain: bool) -> torch.Tensor:
    result = features.clone()
    if not retain:
        result[0, 3:6] = 0
        result[0, FeatureIndex.DT] = 0
    return result


def align_times(stream_times: torch.Tensor, window_times: torch.Tensor) -> torch.Tensor:
    """Nearest exact timestamp match; no millisecond rounding collisions."""
    if len(stream_times) > 1 and torch.any(stream_times[1:] <= stream_times[:-1]):
        raise ValueError("Stream timestamps must increase strictly")
    right = torch.searchsorted(stream_times, window_times).clamp(max=len(stream_times) - 1)
    left = (right - 1).clamp(min=0)
    indices = torch.where(
        (stream_times[left] - window_times).abs() < (stream_times[right] - window_times).abs(),
        left, right,
    )
    if torch.any((stream_times[indices] - window_times).abs() > 2e-5):
        raise ValueError("Saved window cannot be matched to stream timestamps within 20 us")
    if len(indices) > 1 and torch.any(indices[1:] <= indices[:-1]):
        raise ValueError("Ambiguous or repeated window timestamps")
    return indices


def rolling_start(times: torch.Tensor, resets: torch.Tensor, end: int, width: float) -> int:
    start = int(torch.searchsorted(times, times[end] - width - 1e-8))
    last_reset = int(torch.nonzero(resets[:end + 1], as_tuple=False)[-1])
    return max(start, last_reset)


@dataclass
class Stats:
    errors: list[np.ndarray] = field(default_factory=list)
    count: int = 0
    axis_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_sum: float = 0
    residual_sum: float = 0
    hidden_sum: float = 0
    gate_sum: float = 0

    def add(self, prediction, truth, baseline, hidden_norm, gate):
        if len(prediction) == 0:
            return
        difference = prediction.astype(np.float64) - truth.astype(np.float64)
        errors = np.linalg.norm(difference, axis=-1)
        if not np.isfinite(errors).all():
            raise ValueError("Non-finite model output; do not silently remove difficult frames")
        self.errors.append(errors)
        self.count += len(errors)
        self.axis_sum += np.abs(difference).sum(axis=0)
        self.base_sum += np.linalg.norm(baseline - truth, axis=-1).sum()
        self.residual_sum += np.linalg.norm(prediction - baseline, axis=-1).sum()
        self.hidden_sum += hidden_norm.sum()
        self.gate_sum += gate.sum()

    def metrics(self):
        if not self.count:
            return {"n": 0}
        e = np.concatenate(self.errors)
        q = np.quantile(e, (.5, .9, .95, .99))
        return dict(n=self.count, mean=float(e.mean()), p50=q[0], p90=q[1], p95=q[2],
                    p99=q[3], over_0p5=float((e > .5).mean()), over_1=float((e > 1).mean()),
                    over_2=float((e > 2).mean()), mae_x=self.axis_sum[0] / self.count,
                    mae_y=self.axis_sum[1] / self.count, mae_z=self.axis_sum[2] / self.count,
                    baseline_mean=self.base_sum / self.count,
                    residual_norm_mean=self.residual_sum / self.count,
                    hidden_norm_mean=self.hidden_sum / self.count,
                    gate_mean=self.gate_sum / self.count)


@dataclass
class Stream:
    path: Path
    metadata: dict
    arrays: tuple
    world_z: np.ndarray
    score: np.ndarray | None = None
    baseline: torch.Tensor | None = None
    prediction: torch.Tensor | None = None
    hidden_before: torch.Tensor | None = None
    hidden_norm: torch.Tensor | None = None
    gate: torch.Tensor | None = None

    @property
    def uid(self):
        return self.metadata["episode_uid"]

    @property
    def condition(self):
        return self.metadata["condition"]


@dataclass
class Task:
    stream: Stream
    start: int
    end: int
    selected: torch.Tensor
    supervision: torch.Tensor
    original: torch.Tensor | None = None
    original_sources: torch.Tensor | None = None


def read_stream(path: Path) -> Stream:
    payload = load_torch_payload(path)
    arrays = _stream_arrays(payload, path)
    metadata = dict(payload["metadata"])
    for key in ("episode_uid", "trajectory_uid", "condition"):
        if not metadata.get(key):
            raise ValueError(f"Missing {key}: {path}")
    world_z = np.full(len(arrays[0]), np.nan)
    audit_path = path.with_name("velocity_error_audit.jsonl")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Ground scoring requires {audit_path}")
    seen = set()
    with audit_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["frame_index"])
            if row["episode_uid"] != metadata["episode_uid"] or index in seen:
                raise ValueError(f"Duplicate or mismatched audit row: {audit_path}")
            if not 0 <= index < len(world_z):
                raise ValueError(f"Audit frame out of range: {audit_path}")
            seen.add(index)
            world_z[index] = float(row["target_position_world_m"][2])
    if not np.isfinite(world_z).all():
        raise ValueError(f"Missing/non-finite GT height for scoring: {audit_path}")
    return Stream(path, metadata, arrays, world_z)


def prepare_pairs(paths: list[Path], max_pairs: int):
    groups = defaultdict(list)
    for path in paths:
        metadata = load_torch_payload(path)["metadata"]
        groups[str(metadata["trajectory_uid"])].append(path)
    selected = sorted(groups)[:max_pairs or None]
    streams = []
    for trajectory in selected:
        pair = [read_stream(p) for p in sorted(groups[trajectory])]
        if sorted(s.condition for s in pair) != ["A", "B", "C", "D"]:
            raise ValueError(f"Incomplete/duplicate paired trajectory: {trajectory}")
        reference = pair[0]
        starts = [_first_measurement_timestamp(s.arrays[0], s.arrays[1], s.arrays[2],
                                              s.arrays[6]) for s in pair]
        starts = [t for t in starts if t is not None]
        start = min(starts) if starts else float("inf")
        masks = []
        for s in pair:
            f, sources, valid, targets, target_valid, reset, times = s.arrays
            if times.shape != reference.arrays[6].shape or not torch.allclose(
                times, reference.arrays[6], atol=2e-5, rtol=0,
            ):
                raise ValueError(f"Paired timestamps differ: {trajectory}")
            if not np.allclose(s.world_z, reference.world_z, atol=1e-4, rtol=0):
                raise ValueError(f"Paired GT height differs: {trajectory}")
            masks.append((valid & target_valid & (times >= start)).numpy())
        common = np.logical_and.reduce(masks)
        for s in pair:
            s.score = common.copy()
        streams.extend(pair)
    return streams, len(groups)


def physical_baseline(model, features, valid, resets, width):
    # Cutting baseline history creates a buffer boundary, not a GRU reset.
    boundaries = resets.clone()
    boundaries[0] = True
    return _segmented_baseline(features, valid, boundaries, model, width)


@torch.inference_mode()
def run_sequences(model, features, sources, valid, resets, initial_hidden=None,
                  capture_prefix=False):
    """Baseline is deliberately zero: current GRU state is independent of it."""
    batch, steps = valid.shape
    hidden = model.initial_hidden(batch, device=features.device)
    if initial_hidden is not None:
        hidden = initial_hidden.clone()
    residual, norms, gates, before = [], [], [], []
    for i in range(steps):
        hidden = torch.where(resets[:, i, None], torch.zeros_like(hidden), hidden)
        if capture_prefix:
            before.append(hidden.clone())
        mean, _, hidden, gate = model.step(features[:, i], sources[:, i], hidden=hidden,
                                           valid_mask=valid[:, i])
        residual.append(mean)
        norms.append(torch.linalg.vector_norm(hidden, dim=-1))
        gates.append(gate[:, 0])
    result = (torch.stack(residual, 1), torch.stack(norms, 1), torch.stack(gates, 1))
    return (*result, torch.stack(before, 1) if capture_prefix else None)


def pad_sequences(features, sources, resets, device):
    lengths = [len(f) for f in features]
    steps = max(lengths)
    f = torch.zeros((len(lengths), steps, 24), device=device)
    src = torch.full((len(lengths), steps), 4, dtype=torch.long, device=device)
    valid = torch.zeros((len(lengths), steps), dtype=torch.bool, device=device)
    reset = torch.zeros_like(valid)
    for i, length in enumerate(lengths):
        f[i, :length] = features[i].to(device)
        src[i, :length] = sources[i].to(device)
        valid[i, :length] = True
        reset[i, :length] = resets[i].to(device)
    return f, src, valid, reset


def cache_prefix(model, streams, device, width):
    rows = [s.arrays for s in streams]
    f, src, valid, reset = pad_sequences([r[0] for r in rows], [r[1] for r in rows],
                                        [r[5] for r in rows], device)
    for i, s in enumerate(streams):
        valid[i, :len(s.arrays[0])] = s.arrays[2].to(device)
        s.baseline = physical_baseline(model, f[i, :len(s.arrays[0])],
                                      valid[i, :len(s.arrays[0])],
                                      reset[i, :len(s.arrays[0])], width)
    delta, norm, gate, before = run_sequences(model, f, src, valid, reset, capture_prefix=True)
    for i, s in enumerate(streams):
        n = len(s.arrays[0])
        s.prediction = delta[i, :n] + s.baseline
        s.hidden_before = before[i, :n]
        s.hidden_norm = norm[i, :n]
        s.gate = gate[i, :n]


def saved_tasks(streams, alignment, manifest):
    tasks = []
    for s in streams:
        f, sources, valid, targets, _, resets, times = s.arrays
        for path in sorted(s.path.parent.glob("taq_velocity_chunk_*.pt")):
            payload = load_torch_payload(path)
            if payload["metadata"]["episode_uid"] != s.uid:
                raise ValueError(f"Chunk episode mismatch: {path}")
            manifest.append({"path": str(path), "sha256": sha256(path)})
            for sample, raw in enumerate(payload["sequence_features"]):
                n = int(payload["sequence_valid_mask"][sample].sum())
                original = raw[:n]
                window_times = (payload["sample_window_start_s"][sample]
                                + original[:, FeatureIndex.DT].double().cumsum(0))
                indices = align_times(times, window_times)
                if abs(float(times[indices[-1]] - payload["sample_window_end_s"][sample])) > 2e-5:
                    raise ValueError(f"Saved window end mismatch: {path}")
                original_sources = payload["sequence_source_ids"][sample, :n]
                if not torch.equal(sources[indices], original_sources):
                    raise ValueError(f"Saved/stream source mismatch: {path}")
                if not torch.allclose(targets[indices], payload["target_velocity_base"][sample, :n],
                                      atol=1e-5, rtol=0):
                    raise ValueError(f"Saved/stream label mismatch: {path}")
                start, end = int(indices[0]), int(indices[-1])
                mismatch = (original - f[indices]).abs() > 1e-5
                allowed = torch.zeros_like(mismatch)
                allowed[0, 3:6] = True
                allowed[0, FeatureIndex.DT] = True
                other = torch.nonzero((mismatch & ~allowed).any(0)).flatten().tolist()
                alignment.append(dict(episode_uid=s.uid, chunk=str(path), sample=sample,
                                      start_frame=start, end_frame=end,
                                      omitted_stream_frames=end - start + 1 - n,
                                      internal_resets=int(resets[start + 1:end + 1].sum()),
                                      other_differing_features=other))
                tasks.append(Task(s, start, end, indices - start,
                                  payload["target_valid_mask"][sample, :n],
                                  original, original_sources))
    return tasks


class Recorder:
    def __init__(self):
        self.pooled = defaultdict(Stats)
        self.episodes = defaultdict(Stats)
        self.endpoint_errors = defaultdict(dict)

    def add(self, task, protocol, prediction, baseline, norm, gate, scope, source_groups=False):
        s = task.stream
        selected = task.selected.numpy()
        indices = selected + task.start
        prediction, baseline, norm, gate = [x.detach().cpu().numpy()[selected]
                                            for x in (prediction, baseline, norm, gate)]
        self.add_values(s, indices, task.supervision.numpy(), protocol, prediction,
                        baseline, norm, gate, scope, source_groups)

    def add_values(self, s, indices, supervision, protocol, prediction, baseline,
                   norm, gate, scope, source_groups=False):
        truth = s.arrays[3][indices].numpy()
        score = s.score[indices] & supervision
        if scope == "saved_endpoint":
            score[:-1] = False
        for scoring_filter in FILTERS:
            keep = score.copy()
            if scoring_filter == "center_above_ground":
                keep &= s.world_z[indices] >= 0
            key = (scope, scoring_filter, s.condition, protocol, "all")
            arrays = [x[keep] for x in (prediction, truth, baseline, norm, gate)]
            self.pooled[key].add(*arrays)
            self.episodes[(*key, s.uid)].add(*arrays)
            if scope == "saved_endpoint" and protocol.startswith("H") and keep[-1]:
                error = float(np.linalg.norm(prediction[-1].astype(float) - truth[-1]))
                self.endpoint_errors[(scoring_filter, s.condition, s.uid, int(indices[-1]))][protocol] = error
            if source_groups:
                sources = s.arrays[1][indices].numpy()
                for source_id in np.unique(sources[keep]):
                    subset = keep & (sources == source_id)
                    group_key = (*key[:-1], SOURCE_NAMES[int(source_id)])
                    self.pooled[group_key].add(*[x[subset] for x in
                                                (prediction, truth, baseline, norm, gate)])

    def summary_rows(self):
        names = ("scope", "filter", "condition", "protocol", "source")
        return [{**dict(zip(names, key)), **stats.metrics()}
                for key, stats in sorted(self.pooled.items())]

    def episode_rows(self):
        names = ("scope", "filter", "condition", "protocol", "source", "episode_uid")
        return [{**dict(zip(names, key)), **stats.metrics()}
                for key, stats in sorted(self.episodes.items())]

    def effects(self, bootstrap_samples=1000):
        """Factorial mean-error contrasts; CIs resample episodes, not frames."""
        grouped = defaultdict(lambda: defaultdict(list))
        for (scoring_filter, condition, uid, _), values in self.endpoint_errors.items():
            errors = np.array([values[f"H{h}B{b}F{f}"] for h, b, f in BITS])
            for order in (1, 2, 3):
                for axes in itertools.combinations(range(3), order):
                    weights = [np.prod([2 * bits[a] - 1 for a in axes]) / 2**(3 - order)
                               for bits in BITS]
                    effect = float(np.dot(weights, errors))
                    name = ":".join("HBF"[a] for a in axes)
                    grouped[(scoring_filter, condition, name)][uid].append(effect)
        rows = []
        rng = np.random.default_rng(42)
        for (scoring_filter, condition, effect), episodes in sorted(grouped.items()):
            values = [np.array(v) for v in episodes.values()]
            means = np.array([v.mean() for v in values])
            sample = means[rng.integers(0, len(means), size=(bootstrap_samples, len(means)))].mean(1)
            low, high = np.quantile(sample, (.025, .975))
            rows.append(dict(filter=scoring_filter, condition=condition, effect=effect,
                             episodes=len(means), frames=sum(map(len, values)),
                             pooled_frame_effect_mps=float(np.concatenate(values).mean()),
                             episode_macro_effect_mps=float(means.mean()),
                             episode_macro_ci_low=float(low), episode_macro_ci_high=float(high)))
        return rows


def evaluate_saved(model, tasks, recorder, device, width, batch_size, checks):
    for offset in range(0, len(tasks), batch_size):
        batch = tasks[offset:offset + batch_size]
        rows = [t.stream.arrays for t in batch]
        cold = [first_step(r[0][t.start:t.end + 1], False) for t, r in zip(batch, rows)]
        src = [r[1][t.start:t.end + 1] for t, r in zip(batch, rows)]
        resets = [r[5][t.start:t.end + 1] for t, r in zip(batch, rows)]
        local_baselines = [physical_baseline(model, f.to(device),
                          r[2][t.start:t.end + 1].to(device), reset.to(device), width)
                           for f, reset, r, t in zip(cold, resets, rows, batch)]
        for h, fbit in itertools.product((0, 1), repeat=2):
            features = [first_step(r[0][t.start:t.end + 1], bool(fbit)) for t, r in zip(batch, rows)]
            f, source, valid, reset = pad_sequences(features, src, resets, device)
            for i, (t, r) in enumerate(zip(batch, rows)):
                valid[i, :len(features[i])] = r[2][t.start:t.end + 1].to(device)
            initial = torch.stack([t.stream.hidden_before[t.start] for t in batch]) if h else None
            delta, norm, gate, _ = run_sequences(model, f, source, valid, reset, initial)
            for b in (0, 1):
                for i, t in enumerate(batch):
                    n = t.end - t.start + 1
                    baseline = (t.stream.baseline[t.start:t.end + 1] if b else local_baselines[i])
                    prediction = delta[i, :n] + baseline
                    if h == b == fbit == 1:
                        expected = t.stream.prediction[t.start:t.end + 1]
                        deviation = float((prediction - expected).abs().max())
                        checks["H1B1F1_max_abs_deviation_mps"] = max(
                            deviation, checks["H1B1F1_max_abs_deviation_mps"])
                        if deviation > 1e-4:
                            raise ValueError("H1B1F1 did not reproduce full-stream inference")
                    for scope in ("saved_supervised", "saved_endpoint"):
                        recorder.add(t, f"H{h}B{b}F{fbit}", prediction, baseline,
                                     norm[i, :n], gate[i, :n], scope)
        # Reproduce actual saved windows with their original ages and sparse histories.
        original = [t.original for t in batch]
        f, source, valid, reset = pad_sequences(original, [t.original_sources for t in batch],
                                               [torch.zeros(len(x), dtype=torch.bool) for x in original], device)
        base = weighted_velocity_baseline(f, valid, window_s=width,
                    finite_difference_clip_mps=model.finite_difference_clip_mps if model.robust_input_enabled else None)
        delta, norm, gate, _ = run_sequences(model, f, source, valid, reset)
        for i, t in enumerate(batch):
            n = len(t.original)
            # Recorder expects dense stream offsets; scatter sparse anchor predictions.
            length = t.end - t.start + 1
            dense = [torch.zeros((length, *x.shape[1:]), device=device)
                     for x in (delta[i], base[i], norm[i], gate[i])]
            indices = t.selected.to(device)
            dense[0][indices] = delta[i, :n] + base[i, :n]
            dense[1][indices] = base[i, :n]
            dense[2][indices] = norm[i, :n]
            dense[3][indices] = gate[i, :n]
            for scope in ("saved_supervised", "saved_endpoint"):
                recorder.add(t, "saved_original", *dense, scope)


def evaluate_rolling(model, streams, recorder, device, width, lengths, batch_size):
    for history in lengths:
        tasks = []
        for s in streams:
            for end in np.flatnonzero(s.score):
                start = rolling_start(s.arrays[6], s.arrays[5], int(end), history)
                tasks.append(Task(s, start, int(end), torch.tensor([int(end) - start]),
                                  torch.ones(1, dtype=torch.bool)))
        for offset in range(0, len(tasks), batch_size):
            batch = tasks[offset:offset + batch_size]
            features = [first_step(t.stream.arrays[0][t.start:t.end + 1], False) for t in batch]
            src = [t.stream.arrays[1][t.start:t.end + 1] for t in batch]
            reset_rows = [t.stream.arrays[5][t.start:t.end + 1] for t in batch]
            f, source, valid, reset = pad_sequences(features, src, reset_rows, device)
            for i, t in enumerate(batch):
                valid[i, :len(features[i])] = t.stream.arrays[2][t.start:t.end + 1].to(device)
            base = weighted_velocity_baseline(f, valid, window_s=width,
                        finite_difference_clip_mps=model.finite_difference_clip_mps if model.robust_input_enabled else None)
            delta, norm, gate, _ = run_sequences(model, f, source, valid, reset)
            row = torch.arange(len(batch), device=device)
            end = torch.tensor([t.end - t.start for t in batch], device=device)
            arrays = [x[row, end].detach().cpu().numpy() for x in
                      (delta + base, base, norm, gate)]
            # Transfer and aggregate in batches; per-frame CUDA copies distort
            # throughput and can dominate the actual network computation.
            owners = defaultdict(list)
            for i, task in enumerate(batch):
                owners[task.stream.uid].append(i)
            for indices in owners.values():
                s = batch[indices[0]].stream
                frames = np.array([batch[i].end for i in indices])
                recorder.add_values(s, frames, np.ones(len(frames), dtype=bool),
                                    f"rolling_{history:.2f}s", *[x[indices] for x in arrays],
                                    "full_timeline", True)


@torch.inference_mode()
def benchmark_cached_features(model, stream, device, width, lengths, repeats=30):
    """Single-target baseline + model time; excludes sensor/feature IO."""
    end = int(np.flatnonzero(stream.score)[-1])
    result = []
    for protocol, history in [("stream", width), *[(f"rolling_{w:.2f}s", w) for w in lengths]]:
        start = rolling_start(stream.arrays[6], stream.arrays[5], end, history)
        raw = stream.arrays[0][start:end + 1].to(device)
        sources = stream.arrays[1][start:end + 1].to(device)[None]
        valid = stream.arrays[2][start:end + 1].to(device)[None]
        timings = []
        for iteration in range(repeats + 5):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            f = first_step(raw, protocol == "stream")[None]
            baseline = weighted_velocity_baseline(f, valid, window_s=width,
                        finite_difference_clip_mps=model.finite_difference_clip_mps if model.robust_input_enabled else None)
            if protocol == "stream":
                model.step(f[:, -1], sources[:, -1], hidden=stream.hidden_before[end:end + 1],
                           valid_mask=valid[:, -1], velocity_baseline=baseline[:, -1])
            else:
                model(f, sources, valid, velocity_baseline=baseline)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if iteration >= 5:
                timings.append(elapsed_ms)
        result.append(dict(protocol=protocol, device=str(device), iterations=repeats,
                           history_steps=len(raw), p50_ms=float(np.quantile(timings, .5)),
                           p95_ms=float(np.quantile(timings, .95)),
                           scope="one_target_cached_features_plus_baseline_and_model"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-pairs", type=int, default=0, help="0=all; otherwise first N complete trajectory groups")
    parser.add_argument("--stream-batch-size", type=int, default=16)
    parser.add_argument("--window-batch-size", type=int, default=256)
    parser.add_argument("--history-seconds", type=float, nargs="+", default=[.16, .32, .48, .80])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.stream_batch_size, args.window_batch_size, args.threads) < 1 or args.max_pairs < 0:
        parser.error("Batch sizes/threads must be positive and max-pairs nonnegative")
    if any(w <= 0 for w in args.history_seconds) or len(set(args.history_seconds)) != len(args.history_seconds):
        parser.error("History widths must be positive and unique")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory is not empty; use a new directory")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; explicitly use --device cpu")
    payload = load_torch_payload(args.checkpoint)
    if payload.get("baseline_version") != TAQ_BASELINE_VERSION:
        raise ValueError("Checkpoint physical baseline version mismatch")
    model, width = _load_model(args.checkpoint, device)
    started = time.perf_counter()
    paths = sorted(args.data_root.resolve().rglob("*.taqstream"))
    if not paths:
        raise FileNotFoundError("No complete episode streams")
    streams, available_pairs = prepare_pairs(paths, args.max_pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [{"path": str(s.path), "sha256": sha256(s.path),
                 "audit_sha256": sha256(s.path.with_name("velocity_error_audit.jsonl"))} for s in streams]
    recorder, alignment = Recorder(), []
    checks = {"H1B1F1_max_abs_deviation_mps": 0.0}
    latency = []
    for offset in range(0, len(streams), args.stream_batch_size):
        group = streams[offset:offset + args.stream_batch_size]
        with torch.inference_mode():
            cache_prefix(model, group, device, width)
            for s in group:
                n = len(s.arrays[0])
                task = Task(s, 0, n - 1, torch.arange(n), torch.ones(n, dtype=torch.bool))
                recorder.add(task, "stream", s.prediction, s.baseline, s.hidden_norm,
                             s.gate, "full_timeline", True)
            tasks = saved_tasks(group, alignment, manifest)
            evaluate_saved(model, tasks, recorder, device, width, args.window_batch_size, checks)
            evaluate_rolling(model, group, recorder, device, width, args.history_seconds,
                             args.window_batch_size)
            if not latency:
                representative = next((s for s in group if s.score.any()), None)
                if representative is not None:
                    latency = benchmark_cached_features(model, representative, device, width,
                                                        args.history_seconds)
        for s in group:
            s.baseline = s.prediction = s.hidden_before = s.hidden_norm = s.gate = None
        print(f"[E1] streams {min(offset + len(group), len(streams))}/{len(streams)}, "
              f"elapsed {time.perf_counter() - started:.1f}s", flush=True)
    rows = recorder.summary_rows()
    write_csv(args.out_dir / "summary.csv", rows)
    write_csv(args.out_dir / "episode_metrics.csv", recorder.episode_rows())
    write_csv(args.out_dir / "factor_effects.csv", recorder.effects())
    write_csv(args.out_dir / "window_alignment.csv", alignment)
    write_csv(args.out_dir / "latency.csv", latency)
    summary = dict(protocol_version="taq_e1_context_v1", checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=sha256(args.checkpoint), data_root=str(args.data_root.resolve()),
                   device=str(device), selected_pairs=len(streams) // 4, available_pairs=available_pairs,
                   streams=len(streams), baseline_window_s=width, history_seconds=args.history_seconds,
                   elapsed_s=time.perf_counter() - started, checks=checks,
                   windows=len(alignment),
                   windows_with_other_feature_differences=sum(bool(x["other_differing_features"]) for x in alignment),
                   windows_with_omitted_stream_frames=sum(x["omitted_stream_frames"] > 0 for x in alignment),
                   semantics=dict(H="0=zeros; 1=actual stream hidden immediately before window",
                                  B="0=window baseline with first FD/dt zeroed; 1=stream baseline",
                                  F="GRU input only: 0=first FD/dt zero; 1=stream first FD/dt",
                                  fixed="Other features use original stream values, including persistent age; original resets always apply",
                                  anchor="saved_original uses saved features and sparse timeline without injected stream resets",
                                  scoring="Shared earliest direct observation, then A/B/C/D common valid labels; lost/prediction remain scored",
                                  rolling="Past-only <= history width, clipped at latest reset; first FD/dt zeroed, stream ages retained; baseline width stays checkpoint width",
                                  latency="latency.csv is one target with cached device features; excludes tracker, feature builder and CPU/GPU IO; elapsed_s is offline batched processing",
                                  effects="Signed mean-error contrasts; negative main effect favors bit=1; interactions are not error-contribution percentages"),
                   metrics=rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""Check causal boundaries and intervention controls, independent of real data."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_inference_context import (  # noqa: E402
    BITS, Recorder, align_times, first_step, pad_sequences, physical_baseline,
    rolling_start, run_sequences,
)
from speed_prediction.time_aware_velocity_model import TimeAwareQualityGatedGRU  # noqa: E402


def test_timestamp_alignment_retains_distinct_submillisecond_frames():
    times = torch.tensor([1000., 1000.0002, 1000.02], dtype=torch.float64)
    assert align_times(times, times[1:] + 1e-7).tolist() == [1, 2]
    with pytest.raises(ValueError, match="20 us"):
        align_times(times, torch.tensor([1000.01], dtype=torch.float64))


def test_rolling_history_is_past_only_and_does_not_cross_reset():
    times = torch.tensor([0., .1, .2, .7, .8, .9], dtype=torch.float64)
    resets = torch.tensor([True, False, False, True, False, False])
    assert rolling_start(times, resets, 5, .8) == 3
    assert rolling_start(times, resets, 5, .15) == 4
    assert rolling_start(times, resets, 2, .1) == 1


def test_first_step_intervention_preserves_age_and_other_features():
    features = torch.arange(72, dtype=torch.float32).reshape(3, 24)
    original = features.clone()
    changed = first_step(features, False)
    expected = original.clone()
    expected[0, 3:6] = 0
    expected[0, 12] = 0
    assert torch.equal(changed, expected)
    assert torch.equal(features, original)


def test_cached_prefix_reproduces_dense_stream_with_internal_reset_and_padding():
    torch.manual_seed(7)
    model = TimeAwareQualityGatedGRU(dropout=0).eval()
    features = torch.randn(2, 9, 24)
    features[:, :, 12] = .02
    sources = torch.ones(2, 9, dtype=torch.long)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[1, 7:] = False
    resets = torch.zeros_like(valid)
    resets[:, 0] = True
    resets[0, 6] = True
    with torch.inference_mode():
        full, norms, _, before = run_sequences(model, features, sources, valid, resets,
                                               capture_prefix=True)
        start = 3
        sliced, _, _, _ = run_sequences(model, features[:, start:], sources[:, start:],
                                         valid[:, start:], resets[:, start:], before[:, start])
    torch.testing.assert_close(sliced, full[:, start:], atol=1e-6, rtol=1e-6)
    assert norms[1, -1] == norms[1, -2] == norms[1, -3]


def test_padding_and_batch_size_do_not_change_window_end_predictions():
    torch.manual_seed(11)
    model = TimeAwareQualityGatedGRU(dropout=0).eval()
    rows = [torch.randn(4, 24), torch.randn(9, 24)]
    for row in rows:
        row[:, 12] = .02
    sources = [torch.ones(len(row), dtype=torch.long) for row in rows]
    resets = [torch.zeros(len(row), dtype=torch.bool) for row in rows]
    with torch.inference_mode():
        f, src, mask, reset = pad_sequences(rows, sources, resets, torch.device("cpu"))
        batch, _, _, _ = run_sequences(model, f, src, mask, reset)
        for index, row in enumerate(rows):
            f, src, mask, reset = pad_sequences([row], [sources[index]], [resets[index]],
                                               torch.device("cpu"))
            single, _, _, _ = run_sequences(model, f, src, mask, reset)
            torch.testing.assert_close(batch[index, len(row) - 1], single[0, -1],
                                       atol=1e-6, rtol=1e-6)


def test_cut_baseline_buffer_does_not_mutate_gru_reset_bits():
    model = TimeAwareQualityGatedGRU(dropout=0).eval()
    features = torch.zeros(4, 24)
    features[:, 0] = torch.arange(4) * .02
    features[:, 12] = .02
    features[:, 15] = features[:, 22] = features[:, 23] = 1
    resets = torch.zeros(4, dtype=torch.bool)
    baseline = physical_baseline(model, features, torch.ones(4, dtype=torch.bool), resets, .48)
    assert not resets.any()
    torch.testing.assert_close(baseline[-1], torch.tensor([1., 0., 0.]), atol=1e-5, rtol=1e-5)


def test_factorial_effects_include_interactions_instead_of_claiming_error_shares():
    recorder = Recorder()
    values = {f"H{h}B{b}F{f}": 1 + .2*h - .1*b + .3*f + .4*h*b for h, b, f in BITS}
    recorder.endpoint_errors[("unfiltered", "A", "episode", 8)] = values
    effects = {row["effect"]: row["pooled_frame_effect_mps"] for row in recorder.effects(50)}
    assert effects["H"] == pytest.approx(.4)
    assert effects["B"] == pytest.approx(.1)
    assert effects["F"] == pytest.approx(.3)
    assert effects["H:B"] == pytest.approx(.4)
    assert effects["H:B:F"] == pytest.approx(0, abs=1e-12)

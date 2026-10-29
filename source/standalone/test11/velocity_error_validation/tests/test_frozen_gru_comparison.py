"""Protect shared histories, native preprocessing, and causal comparison masks."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_frozen_gru_models import (
    history_tasks, legacy_inputs, prepare_batch, predict_batch, select_candidates, summarize,
)
from optimization_components import ExperimentalGRU
from speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU
from speed_prediction.taq_gru.time_aware_velocity_model import TimeAwareQualityGatedGRU


def make_record(steps=12):
    features = torch.zeros(steps, 24)
    features[:, :3] = torch.tensor([90., 2., 1.]) + torch.arange(steps)[:, None] * torch.tensor([.04, -.02, .01])
    features[:, 3:6] = torch.tensor([2., -1., .5])
    features[:, 12] = .02
    features[:, 15] = .9
    features[:, 22:24] = 1
    score = torch.ones(steps, dtype=torch.bool)
    score[0] = False
    return dict(x=features, source=torch.ones(steps, dtype=torch.long),
                time=torch.arange(steps, dtype=torch.float64) * .02,
                score=score, ground=torch.ones(steps, dtype=torch.bool),
                target=torch.zeros(steps, 3), uid="synthetic", scenario="sd", profile="clean")


def test_histories_never_read_future_or_exceed_width():
    record = make_record(40)
    tasks = history_tasks([record], "rolling_0p48", .48)
    assert len(tasks) == int(record["score"].sum())
    for _, start, end in tasks:
        assert record["score"][end - 1]
        assert float(record["time"][end - 1] - record["time"][start]) <= .48 + 1e-9
    assert tasks[-1] == (0, 15, 40)


def test_adapter_uses_metres_not_velocity_and_no_quality_as_extra_features():
    record = make_record()
    inputs = legacy_inputs(record["x"], 1)
    torch.testing.assert_close(inputs[:, 3:6], record["x"][1:, :3] - record["x"][:-1, :3])
    assert inputs.shape == (11, 13)
    changed = record["x"].clone()
    changed[:, 12:] = 1000
    torch.testing.assert_close(inputs, legacy_inputs(changed, 1))


def test_preparation_is_history_local_and_preserves_age_masks():
    record = make_record(40)
    record["x"][:, 13] = .15
    original = record["x"].clone()
    batch, sequences = prepare_batch([record], [(0, 15, 40)])
    expected = original[15:40].clone()
    expected[0, 3:6] = 0
    expected[0, 12] = 0
    torch.testing.assert_close(sequences[0], expected)
    torch.testing.assert_close(record["x"], original)
    record["x"][:15, :3] = 9999
    changed, _ = prepare_batch([record], [(0, 15, 40)])
    for key in ("x", "old", "legacy_extra", "aligned_extra", "aligned"):
        torch.testing.assert_close(batch[key], changed[key])


def test_batch_predictions_match_native_legacy_and_taq_outputs_with_padding():
    torch.manual_seed(7)
    records = [make_record(12), make_record(8)]
    batch, sequences = prepare_batch(records, [(0, 0, 12), (1, 0, 8)])
    legacy = VelocityPredictorGRU(hidden_dim=16).eval()
    taq = ExperimentalGRU(TimeAwareQualityGatedGRU(dropout=0).config).eval()
    candidates = {"taq": (taq, dict(variant_config={}, scale=1))}
    predictions = predict_batch(legacy, candidates, batch, sequences, 1, torch.device("cpu"))
    with torch.inference_mode():
        for index, sequence in enumerate(sequences):
            native = legacy(legacy_inputs(sequence, 1)[None])[0]
            torch.testing.assert_close(torch.tensor(predictions["legacy_gru"][index][-1]), native)
        mean, _ = taq(batch["x"], batch["source"], batch["valid"], velocity_baseline=batch["old"])
        for index, sequence in enumerate(sequences):
            torch.testing.assert_close(torch.tensor(predictions["taq"][index]), mean[index, :len(sequence)])


def test_scoring_uses_identical_masks_and_rejects_missing_predictions():
    record = make_record()
    record["ground"][5:] = False
    prediction = np.ones((12, 3), dtype=np.float32)
    prediction[0] = np.nan
    rows, _ = summarize([record], {"legacy_gru": [prediction], "taq": [prediction.copy()]}, "rolling_0p48")
    assert {row["n"] for row in rows if row["filter"] == "unfiltered"} == {11}
    assert {row["n"] for row in rows if row["filter"] == "center_above_ground"} == {4}
    prediction[1] = np.nan
    with pytest.raises(ValueError, match="shared scored frame"):
        summarize([record], {"legacy_gru": [prediction]}, "rolling_0p48")


def test_checkpoint_selection_uses_validation_metric_only(tmp_path):
    for seed, validation_metric, test_metric in ((41, 2., .1), (42, 1., 10.)):
        root = tmp_path / f"flow_original_seed{seed}"
        root.mkdir()
        torch.save(dict(variant="flow_original", metric=validation_metric, test_metric=test_metric),
                   root / "best_model.pt")
    path, _ = select_candidates(tmp_path, ["flow_original"])["flow_original"]
    assert path.parent.name == "flow_original_seed42"

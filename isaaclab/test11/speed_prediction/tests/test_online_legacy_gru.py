"""Check live legacy inference against its frozen training input contract."""

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from test11.speed_prediction.legacy_gru.evaluate_vision_gru import VisionGRUMetrics, collector_arguments
from test11.speed_prediction.legacy_gru.compare_vision_gru_initialization import matched_metrics
from test11.speed_prediction.legacy_gru.online_gru_velocity import OnlineLegacyGRU, window_inputs
from test11.speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU


def rows(count=40):
    return [(index * .02, np.array([90 + index * .04, 2 - index * .02, 1 + index * .01,
                                   .1, 0, 0, 0, 0, .01], dtype=np.float32)) for index in range(count)]


def update(adapter, row, episode_key="episode_0", direct_observation=None):
    timestamp, values = row
    return adapter.update(timestamp_s=timestamp, position_base_m=values[:3],
                          robot_linear_velocity_base_mps=values[3:6],
                          robot_angular_velocity_base_rps=values[6:9], episode_key=episode_key,
                          direct_observation=direct_observation)


@pytest.mark.parametrize("smooth_window", [1, 3])
def test_rolling_predictions_match_native_training_adapter(smooth_window):
    torch.manual_seed(11)
    model = VelocityPredictorGRU(hidden_dim=16).eval()
    adapter = OnlineLegacyGRU(model, context="rolling", smooth_window=smooth_window)
    history = rows()
    for index, row in enumerate(history):
        result = update(adapter, row)
        start = next(start for start in range(index + 1) if history[start][0] >= row[0] - .48 - 1e-9)
        if index == start:
            assert result is None
            continue
        with torch.inference_mode():
            expected = model(window_inputs(history[start:index + 1], smooth_window)[None])[0]
        torch.testing.assert_close(torch.from_numpy(result["velocity"]), expected)
        assert result["history_frames"] == index + 1 - start


@pytest.mark.parametrize("smooth_window", [1, 4])
def test_continuous_state_matches_full_sequence_without_future_input(smooth_window):
    torch.manual_seed(12)
    model = VelocityPredictorGRU(hidden_dim=16).eval()
    adapter = OnlineLegacyGRU(model, context="full", smooth_window=smooth_window)
    history = rows()
    for index, row in enumerate(history):
        result = update(adapter, row)
        if index == 0:
            assert result is None
            continue
        with torch.inference_mode():
            expected = model(window_inputs(history[:index + 1], smooth_window)[None])[0]
        torch.testing.assert_close(torch.from_numpy(result["velocity"]), expected, rtol=1e-5, atol=1e-6)


def test_missing_positions_are_zero_filled_and_new_episode_resets_state():
    model = VelocityPredictorGRU(hidden_dim=16).eval()
    adapter = OnlineLegacyGRU(model, context="full")
    history = rows(4)
    for row in history:
        update(adapter, row)
    missing = (.08, np.array([np.nan, np.nan, np.nan, .1, 0, 0, 0, 0, .01], dtype=np.float32))
    assert np.isfinite(update(adapter, missing)["velocity"]).all()
    assert np.array_equal(adapter.rows[-1][1][:3], np.zeros(3))
    assert update(adapter, history[0], episode_key="episode_1") is None
    assert adapter.hidden is None
    with pytest.raises(ValueError, match="strictly increasing"):
        update(adapter, history[0], episode_key="episode_1")


def arguments(context="both"):
    return SimpleNamespace(context=context, device="cpu", window_s=.48, max_history_frames=32,
                           measure_latency=False, burn_in_s=.1, progress_every=0,
                           initialization_policy="first_valid",
                           scenario="sd", profile="clean", condition="D", seed=42,
                           num_envs=1, num_steps=10, headless=True,
                           locomotion_vx_mps=0., locomotion_vy_mps=0., locomotion_yaw_rate_radps=0.,
                           locomotion_policy=Path("policy.pt"), ostrack_repo=Path("ostrack"),
                           ostrack_checkpoint=Path("ostrack.pt"))


def test_live_labels_are_score_only_and_all_ready_frames_remain_in_report(tmp_path):
    model = VelocityPredictorGRU(hidden_dim=16).eval()
    observer = VisionGRUMetrics(model, {"preprocessing": {"pos_smooth_window": 1}}, arguments(), tmp_path)
    metadata = {"validation_gt_direction": False, "validation_gt_range": False}
    for index, (timestamp, values) in enumerate(rows(12)):
        frame = SimpleNamespace(timestamp_s=timestamp, position_base_m=values[:3],
                                robot_linear_velocity_base_mps=values[3:6],
                                robot_angular_velocity_base_rps=values[6:9],
                                source_id=1 if index < 8 else 4, range_valid=index < 8,
                                target_relative_velocity_base_mps=np.array([1., 0., 0.]),
                                audit={"target_position_world_m": [90., 2., 1. if index < 9 else -1.]})
        observer.observe(frame, env_id=0, episode_id=0, metadata=metadata)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen")
    summary = observer.finish(True, checkpoint)
    for context in ("rolling", "full"):
        unfiltered = next(row for row in summary["metrics"] if row["context"] == context
                          and row["filter"] == "unfiltered" and row["group"] == "all")
        ground = next(row for row in summary["metrics"] if row["context"] == context
                      and row["filter"] == "center_above_ground" and row["group"] == "all")
        assert unfiltered["n"] == 7
        assert ground["n"] == 4
    assert json.loads((tmp_path / "summary.json").read_text())["completed"]
    assert observer.finish(False, checkpoint) is summary
    assert json.loads((tmp_path / "summary.json").read_text())["completed"]


def test_observer_rejects_gt_direction_inputs(tmp_path):
    observer = VisionGRUMetrics(VelocityPredictorGRU(hidden_dim=16).eval(),
                               {"preprocessing": {"pos_smooth_window": 1}}, arguments(), tmp_path)
    with pytest.raises(ValueError, match="GT direction"):
        observer.observe(SimpleNamespace(), env_id=0, episode_id=0,
                         metadata={"validation_gt_direction": True})
    observer.handle.close()


def test_live_launcher_preserves_lost_frames_without_gt_visibility_filter():
    command = collector_arguments(arguments(), Path("report"))
    assert "--save_episode_stream" in command
    assert "--keep_measurement_discontinuities" in command
    assert "--no-use_gt_visibility_gate_after_initialization" in command
    assert command[command.index("--validation_condition") + 1] == "D"
    assert command[command.index("--pair_id") + 1] == "legacy_gru_vision_42"


@pytest.mark.parametrize("context", ["rolling", "full"])
@pytest.mark.parametrize("smooth_window", [1, 3])
def test_initialization_wait_does_not_pollute_history_or_hidden_state(context, smooth_window):
    model = VelocityPredictorGRU(hidden_dim=16).eval()
    adapter = OnlineLegacyGRU(model, context=context, smooth_window=smooth_window)
    for row in rows(50):
        assert update(adapter, row, direct_observation=False) is None
        assert len(adapter.rows) == 0
        assert adapter.hidden is None
        assert adapter.frame_count == 0
        assert adapter.first_time is None
    valid_rows = rows(60)[50:]
    for index, row in enumerate(valid_rows):
        result = update(adapter, row, direct_observation=True)
        if index == 0:
            assert result is None
            assert adapter.hidden is None
            continue
        with torch.inference_mode():
            expected = model(window_inputs(valid_rows[:index + 1], smooth_window)[None])[0]
        torch.testing.assert_close(torch.from_numpy(result["velocity"]), expected, rtol=1e-5, atol=1e-6)
    missing = (1.2, np.array([np.nan, np.nan, np.nan, .1, 0, 0, 0, 0, .01], dtype=np.float32))
    assert np.isfinite(update(adapter, missing, direct_observation=False)["velocity"]).all()
    assert np.array_equal(adapter.rows[-1][1][:3], np.zeros(3))
    assert update(adapter, rows(1)[0], episode_key="episode_1", direct_observation=False) is None
    assert adapter.hidden is None
    assert not adapter.rows
    assert not adapter.started


@pytest.mark.parametrize("context", ["rolling", "full"])
def test_first_valid_requires_finite_position_and_old_initialization_is_available(context):
    model = VelocityPredictorGRU(hidden_dim=16).eval()
    missing = (0., np.array([np.nan, np.nan, np.nan, .1, 0, 0, 0, 0, .01], dtype=np.float32))
    adapter = OnlineLegacyGRU(model, context=context)
    assert update(adapter, missing, direct_observation=True) is None
    assert not adapter.rows
    baseline = OnlineLegacyGRU(model, context=context, initialization_policy="zero_filled")
    assert update(baseline, missing, direct_observation=False) is None
    assert len(baseline.rows) == 1
    assert update(baseline, rows(2)[1], direct_observation=False) is not None


def test_observer_reports_initialization_delay_without_dropping_later_loss(tmp_path):
    observer = VisionGRUMetrics(VelocityPredictorGRU(hidden_dim=16).eval(),
                               {"preprocessing": {"pos_smooth_window": 1}}, arguments(), tmp_path)
    for index, (timestamp, values) in enumerate(rows(55)):
        direct = 50 <= index < 53
        observer.observe(SimpleNamespace(
            timestamp_s=timestamp, position_base_m=values[:3] if direct else np.full(3, np.nan),
            robot_linear_velocity_base_mps=values[3:6], robot_angular_velocity_base_rps=values[6:9],
            source_id=1 if direct else 4, range_valid=direct,
            target_relative_velocity_base_mps=np.array([1., 0., 0.]),
            audit={"target_position_world_m": [90., 2., 1.]},
        ), env_id=0, episode_id=0, metadata={})
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen")
    summary = observer.finish(True, checkpoint)
    assert summary["counters"]["initialization_wait_frames"] == 50
    for context in ("rolling", "full"):
        assert summary["counters"][f"scored_{context}"] == 4
        assert summary["episode_timing"][0]["first_prediction_s"][context] == pytest.approx(1.02)
        assert summary["episode_timing"][0]["first_output_after_direct_s"][context] == pytest.approx(.02)
        lost = next(row for row in summary["metrics"] if row["context"] == context
                    and row["filter"] == "unfiltered" and row["group"] == "lost")
        assert lost["n"] == 2


def test_initialization_comparison_uses_identical_scored_timestamps(tmp_path):
    summaries = {}
    for policy in ("zero_filled", "first_valid"):
        directory = tmp_path / policy
        directory.mkdir()
        summaries[policy] = {"episode_timing": [{"env_id": 0, "episode_id": 0, "first_direct_s": .98}]}
        columns = ("env_id", "episode_id", "timestamp_s", "context", "scored",
                   "prediction_vx", "prediction_vy", "prediction_vz", "target_vx", "target_vy", "target_vz")
        with (directory / "frame_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for context in ("rolling", "full"):
                for timestamp, error in ((.98, 100.), (1., 2.), (1.6, .1)):
                    writer.writerow(dict(
                        env_id=0, episode_id=0, timestamp_s=timestamp, context=context,
                        scored=policy == "zero_filled" or timestamp != .98,
                        prediction_vx=error, prediction_vy=0., prediction_vz=0.,
                        target_vx=0., target_vy=0., target_vz=0.,
                    ))
    for row in matched_metrics(tmp_path, summaries):
        assert row["n"] == (2 if row["group"] == "all" else 1)
        expected = {"all": 1.05, "startup_0p5s": 2., "after_startup": .1}[row["group"]]
        assert row["mean"] == pytest.approx(expected)

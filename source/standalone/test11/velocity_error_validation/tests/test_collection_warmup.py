"""Check per-episode startup exclusion and clean capture/serialization clocks."""
import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

TEST11 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST11))
from speed_prediction.sensor_time_contract import EpisodeCollectionWarmup, TIME_CONTRACT_VERSION
from speed_prediction.stage2_degradation import (
    Stage2Degrader, DATA_SPLITS, DEGRADATION_PROFILES, DEGRADATION_SEVERITIES,
)
from speed_prediction.time_aware_velocity_data import (
    TAQ_SCHEMA_VERSION, TAQ_LABEL_SEMANTICS, TAQ_STREAM_FORMAT,
    TrackerSource, build_time_aware_features,
)


def collector_functions():
    # Execute the actual parser and serializer, without importing Isaac Sim.
    path = TEST11 / "collect_vision_data.py"
    tree = ast.parse(path.read_text())
    names = {"_taq_args", "_validate_taq_args", "_TAQFrame", "_json_safe", "_pack_and_save_taq_stream"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    scope = dict(globals())
    exec(compile(module, str(path), "exec"), scope)
    return scope


def test_one_second_excludes_fifty_steps_after_each_independent_reset():
    gate = EpisodeCollectionWarmup(1., .02)
    assert gate.steps == 50
    assert gate.effective_s == 1.
    # Environments reset at different wall/physics steps. Their own age wins.
    ages = [0, 40]
    recorded = [[], []]
    for global_step in range(120):
        for env_id in range(2):
            ages[env_id] += 1
            if ages[env_id] == 80:
                ages[env_id] = 0
                continue
            if gate.ready(ages[env_id]):
                recorded[env_id].append((global_step, ages[env_id]))
    assert all(age > 50 for rows in recorded for _, age in rows)
    assert recorded[0][0] == (50, 51)
    assert recorded[1][0] == (10, 51)
    assert any(step > 39 and age == 51 for step, age in recorded[1])


def test_fractional_warmup_rounds_up_and_zero_allows_first_capture():
    gate = EpisodeCollectionWarmup(.205, .02)
    assert gate.steps == 11
    assert not gate.ready(11)
    assert gate.ready(12)
    assert EpisodeCollectionWarmup(0., .02).ready(1)


@pytest.mark.parametrize("warmup,dt", [(-1, .02), (math.nan, .02), (math.inf, .02), (1, 0), (1, math.nan)])
def test_invalid_timing_is_rejected(warmup, dt):
    with pytest.raises(ValueError):
        EpisodeCollectionWarmup(warmup, dt)


def test_collector_defaults_and_warmup_validation(monkeypatch, tmp_path):
    scope = collector_functions()
    monkeypatch.setattr(sys, "argv", ["collect", "--scenario", "sd", "--out_dir", str(tmp_path)])
    args = scope["_taq_args"]()
    assert args.degradation_profile == "clean"
    assert args.capture_skip_probability == 0
    assert args.episode_warmup_s == 1.
    assert args.target_boundary_mode == "fixed_reference_toi"
    assert args.target_fov_margin_fraction == .1
    args.locomotion_policy = tmp_path / "policy.pt"
    args.locomotion_policy.touch()
    scope["_validate_taq_args"](args)
    for bad in (-.1, math.nan, math.inf):
        args.episode_warmup_s = bad
        with pytest.raises(ValueError, match="episode_warmup_s"):
            scope["_validate_taq_args"](args)


def test_clean_clock_tracks_capture_time_even_across_physical_gaps():
    degrader = Stage2Degrader(profile="clean", severity="hard", seed=10, num_envs=1, nominal_dt_s=.02)
    degrader.reset_episode(0, 0)
    assert degrader.next_timestamp(0, .02) == (.02, .02)
    timestamp, dt = degrader.next_timestamp(0, .10)
    assert timestamp == .10
    assert dt == pytest.approx(.08)
    with pytest.raises(ValueError, match="increase"):
        degrader.next_timestamp(0, .10)
    for i in range(6, 1000):
        timestamp, dt = degrader.next_timestamp(0, i * .02)
        assert timestamp == i * .02
        assert dt == pytest.approx(.02)
    episode = degrader.episode_metadata(0)
    assert episode["enabled_components"] == []
    assert episode["image_delay_steps"] == episode["range_delay_steps"] == episode["proprio_delay_steps"] == 0


def test_stream_contains_only_post_warmup_frames_including_real_tracker_loss(tmp_path):
    scope = collector_functions()
    gate = EpisodeCollectionWarmup(1., .02)
    frames = []
    for step in range(1, 101):
        if not gate.ready(step):
            continue
        lost = step == 70
        frames.append(scope["_TAQFrame"](
            timestamp_s=step*.02,
            position_base_m=np.full(3, np.nan) if lost else np.array([100+step*.02, 0., 0.]),
            robot_linear_velocity_base_mps=np.zeros(3),
            robot_angular_velocity_base_rps=np.zeros(3),
            source_id=int(TrackerSource.INVALID if lost else TrackerSource.FLOW),
            quality=np.ones(7), range_valid=not lost,
            target_relative_velocity_base_mps=np.array([1., 0., 0.]),
            audit={"episode_elapsed_s": step*.02},
        ))
    count = scope["_pack_and_save_taq_stream"](
        frames, output_dir=tmp_path, env_id=0, episode_id=0,
        metadata={"supervision_burn_in_s": .1, "episode_warmup_s": 1.},
    )
    assert count == 50
    stream = torch.load(next(tmp_path.rglob("*.taqstream")), weights_only=False)
    assert stream["timestamps_s"][0].item() == 1.02
    torch.testing.assert_close(torch.diff(stream["timestamps_s"]), torch.full((49,), .02, dtype=torch.float64))
    assert stream["state_reset_mask"][0]
    assert (stream["sequence_source_ids"] == int(TrackerSource.INVALID)).sum() == 1
    assert torch.isfinite(stream["sequence_features"]).all()
    rows = [json.loads(line) for line in next(tmp_path.rglob("velocity_error_audit.jsonl")).read_text().splitlines()]
    assert len(rows) == 50
    assert min(row["episode_elapsed_s"] for row in rows) > 1.

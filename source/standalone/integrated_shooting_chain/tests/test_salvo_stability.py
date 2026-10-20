"""Regressions for cumulative box inflation and bounded visual engagement."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrated_shooting_chain.salvo_engagement import (
    SalvoEngagementConfig, SalvoEngagementScheduler, required_player_steps,
)
from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.multi_target_tracker import MultiTargetConfig, MultiTargetVisionTracker


def make_tracker(**config):
    return MultiTargetVisionTracker(
        CameraIntrinsics(fx=1000., fy=1000., cx=640., cy=360., width=1280, height=720),
        config=MultiTargetConfig(**config),
    )


def test_alternating_rotation_returns_points_and_box_to_original_size(monkeypatch):
    tracker = make_tracker(selection_policy="nearest_until_removed")
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    center = np.array([640., 360.], dtype=np.float32)
    initial = np.array([(x, y) for x in (-4., 0., 4.) for y in (-4., 0., 4.)], dtype=np.float32) + center
    points = initial.copy()
    monkeypatch.setattr(tracker, "_extract_points", lambda *args: points.copy())
    tracker.initialize(image, {0: (634., 354., 646., 366.)})
    for frame in range(1000):
        angle = np.deg2rad(.1 if frame % 2 == 0 else -.1)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
        next_points = (points - center) @ rotation.T + center
        flow = (points.copy(), next_points.copy(), np.zeros(9), 9)
        monkeypatch.setattr(tracker, "_track_all_points", lambda *args: {0: flow})
        points = next_points
        estimate = tracker.update(image, dt_s=.02).selected_target
    assert np.allclose(points, initial, atol=1e-4)
    assert np.allclose(estimate.bbox_xyxy, (634., 354., 646., 366.), atol=.03)
    assert estimate.confidence > .99
    assert estimate.status == "tracking"


def test_similarity_update_preserves_real_scale_and_translation():
    tracker = make_tracker()
    points = np.array([(x, y) for x in (634., 640., 646.) for y in (354., 360., 366.)], dtype=np.float32)
    center = np.array([640., 360.], dtype=np.float32)
    moved = (points-center) * 1.1 + center + [2., -3.]
    result = tracker._fit_motion((634., 354., 646., 366.),
                                 (points, moved.astype(np.float32), np.zeros(9), 9), 1280, 720)
    assert np.allclose(result[0], (635.4, 350.4, 648.6, 363.6), atol=1e-3)


def test_static_scale_guard_rejects_cumulative_growth_and_bounds_predictions(monkeypatch):
    tracker = make_tracker(max_initial_scale_ratio=2.0)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker.initialize(image, {0: (634., 354., 646., 366.)})
    # Even a high-confidence fit must not grow a fixed-distance sphere 3x.
    monkeypatch.setattr(tracker, "_fit_motion", lambda *args: (
        (622., 342., 658., 378.), 1., 9, 0., np.zeros((9, 2), np.float32), 1., 1.))
    tracker._states[0].kalman.x[6:8] = 1000.
    for _ in range(10):
        result = tracker.update(image, dt_s=.02)
        box = result.targets[0].bbox_xyxy
        assert box[2]-box[0] <= 24.001
        assert box[3]-box[1] <= 24.001
    assert result.targets[0].status == "lost"
    assert result.targets[0].measurement_rejected
    assert result.selected_target is None


def test_defer_retry_and_exhaustion_do_not_remove_unhit_targets():
    scheduler = SalvoEngagementScheduler(SalvoEngagementConfig(1., 2., 2))
    candidates = {0: 80., 1: 100.}
    assert scheduler.choose(0., candidates) == 0
    assert scheduler.choose(.99, candidates) == 0
    assert scheduler.choose(1., candidates) == 1
    assert scheduler.choose(1.2, candidates, removed_ids=[1]) is None
    assert scheduler.terminal_reason({0: 80.}) is None  # Await retry deadline.
    assert scheduler.choose(3., candidates) == 0
    assert scheduler.choose(4., candidates) is None
    assert scheduler.terminal_reason(candidates) == "attempts_exhausted"
    assert scheduler.removed == {1}
    assert scheduler.summary()["exhausted_target_ids"] == [0]
    assert scheduler.attempts == {0: 2, 1: 1}


def test_last_attempt_is_not_exhausted_until_it_has_failed():
    scheduler = SalvoEngagementScheduler(SalvoEngagementConfig(1., 1., 1))
    assert scheduler.choose(0., {0: 80.}) == 0
    assert scheduler.summary()["exhausted_target_ids"] == []
    assert scheduler.terminal_reason({0: 80.}) is None
    assert scheduler.choose(1., {0: 80.}) is None
    assert scheduler.summary()["exhausted_target_ids"] == [0]
    scheduler.reset()
    scheduler.choose(0., {0: 80.})
    scheduler.choose(1., {}, removed_ids=[0])
    assert scheduler.summary()["exhausted_target_ids"] == []


def test_loss_selects_another_visual_target_and_nearest_order_uses_ranges():
    scheduler = SalvoEngagementScheduler()
    assert scheduler.choose(2., {0: 120., 9: 80.}) == 9
    assert scheduler.choose(2.02, {0: 120.}) == 0
    assert scheduler.events[-2]["reason"] == "tracking_lost"
    assert not scheduler.removed


def test_player_limit_covers_600_simulation_seconds():
    assert required_player_steps(600., .02, 1) > 30000
    assert required_player_steps(600., .02, 3) > 90000


@pytest.mark.parametrize("kwargs", [dict(stall_timeout_s=0), dict(retry_delay_s=float("nan")), dict(max_attempts=0)])
def test_invalid_retry_configuration(kwargs):
    with pytest.raises(ValueError):
        SalvoEngagementConfig(**kwargs)

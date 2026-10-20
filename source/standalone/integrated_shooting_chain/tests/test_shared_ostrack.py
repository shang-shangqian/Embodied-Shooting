"""Shared weights must never share per-environment tracking state."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

STANDALONE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STANDALONE))

from integrated_shooting_chain import isaac_training_bridge as bridge_module
from test_lk_ostrack.ostrack_tracker import OSTrackBoxTracker, OSTrackFrameTracker


def make_bridge(monkeypatch, *, device="cpu", num_envs=2):
    monkeypatch.setattr(bridge_module, "load_velocity_estimator", lambda *a, **k: SimpleNamespace(reset=lambda i: None))
    repo = STANDALONE / "test_lk_ostrack/ostrack/OSTrack-main"
    config = bridge_module.IsaacTrainingBridgeConfig(
        velocity_checkpoint="unused", device=device, ostrack_repo=repo,
        ostrack_checkpoint=repo / "output/checkpoints/OSTrack_ep0300.pth.tar",
    )
    env = SimpleNamespace(
        num_envs=num_envs, device=device,
        _rl_pred_ready=torch.zeros(num_envs, dtype=torch.bool, device=device),
        _rl_bridge_training_ready=torch.zeros(num_envs, dtype=torch.bool, device=device),
        _rl_bridge_velocity_ready=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )
    return bridge_module.IsaacFullChainPerceptionBridge(env, config)


def add_tracker(bridge):
    tracker = bridge._make_tracker(320, 240, fx=200., fy=200., cx=160., cy=120.)
    return tracker.fine_tracker


def test_bridge_shares_network_but_keeps_state_and_lifetime_independent(monkeypatch):
    networks = []

    class Tracker:
        def __init__(self, network):
            self.network = network

        def initialize(self, image, info):
            self.state = list(info["init_bbox"])
            self.template = image.copy()
            self.frame_id = 0

        def track(self, image, info):
            self.frame_id += 1
            self.state[0] += 1
            return {"target_bbox": list(self.state)}

    def build(adapter, *, shared_network=None):
        if shared_network is None:
            shared_network = object()
            networks.append(shared_network)
        return Tracker(shared_network)

    monkeypatch.setattr(OSTrackBoxTracker, "_build_tracker", build)
    bridge = make_bridge(monkeypatch)
    first, second = add_tracker(bridge), add_tracker(bridge)
    assert len(networks) == 1
    assert first.network is second.network is bridge._ostrack_network
    assert first._tracker is not second._tracker
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    first.initialize(image, (10, 20, 40, 50))
    second.initialize(image + 80, (100, 120, 140, 160))
    first.track(image)
    assert second._tracker.state == [100., 120., 40., 40.]
    assert second._tracker.frame_id == 0
    assert not np.array_equal(first._tracker.template, second._tracker.template)

    # A reset/reinitialization in env 0 must not affect env 1 or reload weights.
    bridge._trackers[:] = [SimpleNamespace(reset=first.reset), SimpleNamespace(reset=second.reset)]
    bridge.reset_envs([0])
    first.initialize(image, (50, 60, 80, 90))
    assert second._initialized
    assert second.track(image).bbox_xyxy == (101., 120., 141., 160.)
    assert len(networks) == 1
    other_bridge = make_bridge(monkeypatch)
    assert add_tracker(other_bridge).network is not first.network
    assert len(networks) == 2


@pytest.mark.skipif(os.environ.get("RUN_OSTRACK_CUDA_TESTS") != "1", reason="Opt-in real checkpoint/CUDA regression")
def test_real_checkpoint_shared_inference_matches_independent_networks(monkeypatch):
    bridge = make_bridge(monkeypatch, device="cuda:0", num_envs=16)
    monkeypatch.syspath_prepend(str(bridge.config.ostrack_repo))
    import lib.test.tracker.ostrack as official

    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    with patch.object(official, "build_ostrack", wraps=official.build_ostrack) as build:
        trackers = [add_tracker(bridge) for _ in range(16)]
        assert build.call_count == 1
    shared = trackers[0].network
    assert all(tracker.network is shared for tracker in trackers)
    assert not shared.training
    assert not any(param.requires_grad for param in shared.parameters())
    assert len({id(tracker._tracker) for tracker in trackers}) == 16
    torch.cuda.synchronize()
    print(f"16 shared trackers allocated {(torch.cuda.memory_allocated()-start)/2**20:.2f} MiB")

    # Real isolated models serve as a reference for interleaved shared inference.
    refs = [OSTrackFrameTracker(bridge.config.ostrack_repo, bridge.config.ostrack_checkpoint) for _ in range(2)]
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 256, (240, 320, 3), dtype=np.uint8) for _ in range(2)]
    boxes = [(30, 40, 62, 76), (160, 120, 204, 160)]
    for i in range(2):
        trackers[i].initialize(images[i], boxes[i])
        refs[i].initialize(images[i], boxes[i])
    assert trackers[0]._tracker.z_dict1 is not trackers[1]._tracker.z_dict1
    assert trackers[0]._tracker.box_mask_z is not trackers[1]._tracker.box_mask_z
    for step in range(3):
        for i in range(2):
            image = np.roll(images[i], step + 1, axis=1)
            np.testing.assert_allclose(trackers[i].track(image).bbox_xyxy, refs[i].track(image).bbox_xyxy, atol=1e-5, rtol=0)
    trackers[0].reset()
    trackers[0].initialize(images[1], boxes[1])
    refs[0].reset()
    refs[0].initialize(images[1], boxes[1])
    for i in range(2):
        np.testing.assert_allclose(trackers[i].track(images[i]).bbox_xyxy, refs[i].track(images[i]).bbox_xyxy, atol=1e-5, rtol=0)

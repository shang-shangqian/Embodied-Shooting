"""Verify the one-frame 100-box contract using real CPU LK tracking."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrated_shooting_chain import isaac_training_bridge
from integrated_shooting_chain.isaac_training_bridge import IsaacTrainingBridgeConfig
from integrated_shooting_chain.static_salvo_perception import IsaacStaticSalvoPerceptionBridge
from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.multi_target_tracker import MultiTargetConfig, MultiTargetVisionTracker


class CountingTracker(MultiTargetVisionTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initializations = []
        self.updates = 0
        self.resets = 0
        self.results = []

    def initialize(self, image, boxes, **kwargs):
        self.initializations.append(dict(boxes))
        result = super().initialize(image, boxes, **kwargs)
        self.results.append(result)
        return result

    def update(self, image, **kwargs):
        self.updates += 1
        result = super().update(image, **kwargs)
        self.results.append(result)
        return result

    def reset(self):
        self.resets += 1
        super().reset()


@pytest.fixture
def pool_bridge(monkeypatch):
    reset_keys = []
    estimator = SimpleNamespace(
        reset=reset_keys.append,
        update=lambda *args: SimpleNamespace(velocity_mean_mps=torch.zeros(3), reliability_gate=1.),
    )
    monkeypatch.setattr(isaac_training_bridge, "load_velocity_estimator", lambda *args, **kwargs: estimator)
    count = 100
    camera = dict(fx=2500., fy=2500., cx=320., cy=320.)
    # Position the rendered patches on a grid; IDs with larger boxes are nearer.
    positions = []
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    rng = np.random.default_rng(17)
    for target_id in range(count):
        cx = 50 + 60 * (target_id % 10)
        cy = 50 + 60 * (target_id // 10)
        depth = 150. - target_id * 0.7
        positions.append(((cx - 320.) / 2500. * depth,
                          (cy - 320.) / 2500. * depth, depth))
        half = int(2500. * 0.35 / depth)
        patch = rng.integers(30, 255, size=(2*half, 2*half, 1), dtype=np.uint8)
        image[cy-half:cy+half, cx-half:cx+half] = patch
    pool = SimpleNamespace(
        positions=torch.tensor([positions]), alive=torch.ones(1, count, dtype=torch.bool),
        ready=torch.ones(1, dtype=torch.bool), bootstrap_frames=torch.zeros(1, dtype=torch.long),
        bootstrap_targets=torch.zeros(1, dtype=torch.long),
    )
    selections = []
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=.02, _static_salvo=pool,
        _rl_pred_ready=torch.zeros(1, dtype=torch.bool),
        _rl_bridge_training_ready=torch.zeros(1, dtype=torch.bool),
        _rl_bridge_velocity_ready=torch.zeros(1, dtype=torch.bool),
        set_static_salvo_selected_target=lambda env_id, target_id: selections.append(target_id),
        _get_runtime_camera_pose_w=lambda: (torch.zeros(1, 3), torch.tensor([[1., 0., 0., 0.]])),
        _rotate_world_to_local=lambda rotations, values: values.clone(),
    )
    config = IsaacTrainingBridgeConfig(velocity_checkpoint="unused", device="cpu", enable_fine_tracking=False)
    bridge = IsaacStaticSalvoPerceptionBridge(env, config)
    tracker = CountingTracker(CameraIntrinsics(**camera, width=640, height=640),
                              config=MultiTargetConfig(selection_policy="nearest_until_removed"))
    bridge._make_tracker = lambda *args, **kwargs: tracker
    return bridge, tracker, image, camera, selections, reset_keys


def test_all_100_boxes_once_and_other_track_histories_survive_a_hit(pool_bridge, monkeypatch):
    bridge, tracker, image, camera, selections, reset_keys = pool_bridge
    pool = bridge.base_env._static_salvo
    first = bridge._track_frame(0, image, 2., camera, {})
    assert len(tracker.initializations) == 1
    assert set(tracker.initializations[0]) == set(range(100))
    assert pool.bootstrap_frames.item() == 1
    assert pool.bootstrap_targets.item() == 100
    assert len(tracker.results[0].targets) == 100
    # The nearest selection is derived from visible box size, not ID order.
    distances = {t.target_id: t.distance_m for t in tracker.results[0].targets}
    assert first.target_id == min(distances, key=lambda key: (distances[key], key))
    survivors = {key: value for key, value in tracker._states.items() if key != first.target_id}
    initial_resets = tracker.resets
    monkeypatch.setattr(bridge, "_initial_pool_boxes", lambda *args: pytest.fail("GT read after bootstrap"))
    # Alter GT ranges so a later privileged sort would yield a different answer.
    pool.positions[:] = torch.flip(pool.positions, dims=[1])
    pool.alive[0, first.target_id] = False
    second = bridge._track_frame(0, image, 2.02, camera, {})
    assert second is not None and second.target_id != first.target_id
    assert tracker.resets == initial_resets
    assert len(tracker.initializations) == 1
    assert tracker.updates == 1
    assert len(tracker.results[-1].targets) == 99
    assert all(tracker._states[key] is value for key, value in survivors.items())
    assert reset_keys == [0, 0]  # Reset only the selected target's velocity history.
    assert selections == [first.target_id, second.target_id]
    for step in range(2, 5):
        bridge._track_frame(0, image, 2. + step*.02, camera, {})
    assert tracker.updates == 4
    assert len(tracker.initializations) == 1


def test_tracking_loss_never_reinitializes_pool_and_keeps_frame_clock(pool_bridge, monkeypatch):
    bridge, tracker, image, camera, selections, _ = pool_bridge
    bridge._track_frame(0, image, 2., camera, {})
    monkeypatch.setattr(bridge, "_initial_pool_boxes", lambda *args: pytest.fail("GT recovery"))
    # Force all tracks into terminal loss; keep them in the pool as unhit targets.
    for track in tracker._states.values():
        track.points = np.empty((0, 2), dtype=np.float32)
        track.missed_frames = 100
        track.status = "lost"
    reset_count = tracker.resets
    for step in range(1, 4):
        assert bridge._track_frame(0, image, 2. + step*.02, camera, {}) is None
        bridge._invalidate_env(0, reset_tracker=True)
    assert selections[-1] is None
    assert bridge.base_env._static_salvo.alive.all()
    assert tracker.resets == reset_count
    assert len(tracker.initializations) == 1
    assert tracker.updates == 3
    assert bridge._pool_timestamps[0] == pytest.approx(2.06)


def test_failed_bootstrap_is_not_retried_on_another_gt_frame(pool_bridge, monkeypatch):
    bridge, tracker, image, camera, _, _ = pool_bridge
    def failure(*args):
        raise RuntimeError("bad initial frame")
    monkeypatch.setattr(bridge, "_initial_pool_boxes", failure)
    with pytest.raises(RuntimeError, match="bad initial frame"):
        bridge._track_frame(0, image, 2., camera, {})
    monkeypatch.setattr(bridge, "_initial_pool_boxes", lambda *args: pytest.fail("second GT attempt"))
    assert bridge._track_frame(0, image, 2.02, camera, {}) is None
    assert not tracker.initializations
    assert bridge.base_env._static_salvo.bootstrap_frames.item() == 1


def test_new_episode_is_the_only_operation_that_allows_new_gt_boxes(pool_bridge):
    bridge, tracker, image, camera, _, _ = pool_bridge
    bridge._track_frame(0, image, 2., camera, {})
    bridge.reset_envs([0])
    bridge._track_frame(0, image, 4., camera, {})
    assert len(tracker.initializations) == 2
    assert bridge._pool_timestamps[0] == 4.


def test_deferral_and_retry_preserve_all_visual_histories_without_new_gt(pool_bridge, monkeypatch):
    from integrated_shooting_chain.salvo_engagement import SalvoEngagementConfig
    bridge, tracker, image, camera, selections, _ = pool_bridge
    bridge._schedulers[0].config = SalvoEngagementConfig(.1, .04, 2)
    first = bridge._track_frame(0, image, 2., camera, {})
    original_states = dict(tracker._states)
    original_resets = tracker.resets
    monkeypatch.setattr(bridge, "_initial_pool_boxes", lambda *args: pytest.fail("GT on retry"))
    second = bridge._track_frame(0, image, 2.1, camera, {})
    assert second.target_id != first.target_id
    # A confirmed hit on the second target selects a current-round target,
    # without jumping back to the deferred nearest target after its deadline.
    bridge.base_env._static_salvo.alive[0, second.target_id] = False
    third = bridge._track_frame(0, image, 2.2, camera, {})
    assert third.target_id not in (first.target_id, second.target_id)
    scheduler = bridge._schedulers[0]
    assert scheduler.retry_ids == [first.target_id]
    assert scheduler.selection_round == 1
    # Clear the other queued targets so the first ID finally gets its retry.
    pool = bridge.base_env._static_salvo
    pool.alive[:] = False
    pool.alive[0, first.target_id] = True
    retry = bridge._track_frame(0, image, 2.22, camera, {})
    assert retry.target_id == first.target_id
    assert scheduler.selection_round == 2
    assert tracker.resets == original_resets
    assert len(tracker.initializations) == 1
    assert scheduler.attempts[first.target_id] == 2
    assert bridge.base_env._static_salvo.alive[0, first.target_id]
    assert tracker._states[first.target_id] is original_states[first.target_id]


def test_full_bridge_uses_rgb_range_and_blocks_firing_until_new_identity_is_ready(pool_bridge, monkeypatch):
    bridge, tracker, image, camera, selections, _ = pool_bridge
    env = bridge.base_env
    env.common_step_counter = 100
    env._aim_warmup_left_steps = torch.zeros(1, dtype=torch.long)
    env._vision_camera = SimpleNamespace(data=SimpleNamespace(intrinsic_matrices=torch.tensor([
        [[camera["fx"], 0., camera["cx"]], [0., camera["fy"], camera["cy"]], [0., 0., 1.]]])))
    env.turret = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.zeros(1, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
        root_lin_vel_b=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
    ))
    env._ensure_vision_supervision_ready = lambda: None
    env._rotate_local_to_world = lambda quat, values: values.clone()
    env._rotate_world_to_yaw_only_local = lambda quat, values: values.clone()
    env.set_dynamic_target_raycast_angles = lambda yaw, pitch: None
    env.run_dynamic_target_raycast_for_current_frame = lambda: None
    def batch():
        # Intentionally omit GT direction and distance labels. Bootstrap and
        # all later actor observations must work without the single-target GT path.
        return dict(images=torch.from_numpy(image.copy()).permute(2, 0, 1)[None],
                    valid_mask=torch.zeros(1, dtype=torch.bool), lidar_distance=torch.tensor([100.]),
                    lidar_valid_mask=torch.tensor([bool(selections and selections[-1] is not None)]))
    env.get_vision_supervision_batch = batch
    def write_prediction(position, velocity, distance, yaw, pitch, *, env_ids):
        env._rl_pred_ready[env_ids] = True
    env.set_rl_predicted_target_state = write_prediction
    assert not bridge.step()["ready"]
    monkeypatch.setattr(bridge, "_initial_pool_boxes", lambda *args: pytest.fail("late GT boxes"))
    for step in range(101, 108):
        env.common_step_counter = step
        bridge.step()
    assert env._rl_pred_ready.item()
    first_id = selections[-1]
    env._static_salvo.alive[0, first_id] = False
    env.common_step_counter = 108
    output = bridge.step()
    assert not output["ready"]
    assert not env._rl_pred_ready.item()
    assert bridge._valid_observations[0] == 1
    assert selections[-1] != first_id
    assert len(tracker.initializations) == 1
    assert len(tracker.results[-1].targets) == 99
    for step in range(109, 116):
        env.common_step_counter = step
        bridge.step()
    assert env._rl_pred_ready.item()

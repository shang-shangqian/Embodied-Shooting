"""Check ordinary GRU loading, independent histories and Isaac write-back."""

from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

STANDALONE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STANDALONE_ROOT))
sys.path.insert(0, str(STANDALONE_ROOT / "test11"))

from integrated_shooting_chain.isaac_training_bridge import IsaacFullChainPerceptionBridge, IsaacTrainingBridgeConfig
from integrated_shooting_chain import factory
from integrated_shooting_chain.legacy_gru_adapter import OnlineLegacyGRUEstimator
from integrated_shooting_chain.model_loaders import load_velocity_estimator
from integrated_shooting_chain.shooting_chain import RobotState, ShootingChain
from speed_prediction.common.time_aware_velocity_data import TAQ_LABEL_SEMANTICS, TAQ_SCHEMA_VERSION
from speed_prediction.legacy_gru.online_gru_velocity import window_inputs
from speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU


@pytest.fixture
def checkpoint_path(tmp_path):
    torch.manual_seed(17)
    model = VelocityPredictorGRU(hidden_dim=16).eval()
    model.set_input_norm_stats(torch.linspace(-1., 1., 13), torch.linspace(1., 2., 13))
    path = tmp_path / "legacy_gru.pt"
    torch.save(dict(
        config=dict(input_dim=13, hidden_dim=16, num_layers=1, dropout=0.),
        model_state_dict=model.state_dict(), data_schema=TAQ_SCHEMA_VERSION,
        label_semantics=TAQ_LABEL_SEMANTICS, preprocessing={"pos_smooth_window": 3},
        input_norm={"mean": model.input_mean.flatten(), "std": model.input_std.flatten()},
    ), path)
    return path


def observation(timestamp, position=(90., 2., 1.), *, source="flow", range_valid=True):
    return SimpleNamespace(
        timestamp_s=timestamp, position_base_m=position,
        robot_linear_velocity_base_mps=(.1, 0., 0.),
        robot_angular_velocity_base_rps=(.02, .03, 99.), yaw_rate_rps=.04,
        source=source, quality=(0.,) * 7, range_valid=range_valid,
    )


def test_loader_defaults_to_ordinary_gru_and_saved_preprocessing(checkpoint_path):
    estimator = load_velocity_estimator(checkpoint_path)
    assert isinstance(estimator, OnlineLegacyGRUEstimator)
    assert estimator.model.input_dim == 13
    assert estimator.smooth_window == 3
    assert not estimator.model.training
    assert estimator.update(0, observation(0.)) is None
    assert estimator.update(0, observation(.02)) is not None


@pytest.mark.parametrize("field", ["data_schema", "label_semantics"])
def test_legacy_loader_rejects_incompatible_checkpoint(checkpoint_path, field):
    payload = torch.load(checkpoint_path, weights_only=False)
    payload[field] = "incompatible"
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError):
        load_velocity_estimator(checkpoint_path)


def test_per_target_adapter_matches_training_contract_and_exact_yaw_rate(checkpoint_path):
    estimator = load_velocity_estimator(checkpoint_path)
    for index in range(5):
        assert estimator.update(0, observation(index * .02, source="lost", range_valid=False)) is None
    assert not estimator._states[0].rows
    assert estimator.update(0, observation(.1, source="initial")) is None
    assert estimator.update(1, observation(.1, position=(130., -4., 5.), source="initial")) is None
    result = estimator.update(0, observation(.12, position=(90.04, 1.98, 1.01)))
    history = [(.1, np.array([90., 2., 1., .1, 0., 0., .02, .03, .04], dtype=np.float32)),
               (.12, np.array([90.04, 1.98, 1.01, .1, 0., 0., .02, .03, .04], dtype=np.float32))]
    with torch.inference_mode():
        expected = estimator.model(window_inputs(history, 3)[None])[0]
    torch.testing.assert_close(result.velocity_mean_mps, expected)
    assert result.reliability_gate == 1.
    assert estimator._states[1].frame_count == 1
    assert estimator._states[0].frame_count == 2


def test_later_missing_frames_keep_state_and_episode_reset_is_selective(checkpoint_path):
    estimator = load_velocity_estimator(checkpoint_path)
    estimator.update(0, observation(0.))
    estimator.update(0, observation(.02))
    estimator.update(1, observation(0.))
    state = estimator._states[0]
    result = estimator.update(0, observation(.04, source="lost", range_valid=False))
    assert estimator._states[0] is state
    assert state.frame_count == 3
    assert np.array_equal(state.rows[-1][1][:3], np.zeros(3))
    assert result.reliability_gate == 0.
    assert torch.isfinite(result.velocity_mean_mps).all()
    estimator.reset(0)
    assert 0 not in estimator._states
    assert 1 in estimator._states
    assert estimator.update(0, observation(0.)) is None


class FakeTracker:
    def __init__(self):
        self.initialized = False
        self.lost = False
        self.config = SimpleNamespace(fb_max_error_px=1.5)
        self.bootstrap_count = 0

    def reset(self):
        self.initialized = False

    def initialize(self, image, boxes, *, timestamp_s, **kwargs):
        self.initialized = True
        self.bootstrap_count += 1
        return self.frame(timestamp_s, "initial", "initialized")

    def update(self, image, *, timestamp_s=None, **kwargs):
        return self.frame(.02 if timestamp_s is None else timestamp_s, "flow", "tracking")

    def frame(self, timestamp, source, status):
        target = None if self.lost else SimpleNamespace(
            target_id=0, yaw_rad=0., pitch_rad=0., distance_m=100., source=source, status=status,
            confidence=1., motion_confidence=1., point_retention=1., ransac_inlier_ratio=1.,
        )
        return SimpleNamespace(timestamp_s=timestamp, selected_target=target)


def test_runtime_policy_waits_for_velocity_and_loss_does_not_reset_gru(checkpoint_path):
    estimator = load_velocity_estimator(checkpoint_path)
    tracker = FakeTracker()
    chain = ShootingChain(tracker, estimator, lambda inputs: np.array([0., 0., 1.]))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    robot = RobotState(turret_yaw_rad=0., turret_pitch_rad=0.)
    first = chain.initialize(image, {0: (24, 24, 40, 40)}, robot)
    assert not first.prediction_ready
    assert not first.fire_command
    second = chain.update(image, robot, dt_s=.02, timestamp_s=.02)
    assert second.prediction_ready
    assert not second.fire_command
    state = estimator._states[0]
    tracker.lost = True
    lost = chain.update(image, robot, dt_s=.02, timestamp_s=.04)
    assert not lost.prediction_ready
    assert not lost.fire_command
    assert estimator._states[0] is state
    assert state.frame_count == 3
    tracker.lost = False
    recovered = chain.update(image, robot, dt_s=.02, timestamp_s=.06)
    assert recovered.prediction_ready
    assert estimator._states[0] is state
    chain.reset()
    assert not estimator._states


class FakeEnv:
    def __init__(self):
        self.num_envs = 2
        self.device = "cpu"
        self.step_dt = .02
        self.common_step_counter = 0
        self.visual_valid = torch.ones(2, dtype=torch.bool)
        self._aim_warmup_left_steps = torch.full((2,), 2, dtype=torch.long)
        self.range_valid = torch.ones(2, dtype=torch.bool)
        intrinsics = torch.tensor([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]]).repeat(2, 1, 1)
        self._vision_camera = SimpleNamespace(data=SimpleNamespace(intrinsic_matrices=intrinsics))
        self.turret = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(2, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
            root_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
        ))
        self._rl_pred_ready = torch.zeros(2, dtype=torch.bool)
        self._rl_bridge_training_ready = torch.zeros(2, dtype=torch.bool)
        self._rl_bridge_velocity_ready = torch.zeros(2, dtype=torch.bool)
        self.positions = torch.zeros(2, 3)
        self.velocities = torch.zeros(2, 3)

    def _ensure_vision_supervision_ready(self):
        pass

    def get_vision_supervision_batch(self):
        return dict(images=torch.zeros(2, 3, 64, 64), valid_mask=self.visual_valid,
                    direction=torch.zeros(2, 2), distance=torch.tensor([100., 120.]),
                    lidar_distance=torch.tensor([100., 120.]), lidar_valid_mask=self.range_valid)

    def set_dynamic_target_raycast_angles(self, yaw, pitch):
        pass

    def run_dynamic_target_raycast_for_current_frame(self):
        pass

    def _get_runtime_camera_pose_w(self):
        return self.turret.data.root_pos_w, self.turret.data.root_quat_w

    def _rotate_local_to_world(self, rotation, values):
        return values.clone()

    def _rotate_world_to_yaw_only_local(self, rotation, values):
        return values.clone()

    def set_rl_predicted_target_state(self, position, velocity, distance, yaw, pitch, *, env_ids):
        self.positions[env_ids] = position[env_ids]
        self.velocities[env_ids] = velocity[env_ids]
        self._rl_pred_ready[env_ids] = True


def test_isaac_bridge_warmup_writeback_missing_and_independent_resets(checkpoint_path):
    env = FakeEnv()
    config = IsaacTrainingBridgeConfig(velocity_checkpoint=checkpoint_path, device="cpu",
                                       enable_fine_tracking=False, min_velocity_history_s=.05)
    bridge = IsaacFullChainPerceptionBridge(env, config)
    bridge._make_tracker = lambda *args, **kwargs: FakeTracker()
    bridge.reset_envs()
    for step in (1, 2):
        env.common_step_counter = step
        env._aim_warmup_left_steps[:] = 3 - step
        bridge.step()
        assert not env._rl_pred_ready.any()
        assert all(not state.rows for state in bridge.velocity_estimator._states.values())
        assert all(tracker is None for tracker in bridge._trackers)
    env._aim_warmup_left_steps[:] = 0
    for step in (3, 4, 5, 6):
        env.common_step_counter = step
        bridge.step()
        if step == 3:
            assert not env._rl_pred_ready.any()
    assert env._rl_pred_ready.all()
    assert env._rl_bridge_velocity_ready.all()
    torch.testing.assert_close(env.positions, torch.tensor([[0., 0., 100.], [0., 0., 120.]]))
    assert torch.isfinite(env.velocities).all()
    first_state = bridge.velocity_estimator._states[0]
    second_state = bridge.velocity_estimator._states[1]
    env.range_valid[0] = False
    env.common_step_counter = 7
    bridge.step()
    assert not env._rl_pred_ready[0]
    assert env._rl_pred_ready[1]
    assert bridge.velocity_estimator._states[0] is first_state
    assert np.array_equal(first_state.rows[-1][1][:3], np.zeros(3))
    env.range_valid[0] = True
    env.common_step_counter = 8
    bridge.step()
    assert env._rl_pred_ready.all()
    bridge._trackers[0].lost = True
    env.common_step_counter = 9
    bridge.step()
    assert not env._rl_pred_ready[0]
    assert env._rl_pred_ready[1]
    assert bridge.velocity_estimator._states[0] is first_state
    assert bridge._trackers[0].bootstrap_count == 1
    env.common_step_counter = 10
    bridge.step()
    assert bridge._trackers[0].bootstrap_count == 1
    bridge.reset_envs([0])
    assert 0 not in bridge.velocity_estimator._states
    assert bridge.velocity_estimator._states[1] is second_state


def test_unknown_velocity_backend_is_rejected(checkpoint_path):
    with pytest.raises(ValueError, match="Unknown velocity backend"):
        load_velocity_estimator(checkpoint_path, backend="unknown")


def test_factory_uses_ordinary_gru_by_default(checkpoint_path, tmp_path, monkeypatch):
    camera_path = tmp_path / "camera.json"
    camera_path.write_text(json.dumps(dict(fx=100., fy=100., cx=32., cy=32., width=64, height=64)))
    vision_package = ModuleType("test_lk_ostrack")
    vision_package.__path__ = []
    camera_module = ModuleType("test_lk_ostrack.camera_geometry")
    camera_module.CameraIntrinsics = SimpleNamespace(from_json=lambda path: SimpleNamespace())
    monkeypatch.setitem(sys.modules, "test_lk_ostrack", vision_package)
    monkeypatch.setitem(sys.modules, "test_lk_ostrack.camera_geometry", camera_module)
    monkeypatch.setattr(factory, "make_lk_ostrack_tracker_with_checkpoint", lambda *args, **kwargs: FakeTracker())
    chain = factory.build_chain(
        intrinsics_json=camera_path, rotation_base_from_camera=np.eye(3), camera_position_base_m=[0., 0., 0.],
        velocity_checkpoint=checkpoint_path, decision_policy=lambda inputs: np.zeros(3),
    )
    assert isinstance(chain.velocity_estimator, OnlineLegacyGRUEstimator)


def test_explicit_taq_backend_does_not_silently_load_ordinary_gru(checkpoint_path):
    with pytest.raises(RuntimeError, match="schema"):
        load_velocity_estimator(checkpoint_path, backend="taq_gru")

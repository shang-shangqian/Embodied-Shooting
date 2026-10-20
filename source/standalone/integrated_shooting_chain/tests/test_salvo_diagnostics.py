from pathlib import Path
import csv
import json
import math
import signal
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from integrated_shooting_chain.salvo_diagnostics import (
    SalvoDiagnostics, firing_block_reason, install_salvo_diagnostics, install_salvo_interrupt_handler,
)
from integrated_shooting_chain.salvo_engagement import SalvoEngagementScheduler
from integrated_shooting_chain.salvo_control_snapshot import control_snapshot


@pytest.mark.parametrize("change,expected", [
    ({}, "intent_below_threshold"),
    ({"intent": .5}, "intent_below_threshold"),
    ({"intent": .9, "confirmed": False}, "intent_unconfirmed"),
    ({"ready": False}, "perception_not_ready|intent_below_threshold"),
    ({"cooldown": 5}, "cooldown|intent_below_threshold"),
    ({"warmup": True}, "warmup|intent_below_threshold"),
    ({"target_id": -1}, "no_target|intent_below_threshold"),
    ({"fired": True}, "fired"),
])
def test_exact_firing_gate_reasons(change, expected):
    inputs = dict(target_id=0, warmup=False, ready=True, cooldown=0,
                  intent=.3, threshold=.5, confirmed=False, fired=False)
    inputs.update(change)
    assert firing_block_reason(**inputs) == expected


def make_control_env():
    scheduler = SalvoEngagementScheduler()
    scheduler.choose(2., {0: 100.})
    pool = SimpleNamespace(
        active=torch.tensor([0]), positions=torch.zeros(1, 100, 3), ready=torch.tensor([True]),
        shots=torch.tensor([10]), hits=torch.tensor([10]), last_hit=torch.tensor([False]),
        engagement_schedulers=[scheduler],
    )
    estimate = SimpleNamespace(target_id=0, bbox_xyxy=(1., 2., 13., 14.), confidence=.8,
                               status="tracking", source="flow", yaw_rad=.01, pitch_rad=.02, distance_m=100.)
    env = SimpleNamespace(
        num_envs=1, step_dt=.02, _static_salvo=pool, episode_length_buf=torch.tensor([500]),
        common_step_counter=500,
        cfg=SimpleNamespace(fire_intent_threshold=.5, fire_intent_hold_steps=1, hit_radius=.35),
        _rl_pred_ready=torch.tensor([True]), _rl_bridge_training_ready=torch.tensor([True]),
        _aim_warmup_left_steps=torch.tensor([0]), _fire_cooldown=torch.tensor([0]),
        _rl_pred_distance=torch.tensor([100.]), _rl_pred_vel_base=torch.tensor([[3., 4., 0.]]),
        _rl_pred_yaw_base=torch.tensor([.2]), _rl_pred_pitch_base=torch.tensor([.1]),
        _full_chain_perception_bridge=SimpleNamespace(_latest_results=[SimpleNamespace(targets=[estimate], timestamp_s=10.)]),
        _yaw_dof_ids=[0], _pitch_dof_ids=[1], _turret_pos_targets=torch.tensor([[.15, .08]]),
        _delta_pos=torch.zeros(1, 2), _raycast_hit_valid=torch.tensor([True]),
        _raycast_hit_distance=torch.tensor([100.]),
        _ballistic_cache_valid=torch.tensor([False]), _cached_min_ballistic_dist=torch.tensor([.12]),
        _cached_ballistic_hit=torch.tensor([True]), _cached_closest_time=torch.tensor([.11]),
    )
    env.turret = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.zeros(1, 3), root_quat_w=torch.tensor([[1., 0, 0, 0]]),
        joint_pos=torch.tensor([[.15, .08]]), joint_vel=torch.zeros(1, 2),
        joint_pos_limits=torch.tensor([[[-math.pi, math.pi], [-1.05, 1.05]]]),
    ))
    env._rl_pred_pos_base = torch.tensor([[100.*math.cos(.2)*math.cos(.1),
                                           100.*math.sin(.2)*math.cos(.1), -100.*math.sin(.1)]])
    pool.positions[0, 0] = env._rl_pred_pos_base[0]
    def rotate(q, v):
        t = 2*torch.cross(q[:, 1:], v, dim=-1)
        return v+q[:, :1]*t+torch.cross(q[:, 1:], t, dim=-1)
    def inverse(q, v):
        q = q.clone()
        q[:, 1:] *= -1
        return rotate(q, v)
    env._rotate_world_to_yaw_only_local = inverse
    env._rotate_world_to_local = inverse
    env.camera_pos = torch.zeros(1, 3)
    env._get_runtime_camera_pose_w = lambda: (env.camera_pos, torch.tensor([[.5, -.5, .5, -.5]]))
    env.forward_angles = [.15, .08]
    def turret_frame():
        yaw, pitch = env.forward_angles
        forward = torch.tensor([[math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.cos(pitch), -math.sin(pitch)]])
        return torch.tensor([[.5, 0., .25]]), torch.tensor([[1., 0, 0, 0]]), forward
    env._compute_turret_frame = turret_frame
    env._vision_camera = SimpleNamespace(data=SimpleNamespace(
        intrinsic_matrices=torch.tensor([[[1000., 0, 640.], [0, 1000., 360.], [0, 0, 1.]]])))
    env.policy_obs = torch.arange(28, dtype=torch.float32).view(1, -1)/10
    env._get_observations = lambda: {"policy":env.policy_obs, "critic":torch.zeros(1, 31)}
    env._reset_idx = lambda ids: None
    def pre(actions):
        env._actions = actions.clamp(-1., 1.)
        env._fire_intent = torch.tensor([.3])
        env._fire_intent_high_steps = torch.tensor([0])
        env._fire_command = torch.tensor([0])
        env._fire_cooldown[:] = 5  # Verify diagnostics saved the BEFORE value.
        env._delta_pos = env._actions[:, :2]*torch.tensor([[.03, .02]])
        env._turret_pos_targets += env._delta_pos
        limits = env.turret.data.joint_pos_limits
        env._turret_pos_targets = torch.maximum(torch.minimum(env._turret_pos_targets, limits[..., 1]), limits[..., 0])
    env._pre_physics_step = pre
    env._get_dones = lambda: (torch.tensor([False]), torch.tensor([False]))
    def rewards():
        env._ballistic_cache_valid[:] = True
        return torch.tensor([7.])
    env._get_rewards = rewards
    return env


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_velocity_wait_logs_skipped_inference_and_hold_no_fire_action(tmp_path):
    from integrated_shooting_chain.salvo_ablation import install_salvo_ablation, install_actor_switch_reset
    env=make_control_env();env.device='cpu'
    env._rl_bridge_velocity_ready=torch.tensor([False])
    experiment=install_salvo_ablation(env)
    diagnostics=SalvoDiagnostics(tmp_path/'waiting.csv')
    install_salvo_diagnostics(env,diagnostics)
    player=SimpleNamespace(states=[torch.ones(1,1,4),torch.ones(1,1,4)],has_batch_dimension=True,
                           get_action=lambda obs:pytest.fail('policy must not run before velocity is ready'))
    install_actor_switch_reset(player,experiment)
    env._get_observations()
    env._pre_physics_step(player.get_action(None))
    env._get_rewards();diagnostics.close()
    row=read_rows(diagnostics.path)[0]
    assert diagnostics.schema_version==4
    assert row['velocity_ready']=='False' and row['actor_inference_skipped']=='True'
    assert row['actor_wait_step_count']=='1' and row['hold_applied']=='True'
    assert float(row['action_fire'])==-1 and not float(row['yaw_command_change_rad'])
    assert not float(row['pitch_command_change_rad'])
    assert 'perception_not_ready' in row['blocked_reason']


def test_hook_logs_pre_action_gates_post_physics_counts_and_target_zero(tmp_path):
    env = make_control_env()
    diagnostics = SalvoDiagnostics(tmp_path / "steps.csv")
    install_salvo_diagnostics(env, diagnostics)
    observations = env._get_observations()
    assert observations["policy"] is env.policy_obs
    env._pre_physics_step(torch.tensor([[0., 0., -.4]]))
    env._get_dones()
    assert not env._ballistic_cache_valid.item()  # Logging must not force integration in dones.
    env.episode_length_buf += 1
    env.common_step_counter += 1
    assert env._get_rewards().item() == 7.
    diagnostics.close()
    rows = read_rows(tmp_path / "steps.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["target_id"] == "0"
    assert row["attempt"] == "1"
    assert row["cooldown_before"] == "0"
    assert row["blocked_reason"] == "intent_below_threshold"
    assert row["remaining"] == "90"
    assert float(row["pred_speed_mps"]) == 5.
    assert float(row["yaw_error_rad"]) == pytest.approx(.05)
    assert float(row["bbox_width_px"]) == 12.
    assert row["decision_frame_step"] == "500"
    assert row["post_frame_step"] == "501"
    assert float(row["decision_sim_s"]) == 10.
    assert float(row["post_sim_s"]) == 10.02
    assert row["actor_obs_available"] == "True"
    assert json.loads(row["actor_observation"]) == env.policy_obs[0].tolist()
    assert float(row["actor_obs_yaw_error"]) == pytest.approx(.7)
    assert row["ballistic_cache_valid"] == "True"
    assert float(row["ballistic_min_dist_m"]) == pytest.approx(.12)
    assert row["ballistic_hit"] == "True"


def test_snapshot_keeps_camera_tracking_separate_from_muzzle_and_matches_origins():
    env = make_control_env()
    first = control_snapshot(env, [0])[0]
    assert first["pred_gt_position_error_m"] == pytest.approx(0.)
    assert first["pred_gt_yaw_error_rad"] == pytest.approx(0., abs=1e-6)
    assert first["gt_center_x_px"] == pytest.approx(640.-1000.*math.tan(.2), abs=1e-4)
    assert first["gt_center_y_px"] == pytest.approx(360.+1000.*math.tan(.1)/math.cos(.2), abs=1e-4)
    # Moving the gun does not move a torso camera or its tracked target projection.
    env.forward_angles = [3., 1.]
    second = control_snapshot(env, [0])[0]
    assert first["gt_center_x_px"] == second["gt_center_x_px"]
    assert first["gt_center_y_px"] == second["gt_center_y_px"]
    assert abs(second["gt_muzzle_yaw_error_rad"]) > 2.7
    assert second["gt_muzzle_pitch_error_rad"] == pytest.approx(second["gt_muzzle_pitch_base_rad"]-1.)
    assert second["gt_muzzle_yaw_base_rad"] == pytest.approx(second["pred_muzzle_yaw_base_rad"])
    assert second["gt_muzzle_pitch_base_rad"] == pytest.approx(second["pred_muzzle_pitch_base_rad"])


def test_invalid_predictions_and_unselected_targets_do_not_produce_gt_comparisons():
    env = make_control_env()
    env._rl_pred_ready[:] = False
    snapshot = control_snapshot(env, [0])[0]
    assert not snapshot["pred_valid"]
    assert snapshot["pred_gt_yaw_error_rad"] == ""
    assert snapshot["pred_muzzle_yaw_error_rad"] == ""
    assert snapshot["gt_muzzle_yaw_error_rad"] != ""  # GT diagnostic remains independent.
    env._static_salvo.positions[0, 0] = torch.tensor([-100., 0, 0])
    assert not control_snapshot(env, [0])[0]["gt_projection_valid"]
    snapshot = control_snapshot(env, [-1])[0]
    assert not snapshot["gt_valid"]
    assert snapshot["gt_pos_w_x"] == ""
    assert snapshot["gt_muzzle_yaw_error_rad"] == ""
    assert snapshot["gt_center_x_px"] == ""


def test_yaw_comparison_wraps_across_pi():
    env = make_control_env()
    env._static_salvo.positions[0, 0] = torch.tensor([-100., 1., 0])
    env._rl_pred_yaw_base[:] = -math.pi+.01
    snapshot = control_snapshot(env, [0])[0]
    assert snapshot["pred_gt_yaw_error_rad"] == pytest.approx(.02, abs=1e-6)


def test_snapshot_uses_translated_rotated_base_and_never_refreshes_camera():
    env = make_control_env()
    root = torch.tensor([[10., -20., 3.]])
    env.turret.data.root_pos_w = root
    env.turret.data.root_quat_w = torch.tensor([[math.sqrt(.5), 0, 0, math.sqrt(.5)]])
    pred = env._rl_pred_pos_base
    env._static_salvo.positions[0, 0] = root[0]+torch.stack((-pred[0, 1], pred[0, 0], pred[0, 2]))
    before = env._static_salvo.positions.clone()
    pred_before = pred.clone()
    command_before = env._turret_pos_targets.clone()
    camera_data = env._vision_camera.data
    class CachedCamera:
        _data = camera_data
        @property
        def data(self):
            pytest.fail("diagnostics must not refresh sensor buffers")
    env._vision_camera = CachedCamera()
    snapshot = control_snapshot(env, [0])[0]
    assert snapshot["gt_distance_m"] == pytest.approx(100., abs=1e-4)
    assert snapshot["pred_gt_position_error_m"] == pytest.approx(0., abs=2e-5)
    assert snapshot["pred_gt_yaw_error_rad"] == pytest.approx(0., abs=1e-6)
    assert torch.equal(before, env._static_salvo.positions)
    assert torch.equal(pred_before, env._rl_pred_pos_base)
    assert torch.equal(command_before, env._turret_pos_targets)


def test_snapshot_selects_gt_from_each_environment_independently():
    env = make_control_env()
    env.num_envs = 2
    for owner in (env, env._static_salvo, env.turret.data):
        for name, value in vars(owner).copy().items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1:
                setattr(owner, name, value.repeat(2, *([1]*(value.ndim-1))))
    env._static_salvo.positions[0, 1] = torch.tensor([40., 0, 0])
    env._static_salvo.positions[1, 0] = torch.tensor([70., 0, 0])
    env._get_runtime_camera_pose_w = lambda: (torch.zeros(2, 3), torch.tensor([[.5, -.5, .5, -.5]]).repeat(2, 1))
    env._compute_turret_frame = lambda: (torch.zeros(2, 3), torch.tensor([[1., 0, 0, 0]]).repeat(2, 1),
                                        torch.tensor([[1., 0, 0]]).repeat(2, 1))
    env._vision_camera.data.intrinsic_matrices = env._vision_camera.data.intrinsic_matrices.repeat(2, 1, 1)
    snapshots = control_snapshot(env, [1, 0])
    assert snapshots[0]["gt_distance_m"] == 40.
    assert snapshots[1]["gt_distance_m"] == 70.


def test_snapshot_matches_rotated_and_translated_base_with_optical_camera():
    env = make_control_env()
    root = torch.tensor([[13., -8., 4.]])
    s = math.sqrt(.5)
    env.turret.data.root_pos_w = root
    env.turret.data.root_quat_w = torch.tensor([[s, 0, 0, s]])  # base faces world +Y
    env._static_salvo.positions[0, 0] = root[0]+torch.tensor([0., 100., 0.])
    env._rl_pred_pos_base = torch.tensor([[100., 0, 0]])
    env._rl_pred_yaw_base[:] = 0
    env._rl_pred_pitch_base[:] = 0
    env._get_runtime_camera_pose_w = lambda: (root, torch.tensor([[s, -s, 0, 0]]))
    env._compute_turret_frame = lambda: (root+torch.tensor([[0, .5, .25]]),
        env.turret.data.root_quat_w, torch.tensor([[0., 1., 0.]]))
    row = control_snapshot(env, [0])[0]
    assert row["pred_gt_position_error_m"] == pytest.approx(0., abs=2e-5)
    assert row["gt_yaw_base_rad"] == pytest.approx(0., abs=1e-6)
    assert row["muzzle_yaw_base_rad"] == pytest.approx(0., abs=1e-6)
    assert row["gt_center_x_px"] == pytest.approx(640., abs=1e-4)
    assert row["gt_center_y_px"] == pytest.approx(360., abs=1e-4)
    assert row["gt_muzzle_pitch_base_rad"] == pytest.approx(math.atan2(.25, 99.5), abs=1e-6)


def test_nonfinite_prediction_and_range_are_blank_in_csv(tmp_path):
    env = make_control_env()
    diagnostics = SalvoDiagnostics(tmp_path/"steps.csv")
    install_salvo_diagnostics(env, diagnostics)
    env._rl_pred_yaw_base[:] = float("nan")
    env._raycast_hit_distance[:] = float("inf")
    env._pre_physics_step(torch.tensor([[0., 0., -.4]]))
    env._get_rewards()
    diagnostics.close()
    row = read_rows(tmp_path/"steps.csv")[0]
    assert row["decision_pred_valid"] == "False"
    assert row["decision_raycast_valid"] == "False"
    assert row["decision_raycast_distance_m"] == ""
    assert row["decision_pred_gt_yaw_error_rad"] == ""
    assert row["yaw_error_rad"] == ""


def test_action_clamping_execution_lag_and_observation_snapshot_are_logged(tmp_path):
    env = make_control_env()
    env._turret_pos_targets[0, 0] = math.pi-.01
    diagnostics = SalvoDiagnostics(tmp_path/"steps.csv")
    install_salvo_diagnostics(env, diagnostics)
    env._get_observations()
    env.policy_obs[:] = 0  # The saved actor input must not alias a mutable tensor.
    env._pre_physics_step(torch.tensor([[2., -.5, -.4]]))
    # Physical motion is distinct from the requested/clamped target change.
    env.turret.data.joint_pos[0, 0] += .002
    env._get_rewards()
    diagnostics.close()
    row = read_rows(tmp_path/"steps.csv")[0]
    assert float(row["action_yaw_raw"]) == 2.
    assert float(row["action_yaw"]) == 1.
    assert float(row["yaw_delta_requested_rad"]) == pytest.approx(.03)
    assert float(row["yaw_command_change_rad"]) == pytest.approx(.01, abs=1e-6)
    assert row["yaw_command_clamped"] == "True"
    assert row["pitch_command_clamped"] == "False"
    assert float(row["post_yaw_joint_pos_rad"])-float(row["decision_yaw_joint_pos_rad"]) == pytest.approx(.002, abs=1e-7)
    assert json.loads(row["actor_observation"])[7] == pytest.approx(.7)


def test_hit_target_geometry_is_preserved_and_reset_clears_actor_snapshot(tmp_path):
    env = make_control_env()
    diagnostics = SalvoDiagnostics(tmp_path/"steps.csv")
    install_salvo_diagnostics(env, diagnostics)
    env._get_observations()
    env._pre_physics_step(torch.tensor([[0., 0., -.4]]))
    original_target = env._static_salvo.positions[0, 0].clone()
    env._static_salvo.last_hit[:] = True
    env._get_rewards()
    env._get_rewards()  # A repeated read must not emit another row.
    env._reset_idx(torch.tensor([0]))
    env._static_salvo.active[:] = 1
    env._static_salvo.positions[0, 1] = torch.tensor([120., 0, 0])
    env._pre_physics_step(torch.tensor([[0., 0., -.4]]))
    env._get_rewards()
    diagnostics.close()
    rows = read_rows(tmp_path/"steps.csv")
    assert len(rows) == 2
    assert rows[0]["target_id"] == "0"
    assert float(rows[0]["post_gt_pos_w_x"]) == pytest.approx(float(original_target[0]))
    assert rows[1]["target_id"] == "1"
    assert rows[1]["actor_obs_available"] == "False"
    assert rows[1]["actor_observation"] == ""
    assert rows[1]["episode_index"] == "1"


def test_switch_without_reset_keeps_each_actor_input_and_target_geometry_separate(tmp_path):
    env = make_control_env()
    diagnostics = SalvoDiagnostics(tmp_path/"steps.csv")
    install_salvo_diagnostics(env, diagnostics)
    first_obs = env._get_observations()["policy"].clone()
    env._pre_physics_step(torch.tensor([[0., 0., -.4]]))
    env._get_rewards()
    env._static_salvo.active[:] = 1
    env._static_salvo.positions[0, 1] = torch.tensor([120., 0, 0])
    env._rl_pred_ready[:] = False
    env._rl_bridge_training_ready[:] = False
    env.policy_obs = torch.zeros(1, 28)
    env.common_step_counter += 1
    env.episode_length_buf += 1
    env._get_observations()
    env._pre_physics_step(torch.tensor([[0., 0., -.4]]))
    env._get_rewards()
    diagnostics.close()
    rows = read_rows(tmp_path/"steps.csv")
    assert rows[0]["actor_obs_target_id"] == rows[0]["target_id"] == "0"
    assert json.loads(rows[0]["actor_observation"]) == first_obs[0].tolist()
    assert rows[1]["actor_obs_target_id"] == rows[1]["target_id"] == "1"
    assert rows[1]["actor_obs_frame_step"] == rows[1]["decision_frame_step"] == "501"
    assert rows[1]["decision_pred_valid"] == "False"
    assert rows[1]["decision_pred_gt_yaw_error_rad"] == ""
    assert rows[1]["decision_gt_distance_m"] == "120.0"


def test_ctrl_c_closes_real_mp4_before_previous_native_handler(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from integrated_shooting_chain.episode_video import EpisodeVideoRecorder
    recorder = EpisodeVideoRecorder(tmp_path / "video", step_dt=.02)
    recorder._compose_frame = lambda *args: np.zeros((64, 64, 3), dtype=np.uint8)
    status = dict(step=1, fires=0, hit=False, warmup=False, ready=True)
    recorder.capture(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8), status)
    original = signal.getsignal(signal.SIGINT)
    native_calls = []
    signal.signal(signal.SIGINT, lambda *args: native_calls.append(True))
    previous = install_salvo_interrupt_handler()
    interrupted = False
    try:
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        recorder.close()
        signal.signal(signal.SIGINT, previous)
        signal.signal(signal.SIGINT, original)
    assert interrupted
    assert not native_calls
    capture = cv2.VideoCapture(str(tmp_path / "video/env0_episode_001.mp4"))
    try:
        assert capture.isOpened()
        ok, frame = capture.read()
        assert ok and frame.shape == (64, 64, 3)
    finally:
        capture.release()

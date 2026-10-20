"""Exercise clearance accounting and Isaac lifecycle hooks without starting Kit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PATH = Path(__file__).resolve().parents[2] / "test7_2_weight/static_salvo_eval.py"
spec = importlib.util.spec_from_file_location("static_salvo_eval", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_hit_accounting_and_per_env_isolation():
    state = module.StaticSalvoState(2, 3, "cpu", 0.02)
    state.positions[:] = torch.tensor([[5., 0, 1], [2., 0, 1], [9., 0, 1]])
    state.alive[:] = True
    state.ready[:] = True
    state.active[:] = 1
    assert state.active.tolist() == [1, 1]
    hit, cleared = state.advance(torch.tensor([1, 1]), torch.tensor([1, 0]))
    assert hit.tolist() == [True, False]
    assert not cleared.any()
    assert state.active.tolist() == [1, 1]
    state.reset(torch.tensor([0]))
    assert state.shots.tolist() == [0, 1]
    assert state.alive[1].all()


def test_shots_misses_clearance_time_and_frequency():
    state = module.StaticSalvoState(1, 2, "cpu", 0.1)
    state.ready[:] = True
    state.alive[:] = True
    state.active[:] = 0
    for fire, hit in [(1, 0), (0, 0), (1, 1)]:
        _, cleared = state.advance(torch.tensor([fire]), torch.tensor([hit]))
        assert not cleared.item()
    state.active[:] = 1
    _, cleared = state.advance(torch.tensor([1]), torch.tensor([1]))
    assert cleared.item()
    record = state.finish(0, "cleared", 2.0)
    assert record["shots"] == 3
    assert record["destroyed"] == 2
    assert record["accuracy"] == pytest.approx(2 / 3)
    assert record["clearance_sim_s"] == pytest.approx(0.4)
    assert record["total_sim_s"] == pytest.approx(2.4)
    assert record["mean_firing_hz"] == pytest.approx(2 / 0.3)


def test_timeout_is_not_reported_as_a_clearance():
    state = module.StaticSalvoState(1, 100, "cpu", 0.02)
    state.ready[:] = True
    state.alive[:] = True
    state.advance(torch.tensor([0]), torch.tensor([0]))
    record = state.finish(0, "timeout", 2.0)
    assert not record["cleared"]
    assert record["remaining"] == 100
    assert record["clearance_sim_s"] is None
    assert record["accuracy"] is None
    assert record["mean_firing_hz"] is None


class FakeEnv:
    def __init__(self):
        self.num_envs, self.device, self.step_dt = 1, "cpu", 0.02
        self.cfg = SimpleNamespace(enable_vision_supervision=False)
        self.scene = SimpleNamespace(env_origins=torch.zeros(1, 3))
        self._rl_pred_ready = torch.zeros(1, dtype=torch.bool)
        self._rl_bridge_training_ready = torch.zeros(1, dtype=torch.bool)
        self._rl_bridge_velocity_ready = torch.zeros(1, dtype=torch.bool)
        self.turret = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(1, 3)))
        self._target_pos_w = torch.zeros(1, 3)
        self._target_vel_w = torch.zeros(1, 3)
        self._target_acc_w = torch.zeros(1, 3)
        self._episode_start_target_pos_w = torch.zeros(1, 3)
        self._target_sphere_visuals_enabled = False
        self._pending_target_resample = torch.zeros(1, dtype=torch.bool)
        self._aim_warmup_left_steps = torch.zeros(1, dtype=torch.long)
        self._hit_flag = torch.zeros(1, dtype=torch.bool)
        self._fire_command = torch.ones(1)
        self._fire_intent_high_steps = torch.ones(1, dtype=torch.long)
        self._fire_cooldown = torch.tensor([5])
        self._target_motion_mask = torch.zeros(1, dtype=torch.bool)
        self._target_out_of_view_termination_mask = torch.ones(1, dtype=torch.bool)
        self.bridge_resets = []
        self._full_chain_perception_bridge = SimpleNamespace(reset_envs=lambda ids: self.bridge_resets.append(ids.tolist()))
        self.timeout = False
        self.reset_count = 0

    def _compute_current_step_hit(self):
        return self._fire_command.bool()

    def _get_dones(self):
        self._hit_flag |= self._compute_current_step_hit()
        return self._hit_flag.clone(), torch.tensor([self.timeout])

    def _get_rewards(self):
        self._hit_flag |= self._compute_current_step_hit()
        return self._hit_flag.float()

    def _get_observations(self):
        return self._target_pos_w.clone()

    def _reset_idx(self, ids):
        self.reset_count += 1
        self._aim_warmup_left_steps[:] = 100

    def _invalidate_ballistic_cache(self):
        pass

    def _update_rendered_target_spheres(self, ids):
        pass


def test_intermediate_hit_switches_after_rewards_without_resetting_robot_or_cooldown():
    env = FakeEnv()
    state = module.install_static_salvo(env, target_count=2)
    state.ready[:] = True
    state.alive[:] = True
    state.positions[0] = torch.tensor([[2., 0, 1], [5., 0, 1]])
    state.active[:] = 0
    env._target_pos_w[:] = state.positions[:, 0]
    original_positions = state.positions.clone()
    terminated, truncated = env._get_dones()
    assert not terminated.item() and not truncated.item()
    assert env._get_rewards().item() == 1
    assert not env._hit_flag.item()
    assert torch.equal(env._target_pos_w[0], original_positions[0, 0])
    observations = env._get_observations()
    assert state.active.item() == -1
    assert observations[0, 2] == -1000
    # The bridge, rather than a GT distance sort, supplies the next tracked ID.
    env.set_static_salvo_selected_target(0, 1)
    observations = env._get_observations()
    assert torch.equal(observations[0], original_positions[0, 1])
    assert torch.equal(state.positions, original_positions)
    assert env._fire_cooldown.item() == 5
    assert env._aim_warmup_left_steps.item() == 0
    assert env.reset_count == 0
    assert env.bridge_resets == []
    env._fire_command[:] = 1
    terminated, _ = env._get_dones()
    assert terminated.item()
    assert len(state.records) == 1
    assert state.records[0]["shots"] == 2
    assert state.records[0]["cleared"]


def test_actual_scoring_drives_five_miss_switch_without_ending_pool_episode():
    from integrated_shooting_chain.salvo_engagement import SalvoEngagementScheduler
    env=FakeEnv()
    env._compute_current_step_hit=lambda:torch.tensor([False])
    state=module.install_static_salvo(env,target_count=2,stall_timeout_s=None,switch_policy='misses')
    state.ready[:]=True;state.alive[:]=True;state.active[:]=0
    from integrated_shooting_chain.salvo_engagement import SalvoEngagementConfig
    scheduler=SalvoEngagementScheduler(SalvoEngagementConfig(switch_policy='misses'));state.engagement_schedulers=[scheduler]
    candidates={0:80.,1:100.};scheduler.choose(0.,candidates)
    for shot in range(1,6):
        # Non-firing steps must not count.
        env._fire_command[:]=0
        env._get_dones();env._get_rewards()
        assert scheduler.attempt_shots==shot-1
        env._fire_command[:]=1
        terminated,truncated=env._get_dones()
        env._get_rewards()
        assert not terminated.item() and not truncated.item()
        selected=scheduler.choose(float(shot)*10,candidates)
        assert selected==(0 if shot<5 else 1)
    assert state.shots.item()==5 and state.hits.item()==0
    assert state.alive.all() and not state.records
    env.set_static_salvo_selected_target(0,1)
    assert state.active.item()==1 and env._fire_command.item()==0
    assert env.reset_count==0
    assert scheduler.events[-2]['reason']=='consecutive_misses'


def test_scored_fifth_shot_hit_removes_target_without_miss_deferral():
    from integrated_shooting_chain.salvo_engagement import SalvoEngagementScheduler
    state=module.StaticSalvoState(2,2,'cpu',.02)
    state.ready[:]=True;state.alive[:]=True;state.active[:]=0
    schedulers=[SalvoEngagementScheduler(),SalvoEngagementScheduler()]
    state.engagement_schedulers=schedulers
    for scheduler in schedulers:scheduler.choose(0.,{0:80.,1:100.})
    for _ in range(4):state.advance(torch.tensor([1,0]),torch.tensor([0,1]))
    assert schedulers[0].consecutive_misses==4 and schedulers[1].attempt_shots==0
    state.advance(torch.tensor([1,1]),torch.tensor([1,0]))
    assert state.shots.tolist()==[5,1] and state.hits.tolist()==[1,0]
    assert schedulers[0].consecutive_misses==0 and schedulers[1].consecutive_misses==1
    assert schedulers[0].choose(1.,{0:80.,1:100.},removed_ids=[0])==1
    assert not any(e['event']=='deferred' for e in schedulers[0].events)
    assert state.alive.tolist()==[[False,True],[True,True]]


def test_default_pool_configuration_switches_without_firing_at_three_seconds():
    from integrated_shooting_chain.salvo_engagement import SalvoEngagementConfig, SalvoEngagementScheduler
    env=FakeEnv();env._fire_command[:]=0
    state=module.install_static_salvo(env,target_count=2)
    state.ready[:]=True;state.alive[:]=True;state.active[:]=0
    scheduler=SalvoEngagementScheduler(SalvoEngagementConfig(**state.engagement_config))
    state.engagement_schedulers=[scheduler]
    candidates={0:80.,1:100.}
    scheduler.choose(0.,candidates)
    env._get_dones();env._get_rewards()
    assert scheduler.choose(2.99,candidates)==0
    selected=scheduler.choose(3.,candidates)
    assert selected==1 and state.shots.item()==0
    env.set_static_salvo_selected_target(0,selected)
    assert state.active.item()==1 and state.alive.all()
    assert env.reset_count==0
    assert scheduler.events[-2]['reason']=='no_hit'


def test_pool_initialization_samples_100_static_targets_inside_sector_above_ground():
    torch.manual_seed(61)
    env = FakeEnv()
    env.cfg.target_radius_range = (80., 150.)
    env.cfg.target_cam_yaw_range = (-0.26, 0.26)
    env.cfg.target_cam_pitch_range = (-0.12, 0.12)
    env.cfg.target_sphere_radius = 0.35
    env.scene = SimpleNamespace(env_origins=torch.zeros(1, 3))
    env.episode_length_buf = torch.tensor([100])
    env._get_runtime_camera_pose_w = lambda: (torch.tensor([[0., 0, 1.]]), torch.tensor([[1., 0, 0, 0]]))
    env._rotate_local_to_world = lambda quat, xyz: torch.stack((xyz[:, 2], xyz[:, 0], -xyz[:, 1]), -1)
    state = module.install_static_salvo(env)
    env._get_observations()
    assert state.ready.item()
    assert state.alive.sum().item() == 100
    assert (state.positions[0, :, 2] >= 0.35).all()
    offset = state.positions[0] - torch.tensor([0., 0, 1.])
    radius = offset.norm(dim=-1)
    yaw = torch.atan2(offset[:, 1], offset[:, 0])
    pitch = torch.atan2(offset[:, 2], offset[:, :2].norm(dim=-1))
    assert ((radius >= 80.) & (radius <= 150.)).all()
    assert (yaw.abs() <= 0.26).all()
    assert (pitch.abs() <= 0.12).all()
    original = state.positions.clone()
    env._get_observations()
    assert torch.equal(state.positions, original)
    assert env._target_vel_w.count_nonzero() == 0
    assert env._target_acc_w.count_nonzero() == 0


def test_saved_layout_replays_exact_positions_without_sampling_again(tmp_path):
    from integrated_shooting_chain.salvo_ablation import save_layout
    path=tmp_path/"layout.json"
    points=torch.tensor([[80.,1.,1.],[120.,-3.,2.]])
    digest=save_layout(path,points,.35)
    env=FakeEnv();env.cfg.target_sphere_radius=.35
    env.scene.env_origins[:]=torch.tensor([20.,30.,0.])
    env.episode_length_buf=torch.tensor([100])
    # No camera sampling APIs are present: replay must not consume random positions.
    state=module.install_static_salvo(env,target_count=2,layout_in=path)
    env._get_observations()
    assert torch.equal(state.positions[0],points+env.scene.env_origins[0])
    assert state.layout_sha256 == digest
    assert state.ready.item() and state.alive.all()


@pytest.mark.parametrize("reason", ["tracking_lost", "attempts_exhausted", "bootstrap_failed"])
def test_unengageable_pool_ends_with_remaining_targets_instead_of_waiting_for_timeout(reason):
    env = FakeEnv()
    env._fire_command[:] = 0
    state = module.install_static_salvo(env, target_count=2)
    state.ready[:] = True
    state.alive[:] = True
    state.unengageable_reasons = [reason]
    terminated, truncated = env._get_dones()
    assert terminated.item()
    assert not truncated.item()
    record = state.records[0]
    assert record["reason"] == reason
    assert not record["cleared"]
    assert record["remaining"] == 2
    assert record["remaining_target_ids"] == [0, 1]


def test_robot_failure_keeps_priority_over_engagement_exhaustion():
    env = FakeEnv()
    env._fire_command[:] = 0
    env._get_dones = lambda: (torch.tensor([True]), torch.tensor([False]))
    state = module.install_static_salvo(env, target_count=2)
    state.ready[:] = True
    state.alive[:] = True
    state.unengageable_reasons = ["attempts_exhausted"]
    env._get_dones()
    assert state.records[0]["reason"] == "failure"


def test_interrupted_partial_result_is_recorded_once():
    state = module.StaticSalvoState(1, 2, "cpu", .02)
    state.ready[:] = True
    state.alive[:] = True
    state.active[:] = 0
    state.advance(torch.tensor([1]), torch.tensor([1]))
    first = state.finish(0, "interrupted", 2.)
    assert state.finish(0, "interrupted", 2.) is first
    assert len(state.records) == 1
    assert first["remaining_target_ids"] == [1]
    assert first["reason"] == "interrupted"
    assert not first["cleared"]

"""Homing state machine, real scheduler timing and policy/actuator integration on CPU."""
import math
import argparse
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from integrated_shooting_chain.salvo_ablation import install_actor_switch_reset, install_salvo_ablation
from integrated_shooting_chain.salvo_engagement import SalvoEngagementScheduler
from integrated_shooting_chain.salvo_return_home import SalvoReturnHomeConfig, install_salvo_return_home


def fixture(q=(.6, -.3), *, count=1, config=None):
    env = SimpleNamespace(num_envs=count, device='cpu', step_dt=.02,
        cfg=SimpleNamespace(max_delta_yaw=.03, max_delta_pitch=.02),
        _yaw_dof_ids=[0], _pitch_dof_ids=[1],
        episode_length_buf=torch.full((count,), 100, dtype=torch.long),
        _aim_warmup_left_steps=torch.zeros(count, dtype=torch.long),
        _rl_pred_ready=torch.ones(count, dtype=torch.bool),
        _rl_bridge_training_ready=torch.ones(count, dtype=torch.bool),
        _rl_bridge_velocity_ready=torch.ones(count, dtype=torch.bool),
        _turret_pos_targets=torch.zeros(count, 2), _delta_pos=torch.zeros(count, 2),
        _fire_command=torch.zeros(count),
        _static_salvo=SimpleNamespace(active=torch.arange(count), engagement_schedulers=[]))
    env.turret = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.tensor([q], dtype=torch.float32).repeat(count, 1), joint_vel=torch.zeros(count, 2),
        joint_pos_limits=torch.tensor([[[-math.pi, math.pi], [-1.05, 1.05]]]).repeat(count, 1, 1)))
    for i in range(count):
        scheduler = SalvoEngagementScheduler()
        scheduler.choose(2., {i: 100.})
        env._static_salvo.engagement_schedulers.append(scheduler)
    def pre(actions):
        env._actions = actions.clone()
        env._delta_pos = actions[:, :2] * torch.tensor([.03, .02])
        env._turret_pos_targets += env._delta_pos
        env._fire_command = (actions[:, 2] > 0).float()
    env._pre_physics_step = pre
    env._reset_idx = lambda ids: None
    home = install_salvo_return_home(env, config or SalvoReturnHomeConfig())
    experiment = install_salvo_ablation(env)
    calls = []
    class Player:
        has_batch_dimension = True
        def __init__(self):
            self.states = [torch.ones(1, count, 4), torch.full((1, count, 4), 7.)]
        def get_action(self, obs):
            calls.append([s.clone() for s in self.states])
            size = self.states[0].shape[1]
            self.states[0].add_(1)
            self.states[1].add_(1)
            return torch.zeros(size, 3)
    player = Player()
    install_actor_switch_reset(player, experiment)
    return env, home, experiment, player, calls


def advance(env, *, move=True):
    data = env.turret.data
    delta = (env._turret_pos_targets - data.joint_pos)*.5 if move else torch.zeros_like(data.joint_pos)
    data.joint_pos += delta
    data.joint_vel = delta/env.step_dt
    env.episode_length_buf += 1
    for i, scheduler in enumerate(env._static_salvo.engagement_schedulers):
        target = int(env._static_salvo.active[i])
        selected = scheduler.choose(float(env.episode_length_buf[i])*env.step_dt, {target: 100.})
        env._static_salvo.active[i] = -1 if selected is None else selected


def test_physical_home_overrides_fire_rate_limits_command_and_never_teleports():
    env, home, exp, player, calls = fixture()
    env._rl_pred_ready[:] = env._rl_bridge_training_ready[:] = env._rl_bridge_velocity_ready[:] = False
    actual = env.turret.data.joint_pos.clone()
    hidden = [s.clone() for s in player.states]
    paused = player.get_action(torch.zeros(1, 59))
    assert paused.tolist() == [[0., 0., -1.]] and not calls
    assert torch.equal(env._turret_pos_targets, actual)
    malicious_actions = torch.ones(1, 3)
    env._pre_physics_step(malicious_actions)
    assert torch.equal(malicious_actions, torch.ones_like(malicious_actions))
    torch.testing.assert_close(env._turret_pos_targets, actual+torch.tensor([[-.03, .02]]))
    assert torch.equal(env.turret.data.joint_pos, actual)  # Only physics may move it.
    assert env._fire_command.item() == 0 and env._actions[0, 2] == -1
    assert not exp.hold_applied.any()  # Unready perception must not cancel physical homing.
    assert all(torch.equal(s, saved) for s, saved in zip(player.states, hidden))
    assert home.steps == [1]
    assert env._static_salvo.engagement_schedulers[0].excluded_attempt_s == pytest.approx(.02)


def test_home_requires_actual_position_velocity_and_consecutive_stable_samples():
    env, home, exp, player, calls = fixture(q=(0., 0.), config=SalvoReturnHomeConfig(settle_frames=3))
    env.turret.data.joint_vel[:] = .2
    for _ in range(4):
        player.get_action(None)
        env._pre_physics_step(torch.ones(1, 3))
        env.episode_length_buf += 1
    assert home.active.item() and home.settled == [0] and not calls
    env.turret.data.joint_vel[:] = 0
    player.get_action(None)
    player.get_action(None)  # Repeated inference in one frame cannot fake settling.
    assert home.settled == [1] and not calls
    env.episode_length_buf += 1
    env.turret.data.joint_pos[:] = .1
    player.get_action(None)
    assert home.settled == [0]
    env.turret.data.joint_pos[:] = 0
    for _ in range(3):
        env.episode_length_buf += 1
        player.get_action(torch.zeros(1, 59))
    assert not home.active.item() and home.completed == [1] and exp.actor_resets == [1]
    assert len(calls) == 1 and torch.count_nonzero(calls[0][0]) == 0
    assert torch.equal(calls[0][1], torch.full((1, 1, 4), 7.))
    env.episode_length_buf += 1
    player.get_action(torch.zeros(1, 59))
    assert exp.actor_resets == [1]


def test_lagged_physical_actuator_settles_then_waits_for_velocity_without_repeat_reset():
    env, home, exp, player, calls = fixture(q=(1.2, -.7))
    env._rl_bridge_velocity_ready[:] = False
    for _ in range(200):
        action = player.get_action(torch.zeros(1, 59))
        if not home.active.item():
            break
        env._pre_physics_step(action)
        advance(env)
    else:
        pytest.fail('physical controller failed to settle')
    assert not calls and home.completed == [1] and exp.actor_resets == [1]
    assert (env.turret.data.joint_pos.abs() <= math.radians(.5)).all()
    assert (env.turret.data.joint_vel.abs() <= .05).all()
    zero_hidden = player.states[0].clone()
    env.episode_length_buf += 1
    player.get_action(None)
    assert torch.equal(player.states[0], zero_hidden) and not calls
    env._rl_bridge_velocity_ready[:] = True
    player.get_action(torch.zeros(1, 59))
    assert len(calls) == 1 and exp.actor_resets == [1]
    assert home.summary(0)['homing_sim_s'] == pytest.approx(home.steps[0]*.02)


def test_homing_does_not_spend_three_second_attempt_budget_and_normal_deadline_resumes():
    env, home, exp, player, calls = fixture(q=(1., .4), config=SalvoReturnHomeConfig(timeout_s=8.))
    for _ in range(200):  # Four seconds of physically stuck actuators, without the 3 s switch.
        env._pre_physics_step(player.get_action(torch.zeros(1, 59)))
        advance(env, move=False)
    scheduler = env._static_salvo.engagement_schedulers[0]
    assert scheduler.attempts == {0: 1} and scheduler.current_id == 0 and not home.failed.any()
    assert scheduler.excluded_attempt_s == pytest.approx(4.)
    assert not calls
    assert scheduler.choose(8.999, {0: 100.}) == 0
    assert scheduler.choose(9., {0: 100.}) is None
    assert scheduler.events[-1]['reason'] == 'no_hit'
    assert scheduler.events[-1]['attempt_elapsed_sim_s'] == pytest.approx(3.)


def test_stuck_home_times_out_and_never_releases_actor_or_fire():
    env, home, exp, player, calls = fixture(q=(1., .4), config=SalvoReturnHomeConfig(timeout_s=.1))
    for _ in range(7):
        action = player.get_action(torch.zeros(1, 59))
        env._pre_physics_step(action)
        advance(env, move=False)
    assert home.failed.item() and not home.active.item() and not calls
    assert env._fire_command.item() == 0 and not exp.actor_resets[0]
    assert home.events[0][-1]['event'] == 'return_home_timeout'


def test_same_id_retry_gets_new_home_and_reset_but_same_attempt_does_not():
    env, home, exp, player, calls = fixture(q=(0., 0.), config=SalvoReturnHomeConfig(settle_frames=1))
    player.get_action(torch.zeros(1, 59))
    assert home.completed == [1] and exp.actor_resets == [1]
    env.episode_length_buf += 1
    player.get_action(torch.zeros(1, 59))
    assert home.completed == [1]
    scheduler = env._static_salvo.engagement_schedulers[0]
    assert scheduler.choose(5., {0: 100.}) is None
    assert scheduler.choose(6., {0: 100.}) == 0
    env.episode_length_buf[:] = 300
    player.get_action(torch.zeros(1, 59))
    assert home.keys == [(0, 2)] and home.completed == [2] and exp.actor_resets == [2]


def test_target_loss_cancels_home_and_starts_new_attempt_without_joint_reset():
    env, home, exp, player, calls = fixture()
    env._pre_physics_step(player.get_action(None))
    pose = env.turret.data.joint_pos.clone()
    scheduler = env._static_salvo.engagement_schedulers[0]
    assert scheduler.choose(2.02, {1: 80.}) == 1
    env._static_salvo.active[:] = 1
    env.episode_length_buf += 1
    player.get_action(None)
    assert home.keys == [(1, 1)] and home.active.item()
    assert [e['event'] for e in home.events[0]] == [
        'return_home_started', 'return_home_cancelled', 'return_home_started']
    assert torch.equal(env.turret.data.joint_pos, pose) and not exp.actor_resets[0]


def test_parallel_home_does_not_block_ready_environment_or_evolve_other_hidden_states():
    env, home, exp, player, calls = fixture(count=2, config=SalvoReturnHomeConfig(settle_frames=1))
    env.turret.data.joint_pos[0] = 0
    action = player.get_action(torch.zeros(2, 59))
    assert home.active.tolist() == [False, True] and exp.actor_resets == [1, 0]
    assert len(calls) == 1 and calls[0][0].shape == (1, 1, 4)
    assert torch.equal(player.states[0][:, 1], torch.ones(1, 4))
    assert torch.equal(player.states[1][:, 1], torch.full((1, 4), 7.))
    assert action[1].tolist() == [0., 0., -1.]
    env._pre_physics_step(action)
    assert home.steps == [0, 1]


def test_reset_clears_only_selected_home_state():
    env, home, exp, player, calls = fixture(count=2)
    env._pre_physics_step(player.get_action(None))
    env._reset_idx(torch.tensor([0]))
    assert home.steps == [0, 1] and home.keys == [None, (1, 1)]
    assert home.active.tolist() == [False, True] and home.events[0] == []


def test_temporary_no_target_gap_does_not_repeat_completed_home_for_same_attempt():
    env, home, exp, player, calls = fixture(q=(0., 0.), config=SalvoReturnHomeConfig(settle_frames=1))
    player.get_action(torch.zeros(1, 59))
    env._static_salvo.active[:] = -1
    env.episode_length_buf += 1
    player.get_action(None)
    env._static_salvo.active[:] = 0
    env.episode_length_buf += 1
    player.get_action(torch.zeros(1, 59))
    assert home.completed == [1] and exp.actor_resets == [1]


@pytest.mark.parametrize('kwargs', [dict(angle_tolerance_deg=0), dict(velocity_tolerance_rps=float('nan')),
                                    dict(timeout_s=-1), dict(settle_frames=0), dict(settle_frames=True)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SalvoReturnHomeConfig(**kwargs)


def test_home_rejects_unreachable_zero_and_conflicting_ablation():
    env, home, exp, player, calls = fixture()
    env.turret.data.joint_pos_limits[0, 0] = torch.tensor([.1, 1.])
    with pytest.raises(ValueError, match='inside'):
        install_salvo_return_home(env)
    with pytest.raises(ValueError, match='group D'):
        install_salvo_ablation(env, 'A')


@pytest.mark.parametrize('options,valid', [
    (['--static_salvo', '--salvo_return_home'], True),
    (['--static_salvo', '--salvo_return_home', '--salvo_ablation', 'A'], False),
    (['--static_salvo', '--salvo_return_home', '--no-salvo_wait_for_velocity'], False),
    (['--salvo_return_home'], False),
    (['--static_salvo', '--salvo_return_home', '--salvo_home_timeout_s', 'nan'], False),
    (['--static_salvo', '--salvo_return_home', '--salvo_home_settle_frames', '0'], False),
    (['--static_salvo', '--no-salvo_return_home'], True),
])
def test_actual_eval_cli_accepts_home_mode_and_rejects_incompatible_options(monkeypatch, options, valid):
    path = Path(__file__).resolve().parents[2]/'test7_2_weight/eval_with_track_gt.py'
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'full_chain' for t in n.targets))
    monkeypatch.setattr(sys, 'argv', ['eval', '--checkpoint', 'old.pth', '--locomotion_policy', 'policy.pt']+options)
    namespace = dict(argparse=argparse, math=math)
    program = compile(ast.Module(body=main.body[:end], type_ignores=[]), str(path), 'exec')
    if valid:
        exec(program, namespace)
        assert namespace['args'].yaw_error_mode == 'wrapped_los'
        assert namespace['args'].salvo_return_home == ('--salvo_return_home' in options)
    else:
        with pytest.raises(ValueError):
            exec(program, namespace)

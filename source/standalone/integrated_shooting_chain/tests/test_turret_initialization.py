"""Reset distributions and actual environment reset/warmup code, without Kit."""
import argparse
import ast
from collections.abc import Sequence
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

TRAIN = Path(__file__).resolve().parents[2]/'test7_2_weight'
sys.path.insert(0, str(TRAIN))
from turret_initialization import TurretInitConfig, resolve_training_cases, sample_turret_initialization
from ppo_training_config import add_ppo_arguments


def limits(count=1000):
    return torch.tensor([[[-math.pi, math.pi], [-math.pi/3, math.pi/3]]]).repeat(count, 1, 1)


def test_mixture_proportions_axis_coverage_signs_and_authored_limit_margin():
    torch.manual_seed(61)
    cfg = TurretInitConfig(mode='mixed')
    lim = limits(60000)
    sample = sample_turret_initialization(lim, cfg)
    fractions = torch.bincount(sample.groups, minlength=3).float()/len(lim)
    torch.testing.assert_close(fractions, torch.tensor([.4, .4, .2]), atol=.01, rtol=0)
    assert torch.count_nonzero(sample.positions[sample.groups == 0]) == 0
    degrees = sample.positions.rad2deg()
    general = sample.groups == 1
    assert (degrees[general, 0].abs() <= 90.).all()
    assert (degrees[general, 1].abs() <= 20.).all()
    assert ((degrees[:, 0] >= -175.001) & (degrees[:, 0] <= 175.001)).all()
    assert ((degrees[:, 1] >= -55.001) & (degrees[:, 1] <= 55.001)).all()
    edge = sample.groups == 2
    for code in (1, 2, 3):
        assert abs(float((sample.limit_axes[edge] == code).float().mean())-1/3) < .02
    for axis, bit, inner, outer in ((0, 1, 150., 175.), (1, 2, 40., 55.)):
        chosen = edge & ((sample.limit_axes & bit) != 0)
        values = degrees[chosen, axis]
        assert (values.abs() >= inner-.001).all() and (values.abs() <= outer+.001).all()
        assert abs(float((values > 0).float().mean())-.5) < .02
    both = degrees[sample.limit_axes == 3]
    assert all(((both[:, 0] > 0) == yaw_positive).logical_and((both[:, 1] > 0) == pitch_positive).any()
               for yaw_positive in (False, True) for pitch_positive in (False, True))


def test_zero_mode_preserves_exact_zero_and_does_not_consume_rng():
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    sample = sample_turret_initialization(limits())
    assert torch.equal(torch.get_rng_state(), before)
    assert not sample.positions.any() and not sample.groups.any() and not sample.limit_axes.any()


def test_mixed_sampling_is_seeded_and_supports_empty_batches_and_asymmetric_limits():
    lim = torch.deg2rad(torch.tensor([[[-100., 170.], [-50., 40.]], [[-80., 90.], [-30., 55.]]]))
    cfg = TurretInitConfig(mode='mixed', probabilities=(0., 0., 1.))
    torch.manual_seed(9)
    first = sample_turret_initialization(lim, cfg)
    torch.manual_seed(9)
    second = sample_turret_initialization(lim, cfg)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert (first.positions >= lim[:, :, 0]+math.radians(5)-1e-6).all()
    assert (first.positions <= lim[:, :, 1]-math.radians(5)+1e-6).all()
    assert sample_turret_initialization(limits(0), cfg).positions.shape == (0, 2)


@pytest.mark.parametrize('kwargs', [dict(mode='bad'), dict(probabilities=(.4, .4, .3)),
    dict(probabilities=(-.1, .9, .2)), dict(probabilities=(float('nan'), .4, .2)),
    dict(probabilities=(1., 0.)), dict(general_yaw_deg=-1.), dict(general_pitch_deg=float('inf')),
    dict(limit_margin_deg=0.), dict(limit_yaw_band_deg=-1.), dict(limit_pitch_band_deg=float('nan'))])
def test_invalid_config_fails_before_simulation(kwargs):
    with pytest.raises(ValueError):
        TurretInitConfig(**kwargs)


def test_invalid_joint_limits_and_excessive_margin_are_rejected():
    with pytest.raises(ValueError, match='containing zero'):
        sample_turret_initialization(torch.tensor([[[.1, 1.], [-1., 1.]]]))
    with pytest.raises(ValueError, match='margin'):
        sample_turret_initialization(limits(1), TurretInitConfig(mode='mixed', limit_margin_deg=70.))
    with pytest.raises(ValueError, match='floating'):
        sample_turret_initialization(torch.ones(3, 2, 2, dtype=torch.long))


def env_method(name):
    tree = ast.parse((TRAIN/'turret_aim_env_gt.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TurretAimEnv')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def make_env(count=64):
    q = torch.full((count, 5), .25)
    lim = torch.tensor([[[-2., 2.]]*5]).repeat(count, 1, 1)
    lim[:, [3, 1]] = limits(count)
    env = SimpleNamespace(num_envs=count, device='cpu', step_dt=.02,
        cfg=SimpleNamespace(turret_init_config=TurretInitConfig(mode='mixed'), debug_diagnostics=False,
                            max_delta_yaw=.03, max_delta_pitch=.02, enable_dynamic_target=True,
                            locomotion_command_resample_s=10.),
        _turret_dof_ids=[3, 1], _yaw_dof_ids=[3], _pitch_dof_ids=[1],
        _turret_pos_targets=torch.full((count, 2), .7), _turret_reset_joint_pos=torch.zeros(count, 2),
        _turret_init_group=torch.zeros(count, dtype=torch.long),
        _turret_init_limit_axes=torch.zeros(count, dtype=torch.long),
        _aim_warmup_left_steps=torch.full((count,), 100, dtype=torch.long),
        _actions=torch.zeros(count, 3), _prev_actions=torch.zeros(count, 3), _delta_pos=torch.zeros(count, 2),
        _locomotion_commands=torch.ones(count, 3), _locomotion_command_time_left=torch.zeros(count),
        _locomotion_last_actions=torch.zeros(count, 12),
        _target_pos_w=torch.zeros(count, 3), _target_vel_w=torch.ones(count, 3), _target_acc_w=torch.zeros(count, 3),
        _episode_start_target_pos_w=torch.zeros(count, 3), _target_motion_mask=torch.ones(count, dtype=torch.bool),
        _target_relative_constraint_mask=torch.zeros(count, dtype=torch.bool),
        _target_constraint_fixed_anchor_mask=torch.zeros(count, dtype=torch.bool),
        _target_out_of_view_termination_mask=torch.zeros(count, dtype=torch.bool),
        _maybe_finalize_pending_target_resample=lambda: None, _invalidate_ballistic_cache=lambda: None,
        _handle_target_out_of_bounds=lambda *a, **kw: None, _enforce_target_above_ground=lambda ids: None,
        _reset_idx=lambda ids: None)
    env.turret = SimpleNamespace(data=SimpleNamespace(default_joint_pos=q,
        default_joint_vel=torch.full_like(q, .2), joint_pos=q.clone(), joint_pos_limits=lim))
    namespace = dict(sample_turret_initialization=sample_turret_initialization)
    node = env_method('_initialize_turret_joints')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual init method>', 'exec'), namespace)
    env._initialize_turret_joints = MethodType(namespace[node.name], env)
    return env


def test_actual_reset_writes_same_sample_to_simulation_and_drive_and_preserves_other_envs_and_legs():
    env = make_env(6)
    ids = torch.tensor([1, 4])
    saved = env._turret_pos_targets.clone()
    calls = {}
    env.turret.set_joint_position_target = lambda values, **kwargs: calls.update(command=values.clone(), **kwargs)
    env.turret.write_root_pose_to_sim = lambda *a: None
    env.turret.write_root_velocity_to_sim = lambda *a: None
    env.turret.write_joint_state_to_sim = lambda q, v, _, which: calls.update(q=q.clone(), v=v.clone(), ids=which.clone())
    body = env_method('_reset_idx').body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'joint_pos' for t in n.targets))
    stop = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == '_resample_targets')
    exec(compile(ast.Module(body=body[start:stop], type_ignores=[]), '<actual reset block>', 'exec'),
         dict(self=env, env_ids_t=ids, cast=cast, Sequence=Sequence, default_root_state=torch.zeros(2, 13)))
    torch.testing.assert_close(calls['q'][:, [3, 1]], calls['command'])
    torch.testing.assert_close(calls['command'], env._turret_reset_joint_pos[ids])
    assert torch.count_nonzero(calls['v'][:, [3, 1]]) == 0
    torch.testing.assert_close(calls['q'][:, [0, 2, 4]], torch.full((2, 3), .25))
    torch.testing.assert_close(calls['v'][:, [0, 2, 4]], torch.full((2, 3), .2))
    assert torch.equal(env._turret_pos_targets[[0, 2, 3, 5]], saved[[0, 2, 3, 5]])


def test_both_ss_and_sd_hold_sampled_pose_for_full_warmup_while_sd_target_moves():
    torch.manual_seed(12)
    env = make_env()
    ids = torch.arange(env.num_envs)
    env._initialize_turret_joints(ids, env.turret.data.joint_pos, torch.zeros(env.num_envs, 5))
    initial = env._turret_reset_joint_pos.clone()
    method = env_method('_pre_physics_step')
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'fire_intent' for t in n.targets))
    decrement = next(n for n in method.body if isinstance(n, ast.Assign)
                     and any(ast.unparse(t) == 'self._aim_warmup_left_steps' for t in n.targets))
    motion = next(n for n in method.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'self.cfg.enable_dynamic_target')
    prefix = compile(ast.Module(body=method.body[:end]+[decrement, motion], type_ignores=[]), '<actual warmup/motion>', 'exec')
    def actual_pre(actions):
        exec(prefix, dict(self=env, actions=actions, torch=torch))
    env._pre_physics_step = actual_pre
    path = TRAIN/'train_turret_mixed_gt.py'
    installer = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                     and n.name == '_install_mixed_training_logic')
    namespace = dict(Any=Any)
    exec(compile(ast.Module(body=[installer], type_ignores=[]), str(path), 'exec'), namespace)
    namespace[installer.name](env, (.4, .6, 0., 0.))
    static, dynamic = env._mix_case_id == 0, env._mix_case_id == 1
    assert static.any() and dynamic.any() and (static | dynamic).all()
    for _ in range(100):
        env._turret_pos_targets[:] = 0  # Even a stale external command cannot undo the warmup hold.
        env._pre_physics_step(torch.ones(env.num_envs, 3))
        torch.testing.assert_close(env._turret_pos_targets, initial)
    assert torch.count_nonzero(env._aim_warmup_left_steps) == 0
    assert torch.count_nonzero(env._target_pos_w[static]) == 0
    assert (env._target_pos_w[dynamic] > 0).all()
    env._pre_physics_step(torch.ones(env.num_envs, 3))
    torch.testing.assert_close(env._turret_pos_targets, initial+torch.tensor([.03, .02]))
    # The actual physics apply method sends these targets to the original drive.
    calls = []
    env._leg_dof_ids = [0, 2, 4]
    env._locomotion_joint_targets = torch.zeros(env.num_envs, 3)
    env.turret.set_joint_position_target = lambda values, **kw: calls.append((values.clone(), kw))
    namespace = {}
    apply = env_method('_apply_action')
    exec(compile(ast.Module(body=[apply], type_ignores=[]), '<actual apply>', 'exec'), namespace)
    namespace[apply.name](env)
    torch.testing.assert_close(calls[0][0], env._turret_pos_targets)


def training_args(*flags):
    path = TRAIN/'train_turret_mixed_gt.py'
    main = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
    namespace = dict(argparse=argparse, add_ppo_arguments=add_ppo_arguments)
    exec(compile(ast.Module(body=main.body[:end], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['parser'].parse_args(['--locomotion_policy', 'policy.pt', *flags])


@pytest.mark.parametrize('mode', ['gt', 'full_chain'])
def test_training_defaults_mix_recovery_poses_and_ss_sd_only(mode):
    args = training_args('--training_mode', mode)
    assert args.turret_init_mode == 'mixed'
    assert tuple(args.turret_init_probabilities) == (.4, .4, .2)
    assert resolve_training_cases(args) == (.4, .6, 0., 0.)
    assert args.yaw_error_mode == 'wrapped_los'


def test_zero_mode_can_reproduce_legacy_gt_case_mix_and_explicit_ss_sd_mix_is_honored():
    args = training_args('--turret_init_mode', 'zero')
    assert resolve_training_cases(args) == (.3, .3, .2, .2)
    args = training_args('--p_ss', '.5', '--p_sd', '.5', '--p_ds', '0', '--p_dd', '0')
    assert resolve_training_cases(args) == (.5, .5, 0., 0.)
    args = training_args('--p_ss', '.3', '--p_sd', '.3', '--p_ds', '.2', '--p_dd', '.2')
    assert resolve_training_cases(args) == (.3, .3, .2, .2)


@pytest.mark.parametrize('flags', [('--p_ss', '-1'), ('--p_sd', 'nan'),
    ('--p_ss', '0', '--p_sd', '0', '--p_ds', '0', '--p_dd', '0'),
    ('--training_mode', 'full_chain', '--p_ds', '.5')])
def test_invalid_case_mix_is_rejected_before_simulation(flags):
    with pytest.raises(ValueError):
        resolve_training_cases(training_args(*flags))

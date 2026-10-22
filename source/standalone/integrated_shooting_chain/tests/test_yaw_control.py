"""Real vector geometry and extracted environment observation tests, CPU only."""
import ast
import argparse
import math
from pathlib import Path
import sys
import types

import pytest
import torch

TRAIN = Path(__file__).resolve().parents[2] / 'test7_2_weight'
sys.path.insert(0, str(TRAIN))
from yaw_control import (CONTRACT_KEY, checkpoint_contract, contracts_match,
                         make_observation_contract_agent, observation_contract, reachable_yaw_goal)
from ppo_training_config import add_ppo_arguments


def rotate(vector, axis, angle):
    vector = torch.as_tensor(vector, dtype=torch.float64)
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    return (vector * math.cos(angle) + torch.cross(axis, vector, dim=-1) * math.sin(angle)
            + axis * torch.dot(axis, vector) * (1. - math.cos(angle)))


def solve(current, goal, lower=-180., upper=180., axis=(0., 0., 1.), mount_angle=0., pitch=0., offset=(.09, 0., .06)):
    axis = torch.tensor(axis, dtype=torch.float64)
    zero_forward = torch.tensor([math.cos(pitch), 0., -math.sin(pitch)], dtype=torch.float64)
    offset = torch.tensor(offset, dtype=torch.float64)
    forward = rotate(zero_forward, axis, math.radians(current))
    muzzle = rotate(offset, axis, math.radians(current))
    target = rotate(offset + 100. * zero_forward, axis, math.radians(goal))
    if mount_angle:
        mounting_axis = (1., 2., 3.)
        forward, muzzle, target, axis = [rotate(v, mounting_axis, mount_angle) for v in (forward, muzzle, target, axis)]
    return reachable_yaw_goal(torch.tensor([math.radians(current)], dtype=torch.float64),
                              forward[None], axis[None], target[None], muzzle[None],
                              torch.deg2rad(torch.tensor([[lower, upper]], dtype=torch.float64)))


@pytest.mark.parametrize('current,goal,error', [(180., -3., -183.), (-180., 3., 183.), (12., 15., 3.), (180., 180., 0.)])
def test_joint_error_uses_reachable_path_not_circle_shortcut(current, goal, error):
    result = solve(current, goal)
    assert result.exact_reachable.item()
    assert math.degrees(result.angle.item()) == pytest.approx(goal, abs=1e-7)
    assert math.degrees(result.error.item()) == pytest.approx(error, abs=1e-7)


@pytest.mark.parametrize('axis', [(0., 0., 1.), (0., 0., -1.)])
@pytest.mark.parametrize('mount_angle', [0., .35, 1.2])
@pytest.mark.parametrize('pitch', [0., .7, -1.0])
def test_geometry_accounts_for_mounting_pitch_axis_sign_and_lateral_offset(axis, mount_angle, pitch):
    result = solve(175., -37., axis=axis, mount_angle=mount_angle, pitch=pitch, offset=(.12, .04, .06))
    assert result.exact_reachable.item()
    assert math.degrees(result.angle.item()) == pytest.approx(-37., abs=1e-7)
    assert math.degrees(result.error.item()) == pytest.approx(-212., abs=1e-7)


def test_asymmetric_limits_and_multiple_turns_choose_nearest_equivalent():
    result = solve(100., -110., lower=-120., upper=110.)
    assert math.degrees(result.error.item()) == pytest.approx(-210.)
    result = solve(370., 10., lower=-720., upper=720.)
    assert result.exact_reachable.item()
    assert math.degrees(result.angle.item()) == pytest.approx(370.)
    assert result.error.item() == pytest.approx(0., abs=1e-9)


def test_unreachable_heading_returns_best_legal_boundary_and_marks_inexact():
    result = solve(0., 120., lower=-90., upper=90.)
    assert not result.exact_reachable.item()
    assert math.degrees(result.angle.item()) == pytest.approx(90.)


def test_usd_rounded_pi_is_tolerated_without_exceeding_limits():
    result = solve(math.degrees(3.14159), 180., lower=math.degrees(-3.14159), upper=math.degrees(3.14159))
    assert result.exact_reachable.item()
    assert result.angle.item() <= 3.14159
    assert result.error.item() == pytest.approx(0., abs=1e-6)


def test_parallel_target_direction_has_no_defined_yaw_and_holds_current():
    result = reachable_yaw_goal(torch.tensor([.4]), torch.tensor([[1., 0., 0.]]),
                                torch.tensor([[0., 0., 1.]]), torch.tensor([[0., 0., 100.]]),
                                torch.zeros(1, 3), torch.tensor([[-math.pi, math.pi]]))
    assert not result.exact_reachable.item()
    assert result.error.item() == 0.


def test_batched_mixed_boundaries():
    angles = torch.deg2rad(torch.tensor([180., -180., 10.]))
    goals = torch.deg2rad(torch.tensor([-3., 3., 20.]))
    forward = torch.stack((angles.cos(), angles.sin(), torch.zeros(3)), -1)
    targets = 100. * torch.stack((goals.cos(), goals.sin(), torch.zeros(3)), -1)
    result = reachable_yaw_goal(angles, forward, torch.tensor([[0., 0., 1.]]).expand(3, -1),
                                targets, torch.zeros(3, 3), torch.tensor([[-math.pi, math.pi]]).expand(3, -1))
    torch.testing.assert_close(torch.rad2deg(result.error), torch.tensor([-183., 183., 10.]), atol=1e-4, rtol=0)


def extract_method(name):
    path = TRAIN / 'turret_aim_env_gt.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TurretAimEnv')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(torch=torch, reachable_yaw_goal=reachable_yaw_goal,
                     quat_apply=lambda quat, vector: vector.clone(),
                     wrap_to_pi=lambda x: torch.atan2(torch.sin(x), torch.cos(x)))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def fake_env(mode):
    env = types.SimpleNamespace(num_envs=1, cfg=types.SimpleNamespace(
        yaw_error_mode=mode, fire_cooldown_steps=5, target_radius_range=(80., 150.),
        target_speed_range=(0., 5.), target_acc_range=(0., 0.), target_vertical_acc_range=(0., 0.),
        observation_space=28, state_space=31))
    env._yaw_dof_ids, env._pitch_dof_ids, env._turret_dof_ids = [0], [1], [0, 1]
    env._yaw_parent_body_id = 0
    env._yaw_axis_parent, env._yaw_pivot_parent = torch.tensor([0., 0., 1.]), torch.zeros(3)
    data = types.SimpleNamespace(joint_pos=torch.tensor([[math.pi, 0.]]), joint_vel=torch.zeros(1, 2),
        joint_pos_limits=torch.tensor([[[-math.pi, math.pi], [-1., 1.]]]),
        root_pos_w=torch.zeros(1, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
        body_pos_w=torch.zeros(1, 1, 3), body_quat_w=torch.tensor([[[1., 0., 0., 0.]]]),
        root_lin_vel_b=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]))
    env.turret = types.SimpleNamespace(data=data)
    env._actions = torch.zeros(1, 3)
    env._fire_intent = torch.zeros(1)
    env._fire_cooldown = torch.zeros(1, dtype=torch.long)
    env._target_pos_w = torch.tensor([[100. * math.cos(math.radians(-3.)), 100. * math.sin(math.radians(-3.)), 0.]])
    env._target_acc_w = torch.zeros(1, 3)
    env._rl_pred_pos_base = env._target_pos_w.clone()
    env._rl_pred_vel_base = torch.zeros(1, 3)
    env._rl_pred_distance = env._rl_pred_pos_base.norm(dim=-1)
    env._rl_pred_yaw_base = torch.tensor([math.radians(-3.)])
    env._rl_pred_pitch_base = torch.zeros(1)
    env._rl_pred_ready = torch.ones(1, dtype=torch.bool)
    env._use_gt_fallback_for_missing_rl_pred = False
    env._rotate_world_to_yaw_only_local = lambda quat, vector: vector.clone()
    env._rotate_yaw_only_local_to_world = lambda quat, vector: vector.clone()
    env._compute_base_relative_velocity_labels = lambda: torch.zeros(1, 3)
    env._compute_base_yaw_pitch_from_pos = lambda p: (torch.atan2(p[:, 1], p[:, 0]),
                                                      torch.atan2(-p[:, 2], p[:, :2].norm(dim=-1)))
    env._compute_turret_frame = lambda: (torch.tensor([[-.09, 0., 0.]]), data.root_quat_w, torch.tensor([[-1., 0., 0.]]))
    env._compute_muzzle_actual_yaw_pitch_in_yaw_only_base = lambda: (torch.tensor([math.pi]), torch.zeros(1))
    env._compute_muzzle_based_aim_refs = lambda: (torch.atan2(env._target_pos_w[:, 1], env._target_pos_w[:, 0]), torch.zeros(1), torch.zeros(1))
    env._compute_actor_reachable_yaw_error = types.MethodType(extract_method('_compute_actor_reachable_yaw_error'), env)
    env._get_observations = types.MethodType(extract_method('_get_observations'), env)
    return env


def test_actual_environment_observation_changes_only_actor_yaw_channel():
    old, new = fake_env('wrapped_los'), fake_env('joint_reachable')
    legacy, bounded = old._get_observations(), new._get_observations()
    assert legacy['policy'].shape == bounded['policy'].shape == (1, 28)
    assert bounded['critic'].shape == (1, 31)
    assert legacy['policy'][0, 7] == 5. and bounded['policy'][0, 7] == -5.
    indices = [i for i in range(28) if i != 7]
    torch.testing.assert_close(legacy['policy'][:, indices], bounded['policy'][:, indices])
    torch.testing.assert_close(legacy['critic'], bounded['critic'])
    assert math.degrees(new._actor_yaw_joint_error_rad.item()) == pytest.approx(-183., abs=1e-4)
    assert new._actor_yaw_goal_reachable.item()


def test_strict_actor_never_reads_gt_target_for_new_joint_goal():
    env = fake_env('joint_reachable')
    first = env._get_observations()['policy'].clone()
    env._target_pos_w[:] = torch.tensor([[0., 200., 10.]])
    torch.testing.assert_close(env._get_observations()['policy'], first)
    env._rl_pred_ready[:] = False
    env._get_observations()
    assert env._actor_yaw_joint_error_rad.item() == 0.
    assert not env._actor_yaw_goal_reachable.item()


def test_checkpoint_contract_roundtrip_and_untagged_legacy(tmp_path, capsys):
    class Agent:
        def get_full_state_weights(self):
            return dict(model={'weight': torch.tensor([1.])}, epoch=42)

        def set_full_state_weights(self, weights, set_epoch=True):
            self.received = weights
            return 'restored'

    expected = observation_contract('joint_reachable')
    agent = make_observation_contract_agent(Agent, expected)()
    state = agent.get_full_state_weights()
    path = tmp_path / 'new.pth'
    torch.save(state, path)
    assert contracts_match(checkpoint_contract(path), expected)
    state[CONTRACT_KEY]['version'] = 'mutated_copy'
    assert agent.get_full_state_weights()[CONTRACT_KEY]['version'] == 'joint_reachable_yaw_v2'
    old = Agent().get_full_state_weights()
    torch.save(old, path)
    previous = checkpoint_contract(path)
    assert previous['inferred_from_untagged_checkpoint']
    assert previous['yaw_error_mode'] == 'wrapped_los'
    assert not contracts_match(previous, expected)
    assert agent.set_full_state_weights(old) == 'restored'
    assert 'Warm-start observation change' in capsys.readouterr().out
    assert CONTRACT_KEY not in old


def test_unknown_observation_mode_is_rejected():
    with pytest.raises(ValueError, match='yaw_error_mode'):
        observation_contract('invalid')


@pytest.mark.parametrize('filename,required', [
    ('train_turret_mixed_gt.py', ['--locomotion_policy', 'policy.pt']),
    ('eval_with_track_gt.py', ['--checkpoint', 'old.pth', '--locomotion_policy', 'policy.pt']),
])
def test_train_and_eval_parser_default_and_legacy_override_match(filename, required):
    path = TRAIN / filename
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
    namespace = dict(argparse=argparse, add_ppo_arguments=add_ppo_arguments)
    exec(compile(ast.Module(body=main.body[:end], type_ignores=[]), str(path), 'exec'), namespace)
    assert namespace['parser'].parse_args(required).yaw_error_mode == 'wrapped_los'
    for mode in ('wrapped_los', 'joint_reachable'):
        assert namespace['parser'].parse_args(required + ['--yaw_error_mode', mode]).yaw_error_mode == mode


def test_environment_default_matches_shortest_turn_entry_points():
    tree = ast.parse((TRAIN / 'turret_aim_env_gt.py').read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TurretAimEnvCfg')
    mode = next(n.value for n in cfg.body if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name) and n.target.id == 'yaw_error_mode')
    assert ast.literal_eval(mode) == 'wrapped_los'


def test_changed_normalization_is_a_different_checkpoint_contract():
    assert not contracts_match(observation_contract('joint_reachable', scale=.1), observation_contract('joint_reachable'))

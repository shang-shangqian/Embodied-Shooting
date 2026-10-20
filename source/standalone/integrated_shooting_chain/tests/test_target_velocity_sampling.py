"""Check the actual environment methods without starting Isaac Sim."""

import ast
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


ENV_PATH = Path(__file__).resolve().parents[2] / "test7_2_weight/turret_aim_env_gt.py"
METHOD_NAMES = {
    "_sample_target_velocities", "_resample_targets", "_maybe_finalize_pending_target_resample",
    "_handle_target_out_of_bounds", "_enforce_target_above_ground",
}


@pytest.fixture
def env():
    tree = ast.parse(ENV_PATH.read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name in METHOD_NAMES]
    namespace = {
        "torch": torch, "math": math,
        "sample_uniform": lambda low, high, shape, device: torch.empty(shape, device=device).uniform_(low, high),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    count = 10000
    instance = SimpleNamespace(
        cfg=SimpleNamespace(
            target_speed_range=(0.6, 5.0), enable_dynamic_target=True,
            target_acc_range=(0.0, 0.0), target_vertical_acc_range=(0.0, 0.0),
            sample_target_in_camera_sector=False, debug_diagnostics=False,
            target_radius_range=(80.0, 150.0), target_yaw_range=(-math.pi / 2, math.pi / 2),
            target_cam_yaw_range=(-0.26, 0.26), target_cam_pitch_range=(-0.12, 0.12),
            vision_valid_distance_range=(80.0, 150.0), vision_distance_bounce_margin=0.2,
            constrain_target_relative_height=False, target_sphere_radius=0.35,
        ),
        device="cpu", num_envs=count, _vision_camera=None,
        _target_vel_w=torch.zeros(count, 3), _target_acc_w=torch.zeros(count, 3),
        _target_pos_w=torch.zeros(count, 3), _target_anchor_w=torch.zeros(count, 3),
        _episode_start_target_pos_w=torch.zeros(count, 3),
        _pending_target_resample=torch.ones(count, dtype=torch.bool),
        _vision_collect_warmup_left=torch.zeros(count, dtype=torch.long),
        _target_sphere_visuals_enabled=False,
        scene=SimpleNamespace(env_origins=torch.zeros(count, 3)),
        _sample_targets_in_camera_sector=lambda ids: None,
        _rotate_world_to_local=lambda quat, vector: vector.clone(),
        _rotate_local_to_world=lambda quat, vector: vector.clone(),
        _get_runtime_camera_pose_w=lambda: (torch.zeros(count, 3), torch.zeros(count, 4)),
    )
    for name in METHOD_NAMES:
        setattr(instance, name, MethodType(namespace[name], instance))
    return instance


def test_total_speed_is_uniform_and_directions_are_isotropic(env):
    torch.manual_seed(20261013)
    velocities = env._sample_target_velocities(60000)
    speeds = velocities.norm(dim=-1)
    assert speeds.min() >= 0.6 - 1e-6
    assert speeds.max() <= 5.0 + 1e-6
    # Equal-width speed bins should have equal probability.
    histogram = torch.histc(speeds, bins=10, min=0.6, max=5.0)
    assert torch.all((histogram / speeds.numel() - 0.1).abs() < 0.006)
    directions = velocities / speeds[:, None]
    assert torch.all(directions.mean(dim=0).abs() < 0.012)
    assert torch.all((directions.square().mean(dim=0) - 1.0 / 3.0).abs() < 0.012)
    for axis in range(3):
        assert (velocities[:, axis] > 0.5).any()
        assert (velocities[:, axis] < -0.5).any()
    # Direction remains isotropic at both ends of the speed distribution.
    for mask in (speeds < 1.5, speeds > 4.0):
        assert torch.all((directions[mask].square().mean(dim=0) - 1.0 / 3.0).abs() < 0.02)


@pytest.mark.parametrize("camera", [False, True])
def test_both_reset_paths_sample_three_dimensional_speed(env, camera):
    torch.manual_seed(42)
    ids = torch.arange(env.num_envs)
    if camera:
        env._vision_camera = object()
        env.cfg.sample_target_in_camera_sector = True
        env._maybe_finalize_pending_target_resample()
        assert not env._pending_target_resample.any()
    else:
        env._resample_targets(ids, sample_position=False, sample_velocity=True)
    speeds = env._target_vel_w.norm(dim=-1)
    assert torch.all((speeds >= 0.6 - 1e-6) & (speeds <= 5.0 + 1e-6))
    assert (env._target_vel_w[:, 2].abs() > 2.0).any()
    assert torch.count_nonzero(env._target_acc_w) == 0


def test_distance_fov_and_ground_reflections_preserve_speed(env):
    env._vision_camera = object()
    ids = torch.arange(4)
    env._target_pos_w[ids] = torch.tensor([
        [90.0, 0.0, 130.0], [80.0, 0.0, 0.0], [100.0, 50.0, 0.0], [100.0, 0.0, -0.1],
    ])
    env._target_vel_w[ids] = torch.tensor([
        [0.36, 0.0, 0.48], [-0.6, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, -2.0],
    ])
    speeds = env._target_vel_w[ids].norm(dim=-1)
    env._handle_target_out_of_bounds(ids, fixed_anchor=True)
    env._enforce_target_above_ground(ids)
    torch.testing.assert_close(env._target_vel_w[ids].norm(dim=-1), speeds)


@pytest.mark.parametrize("speed_range", [(2.0, 2.0), (0.0, 0.0)])
def test_fixed_speed(env, speed_range):
    env.cfg.target_speed_range = speed_range
    speeds = env._sample_target_velocities(100).norm(dim=-1)
    torch.testing.assert_close(speeds, torch.full_like(speeds, speed_range[0]))


@pytest.mark.parametrize("speed_range", [(-0.1, 5.0), (5.0, 0.6), (0.6, float("inf")), (float("nan"), 5.0)])
def test_invalid_speed_range(env, speed_range):
    env.cfg.target_speed_range = speed_range
    with pytest.raises(ValueError, match="target_speed_range"):
        env._sample_target_velocities(1)

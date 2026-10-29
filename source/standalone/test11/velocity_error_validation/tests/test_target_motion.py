"""Analytical collisions, speed bounds, and independence from camera motion."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speed_prediction.common.target_motion import advance_reflecting_targets, initialize_in_domain
from test_collection_ground_constraint import environment


def box_motion(position, velocity, dt):
    p = torch.tensor(position, dtype=torch.float64).reshape(-1, 3)
    v = torch.tensor(velocity, dtype=torch.float64).reshape(-1, 3)
    normals = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=p.dtype)[None].repeat(len(p), 1, 1)
    offsets = torch.tensor([0, 1, 0, 1, 0, 1], dtype=p.dtype)[None].repeat(len(p), 1)
    center = torch.tensor([[-10, 0, 0]], dtype=p.dtype).repeat(len(p), 1)
    return advance_reflecting_targets(p, v, dt, normals, offsets, center, 1., 100.)


def test_plane_collision_consumes_remaining_time():
    p, v, count = box_motion([.9, .5, .5], [1., 0., 0.], .2)
    torch.testing.assert_close(p, torch.tensor([[.9, .5, .5]], dtype=p.dtype))
    torch.testing.assert_close(v, torch.tensor([[-1., 0., 0.]], dtype=v.dtype))
    assert count.item() == 1


def test_multiple_collisions_in_one_step():
    p, v, count = box_motion([.25, .5, .5], [5., 0., 0.], .5)
    assert p[0, 0].item() == pytest.approx(.75)
    assert v[0, 0].item() == 5.
    assert count.item() == 2


def test_corner_and_boundary_start():
    p, v, count = box_motion([[.9, .9, .5], [1, .5, .5], [1, .5, .5]],
                             [[1, 1, 0], [1, 0, 0], [-1, 0, 0]], .2)
    torch.testing.assert_close(p, torch.tensor([[.9, .9, .5], [.8, .5, .5], [.8, .5, .5]], dtype=p.dtype))
    torch.testing.assert_close(v, torch.tensor([[-1, -1, 0], [-1, 0, 0], [-1, 0, 0]], dtype=v.dtype))
    assert count.tolist() == [2, 1, 0]


def shell_motion(p, v, dt):
    p = torch.as_tensor(p, dtype=torch.float64).reshape(-1, 3)
    v = torch.as_tensor(v, dtype=p.dtype).reshape(-1, 3)
    normals = torch.tensor([[[0., 0., 1.]]], dtype=p.dtype).repeat(len(p), 1, 1)
    offsets = torch.full((len(p), 1), 10., dtype=p.dtype)
    return advance_reflecting_targets(p, v, dt, normals, offsets, torch.zeros_like(p), 1., 3.)


def test_inner_and_outer_range_collisions():
    p, v, count = shell_motion([[2.9, 0, 0], [1.1, 0, 0], [3, 0, 0], [1, 0, 0]],
                               [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]], .2)
    torch.testing.assert_close(p[:, 0], torch.tensor([2.9, 1.1, 2.8, 1.2], dtype=p.dtype))
    torch.testing.assert_close(v[:, 0], torch.tensor([-1, 1, -1, 1], dtype=v.dtype))
    assert count.tolist() == [1, 1, 1, 1]


def test_oblique_range_collision_is_reversible_and_preserves_speed():
    original = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
    velocity = torch.tensor([[0., 2., 0.]], dtype=original.dtype)
    p, v, count = shell_motion(original, velocity, 2.)
    assert count.item() >= 1
    torch.testing.assert_close(torch.linalg.vector_norm(v, dim=-1), torch.tensor([2.], dtype=v.dtype))
    back, back_v, _ = shell_motion(p, -v, 2.)
    torch.testing.assert_close(back, original, atol=1e-12, rtol=0)
    torch.testing.assert_close(back_v, -velocity, atol=1e-12, rtol=0)


def test_zero_speed_does_not_generate_collisions():
    p, v, count = box_motion([.5, .5, .5], [0, 0, 0], .02)
    assert torch.isfinite(p).all() and torch.isfinite(v).all()
    assert count.item() == 0


def test_invalid_start_fails_without_teleporting():
    with pytest.raises(RuntimeError, match="outside fixed motion domain"):
        box_motion([1.1, .5, .5], [-1, 0, 0], .02)


def test_float32_sphere_roundoff_reflects_slow_velocity_without_position_correction():
    p = torch.tensor([[150., .03, 0.]], dtype=torch.float32)
    v = torch.tensor([[.01, 0., 0.]], dtype=p.dtype)
    normals = torch.tensor([[[0., 0., 1.]]], dtype=p.dtype)
    offsets = torch.tensor([[10.]], dtype=p.dtype)
    result, velocity, count = advance_reflecting_targets(p, v, .02, normals, offsets, torch.zeros_like(p), 80., 150.)
    assert count.item() == 1
    assert velocity[0, 0] < 0
    assert torch.linalg.vector_norm((result-p).double()) <= .01*.02+1e-5
    assert torch.linalg.vector_norm(result.double()) < 150.


def test_exhausted_collision_budget_fails_instead_of_clamping_position():
    with pytest.raises(RuntimeError, match="iteration limit"):
        box_motion([.5, .5, .5], [5., 0., 0.], 20.)


def fixed_env(positions, velocities):
    env = environment(positions, velocities)
    env.cfg.target_boundary_mode = "fixed_reference_toi"
    env.cfg.target_fov_margin_fraction = .1
    env._initialize_fixed_target_motion(torch.arange(env.num_envs))
    return env


def test_ground_collision_uses_remaining_time_and_preserves_speed():
    env = fixed_env([[100, 0, .45]], [[0, 0, -5]])
    env._advance_dynamic_target(.02)
    assert env._target_pos_w[0, 2].item() == pytest.approx(.452)
    assert env._target_vel_w[0, 2].item() == 5.
    assert env._target_boundary_collision_count.item() == 1


def test_motion_never_reads_live_camera_or_projects_positions():
    env = fixed_env([[100, 20, 14], [149, 0, .5]], [[0, 4, 2], [4, 0, -2]])
    initial_speed = torch.linalg.vector_norm(env._target_vel_w, dim=-1)

    def forbidden(*args):
        raise AssertionError("Live camera/position projection must not affect fixed trajectories")

    env.get_mounted_camera_pose_w = forbidden
    env._get_runtime_camera_pose_w = forbidden
    env._constrain_dynamic_target_to_camera_views = forbidden
    env._enforce_target_above_ground = forbidden
    collisions = 0
    for _ in range(600):
        previous = env._target_pos_w.clone()
        env._advance_dynamic_target(.02)
        assert torch.all(torch.linalg.vector_norm(env._target_pos_w-previous, dim=-1) <= initial_speed*.02+1e-10)
        assert torch.all(env._target_pos_w[:, 2] >= .401-1e-10)
        torch.testing.assert_close(torch.linalg.vector_norm(env._target_vel_w, dim=-1), initial_speed)
        collisions += int(env._target_boundary_collision_count.sum())
    assert collisions > 0


def test_partial_reset_does_not_modify_another_targets_position_or_velocity():
    env = fixed_env([[100, 0, 2], [100, 0, 4]], [[1, 2, 3], [1, 2, 3]])
    untouched = env._target_pos_w[1].clone()
    velocity = env._target_vel_w.clone()
    env._target_pos_w[0, 2] = -10
    env._initialize_fixed_target_motion(torch.tensor([0]))
    torch.testing.assert_close(env._target_pos_w[1], untouched, atol=0, rtol=0)
    torch.testing.assert_close(env._target_vel_w, velocity, atol=0, rtol=0)
    assert env._target_pos_w[0, 2] >= .401-1e-9


def test_float32_steps_keep_speed_bound_and_label_consistency_away_from_collisions():
    env = fixed_env([[100, 0, .45], [149, 0, 2]], [[0, 0, -5], [4, 1, 2]])
    env._target_pos_w = env._target_pos_w.float()
    env._target_vel_w = env._target_vel_w.float()
    initial_speed = torch.linalg.vector_norm(env._target_vel_w, dim=-1)
    for _ in range(500):
        previous = env._target_pos_w.clone()
        env._advance_dynamic_target(.02)
        average = (env._target_pos_w-previous)/.02
        assert torch.all(torch.linalg.vector_norm(average, dim=-1) <= initial_speed+1e-3)
        no_collision = env._target_boundary_collision_count == 0
        torch.testing.assert_close(average[no_collision], env._target_vel_w[no_collision], atol=8e-4, rtol=0)
        torch.testing.assert_close(torch.linalg.vector_norm(env._target_vel_w, dim=-1), initial_speed, atol=2e-5, rtol=0)


def test_base_motion_hook_preserves_legacy_update_order():
    import ast
    path = Path(__file__).resolve().parents[2] / "turret_aim_env.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TurretAimEnv")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_advance_dynamic_target")
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    from types import SimpleNamespace
    calls = []
    env = SimpleNamespace(_target_pos_w=torch.zeros(1, 3), _target_vel_w=torch.ones(1, 3),
                          _handle_target_out_of_bounds=lambda: calls.append("bounds"),
                          _enforce_target_above_ground=lambda: calls.append("ground"))
    scope["_advance_dynamic_target"](env, .02)
    torch.testing.assert_close(env._target_pos_w, torch.full((1, 3), .02))
    assert calls == ["bounds", "ground"]

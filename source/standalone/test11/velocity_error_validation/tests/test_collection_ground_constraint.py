"""Exercise the actual collection geometry on CPU without starting Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def collection_class():
    # Load the class body itself, substituting only the Isaac base class.
    # Geometry and sampling methods remain the production implementation.
    path = Path(__file__).resolve().parents[2] / "turret_aim_env_collect.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TurretAimCollectEnv")

    class Base:
        def _enforce_target_above_ground(self, env_ids_t=None):
            self.legacy_height_constraint_called = True

    scope = {"torch": torch, "TurretAimEnv": Base}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["TurretAimCollectEnv"]


def environment(positions, velocities, paired=True):
    env = collection_class()()
    env.device = "cpu"
    env._target_pos_w = torch.tensor(positions, dtype=torch.float64)
    env._target_vel_w = torch.tensor(velocities, dtype=torch.float64)
    env._target_acc_w = torch.zeros_like(env._target_vel_w)
    env.num_envs = len(positions)
    env.cfg = SimpleNamespace(
        constrain_dynamic_target_to_camera_fov=True,
        validation_reference_target_trajectory=paired,
        target_sphere_radius=.4,
        target_ground_clearance_m=.001,
        camera_width=1024,
        camera_height=576,
        camera_clipping_range=(.1, 1000000),
        target_radius_range=(80., 150.),
        target_cam_yaw_range=(-.26, .26),
        target_cam_pitch_range=(-.14, .14),
        target_speed_range=(0., 5.),
        target_acc_range=(0., 0.),
        target_vertical_acc_range=(0., 0.),
        enable_dynamic_target=True,
        validation_trajectory_seed=20261012,
    )
    # ROS optical camera: right=-world_y, down=-world_z, forward=world_x.
    env._rotate_world_to_local = lambda q, v: torch.stack((-v[:, 1], -v[:, 2], v[:, 0]), dim=-1)
    env._rotate_local_to_world = lambda q, v: torch.stack((v[:, 2], -v[:, 0], -v[:, 1]), dim=-1)
    camera_position = torch.tensor([[0., 0., .45]], dtype=torch.float64).repeat(env.num_envs, 1)
    camera_quaternion = torch.zeros(env.num_envs, 4, dtype=torch.float64)
    env._get_validation_reference_camera_pose_w = lambda: (camera_position, camera_quaternion)
    env.get_mounted_camera_pose_w = lambda: (camera_position, camera_quaternion)
    env._get_runtime_camera_pose_w = lambda: (camera_position, camera_quaternion)
    intrinsics = torch.tensor([[1024*10/5.76, 0, 512], [0, 576*10/3.24, 288], [0, 0, 1]], dtype=torch.float64)
    env._vision_camera = SimpleNamespace(data=SimpleNamespace(intrinsic_matrices=intrinsics.repeat(env.num_envs, 1, 1)))
    env._pending_target_resample = torch.ones(env.num_envs, dtype=torch.bool)
    return env


def assert_feasible(env):
    assert torch.all(env._target_pos_w[:, 2] >= .401-1e-9)
    camera, q = env._get_runtime_camera_pose_w()
    local = env._rotate_world_to_local(q, env._target_pos_w-camera)
    distance = torch.linalg.vector_norm(local, dim=-1)
    assert torch.all(distance >= 80-1e-3)
    assert torch.all(distance <= 150+1e-3)
    fx = env._vision_camera.data.intrinsic_matrices[:, 0, 0]
    fy = env._vision_camera.data.intrinsic_matrices[:, 1, 1]
    x, y, z = local.unbind(-1)
    assert torch.all(z >= .5-1e-6)
    assert torch.all(fx*x+512*z >= fx*.4-1e-3)
    assert torch.all(-fx*x+511*z >= fx*.4-1e-3)
    assert torch.all(fy*y+288*z >= fy*.4-1e-3)
    assert torch.all(-fy*y+287*z >= fy*.4-1e-3)


def test_whole_sphere_clears_ground_and_reflection_preserves_speed():
    env = environment([[100, 0, -3], [100, 0, .2], [100, 0, 4]], [[1, 2, -3], [1, 2, 3], [1, 2, -3]])
    before = torch.linalg.vector_norm(env._target_vel_w, dim=-1)
    env._enforce_target_above_ground()
    torch.testing.assert_close(env._target_pos_w[:, 2], torch.tensor([.401, .401, 4], dtype=torch.float64))
    torch.testing.assert_close(env._target_vel_w[:, 2], torch.tensor([3, 3, -3], dtype=torch.float64))
    torch.testing.assert_close(torch.linalg.vector_norm(env._target_vel_w, dim=-1), before)


def test_subset_and_empty_reset_do_not_change_other_environments():
    env = environment([[100, 0, -2], [100, 0, -3]], [[0, 0, -1], [0, 0, -2]])
    env._enforce_target_above_ground(torch.tensor([], dtype=torch.long))
    assert env._target_pos_w[0, 2] == -2
    env._enforce_target_above_ground(torch.tensor([1]))
    assert env._target_pos_w[0, 2] == -2
    assert env._target_vel_w[0, 2] == -1
    assert env._target_pos_w[1, 2] == .401
    assert env._target_vel_w[1, 2] == 2


def test_ground_range_and_camera_constraints_can_hold_together():
    env = environment([[100, 0, -10], [79, 0, .35], [151, 0, .5], [100, 40, 2], [100, -40, 35]], [[1, 2, -3]]*5)
    speed = torch.linalg.vector_norm(env._target_vel_w, dim=-1)
    env._constrain_dynamic_target_to_camera_views()
    assert_feasible(env)
    torch.testing.assert_close(torch.linalg.vector_norm(env._target_vel_w, dim=-1), speed)


def test_continuous_motion_bounces_at_ground_without_speed_growth():
    env = environment([[81, 0, .41], [149, 0, .6], [100, 25, 4]], [[0, 0, -5], [4, 0, -2], [0, 4, -2]], paired=False)
    speed = torch.linalg.vector_norm(env._target_vel_w, dim=-1)
    for _ in range(200):
        env._target_pos_w += .02*env._target_vel_w
        env._constrain_dynamic_target_to_camera_views()
        assert_feasible(env)
    torch.testing.assert_close(torch.linalg.vector_norm(env._target_vel_w, dim=-1), speed)


def test_ground_has_priority_if_a_camera_projection_is_incompatible():
    env = environment([[100, 0, 2]], [[0, 0, -1]])

    def incompatible_camera(*args):
        env._target_pos_w[:, 2] = -1

    env._reflect_target_into_camera_frustum = incompatible_camera
    env._constrain_dynamic_target_to_camera_views()
    assert env._target_pos_w[0, 2] >= .401
    assert env._target_vel_w[0, 2] > 0


def test_d_condition_initializes_directly_above_ground_without_placeholder():
    env = environment([[0, 0, 0]]*16, [[0, 0, 0]]*16)

    def unexpected_placeholder(*args):
        raise AssertionError("D reference trajectory must not use placeholder targets")

    env._sample_placeholder_targets = unexpected_placeholder
    env._resample_targets(torch.arange(16), True, True)
    assert not env._pending_target_resample.any()
    assert_feasible(env)
    assert torch.all(torch.linalg.vector_norm(env._target_vel_w, dim=-1) <= 5+1e-6)


def test_legacy_height_constraint_remains_for_non_camera_constrained_mode():
    env = environment([[100, 0, 2]], [[0, 0, 0]])
    env.cfg.constrain_dynamic_target_to_camera_fov = False
    env._enforce_target_above_ground()
    assert env.legacy_height_constraint_called

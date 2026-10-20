from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrated_shooting_chain.video_scene_view import (
    SceneVideoView, fitted_camera_pose, fitted_pool_camera_pose, hit_ballistic_path, project_world_points, truncate_ballistic_path,
)


def observer_matrices(eye, lookat, horizontal_fov, aspect):
    forward = (lookat - eye) / np.linalg.norm(lookat - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4)
    view[:3, :3] = np.column_stack([right, up, -forward])
    view[3, :3] = -eye @ view[:3, :3]
    projection = np.zeros((4, 4))
    projection[0, 0] = 1.0 / np.tan(horizontal_fov / 2.0)
    projection[1, 1] = projection[0, 0] * aspect
    projection[2, 3] = -1.0
    return view, projection


@pytest.mark.parametrize("target", [[100.0, 0.0, 2.0], [10.0, 150.0, 4.0], [-80.0, -20.0, 15.0]])
def test_observer_frames_both_robot_and_target_with_margin(target):
    robot = np.array([500.0, 200.0, 0.4])
    target = np.asarray(target) + np.array([500.0, 200.0, 0.0])
    horizontal_fov = np.deg2rad(60.0)
    eye, lookat = fitted_camera_pose(robot, target, horizontal_fov)
    view, projection = observer_matrices(eye, lookat, horizontal_fov, 1280 / 720)
    pixels = project_world_points([robot, target], view, projection, (1280, 720))
    assert np.isfinite(pixels).all()
    assert (pixels[:, 0] > 100).all() and (pixels[:, 0] < 1180).all()
    assert (pixels[:, 1] > 280).all() and (pixels[:, 1] < 660).all()
    assert pixels[0, 0] < pixels[1, 0]


def test_projection_rejects_points_behind_camera():
    view = np.eye(4)
    projection = np.zeros((4, 4))
    projection[0, 0] = projection[1, 1] = 1.0
    projection[2, 3] = -1.0
    pixels = project_world_points([[0, 0, -1], [0, 0, 1]], view, projection, (1280, 720))
    np.testing.assert_allclose(pixels[0], [640, 360])
    assert np.isnan(pixels[1]).all()


def test_observer_frames_all_100_targets_and_robot_with_margin():
    rng = np.random.default_rng(61)
    robot = np.array([500., 200., .4])
    targets = robot + np.column_stack([rng.uniform(80., 150., 100),
                                      rng.uniform(-40., 40., 100), rng.uniform(0., 18., 100)])
    eye, lookat = fitted_pool_camera_pose(robot, targets, np.deg2rad(60.), 1280 / 720)
    view, projection = observer_matrices(eye, lookat, np.deg2rad(60.), 1280 / 720)
    pixels = project_world_points(np.concatenate([robot[None], targets]), view, projection, (1280, 720))
    assert np.isfinite(pixels).all()
    assert ((pixels[:, 0] > 20) & (pixels[:, 0] < 1260)).all()
    assert ((pixels[:, 1] > 20) & (pixels[:, 1] < 700)).all()


def test_path_stops_at_evaluated_closest_time():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    positions = np.array([[0, 0, 0], [10, 0, 1], [20, 0, 2], [30, 0, 3]])
    path = truncate_ballistic_path(times, positions, 0.15)
    np.testing.assert_allclose(path[0], positions[0])
    np.testing.assert_allclose(path[-1], [15, 0, 1.5])
    assert (path[:, 0] <= 15.0).all()


@pytest.mark.parametrize("enable_air_drag", [False, True])
def test_path_uses_postphysics_muzzle_and_preserves_environment_state(enable_air_drag):
    root_pos = torch.tensor([[0.0, 0.0, 1.0], [500.0, 0.0, 1.0]])
    initial = root_pos.clone()
    env = SimpleNamespace(
        turret=SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos)), _muzzle_body_id=None,
        _compute_turret_frame=lambda: (root_pos, None, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
        _compute_muzzle_velocity_w=lambda **kwargs: torch.zeros_like(root_pos),
        _compute_air_drag_accel_w=lambda velocity: torch.zeros_like(velocity),
        cfg=SimpleNamespace(muzzle_speed=100.0, enable_air_drag=enable_air_drag),
        _gravity_w=torch.tensor([0.0, 0.0, -10.0]),
        _ballistic_time_samples=torch.tensor([0.0, 0.1, 0.2, 0.3]),
        _ballistic_dt_samples=torch.tensor([0.1, 0.1, 0.1]),
        _cached_closest_time=torch.tensor([0.1, 0.15]),
    )
    path = hit_ballistic_path(env, 1)
    np.testing.assert_allclose(path[0], [500.0, 0.0, 1.0])
    np.testing.assert_allclose(path[-1], [515.0, 0.0, 0.875], atol=1e-5)
    torch.testing.assert_close(root_pos, initial)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("hit", [False, True])
def test_observer_robot_detail_only_and_no_object_markers(terminal, hit, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    poses = []
    render_calls = []
    captions = []
    lines = []
    robot_rgb = torch.full((1, 576, 1024, 3), 123, dtype=torch.uint8)
    original_put_text = cv2.putText
    original_line = cv2.line

    def capture_caption(image, text, *args, **kwargs):
        captions.append(text)
        return original_put_text(image, text, *args, **kwargs)

    def capture_line(image, *args, **kwargs):
        lines.append(args)
        return original_line(image, *args, **kwargs)

    def reject_object_circle(*args, **kwargs):
        pytest.fail("Object annotation circles should not be drawn.")

    monkeypatch.setattr(cv2, "putText", capture_caption)
    monkeypatch.setattr(cv2, "circle", reject_object_circle)
    monkeypatch.setattr(cv2, "line", capture_line)
    monkeypatch.setattr("integrated_shooting_chain.video_scene_view.hit_ballistic_path",
                        lambda base_env, env_id: np.array([[0.0, 0.0, 0.4], [100.0, 0.0, 2.0]]))

    def set_camera_view(**kwargs):
        poses.append(kwargs)

    def render():
        render_calls.append("observer")

    def frustum():
        pose = poses[-1]
        view, projection = observer_matrices(np.asarray(pose["eye"]), np.asarray(pose["target"]),
                                            np.deg2rad(60), 1280 / 720)
        return SimpleNamespace(ComputeViewMatrix=lambda: view, ComputeProjectionMatrix=lambda: projection)

    env = SimpleNamespace(
        turret=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, 0.4]]))),
        _target_pos_w=torch.tensor([[100.0, 0.0, 2.0]]),
        _hit_flag=torch.tensor([hit]),
        _vision_camera=SimpleNamespace(data=SimpleNamespace(output={"rgb": robot_rgb})),
        sim=SimpleNamespace(set_camera_view=set_camera_view, render=render),
        render=lambda: np.full((720, 1280, 3), 45, dtype=np.uint8),
        _sync_render_camera_before_read=lambda: render_calls.append("terminal_sync"),
        cfg=SimpleNamespace(viewer=SimpleNamespace(cam_prim_path="observer_only", resolution=(1280, 720))),
    )
    observer = SceneVideoView.__new__(SceneVideoView)
    observer.env = env
    observer.env_id = 0
    observer.horizontal_fov = np.deg2rad(60)
    observer.camera = SimpleNamespace(GetCamera=lambda: SimpleNamespace(frustum=frustum()))
    scene, camera_image = observer(terminal)
    assert scene.shape == (720, 1280, 3)
    assert len(render_calls) == 2
    assert len(poses) == 3
    assert poses[0] == poses[-1]
    assert all(pose["camera_prim_path"] == "observer_only" for pose in poses)
    assert np.all(camera_image == 123)
    assert "Robot detail (observer)" in captions
    assert not any(text in ("ROBOT", "TARGET") or "Target detail" in text for text in captions)
    assert poses[1]["target"] == env.turret.data.root_pos_w[0].tolist()
    assert np.all(scene[440:595, 980:1265] == 45)
    assert len(lines) == int(hit)
    torch.testing.assert_close(robot_rgb, torch.full_like(robot_rgb, 123))

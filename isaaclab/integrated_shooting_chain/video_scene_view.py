"""Diagnostic-only observer view and projected ballistic annotations for videos."""

from __future__ import annotations

import numpy as np


def fitted_camera_pose(robot, target, horizontal_fov_radians):
    robot = np.asarray(robot, dtype=float)
    target = np.asarray(target, dtype=float)
    heading = target - robot
    heading[2] = 0.0
    heading_norm = np.linalg.norm(heading)
    heading = heading / heading_norm if heading_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    side = np.cross(heading, [0.0, 0.0, 1.0])
    radius = 0.5 * np.linalg.norm(target - robot) + 3.0
    distance = max(8.0, 1.35 * radius / np.tan(horizontal_fov_radians / 2.0))
    lookat = (robot + target) / 2.0
    eye = lookat + side * distance + np.array([0.0, 0.0, 0.32 * distance])
    return eye, lookat


def fitted_pool_camera_pose(robot, targets, horizontal_fov_radians, aspect):
    """Keep the robot and the whole original pool in a stable observer view."""
    robot = np.asarray(robot, dtype=float)
    targets = np.asarray(targets, dtype=float).reshape(-1, 3)
    points = np.concatenate([robot[None], targets], axis=0)
    lookat = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = np.linalg.norm(points - lookat, axis=-1).max() + 3.0
    half_fov = min(horizontal_fov_radians / 2.0,
                   np.arctan(np.tan(horizontal_fov_radians / 2.0) / aspect))
    distance = max(8.0, 1.15 * radius / np.sin(half_fov))
    heading = targets.mean(axis=0) - robot
    heading[2] = 0.0
    if np.linalg.norm(heading) < 1e-6:
        heading = np.array([1., 0., 0.])
    heading /= np.linalg.norm(heading)
    direction = np.cross(heading, [0., 0., 1.]) + np.array([0., 0., 0.32])
    direction /= np.linalg.norm(direction)
    return lookat + distance * direction, lookat


def project_world_points(points, view_matrix, projection_matrix, resolution):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    clip = homogeneous @ np.asarray(view_matrix) @ np.asarray(projection_matrix)
    valid = clip[:, 3] > 1e-6
    normalized = clip[:, :2] / np.where(valid, clip[:, 3], 1.0)[:, None]
    width, height = resolution
    pixels = np.column_stack([(normalized[:, 0] + 1.0) * width / 2.0,
                              (1.0 - normalized[:, 1]) * height / 2.0])
    pixels[~valid] = np.nan
    return pixels


def truncate_ballistic_path(times, positions, closest_time, stride=3):
    times = np.asarray(times)
    positions = np.asarray(positions)
    closest_time = float(np.clip(closest_time, times[0], times[-1]))
    last_index = max(1, int(np.searchsorted(times, closest_time, side="left")))
    last_index = min(last_index, len(times) - 1)
    fraction = (closest_time - times[last_index - 1]) / (times[last_index] - times[last_index - 1])
    endpoint = positions[last_index - 1] + fraction * (positions[last_index] - positions[last_index - 1])
    selected = positions[:last_index:max(1, stride)]
    return np.concatenate([selected, endpoint[None]], axis=0)


def hit_ballistic_path(base_env, env_id):
    import torch

    muzzle_pos, _, muzzle_forward = base_env._compute_turret_frame()
    body_pos = (base_env.turret.data.root_pos_w if base_env._muzzle_body_id is None
                else base_env.turret.data.body_pos_w[:, base_env._muzzle_body_id])
    muzzle_velocity = base_env._compute_muzzle_velocity_w(body_pos_w=body_pos, muzzle_pos_w=muzzle_pos)
    position = muzzle_pos[env_id:env_id + 1].clone()
    velocity = muzzle_velocity[env_id:env_id + 1] + muzzle_forward[env_id:env_id + 1] * base_env.cfg.muzzle_speed
    gravity = base_env._gravity_w.to(position).reshape(1, 3)
    times = base_env._ballistic_time_samples.to(position)
    if not base_env.cfg.enable_air_drag:
        trajectory = position + velocity * times[:, None] + 0.5 * gravity * times[:, None].square()
    else:
        history = [position[0].clone()]
        for duration in base_env._ballistic_dt_samples:
            duration = duration.to(position)
            acceleration = gravity + base_env._compute_air_drag_accel_w(velocity)
            midpoint_velocity = velocity + 0.5 * acceleration * duration
            midpoint_acceleration = gravity + base_env._compute_air_drag_accel_w(midpoint_velocity)
            position = position + midpoint_velocity * duration
            velocity = velocity + midpoint_acceleration * duration
            history.append(position[0].clone())
        trajectory = torch.stack(history)
    return truncate_ballistic_path(
        times.detach().cpu().numpy(), trajectory.detach().cpu().numpy(),
        base_env._cached_closest_time[env_id].item(),
    )


class SceneVideoView:
    def __init__(self, base_env, env_id):
        from pxr import UsdGeom

        self.env = base_env
        self.env_id = env_id
        self.camera = UsdGeom.Camera(base_env.sim.stage.GetPrimAtPath(base_env.cfg.viewer.cam_prim_path))
        width, height = base_env.cfg.viewer.resolution
        aperture = self.camera.GetHorizontalApertureAttr().Get()
        self.camera.GetVerticalApertureAttr().Set(aperture * height / width)
        self.horizontal_fov = 2.0 * np.arctan(aperture / (2.0 * self.camera.GetFocalLengthAttr().Get()))

    def __call__(self, terminal):
        import cv2

        env = self.env
        robot = env.turret.data.root_pos_w[self.env_id].detach().cpu().numpy()
        target = env._target_pos_w[self.env_id].detach().cpu().numpy()
        salvo = getattr(env, "_static_salvo", None)
        if salvo is not None:
            if bool(salvo.ready[self.env_id]):
                targets = salvo.positions[self.env_id].detach().cpu().numpy()
                target = targets.mean(axis=0)
                width, height = env.cfg.viewer.resolution
                eye, lookat = fitted_pool_camera_pose(robot, targets, self.horizontal_fov, width / height)
            else:
                target = robot + np.array([10., 0., 0.])
                eye, lookat = fitted_camera_pose(robot, target, self.horizontal_fov)
        else:
            eye, lookat = fitted_camera_pose(robot, target, self.horizontal_fov)
        env.sim.set_camera_view(eye=eye.tolist(), target=lookat.tolist(),
                                camera_prim_path=env.cfg.viewer.cam_prim_path)
        if terminal:
            env._sync_render_camera_before_read()
        else:
            env.sim.render()
        scene = env.render().copy()
        camera_image = env._vision_camera.data.output["rgb"][self.env_id, ..., :3].detach().cpu().numpy().copy()
        frustum = self.camera.GetCamera().frustum
        view_matrix = np.asarray(frustum.ComputeViewMatrix()).copy()
        projection_matrix = np.asarray(frustum.ComputeProjectionMatrix()).copy()

        def project(points):
            return project_world_points(points, view_matrix, projection_matrix,
                                        env.cfg.viewer.resolution)

        hit = bool(env._hit_flag[self.env_id].item()) if salvo is None else bool(salvo.last_hit[self.env_id].item())
        if hit:
            trajectory = project(hit_ballistic_path(env, self.env_id))
            for start, end in zip(trajectory[:-1], trajectory[1:]):
                if np.isfinite(start).all() and np.isfinite(end).all():
                    cv2.line(scene, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                             (255, 220, 0), 4, cv2.LINE_AA)
        distance = np.linalg.norm(target - robot)
        caption = (f"Observer overview | robot-target distance={distance:.1f}m" if salvo is None
                   else "SS target pool | yellow: evaluated hit path")
        cv2.putText(scene, caption, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        heading = target - robot
        heading[2] = 0.0
        heading /= max(np.linalg.norm(heading), 1e-6)
        side = np.cross(heading, [0.0, 0.0, 1.0])
        width, height = env.cfg.viewer.resolution
        detail_width = min(288, width // 4)
        detail_height = round(detail_width * height / width)
        detail_top = height - 120 - detail_height
        try:
            detail_eye = robot - heading * 2.5 + side * 2.5 + np.array([0.0, 0.0, 1.5])
            env.sim.set_camera_view(eye=detail_eye.tolist(), target=robot.tolist(),
                                    camera_prim_path=env.cfg.viewer.cam_prim_path)
            env.sim.render()
            detail = cv2.resize(env.render(), (detail_width, detail_height))
            detail_left = 12
            scene[detail_top:detail_top + detail_height, detail_left:detail_left + detail_width] = detail
            cv2.rectangle(scene, (detail_left - 1, detail_top - 1),
                          (detail_left + detail_width, detail_top + detail_height), (255, 255, 255), 1)
            cv2.putText(scene, "Robot detail (observer)", (detail_left, detail_top - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        finally:
            env.sim.set_camera_view(eye=eye.tolist(), target=lookat.tolist(),
                                    camera_prim_path=env.cfg.viewer.cam_prim_path)
        return scene, camera_image

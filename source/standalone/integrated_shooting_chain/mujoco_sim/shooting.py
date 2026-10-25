"""MuJoCo turret execution and source-compatible shooting evaluation.

Target truth is accepted only by ballistic_metrics(), never by a policy or
perception method. The actor's robot channels use the full body frame; GRU
ego-motion and target channels use the yaw-only base frame.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..shooting_chain import RobotState

WORKSPACE = Path(__file__).resolve().parents[4]
SOURCE_ENV = WORKSPACE / 'source/standalone/test7_2_weight/turret_aim_env_gt.py'


def source_settings():
    """Read literal training/evaluation defaults without importing Isaac Sim."""
    tree = ast.parse(SOURCE_ENV.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TurretAimEnvCfg')
    values = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            key, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key, value = node.targets[0].id, node.value
        else:
            continue
        try:
            values[key] = ast.literal_eval(value)
        except (ValueError, TypeError):
            pass
    return values


def checkpoint_settings(checkpoint):
    """Keep saved firing/training parameters paired with their policy weights."""
    settings = source_settings()
    path = Path(checkpoint).resolve().parent.parent/'training_settings.json'
    if path.is_file():
        arguments = json.loads(path.read_text())['arguments']
        for name in ('fire_intent_threshold', 'fire_intent_hold_steps', 'max_shots_per_episode'):
            if name in arguments:
                settings[name] = arguments[name]
        if 'target_speed_min' in arguments and 'target_speed_max' in arguments:
            settings['target_speed_range'] = (arguments['target_speed_min'], arguments['target_speed_max'])
    return settings


def yaw_rotation(rotation):
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def camera_rotation_homography(intrinsics, rotation_current, rotation_reference):
    """Current pixels -> fixed-orientation pixels, using camera pose only."""
    return intrinsics @ rotation_reference.T @ rotation_current @ np.linalg.inv(intrinsics)


def transform_image_points(points, homography):
    points = np.asarray(points, dtype=float)
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    transformed = homogeneous @ homography.T
    return transformed[..., :2] / transformed[..., 2:3]


def transform_image_box(box, homography):
    x1, y1, x2, y2 = box
    corners = transform_image_points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], homography)
    return tuple(np.r_[corners.min(axis=0), corners.max(axis=0)])


class MujocoShootingAdapter:
    def __init__(self, model, control, data, *, settings=None):
        self.model = model
        self.cfg = source_settings() if settings is None else dict(settings)
        self.root = model.body('TORSO').id
        self.muzzle_body = model.body(self.cfg['muzzle_body_name_expr']).id
        self.joints = np.array([model.joint(n).id for n in ('yaw_joint', 'pitch_joint')])
        self.qadr = model.jnt_qposadr[self.joints]
        self.vadr = model.jnt_dofadr[self.joints]
        self.aids = np.array([model.actuator(n + '_motor').id for n in ('yaw_joint', 'pitch_joint')])
        self.limits = model.jnt_range[self.joints].copy()
        self.target = data.qpos[self.qadr].copy()
        self.last_action = np.zeros(3, dtype=np.float32)
        self.fire_intent = 0.
        self.cooldown = self.high_steps = 0
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        # PhysX evaluates drive position and velocity at the end of the step.
        # implicitfast is implicit in velocity only. Substitute q+ = q + h*v+
        # into kp*(target-q+) - kd*v+: the velocity coefficient is kd + h*kp.
        # This retains the source kp/kd while accounting for implicit position
        # integration for the linear drive. Contact/force-limit solvers can differ.
        for i, name in enumerate(('yaw_joint', 'pitch_joint')):
            aid, settings = self.aids[i], control['joints'][name]
            model.actuator_gainprm[aid, 0] = settings['kp']
            model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
            position_term = (0. if model.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE
                             else model.opt.timestep*settings['kp'])
            model.actuator_biasprm[aid, :3] = [0., -settings['kp'], -(settings['kd']+position_term)]
            model.actuator_ctrllimited[aid] = False
            model.actuator_forcelimited[aid] = True
            model.actuator_forcerange[aid] = [-settings['effort_limit'], settings['effort_limit']]

    def point_velocity(self, data, point, body):
        mujoco.mj_jac(self.model, data, self.jacp, self.jacr, np.asarray(point), body)
        return self.jacp @ data.qvel, self.jacr @ data.qvel

    def muzzle(self, data):
        rotation = data.xmat[self.muzzle_body].reshape(3, 3)
        position = data.xpos[self.muzzle_body] + rotation @ self.cfg['muzzle_offset_b']
        forward = rotation @ self.cfg['muzzle_forward_axis']
        forward /= np.linalg.norm(forward)
        velocity, _ = self.point_velocity(data, position, self.muzzle_body)
        return position.copy(), forward, velocity

    def ego_motion(self, data):
        rotation = data.xmat[self.root].reshape(3, 3)
        yaw = yaw_rotation(rotation)
        linear, angular = self.point_velocity(data, data.xpos[self.root], self.root)
        forward = rotation[:, 0]
        forward_dot = np.cross(angular, forward)
        yaw_rate = (forward[0]*forward_dot[1] - forward[1]*forward_dot[0]) / max(float(forward[:2] @ forward[:2]), 1e-8)
        angular_yaw = yaw.T @ angular
        angular_yaw[2] = yaw_rate
        return yaw.T @ linear, angular_yaw, yaw_rate

    def robot_state(self, data):
        rotation = data.xmat[self.root].reshape(3, 3)
        linear, angular = self.point_velocity(data, data.xpos[self.root], self.root)
        _, forward, _ = self.muzzle(data)
        forward = yaw_rotation(rotation).T @ forward
        return RobotState(
            turret_yaw_rad=float(data.qpos[self.qadr[0]]), turret_pitch_rad=float(data.qpos[self.qadr[1]]),
            turret_yaw_velocity_rps=float(data.qvel[self.vadr[0]]), turret_pitch_velocity_rps=float(data.qvel[self.vadr[1]]),
            base_linear_velocity_base_mps=rotation.T @ linear,
            base_angular_velocity_base_rps=rotation.T @ angular,
            projected_gravity_base=-rotation[2, :], turret_joint_limits_rad=self.limits,
            muzzle_yaw_base_rad=float(np.arctan2(forward[1], forward[0])),
            muzzle_pitch_base_rad=float(np.arctan2(-forward[2], np.linalg.norm(forward[:2]))),
            prev_action=self.last_action.copy(), fire_intent=self.fire_intent,
            fire_cooldown_steps=self.cooldown)

    def command(self, action, *, ready, warmup=False, hold_when_unready=True):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('Expected three finite shooting actions.')
        # Same hold/no-fire protocol as the last full-chain Isaac evaluation.
        action = np.clip(action, -1., 1.) if ready or warmup or not hold_when_unready else np.array([0., 0., -1.], dtype=np.float32)
        if not warmup and (ready or not hold_when_unready):
            self.target += action[:2] * [self.cfg['max_delta_yaw'], self.cfg['max_delta_pitch']]
        self.target = np.clip(self.target, self.limits[:, 0], self.limits[:, 1])
        self.fire_intent = float((action[2] + 1.) * .5)
        high = ready and not warmup and self.fire_intent > self.cfg['fire_intent_threshold']
        self.high_steps = min(self.high_steps+1, self.cfg['fire_intent_hold_steps']) if high else 0
        fired = bool(ready and not warmup and self.cooldown <= 0 and self.high_steps >= self.cfg['fire_intent_hold_steps'])
        self.cooldown = max(self.cooldown-1, 0)
        if fired:
            self.cooldown = self.cfg['fire_cooldown_steps']
        self.last_action = action.copy()
        return fired

    def apply(self, data):
        # Call after the leg controller: only this adapter owns these motors.
        data.ctrl[self.aids] = self.target


@dataclass(frozen=True)
class BallisticResult:
    minimum_distance_m: float
    hit: bool
    closest_time_s: float
    bullet_path: np.ndarray
    target_path: np.ndarray


def ballistic_metrics(muzzle_position, muzzle_forward, muzzle_velocity,
                      target_position, target_velocity, target_acceleration=(0., 0., 0.), *, config=None):
    """Port the original synchronous trajectory metric, including optional RK2 drag."""
    cfg = source_settings() if config is None else config
    times = np.linspace(0., cfg['ballistic_max_time'], cfg['ballistic_num_samples'])
    p0, v0 = np.asarray(muzzle_position), np.asarray(muzzle_velocity) + np.asarray(muzzle_forward)*cfg['muzzle_speed']
    tp, tv, ta = (np.asarray(x, dtype=float) for x in (target_position, target_velocity, target_acceleration))
    gravity = np.array([0., 0., -cfg['gravity_mps2']])
    target = tp + times[:, None]*tv + .5*times[:, None]**2*ta
    if not cfg['enable_air_drag']:
        bullet = p0 + times[:, None]*v0 + .5*times[:, None]**2*gravity
    else:
        def acceleration(velocity):
            speed = max(float(np.linalg.norm(velocity)), 1e-6)
            cd = (np.interp(speed/cfg['speed_of_sound_mps'], cfg['mach_cd_mach'], cfg['mach_cd_cd'])
                  if cfg['use_mach_dependent_cd'] else cfg['drag_coefficient'])
            area = np.pi * (cfg['projectile_diameter_m']*.5)**2
            return gravity - .5*cfg['air_density']*cd*area/cfg['projectile_mass_kg']*speed*velocity
        history = [p0.copy()]
        position, velocity = p0.copy(), v0.copy()
        for dt in np.diff(times):
            midpoint_velocity = velocity + .5*acceleration(velocity)*dt
            position = position + midpoint_velocity*dt
            velocity = velocity + acceleration(midpoint_velocity)*dt
            history.append(position.copy())
        bullet = np.asarray(history)
    relative = bullet - target
    delta = np.diff(relative, axis=0)
    alpha = np.clip(-np.sum(relative[:-1]*delta, axis=1)/np.maximum(np.sum(delta*delta, axis=1), 1e-8), 0., 1.)
    distances = np.linalg.norm(relative[:-1] + alpha[:, None]*delta, axis=1)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    return BallisticResult(distance, distance < cfg['hit_radius'], float(times[index]+alpha[index]*(times[index+1]-times[index])), bullet, target)

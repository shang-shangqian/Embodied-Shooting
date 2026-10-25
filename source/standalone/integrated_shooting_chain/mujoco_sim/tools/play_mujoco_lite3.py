"""Play the user's exported TorchScript locomotion policy on the USD-derived MJCF.

Run with isaac_lab_232 Python. If necessary, the existing sibling mujoco_test
site-packages are appended for MuJoCo; neither environment is modified.
Video is rendered from the saved trajectory, preserving the measured rollout.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

try:
    import mujoco
except ModuleNotFoundError as error:
    if error.name != 'mujoco':
        raise
    package_dirs = [Path(sys.prefix).parent / name / 'lib'
                    / f'python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
                    for name in ('mujoco_sim', 'mujoco_test')]
    package_dir = next((p for p in package_dirs if (p/'mujoco').is_dir()), None)
    if package_dir is None:
        raise RuntimeError('MuJoCo is unavailable; run with the configured Isaac/MuJoCo Python environments.') from error
    sys.path.append(str(package_dir))
    import mujoco

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
DEFAULT_POLICY = WORKSPACE / 'logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt'
DEFAULT_MODEL = ROOT / 'mjcf/lite3_with_turret/scene.xml'
TRANSITIONS = [
    (0., 2., 'stand', (0., 0., 0.)),
    (2., 8., 'forward', (.5, 0., 0.)),
    (8., 12., 'stop', (0., 0., 0.)),
    (12., 16., 'turn_left', (0., 0., .4)),
    (16., 20., 'turn_right', (0., 0., -.4)),
    (20., 24., 'sideways', (0., .3, 0.)),
    (24., 28., 'backward', (-.3, 0., 0.)),
    (28., float('inf'), 'final_stop', (0., 0., 0.)),
]


def resolve_model_path(path):
    """Locate models recorded before the MuJoCo directory reorganization."""
    path = Path(path)
    if path.is_file():
        return path
    try:
        relative = path.resolve().relative_to((ROOT.parent / 'mjcf').resolve())
    except ValueError:
        return path
    migrated = ROOT / 'mjcf' / relative
    return migrated if migrated.is_file() else path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_for_time(t, args, rotation):
    if args.scenario == 'transitions':
        for start, end, phase, command in TRANSITIONS:
            if start <= t < end:
                return np.array(command, dtype=np.float32), phase
    command = np.zeros(3, dtype=np.float32) if t < args.settle_seconds else np.array(args.command, dtype=np.float32)
    phase = 'warmup' if t < args.settle_seconds else 'constant'
    if args.heading_target is not None:
        # Same outer command law as the training configuration: gain=.5, max=.8.
        heading = np.arctan2(rotation[1, 0], rotation[0, 0])
        error = np.arctan2(np.sin(args.heading_target-heading), np.cos(args.heading_target-heading))
        command[2] = np.clip(.5 * error, -.8, .8)
        phase = 'heading_hold'
    return command, phase


def phase_summary(rows, dt):
    groups = []
    for row in rows:
        if not groups or groups[-1][0]['phase'] != row['phase']:
            groups.append([])
        groups[-1].append(row)
    result = []
    for group in groups:
        start, end = max(0., group[0]['time_s']-dt), group[-1]['time_s']
        steady = [r for r in group if r['time_s'] >= start + min(1., (end-start)/4)]
        result.append({'phase': group[0]['phase'], 'start_s': start, 'end_s': end,
                       'mean_command': [float(np.mean([r[k] for r in steady])) for k in ('command_vx','command_vy','command_wz')],
                       'mean_velocity': [float(np.mean([r[k] for r in steady])) for k in ('vx_body','vy_body','wz_body')],
                       'minimum_height_m': min(r['z'] for r in group),
                       'maximum_tilt_deg': max(r['tilt_deg'] for r in group)})
    return result


def write_subtitles(path, segments, heading_target=None):
    def stamp(seconds):
        milliseconds = int(round(seconds * 1000))
        hours, milliseconds = divmod(milliseconds, 3600000)
        minutes, milliseconds = divmod(milliseconds, 60000)
        seconds, milliseconds = divmod(milliseconds, 1000)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'
    pieces = []
    for i, segment in enumerate(segments, 1):
        x, y, z = segment['mean_command']
        label = segment['phase'].upper().replace('_', ' ')
        command_label = (f'cmd vx {x:+.2f} m/s, vy {y:+.2f} m/s, wz {z:+.2f} rad/s'
                         if heading_target is None else f'cmd vx {x:+.2f} m/s, target yaw {np.rad2deg(heading_target):+.1f} deg')
        pieces.append(f'{i}\n{stamp(segment["start_s"])} --> {stamp(segment["end_s"])}\n'
                      f'{label} | {command_label}\n')
    path.write_text('\n'.join(pieces))


class LocomotionBridge:
    """Named-joint 45-D observation and delayed target-position PD adapter."""

    observation_slices = {'base_ang_vel': (0, 3), 'projected_gravity': (3, 6),
                          'velocity_commands': (6, 9), 'joint_pos_rel': (9, 21),
                          'joint_vel_rel': (21, 33), 'last_action': (33, 45)}

    def __init__(self, model, control, hip_delay=0, knee_delay=0):
        self.model, self.control = model, control
        self.policy_names = control['policy_joint_order']
        self.motor_names = control['actuator_order']
        if len(self.policy_names) != 12 or self.motor_names[:12] != self.policy_names:
            raise ValueError('Expected twelve policy joints followed by turret motors.')
        joint_ids = [model.joint(name).id for name in self.motor_names]
        self.qadr = model.jnt_qposadr[joint_ids]
        self.vadr = model.jnt_dofadr[joint_ids]
        self.aids = np.array([model.actuator(name + '_motor').id for name in self.motor_names])
        self.root_dof = int(model.joint('floating_base').dofadr[0])
        self.body_id = model.body('TORSO').id
        settings = [control['joints'][name] for name in self.motor_names]
        self.default = np.array([v['default_position'] for v in settings], dtype=np.float32)
        self.scale = np.array([v['action_scale'] for v in settings], dtype=np.float32)
        self.kp = np.array([v['kp'] for v in settings])
        self.kd = np.array([v['kd'] for v in settings])
        self.limits = np.array([v['effort_limit'] for v in settings])
        allowed = control['leg_delay_range_physics_steps']
        if not all(isinstance(v, int) and allowed[0] <= v <= allowed[1] for v in (hip_delay, knee_delay)):
            raise ValueError(f'Leg delays must be integers in {allowed}.')
        self.delays = np.array([knee_delay if 'Knee' in n else hip_delay if n in self.policy_names else 0
                                for n in self.motor_names])
        self.history = deque(maxlen=int(max(self.delays)) + 1)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.target = self.default.copy()

    def observe(self, data, command):
        rotation = data.xmat[self.body_id].reshape(3, 3)
        # Free-joint rotation velocity uses the body frame, not the rotated
        # principal-inertia frame returned by mj_objectVelocity(BODY, local=1).
        angular_local = data.qvel[self.root_dof+3:self.root_dof+6]
        gravity_local = -rotation[2, :]
        # Original observation terms clip before scaling; corruption is disabled in play.
        clip = lambda value: np.clip(value, -100., 100.)
        result = np.concatenate([
            clip(angular_local) * self.control['angular_velocity_observation_scale'],
            clip(gravity_local), clip(command),
            clip(data.qpos[self.qadr[:12]] - self.default[:12]),
            clip(data.qvel[self.vadr[:12]]) * self.control['joint_velocity_observation_scale'],
            clip(self.previous_action),
        ]).astype(np.float32)
        if result.shape != (45,) or not np.isfinite(result).all():
            raise RuntimeError('Invalid 45-D policy observation.')
        return result

    def set_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError('Invalid twelve-joint policy action.')
        self.previous_action[:] = np.clip(action, -self.control['clip_actions'], self.control['clip_actions'])
        self.target[:12] = self.default[:12] + self.scale[:12] * self.previous_action

    def delayed_target(self):
        # Called once per PHYSICS step, as in DelayedPDActuator.compute.
        # During startup, the oldest available command is the first command.
        self.history.appendleft(self.target.copy())
        return np.array([self.history[min(delay, len(self.history)-1)][i]
                         for i, delay in enumerate(self.delays)])

    def apply(self, data):
        target = self.delayed_target()
        raw = self.kp * (target - data.qpos[self.qadr]) - self.kd * data.qvel[self.vadr]
        applied = np.clip(raw, -self.limits, self.limits)
        data.ctrl[self.aids] = applied
        return raw, applied


def render_trajectory(model_path, trace_path, video_path, *, fps=50, width=1280, height=720):
    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(model_path)))
    data = mujoco.MjData(model)
    trace = np.load(trace_path)
    duration = float(trace['physics_time'][-1])
    frame_times = np.arange(0, duration, 1/fps)
    indices = np.minimum(np.rint(frame_times/model.opt.timestep).astype(int), len(trace['physics_time'])-1)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.azimuth, camera.elevation = 1.65, 135, -20
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    # Replay view only: retain precision on the nearby robot; no dynamics are stepped.
    model.vis.map.zfar = 100. / model.stat.extent
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an']
    subtitles = video_path.with_suffix('.srt').resolve()
    if subtitles.is_file():
        command += ['-vf', f"subtitles=filename='{subtitles}':force_style='Alignment=7,Fontsize=18,Outline=1,MarginL=12,MarginV=12'"]
    command += ['-c:v', 'libx264',
               '-preset', 'fast', '-threads', '4', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_path)]
    log_path = video_path.with_suffix('.ffmpeg.log')
    with log_path.open('w') as log, mujoco.Renderer(model, height=height, width=width) as renderer:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame_number, index in enumerate(indices):
                data.qpos[:] = trace['physics_qpos'][index]
                data.qvel[:] = trace['physics_qvel'][index]
                data.time = float(trace['physics_time'][index])
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.body('TORSO').xpos + [0, 0, .02]
                renderer.update_scene(data, camera=camera, scene_option=option)
                process.stdin.write(renderer.render().tobytes())
                if frame_number % 200 == 0:
                    print(f'[VIDEO] {frame_number}/{len(indices)} frames', flush=True)
        finally:
            process.stdin.close()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f'ffmpeg failed; see {log_path}')
    return {'path': str(video_path), 'frames': len(indices), 'fps': fps,
            'resolution': [width, height], 'duration_seconds': len(indices)/fps,
            'source': 'Replay of saved measured physics states; no resimulation or resets.'}


def rollout(args):
    model_path, policy_path = args.model.resolve(), args.policy.resolve()
    control = json.loads((model_path.parent / 'control.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if args.contact_time_constant is not None:
        if args.contact_time_constant < 2 * model.opt.timestep:
            raise ValueError('Contact time constant must be at least two physics steps.')
        contact_geoms = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_solref[contact_geoms, 0] = args.contact_time_constant
    if not np.isclose(model.opt.timestep, control['physics_dt']):
        raise ValueError('Physics timestep differs from the control configuration.')
    if not np.isclose(control['physics_dt'] * control['control_decimation'], control['control_dt']):
        raise ValueError('Inconsistent policy period.')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    mujoco.mj_forward(model, data)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    policy = torch.jit.load(str(policy_path), map_location='cpu').eval()
    rng = np.random.default_rng(args.seed)
    hip_delay, knee_delay = (map(int, rng.integers(0, 6, size=2)) if args.source_delay
                              else (args.hip_delay, args.knee_delay))
    bridge = LocomotionBridge(model, control, hip_delay, knee_delay)
    args.output.mkdir(parents=True, exist_ok=True)
    physics_times, physics_qpos, physics_qvel = [0.], [data.qpos.copy()], [data.qvel.copy()]
    observations, actions, torques, rows = [], [], [], []
    maximum_joint_speed = np.zeros(14)
    saturated_count = torque_count = 0
    first_fall_time = None
    command_requested = np.array(args.command, dtype=np.float32)
    started = time.monotonic()
    print(f'[PLAY] policy={policy_path}', flush=True)
    print(f'[PLAY] command={command_requested.tolist()}, physics_dt={model.opt.timestep}, '
          f'policy_dt={control["control_dt"]}, delays hip/knee={hip_delay}/{knee_delay}', flush=True)
    control_steps = int(round(args.duration / control['control_dt']))
    with torch.inference_mode():
        for step in range(control_steps):
            command, phase = command_for_time(step * control['control_dt'], args,
                                           data.xmat[bridge.body_id].reshape(3, 3))
            observation = bridge.observe(data, command)
            action = policy(torch.from_numpy(observation).unsqueeze(0)).cpu().numpy()[0]
            bridge.set_action(action)
            observations.append(observation)
            actions.append(bridge.previous_action.copy())
            for _ in range(control['control_decimation']):
                raw, applied = bridge.apply(data)
                saturated_count += int(np.count_nonzero(np.abs(raw) > bridge.limits))
                torque_count += len(applied)
                torques.append(applied.copy())
                mujoco.mj_step(model, data)
                if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                        or any(w.number for w in data.warning)):
                    raise RuntimeError('Invalid physics state or MuJoCo warning; refusing a reset-contaminated rollout.')
                mujoco.mj_forward(model, data)
                maximum_joint_speed = np.maximum(maximum_joint_speed, np.abs(data.qvel[bridge.vadr]))
                physics_times.append(float(data.time))
                physics_qpos.append(data.qpos.copy())
                physics_qvel.append(data.qvel.copy())
            rotation = data.xmat[bridge.body_id].reshape(3, 3)
            velocity = rotation.T @ data.qvel[bridge.root_dof:bridge.root_dof+3]
            tilt = float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1., 1.))))
            position = data.body('TORSO').xpos
            if first_fall_time is None and (position[2] < .12 or tilt > 60.):
                first_fall_time = float(data.time)
            rows.append({'time_s': float(data.time), 'x': float(position[0]), 'y': float(position[1]),
                         'z': float(position[2]), 'vx_body': float(velocity[0]), 'vy_body': float(velocity[1]),
                         'wz_body': float(data.qvel[bridge.root_dof+5]), 'tilt_deg': tilt,
                         'command_vx': float(command[0]), 'command_vy': float(command[1]),
                         'command_wz': float(command[2]), 'contact_count': int(data.ncon), 'phase': phase})
            if step % 100 == 0:
                print(f'[PLAY] t={data.time:.2f}s vx={velocity[0]:.3f} z={position[2]:.3f} '
                      f'tilt={tilt:.1f} fall={first_fall_time}', flush=True)
    trace_path = args.output / 'trajectory.npz'
    np.savez_compressed(trace_path, physics_time=physics_times, physics_qpos=physics_qpos,
                        physics_qvel=physics_qvel, observations=observations, actions=actions, torques=torques)
    with (args.output / 'motion.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    steady = [r for r in rows if r['time_s'] >= max(2., args.settle_seconds + 1.)] or rows
    limits = [control['joints'][n]['velocity_limit'] for n in bridge.motor_names]
    segments = phase_summary(rows, control['control_dt'])
    write_subtitles(args.output / 'walk.srt', segments, args.heading_target)
    report = {'policy': str(policy_path), 'policy_sha256': sha256(policy_path), 'model': str(model_path),
              'robot_xml_sha256': sha256(model_path.parent / 'robot.xml'),
              'source_usd_sha256': json.loads((model_path.parent / 'source_manifest.json').read_text())['source_sha256'],
              'mujoco_version': mujoco.__version__, 'torch_version': torch.__version__,
              'simulation_seconds': float(data.time), 'rollout_wall_seconds': time.monotonic() - started,
              'command': command_requested.tolist(), 'zero_command_settle_seconds': args.settle_seconds,
              'scenario': args.scenario, 'command_segments': segments,
              'heading_target_rad': args.heading_target,
              'contact_time_constant_override_s': args.contact_time_constant,
              'physics_dt': float(model.opt.timestep), 'policy_dt': control['control_dt'],
              'hip_delay_physics_steps': hip_delay, 'knee_delay_physics_steps': knee_delay,
              'delay_selection': 'source range sampled once per actuator group' if args.source_delay else 'fixed CLI values',
              'seed': args.seed, 'observation_slices': bridge.observation_slices,
              'policy_joint_order': bridge.policy_names, 'actuator_order': bridge.motor_names,
              'first_fall_time_s': first_fall_time, 'fall_definition': 'TORSO height < 0.12 m or tilt > 60 degrees',
              'resets': 0, 'minimum_height_m': min(r['z'] for r in rows),
              'maximum_tilt_deg': max(r['tilt_deg'] for r in rows),
              'final_position_m': [rows[-1][k] for k in ('x', 'y', 'z')],
              'mean_vx_body_after_warmup': float(np.mean([r['vx_body'] for r in steady])),
              'mean_vy_body_after_warmup': float(np.mean([r['vy_body'] for r in steady])),
              'mean_wz_body_after_warmup': float(np.mean([r['wz_body'] for r in steady])),
              'velocity_tracking_rmse_after_warmup': float(np.sqrt(np.mean([
                  (r['vx_body']-r['command_vx'])**2 + (r['vy_body']-r['command_vy'])**2 for r in steady]))),
              'torque_saturation_fraction': saturated_count / torque_count,
              'maximum_joint_speeds_rad_s': dict(zip(bridge.motor_names, maximum_joint_speed.tolist())),
              'configured_soft_velocity_limits_rad_s': dict(zip(bridge.motor_names, limits)),
              'notes': ['Actual user-specified TorchScript is evaluated on CPU; no policy weight conversion.',
                        'Deterministic initial state; no observation noise, pushes, or physics randomization.',
                        'Leg delayed target PD is explicit; turret hold approximates original implicit PhysX drives.',
                        'Velocity limits are reported, not imposed by qvel clipping or a torque-speed model.',
                        'No camera, LiDAR or perception inputs enter this 45-D locomotion policy.',
                        'Foot inertia inference and uncalibrated PhysX/MuJoCo contact differences remain.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('simulation_seconds', 'first_fall_time_s', 'minimum_height_m',
                     'maximum_tilt_deg', 'mean_vx_body_after_warmup', 'velocity_tracking_rmse_after_warmup')}, indent=2), flush=True)
    if args.video:
        report['video'] = render_trajectory(model_path, trace_path, args.output / 'walk.mp4', fps=args.fps)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('[PLAY] Saved video: ' + str((args.output / 'walk.mp4').resolve()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--duration', type=float, default=24.)
    parser.add_argument('--scenario', choices=('constant', 'transitions'), default='constant')
    parser.add_argument('--command', nargs=3, type=float, default=[.5, 0., 0.], metavar=('VX', 'VY', 'WZ'))
    parser.add_argument('--settle-seconds', type=float, default=0., help='Optional zero-command policy warmup, not a support fixture.')
    parser.add_argument('--hip-delay', type=int, default=0)
    parser.add_argument('--knee-delay', type=int, default=0)
    parser.add_argument('--source-delay', action='store_true', help='Sample each actuator group delay from 0..5 once at reset.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--heading-target', type=float, help='World heading target in radians; use original .5 gain and .8 rad/s limit.')
    parser.add_argument('--contact-time-constant', type=float, help='In-memory contact experiment only; original model files remain the baseline.')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('--render-only', type=Path, help='Render an existing rollout directory without running the policy again.')
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'outputs/mujoco_walk_validation_20261021/baseline_zero_delay')
    args = parser.parse_args()
    if (not np.isfinite([args.duration, args.settle_seconds, *args.command]).all()
            or args.duration < .02 or args.settle_seconds < 0 or args.fps < 1):
        parser.error('Invalid duration, settle time, command or fps.')
    if args.heading_target is not None and (not np.isfinite(args.heading_target) or args.scenario != 'constant'):
        parser.error('--heading-target requires a finite target and the constant scenario')
    if args.contact_time_constant is not None and not np.isfinite(args.contact_time_constant):
        parser.error('--contact-time-constant must be finite')
    if args.render_only:
        folder = args.render_only.resolve()
        report = json.loads((folder / 'report.json').read_text())
        report['video'] = render_trajectory(Path(report['model']), folder / 'trajectory.npz', folder / 'walk.mp4', fps=args.fps)
        (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    else:
        rollout(args)

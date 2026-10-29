"""Run RGB -> LK/OSTrack -> array LiDAR -> GRU -> recurrent shooting policy.

Run with isaac_lab_232 Python. CPU software rendering is supported. The target
is a prescribed mocap stimulus; its truth enters scoring/logs only. Bootstrap
uses a colour detector on the first RGB image, once per run, without GT boxes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

# Reuse the existing local MuJoCo environment bootstrap and locomotion bridge.
if __package__:
    from .play_mujoco_lite3 import LocomotionBridge, DEFAULT_POLICY, mujoco, np, torch
else:
    from play_mujoco_lite3 import LocomotionBridge, DEFAULT_POLICY, mujoco, np, torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
sys.path.insert(0, str(ROOT.parents[1]))
from integrated_shooting_chain.mujoco_sim.shooting_models import (
    RecurrentShootingPolicy, ensure_opencv, make_device_ostrack,
)
from integrated_shooting_chain.mujoco_sim.model_paths import resolve_model_path
cv2 = ensure_opencv()
from integrated_shooting_chain.mujoco_sim.shooting import (
    MujocoShootingAdapter, ballistic_metrics, checkpoint_settings, yaw_rotation,
    camera_rotation_homography, transform_image_points, transform_image_box,
)
from integrated_shooting_chain.mujoco_sim.mujoco_camera import MujocoCameraSensor
from integrated_shooting_chain.mujoco_sim.mujoco_lidar import ArrayLidarConfig, MujocoArrayLidar
from integrated_shooting_chain.model_loaders import load_velocity_estimator
from integrated_shooting_chain.shooting_chain import build_policy_observation, tracker_quality_features, ShootingChainConfig
from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.multi_target_tracker import MultiTargetConfig, MultiTargetVisionTracker

ASSET = ROOT/'mjcf/lite3_with_turret'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_box(rgb):
    """Image-only detector for the red simulated target; never used after init."""
    red, green, blue = (rgb[:, :, i].astype(float) for i in range(3))
    mask = ((red > 60) & (red > 1.5*green+10) & (red > 1.5*blue+10)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if not candidates:
        return None
    i = max(candidates, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    x, y, w, h, area = stats[i]
    return (float(x), float(y), float(x+w), float(y+h))


class PerceptionPipeline:
    """Perception has no interface accepting target positions or velocities."""
    def __init__(self, model, camera, config, *, device, stabilize_camera=False):
        self.config = config
        self.stabilize_camera = stabilize_camera
        self.reference_rotation = None
        fine = make_device_ostrack(config['ostrack_repo'], config['ostrack_checkpoint'],
                                   config=config['ostrack_config'], device=device)
        k = camera.intrinsic_matrix
        intrinsics = CameraIntrinsics(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2],
                                      width=camera.width, height=camera.height)
        self.tracker = MultiTargetVisionTracker(intrinsics, config=MultiTargetConfig(
            fine_tracker_policy=config['fine_tracker_policy']), fine_tracker=fine)
        self.velocity = load_velocity_estimator(config['velocity_checkpoint'], backend=config['velocity_backend'], device=device)
        self.lidar = MujocoArrayLidar(model, ArrayLidarConfig.from_json(ASSET/'lidar_config.json'))
        self.direct_count = 0
        self.first_valid = None
        self.bootstrap_calls = 0
        self.last_scan = None

    def update(self, frame, data, adapter):
        image = frame.rgb
        homography = np.eye(3)
        if self.stabilize_camera:
            if self.reference_rotation is None:
                self.reference_rotation = frame.rotation_w_optical.copy()
            homography = camera_rotation_homography(frame.intrinsic_matrix, frame.rotation_w_optical, self.reference_rotation)
            image = cv2.warpPerspective(frame.rgb, homography, (frame.rgb.shape[1], frame.rgb.shape[0]))
        if not self.tracker.initialized:
            if self.bootstrap_calls:
                raise RuntimeError('Tracker lost initialization; GT/re-detection fallback is disabled.')
            box = bootstrap_box(image)
            if box is None:
                raise RuntimeError('No red target detected in first RGB frame.')
            self.bootstrap_calls += 1
            result = self.tracker.initialize(image, {0: box}, timestamp_s=frame.time)
        else:
            result = self.tracker.update(image, dt_s=.02, timestamp_s=frame.time)
        selected = result.selected_target
        if self.stabilize_camera and selected is not None:
            inverse = np.linalg.inv(homography)
            center = transform_image_points(selected.center_xy, inverse)
            xy = (center-frame.intrinsic_matrix[:2, 2])/np.diag(frame.intrinsic_matrix)[:2]
            selected = replace(selected, center_xy=tuple(center),
                bbox_xyxy=transform_image_box(selected.bbox_xyxy, inverse),
                yaw_rad=float(np.arctan2(xy[0], 1.)),
                pitch_rad=float(np.arctan2(-xy[1], np.sqrt(1.+xy[0]**2))))
        return self.estimate_selected(frame, data, adapter, selected)

    def estimate_selected(self, frame, data, adapter, selected):
        """Fuse a visual estimate with ray measurements; no target truth is accepted."""
        self.last_scan = self.lidar.scan(data, update_kinematics=False)
        position = np.full(3, np.nan)
        range_estimate = None
        if selected is not None and selected.status != 'lost':
            range_estimate = self.lidar.estimate_target(self.last_scan, yaw=selected.yaw_rad,
                pitch=selected.pitch_rad, camera_position_w=frame.position_w,
                camera_rotation_w=frame.rotation_w_optical, target_radius=self.config['target_radius_m'])
            if range_estimate.hit_valid:
                yaw, pitch = selected.yaw_rad, selected.pitch_rad
                ray = np.array([np.sin(yaw)*np.cos(pitch), -np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
                world = frame.position_w + frame.rotation_w_optical @ (ray*range_estimate.center_distance)
                rotation = data.xmat[adapter.root].reshape(3, 3)
                position = yaw_rotation(rotation).T @ (world-data.xpos[adapter.root])
        valid = bool(np.isfinite(position).all())
        linear, angular, rate = adapter.ego_motion(data)
        source = selected.source if selected is not None else 'lost'
        estimate = self.velocity.update(0, SimpleNamespace(
            timestamp_s=frame.time, position_base_m=position,
            robot_linear_velocity_base_mps=linear, robot_angular_velocity_base_rps=angular,
            yaw_rate_rps=rate, source=source, range_valid=valid,
            quality=tracker_quality_features(selected, self.tracker.config) if selected is not None else (0.,)*7))
        if valid and source in {'initial', 'flow', 'ostrack'}:
            self.direct_count += 1
            if self.first_valid is None:
                self.first_valid = frame.time
        velocity = (estimate.velocity_mean_mps.detach().cpu().numpy() if estimate is not None else np.zeros(3))
        ready = bool(valid and estimate is not None and np.isfinite(velocity).all()
                     and self.direct_count >= self.config['min_velocity_observations']
                     and frame.time-self.first_valid >= self.config['min_velocity_history_s'])
        return SimpleNamespace(ready=ready, position=position, velocity=velocity,
            selected=selected, range_estimate=range_estimate, source=source)


class VideoWriter:
    def __init__(self, path, fps):
        self.path, self.frames = path, 0
        self.log = path.with_suffix('.ffmpeg.log').open('w')
        self.process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', str(fps), '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-preset', 'fast', '-threads', '4', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE, stderr=self.log)

    def write(self, rgb):
        self.process.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        self.log.close()
        if code:
            raise RuntimeError(f'ffmpeg failed: {self.path.with_suffix(".ffmpeg.log")}')


def draw_frame(frame, perception, adapter, data, renderer, view, option, row, shots, last_shot):
    canvas = np.full((720, 1280, 3), 22, dtype=np.uint8)
    rgb = frame.rgb.copy()
    if perception.selected is not None:
        x1, y1, x2, y2 = np.rint(perception.selected.bbox_xyxy).astype(int)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.drawMarker(rgb, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
        # Actual RGB crop, enlarged for distant-target inspection.
        x1, x2 = max(0, cx-24), min(frame.rgb.shape[1], cx+24)
        y1, y2 = max(0, cy-24), min(frame.rgb.shape[0], cy+24)
        if x2 > x1 and y2 > y1:
            canvas[390:646, 1024:] = cv2.resize(rgb[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_NEAREST)
    muzzle, forward, _ = adapter.muzzle(data)
    aim = frame.rotation_w_optical.T @ (muzzle+forward*100.-frame.position_w)
    if aim[2] > 0:
        uv = aim[:2]/aim[2]*np.diag(frame.intrinsic_matrix)[:2]+frame.intrinsic_matrix[:2, 2]
        if np.isfinite(uv).all() and 0 <= uv[0] < 1024 and 0 <= uv[1] < 576:
            cv2.drawMarker(rgb, tuple(np.rint(uv).astype(int)), (255, 220, 0), cv2.MARKER_TILTED_CROSS, 17, 2)
    canvas[:576, :1024] = rgb
    view.lookat[:] = data.body('TORSO').xpos+[0., 0., .04]
    renderer.update_scene(data, camera=view, scene_option=option)
    canvas[:360, 1024:] = renderer.render()
    lines = [f't={frame.time:6.2f}s  mode={row["mode"]}  ready={perception.ready}  source={perception.source}',
             f'shots={len(shots)}  hits={sum(s["hit"] for s in shots)}  vx={row["robot_vx_mps"]:+.3f} m/s  range={row["range_m"]}',
             f'action={np.round(row["action"], 3)}  turret={np.round(row["turret_qpos_rad"], 3)}',
             'GREEN: tracked target | YELLOW: physical muzzle direction at 100 m',
             'RGB colour bootstrap once; LK/OSTrack + LiDAR + GRU afterwards']
    if last_shot is not None:
        lines[-1] = f'Last shot: {"HIT" if last_shot["hit"] else "MISS"} | closest distance {last_shot["minimum_distance_m"]:.3f} m'
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 598+24*i), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Robot view', (1030, 378), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1)
    return canvas


def run(args):
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    model_root = (WORKSPACE/(args.model_root or config.get('model_root', WORKSPACE))).resolve()
    config['model_root'] = str(model_root)
    def model_path(value):
        return resolve_model_path(value, model_root=model_root, project_root=WORKSPACE)
    if args.decision_checkpoint is not None:
        config['decision_checkpoint'] = str(args.decision_checkpoint)
        config['reference_report'] = None
        training_path = model_path(args.decision_checkpoint).parent.parent/'training_settings.json'
        if training_path.is_file():
            config['training_settings'] = str(training_path)
            training = json.loads(training_path.read_text())['arguments']
            if 'velocity_model' in training:
                config['velocity_checkpoint'] = training['velocity_model']
    config['locomotion_policy'] = str(args.locomotion_policy or config.get('locomotion_policy', DEFAULT_POLICY))
    for name in ('decision_checkpoint', 'velocity_checkpoint', 'ostrack_repo', 'ostrack_checkpoint', 'locomotion_policy'):
        config[name] = str(model_path(config[name]))
    args.output.mkdir(parents=True, exist_ok=False)
    settings = checkpoint_settings(config['decision_checkpoint'])
    print('[MODELS] decision='+config['decision_checkpoint'], flush=True)
    print('[MODELS] velocity='+config['velocity_checkpoint'], flush=True)
    print(f'[CONFIG] integrator={args.integrator} fire_threshold={settings["fire_intent_threshold"]} '
          f'hold_steps={settings["fire_intent_hold_steps"]} budget={settings["max_shots_per_episode"]}', flush=True)
    control = json.loads((ASSET/'control.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(ASSET/'scene_shooting.xml'))
    model.opt.integrator = (mujoco.mjtIntegrator.mjINT_DISCRETE if args.integrator == 'discrete'
                            else mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    adapter = MujocoShootingAdapter(model, control, data, settings=settings)
    legs = LocomotionBridge(model, control, hip_delay=0, knee_delay=4)
    locomotion = torch.jit.load(config['locomotion_policy'], map_location='cpu').eval()
    policy = RecurrentShootingPolicy(config['decision_checkpoint'], device=args.device)
    chain_config = ShootingChainConfig(target_distance_range_m=tuple(settings['target_radius_range']),
        target_speed_range_mps=tuple(settings['target_speed_range']))
    warmup = config['robot_warmup_s']
    target0 = np.array([args.distance, args.target_y, args.target_z])
    target_velocity = np.array(args.target_velocity)
    mocap = int(model.body('shooting_target').mocapid[0])
    data.mocap_pos[mocap] = target0
    mujoco.mj_forward(model, data)
    view = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(view)
    view.distance, view.azimuth, view.elevation = 1.45, 135, -15
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3] = 0
    option.geomgroup[4] = 1
    option.sitegroup[:] = 0
    physics_qpos, physics_qvel, observations, actions, rows, shots = [], [], [], [], [], []
    sources = Counter()
    first_ready, first_fall = None, None
    start = time.perf_counter()
    video = VideoWriter(args.output/'shooting.mp4', 25) if args.video else None
    try:
        with MujocoCameraSensor(model, ASSET/'camera_config.json') as camera:
            perception = PerceptionPipeline(model, camera, config, device=args.device, stabilize_camera=args.stabilize_camera)
            with mujoco.Renderer(model, height=360, width=256) as renderer:
                for step in range(round(args.duration/.02)):
                    t = float(data.time)
                    rotation = data.xmat[adapter.root].reshape(3, 3)
                    heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                    # Establish perception before the walking command, then ramp
                    # the command to avoid changing gait and bootstrapping together.
                    speed = args.walk_speed*np.clip((t-args.walk_start)/args.walk_ramp, 0., 1.)
                    command = np.array([speed, 0., np.clip(-.5*heading, -.8, .8)])
                    with torch.inference_mode():
                        leg_action = locomotion(torch.from_numpy(legs.observe(data, command))[None])[0].numpy()
                    legs.set_action(leg_action)
                    sensed = frame = None
                    obs = np.zeros(28, dtype=np.float32)
                    action = np.array([0., 0., -1.], dtype=np.float32)
                    ready = False
                    if t >= warmup-1e-8:
                        frame = camera.capture(data)
                        sensed = perception.update(frame, data, adapter)
                        ready = sensed.ready
                        if ready:
                            if first_ready is None:
                                first_ready = t
                            obs = build_policy_observation(sensed.position, sensed.velocity, adapter.robot_state(data), config=chain_config)
                            action = policy(obs)
                    in_warmup = t < warmup-1e-8
                    if in_warmup:
                        # The original salvo evaluator infers during standing
                        # warmup and preserves that RNN history at first target
                        # acquisition. Only the actuator/fire gates are disabled.
                        obs = build_policy_observation(np.zeros(3), np.zeros(3), adapter.robot_state(data),
                            prediction_ready=False, config=chain_config)
                        action = policy(obs)
                    fired = adapter.command(action, ready=ready, warmup=in_warmup)
                    robot_state = adapter.robot_state(data)
                    # Evaluation-only branch; these values never enter perception or actor.
                    truth_position = data.mocap_pos[mocap].copy()
                    truth_relative = yaw_rotation(rotation).T @ (truth_position-data.xpos[adapter.root])
                    ego_linear, _, _ = adapter.ego_motion(data)
                    truth_relative_velocity = yaw_rotation(rotation).T @ target_velocity - ego_linear
                    position_error = float(np.linalg.norm(sensed.position-truth_relative)) if sensed is not None and np.isfinite(sensed.position).all() else None
                    velocity_error = float(np.linalg.norm(sensed.velocity-truth_relative_velocity)) if ready else None
                    source = sensed.source if sensed is not None else 'warmup'
                    sources[source] += 1
                    row = {'time_s': t, 'mode': 'walking' if abs(speed) > 1e-8 else 'standing', 'ready': ready,
                        'command_vx_mps': float(speed), 'turret_position_target_rad': adapter.target.tolist(),
                        'source': source, 'range_valid': bool(sensed is not None and sensed.range_estimate is not None and sensed.range_estimate.hit_valid),
                        'range_m': float(sensed.range_estimate.center_distance) if sensed is not None and sensed.range_estimate is not None and sensed.range_estimate.hit_valid else None,
                        'bbox_xyxy': list(sensed.selected.bbox_xyxy) if sensed is not None and sensed.selected is not None else None,
                        'fine_action': sensed.selected.fine_action if sensed is not None and sensed.selected is not None else None,
                        'fine_attempted': bool(sensed is not None and sensed.selected is not None and sensed.selected.fine_attempted),
                        'fine_applied': bool(sensed is not None and sensed.selected is not None and sensed.selected.fine_applied),
                        'action': action.tolist(), 'turret_qpos_rad': data.qpos[adapter.qadr].tolist(),
                        'robot_vx_mps': float(robot_state.base_linear_velocity_base_mps[0]),
                        'robot_z_m': float(data.xpos[adapter.root, 2]),
                        'tilt_deg': float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1., 1.)))),
                        'position_error_m': position_error, 'velocity_error_mps': velocity_error,
                        'target_position_estimate': sensed.position.tolist() if sensed is not None and np.isfinite(sensed.position).all() else None,
                        'target_velocity_estimate': sensed.velocity.tolist() if ready else None,
                        'decision_fire': fired}
                    rows.append(row)
                    observations.append(obs.copy())
                    actions.append(action.copy())
                    if video is not None and frame is not None and step % 2 == 0:
                        video.write(draw_frame(frame, sensed, adapter, data, renderer, view, option, row, shots, shots[-1] if shots else None))
                        if video.frames == 1:
                            from preview_lite3_mjcf import save_rgb_png
                            save_rgb_png(args.output/'first_camera.png', frame.rgb)
                    for substep in range(control['control_decimation']):
                        legs.apply(data)
                        adapter.apply(data)
                        mujoco.mj_step(model, data)
                        data.mocap_pos[mocap] = target0+target_velocity*max(0., float(data.time)-warmup)
                        mujoco.mj_forward(model, data)
                        physics_qpos.append(data.qpos.copy())
                        physics_qvel.append(data.qvel.copy())
                    metric = None
                    if ready:
                        muzzle, forward, muzzle_velocity = adapter.muzzle(data)
                        metric = ballistic_metrics(muzzle, forward, muzzle_velocity, data.mocap_pos[mocap], target_velocity, config=settings)
                    row['ballistic_minimum_distance_m'] = metric.minimum_distance_m if metric is not None else None
                    row['ballistic_would_hit'] = metric.hit if metric is not None else None
                    row['fire_intent'] = adapter.fire_intent
                    row['fire_high_steps'] = adapter.high_steps
                    row['cooldown_steps'] = adapter.cooldown
                    row['aim_errors_rad'] = (obs[7:9]*chain_config.aim_error_obs_scale_rad).tolist() if ready else None
                    if fired:
                        shot = {'decision_time_s': t, 'time_s': float(data.time), 'hit': metric.hit,
                            'minimum_distance_m': metric.minimum_distance_m, 'closest_time_s': metric.closest_time_s,
                            'muzzle_position_w': muzzle.tolist(), 'muzzle_forward_w': forward.tolist(), 'muzzle_velocity_w': muzzle_velocity.tolist(),
                            'target_position_w': data.mocap_pos[mocap].tolist(), 'target_velocity_w': target_velocity.tolist()}
                        shots.append(shot)
                        print(f'[SHOT] t={data.time:.2f} hit={metric.hit} miss_distance={metric.minimum_distance_m:.3f}m', flush=True)
                    if first_fall is None and (row['robot_z_m'] < .12 or row['tilt_deg'] > 60.):
                        first_fall = t
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
                        raise RuntimeError('Nonfinite state or MuJoCo warning; aborting without reset.')
                    if step % 50 == 0:
                        print(f'[CHAIN] t={data.time:.2f}s ready={ready} source={source} shots={len(shots)} hits={sum(s["hit"] for s in shots)}', flush=True)
                    budget = settings.get('max_shots_per_episode', 0)
                    if budget > 0 and len(shots) >= budget:
                        break
                if frame is not None:
                    from preview_lite3_mjcf import save_rgb_png
                    save_rgb_png(args.output/'last_camera.png', frame.rgb)
                if perception.last_scan is not None:
                    perception.last_scan.save_npz(args.output/'last_lidar_scan.npz')
                np.savez_compressed(args.output/'trajectory.npz', physics_qpos=physics_qpos, physics_qvel=physics_qvel,
                    observations=observations, actions=actions)
                report = {'completed': True, 'duration_s': float(data.time), 'engagement_duration_s': max(0., float(data.time)-warmup),
                    'wall_seconds': time.perf_counter()-start, 'physics_dt_s': model.opt.timestep, 'control_dt_s': .02,
                    'inference_device': args.device, 'mujoco_version': mujoco.__version__, 'seed': args.seed,
                    'model': str(ASSET/'scene_shooting.xml'), 'config': config,
                    'hashes': {name: sha256(config[name]) for name in ('decision_checkpoint', 'velocity_checkpoint', 'ostrack_checkpoint')},
                    'robot_xml_sha256': sha256(ASSET/'robot.xml'), 'locomotion_policy_sha256': sha256(config['locomotion_policy']),
                    'ballistic_source_settings': {k: settings[k] for k in ('muzzle_speed', 'gravity_mps2', 'hit_radius', 'ballistic_max_time', 'ballistic_num_samples', 'enable_air_drag')},
                    'fire_settings': {k: settings[k] for k in ('fire_intent_threshold', 'fire_intent_hold_steps', 'fire_cooldown_steps', 'max_shots_per_episode')},
                    'integrator': args.integrator,
                    'turret_drive': ('native position/velocity affine drive, original USD gains and effort limits' if args.integrator == 'discrete' else 'affine drive with kd+h*kp for end-of-step position approximation'),
                    'actor_contract': '28 actor + 31 zero critic placeholders; saved RL-Games normalization, separate recurrent states, tanh(mu)',
                    'target_stimulus': {'position0_w': target0.tolist(), 'velocity_w': target_velocity.tolist(), 'motion_starts_s': warmup, 'retained_after_hit': True},
                    'walking_command': {'vx_mps': args.walk_speed, 'starts_s': args.walk_start, 'ramp_s': args.walk_ramp, 'heading_target_rad': 0.},
                    'bootstrap_protocol': 'image-only red connected-component box once; no GT box or re-detection',
                    'bootstrap_calls': perception.bootstrap_calls, 'ground_truth_feedback_to_policy': False,
                    'camera_rotation_compensation': args.stabilize_camera,
                    'ground_truth_usage': 'scene stimulus, scoring and diagnostic errors only',
                    'scoring': 'source synchronous predictive ballistics after physics step; not physical projectile contacts',
                    'first_ready_time_s': first_ready, 'actor_calls': policy.calls,
                    'actor_warmup_protocol': 'infer on robot state and zero target during standing warmup; preserve RNN at first acquisition; pause only while perception is unready after warmup',
                    'ready_frames': sum(r['ready'] for r in rows), 'source_counts': dict(sources),
                    'fine_attempts': sum(r['fine_attempted'] for r in rows), 'fine_applied': sum(r['fine_applied'] for r in rows),
                    'shots': shots, 'shot_count': len(shots), 'hit_count': sum(s['hit'] for s in shots),
                    'accuracy': sum(s['hit'] for s in shots)/len(shots) if shots else None,
                    'diagnostic_would_hit_frames': sum(r['ballistic_would_hit'] is True for r in rows),
                    'first_fall_time_s': first_fall, 'automatic_resets': 0,
                    'maximum_tilt_deg': max(r['tilt_deg'] for r in rows), 'minimum_robot_z_m': min(r['robot_z_m'] for r in rows),
                    'mean_robot_vx_after_warmup_mps': float(np.mean([r['robot_vx_mps'] for r in rows if r['time_s'] >= warmup])),
                    'mean_position_error_m': float(np.mean([r['position_error_m'] for r in rows if r['position_error_m'] is not None])) if any(r['position_error_m'] is not None for r in rows) else None,
                    'mean_velocity_error_mps': float(np.mean([r['velocity_error_mps'] for r in rows if r['velocity_error_mps'] is not None])) if any(r['velocity_error_mps'] is not None for r in rows) else None,
                    'video': {'path': str(video.path), 'fps': 25, 'frames': video.frames} if video else None}
                (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                (args.output/'steps.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
    finally:
        if video is not None:
            video.close()
    print('[DONE] '+str(args.output/'report.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'shooting_config.json')
    parser.add_argument('--decision-checkpoint', type=Path, help='Override the shooting checkpoint; relative paths use model-root.')
    parser.add_argument('--model-root', type=Path, help='Override the model asset root from the config')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing results are never overwritten.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--integrator', choices=('discrete', 'implicitfast'), default='discrete',
                        help='Discrete integrates position feedback implicitly; implicitfast uses kd+h*kp approximation.')
    parser.add_argument('--seed', type=int, default=61)
    parser.add_argument('--duration', type=float, default=12.)
    parser.add_argument('--distance', type=float, default=100.)
    parser.add_argument('--target-y', type=float, default=-2.)
    parser.add_argument('--target-z', type=float, default=2.)
    parser.add_argument('--target-velocity', type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument('--walk-speed', type=float, default=0.)
    parser.add_argument('--walk-start', type=float, default=4.)
    parser.add_argument('--walk-ramp', type=float, default=1.)
    parser.add_argument('--locomotion-policy', type=Path, help='Override locomotion weights; relative to model-root')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--stabilize-camera', action='store_true', help='Rotate RGB to the first camera orientation using proprioception only, then map tracked bearings back.')
    args = parser.parse_args()
    if not np.isfinite([args.duration, args.distance, args.target_y, args.target_z, args.walk_speed, args.walk_start, args.walk_ramp, *args.target_velocity]).all():
        parser.error('All scenario values must be finite.')
    if args.duration <= 2.2 or not 80 <= args.distance <= 150 or args.threads < 1:
        parser.error('Require duration > 2.2s, distance within trained 80..150m and threads >= 1.')
    if args.walk_start < 2. or args.walk_ramp <= 0.:
        parser.error('Walking must start after warmup, with a positive ramp duration.')
    run(args)


if __name__ == '__main__':
    main()

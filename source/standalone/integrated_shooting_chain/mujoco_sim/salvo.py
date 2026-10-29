"""Standing robot / static target clearance, sharing Isaac's engagement rules."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from .tools.play_mujoco_shooting import (
    ASSET, ROOT, WORKSPACE, DEFAULT_POLICY, LocomotionBridge, PerceptionPipeline,
    RecurrentShootingPolicy, VideoWriter, cv2, mujoco, np, torch, sha256,
    MujocoCameraSensor, MujocoShootingAdapter, checkpoint_settings,
    ShootingChainConfig, build_policy_observation, ballistic_metrics,
    camera_rotation_homography, transform_image_points, transform_image_box,
)
from .salvo_scene import build_salvo_model, sample_layout, MujocoStaticTargetPool
from .model_paths import resolve_model_path
from ..salvo_engagement import SalvoEngagementConfig, SalvoEngagementScheduler
from ..salvo_ablation import load_layout, save_layout


def project_path(value):
    """Resolve relative inputs and outputs against this checkout's root."""
    return (WORKSPACE / Path(value).expanduser()).resolve()


class SalvoActorGate:
    """Isaac group D: wait for perception; reset only Actor on identity change."""
    def __init__(self, policy):
        self.policy, self.last_id = policy, None
        self.resets = self.wait_steps = 0

    def action(self, target_id, ready, observation):
        if target_id is not None:
            if self.last_id is not None and target_id != self.last_id:
                self.policy.reset_actor()
                self.resets += 1
            self.last_id = target_id
        if not ready or target_id is None:
            self.wait_steps += 1
            return np.array([0., 0., -1.], dtype=np.float32)
        return self.policy(observation)


class SalvoPerceptionPipeline(PerceptionPipeline):
    """One pool bootstrap; continuously track all surviving IDs using RGB only."""
    def __init__(self, *args, engagement_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker.config = replace(self.tracker.config,
            selection_policy='nearest_until_removed', max_initial_scale_ratio=2.)
        self.scheduler = SalvoEngagementScheduler(engagement_config)
        self.removed = set()
        self.selected_id = self.previous_time = None
        self.bootstrap_targets = 0
        self.latest_result = None

    def track(self, frame, *, initial_boxes=None, removed_ids=()):
        image, homography = frame.rgb, np.eye(3)
        if self.stabilize_camera:
            if self.reference_rotation is None:
                self.reference_rotation = frame.rotation_w_optical.copy()
            homography = camera_rotation_homography(frame.intrinsic_matrix,
                frame.rotation_w_optical, self.reference_rotation)
            image = cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]))
        if self.bootstrap_calls == 0:
            self.bootstrap_calls += 1
            if not initial_boxes:
                raise ValueError('The first frame requires the complete pool of initial boxes')
            boxes = {i: transform_image_box(b, homography) for i, b in initial_boxes.items()}
            self.bootstrap_targets = len(boxes)
            result = self.tracker.initialize(image, boxes, timestamp_s=frame.time)
            chosen = self.scheduler.choose(frame.time, self.tracker.selectable_target_distances())
            self.tracker.set_selection_candidates([] if chosen is None else [chosen])
        else:
            if initial_boxes is not None:
                raise RuntimeError('GT rebootstrap is forbidden')
            removed = set(removed_ids)
            self.tracker.remove_targets(removed-self.removed)
            self.removed = removed
            chosen = self.scheduler.choose(frame.time, self.tracker.selectable_target_distances(),
                                            removed_ids=removed)
            self.tracker.set_selection_candidates([] if chosen is None else [chosen])
            dt = frame.time-self.previous_time
            if dt <= 0:
                raise ValueError('Tracking requires increasing frame timestamps')
            result = self.tracker.update(image, dt_s=dt, timestamp_s=frame.time)
        self.previous_time, self.latest_result = frame.time, result
        selected = next((t for t in result.targets if t.target_id == chosen and t.status != 'lost'), None)
        selected_id = None if selected is None else int(selected.target_id)
        if selected_id != self.selected_id:
            self.velocity.reset(0)
            self.direct_count, self.first_valid = 0, None
            self.selected_id = selected_id
        if self.stabilize_camera and selected is not None:
            inverse = np.linalg.inv(homography)
            center = transform_image_points(selected.center_xy, inverse)
            xy = (center-frame.intrinsic_matrix[:2, 2])/np.diag(frame.intrinsic_matrix)[:2]
            selected = replace(selected, center_xy=tuple(center),
                bbox_xyxy=transform_image_box(selected.bbox_xyxy, inverse),
                yaw_rad=float(np.arctan2(xy[0], 1.)),
                pitch_rad=float(np.arctan2(-xy[1], np.hypot(1., xy[0]))))
        return selected

    def terminal_reason(self):
        return self.scheduler.terminal_reason(self.tracker.selectable_target_distances())


def prepare_config(args):
    config = json.loads(project_path(args.config).read_text())
    root = project_path(getattr(args, 'model_root', None) or config.get('model_root', WORKSPACE))
    config['model_root'] = str(root)
    def model_path(value):
        return resolve_model_path(value, model_root=root, project_root=WORKSPACE)
    if args.decision_checkpoint is not None:
        config['decision_checkpoint'] = str(args.decision_checkpoint)
        config['reference_report'] = None
    config['decision_checkpoint'] = str(model_path(config['decision_checkpoint']))
    paired = Path(config['decision_checkpoint']).parent.parent/'training_settings.json'
    if paired.is_file():
        training = json.loads(paired.read_text())['arguments']
        if 'velocity_model' in training:
            config['velocity_checkpoint'] = training['velocity_model']
        config['training_settings'] = str(paired)
    config['locomotion_policy'] = str(args.locomotion_policy or config.get('locomotion_policy', DEFAULT_POLICY))
    for key in ('velocity_checkpoint', 'ostrack_repo', 'ostrack_checkpoint', 'locomotion_policy'):
        config[key] = str(model_path(config[key]))
    if not args.dry_run:
        for key in ('decision_checkpoint', 'velocity_checkpoint', 'ostrack_checkpoint', 'locomotion_policy'):
            if not Path(config[key]).is_file():
                raise FileNotFoundError(f'{key}: {config[key]}')
        if not Path(config['ostrack_repo']).is_dir():
            raise FileNotFoundError(config['ostrack_repo'])
    return config


def place_layout(args, pool, camera):
    if args.layout_in:
        positions, _ = load_layout(project_path(args.layout_in), args.salvo_targets, pool.radius)
        positions = positions.numpy()
    else:
        positions = sample_layout(*camera.pose(pool.data), args.salvo_targets,
                                  seed=args.seed, radius=pool.radius)
    pool.place(positions)
    return save_layout(args.output/'layout.json', torch.tensor(positions, dtype=torch.float32), pool.radius)


def make_report(pool, shots, *, reason, dry_run=False, **details):
    hits = sum(bool(s['hit']) for s in shots)
    destroyed = len(pool.removed_ids)
    return dict(platform='mujoco', scenario='ss', mode='dry_run' if dry_run else 'evaluation',
        completed=not dry_run and reason != 'simulator_error', cleared=not dry_run and destroyed == len(pool.alive),
        termination_reason=reason, targets_per_episode=len(pool.alive), destroyed=destroyed,
        remaining=int(pool.alive.sum()), destroyed_target_ids=sorted(pool.removed_ids),
        shot_count=len(shots), hit_count=hits, accuracy=hits/len(shots) if shots else None,
        clearance_rate=destroyed/len(pool.alive), shots=shots, **details)


def draw_salvo_frame(frame, perception, pool, shots):
    canvas = np.full((720, 1280, 3), 22, dtype=np.uint8)
    rgb = frame.rgb.copy()
    for target in perception.latest_result.targets:
        if target.status == 'lost':
            continue
        box = np.rint(target.bbox_xyxy).astype(int)
        color = (0, 255, 0) if target.target_id == pool.selected_id else (255, 200, 50)
        cv2.rectangle(rgb, tuple(box[:2]), tuple(box[2:]), color, 1)
    canvas[:576, :1024] = rgb
    lines = [f'MuJoCo SS | t={frame.time:.2f}s | selected={pool.selected_id}',
        f'Destroyed={len(pool.removed_ids)}/{len(pool.alive)} | shots={len(shots)}',
        'One GT-box bootstrap; RGB LK/OSTrack + selected-target LiDAR + GRU afterwards',
        'Standing robot; predictive ballistic scoring; group D actor wait/reset protocol']
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 600+27*i), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def run(args):
    engagement = SalvoEngagementConfig(stall_timeout_s=args.salvo_stall_timeout_s,
        retry_delay_s=args.salvo_retry_delay_s, max_attempts=args.salvo_max_attempts,
        miss_limit=args.salvo_miss_limit, switch_policy=args.salvo_switch_policy)
    if args.threads < 1 or not np.isfinite(args.salvo_timeout_s) or args.salvo_timeout_s <= 0:
        raise ValueError('Threads and timeout must be positive')
    config = prepare_config(args)
    warmup = float(config['robot_warmup_s'])
    if not np.isfinite(warmup) or warmup < 0:
        raise ValueError('robot_warmup_s must be finite and nonnegative')
    if not args.dry_run and args.salvo_timeout_s <= warmup:
        raise ValueError('The episode timeout must exceed standing warmup')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    torch.manual_seed(args.seed)
    model = build_salvo_model(ASSET, args.salvo_targets, config['target_radius_m'])
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    mujoco.mj_forward(model, data)
    pool = MujocoStaticTargetPool(model, data, args.salvo_targets, config['target_radius_m'])
    args.output = project_path(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    with MujocoCameraSensor(model, ASSET/'camera_config.json') as camera:
        camera._option.geomgroup[5] = 1  # All nonselected pool targets remain visible.
        if args.dry_run:
            digest = place_layout(args, pool, camera)
            boxes = pool.bootstrap_boxes(camera)
            report = make_report(pool, [], reason='dry_run', dry_run=True, layout_sha256=digest,
                initial_boxes={str(i): list(b) for i, b in boxes.items()}, config=config,
                preview_pose='stand keyframe before locomotion warmup; no policy execution',
                engagement_config=asdict(engagement), camera=camera.describe())
            (args.output/'salvo_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('[DRY RUN] '+str(args.output/'salvo_report.json'), flush=True)
            return report
        settings = checkpoint_settings(config['decision_checkpoint'])
        original_budget = settings['max_shots_per_episode']
        settings['max_shots_per_episode'] = 0  # The saved single-target budget must not end pool clearance.
        control = json.loads((ASSET/'control.json').read_text())
        adapter = MujocoShootingAdapter(model, control, data, settings=settings)
        legs = LocomotionBridge(model, control, hip_delay=0, knee_delay=4)
        locomotion = torch.jit.load(config['locomotion_policy'], map_location='cpu').eval()
        policy = RecurrentShootingPolicy(config['decision_checkpoint'], device=args.device)
        gate = SalvoActorGate(policy)
        chain = ShootingChainConfig(target_distance_range_m=tuple(settings['target_radius_range']),
            target_speed_range_mps=tuple(settings['target_speed_range']))
        perception = SalvoPerceptionPipeline(model, camera, config, device=args.device,
            stabilize_camera=args.stabilize_camera, engagement_config=engagement)
        dt = float(model.opt.timestep)*control['control_decimation']
        shots, digest, reason, error = [], None, 'timeout', None
        previous_id = None
        start = time.perf_counter()
        video = VideoWriter(args.output/'salvo.mp4', 1/(2*dt)) if args.video else None
        print(f'[SALVO] targets={args.salvo_targets} timeout={args.salvo_timeout_s}s (includes {warmup}s warmup)', flush=True)
        print('[MODELS] '+json.dumps(config), flush=True)
        try:
            with (args.output/'steps.jsonl').open('x') as stream:
                for step in range(int(np.ceil(args.salvo_timeout_s/dt))):
                    t = float(data.time)
                    rotation = data.xmat[adapter.root].reshape(3, 3)
                    heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                    command = np.array([0., 0., np.clip(-.5*heading, -.8, .8)])
                    with torch.inference_mode():
                        leg_action = locomotion(torch.from_numpy(legs.observe(data, command))[None])[0].numpy()
                    legs.set_action(leg_action)
                    in_warmup = t < warmup-1e-8
                    selected_id, ready, frame = None, False, None
                    if in_warmup:
                        obs = build_policy_observation(np.zeros(3), np.zeros(3), adapter.robot_state(data),
                            prediction_ready=False, config=chain)
                        action = policy(obs)
                    else:
                        initial_boxes = None
                        if digest is None:
                            digest = place_layout(args, pool, camera)
                            initial_boxes = pool.bootstrap_boxes(camera)
                        frame = camera.capture(data)
                        selected = perception.track(frame, initial_boxes=initial_boxes, removed_ids=pool.removed_ids)
                        selected_id = perception.selected_id
                        pool.select(selected_id)
                        if selected_id != previous_id:
                            adapter.high_steps, adapter.fire_intent = 0, 0.
                            previous_id = selected_id
                        sensed = perception.estimate_selected(frame, data, adapter, selected)
                        ready = sensed.ready
                        obs = build_policy_observation(sensed.position if ready else np.zeros(3),
                            sensed.velocity if ready else np.zeros(3), adapter.robot_state(data),
                            prediction_ready=ready, config=chain)
                        action = gate.action(selected_id, ready, obs)
                        terminal = perception.terminal_reason()
                        if terminal is not None:
                            reason = terminal
                            break
                    fired = adapter.command(action, ready=ready, warmup=in_warmup)
                    if video is not None and frame is not None and step % 2 == 0:
                        video.write(draw_salvo_frame(frame, perception, pool, shots))
                    for _ in range(control['control_decimation']):
                        legs.apply(data)
                        adapter.apply(data)
                        mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
                        raise RuntimeError('Nonfinite state or MuJoCo warning; aborted without reset')
                    if fired:
                        muzzle, forward, velocity = adapter.muzzle(data)
                        metric = ballistic_metrics(muzzle, forward, velocity, pool.positions[selected_id],
                                                  np.zeros(3), config=settings)
                        shot = dict(target_id=selected_id, decision_time_s=t, time_s=float(data.time),
                            hit=bool(metric.hit), minimum_distance_m=metric.minimum_distance_m,
                            closest_time_s=metric.closest_time_s,
                            target_position_w=pool.positions[selected_id].tolist(),
                            muzzle_position_w=muzzle.tolist(), muzzle_forward_w=forward.tolist())
                        shots.append(shot)
                        perception.scheduler.record_shot(selected_id, hit=metric.hit)
                        if metric.hit:
                            pool.remove(selected_id)
                        print(f'[SHOT] id={selected_id} hit={metric.hit} destroyed={len(pool.removed_ids)}/{len(pool.alive)}', flush=True)
                    tilt = float(np.rad2deg(np.arccos(np.clip(data.xmat[adapter.root].reshape(3, 3)[2, 2], -1., 1.))))
                    row = dict(time_s=t, target_id=selected_id, ready=ready, fired=fired,
                        action=action.tolist(), turret_target_rad=adapter.target.tolist(),
                        destroyed=len(pool.removed_ids), shots=len(shots),
                        robot_z_m=float(data.xpos[adapter.root, 2]), tilt_deg=tilt,
                        actor_resets=gate.resets, cooldown_steps=adapter.cooldown)
                    row['source'] = 'warmup' if in_warmup else sensed.source
                    row['estimated_position_base_m'] = sensed.position.tolist() if ready else None
                    row['estimated_velocity_mps'] = sensed.velocity.tolist() if ready else None
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    if not pool.alive.any():
                        reason = 'cleared'
                        break
                    if row['robot_z_m'] < .12 or tilt > 60.:
                        reason = 'base_failure'
                        break
                    if step % 50 == 0:
                        print(f'[CHAIN] t={data.time:.2f}s selected={selected_id} ready={ready} remaining={pool.alive.sum()}', flush=True)
        except Exception as exc:
            reason, error = 'simulator_error', f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if video is not None:
                video.close()
            if perception.scheduler.current_id in pool.removed_ids:
                perception.scheduler.choose(float(data.time), {}, removed_ids=pool.removed_ids)
            perception.scheduler.removed.update(pool.removed_ids)
            report = make_report(pool, shots, reason=reason, error=error, layout_sha256=digest,
                seed=args.seed, duration_s=float(data.time), wall_seconds=time.perf_counter()-start,
                timeout_s=args.salvo_timeout_s, warmup_s=warmup, config=config,
                hashes={k: sha256(config[k]) for k in ('decision_checkpoint', 'velocity_checkpoint', 'ostrack_checkpoint')},
                locomotion_policy=config['locomotion_policy'],
                locomotion_policy_sha256=sha256(config['locomotion_policy']),
                mujoco_version=mujoco.__version__, physics_dt_s=float(model.opt.timestep), control_dt_s=dt,
                camera=camera.describe(), salvo_ablation_group='D', actor_calls=policy.calls,
                actor_resets=gate.resets, actor_wait_steps=gate.wait_steps,
                bootstrap_frames=perception.bootstrap_calls, bootstrap_targets=perception.bootstrap_targets,
                ground_truth_feedback_to_policy=False,
                ground_truth_usage='Scene placement; initial boxes once for every ID; shot scoring only',
                lidar_protocol='Only selected target is raycastable; all alive targets remain visible',
                scoring='Synchronous predictive ballistics after physics; no physical projectile contacts',
                fire_settings={k: settings[k] for k in ('fire_intent_threshold', 'fire_intent_hold_steps', 'fire_cooldown_steps', 'max_shots_per_episode')},
                saved_single_target_shot_budget=original_budget, automatic_resets=0,
                camera_rotation_compensation=args.stabilize_camera,
                engagement=perception.scheduler.summary())
            (args.output/'salvo_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('[DONE] '+str(args.output/'salvo_report.json'), flush=True)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'shooting_config.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing runs are never overwritten')
    parser.add_argument('--case', choices=['ss'], default='ss')
    parser.add_argument('--salvo-targets', '--salvo_targets', type=int, default=100)
    parser.add_argument('--salvo-timeout-s', '--salvo_timeout_s', type=float, default=600.)
    parser.add_argument('--salvo-switch-policy', '--salvo_switch_policy', choices=['both', 'misses', 'timeout'], default='both')
    parser.add_argument('--salvo-stall-timeout-s', '--salvo_stall_timeout_s', type=float, default=3.)
    parser.add_argument('--salvo-retry-delay-s', '--salvo_retry_delay_s', type=float, default=1.)
    parser.add_argument('--salvo-max-attempts', '--salvo_max_attempts', type=int, default=3)
    parser.add_argument('--salvo-miss-limit', '--salvo_miss_limit', type=int, default=5)
    parser.add_argument('--layout-in', '--salvo_layout_in', type=Path)
    parser.add_argument('--decision-checkpoint', type=Path)
    parser.add_argument('--model-root', type=Path, help='Override the model asset root from the config')
    parser.add_argument('--locomotion-policy', type=Path, help='Override locomotion weights; relative to model-root')
    parser.add_argument('--seed', type=int, default=61)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--stabilize-camera', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Build and validate scene/layout without inference or rendering')
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()

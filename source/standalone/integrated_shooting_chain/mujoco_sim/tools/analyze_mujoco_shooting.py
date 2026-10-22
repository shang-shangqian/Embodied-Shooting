"""Audit saved shooting results without policy inference or physics stepping."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from play_mujoco_lite3 import mujoco, np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from integrated_shooting_chain.mujoco_sim.mujoco_camera import MujocoCameraSensor
from integrated_shooting_chain.mujoco_sim.shooting import checkpoint_settings


def analyze(folder):
    report = json.loads((folder/'report.json').read_text())
    rows = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    trace = np.load(folder/'trajectory.npz')
    model = mujoco.MjModel.from_xml_path(report['model'])
    data = mujoco.MjData(model)
    stimulus = report['target_stimulus']
    settings = checkpoint_settings(report['config']['decision_checkpoint'])
    warmup = stimulus['motion_starts_s']
    high_count = maximum_high = 0
    eligible = visible = lost_visible = lost_count = 0
    pixel_errors, yaw_errors, pitch_errors = [], [], []
    blocked = Counter()
    with MujocoCameraSensor(model, ROOT/'mjcf/lite3_with_turret/camera_config.json') as camera:
        for i, row in enumerate(rows):
            if row['time_s'] < warmup-1e-8:
                continue
            eligible += 1
            index = max(0, i*4-1)
            data.qpos[:] = trace['physics_qpos'][index]
            data.qvel[:] = trace['physics_qvel'][index]
            data.time = row['time_s']
            truth = np.asarray(stimulus['position0_w'])+np.asarray(stimulus['velocity_w'])*max(0., data.time-warmup)
            uv, depth, valid = camera.project_world(data, truth)
            visible += int(valid)
            lost = row['source'] == 'lost'
            lost_count += int(lost)
            lost_visible += int(lost and valid)
            if row['bbox_xyxy'] is not None and valid and not lost:
                box = np.array(row['bbox_xyxy'])
                pixel_errors.append(float(np.linalg.norm((box[:2]+box[2:])/2-uv)))
            if row['ready']:
                yaw_errors.append(float(trace['observations'][i, 7])*.05)
                pitch_errors.append(float(trace['observations'][i, 8])*.05)
            intent = (row['action'][2]+1)*.5
            high = row['ready'] and intent > settings['fire_intent_threshold']
            high_count = high_count+1 if high else 0
            maximum_high = max(maximum_high, high_count)
            if not row['ready']:
                blocked['prediction_not_ready'] += 1
            elif not high:
                blocked['intent_below_threshold'] += 1
            elif high_count < settings['fire_intent_hold_steps']:
                blocked['intent_hold_not_confirmed'] += 1
            else:
                blocked['confirmed_intent_or_cooldown'] += 1
    result = {'scope': 'evaluation only: geometric projection replay, no target truth feedback',
        'engagement_frames': eligible, 'ready_frames': report['ready_frames'],
        'target_center_visible_frames': visible, 'tracking_lost_frames': lost_count,
        'lost_while_target_center_visible_frames': lost_visible,
        'mean_tracker_center_error_px_when_not_lost': float(np.mean(pixel_errors)) if pixel_errors else None,
        'maximum_consecutive_high_intent_frames': maximum_high,
        'fire_intent_threshold': settings['fire_intent_threshold'], 'fire_intent_hold_steps': settings['fire_intent_hold_steps'],
        'maximum_fire_intent': max((r['action'][2]+1)*.5 for r in rows),
        'fire_gate_frame_categories': dict(blocked),
        'rms_actor_yaw_error_rad_ready': float(np.sqrt(np.mean(np.square(yaw_errors)))) if yaw_errors else None,
        'rms_actor_pitch_error_rad_ready': float(np.sqrt(np.mean(np.square(pitch_errors)))) if pitch_errors else None,
        'shot_count': report['shot_count'], 'hit_count': report['hit_count']}
    (folder/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'folder': str(folder), **result}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', type=Path, nargs='+')
    for folder in parser.parse_args().folders:
        analyze(folder.resolve())

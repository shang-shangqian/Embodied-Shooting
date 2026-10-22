"""Capture the TORSO-mounted camera and optionally query LiDAR using an RGB bearing.

The robot is held at stand; target motion is prescribed. A simple red-pixel box
provides the demo visual bearing. This is not OSTrack or closed-loop aiming.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mujoco_camera import MujocoCameraSensor
from mujoco_lidar import ArrayLidarConfig, MujocoArrayLidar
from preview_lite3_mjcf import save_rgb_png


def red_box(rgb):
    """Image-only toy detector: no segmentation IDs or target truth are used."""
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y, x = np.nonzero((red > 70) & (red > 1.5 * green + 10) & (red > 1.5 * blue + 10))
    return [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1] if len(x) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'mjcf/lite3_with_turret/scene_lidar_demo.xml')
    parser.add_argument('--config', type=Path, default=ROOT / 'mjcf/lite3_with_turret/camera_config.json')
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--frame-dt', type=float, default=.02)
    parser.add_argument('--distance', type=float, default=10.)
    parser.add_argument('--depth', action='store_true', help='Save axial depth in metres as NPY.')
    parser.add_argument('--lidar', action='store_true', help='Use RGB detection to query the same-frame LiDAR hits.')
    parser.add_argument('--output', type=Path, default=Path('outputs/usd_mjcf_validation_20261021/camera'))
    args = parser.parse_args()
    if args.frames < 1 or not np.isfinite([args.frame_dt, args.distance]).all() or min(args.frame_dt, args.distance) <= 0:
        parser.error('frames, frame-dt and distance must be positive')
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    target_id = model.body('lidar_demo_target').mocapid[0]
    target_geom_id = model.geom('lidar_demo_target_surface').id
    if target_id < 0:
        raise ValueError('Demo target must be a mocap body.')
    lidar = MujocoArrayLidar(model, ArrayLidarConfig.from_json(args.config.parent / 'lidar_config.json')) if args.lidar else None
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with MujocoCameraSensor(model, args.config) as camera:
        k = camera.intrinsic_matrix
        (args.output / 'intrinsics.json').write_text(json.dumps({
            'K': k.tolist(), 'fx': float(k[0, 0]), 'fy': float(k[1, 1]),
            'cx': float(k[0, 2]), 'cy': float(k[1, 2]),
            'width': camera.width, 'height': camera.height}, indent=2) + '\n')
        for i in range(args.frames):
            data.time = i * args.frame_dt
            position, rotation = camera.pose(data)
            # This trajectory is only a scene stimulus; it is not a sensor input.
            target_optical = np.array([0.5 * np.sin(data.time), -0.3, args.distance])
            data.mocap_pos[target_id] = position + rotation @ target_optical
            frame = camera.capture(data, depth=args.depth)
            save_rgb_png(args.output / f'rgb_{i:06d}.png', frame.rgb)
            bbox = red_box(frame.rgb)
            row = {'time_s': frame.time, 'rgb_bbox_xyxy': bbox,
                   'camera_position_w': frame.position_w.tolist(),
                   'camera_rotation_w_optical': frame.rotation_w_optical.tolist()}
            if bbox is not None:
                uv = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
                yaw, pitch = camera.pixel_to_angles(uv)
                row['visual_yaw_rad'], row['visual_pitch_rad'] = float(yaw), float(pitch)
                if lidar:
                    scan = lidar.scan(data)
                    result = lidar.estimate_target(scan, yaw=float(yaw), pitch=float(pitch),
                        camera_position_w=frame.position_w, camera_rotation_w=frame.rotation_w_optical,
                        target_radius=.35, target_geom_id=target_geom_id)
                    row['lidar'] = {'valid': result.hit_valid,
                        'center_distance_m': result.center_distance if result.hit_valid else None,
                        'surface_distance_m': result.surface_distance if result.hit_valid else None,
                        'hit_count': int(scan.valid.sum())}
                    scan.save_npz(args.output / 'last_lidar_scan.npz')
            if args.depth:
                np.save(args.output / f'depth_{i:06d}.npy', frame.depth_m)
            rows.append(row)
        save_rgb_png(args.output / 'camera_rgb.png', frame.rgb)
        np.savez_compressed(args.output / 'last_camera_frame.npz', rgb=frame.rgb,
                            intrinsic_matrix=frame.intrinsic_matrix, position_w=frame.position_w,
                            rotation_w_optical=frame.rotation_w_optical, time=frame.time)
        report = {'camera': camera.name, 'resolution': [camera.width, camera.height],
                  'intrinsic_matrix': k.tolist(), 'clip_range_m': [camera.config['near'], camera.config['far']],
                  'frames': args.frames, 'detected_frames': sum(r['rgb_bbox_xyxy'] is not None for r in rows),
                  'scope': 'Kinematic mounted-camera demo; RGB colour threshold, not OSTrack or control.',
                  'samples': rows}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))
        print('Saved RGB image: ' + str((args.output / 'camera_rgb.png').resolve()))


if __name__ == '__main__':
    main()

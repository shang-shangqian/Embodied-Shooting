"""Run the array LiDAR on a prescribed moving target, with the robot held at stand.

The visual direction input is fixed at (0, 0), not computed from target truth.
The demo camera optical frame is co-located with the LiDAR, not the missing USD
camera. Ground-truth distance is used only for the diagnostic error metric.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mujoco_lidar import ArrayLidarConfig, MujocoArrayLidar
from preview_lite3_mjcf import save_rgb_png

# Columns: optical right, down, forward, expressed in LiDAR coordinates.
OPTICAL_TO_LIDAR = np.array([[0., 0, 1], [-1., 0, 0], [0., -1, 0]])


def draw_scan(scene, scan, maximum=128):
    """Display a subset of actual hits; the measurement still uses all rays."""
    valid = np.flatnonzero(scan.valid)
    if not len(valid):
        return
    indices = valid[np.linspace(0, len(valid) - 1, min(len(valid), maximum)).astype(int)]
    for index in indices:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.array([0.1, 1., 0.25, 1.], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1., scan.origin_w, scan.ray_hits_w[index])
        scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'mjcf/lite3_with_turret/scene_lidar_demo.xml')
    parser.add_argument('--config', type=Path, default=ROOT / 'mjcf/lite3_with_turret/lidar_config.json')
    parser.add_argument('--headless', action='store_true', help='No graphics context or window.')
    parser.add_argument('--snapshot', type=Path, help='Save one rendered scan and exit.')
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--scan-hz', type=float, default=50.)
    parser.add_argument('--target-distance', type=float, default=10.)
    parser.add_argument('--output', type=Path, default=Path('outputs/usd_mjcf_validation_20261021/lidar'))
    args = parser.parse_args()
    if not np.isfinite([args.duration, args.scan_hz, args.target_distance]).all() or min(args.duration, args.scan_hz) <= 0:
        parser.error('duration and scan-hz must be positive finite numbers')
    if args.target_distance <= 0.35:
        parser.error('target-distance must exceed the target radius (0.35 m)')
    if args.snapshot and args.headless:
        parser.error('--snapshot cannot be combined with --headless')
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    config = ArrayLidarConfig.from_json(args.config)
    sensor = MujocoArrayLidar(model, config)
    target_id = model.body('lidar_demo_target').mocapid[0]
    if target_id < 0:
        raise ValueError('The demo target must be a mocap body.')
    target_geom_id = model.geom('lidar_demo_target_surface').id
    rows = []
    print(f'Kinematic LiDAR demo: {sensor.num_rays} rays, {sensor.grid_shape}; robot held at stand.')
    print('Fixed optical yaw/pitch prior = 0/0; no camera inference or learned policy.')

    def sample(t):
        data.time = t
        mujoco.mj_forward(model, data)
        parent_rotation = data.body(config.parent_body).xmat.reshape(3, 3)
        origin = data.body(config.parent_body).xpos + parent_rotation @ np.asarray(config.offset_position)
        # Scripted target motion is for demonstration, not a range/bearing input.
        data.mocap_pos[target_id] = origin + parent_rotation @ np.array([
            args.target_distance + 0.5 * np.sin(0.7*t), 0.4 * np.sin(0.9*t), 0.1 * np.sin(0.5*t)])
        scan = sensor.scan(data)
        estimate = sensor.estimate_target(scan, yaw=0., pitch=0., camera_position_w=scan.origin_w,
                                          camera_rotation_w=scan.rotation_w @ OPTICAL_TO_LIDAR,
                                          target_radius=0.35, target_geom_id=target_geom_id)
        truth = float(np.linalg.norm(data.mocap_pos[target_id] - scan.origin_w))
        rows.append({'time_s': t, 'hit_count': int(scan.valid.sum()), 'valid': estimate.hit_valid,
                     'center_distance_m': estimate.center_distance if estimate.hit_valid else None,
                     'surface_distance_m': estimate.surface_distance if estimate.hit_valid else None,
                     'refined_yaw_rad': estimate.refined_yaw, 'refined_pitch_rad': estimate.refined_pitch,
                     'diagnostic_true_center_distance_m': truth,
                     'diagnostic_center_error_m': estimate.center_distance - truth if estimate.hit_valid else None,
                     'scan_ms': 1000 * scan.scan_seconds})
        return scan

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [args.target_distance / 2, 0, 0.35]
    camera.distance = max(3., args.target_distance * 1.4)
    camera.azimuth, camera.elevation = 100, -20
    if args.snapshot:
        scan = sample(0.)
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        option.geomgroup[3] = 0
        option.geomgroup[config.target_geom_group] = 1
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            draw_scan(renderer.scene, scan)
            save_rgb_png(args.snapshot, renderer.render())
        print('Saved LiDAR preview: ' + str(args.snapshot.resolve()))
    elif args.headless:
        for frame in range(int(np.ceil(args.duration * args.scan_hz))):
            scan = sample(frame / args.scan_hz)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[config.target_geom_group] = 1
            for frame in range(int(np.ceil(args.duration * args.scan_hz))):
                if not viewer.is_running():
                    break
                start = time.monotonic()
                scan = sample(frame / args.scan_hz)
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_scan(viewer.user_scn, scan)
                viewer.sync()
                time.sleep(max(0., 1/args.scan_hz - (time.monotonic() - start)))
    if not rows:
        print('No scan captured.')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    scan.save_npz(args.output / 'last_scan.npz')
    scan_times = [r['scan_ms'] for r in rows]
    errors = [r['diagnostic_center_error_m'] for r in rows if r['valid']]
    report = {'mujoco_version': mujoco.__version__, 'num_rays': sensor.num_rays,
              'grid_shape': sensor.grid_shape, 'frames': len(rows),
              'valid_frames': sum(r['valid'] for r in rows), 'requested_scan_hz': args.scan_hz,
              'scan_ms_median': float(np.median(scan_times)), 'scan_ms_max': max(scan_times),
              'maximum_absolute_center_error_m': max(map(abs, errors)) if errors else None,
              'scope': 'Kinematic moving-sphere demo; synthetic co-located optical frame and fixed visual angle prior.',
              'not_validated': ['Real camera alignment', 'Isaac-to-MuJoCo sensor parity in a common scene',
                                'Closed-loop perception/control or real-time scheduling'],
              'config': asdict(config), 'samples': rows}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('samples', 'config')}, indent=2))


if __name__ == '__main__':
    main()

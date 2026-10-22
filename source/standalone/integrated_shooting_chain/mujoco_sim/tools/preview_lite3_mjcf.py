"""Preview the converted robot, optionally holding its default joint targets.

Run in the mujoco_test environment. All log messages use ASCII.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import time
import zlib

import mujoco
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'mjcf/lite3_with_turret/scene.xml'


def save_snapshot(model, data, path, show_collisions):
    """Render a static stand pose; write RGB PNG without extra dependencies."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.25]
    camera.distance, camera.azimuth, camera.elevation = 1.5, 135, -20
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3] = show_collisions
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        pixels = renderer.render()

    save_rgb_png(path, pixels)
    print('Saved static stand preview: ' + str(path.resolve()))


def save_rgb_png(path, pixels):
    def chunk(kind, payload):
        return (struct.pack('!I', len(payload)) + kind + payload
                + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff))

    rows = b''.join(b'\x00' + row.tobytes() for row in pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'
                     + chunk(b'IHDR', struct.pack('!2I5B', pixels.shape[1], pixels.shape[0], 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


class StandController:
    """Deterministic torque PD; this is not the learned locomotion policy."""

    def __init__(self, model, control):
        names = control['actuator_order']
        jids = [model.joint(name).id for name in names]
        self.qadr = model.jnt_qposadr[jids]
        self.vadr = model.jnt_dofadr[jids]
        self.aids = [model.actuator(name + '_motor').id for name in names]
        settings = [control['joints'][name] for name in names]
        self.target = np.array([s['default_position'] for s in settings])
        self.kp = np.array([s['kp'] for s in settings])
        self.kd = np.array([s['kd'] for s in settings])
        self.limit = np.array([s['effort_limit'] for s in settings])

    def apply(self, data):
        torque = self.kp * (self.target - data.qpos[self.qadr]) - self.kd * data.qvel[self.vadr]
        data.ctrl[self.aids] = np.clip(torque, -self.limit, self.limit)


def stand_smoke(model, control, duration=5.0):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    mujoco.mj_forward(model, data)
    controller = StandController(model, control)
    heights, tilts = [], []
    max_contacts = 0
    min_contact_distance = 0.0
    for _ in range(int(np.ceil(duration / model.opt.timestep))):
        controller.apply(data)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError('Non-finite simulation state.')
        # Refresh all derived fields at the new qpos before collecting metrics.
        mujoco.mj_forward(model, data)
        heights.append(float(data.body('TORSO').xpos[2]))
        tilts.append(float(np.arccos(np.clip(data.body('TORSO').xmat.reshape(3, 3)[2, 2], -1, 1))))
        max_contacts = max(max_contacts, data.ncon)
        if data.ncon:
            min_contact_distance = min(min_contact_distance, float(min(c.dist for c in data.contact)))
    warnings = {mujoco.mjtWarning(i).name: int(w.number)
                for i, w in enumerate(data.warning) if w.number}
    return {
        'duration_seconds': float(data.time), 'finite_state': True,
        'controller': 'deterministic default-position torque PD; no learned policy or actuator delay',
        'initial_base_height_m': 0.35,
        'final_base_position_m': data.body('TORSO').xpos.tolist(),
        'base_height_range_m': [min(heights), max(heights)],
        'maximum_tilt_degrees': float(np.rad2deg(max(tilts))),
        'maximum_contacts': int(max_contacts),
        'minimum_contact_distance_m': min_contact_distance,
        'solver_warnings': warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--simulate', action='store_true', help='Step with stand PD instead of a static preview.')
    parser.add_argument('--show-collisions', action='store_true')
    parser.add_argument('--headless', action='store_true', help='Run CPU stand smoke test, without rendering.')
    parser.add_argument('--snapshot', type=Path, help='Save a static stand PNG and exit; select MUJOCO_GL before launching.')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--camera', help='Use a named mounted camera in the interactive viewer.')
    args = parser.parse_args()
    if args.snapshot and (args.headless or args.simulate):
        parser.error('--snapshot cannot be combined with --headless or --simulate')
    if args.camera and (args.snapshot or args.headless):
        parser.error('--camera is for the interactive viewer; use demo_mujoco_camera.py to capture calibrated RGB')
    if args.duration <= 0:
        parser.error('--duration must be positive')
    model_path = args.model.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    control = json.loads((model_path.parent / 'control.json').read_text())
    if args.headless:
        print(json.dumps(stand_smoke(model, control, args.duration), indent=2))
        return
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    mujoco.mj_forward(model, data)
    if args.snapshot:
        save_snapshot(model, data, args.snapshot, args.show_collisions)
        return
    from mujoco import viewer as mj_viewer
    controller = StandController(model, control)
    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[3] = args.show_collisions
        viewer.cam.lookat[:] = [0, 0, 0.25]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth, viewer.cam.elevation = 135, -20
        if args.camera:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera(args.camera).id
            viewer.opt.geomgroup[4] = 1
        print('Preview mode: ' + ('stand PD simulation' if args.simulate else 'static stand pose'))
        while viewer.is_running():
            start = time.monotonic()
            if args.simulate:
                controller.apply(data)
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - start)))


if __name__ == '__main__':
    main()

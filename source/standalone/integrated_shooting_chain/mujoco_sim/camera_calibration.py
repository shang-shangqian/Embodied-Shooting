"""Convert a measured ordinary Isaac evaluation camera to a MuJoCo profile.

No Isaac, USD or MuJoCo runtime is required. The bundled snapshot comes from
the actual TiledCamera and its render proxy, not the collector's fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CALIBRATION_PATH = Path(__file__).resolve().parent/'mjcf/lite3_with_turret/isaac_camera_calibration.json'


def camera_profile_from_snapshot(path=CALIBRATION_PATH):
    path = Path(path)
    snapshot = json.loads(path.read_text())
    width, height = snapshot['resolution']
    k = np.asarray(snapshot['intrinsic_matrix'], dtype=float)
    attrs = snapshot['proxy_camera']['attributes']
    value = lambda name: attrs[name]['value']
    if (value('projection') != 'perspective' or k.shape != (3, 3)
            or not np.isfinite(k).all() or min(width, height, k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('Expected a finite perspective pinhole calibration without skew')
    position = np.array(snapshot['local_position_torso_m'], dtype=float)
    q = np.array(snapshot['local_quaternion_usd_wxyz'], dtype=float)
    if position.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[position, q]).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Expected a finite camera mount in TORSO')
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1
    w, x, y, z = q
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    local = np.eye(4)
    local[:3, :3], local[:3, 3] = rotation, position
    near, far = value('clippingRange')
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError('Invalid captured clipping range')
    return dict(name='tracking_camera', parent_body='TORSO', profile_id=snapshot['profile_id'],
        width=width, height=height, fx=float(k[0, 0]), fy=float(k[1, 1]),
        cx=float(k[0, 2]), cy=float(k[1, 2]), near=near, far=far,
        position_body=position.tolist(), quaternion_body_wxyz=q.tolist(),
        sensor_size=[value('horizontalAperture')/1000, value('verticalAperture')/1000],
        source=dict(calibration_file=path.name, calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **snapshot['provenance'], camera_prim=snapshot['source_camera']['path'],
            render_camera_prim=snapshot['proxy_camera']['path'], fallback_camera_used=False,
            camera_pose_body_usd=local.tolist(),
            parameters=dict(camera_width=width, camera_height=height,
                camera_projection_type=value('projection'), camera_focal_length=value('focalLength'),
                camera_horizontal_aperture=value('horizontalAperture'),
                camera_vertical_aperture=value('verticalAperture'), camera_clipping_range=[near, far],
                camera_f_stop=value('fStop'), camera_focus_distance=value('focusDistance')),
            intrinsics_rule='Measured ordinary test7_2_weight TiledCamera K, stored as float32.'),
        notes=['USD/OpenGL mount is used directly; ROS optical rotation flips camera Y and Z.',
               'Intrinsics and mount come from the actual ordinary Isaac evaluation render proxy.',
               'fStop=0 disables depth of field; focus distance is metadata, not optical focus simulation.',
               'Rendering and lighting differ from Isaac RTX; matching calibration does not match image appearance.'])

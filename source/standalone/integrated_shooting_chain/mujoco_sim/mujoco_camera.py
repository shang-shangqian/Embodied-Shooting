"""Mounted RGB pinhole camera matching the ordinary Isaac evaluation profile.

Set MUJOCO_GL before importing mujoco for headless rendering. Geometry methods
work without creating a graphics context. Camera outputs use RGB uint8 HWC and
ROS optical coordinates (right, down, forward); depth is axial Z in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

OPENGL_TO_OPTICAL = np.diag([1., -1., -1.])


@dataclass(frozen=True)
class CameraFrame:
    time: float
    rgb: np.ndarray
    intrinsic_matrix: np.ndarray
    position_w: np.ndarray
    rotation_w_optical: np.ndarray
    depth_m: np.ndarray | None = None
    geom_ids: np.ndarray | None = None


class MujocoCameraSensor:
    def __init__(self, model, config):
        self.model = model
        self.config = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else dict(config)
        self.name = self.config['name']
        self.camera_id = model.camera(self.name).id
        self.width, self.height = int(self.config['width']), int(self.config['height'])
        if min(self.width, self.height) <= 0:
            raise ValueError('Camera resolution must be positive.')
        if not np.array_equal(model.cam_resolution[self.camera_id], [self.width, self.height]):
            raise ValueError('Camera config resolution differs from compiled camera.')
        if model.cam_bodyid[self.camera_id] != model.body(self.config['parent_body']).id:
            raise ValueError('Camera is attached to an unexpected body.')
        if not np.allclose(model.cam_pos[self.camera_id], self.config['position_body'], atol=1e-10, rtol=0):
            raise ValueError('Camera config mount position differs from compiled camera.')
        expected_quat = np.array(self.config['quaternion_body_wxyz'], dtype=float)
        expected_quat /= np.linalg.norm(expected_quat)
        if not np.isclose(abs(model.cam_quat[self.camera_id] @ expected_quat), 1., atol=1e-10, rtol=0):
            raise ValueError('Camera config mount orientation differs from compiled camera.')
        near, far = self.config['near'], self.config['far']
        if not np.isfinite([near, far]).all() or not 0 < near < far:
            raise ValueError('Require finite 0 < near < far.')
        intrinsic = model.cam_intrinsic[self.camera_id].astype(float)
        size = model.cam_sensorsize[self.camera_id].astype(float)
        if np.any(size <= 0):
            raise ValueError('Expected a sensor-size/intrinsic calibrated MJCF camera.')
        # MuJoCo's principal offsets are measured from sensor centre. Image Y is down.
        fx, fy = intrinsic[:2] / size * [self.width, self.height]
        cx = self.width/2 - intrinsic[2] / size[0] * self.width
        cy = self.height/2 + intrinsic[3] / size[1] * self.height
        self.intrinsic_matrix = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        expected = np.array([[self.config['fx'], 0, self.config['cx']],
                             [0, self.config['fy'], self.config['cy']], [0, 0, 1.]])
        if not np.allclose(self.intrinsic_matrix, expected, atol=1e-3, rtol=0):
            raise ValueError('Camera config intrinsics differ from compiled camera.')
        self._renderer = None
        self._option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._option)
        self._option.geomgroup[3] = 0  # Hide contact proxies.
        self._option.geomgroup[4] = 1  # Show ray-detectable target surfaces.
        self._option.sitegroup[:] = 0

    def describe(self):
        """Report the compiled renderer calibration used by perception."""
        fx, fy = self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]
        return dict(profile_id=self.config.get('profile_id'), projection='perspective',
            resolution=[self.width, self.height], intrinsic_matrix=self.intrinsic_matrix.tolist(),
            fov_deg=np.rad2deg(2*np.arctan([self.width/(2*fx), self.height/(2*fy)])).tolist(),
            position_body=self.model.cam_pos[self.camera_id].tolist(),
            quaternion_body_wxyz=self.model.cam_quat[self.camera_id].tolist(),
            clipping_range_m=[self.config['near'], self.config['far']],
            calibration_sha256=self.config['source']['calibration_sha256'])

    def pose(self, data, *, update_kinematics=True):
        if update_kinematics:
            mujoco.mj_forward(self.model, data)
        position = data.cam_xpos[self.camera_id].copy()
        rotation = data.cam_xmat[self.camera_id].reshape(3, 3) @ OPENGL_TO_OPTICAL
        return position, rotation

    def project_world(self, data, points_w, *, update_kinematics=True):
        """Return pixel-boundary coordinates, axial depth and frustum validity.

        Raster pixel (i,j) has centre (j+0.5,i+0.5). The source calibration uses
        cx=width/2, cy=height/2. Behind-camera points return NaN pixel positions.
        """
        position, rotation = self.pose(data, update_kinematics=update_kinematics)
        points = np.asarray(points_w, dtype=float)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError('points_w must end with dimension 3.')
        local = (points - position) @ rotation
        depth = local[..., 2]
        safe_depth = np.where(depth > 0, depth, np.nan)
        uv = local[..., :2] / safe_depth[..., None]
        uv = uv * np.diag(self.intrinsic_matrix)[:2] + self.intrinsic_matrix[:2, 2]
        valid = (np.isfinite(uv).all(axis=-1) & (depth >= self.config['near'])
                 & (depth <= self.config['far']) & (uv[..., 0] >= 0) & (uv[..., 0] < self.width)
                 & (uv[..., 1] >= 0) & (uv[..., 1] < self.height))
        return uv, depth, valid

    def pixel_to_angles(self, uv):
        uv = np.asarray(uv, dtype=float)
        if uv.ndim == 0 or uv.shape[-1] != 2 or not np.isfinite(uv).all():
            raise ValueError('uv must contain finite pixel pairs.')
        xy = (uv - self.intrinsic_matrix[:2, 2]) / np.diag(self.intrinsic_matrix)[:2]
        return np.stack([np.arctan2(xy[..., 0], 1.),
                         np.arctan2(-xy[..., 1], np.sqrt(xy[..., 0]**2 + 1))], axis=-1)

    def capture(self, data, *, depth=False, segmentation=False, update_kinematics=True):
        position, rotation = self.pose(data, update_kinematics=update_kinematics)
        # Clipping in MuJoCo is global and expressed relative to model extent.
        self.model.vis.map.znear = self.config['near'] / self.model.stat.extent
        self.model.vis.map.zfar = self.config['far'] / self.model.stat.extent
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        renderer = self._renderer
        renderer.update_scene(data, camera=self.name, scene_option=self._option)
        rgb = renderer.render().copy()
        depth_image = geom_ids = None
        try:
            if depth:
                renderer.enable_depth_rendering()
                depth_image = renderer.render().copy()
                renderer.disable_depth_rendering()
            if segmentation:
                renderer.enable_segmentation_rendering()
                segmentation_image = renderer.render()
                geom_ids = np.where(segmentation_image[..., 1] == mujoco.mjtObj.mjOBJ_GEOM,
                                    segmentation_image[..., 0], -1).astype(np.int32)
        finally:
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
        return CameraFrame(float(data.time), rgb, self.intrinsic_matrix.copy(), position, rotation,
                           depth_image, geom_ids)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

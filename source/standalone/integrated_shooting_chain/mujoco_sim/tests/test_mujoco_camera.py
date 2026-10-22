"""Check camera calibration, body attachment and actual rasterized observations.

Run with MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 in the
mujoco_test environment. No Isaac Sim process or GPU is required.
"""
import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
sys.path.insert(0, str(ROOT))
from mujoco_camera import MujocoCameraSensor, OPENGL_TO_OPTICAL
sys.path.insert(0, str(WORKSPACE / 'source/standalone/test_lk_ostrack'))
from camera_geometry import CameraIntrinsics

ASSET = ROOT / 'mjcf/lite3_with_turret'


class TestMujocoCamera(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ASSET / 'scene_lidar_demo.xml'))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('stand').id)
        self.camera = MujocoCameraSensor(self.model, ASSET / 'camera_config.json')
        self.addCleanup(self.camera.close)

    def place_target_optical(self, xyz):
        # Independent reference from the extracted USD local camera transform.
        authored = np.array(self.camera.config['source']['camera_pose_body_usd'])
        optical = authored[:3, :3] @ OPENGL_TO_OPTICAL
        position = authored[:3, 3] + [0, 0, .35] + optical @ np.asarray(xyz)
        self.data.mocap_pos[self.model.body('lidar_demo_target').mocapid[0]] = position

    def test_intrinsics_match_recorded_isaac_calibration(self):
        recorded = WORKSPACE / 'source/standalone/test_lk_ostrack/outputs/formal_test_r035/sequences/seq_000005_env_005/intrinsics.json'
        # The source profile is also retained when the optional dataset is moved.
        expected = np.array([[1777.7777099609375, 0, 512], [0, 1777.7777099609375, 288], [0, 0, 1]])
        if recorded.is_file():
            expected = np.array(json.loads(recorded.read_text())['K'])
        np.testing.assert_allclose(self.camera.intrinsic_matrix, expected, atol=1e-3, rtol=0)
        self.assertEqual((self.camera.width, self.camera.height), (1024, 576))

    def test_mount_optical_axes_and_no_turret_tracking(self):
        position, optical = self.camera.pose(self.data)
        np.testing.assert_allclose(position, [.25, -.0005, .453], atol=1e-10)
        np.testing.assert_allclose(optical, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-8)
        self.data.qpos[self.model.joint('yaw_joint').qposadr[0]] = .4
        self.data.qpos[self.model.joint('pitch_joint').qposadr[0]] = .2
        after = self.camera.pose(self.data)
        np.testing.assert_allclose(after[0], position, atol=1e-12)
        np.testing.assert_allclose(after[1], optical, atol=1e-12)

    def test_mount_follows_base_roll_pitch_translation(self):
        rng = np.random.default_rng(20261021)
        local = np.array(self.camera.config['source']['camera_pose_body_usd'])
        for _ in range(8):
            translation = rng.normal(size=3)
            quaternion = rng.normal(size=4)
            quaternion /= np.linalg.norm(quaternion)
            self.data.qpos[:3], self.data.qpos[3:7] = translation, quaternion
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, quaternion)
            rotation = matrix.reshape(3, 3)
            position, optical = self.camera.pose(self.data)
            np.testing.assert_allclose(position, translation + rotation @ local[:3, 3], atol=1e-12)
            np.testing.assert_allclose(optical, rotation @ local[:3, :3] @ OPENGL_TO_OPTICAL, atol=1e-12)

    def test_projection_and_angles_match_test_lk_ostrack(self):
        k = self.camera.intrinsic_matrix
        original = CameraIntrinsics(k[0, 0], k[1, 1], k[0, 2], k[1, 2], 1024, 576)
        local = np.array([[.4, -.25, 5.], [-1., .2, 10.], [0., 0., 100.]])
        authored = np.array(self.camera.config['source']['camera_pose_body_usd'])
        world = local @ (authored[:3, :3] @ OPENGL_TO_OPTICAL).T + authored[:3, 3] + [0, 0, .35]
        uv, depth, valid = self.camera.project_world(self.data, world)
        np.testing.assert_allclose(uv, [original.project(p) for p in local], atol=1e-10)
        np.testing.assert_allclose(depth, local[:, 2], atol=1e-12)
        np.testing.assert_allclose(self.camera.pixel_to_angles(uv),
                                   [original.direction_from_pixel(*p) for p in uv], atol=1e-14)
        self.assertTrue(valid.all())
        position, rotation = self.camera.pose(self.data)
        _, _, valid = self.camera.project_world(self.data,
            position + np.array([[0, 0, -1], [0, 0, .05], [0, 0, 1100000]]) @ rotation.T)
        self.assertFalse(valid.any())

    def test_rendered_markers_match_pixels_without_flip(self):
        gid = self.model.geom('lidar_demo_target_surface').id
        self.model.geom_size[gid, 0] = .025
        for xyz in ([.4, -.25, 5.], [-.4, .25, 5.]):
            self.place_target_optical(xyz)
            frame = self.camera.capture(self.data, segmentation=True)
            y, x = np.nonzero(frame.geom_ids == gid)
            self.assertGreater(len(x), 100)
            observed = np.array([x.mean()+.5, y.mean()+.5])
            expected = np.array(xyz[:2]) / xyz[2] * np.diag(frame.intrinsic_matrix)[:2] + frame.intrinsic_matrix[:2, 2]
            np.testing.assert_allclose(observed, expected, atol=.75)
            self.assertEqual(frame.rgb.shape, (576, 1024, 3))
            self.assertEqual(frame.rgb.dtype, np.uint8)

    def test_far_target_visible_and_depth_is_metric_axial(self):
        gid = self.model.geom('lidar_demo_target_surface').id
        self.place_target_optical([0, 0, 100.])
        far = self.camera.capture(self.data, segmentation=True)
        self.assertGreater(int((far.geom_ids == gid).sum()), 50)
        self.place_target_optical([0, 0, 10.])
        self.data.time = .02
        frame = self.camera.capture(self.data, depth=True, segmentation=True)
        self.assertEqual(frame.time, .02)
        self.assertEqual(frame.depth_m.shape, (576, 1024))
        self.assertAlmostEqual(float(frame.depth_m[287, 511]), 9.65, delta=.01)
        self.assertEqual(frame.geom_ids[287, 511], gid)
        # Capture modes must be reset, so a subsequent RGB-only frame remains RGB.
        again = self.camera.capture(self.data)
        self.assertIsNone(again.depth_m)
        self.assertIsNone(again.geom_ids)
        self.assertEqual(again.rgb.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main(verbosity=2)

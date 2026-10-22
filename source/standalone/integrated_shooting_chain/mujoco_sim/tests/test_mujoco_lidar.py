"""Independent geometric checks for the MuJoCo LiDAR; run with mujoco_test Python."""
from pathlib import Path
from dataclasses import replace
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mujoco_lidar import ArrayLidarConfig, MujocoArrayLidar, lidar_directions


def make_sensor(**overrides):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="TORSO" mocap="true"><geom type="box" size=".1 .1 .1" group="3"/></body>
      <geom name="ignored_obstacle" type="box" pos="1 0 0" size=".1 1 1" group="0"/>
      <body name="target" mocap="true" pos="10 0 0">
        <geom name="surface" type="sphere" size=".35" group="4" contype="0" conaffinity="0"/>
      </body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    config = ArrayLidarConfig(offset_position=(0., 0., 0.), horizontal_fov_deg=0.,
                              vertical_fov_deg=0., **overrides)
    return model, data, MujocoArrayLidar(model, config)


OPTICAL_TO_WORLD = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


class TestMujocoLidar(unittest.TestCase):
    def test_default_grid_and_full_circle_seam(self):
        directions, shape = lidar_directions(ArrayLidarConfig())
        self.assertEqual(shape, (134, 901))
        self.assertEqual(len(directions), 120734)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.)
        np.testing.assert_allclose(directions[0], [0., -np.cos(np.deg2rad(20)), -np.sin(np.deg2rad(20))], atol=1e-14)
        np.testing.assert_allclose(directions[900], [0., np.cos(np.deg2rad(20)), -np.sin(np.deg2rad(20))], atol=1e-14)
        full, grid = lidar_directions(ArrayLidarConfig(horizontal_fov_deg=360., horizontal_resolution_deg=90.,
                                                      vertical_fov_deg=0.))
        self.assertEqual(grid, (1, 4))
        np.testing.assert_allclose(full, [[-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0]], atol=1e-14)

    def test_dynamic_target_and_contact_independent_detection(self):
        model, data, sensor = make_sensor()
        self.assertEqual(model.geom('surface').contype[0], 0)
        for center in (10., 100., 150.):
            data.mocap_pos[model.body('target').mocapid[0]] = [center, 0, 0]
            scan = sensor.scan(data)
            self.assertTrue(scan.valid[0])
            # The nearer group-0 obstacle is intentionally excluded.
            self.assertAlmostEqual(scan.distances[0], center - .35, places=9)
            np.testing.assert_allclose(scan.ray_hits_w[0], [center - .35, 0, 0], atol=1e-9)
            self.assertEqual(scan.geom_ids[0], model.geom('surface').id)

    def test_near_far_and_miss_contract(self):
        model, data, sensor = make_sensor(min_range=1., max_range=20.)
        for center in ([0.8, 0, 0], [21., 0, 0], [10., 10., 0]):
            data.mocap_pos[model.body('target').mocapid[0]] = center
            scan = sensor.scan(data)
            self.assertFalse(scan.valid[0])
            self.assertTrue(np.isinf(scan.distances[0]))
            self.assertTrue(np.isinf(scan.ray_hits_w[0]).all())
            self.assertEqual(scan.geom_ids[0], -1)

    def test_roll_and_mount_offset_rotation(self):
        model, data, _ = make_sensor()
        cfg = ArrayLidarConfig(horizontal_fov_deg=0., vertical_fov_deg=0., offset_position=(.23, .02, .073),
                               offset_quaternion_wxyz=(np.sqrt(.5), 0., 0., np.sqrt(.5)))
        sensor = MujocoArrayLidar(model, cfg)
        torso_id, target_id = model.body('TORSO').mocapid[0], model.body('target').mocapid[0]
        data.mocap_pos[torso_id] = [2., -3., 4.]
        data.mocap_quat[torso_id] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]  # +90 deg roll
        # +90 deg mount yaw then +90 deg body roll maps forward X to world Z.
        expected_origin = np.array([2.23, -3.073, 4.02])
        data.mocap_pos[target_id] = expected_origin + [0, 0, 5.]
        scan = sensor.scan(data)
        np.testing.assert_allclose(scan.origin_w, expected_origin, atol=1e-12)
        np.testing.assert_allclose(scan.directions_w[0], [0, 0, 1.], atol=1e-12)
        self.assertAlmostEqual(scan.distances[0], 4.65, places=10)

    def test_all_returns_against_analytic_sphere(self):
        model, data, _ = make_sensor()
        sensor = MujocoArrayLidar(model, ArrayLidarConfig(offset_position=(0., 0., 0.),
            horizontal_fov_deg=20., vertical_fov_deg=12., horizontal_resolution_deg=.5,
            vertical_resolution_deg=.5))
        center = np.array([10., .5, .15])
        data.mocap_pos[model.body('target').mocapid[0]] = center
        scan = sensor.scan(data)
        projection = scan.directions_w @ center
        discriminant = projection**2 - (center @ center - .35**2)
        expected_valid = (discriminant >= 0) & (projection > 0)
        np.testing.assert_array_equal(scan.valid, expected_valid)
        expected_distance = projection[expected_valid] - np.sqrt(discriminant[expected_valid])
        np.testing.assert_allclose(scan.distances[expected_valid], expected_distance, atol=1e-10)

    def test_center_estimate_and_invalid_visual_window(self):
        model, data, sensor = make_sensor()
        scan = sensor.scan(data)
        kwargs = dict(camera_position_w=[0., 0., 0.], camera_rotation_w=OPTICAL_TO_WORLD, target_radius=.35)
        result = sensor.estimate_target(scan, yaw=0., pitch=0., **kwargs)
        self.assertTrue(result.hit_valid)
        self.assertAlmostEqual(result.center_distance, 10., places=10)
        self.assertAlmostEqual(result.surface_distance, 9.65, places=10)
        self.assertAlmostEqual(result.refined_yaw, 0.)
        self.assertAlmostEqual(result.refined_pitch, 0.)
        # Large incorrect prior must not fall back to target ground truth.
        invalid = sensor.estimate_target(scan, yaw=.5, pitch=.1, **kwargs)
        self.assertFalse(invalid.hit_valid)
        self.assertTrue(np.isinf(invalid.center_distance))
        self.assertEqual(invalid.refined_yaw, .5)
        self.assertEqual(invalid.refined_pitch, .1)
        self.assertIsNone(invalid.estimated_center_w)
        excluded = sensor.estimate_target(scan, yaw=0., pitch=0., target_geom_id=999, **kwargs)
        self.assertFalse(excluded.hit_valid)

    def test_fallback_cone_and_camera_offset(self):
        model, data, sensor = make_sensor()
        # One hit 4 degrees from the visual prior: primary cone empty, fallback hits.
        data.mocap_pos[model.body('target').mocapid[0]] = [10., 0., 0.]
        scan = sensor.scan(data)
        result = sensor.estimate_target(scan, yaw=np.deg2rad(4), pitch=0., camera_position_w=[.2, 0, 0],
                                         camera_rotation_w=OPTICAL_TO_WORLD, target_radius=.35)
        self.assertTrue(result.hit_valid)
        self.assertAlmostEqual(result.center_distance, 9.8, places=10)

    def test_selected_surfaces_occlude_each_other(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="TORSO"/>
          <geom name="near" type="sphere" pos="2 0 0" size=".2" group="4"/>
          <geom name="far" type="sphere" pos="4 0 0" size=".2" group="4"/>
        </worldbody></mujoco>''')
        sensor = MujocoArrayLidar(model, ArrayLidarConfig(offset_position=(0., 0., 0.),
                                                        horizontal_fov_deg=0., vertical_fov_deg=0.))
        scan = sensor.scan(mujoco.MjData(model))
        self.assertEqual(scan.geom_ids[0], model.geom('near').id)
        result = sensor.estimate_target(scan, yaw=0., pitch=0., camera_position_w=[0, 0, 0],
                                         camera_rotation_w=OPTICAL_TO_WORLD, target_radius=.2,
                                         target_geom_id=model.geom('far').id)
        self.assertFalse(result.hit_valid)

    def test_raw_range_and_legacy_parent_reference_are_separate(self):
        model, data, _ = make_sensor()
        config = ArrayLidarConfig(offset_position=(.23, 0, 0), horizontal_fov_deg=0., vertical_fov_deg=0.)
        sensor = MujocoArrayLidar(model, config)
        scan = sensor.scan(data)
        self.assertAlmostEqual(scan.distances[0], 9.42, places=10)
        kwargs = dict(yaw=0., pitch=0., camera_position_w=scan.origin_w,
                      camera_rotation_w=OPTICAL_TO_WORLD, target_radius=.35)
        legacy = sensor.estimate_target(scan, **kwargs)
        self.assertAlmostEqual(legacy.surface_distance, 9.65, places=10)
        self.assertAlmostEqual(legacy.center_distance, 9.77, places=10)
        physical = MujocoArrayLidar(model, replace(config, postprocess_reference='lidar')).estimate_target(scan, **kwargs)
        self.assertAlmostEqual(physical.surface_distance, 9.42, places=10)
        self.assertAlmostEqual(physical.center_distance, 9.77, places=10)

    def test_invalid_configuration_and_empty_target_group(self):
        for changes in ({'max_range': -.1}, {'horizontal_resolution_deg': 0.}, {'target_geom_group': 6},
                        {'offset_quaternion_wxyz': (0, 0, 0, 0)}):
            with self.assertRaises(ValueError):
                ArrayLidarConfig(**changes)
        model, _, _ = make_sensor()
        with self.assertRaises(ValueError):
            MujocoArrayLidar(model, ArrayLidarConfig(target_geom_group=5))

    def test_moving_mesh_target(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
          <mesh name="box_mesh" vertex="-.2 -.2 -.2  .2 -.2 -.2  .2 .2 -.2  -.2 .2 -.2
                                            -.2 -.2 .2  .2 -.2 .2  .2 .2 .2  -.2 .2 .2"
                face="0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7"/>
          </asset><worldbody><body name="TORSO"/>
          <body name="target" mocap="true" pos="3 0 0">
            <geom name="surface" type="mesh" mesh="box_mesh" group="4" contype="0" conaffinity="0"/>
          </body></worldbody></mujoco>''')
        sensor = MujocoArrayLidar(model, ArrayLidarConfig(offset_position=(0., 0., 0.),
                                                        horizontal_fov_deg=0., vertical_fov_deg=0.))
        data = mujoco.MjData(model)
        for x in (3., 5.):
            data.mocap_pos[0] = [x, 0, 0]
            scan = sensor.scan(data)
            self.assertTrue(scan.valid[0])
            self.assertAlmostEqual(scan.distances[0], x - .2, places=7)


if __name__ == '__main__':
    unittest.main(verbosity=2)

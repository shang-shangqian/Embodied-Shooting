"""Check the policy adapter's joint/frame ordering and physics-step delays."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from play_mujoco_lite3 import LocomotionBridge, mujoco, np, command_for_time


class TestLocomotionBridge(unittest.TestCase):
    def setUp(self):
        root = ROOT / 'mjcf/lite3_with_turret'
        self.model = mujoco.MjModel.from_xml_path(str(root / 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('stand').id)
        self.control = json.loads((root / 'control.json').read_text())
        self.bridge = LocomotionBridge(self.model, self.control)

    def test_joint_order_observation_and_action_scales(self):
        b, d = self.bridge, self.data
        offsets = np.arange(12) * .01
        velocities = np.arange(12) * .1
        d.qpos[b.qadr[:12]] = b.default[:12] + offsets
        d.qvel[b.vadr[:12]] = velocities
        d.qvel[3:6] = [.4, -.8, 1.2]
        b.previous_action[:] = np.arange(12) * .2
        mujoco.mj_forward(self.model, d)
        observation = b.observe(d, [.5, .1, -.2])
        np.testing.assert_allclose(observation[:3], [.1, -.2, .3], atol=1e-7)
        np.testing.assert_allclose(observation[3:6], [0, 0, -1], atol=1e-7)
        np.testing.assert_allclose(observation[6:9], [.5, .1, -.2], atol=1e-7)
        np.testing.assert_allclose(observation[9:21], offsets, atol=1e-7)
        np.testing.assert_allclose(observation[21:33], velocities * .05, atol=1e-7)
        np.testing.assert_allclose(observation[33:], np.arange(12)*.2, atol=1e-7)
        b.set_action(np.ones(12))
        np.testing.assert_allclose(b.target[:12]-b.default[:12], [.125, .25, .25]*4, atol=1e-7)
        np.testing.assert_allclose(b.target[12:], [0, 0])

    def test_gravity_rotation_and_free_joint_angular_velocity(self):
        self.data.qpos[3:7] = [np.sqrt(.5), np.sqrt(.5), 0, 0]  # +90 deg roll.
        self.data.qvel[3:6] = [.4, -.8, 1.2]
        mujoco.mj_forward(self.model, self.data)
        observation = self.bridge.observe(self.data, [0, 0, 0])
        np.testing.assert_allclose(observation[3:6], [0, -1, 0], atol=1e-7)
        object_velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.model.body('TORSO').id, object_velocity, 0)
        # BODY local=1 uses the inertial frame, which differs from TORSO axes.
        rotation = self.data.body('TORSO').xmat.reshape(3, 3)
        np.testing.assert_allclose(observation[:3], rotation.T @ object_velocity[:3]*.25, atol=1e-7)

    def test_group_delays_tick_per_physics_step_and_fill_startup(self):
        b = LocomotionBridge(self.model, self.control, hip_delay=1, knee_delay=3)
        # Each call represents one physics step, including repeated policy targets.
        for number in range(6):
            b.target[:] = number
            delayed = b.delayed_target()
            self.assertEqual(delayed[0], max(0, number-1))
            self.assertEqual(delayed[2], max(0, number-3))
            self.assertEqual(delayed[3], max(0, number-1))
            self.assertEqual(delayed[12], number)

    def test_pd_effort_limits(self):
        self.bridge.target[:] = 100.
        raw, torque = self.bridge.apply(self.data)
        self.assertTrue(np.all(raw > self.bridge.limits))
        np.testing.assert_allclose(torque, [24, 24, 36]*4 + [3, 2])
        self.assertTrue(np.isfinite(self.data.ctrl).all())

    def test_transition_boundaries(self):
        args = SimpleNamespace(scenario='transitions', settle_seconds=0, command=[.5, 0, 0], heading_target=None)
        for t, phase, expected in [(0, 'stand', [0,0,0]), (2, 'forward', [.5,0,0]),
                                    (8, 'stop', [0,0,0]), (12, 'turn_left', [0,0,.4]),
                                    (16, 'turn_right', [0,0,-.4]), (20, 'sideways', [0,.3,0]),
                                    (24, 'backward', [-.3,0,0]), (28, 'final_stop', [0,0,0])]:
            command, name = command_for_time(t, args, np.eye(3))
            self.assertEqual(name, phase)
            np.testing.assert_allclose(command, expected, atol=1e-7)

    def test_training_heading_law_wraps_and_saturates(self):
        args = SimpleNamespace(scenario='constant', settle_seconds=0, command=[.5, 0, 0], heading_target=np.deg2rad(-179))
        angle = np.deg2rad(179)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])
        command, phase = command_for_time(0, args, rotation)
        self.assertEqual(phase, 'heading_hold')
        self.assertAlmostEqual(command[2], np.deg2rad(1), places=7)
        args.heading_target = np.pi
        command, _ = command_for_time(0, args, np.eye(3))
        self.assertAlmostEqual(abs(command[2]), .8, places=7)


if __name__ == '__main__':
    unittest.main(verbosity=2)

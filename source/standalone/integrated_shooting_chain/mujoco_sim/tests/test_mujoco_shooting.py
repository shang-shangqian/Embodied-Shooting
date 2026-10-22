"""Check shooting migration against original source functions and checkpoint."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from play_mujoco_lite3 import mujoco, np, torch
sys.path.insert(0, str(ROOT.parents[1]))
from integrated_shooting_chain.mujoco_sim.shooting import (
    SOURCE_ENV, MujocoShootingAdapter, ballistic_metrics, source_settings, yaw_rotation,
    camera_rotation_homography, transform_image_points,
)
from integrated_shooting_chain.mujoco_sim.shooting_models import RecurrentShootingPolicy

ASSET = ROOT/'mjcf/lite3_with_turret'


class TestMujocoShooting(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ASSET/'scene_shooting.xml'))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('stand').id)
        mujoco.mj_forward(self.model, self.data)
        self.adapter = MujocoShootingAdapter(self.model, json.loads((ASSET/'control.json').read_text()), self.data)

    def test_muzzle_velocity_matches_finite_difference(self):
        rng = np.random.default_rng(61)
        self.data.qvel[:] = rng.normal(size=self.model.nv)*.3
        mujoco.mj_forward(self.model, self.data)
        position, _, velocity = self.adapter.muzzle(self.data)
        dt = 1e-7
        mujoco.mj_integratePos(self.model, self.data.qpos, self.data.qvel, dt)
        mujoco.mj_forward(self.model, self.data)
        next_position, _, _ = self.adapter.muzzle(self.data)
        np.testing.assert_allclose(velocity, (next_position-position)/dt, atol=1e-7)

    def test_actor_body_frame_and_gru_yaw_frame_remain_distinct(self):
        q = np.array([.91, .22, -.13, .30])
        self.data.qpos[3:7] = q/np.linalg.norm(q)
        self.data.qvel[:6] = [1.2, -.4, .2, .7, -.3, .9]
        mujoco.mj_forward(self.model, self.data)
        state = self.adapter.robot_state(self.data)
        linear, angular, yaw_rate = self.adapter.ego_motion(self.data)
        rotation = self.data.xmat[self.adapter.root].reshape(3, 3)
        yaw = yaw_rotation(rotation)
        np.testing.assert_allclose(yaw@linear, rotation@state.base_linear_velocity_base_mps, atol=1e-12)
        np.testing.assert_allclose(state.projected_gravity_base, -rotation[2], atol=1e-12)
        self.assertFalse(np.allclose(linear, state.base_linear_velocity_base_mps, atol=1e-3))
        self.assertEqual(angular[2], yaw_rate)

    def test_fire_confirmation_cooldown_and_missing_hold(self):
        a = self.adapter
        fired = [i for i in range(16) if a.command([0., 0., 1.], ready=True)]
        self.assertEqual(fired, [1, 7, 13])
        target = a.target.copy()
        self.assertFalse(a.command([1., 1., 1.], ready=False))
        np.testing.assert_array_equal(a.target, target)
        self.assertEqual(a.high_steps, 0)
        self.assertEqual(a.fire_intent, 0.)

    def test_warmup_preserves_raw_action_history_without_moving_or_firing(self):
        a = self.adapter
        initial = a.target.copy()
        for _ in range(10):
            self.assertFalse(a.command([.7, -.4, .9], ready=False, warmup=True))
        np.testing.assert_array_equal(a.target, initial)
        np.testing.assert_allclose(a.last_action, [.7, -.4, .9])
        self.assertAlmostEqual(a.fire_intent, .95, places=6)
        self.assertEqual(a.high_steps, 0)
        self.assertEqual(a.cooldown, 0)

    def test_turret_increments_clamps_and_force_limits(self):
        a = self.adapter
        a.command([1., -1., -1.], ready=True)
        np.testing.assert_allclose(a.target, [.03, -.02])
        for _ in range(200):
            a.command([1., -1., -1.], ready=True)
        np.testing.assert_array_equal(a.target, [a.limits[0, 1], a.limits[1, 0]])
        a.apply(self.data)
        mujoco.mj_forward(self.model, self.data)
        np.testing.assert_array_less(np.abs(self.data.actuator_force[a.aids]), np.array([3., 2.])+1e-12)

    def test_native_discrete_drive_matches_implicit_spring_equation(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep="0.005" integrator="discrete" gravity="0 0 0"/>
          <worldbody><body><joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/></body></worldbody>
          <actuator><position joint="hinge" kp="8" kv="0.2"/></actuator></mujoco>''')
        data = mujoco.MjData(model)
        data.qpos[0], data.qvel[0], data.ctrl[0] = .1, .3, .14
        mujoco.mj_forward(model, data)
        inertia = np.zeros((1, 1))
        mujoco.mj_fullM(model, data, inertia)
        h = model.opt.timestep
        expected_v = (inertia[0, 0]*.3+h*8*(.14-.1))/(inertia[0, 0]+h*.2+h*h*8)
        mujoco.mj_step(model, data)
        self.assertAlmostEqual(data.qvel[0], expected_v, places=12)
        self.assertAlmostEqual(data.qpos[0], .1+h*expected_v, places=12)

    def test_camera_rotation_warp_matches_ray_projection(self):
        k = np.array([[1777., 0., 512.], [0., 1777., 288.], [0., 0., 1.]])
        ref = np.eye(3)
        current = np.array([[1., 0., 0.], [0., np.cos(.07), -np.sin(.07)], [0., np.sin(.07), np.cos(.07)]])
        rays_world = np.array([[.02, .03, 1.], [-.05, .04, 1.]])
        current_rays = rays_world@current
        pixels_current = current_rays[:, :2]/current_rays[:, 2:]*[1777., 1777.]+[512., 288.]
        expected = rays_world[:, :2]/rays_world[:, 2:]*[1777., 1777.]+[512., 288.]
        h = camera_rotation_homography(k, current, ref)
        actual = transform_image_points(pixels_current, h)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(transform_image_points(actual, np.linalg.inv(h)), pixels_current, atol=1e-10)

    def test_ballistic_parity_with_actual_source_gravity_and_drag(self):
        names = {'_compute_ballistic_metrics', '_interp_cd_from_mach', '_compute_air_drag_accel_w'}
        functions = [n for n in ast.walk(ast.parse(SOURCE_ENV.read_text())) if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(SOURCE_ENV), 'exec'), namespace)
        rng = np.random.default_rng(17)
        for drag in (False, True):
            cfg = source_settings()
            cfg['enable_air_drag'] = drag
            for _ in range(3):
                position = rng.normal(size=3)
                forward = np.array([1., .04, .03]); forward /= np.linalg.norm(forward)
                velocity = rng.normal(size=3)
                tp = np.array([100., 4., 3.])+rng.normal(size=3)
                tv, ta = rng.normal(size=3), rng.normal(size=3)*.1
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
                batch = lambda x: tensor(x).reshape(1, 3)
                times = tensor(np.linspace(0., cfg['ballistic_max_time'], cfg['ballistic_num_samples']))
                source = SimpleNamespace(cfg=SimpleNamespace(**cfg), _muzzle_body_id=None, num_envs=1,
                    turret=SimpleNamespace(data=SimpleNamespace(root_pos_w=batch(position))),
                    _compute_turret_frame=lambda: (batch(position), None, batch(forward)),
                    _compute_muzzle_velocity_w=lambda **kw: batch(velocity),
                    _target_pos_w=batch(tp), _target_vel_w=batch(tv), _target_acc_w=batch(ta),
                    _gravity_w=tensor([0., 0., -cfg['gravity_mps2']]), _ballistic_time_samples=times,
                    _ballistic_dt_samples=times[1:]-times[:-1],
                    _mach_table=tensor(cfg['mach_cd_mach']), _cd_table=tensor(cfg['mach_cd_cd']),
                    _speed_of_sound_mps=tensor(cfg['speed_of_sound_mps']), _air_density=tensor(cfg['air_density']),
                    _projectile_mass=tensor(cfg['projectile_mass_kg']),
                    _projectile_area=tensor(np.pi*(cfg['projectile_diameter_m']*.5)**2))
                source._interp_cd_from_mach = lambda mach: namespace['_interp_cd_from_mach'](source, mach)
                source._compute_air_drag_accel_w = lambda vel: namespace['_compute_air_drag_accel_w'](source, vel)
                expected = namespace['_compute_ballistic_metrics'](source)
                actual = ballistic_metrics(position, forward, velocity, tp, tv, ta, config=cfg)
                self.assertAlmostEqual(actual.minimum_distance_m, expected[0].item(), places=10)
                self.assertEqual(actual.hit, expected[1].item())
                self.assertAlmostEqual(actual.closest_time_s, expected[2].item(), places=12)

    def test_actor_matches_original_network_with_saved_normalization_and_state(self):
        torch.set_num_threads(1)
        config = json.loads((ROOT/'shooting_config.json').read_text())
        checkpoint = ROOT.parents[3]/config['decision_checkpoint']
        policy = RecurrentShootingPolicy(checkpoint)
        reference_states = [x.clone() for x in policy.states]
        rng = np.random.default_rng(61)
        for _ in range(12):
            actor = rng.normal(size=28).astype(np.float32)
            # Critic truth placeholders cannot influence actor inference.
            full = torch.from_numpy(np.r_[actor, rng.normal(size=31).astype(np.float32)])[None]
            with torch.inference_mode():
                normalized = policy.model.running_mean_std(full)
                mu, _, _, reference_states = policy.model.a2c_network({'obs': normalized, 'rnn_states': reference_states})
                expected = torch.tanh(mu)[0].numpy()
            np.testing.assert_allclose(policy(actor), expected, atol=1e-7, rtol=0.)
        policy.reset()
        first = policy(actor)
        policy.reset()
        np.testing.assert_array_equal(first, policy(actor))


if __name__ == '__main__':
    unittest.main(verbosity=2)

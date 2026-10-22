"""Check compiled MJCF against the original USD extraction and run a PD smoke test.

Requires only numpy and mujoco. This does not evaluate the locomotion policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from preview_lite3_mjcf import DEFAULT_MODEL, stand_smoke


def quatmat(q):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(q, dtype=float))
    return result.reshape(3, 3)


def axis_rotation(axis, angle):
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    result = np.eye(4)
    result[:3, :3] = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    return result


def validate(model_path, duration):
    root = model_path.parent
    source = json.loads((root / 'source_manifest.json').read_text())
    control = json.loads((root / 'control.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(model_path))
    original = Path(source['source_usd'])
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source['source_sha256'], 'Source USD changed'
    for path, digest in source['source_layer_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'Source layer changed: {path}'
    assert model.nbody - 1 == len(source['bodies']) == 20
    assert model.njnt == 15 and model.nu == 14 and model.nq == 21 and model.nv == 20
    assert model.joint('floating_base').type == mujoco.mjtJoint.mjJNT_FREE
    max_inertia_error = 0.0
    for body in source['bodies']:
        compiled = model.body(body['name'])
        np.testing.assert_allclose(compiled.mass, body['mass'], atol=1e-12)
        np.testing.assert_allclose(compiled.ipos, body['com'], atol=1e-12)
        rotation = quatmat(compiled.iquat)
        tensor = rotation @ np.diag(compiled.inertia) @ rotation.T
        error = float(np.abs(tensor - body['inertia_body']).max())
        max_inertia_error = max(max_inertia_error, error)
        np.testing.assert_allclose(tensor, body['inertia_body'], atol=1e-10)

    for joint in source['joints']:
        if joint['type'] != 'hinge':
            continue
        compiled = model.joint(joint['name'])
        np.testing.assert_allclose(compiled.axis, joint['axis_body'], atol=1e-12)
        np.testing.assert_allclose(compiled.pos, joint['anchor_body'], atol=1e-12)
        np.testing.assert_allclose(compiled.range, joint['limits'], atol=1e-12)
    for i, name in enumerate(control['actuator_order']):
        actuator = model.actuator(name + '_motor')
        assert actuator.id == i and actuator.trnid[0] == model.joint(name).id
        np.testing.assert_allclose(actuator.ctrlrange, [-control['joints'][name]['effort_limit'],
                                                       control['joints'][name]['effort_limit']])

    camera = source['camera']
    cid = model.camera(camera['name']).id
    assert model.cam_bodyid[cid] == model.body(camera['parent_body']).id
    np.testing.assert_allclose(model.cam_pos[cid], camera['position_body'], atol=1e-12)
    np.testing.assert_allclose(quatmat(model.cam_quat[cid]),
                               np.asarray(camera['source']['camera_pose_body_usd'])[:3, :3], atol=1e-12)
    np.testing.assert_array_equal(model.cam_resolution[cid], [camera['width'], camera['height']])
    np.testing.assert_allclose(model.cam_intrinsic[cid, :2].astype(float) / model.cam_sensorsize[cid].astype(float)
                               * model.cam_resolution[cid], [camera['fx'], camera['fy']], atol=1e-3, rtol=0)

    collider_counts = Counter()
    max_mesh_error = 0.0
    rng = np.random.default_rng(20261021)
    directions = np.vstack([np.eye(3), -np.eye(3), rng.normal(size=(26, 3))])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for geometry in source['geometry']:
        shape = geometry['shape']
        for name in geometry['geom_names']:
            compiled = model.geom(name)
            assert compiled.bodyid == model.body(geometry['body']).id
            collision = name.startswith('collision__')
            assert compiled.contype == int(collision)
            assert compiled.conaffinity == (2 if collision else 0)
            assert compiled.group == (3 if collision else 2)
            assert int(compiled.type[0]) == int(getattr(mujoco.mjtGeom, 'mjGEOM_' + shape['type'].upper()))
            if collision:
                collider_counts[geometry['usd_type']] += 1
            if shape['type'] == 'mesh':
                mesh = model.mesh(int(compiled.dataid[0]))
                vertices = model.mesh_vert[int(mesh.vertadr[0]):int(mesh.vertadr[0] + mesh.vertnum[0])]
                baked = vertices @ quatmat(compiled.quat).T + compiled.pos
                obj_vertices = np.array([[float(v) for v in line.split()[1:]]
                                         for line in (root / geometry['mesh_file']).read_text().splitlines()
                                         if line.startswith('v ')])
                # MuJoCo recentres and rotates mesh vertices when compiling.
                # Compare support in 32 directions after undoing that transform.
                error = float(np.abs((baked @ directions.T).max(axis=0)
                                     - (obj_vertices @ directions.T).max(axis=0)).max())
                max_mesh_error = max(max_mesh_error, error)
                assert error < 2e-6, (name, error)
            elif 'fromto' in shape:
                endpoints = np.fromstring(shape['fromto'], sep=' ').reshape(2, 3)
                np.testing.assert_allclose(compiled.pos, endpoints.mean(axis=0), atol=1e-12)
                direction = quatmat(compiled.quat)[:, 2]
                actual = np.array([compiled.pos - compiled.size[1] * direction,
                                   compiled.pos + compiled.size[1] * direction])
                error = min(np.abs(actual - endpoints).max(), np.abs(actual[::-1] - endpoints).max())
                assert error < 1e-10, (name, error)
                np.testing.assert_allclose(compiled.size[0], float(shape['size']), atol=1e-12)
            else:
                np.testing.assert_allclose(compiled.pos, np.fromstring(shape['pos'], sep=' '), atol=1e-12)
                np.testing.assert_allclose(quatmat(compiled.quat),
                                           quatmat(np.fromstring(shape['quat'], sep=' ')), atol=1e-12)
                sizes = np.fromstring(shape['size'], sep=' ')
                np.testing.assert_allclose(compiled.size[:len(sizes)], sizes, atol=1e-12)
    assert dict(collider_counts) == source['collider_types']
    assert sum(collider_counts.values()) == 22 and collider_counts['Capsule'] == 12

    # Reconstruct source USD joint-frame FK independently of compiled body/joint transforms.
    hinges = [j for j in source['joints'] if j['type'] == 'hinge']
    poses = [{j['name']: 0.0 for j in hinges},
             {j['name']: control['joints'][j['name']]['default_position'] for j in hinges}]
    poses += [{j['name']: rng.uniform(*j['limits']) for j in hinges} for _ in range(20)]
    data = mujoco.MjData(model)
    max_position_error = max_rotation_error = 0.0
    for angles in poses:
        mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
        for name, angle in angles.items():
            data.qpos[model.joint(name).qposadr[0]] = angle
        mujoco.mj_forward(model, data)
        base = np.eye(4)
        base[:3, 3] = source['root_position']
        expected = {'TORSO': base}
        pending = list(source['joints'])
        while pending:
            ready = [j for j in pending if j['parent'] in expected]
            assert ready, 'Invalid source joint tree'
            for joint in ready:
                turn = axis_rotation(joint['axis_joint'], angles[joint['name']]) if joint['type'] == 'hinge' else np.eye(4)
                expected[joint['child']] = (expected[joint['parent']] @ np.asarray(joint['parent_joint_pose'])
                                             @ turn @ np.linalg.inv(joint['child_joint_pose']))
                pending.remove(joint)
        for name, transform in expected.items():
            observed = data.body(name)
            pe = float(np.abs(observed.xpos - transform[:3, 3]).max())
            re = float(np.abs(observed.xmat.reshape(3, 3) - transform[:3, :3]).max())
            max_position_error, max_rotation_error = max(max_position_error, pe), max(max_rotation_error, re)
    assert max_position_error < 1e-9 and max_rotation_error < 1e-9
    smoke = stand_smoke(model, control, duration)
    assert not smoke['solver_warnings'], smoke
    assert smoke['maximum_contacts'] > 0, smoke
    assert smoke['maximum_tilt_degrees'] < 30, smoke
    assert smoke['base_height_range_m'][0] > 0.20, smoke
    report = {
        'passed': True, 'mujoco_version': mujoco.__version__,
        'source_usd': source['source_usd'], 'source_sha256': source['source_sha256'],
        'source_unchanged': True, 'source_layers_checked': len(source['source_layer_sha256']),
        'rigid_bodies': model.nbody - 1, 'hinges': model.njnt - 1,
        'total_mass_kg': float(model.body_mass.sum()), 'collider_types': dict(collider_counts),
        'inferred_foot_inertias': [b['name'] for b in source['bodies'] if b['inferred_uniform_sphere']],
        'maximum_inertia_tensor_error': max_inertia_error,
        'maximum_compiled_mesh_support_error_m': max_mesh_error,
        'kinematic_poses_checked': len(poses),
        'maximum_body_position_error_m': max_position_error,
        'maximum_body_rotation_matrix_error': max_rotation_error,
        'mounted_camera': {'name': camera['name'], 'resolution': [camera['width'], camera['height']],
                           'parent_body': camera['parent_body'], 'source': camera['source']['parameters_python']},
        'stand_pd_smoke_test': smoke,
        'scope': 'USD extraction versus native MuJoCo compilation and short deterministic PD stance only.',
        'not_validated': ['Learned-policy locomotion', 'Exact PhysX contact/drive equivalence',
                          'PhysX auto-computed foot inertia measured at runtime',
                          'External camera housing geometry and Isaac RTX image appearance',
                          'OSTrack/LK integration and camera update scheduling',
                          'Four unresolved foot visual mesh references',
                          'Delayed actuator behavior and joint velocity limits'],
    }
    report_path = root / 'validation.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--duration', type=float, default=5.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be positive')
    validate(args.model.resolve(), args.duration)

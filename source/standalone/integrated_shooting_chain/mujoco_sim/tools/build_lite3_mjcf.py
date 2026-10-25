"""Build Lite3 + turret MJCF directly from the original USD, without URDF.

Run with the Isaac Lab Python environment. The local USD bindings are located
automatically; MuJoCo compilation/validation runs separately in mujoco_test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

WORKSPACE = Path(__file__).resolve().parents[5]


def bootstrap_usd():
    if os.environ.get('LITE3_MJCF_USD_READY') == '1':
        return
    sim = (WORKSPACE / '_isaac_sim').resolve()
    candidates = sorted((sim / 'extscache').glob('omni.usd.libs-*cp311*'))
    if not candidates:
        raise RuntimeError('Cannot locate the local Isaac Sim USD Python bindings.')
    usd = candidates[-1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(usd) + os.pathsep + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = os.pathsep.join([
        str(Path(sys.prefix) / 'lib'), str(usd / 'bin'), str(sim / 'kit/lib'),
        env.get('LD_LIBRARY_PATH', ''),
    ])
    env['LITE3_MJCF_USD_READY'] = '1'
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


bootstrap_usd()
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_calibration import camera_profile_from_snapshot

POLICY_JOINTS = [f'{leg}_{joint}' for leg in ('FL', 'FR', 'HL', 'HR')
                 for joint in ('HipX_joint', 'HipY_joint', 'Knee_joint')]
MOTOR_JOINTS = POLICY_JOINTS + ['yaw_joint', 'pitch_joint']


def numbers(value):
    return ' '.join(format(float(x), '.16g') for x in np.asarray(value).reshape(-1))


def rotation(quat):
    return Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()


def wxyz(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    if q[3] < 0:
        q = -q
    return np.roll(q, 1)


def joint_pose(joint, side):
    matrix = np.eye(4)
    matrix[:3, 3] = getattr(joint, f'GetLocalPos{side}Attr')().Get()
    matrix[:3, :3] = rotation(getattr(joint, f'GetLocalRot{side}Attr')().Get())
    return matrix


def default_angle(name):
    return -0.8 if 'HipY' in name else 1.6 if 'Knee' in name else 0.0


def tracking_camera_profile(stage):
    """Keep rebuilds on the measured ordinary evaluator's calibration.

    Reading the bundled runtime capture does not depend on resolving the
    remote SG2 reference or on the collector's different fallback focal length.
    """
    if not stage.GetPrimAtPath('/World/Lite3/TORSO'):
        raise ValueError('Missing source TORSO camera parent')
    return camera_profile_from_snapshot()


def mesh_export(prim, transform, path):
    """Bake the complete visual transform, retaining reflection and normals."""
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    baked = points @ transform[:3, :3].T + transform[:3, 3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    assert counts.sum() == len(indices)
    flip = (np.linalg.det(transform[:3, :3]) < 0) != (mesh.GetOrientationAttr().Get() == 'leftHanded')
    raw_normals = mesh.GetNormalsAttr().Get()
    normals = np.asarray(raw_normals, dtype=float) if raw_normals is not None else np.empty((0, 3))
    interpolation = mesh.GetNormalsInterpolation()
    if len(normals):
        assert interpolation in ('faceVarying', 'vertex', 'varying')
        normals = normals @ np.linalg.inv(transform[:3, :3])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    triangles = []
    cursor = 0
    for count in counts:
        order = list(range(cursor, cursor + int(count)))
        if flip:
            order.reverse()
        for i in range(1, len(order) - 1):
            corners = [order[0], order[i], order[i + 1]]
            triangles.append((indices[corners], np.asarray(corners)))
        cursor += int(count)
    with path.open('w') as stream:
        stream.write('# Visual vertices baked into the original USD rigid-body frame.\n')
        np.savetxt(stream, baked, fmt='v %.12g %.12g %.12g')
        if len(normals):
            np.savetxt(stream, normals, fmt='vn %.12g %.12g %.12g')
        for vertices, corners in triangles:
            if len(normals):
                normal_ids = corners if interpolation == 'faceVarying' else vertices
                fields = [f'{v + 1}//{n + 1}' for v, n in zip(vertices, normal_ids)]
            else:
                fields = [str(v + 1) for v in vertices]
            stream.write('f ' + ' '.join(fields) + '\n')
    return {'vertices': len(points), 'triangles': len(triangles), 'reflection_baked': bool(flip)}


def primitive(prim, transform):
    linear = transform[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    proper = linear / scale
    if not np.allclose(proper.T @ proper, np.eye(3), atol=1e-6):
        raise ValueError(f'Shear in primitive: {prim.GetPath()}')
    if np.linalg.det(proper) < 0:
        proper[:, 0] *= -1
    attrs = {'pos': numbers(transform[:3, 3]), 'quat': numbers(wxyz(proper))}
    if prim.IsA(UsdGeom.Cube):
        attrs.update(type='box', size=numbers(scale * float(prim.GetAttribute('size').Get()) / 2))
    elif prim.IsA(UsdGeom.Sphere):
        radii = scale * float(prim.GetAttribute('radius').Get())
        attrs.update(type='sphere' if np.allclose(radii, radii[0]) else 'ellipsoid',
                     size=numbers([radii[0]]) if np.allclose(radii, radii[0]) else numbers(radii))
    elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
        axis = 'XYZ'.index(str(prim.GetAttribute('axis').Get()))
        radial_axes = [i for i in range(3) if i != axis]
        if not np.isclose(scale[radial_axes[0]], scale[radial_axes[1]], atol=1e-8):
            raise ValueError(f'Unequal radial scale: {prim.GetPath()}')
        if prim.IsA(UsdGeom.Capsule) and not np.allclose(scale, scale[0], atol=1e-8):
            raise ValueError(f'Nonuniform capsule scale: {prim.GetPath()}')
        half_segment = np.eye(3)[axis] * float(prim.GetAttribute('height').Get()) / 2
        endpoints = np.asarray([-half_segment, half_segment]) @ linear.T + transform[:3, 3]
        radius = float(prim.GetAttribute('radius').Get()) * scale[radial_axes[0]]
        attrs = {'type': 'capsule' if prim.IsA(UsdGeom.Capsule) else 'cylinder',
                 'fromto': numbers(endpoints), 'size': numbers([radius])}
    else:
        raise ValueError(f'Unsupported Gprim {prim.GetPath()}: {prim.GetTypeName()}')
    return attrs


def build(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets_dir = output / 'assets'
    assets_dir.mkdir(exist_ok=True)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f'Cannot open {source}')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('This converter expects the source model in metres with Z up.')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    prims = [p for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
             if str(p.GetPath()).startswith('/World/Lite3/')]
    body_prims = {p.GetName(): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert len(body_prims) == 20
    cache = UsdGeom.XformCache()
    world_body = {name: np.asarray(cache.GetLocalToWorldTransform(p), dtype=float).T for name, p in body_prims.items()}
    joints = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if not joint.GetJointEnabledAttr().Get():
            continue
        parent_targets, child_targets = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        assert len(parent_targets) == len(child_targets) == 1
        parent, child = stage.GetPrimAtPath(parent_targets[0]).GetName(), stage.GetPrimAtPath(child_targets[0]).GetName()
        assert parent in body_prims and child in body_prims
        t0, t1 = joint_pose(joint, 0), joint_pose(joint, 1)
        rest = t0 @ np.linalg.inv(t1)
        assert np.allclose(world_body[parent] @ rest, world_body[child], atol=2e-6), prim.GetPath()
        item = {'name': prim.GetName(), 'source_path': str(prim.GetPath()), 'parent': parent, 'child': child,
                'type': 'hinge' if prim.IsA(UsdPhysics.RevoluteJoint) else 'fixed',
                'parent_joint_pose': t0.tolist(), 'child_joint_pose': t1.tolist(), 'rest_pose': rest.tolist()}
        if item['type'] == 'hinge':
            revolute = UsdPhysics.RevoluteJoint(prim)
            axis = np.eye(3)['XYZ'.index(str(revolute.GetAxisAttr().Get()))]
            item.update(axis_joint=axis.tolist(), axis_body=(t1[:3, :3] @ axis).tolist(),
                        anchor_body=t1[:3, 3].tolist(),
                        limits=np.deg2rad([revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get()]).tolist())
        else:
            assert prim.IsA(UsdPhysics.FixedJoint), prim.GetPath()
        joints.append(item)
    assert len(joints) == 19
    incoming = {j['child']: j for j in joints}
    assert len(incoming) == 19 and set(body_prims) - set(incoming) == {'TORSO'}

    model = ET.Element('mujoco', model='lite3_with_turret_from_original_usd')
    ET.SubElement(model, 'compiler', angle='radian', meshdir='assets', inertiafromgeom='false',
                  fusestatic='false', balanceinertia='false', autolimits='true', alignfree='false')
    ET.SubElement(model, 'option', timestep='0.005', gravity='0 0 -9.81', integrator='implicitfast', iterations='50')
    defaults = ET.SubElement(model, 'default')
    ET.SubElement(defaults, 'joint', damping='0', armature='0', frictionloss='0')
    ET.SubElement(defaults, 'geom', friction='1 0 0', condim='3', solref='0.02 1', solimp='0.9 0.95 0.001')
    asset_xml = ET.SubElement(model, 'asset')
    world_xml = ET.SubElement(model, 'worldbody')
    body_xml = {}
    native_joint_order = []

    def add_body(name, parent):
        matrix = np.eye(4) if name == 'TORSO' else np.asarray(incoming[name]['rest_pose'])
        if name == 'TORSO':
            matrix[2, 3] = 0.35
        element = ET.SubElement(parent, 'body', name=name, pos=numbers(matrix[:3, 3]), quat=numbers(wxyz(matrix[:3, :3])))
        body_xml[name] = element
        if name == 'TORSO':
            ET.SubElement(element, 'freejoint', name='floating_base')
        elif incoming[name]['type'] == 'hinge':
            j = incoming[name]
            ET.SubElement(element, 'joint', name=j['name'], type='hinge', axis=numbers(j['axis_body']),
                          pos=numbers(j['anchor_body']), range=numbers(j['limits']), limited='true')
            native_joint_order.append(j['name'])
        for j in joints:
            if j['parent'] == name:
                add_body(j['child'], element)

    add_body('TORSO', world_xml)
    # Sensor mount marker only: no added rigid body, inertia or contact shape.
    ET.SubElement(body_xml['TORSO'], 'site', name='array_lidar', pos='0.23 0 0.073',
                  quat='1 0 0 0', size='0.004', rgba='0.1 0.8 1 1', group='5')
    geometry = []
    for prim in prims:
        if not prim.IsA(UsdGeom.Gprim):
            continue
        ancestor = prim
        while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            ancestor = ancestor.GetParent()
        if not ancestor:
            continue
        body = ancestor.GetName()
        collision_api = UsdPhysics.CollisionAPI(prim)
        collision = bool(collision_api) and collision_api.GetCollisionEnabledAttr().Get()
        imageable = UsdGeom.Imageable(prim)
        visible = imageable.ComputeVisibility() != 'invisible' and imageable.ComputePurpose() in ('default', 'render')
        if not visible and not collision:
            continue
        local = np.linalg.inv(world_body[body]) @ np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        relative = str(prim.GetPath()).removeprefix(str(body_prims[body].GetPath()) + '/')
        label = re.sub(r'[^a-zA-Z0-9_]', '_', body + '__' + relative)
        entry = {'source_path': str(prim.GetPath()), 'body': body, 'usd_type': prim.GetTypeName(),
                 'body_pose': local.tolist(), 'visible': bool(visible), 'collision': bool(collision), 'geom_names': []}
        if prim.IsA(UsdGeom.Mesh):
            filename = label + '.obj'
            mesh_data = mesh_export(prim, local, assets_dir / filename)
            ET.SubElement(asset_xml, 'mesh', name=label, file=filename)
            shape = {'type': 'mesh', 'mesh': label}
            entry.update(mesh_data, mesh_asset=label, mesh_file='assets/' + filename)
        else:
            shape = primitive(prim, local)
        entry['shape'] = shape
        if visible:
            geom_name = 'visual__' + label
            ET.SubElement(body_xml[body], 'geom', name=geom_name, **shape, group='2',
                          contype='0', conaffinity='0', rgba='0.72 0.74 0.78 1')
            entry['geom_names'].append(geom_name)
        if collision:
            geom_name = 'collision__' + label
            ET.SubElement(body_xml[body], 'geom', name=geom_name, **shape, group='3',
                          contype='1', conaffinity='2', rgba='0.1 0.7 0.2 0.35')
            entry['geom_names'].append(geom_name)
        geometry.append(entry)

    bodies = []
    for name, prim in body_prims.items():
        api = UsdPhysics.MassAPI(prim)
        mass = float(api.GetMassAttr().Get())
        com = np.asarray(api.GetCenterOfMassAttr().Get(), dtype=float)
        diagonal = np.asarray(api.GetDiagonalInertiaAttr().Get(), dtype=float)
        principal = rotation(api.GetPrincipalAxesAttr().Get())
        inferred = False
        if not np.isfinite(com).all() or not np.any(diagonal):
            shapes = [g for g in geometry if g['body'] == name and g['collision']]
            assert len(shapes) == 1 and shapes[0]['shape']['type'] == 'sphere', name
            sphere = shapes[0]['shape']
            radius = float(sphere['size'])
            com = np.fromstring(sphere['pos'], sep=' ')
            diagonal = np.repeat(0.4 * mass * radius**2, 3)
            principal = np.eye(3)
            inferred = True
        tensor = principal @ np.diag(diagonal) @ principal.T
        assert mass > 0 and np.linalg.eigvalsh(tensor).min() > 0
        full = tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        inertial = ET.Element('inertial', mass=numbers([mass]), pos=numbers(com), fullinertia=numbers(full))
        body_xml[name].insert(0, inertial)
        bodies.append({'name': name, 'source_path': str(prim.GetPath()), 'mass': mass, 'com': com.tolist(),
                       'inertia_body': tensor.tolist(), 'inferred_uniform_sphere': inferred})

    actuator_xml = ET.SubElement(model, 'actuator')
    control = {'physics_dt': 0.005, 'control_decimation': 4, 'control_dt': 0.02,
               'provenance': 'Leg control settings from outputs/urdf_walk_validation_20261021/original/effective_env.yaml; turret gains from original USD.',
               'notes': ['This file specifies the controller interface; robot.xml contains torque motors only.',
                         'The preview controller implements deterministic PD with no actuator delay.',
                         'Velocity limits are metadata here; the preview controller does not enforce them.',
                         'Turret PD is explicit in this preview, unlike the original PhysX implicit drives.'],
               'policy_joint_order': POLICY_JOINTS, 'actuator_order': MOTOR_JOINTS,
               'policy_observation_size': 45, 'policy_action_size': 12,
               'angular_velocity_observation_scale': 0.25, 'joint_velocity_observation_scale': 0.05,
               'clip_observations': 100.0, 'clip_actions': 100.0,
               'policy_output': 'target_position = default_position + action_scale * action',
               'pd_torque': 'clip(kp * (target_position - position) - kd * velocity, -effort_limit, effort_limit)',
               'leg_delay_range_physics_steps': [0, 5], 'joints': {}}
    by_name = {j['name']: j for j in joints}
    for name in MOTOR_JOINTS:
        if name in POLICY_JOINTS:
            effort = 36.0 if 'Knee' in name else 24.0
            kp, kd = 30.0, 1.0
            scale = 0.125 if 'HipX' in name else 0.25
        else:
            joint_prim = stage.GetPrimAtPath(by_name[name]['source_path'])
            effort = float(joint_prim.GetAttribute('drive:angular:physics:maxForce').Get())
            kp = float(joint_prim.GetAttribute('drive:angular:physics:stiffness').Get())
            kd = float(joint_prim.GetAttribute('drive:angular:physics:damping').Get())
            scale = 0.0
        ET.SubElement(actuator_xml, 'motor', name=name + '_motor', joint=name, gear='1',
                      ctrllimited='true', ctrlrange=numbers([-effort, effort]))
        control['joints'][name] = {'default_position': default_angle(name), 'kp': kp, 'kd': kd,
                                  'effort_limit': effort, 'action_scale': scale,
                                  'velocity_limit': 17.3 if 'Knee' in name else 26.2 if name in POLICY_JOINTS else None}
    ET.SubElement(ET.SubElement(model, 'keyframe'), 'key', name='stand',
                  qpos=numbers([0, 0, 0.35, 1, 0, 0, 0] + [default_angle(n) for n in native_joint_order]))
    camera_profile = tracking_camera_profile(stage)
    ET.SubElement(body_xml['TORSO'], 'camera', name=camera_profile['name'], mode='fixed',
                  pos=numbers(camera_profile['position_body']), quat=numbers(camera_profile['quaternion_body_wxyz']),
                  sensorsize=numbers(camera_profile['sensor_size']), resolution=f"{camera_profile['width']} {camera_profile['height']}",
                  focalpixel=numbers([camera_profile['fx'], camera_profile['fy']]),
                  principalpixel=numbers([camera_profile['width']/2-camera_profile['cx'],
                                          camera_profile['cy']-camera_profile['height']/2]), ipd='0')
    (output / 'camera_config.json').write_text(json.dumps(camera_profile, indent=2) + '\n')
    ET.indent(model, space='  ')
    ET.ElementTree(model).write(output / 'robot.xml', encoding='utf-8', xml_declaration=True)

    scene = ET.Element('mujoco', model='lite3_with_turret_flat_scene')
    ET.SubElement(scene, 'include', file='robot.xml')
    ET.SubElement(scene, 'statistic', center='0 0 0.25', extent='1.2')
    visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='720', azimuth='135', elevation='-20')
    ET.SubElement(visual, 'map', znear=numbers([camera_profile['near'] / 1.2]),
                  zfar=numbers([camera_profile['far'] / 1.2]))
    scene_asset = ET.SubElement(scene, 'asset')
    ET.SubElement(scene_asset, 'texture', name='ground_checker', type='2d', builtin='checker',
                  rgb1='0.15 0.18 0.2', rgb2='0.25 0.28 0.3', width='512', height='512')
    ET.SubElement(scene_asset, 'material', name='ground_material', texture='ground_checker',
                  texrepeat='10 10', texuniform='true', reflectance='0.05')
    world = ET.SubElement(scene, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -2 3', dir='0 0 -1', directional='true')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='0 0 0.1', pos='0 0 0',
                  contype='2', conaffinity='1', material='ground_material', friction='1 0 0', condim='3')
    ET.indent(scene, space='  ')
    ET.ElementTree(scene).write(output / 'scene.xml', encoding='utf-8', xml_declaration=True)
    (output / 'control.json').write_text(json.dumps(control, indent=2) + '\n')
    errors = sorted({str(e) for p in prims for e in p.GetPrimIndex().localErrors})
    layer_hashes = {str(Path(layer.realPath).resolve()): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                    for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    manifest = {'source_usd': str(source), 'source_sha256': source_hash, 'source_layer_sha256': layer_hashes, 'used_urdf': False,
                'source_root_body': 'TORSO', 'root_position': [0, 0, 0.35],
                'bodies': bodies, 'joints': joints, 'geometry': geometry,
                'native_joint_order': native_joint_order, 'source_composition_errors': errors,
                'camera': camera_profile,
                'collider_types': dict(Counter(g['usd_type'] for g in geometry if g['collision'])),
                'notes': ['Rigid-body frames, full inertia tensors and local COM are retained from the original USD.',
                          'All active UsdPhysics collision primitives are included, including Capsule.',
                          'Guide-only shapes without CollisionAPI are not promoted into colliders.',
                          'Mirrors are baked into visual vertices and winding; source USD is not modified.',
                          'Source world/ground/render content is excluded; scene.xml defines one plane at z=0.',
                          'Four auto-computed foot COM/inertias are derived from their sole spherical collision shape.',
                          'PhysX contact solver, friction-combine rules and drive implementation have no exact MJCF translation.',
                          'Camera calibration comes from the measured ordinary Isaac evaluation render proxy; appearance differs from RTX.']}
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    print(json.dumps({'output': str(output), 'bodies': len(bodies), 'hinges': len(native_joint_order),
                      'colliders': sum(g['collision'] for g in geometry), 'collider_types': manifest['collider_types'],
                      'total_mass': sum(b['mass'] for b in bodies), 'source_unchanged': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(WORKSPACE / 'source/standalone/test7_2_weight/lite3_with_turret.usd'))
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'mjcf/lite3_with_turret'))
    build(parser.parse_args())

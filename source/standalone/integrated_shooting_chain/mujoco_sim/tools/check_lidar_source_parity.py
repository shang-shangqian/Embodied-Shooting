"""Compare saved MuJoCo demo outputs with the actual IsaacLab source functions.

Run using isaac_lab_232 Python (Torch + SciPy). Only the two source functions are
loaded via AST, so no SimulationApp, GPU or IsaacLab environment is started.
This checks the pattern and postprocessing, not cross-engine mesh intersections.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]


def source_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    for argument in function.args.args:
        argument.annotation = None
    function.returns = None
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


def quat_apply(quat, vec):
    imaginary = quat[:, 1:]
    temp = 2 * torch.cross(imaginary, vec, dim=-1)
    return vec + quat[:, :1] * temp + torch.cross(imaginary, temp, dim=-1)


def check(folder, pattern, postprocess):
    report = json.loads((folder / 'report.json').read_text())
    inputs = np.load(folder / 'last_scan.npz')
    config = report['config']
    cfg = SimpleNamespace(
        channels=max(1, round(config['vertical_fov_deg'] / config['vertical_resolution_deg']) + 1),
        vertical_fov_range=(-config['vertical_fov_deg']/2, config['vertical_fov_deg']/2),
        horizontal_fov_range=(-config['horizontal_fov_deg']/2, config['horizontal_fov_deg']/2),
        horizontal_res=config['horizontal_resolution_deg'])
    _, directions = pattern(cfg, 'cpu')
    local = inputs['directions_w'] @ inputs['rotation_w']
    np.testing.assert_allclose(local, directions.numpy(), atol=4e-7, rtol=0)
    pattern_error = float(np.abs(local - directions.numpy()).max())
    optical_to_lidar = np.array([[0., 0, 1], [-1., 0, 0], [0., -1, 0]])
    camera_rotation = inputs['rotation_w'] @ optical_to_lidar
    xyzw = Rotation.from_matrix(camera_rotation).as_quat()
    camera_quat = torch.tensor(np.roll(xyzw, 1)[None], dtype=torch.float32)
    camera_pos = torch.tensor(inputs['origin_w'][None], dtype=torch.float32)
    camera_matrix = torch.tensor(camera_rotation, dtype=torch.float32)
    env = SimpleNamespace(
        cfg=SimpleNamespace(enable_dynamic_target_raycast_debug=True, debug_diagnostics=False,
            dynamic_target_raycast_horizontal_resolution_deg=config['horizontal_resolution_deg'],
            dynamic_target_raycast_vertical_resolution_deg=config['vertical_resolution_deg'],
            dynamic_target_raycast_debug_every=20, target_sphere_radius=.35),
        _dynamic_target_raycast_sensor=SimpleNamespace(update=lambda **kw: None,
            data=SimpleNamespace(pos_w=torch.tensor(inputs['parent_position_w'][None], dtype=torch.float32),
                                 ray_hits_w=torch.tensor(inputs['ray_hits_w'][None], dtype=torch.float32))),
        _vision_camera=object(), device='cpu', physics_dt=.005,
        _external_raycast_yaw=torch.zeros(1), _external_raycast_pitch=torch.zeros(1),
        _raycast_hit_valid=torch.zeros(1, dtype=torch.bool),
        _raycast_hit_distance=torch.full((1,), float('inf')),
        _raycast_expected_surface_distance=torch.full((1,), float('inf')),
        _get_runtime_camera_pose_w=lambda: (camera_pos, camera_quat),
        _rotate_world_to_local=lambda quat, vector: vector @ camera_matrix)
    postprocess(env)
    row = report['samples'][-1]
    assert abs(float(inputs['time']) - row['time_s']) < 1e-10
    assert bool(env._raycast_hit_valid[0]) == row['valid']
    differences = {}
    for field, tensor in [('center_distance_m', env._raycast_hit_distance),
                          ('surface_distance_m', env._raycast_expected_surface_distance),
                          ('refined_yaw_rad', env._raycast_refined_yaw),
                          ('refined_pitch_rad', env._raycast_refined_pitch)]:
        if row[field] is None:
            assert torch.isinf(tensor).all()
        else:
            expected = float(tensor[0])
            tolerance = 1e-5 if field.endswith('_m') else 1e-6
            np.testing.assert_allclose(row[field], expected, atol=tolerance, rtol=0)
            differences[field] = abs(row[field] - expected)
    return {'folder': str(folder), 'passed': True, 'pattern_maximum_component_error': pattern_error,
            'postprocessing_absolute_differences': differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='*', type=Path, default=[WORKSPACE / 'outputs/usd_mjcf_validation_20261021/lidar',
                                                                WORKSPACE / 'outputs/usd_mjcf_validation_20261021/lidar_snapshot'])
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'outputs/usd_mjcf_validation_20261021/lidar_source_parity.json')
    args = parser.parse_args()
    sim_source = (WORKSPACE / 'source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns.py')
    if not sim_source.is_file():
        import importlib.util
        package = importlib.util.find_spec('isaaclab')
        sim_source = Path(next(iter(package.submodule_search_locations))) / 'sensors/ray_caster/patterns/patterns.py'
    env_source = WORKSPACE / 'source/standalone/test7_2_weight/turret_aim_env_gt.py'
    namespace = {'torch': torch, 'math': math, 'quat_apply': quat_apply, 'traceback': traceback}
    torch.set_num_threads(1)
    pattern = source_function(sim_source, 'lidar_pattern', namespace)
    postprocess = source_function(env_source, '_debug_raycast_dynamic_target', namespace)
    result = {'passed': True, 'scope': 'Original source functions versus saved MuJoCo demo pattern/postprocessing only.',
              'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (sim_source, env_source)},
              'cases': [check(folder.resolve(), pattern, postprocess) for folder in args.folders]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

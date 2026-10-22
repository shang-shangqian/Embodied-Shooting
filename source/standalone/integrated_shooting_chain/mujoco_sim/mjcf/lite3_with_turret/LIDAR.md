# MuJoCo array LiDAR

The sensor implementation is `source/standalone/integrated_shooting_chain/mujoco_sim/mujoco_lidar.py`.
It runs with NumPy and MuJoCo 3.16.0; it does not require Isaac Sim, Torch, Warp,
a renderer or a GPU. Its default profile follows **test7_2_weight**, not the
different settings in test11 or an unspecified command-line override.

## Run the example

From `/home/jby/IsaacLab-2.3.2`:

```bash
conda activate mujoco_test
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/demo_mujoco_lidar.py --duration 20
```

The viewer shows the robot held at its stand pose, a red moving sphere, and up
to 128 green lines from actual hits. Every measurement uses the full array.
This is a kinematic sensor example, not walking or closed-loop aiming. The
synthetic camera optical frame is co-located with the LiDAR. To use the actual
mounted RGB camera instead, see `demo_mujoco_camera.py --lidar` in
[CAMERA.md](CAMERA.md). The supplied visual direction in this LiDAR-only demo stays at yaw/pitch = 0/0,
without using the target's true position to steer the sensor or set the prior.

Headless example, without any graphics initialization:

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/demo_mujoco_lidar.py --headless --duration 2
```

The results are saved under `outputs/usd_mjcf_validation_20261021/lidar/`:

- `report.json`: per-frame range, validity, angles, timing and diagnostic errors.
- `last_scan.npz`: full final ray directions, distances, hit points and geom IDs.

The sphere's true distance is recorded only for evaluation. Measurements use ray
intersections and the supplied angle prior. The centre estimate keeps the
original heuristic; it is not exact sphere fitting. The initial 100-frame run
had 100 valid frames, a maximum centre-estimation error of about 0.115 m and a
median native scan time around 10 ms. These timings exclude postprocessing,
rendering and application scheduling, and do not establish real-time operation.

Static preview and reproducible screenshot command:

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 python source/standalone/integrated_shooting_chain/mujoco_sim/tools/demo_mujoco_lidar.py --snapshot outputs/usd_mjcf_validation_20261021/lidar_preview.png --target-distance 3 --output outputs/usd_mjcf_validation_20261021/lidar_snapshot
```

## Configuration and data

`lidar_config.json` specifies a fixed array attached to TORSO at
`(0.23, 0, 0.073)` with identity mount rotation. It uses full body orientation,
including roll and pitch. The `array_lidar` site in robot.xml is a mount marker
only; the Python module performs the measurements using the configured offset.

The default field of view is 180 by 40 degrees, with requested angular spacing
0.2 by 0.3 degrees. As in IsaacLab, the generated horizontal linspace includes
both endpoints unless the horizontal span is 360 degrees. Vertical channels are
`round(vertical_fov / vertical_resolution) + 1`. This gives 134 rows by 901
columns, **120,734 rays**, in vertical-major order. Math uses float64, whereas the
original Torch pattern uses float32, so bitwise identity is not expected.

Only **geom group 4** is queried. The robot's appearance/collision geoms and
ground are excluded. This reproduces the source target-only mesh-list behavior:
the ground and other excluded objects do not occlude the targets. Selected group
4 objects can occlude each other. Place additional desired ray surfaces in this
group, or change `target_geom_group` deliberately. The visualizer explicitly
enables this group, which MuJoCo normally hides by default.

The demo target has `contype=0 conaffinity=0`, allowing detection without contact
forces. Ray selection is independent of physical contact masks. Mesh and
primitive targets are supported by the native ray query. Native analytic sphere
intersections need not match a triangulated USD sphere exactly.

The reusable API is:

```python
# Add source/standalone/integrated_shooting_chain/mujoco_sim to sys.path when running standalone.
from mujoco_lidar import ArrayLidarConfig, MujocoArrayLidar

config = ArrayLidarConfig.from_json("path/to/lidar_config.json")
lidar = MujocoArrayLidar(model, config)
scan = lidar.scan(data)

estimate = lidar.estimate_target(
    scan,
    yaw=visual_yaw_radians,
    pitch=visual_pitch_radians,
    camera_position_w=camera_position,
    camera_rotation_w=optical_to_world_rotation,
    target_radius=0.35,
)
```

The camera matrix maps **ROS optical** axes (right, down, forward) into world
coordinates. MuJoCo camera axes require conversion; do not pass them directly.
The method requires a visual angle input and does not fall back to target truth.
An optional `target_geom_id` filters already-visible returns for target
association; it does not query through nearer selected targets.

`scan()` calls `mj_forward` to refresh geometry after target/base motion. Set
`update_kinematics=False` only if you have already refreshed the derived state
for the current qpos/mocap values. `mj_step` alone can leave derived transforms
corresponding to the earlier integration state. Independent environments should
use separate MjData objects. There is no GPU batch-environment implementation.

Returned quantities use metres and radians:

| Quantity | Meaning |
| --- | --- |
| `scan.origin_w` | Physical ray origin, including mount offset |
| `scan.parent_position_w` | TORSO origin; original RayCaster `data.pos_w` reference |
| `scan.distances` | Per-ray surface distance from the physical ray origin |
| `scan.ray_hits_w` | World hit positions; infinity on misses |
| `scan.geom_ids` | Native MuJoCo geom IDs; -1 on misses |
| `scan.valid` | Per-ray hit and range validity |
| `estimate.center_distance` | Estimated target-centre distance from camera |
| `estimate.surface_distance` | Mean surface distance using the configured postprocessing reference |
| `estimate.refined_yaw/pitch` | Estimated centre angles in ROS optical coordinates |
| `estimate.hit_valid` | Whether the visual-window-filtered estimate succeeded |

The raw scan explicitly enforces the configured 0.3--165 m interval. Near
clipping invalidates the first return; it does not search through it. The old
test7_2_weight postprocessor declared a minimum range but did not apply that
check. Its normal distant-target operating range is unaffected by this change.

There is a source-specific distinction: the original RayCaster's `data.pos_w`
is the parent TORSO position, while its ray starts include the mount offset.
To preserve the original postprocessing, `postprocess_reference` defaults to
`parent`: cone selection, median clustering and the returned surface-distance
summary use that reference. Raw `scan.distances` always use the actual ray
origin. Set `postprocess_reference` to `lidar` only when deliberately changing
the estimator's distance convention.

## Validation

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tests/test_mujoco_lidar.py
```

The tests check analytic sphere distances, moving sphere/mesh targets, ray
ordering and the 360-degree seam, roll plus mount transforms, range clipping,
group selection, target association/occlusion, optical angles and invalid-return
handling. They also distinguish physical ray range from the legacy parent-origin
postprocessing range.

After running the headless and screenshot examples, the source-parity check can
be run using the Isaac Python environment:

```bash
/home/jby/miniconda3/envs/isaac_lab_232/bin/python source/standalone/integrated_shooting_chain/mujoco_sim/tools/check_lidar_source_parity.py
```

This loads the actual `lidar_pattern` and
`_debug_raycast_dynamic_target` function bodies without starting Isaac Sim, then
compares them against the saved scans and estimates. Results are in
`outputs/usd_mjcf_validation_20261021/lidar_source_parity.json`.
It validates the pattern and postprocessing, not full sensor parity between two
running engines or integration with the real camera/perception/control chain.

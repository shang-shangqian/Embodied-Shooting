# Lite3 + turret: original USD to MJCF

Source: `source/standalone/test7_2_weight/lite3_with_turret.usd`.
This conversion reads the original USD and its composed layers directly. It does
not use the earlier URDF or URDF-imported USD, and does not modify the source.

## Files

| File | Purpose |
| --- | --- |
| `scene.xml` | Load this in MuJoCo: robot and one ground plane at z = 0 |
| `robot.xml` | Robot definition; include it in another MJCF scene if needed |
| `assets/*.obj` | Visual meshes in the original rigid-body frames |
| `control.json` | Joint order, PD gains, effort limits and policy interface metadata |
| `source_manifest.json` | Extracted USD bodies, joints, geometry, source layer hashes and unresolved references |
| `validation.json` | Results from the native MuJoCo checks and a short PD test |
| `lidar_config.json` | Array LiDAR mount, scan grid, range and target group |
| `scene_lidar_demo.xml` | Robot plus a scripted moving, ray-detectable target |
| `camera_config.json` | test_lk_ostrack camera intrinsics, TORSO mount and source provenance |

The MuJoCo array LiDAR is documented in [LIDAR.md](LIDAR.md). The mounted RGB
camera, image capture and same-frame camera/LiDAR example are documented in
[CAMERA.md](CAMERA.md). These examples do not run the walking/aiming policy.
Locomotion rollout/video commands and the matched policy interface are described
in [WALKING.md](WALKING.md).

## Preview and validation

Run these commands from `/home/jby/IsaacLab-2.3.2`:

```bash
conda activate mujoco_test
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/preview_lite3_mjcf.py
```

The default preview displays the `stand` keyframe without advancing physics.
Close the viewer to exit. Display the collision primitives as green shapes with:

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/preview_lite3_mjcf.py --show-collisions
```

Apply a deterministic joint PD controller and advance physics with:

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/preview_lite3_mjcf.py --simulate
```

This holds the default joint targets; it is not a balance controller or the
trained walking policy. For CPU validation without a display:

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/validate_lite3_mjcf.py
```

Static render previews from this build are saved in
`outputs/usd_mjcf_validation_20261021/stand.png` and `collisions.png`.
To regenerate a preview without a desktop window, using Mesa software rendering:

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 python source/standalone/integrated_shooting_chain/mujoco_sim/tools/preview_lite3_mjcf.py --snapshot outputs/usd_mjcf_validation_20261021/stand.png
```

The validator requires the original source layers to remain at their recorded
paths. The MJCF scene and its OBJ assets can be loaded independently of USD.
Validated with MuJoCo 3.16.0; compatibility with older versions is not tested.

## What is retained

- 20 robot rigid bodies, 14 revolute joints, 5 fixed connections and a floating base.
- Total mass: 14.497599884867668 kg.
- All 22 active collision primitives: 6 boxes, 12 capsules and 4 spheres.
  The earlier URDF exporter omitted the capsules. That omission is one identified
  model difference; it is not established as the only cause of its poor walking.
- Original body frames, local centres of mass and complete inertia tensors.
  Joint anchors and axes account for both USD joint frames. Some compiled axes
  point along negative X or Y; changing their signs would change joint semantics.
- Four mirrored visual meshes have their transforms baked into vertices, normals
  and face winding. Their reflected shapes are retained without reflected body
  frames. Visual materials/textures are replaced by a neutral colour.
- Self collisions are disabled, matching the recorded locomotion configuration.
  Active robot shapes collide with the new floor. Guide shapes without a USD
  CollisionAPI are not silently made into collision shapes.

The original environment and its embedded floor are excluded from the MJCF.

## Validation scope and remaining differences

The validator checks compiled masses, COMs, inertia tensors, joint axes/limits,
actuators, collision primitives and visual mesh extents. It reconstructs forward
kinematics from the extracted USD joint frames at 22 poses, including nonzero
angles, and compares every body against native MuJoCo forward kinematics.

The 5-second PD test checks finite state, contacts, solver warnings and gross
collapse. It does not establish good standing or walking: the initial run moved
approximately 0.13 m forward, reached about 14.37 degrees of tilt, and settled
near a torso height of 0.226 m. Minimum contact distance was approximately
-0.00867 m. See `validation.json` for exact results. Contact softness has not been
calibrated against PhysX.

Known approximations and unresolved source content:

- Each foot's USD COM/inertia is marked for automatic calculation. This model
  derives it from that foot's sole sphere collider and authored mass. The result
  has not been compared to the inertia returned by a running PhysX simulation.
- The camera pinhole definition is rebuilt using test_lk_ostrack's actual local
  fallback and imaging profile. Its unresolved external housing mesh remains
  absent. The source also has four unresolved foot visual mesh references. The
  foot sphere collision shapes remain present; use `--show-collisions` to see
  them. Complete decorative geometry and Isaac RTX image appearance are not
  reproduced.
- MuJoCo contact/solver parameters are explicit initial choices, not an exact
  PhysX translation. Friction randomization, training events and actuator delays
  are outside the USD model.
- `control.json` records the leg gains/limits from the previous original-USD play
  run and the turret gains from USD. The preview uses explicit torque PD, no
  delay, and no enforced joint velocity cap. The original turret uses implicit
  PhysX drives. The recorded policy order is FL, FR, HL, HR, each HipX/HipY/Knee;
  do not assume the USD or MuJoCo native joint storage order is the policy order.
- The learned policy has now been tested in MuJoCo with matched observation
  ordering/scales, named-joint actions, 0.02 s policy period and delayed target PD.
  See [WALKING.md](WALKING.md) and the separate rollout reports. These tests do not
  establish full PhysX dynamics equivalence or closed-loop perception/aiming.

## Rebuild

```bash
/home/jby/miniconda3/envs/isaac_lab_232/bin/python source/standalone/integrated_shooting_chain/mujoco_sim/tools/build_lite3_mjcf.py
/home/jby/miniconda3/envs/mujoco_test/bin/python source/standalone/integrated_shooting_chain/mujoco_sim/tools/validate_lite3_mjcf.py
```

The builder locates the local Isaac Sim USD Python bindings automatically. Its
defaults target this specific robot; it deliberately stops on unsupported shapes
or unexpected joint/scale layouts instead of silently exporting partial physics.

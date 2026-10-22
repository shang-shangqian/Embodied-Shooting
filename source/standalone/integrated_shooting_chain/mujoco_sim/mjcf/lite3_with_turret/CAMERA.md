# Mounted camera matching test_lk_ostrack

The camera is named **tracking_camera**, is attached to **TORSO**, and is included
in `robot.xml`, `scene.xml` and `scene_lidar_demo.xml`. It follows TORSO translation,
roll, pitch and yaw. It does not turn with the turret joints.

The builder reads `test_lk_ostrack/multi_target_collect_env.py` and applies its
actual `_ensure_vision_camera_prim` function to an anonymous USD session layer.
This reproduces the local pinhole fallback and preserves the source mount while
avoiding the unresolved remote H60YA asset. No source USD is saved or changed.
The configuration and source hashes are saved in `camera_config.json`.

## Calibration

| Parameter | Matched value |
| --- | --- |
| Resolution | 1024 x 576 |
| USD focal length | 10.0 |
| USD horizontal/vertical aperture | 5.76 / 3.24 |
| Target fx / fy | 1777.7777099609375 pixels |
| Target cx / cy | 512 / 288 pixels |
| Approximate horizontal/vertical FOV | 32.13 / 18.40 degrees |
| Near/far clipping | 0.1 / 1,000,000 metres |
| Mount position in TORSO | (0.25, -0.0005, 0.103) metres |
| Nominal forward direction | TORSO +X |
| Output | RGB uint8 HWC, optionally metric axial depth and geom IDs |

The K calculation follows IsaacLab's actual square-pixel intrinsic calculation
and float32 storage. The compiled MuJoCo K differs by less than 0.001 pixel due
to float32 sensor-size/focal representation. The wrapper exports the compiled K
used by the renderer, so downstream projection uses that calibration.

MuJoCo/USD camera axes are right/up/backward. Geometry/perception outputs are
converted to ROS optical right/down/forward by flipping camera Y and Z. The
camera mounting position differs from the LiDAR position; the joint demo uses
each sensor's own position. It does not reuse a co-located synthetic camera.

Source f_stop=0 means no depth of field. The focus-distance setting of 400 is
recorded but no optical focus simulation is added. No lens distortion, rolling
shutter, exposure/noise model or Isaac RTX lighting reproduction is claimed.
The remote camera **housing mesh** is still absent; a functional camera does
not automatically add that visual asset, mass or collision geometry.

## Capture images

From `/home/jby/IsaacLab-2.3.2`:

```bash
conda activate mujoco_test
MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 python source/standalone/integrated_shooting_chain/mujoco_sim/tools/demo_mujoco_camera.py --frames 3 --depth --lidar
```

This runs without a desktop window using Mesa software rendering. For hardware
EGL, omit LIBGL_ALWAYS_SOFTWARE when a working GPU EGL setup is available.

The robot is held at the stand pose. A red sphere is placed on a prescribed
trajectory as a scene stimulus. A simple RGB colour box supplies visual angles;
those angles query the same-frame LiDAR hits using the actual mounted camera
extrinsics. The estimator receives no target-truth position/bearing. This is
not OSTrack/LK inference or closed-loop walking/aiming.

The outputs in `outputs/usd_mjcf_validation_20261021/camera/` are:

- `camera_rgb.png`, `rgb_*.png`: actual sensor RGB frames, without ray overlays.
- `intrinsics.json`: K compatible with test_lk_ostrack CameraIntrinsics.from_json.
- `last_camera_frame.npz`: RGB, K, ROS optical world pose and capture time.
- `depth_*.npy`: optional axial Z depth in metres; this is not Euclidean range.
- `last_lidar_scan.npz`: optional full final ray scan.
- `report.json`: RGB detections, camera poses and optional LiDAR estimates.

For a distant-target visibility check:

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 python source/standalone/integrated_shooting_chain/mujoco_sim/tools/demo_mujoco_camera.py --distance 100 --output outputs/usd_mjcf_validation_20261021/camera_100m
```

For an interactive view through the mounted camera:

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/preview_lite3_mjcf.py --model source/standalone/integrated_shooting_chain/mujoco_sim/mjcf/lite3_with_turret/scene_lidar_demo.xml --camera tracking_camera
```

The viewer window may have a different aspect ratio. Use the capture script's
fixed 1024x576 images for calibrated tracking and pixel geometry.

## Reusable sensor interface

Add `source/standalone/integrated_shooting_chain/mujoco_sim` to sys.path when running
standalone; the module only imports NumPy and MuJoCo.

```python
from mujoco_camera import MujocoCameraSensor

with MujocoCameraSensor(model, "path/to/camera_config.json") as camera:
    frame = camera.capture(data, depth=True)
    image_rgb = frame.rgb
    camera_position = frame.position_w
    optical_to_world_rotation = frame.rotation_w_optical
    # Feed image_rgb and frame.intrinsic_matrix into the tracker.
    yaw_pitch = camera.pixel_to_angles(predicted_pixel_center)
```

Rendering starts only on the first capture; pose/projection methods require no
graphics context. `capture` refreshes kinematics by default, and a returned frame
is a copy. Call it at the required simulation timestamps; no automatic camera
clock or asynchronous latency is introduced. Explicitly schedule captures when
integrating the tracker. Separate multi-environment rendered batching is not
implemented.

Clipping in MuJoCo is global and scaled by model extent. Capture sets it from
the profile, preserving visibility beyond the old scene's approximately 60 m
default far plane. These visual settings do not change collision physics.

Pixel coordinates follow the source K convention, with cx=width/2. Raster pixel
(row i, column j) has centre (j+0.5, i+0.5). `project_world` reports continuous
pixel-boundary coordinates, camera Z depth and frustum validity. A projected
visible point can still be occluded by a nearer surface; frustum validity is
not an occlusion test.

## Validation

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 python source/standalone/integrated_shooting_chain/mujoco_sim/tests/test_mujoco_camera.py
```

Six checks cover the recorded Isaac K, TORSO mount and independence from turret
angles, full base-pose changes, projection/yaw/pitch parity with the actual
test_lk_ostrack CameraIntrinsics implementation, asymmetric rendered marker
locations (detecting flips), 100 m target visibility and metric depth/capture
mode reset. Full tracker accuracy and image-domain compatibility remain to be
evaluated.

MuJoCo's camera attributes and axis conventions are documented in the
[official camera reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera).

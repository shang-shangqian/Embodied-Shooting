# MuJoCo locomotion policy tests

The adapter is `source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_lite3.py`.
It evaluates the user-specified **TorchScript policy directly**, with no weight
conversion or retraining:

`logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt`

Policy SHA-256:
`1f8a1a89541132126af48506505a3f3e9272b955c439a5353cc8df0b034d7e34`.
Its weights match the exported policy used in the earlier original-USD Isaac
reference run (maximum parameter difference zero). The native physics tests use
`scene.xml`, not the URDF round-trip model. Source USD and policy files are not
modified.

## Reproduce

From `/home/jby/IsaacLab-2.3.2`:

```bash
conda activate isaac_lab_232
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=4

# Straight-ahead velocity command, with source-range actuator delays.
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_lite3.py --source-delay --video --output outputs/mujoco_walk_validation_20261021/source_delay_seed42

# Restore the training command generator's outer heading feedback.
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_lite3.py --source-delay --heading-target 0 --video --fps 25 --output outputs/mujoco_walk_validation_20261021/heading_hold_source_delay_seed42

# Stand, walk forward, stop, turn left/right, move sideways, walk backward, stop.
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_lite3.py --source-delay --scenario transitions --duration 32 --video --fps 25 --output outputs/mujoco_walk_validation_20261021/transitions_source_delay_seed42
```

The Isaac Python environment supplies Torch. If MuJoCo is absent there, the
script appends the existing sibling `mujoco_test` site-packages to sys.path.
Neither environment is installed into or modified. Software rendering is used
only for the video; simulation and policy inference both run on CPU.

Omit `--video` for a fast headless simulation. Re-render a saved trajectory with:

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_lite3.py --render-only outputs/mujoco_walk_validation_20261021/transitions_source_delay_seed42 --fps 25
```

Videos are generated from the exact saved physics states. Rendering performs no
new dynamics, policy calls or resets. Command phase subtitles appear in the
transition and heading-hold videos. The earlier constant-command video was
recorded before subtitle support was added.

## Matched policy interface

- Physics period 0.005 s; policy period 0.02 s, four physics steps per action.
- Observation: local angular velocity x0.25 (3), projected gravity (3), velocity
  command (3), relative joint positions (12), relative joint velocity x0.05 (12),
  previous clipped action (12): **45 values**. Clipping occurs before term scales.
- Policy joints are FL, FR, HL, HR, each HipX/HipY/Knee. Explicit names map to
  native MuJoCo qpos/dof/actuator addresses; native storage order is not assumed.
- Action target is default joint position plus scaled action. HipX scale is
  0.125; other leg joint scales are 0.25. Leg PD is kp=30, kd=1 with 24 Nm hip
  and 36 Nm knee effort limits. Torso and joint frames retain USD semantics.
- Free-joint rotational velocity is used in TORSO axes. MuJoCo BODY-local object
  velocity uses the principal-inertia frame for this model; the adapter avoids
  confusing that with the policy body frame. An independent world-velocity
  transformation test covers this distinction.
- `--source-delay` samples one command delay per actuator group at reset from
  0..5 physics steps. Seed 42 selects hip=0, knee=4 (0 and 20 ms). The delayed
  target buffer advances every physics step, including repeated policy targets.
- Default fixed delays are zero. The delay groups, seed and selected values are
  recorded. The sampled values follow the source distribution; NumPy sampling
  does not reproduce an Isaac/Torch RNG stream or its previous episode draws.
- `--heading-target` uses the original heading gain 0.5 and yaw-rate limits
  +/-0.8 rad/s, with wrapped world-yaw error. A direct zero yaw-rate command
  otherwise provides no outer heading correction.

There is no observation corruption, pushes or mass/material randomization in
these controlled tests. No camera, LiDAR or target-truth observation is supplied
to the locomotion policy. The camera and LiDAR remain separate sensor modules.

## Outputs and interpretation

Each rollout directory contains `report.json`, `motion.csv`, `trajectory.npz`
and optional `walk.mp4`/`walk.srt`. The trace preserves physics qpos/qvel at 200 Hz,
observations/actions at 50 Hz and commanded torques at 200 Hz.

`first_fall_time_s` reports TORSO height below 0.12 m or tilt above 60 degrees.
The tests also report minimum height and tilt rather than relying solely on this
coarse fall criterion. No automatic resets or support fixtures are used. A
MuJoCo numerical warning aborts the rollout instead of hiding a reset.

Audit contacts and heading from recorded states with:

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/analyze_mujoco_walk.py outputs/mujoco_walk_validation_20261021/source_delay_seed42 outputs/mujoco_walk_validation_20261021/heading_hold_source_delay_seed42 outputs/mujoco_walk_validation_20261021/transitions_source_delay_seed42
```

The audit samples contacts every 0.04 s after 2 s. It is not a force/friction
equivalence test. A contact-frame fraction indicates the fraction of sampled
states with contact, not the number of footfalls or independent collisions.

The test results and remaining issues are summarized in
`outputs/mujoco_walk_validation_20261021/README.md`.

Known limitations remain: inferred foot inertias, uncalibrated contact compliance,
and explicit turret PD replacing implicit PhysX drives. Velocity limits are
reported as metadata; the adapter does not clip qvel or add a DC torque-speed
curve to the source IdealPD model. No friction/contact tuning experiment is
promoted automatically into `robot.xml`.

The optional `--contact-time-constant 0.01` experiment changes only the in-memory
contact solver profile and records that override. It does not write model files.
Reports distinguish this run from the unmodified MJCF baseline.

## Adapter checks

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tests/test_mujoco_locomotion_bridge.py
```

Checks cover joint ordering/scales, body angular velocity and gravity under a
90-degree roll, physics-step delay buffers, torque limits, command-transition
boundaries, and heading wrapping/saturation.

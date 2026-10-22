# Integrated shooting chain

MuJoCo-specific sensors, conversion/preview/walking tools, MJCF assets and tests
are grouped under [`mujoco_sim/`](mujoco_sim/README.md). Start there for MuJoCo
commands and model documentation. Shared shooting-chain code and Isaac training
entry points remain in this directory.

This directory connects the three existing modules for both runtime inference
and perception-in-the-loop PPO training:

```text
test_lk_ostrack.MultiTargetVisionTracker
        -> camera position + target quality
test11 legacy_gru.OnlineLegacyGRU (13-D ordinary GRU)
        -> relative target velocity in yaw-only base coordinates
test7_2_weight Actor (28-D observation)
        -> yaw delta + pitch delta + fire intent
```

## Runtime contract

Use `ShootingChain.initialize()` for the first RGB frame (with detector-provided
initial boxes); it resets estimator history and recurrent policy state for a new
sequence. Then call `ShootingChain.update()` once per frame. The caller
provides `RobotState` and receives a `DecisionCommand`. Apply
`command.action` to the turret and use `command.fire_command` as the external
fire command. `isaac_adapter.write_prediction_to_turret_env()` writes the
prediction into the existing `TurretAimEnv` bridge buffers.

`factory.build_chain()` constructs all three modules from the camera intrinsics,
ordinary-GRU checkpoint and policy. Pass a caller-owned `policy_adapters.RLGamesPlayerPolicy`
when using the original RL-Games `.pth` policy so its recurrent state and input
normalization remain active.

Minimal call pattern:

```python
from integrated_shooting_chain import RobotState
from integrated_shooting_chain.factory import build_chain
from integrated_shooting_chain.policy_adapters import RLGamesPlayerPolicy

chain = build_chain(
    intrinsics_json=".../camera_intrinsics.json",
    rotation_base_from_camera=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    camera_position_base_m=[0.23, 0.0, 0.073],
    velocity_checkpoint="source/standalone/test11/speed_prediction/legacy_gru/full_episode_finetune_runs/20261013_163052_278937/best_model.pt",
    velocity_backend="legacy_gru",
    ostrack_repo=".../test_lk_ostrack/ostrack/OSTrack-main",
    ostrack_checkpoint=".../OSTrack_ep0300.pth.tar",
    decision_policy=RLGamesPlayerPolicy(player),
    device="cuda:0",
)
robot = RobotState(turret_yaw_rad=0.0, turret_pitch_rad=0.0)
command = chain.initialize(rgb_uint8, initial_boxes, robot)
command = chain.update(next_rgb_uint8, robot, dt_s=1.0 / camera_fps)
```

`initial_boxes` is the detector output, in `{target_id: (x1, y1, x2, y2)}`
format. The existing tracker then maintains those targets with LK and optionally
refines the selected target with OSTrack. Pass both `ostrack_repo` and
`ostrack_checkpoint` to `build_chain()` to enable refinement; omit both for
LK-only tracking.

The camera extrinsics must be supplied in the same convention as the vision
dataset: `rotation_base_from_camera` maps `[right, down, forward]` to
`[forward, left, up]`, and `camera_position_base_m` is the camera origin in the
yaw-only base frame. Do not use an identity extrinsic for the real robot.

The default velocity backend is the ordinary GRU in `test11/speed_prediction/legacy_gru`.
`legacy_gru_adapter.py` reuses its tested rolling inference adapter: measured
position (3), causal position increment in metres, not divided by dt (3), robot
linear velocity (3), angular velocity with exact yaw-only heading rate (3),
and a horizontal-moving flag (1). The frozen checkpoint's normalization and
`pos_smooth_window` are used unchanged; its output is relative translation
velocity in yaw-only base coordinates, not the derivative of rotated position.

Only TAQ-v2-trained ordinary-GRU checkpoints with compatible label semantics are
accepted. The first direct valid position seeds history without predicting;
pre-initialization missing positions do not enter the GRU. Later missing frames
retain the original zero-fill handling and do not reset history. Each target/env
has independent state, cleared on a new episode/sequence. There is no learned
uncertainty head: the exposed `reliability_gate` is only a direct-observation flag,
not TAQ's learned quality gate or a calibrated confidence probability.

The Isaac bridge also waits for robot/aim warmup before bootstrapping vision:
test7_2_weight's camera-valid flag alone does not include that warmup. The existing
actor-side readiness threshold (three direct observations and 0.10 s history by
default) is retained; it is separate from the GRU's first-position seed rule.

To explicitly use the old TAQ module, select `velocity_backend="taq_gru"` and
provide the Stage1 TAQ-v2 checkpoint. Model types are not silently autodetected.

The RL loader in `model_loaders.py` accepts a TorchScript 28-to-3 export. The
original RL-Games `.pth` checkpoint is stateful (normalization and GRU state),
so it should be exported through the test7_2_weight evaluation/RL-Games setup
before use here, or passed through a caller-owned policy callable.

## Full-chain PPO training

`isaac_training_bridge.py` installs the online perception path into
`test7_2_weight/train_turret_mixed_gt.py`.  At every environment step it:

1. reads the current rendered RGB image;
2. updates LK and optional OSTrack using independent state per Isaac env;
3. sends the predicted angle into the current-frame raycaster;
4. feeds the measured 3-D position and robot velocities to the frozen ordinary GRU;
5. writes only valid, warmed-up predictions into the actor observation;
6. returns the rebuilt observation to RL-Games, which remains the sole owner
   of the recurrent actor/critic and PPO update.

The vision and velocity modules are frozen.  “Full-chain training” therefore
means training the RL policy against their real online errors; it does not
jointly backpropagate through LK/OSTrack or the velocity GRU.

Activate the `isaac_lab_232` environment, then run the provided launcher from
the IsaacLab repository root:

```bash
bash source/standalone/integrated_shooting_chain/train_full_chain.sh \
  logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt \
  --max_epochs 15000
```

The launcher defaults to 16 environments for the memory-limited fine-tuning setup;
all environments in a perception bridge share one frozen OSTrack network, while
templates, masks, boxes, frame counters and LK state remain independent. The
network is loaded once per bridge/device and retained across episode resets.
There is no hard two-environment limit
in full-chain training or evaluation: override `--num_envs <N>` while keeping
OSTrack enabled. Larger counts still consume more GPU memory for cameras and
per-environment state; reduce the count if allocation fails. Inference remains
sequential across environments. For a cheaper
LK-only run, pass `--no-fine_tracking --num_envs <N>`.

Shared-weight regression tests are in `tests/test_shared_ostrack.py`. Set
`RUN_OSTRACK_CUDA_TESTS=1` when running that file with pytest to additionally
check the real checkpoint on CUDA: 16 trackers build one network, and interleaved
tracking plus a single-environment reset agree with independent-model references.

### Conservative PPO fine-tuning defaults

Full-chain mode defaults to `num_envs=16`, `horizon_length=128` (2048 samples
per update), `minibatch_size=1024`, `mini_epochs=2`, `seq_length=8`, and a fixed
`learning_rate=3e-6` (`--lr_schedule none`). Entropy, clipping and gradient norm
remain `0.003`, `0.1` and `0.5`. These values are CLI-overridable; the minibatch
must divide the rollout batch and be divisible by the sequence length, and the
horizon must also be divisible by the sequence length. A smaller environment
count automatically reduces the default minibatch when necessary. GT mode keeps
its previous defaults instead of inheriting the full-chain fine-tuning settings.

Checkpoint restoration keeps model weights, normalization, optimizer moments,
epoch and frame counters. In full-chain mode, `--reset_learning_rate` is enabled
by default: after restoration, both the optimizer and scheduler learning-rate
state are set to the configured rate before the first update. To keep the
checkpoint's rate instead, use `--no-reset_learning_rate`; the scheduler state
is then synchronized to that restored rate. `--lr_schedule adaptive` explicitly
re-enables adaptive scheduling; the conservative default is fixed-rate training.

Best-reward saving starts immediately (`--save_best_after 0`), and periodic
checkpoints are saved every 50 absolute epochs (`--save_frequency 50`). Each
run also saves `training_settings.json` with resolved CLI and PPO settings.
Best training reward still does not guarantee best held-out shooting accuracy.

### Optional five-shot budget experiment

Pass `--max_shots_per_episode 5` to train with at most five actual fires per
episode, while retaining the existing 10 s timeout (including 2 s warmup).
The fifth shot is evaluated normally: a hit is success, a miss ends the episode
as a task failure, without time-limit value bootstrapping. Fire intent during
warmup, unavailable perception or cooldown does not consume the budget.
Existing hit/failure/out-of-view endings still apply. No reward formula, reward
weight or actor/critic observation dimension is changed; old checkpoints load
as before. Budget exhaustion appears as `shot_limit` in training logs and
`Episode_Termination/shot_budget_exhausted_rate` in TensorBoard.

The default is `--max_shots_per_episode 0` (unlimited), in both training and
single-target evaluation. Evaluation also accepts `--max_shots_per_episode 5`
to measure the same budget; this option is incompatible with `--static_salvo`.
Both the aggregated summary and the per-case SS/SD/DS/DD summaries report
`shot_budget_exhausted` and `shot_budget_exhausted_rate`, separately from timeouts.
The final summary repeats the configured time limit and shot budget.
For example, to evaluate the five-shot run's best policy on SD:

```bash
CUDA_VISIBLE_DEVICES=0 python -u source/standalone/test7_2_weight/eval_with_track_gt.py \
  --headless \
  --checkpoint runs/test7_2_weight/full_chain_5shots_20261017_201341 \
  --yaw_error_mode wrapped_los \
  --which best \
  --locomotion_policy logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt \
  --full_chain_perception --case sd --episodes 200 \
  --max_shots_per_episode 5 --no-pause_between_episodes
```

Compare old and new checkpoints under the same evaluation budget. In particular,
keep the original unlimited-shot evaluation as a control: removing later shots
can increase `accuracy_rate` (hits/fires) without improving the policy, and may
reduce episode success. Shorter failures also change returns and the training
distribution despite leaving the per-step reward formula unchanged.

For 200 additional updates from the original epoch-114000 policy, run from the
repository root after activating `isaac_lab_232`:

```bash
bash source/standalone/integrated_shooting_chain/train_full_chain.sh \
  logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt \
  --device cuda:0 --num_envs 16 \
  --checkpoint runs/test7_2_weight/turret_aim_mixed_09-08-16-46-24 --which last \
  --horizon_length 128 --minibatch_size 1024 --mini_epochs 2 \
  --learning_rate 3e-6 --lr_schedule none --reset_learning_rate \
  --p_ss 0.5 --p_sd 0.5 --p_ds 0 --p_dd 0 \
  --max_epochs 114200 --save_best_after 0 --save_frequency 50 \
  --experiment_name full_chain_ss_sd_gru_small_lr_$(date +%Y%m%d_%H%M%S)
```

`--max_epochs` is an absolute limit, not an additional-update count. Evaluation
is separate: compare saved checkpoints on the same SS/SD conditions and retain
the original policy as a baseline. Longer rollouts cost more simulation time
and buffer memory without creating additional camera or OSTrack instances.

Environment camera/raycast diagnostic prints are disabled by default in both
training and evaluation. Use `--debug_diagnostics` to enable them, independently
of `--bridge_debug` for perception-bridge statistics. Disabling diagnostics does
not disable the raycaster or suppress training metrics and actual sensor errors.

The default velocity checkpoint is
`legacy_gru/full_episode_finetune_runs/20261013_163052_278937/best_model.pt`.
Full-chain mode retains its static-robot (SS/SD) restriction and defaults to `p_ss=0.4`, `p_sd=0.6`,
`p_ds=0`, `p_dd=0`, and rejects explicit DS/DD probabilities. Train a moving-
base velocity model before enabling DS/DD.

Training and evaluation sample moving-target **3-D speed magnitude** uniformly
from 0.6 to 5.0 m/s (`--target_speed_min` / `--target_speed_max`), then independently
sample a uniform direction on the unit sphere. The resulting x/y/z velocity
components have no separate bounds; the former `--target_vz_min` and
`--target_vz_max` options have been removed. Both entry points set acceleration
to zero. Distance, camera-sector and ground boundaries still reflect the target
velocity, preserving its speed magnitude. SS targets remain stationary.

Simulation GT direction and range are used once per episode to create the
initial box, acting as a detector proxy. Once that bootstrap is consumed,
tracker loss is not repaired with GT; recovery waits for the next episode (or
a future real detector/re-identification module). Tracker updates come from RGB
and range comes only from the raycaster's valid hits; there is no actor-side GT
fallback.


## Evaluate an existing RL policy with the full chain

As of 2026-10-21, training and evaluation default to `--yaw_error_mode wrapped_los`
(`wrapped_los_v1`): Actor yaw is target yaw minus measured muzzle yaw, wrapped
to [-pi, pi] for the circular shortest turn. This matches historical checkpoints.
`--yaw_error_mode joint_reachable` remains an explicit experimental option
requiring adapted weights.
Checkpoints and reports record the observation contract. See the
[geometry and compatibility notes](../test7_2_weight/YAW_REACHABLE_OBSERVATION_20261020.md).
CSV schema 5 adds the raw Actor yaw goal, joint error, and yaw reachability flag;
physical LOS diagnostic errors retain their existing meaning.

Optional `--salvo_return_home` physically returns yaw/pitch to zero before each
attempt. It is off by default and requires D with velocity waiting. Actor inference
and firing pause until measured joint positions and velocities settle; Actor state
is cleared once after homing, including the first acquisition and same-ID retries.
Homing uses the existing incremental position commands, without a physics reset.
Its time counts toward the episode but is excluded from the 3 s attempt deadline.
CSV schema 6 records homing control and excluded time. See the
[return-home experiment and command](SS_RETURN_HOME_20261021.md).

For SS clearance of 100 stationary targets, add `--case ss --static_salvo
--salvo_targets 100 --num_envs 1 --episodes 1`. This mode automatically enables
full-chain perception. After standing warmup, one RGB frame and all 100 GT boxes
initialize one multi-target tracker per environment. LK then tracks all remaining
targets continuously; OSTrack refines the selected target. At each selection,
choose the nearest available target not yet attempted in the current round,
using the latest distance inferred from tracked box size and the known sphere
diameter. Retain the selected target until hit, lost, five consecutive actual shots
miss (`--salvo_miss_limit 5`), or 3 simulation seconds elapse without a hit
(`--salvo_stall_timeout_s 3`). Default `--salvo_switch_policy both` enables the
two failure triggers together; the first one reached wins. Waiting, cooldown,
and non-firing steps do not count as shots, but do consume the 3 s deadline from
selection. Firing does not restart that deadline; a boundary hit is a normal hit.
Unhit targets enter a separate next-round retry pool, are deferred for at least 1 simulation second,
and may be tried at most 3 times (`--salvo_retry_delay_s`, `--salvo_max_attempts`).
Cooldown expiry cannot return a failed ID to the same round. Once all other
available current-round targets have had a turn, promote the retry pool and
start a new round, again choosing by current distance rather than initial FIFO
order. Removed and exhausted IDs leave both pools. Unavailable tracks do not
block round promotion; their outstanding turn survives if tracking recovers.
A visible current-round target still cooling down waits instead of allowing
another round to start early. Newly available IDs join the current round. Each new attempt starts
with zero misses. A target that never becomes ready or never receives a fire
action is deferred when the 3 s deadline is reached.
Use `--salvo_switch_policy timeout` for the old timer-only rule, or `misses` for
shot-count-only switching. Existing commands specifying `misses` must change it
to `both` to enable the deadline too.
Deferral only excludes engagement; LK keeps tracking
every remaining ID. Hits remove only that track ID;
switches reset selected-target velocity history, preserving other visual tracks.
Static salvo evaluation defaults to group D: hold the turret position command
while perception is unready, and clear the Actor GRU state once before inference
on a different target ID. First acquisition retains warmup history; same-target
frames do not repeatedly clear the state. D also waits for the selected target's
range/velocity chain to become ready before policy inference. During the wait,
the policy is not called, recurrent states stay unchanged, and `[0, 0, -1]`
holds the turret command and prevents firing. Perception and standing control
continue. Later perception loss pauses inference again without clearing state.
Use `--no-salvo_wait_for_velocity` to reproduce the original D protocol.
CSV schema 3 and later record `velocity_ready`, `actor_inference_skipped`, and
`actor_wait_step_count`; JSON records the wait protocol and episode wait totals.
Use `--salvo_ablation A`, `B`, or `C`
to select a comparison group, or `--salvo_ablation D` to make the default explicit.
CSV schema 4 adds `attempt_shots` and `consecutive_misses`; reports and failure
events identify the switch policy and actual missed-shot count. Reports identify
`selection: dynamic_nearest_per_round_deferred_retries`. Each episode records
`selection_round` (1-based), `round_pending_target_ids`, and `next_round_target_ids`.
The latter records failure insertion order, not a fixed next-round selection order.
`pending_target_ids` combines both pools, excluding the current target; selection
and failure events also record their round. Historical saved runs keep their
original selection metadata and are not relabeled by this change.
The four-group campaign passes each group explicitly, uses legacy wrapped yaw observations and the old 3 s timeout,
and disables velocity waits to retain the original experiment. These controls apply only
to static salvo evaluation.
No later GT boxes are supplied, including on tracking loss. Lost, unhit targets
remain in the clearance denominator. All-lost pools terminate with
`tracking_lost`; exhausted attempts terminate with `attempts_exhausted`, while
temporary retry delays keep the episode running. The JSON report includes
remaining IDs, attempt histories, `gt_bootstrap_frames` and
`gt_bootstrap_targets` for each episode. LK updates center and scalar size
without recursively enclosing rotated rectangles. This static SS mode also
rejects dimensions outside 0.5--2x the first visual box size; the cumulative
size guard is disabled for ordinary tracking.
Add `--record_video --video_episodes 1 --video_env_id 0 --video_hit_hold_s 0`
to record the whole clearance attempt in one MP4, including intermediate hits.
For this mode the main view shows robot RGB with tracked boxes, and an inset
frames the entire original target pool and robot. Counters show shots, destroyed
targets, remaining targets, accuracy and simulation time. The clearance report
defaults to `salvo_report.json` beside the video when recording is enabled.
Per-step firing gates, intent, actor prediction errors and visual quality are
saved beside the report as `salvo_report.steps.csv` (`--salvo_diagnostics` to
override). Progress is printed every 10 wall-clock seconds
(`--salvo_log_interval_s`), separately from simulation-time deadlines.
CSV schema v2 adds paired decision/post-physics snapshots of predicted/GT/muzzle
directions, the returned 28-D actor observation, yaw/pitch actions, joint commands,
actual joint motion/limits and cached ballistic scores. GT comparisons are
diagnostic-only and never enter vision or actor inputs. See
[SS control diagnostics](SS_CONTROL_DIAGNOSTICS.md) for timing, coordinates and validity rules.
During play, Ctrl+C finalizes the video/CSV and saves an interrupted result
before Isaac shutdown. Initialization interruption or forcible termination
cannot guarantee a finalized MP4. Start with 10 targets and a 30 simulation-second
timeout for a short validation; GPU evaluation can contend with concurrent training.
The selected ID is mapped to a simulation sphere for raycast measurement and
ballistic scoring; other-target raycast occlusion and accidental hits are not
modeled. See [the SS scenario parameters and command](../test7_2_weight/README.md).

Use the trained `test7_2_weight` checkpoint as the decision policy. This mode
runs inference only; it does not update PPO weights. `--case sd` evaluates the
supported static-robot, moving-target condition. From the repository root:

```bash
conda activate isaac_lab_232
./isaaclab.sh -p source/standalone/test7_2_weight/eval_with_track_gt.py \
  --headless \
  --num_envs 2 \
  --checkpoint runs/test7_2_weight/YOUR_RUN_DIR \
  --which best \
  --locomotion_policy /path/to/locomotion_policy.pt \
  --full_chain_perception \
  --case sd \
  --episodes 10 \
  --bridge_debug
```

Replace `YOUR_RUN_DIR` with the run directory containing the RL checkpoint and
set the locomotion policy path. The ordinary-GRU and OSTrack paths default to the
checkpoints under `source/standalone`; override them with
`--full_chain_velocity_checkpoint`, `--ostrack_repo`, and `--ostrack_checkpoint` if
needed. Add `--no-fine_tracking` to evaluate LK without OSTrack corrections.

Training uses `--velocity_model` for the speed checkpoint. Both entry points
default to `--velocity_backend legacy_gru`; to reproduce the previous TAQ chain,
pass `--velocity_backend taq_gru` and its matching checkpoint explicitly.
Evaluation retains `--gru_velocity_checkpoint` and `--taq_velocity_checkpoint`
as aliases of `--full_chain_velocity_checkpoint`.

### Record ten episodes with the robot camera visible

Add these options to the full-chain evaluation command above:

```bash
--num_envs 1 --episodes 10 --record_video --video_episodes 10 --video_env_id 0
```

Each completed episode is saved as a separate 1280x720 MP4: the main view uses
an automatically fitted side overview that includes both the robot and target,
without ROBOT/TARGET labels or annotation circles. The upper-right inset shows
the unchanged RGB camera used by LK/OSTrack. A lower-left observer-detail window
shows a close-up render of the robot; there is no target-detail window. This is
a separate observer viewpoint, not a change to target size or the perception
camera. The caption displays simulation time,
fire count, hit status, warmup and perception readiness. Recording includes
warmup but does not feed additional observations into the GRU or advance physics.
Headless recording is supported. The default output directory is
`source/standalone/integrated_shooting_chain/video/<timestamp>/` (relative to the
repository, regardless of the working directory); `--video_dir` overrides it and must name
a new directory. `manifest.json` lists the videos, completion status and outcomes.

On a hit, the final pre-reset frame shows a yellow ballistic trajectory and a
HIT banner, then is frozen for 1.5 seconds of video playback. Change this with
`--video_hit_hold_s`; use `0` to disable the freeze. This adds encoded frames,
not physics steps or wall-clock sleeps. The manifest distinguishes playback
duration from simulation duration and records the hold-frame count.

The yellow path is a diagnostic reconstruction of the environment's evaluated
ballistics (including muzzle velocity, gravity and configured air drag),
truncated at its synchronous closest-approach time. This environment determines
hits by future ballistic evaluation, not a flying bullet entity, so the overlay
is not an animation of a physical projectile. It is generated from the same
post-physics state used by hit scoring, before autoreset. GT positions are used
only for observer framing; the path and status captions are drawn onto the recorded
scene RGB array, never into USD or the robot-camera image. They therefore cannot
leak into LK/OSTrack or change GRU/RL observations.
Recording uses these overlays instead of the GUI's legacy world-space trajectory
debug drawing, including when `--draw_tracks` is enabled.

Only `--video_env_id` is recorded. With multiple parallel environments,
`--episodes` counts completed episodes across all environments, not just the
recorded one; it may therefore stop before ten videos are complete. Use one
environment for ten guaranteed completed videos, or increase the evaluation
episode count. The recorder stops at ten complete episodes while evaluation
can continue. An unfinished episode at interruption/end is retained and marked
`complete: false`, not counted toward the quota. Writers are flushed before
Isaac Sim shutdown, so a shutdown hang does not leave completed MP4s unfinalized.
Recording adds rendering/encoding overhead; keep it disabled for throughput
benchmarking. Without `--record_video`, existing evaluation behavior is unchanged.

`source/standalone/model_with_recoil/` currently contains USD/URDF assets and a
modeling plan, not the Python chain entry point. This change affects the shared
integration code and the test7_2_weight full-chain train/eval paths; it does not
change asset selection, recoil dynamics, tracker implementation or the decision
network's 28-D observation / 3-D action contract.

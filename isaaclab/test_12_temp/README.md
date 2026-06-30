# test_12_temp: direction + velocity integration

This folder combines:

- `test12`: RGB frames + first-frame box -> OSTrack boxes -> yaw/pitch direction.
- `test_temp`: GRU velocity model -> target velocity from a short 3D position sequence.

Important: the velocity GRU is not an image-only model. It was trained on 13-D
sequence features:

```text
refined_pos_abs(3), refined_pos_diff(3),
robot_lin_vel(3), robot_ang_vel(3), dog_moving_flag(1)
```

For datasets collected by `test10/collect_direction_eval_data.py`, the script
now first looks for each sequence's `sensor_features.csv` as the velocity-model
input. If it is absent, it falls back to `velocity_features.csv` from simulation.
Velocity metrics still use `velocity_features.csv` as ground truth when present.

## Dataset Evaluation

Pass the collected dataset root directly. The dataset should contain
`manifest.csv` and `sequences/seq_xxxxxx_env_xxx/` folders.

```powershell
python isaaclab/test_12_temp/estimate_direction_velocity.py `
  --dataset_dir source/standalone/test10/direction_eval_dataset_red_combination `
  --out_dir isaaclab/test_12_temp/outputs/red_combination_eval `
  --ostrack_repo E:/path/to/OSTrack `
  --ostrack_checkpoint isaaclab/test12/OSTrack_ep0300.pth.tar `
  --velocity_model_path E:/path/to/best_model.pt `
  --sequence_length 9
```

Useful options:

```powershell
--max_sequences 5
--target_color_name red
--target_object_id 0
--overlay
--force
```

Outputs:

- `out_dir/estimates/*.csv`: one combined direction+velocity CSV per target run.
- `out_dir/summary.csv`: per-sequence counts plus direction and velocity metrics. The `input_features_source` column shows whether velocity input came from `sensor_features.csv` or the simulation fallback.
- `out_dir/overlays/...`: optional tracked-box visualizations when `--overlay` is set.

`summary.csv` computes direction errors against `ground_truth_direction.csv` and
velocity errors against `velocity_features.csv` column
`target_velocity_base_x/y/z` when those files are present.

## Single Sequence Mode

You can still run one sequence manually:

```powershell
python isaaclab/test_12_temp/estimate_direction_velocity.py `
  --rgb_dir test10_direction_eval_dataset/sequences/seq_000000_env_000/rgb `
  --initial_box_file test10_direction_eval_dataset/sequences/seq_000000_env_000/initial_box.txt `
  --intrinsics_json test10_direction_eval_dataset/sequences/seq_000000_env_000/intrinsics.json `
  --fps 50 `
  --ostrack_repo E:/path/to/OSTrack `
  --ostrack_checkpoint isaaclab/test12/OSTrack_ep0300.pth.tar `
  --velocity_model_path E:/path/to/best_model.pt `
  --position_csv test10_direction_eval_dataset/sequences/seq_000000_env_000/velocity_features.csv `
  --robot_motion_csv test10_direction_eval_dataset/sequences/seq_000000_env_000/velocity_features.csv `
  --out_csv isaaclab/test_12_temp/outputs/seq_000000_direction_velocity.csv
```

## 3D Input Options

Use one of these for velocity:

- `--position_csv`: CSV with `frame_index` and one recognized 3D column set:
  `x_input_m,y_input_m,z_input_m`, `sensor_pos_base_x,sensor_pos_base_y,sensor_pos_base_z`,
  `refined_pos_base_x,refined_pos_base_y,refined_pos_base_z`, or `x,y,z`.
- `--distance_csv`: CSV with `frame_index` and one of
  `distance_m`, `range_m`, `depth_m`, `z_cam_m`, `target_distance_m`.
- `--constant_distance_m`: quick sanity-check mode, useful only when the target
  distance is nearly constant.
- `--target_diameter_m`: rough monocular depth from tracked bbox size and known
  target diameter.

If your distance column is optical-frame z depth, add `--distance_kind optical_z`.
Otherwise distance values are treated as range along the predicted yaw/pitch ray.
## Sensor Feature Input

For sensor-based velocity evaluation, put this file in each sequence folder:

```text
sequences/seq_xxxxxx_env_xxx/sensor_features.csv
```

Recommended columns, already in the velocity GRU's expected yaw-only base frame:

```text
frame_index,object_id,target_color_name
sensor_pos_base_x,sensor_pos_base_y,sensor_pos_base_z
robot_lin_vel_x,robot_lin_vel_y,robot_lin_vel_z
robot_ang_vel_x,robot_ang_vel_y,robot_ang_vel_z
dog_moving_flag,measurement_valid
```

If you only have a sensor range instead of `sensor_pos_base_*`, provide range and
pose columns so the script can reconstruct the base-frame position in the same
style as `test7_1/rl_perception_bridge.py`:

```text
lidar_distance,refined_yaw,refined_pitch
camera_pos_w_x,camera_pos_w_y,camera_pos_w_z
camera_quat_w,camera_quat_x,camera_quat_y,camera_quat_z
root_pos_w_x,root_pos_w_y,root_pos_w_z
root_quat_w,root_quat_x,root_quat_y,root_quat_z
```

Use `--sensor_features_name other_name.csv` if the per-sequence file has a
different name. In single-sequence mode, pass the same file through
`--position_csv`, `--distance_csv`, `--direction_csv`, `--frame_pose_csv`, and
`--robot_motion_csv` as needed.

## Robot Motion Input

`--robot_motion_csv` is optional. If omitted, robot linear/angular velocity and
`dog_moving_flag` default to zero.

Recognized columns:

```text
frame_index
robot_lin_vel_x, robot_lin_vel_y, robot_lin_vel_z
robot_ang_vel_x, robot_ang_vel_y, robot_ang_vel_z
dog_moving_flag
```

The first `sequence_length - 1` frames cannot have GRU velocity because the GRU
needs a complete history window. Default `sequence_length` is `9`.
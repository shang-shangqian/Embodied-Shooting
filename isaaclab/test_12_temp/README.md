本文件夹把 `test12` 的 OSTrack 目标框跟踪/方向估计思路和 `test_temp` 的 GRU 速度估计接在一起：先用第一帧目标框和 OSTrack 跟踪 RGB 序列，得到每帧 bbox、yaw、pitch；再根据 3D 位置序列、机器人运动信息和速度 GRU 估计目标速度；最后输出方向 + 速度的合并 CSV，并在数据集模式下汇总误差指标。

## 文件说明

- `estimate_direction_velocity.py`：主入口。支持单序列和数据集两种模式，负责调度跟踪、方向估计、位置构建、速度估计、CSV 输出和数据集汇总。
- `camera_geometry.py`：本地相机内参和几何工具，包含 `CameraIntrinsics`、内参 JSON 读取、像素转 yaw/pitch、反投影和 bbox 中心计算。
- `direction_pipeline.py`：本地方向估计流水线，包含 RGB 帧读取、bbox 跟踪结果转方向、角速度计算和 overlay 输出。它复用 `test12.ostrack_tracker.TrackedBox` 类型。

## 外部复用


- OSTrack 适配器：`isaaclab/test12/ostrack_tracker.py`。
- 速度模型定义：`isaaclab/test_temp/speed_prediction_model.py`。



## 数据集模式

数据集根目录需要包含 `manifest.csv` 和 `sequences/seq_xxxxxx_env_xxx/` 子目录。每个序列通常包含：

- `rgb/`：按文件名排序的 RGB 帧。
- `intrinsics.json`：相机内参，支持 `fx/fy/cx/cy` 字段，或 `intrinsic_matrix` / `camera_matrix` / `K` 形式的 3x3 矩阵。
- `initial_boxes.csv`：多目标初始框，字段包含 `object_id,x1,y1,x2,y2`，可选 `target_color_name,target_color_rgb`。
- `initial_box.txt`：单目标初始框，格式为 `x1 y1 x2 y2`。当没有 `initial_boxes.csv` 时使用。
- `sensor_features.csv` 或 `velocity_features.csv`：速度输入特征，通常包含 `sensor_pos_base_x/y/z`、机器人线速度/角速度、`dog_moving_flag` 和有效性字段。
- `ground_truth_direction.csv`、`velocity_features.csv`：可选，用于汇总方向/速度误差。

示例：

```
python isaaclab/test_12_temp/estimate_direction_velocity.py `
  --dataset_dir path/to/direction_velocity_dataset `
  --out_dir isaaclab/test_12_temp/outputs/dataset_eval `
  --ostrack_repo E:/path/to/OSTrack `
  --ostrack_checkpoint isaaclab/test12/OSTrack_ep0300.pth.tar `
  --velocity_model_path E:/path/to/best_model.pt `
  --sequence_length 9
```

## 单序列模式

```
python isaaclab/test_12_temp/estimate_direction_velocity.py `
  --rgb_dir path/to/sequence/rgb `
  --initial_box_file path/to/sequence/initial_box.txt `
  --intrinsics_json path/to/sequence/intrinsics.json `
  --fps 50 `
  --ostrack_repo E:/path/to/OSTrack `
  --ostrack_checkpoint isaaclab/test12/OSTrack_ep0300.pth.tar `
  --velocity_model_path E:/path/to/best_model.pt `
  --position_csv path/to/sequence/velocity_features.csv `
  --robot_motion_csv path/to/sequence/velocity_features.csv `
  --out_csv isaaclab/test_12_temp/outputs/seq_direction_velocity.csv
```

## 输出文件

- `out_dir/estimates/*.csv`：每个目标 run 的方向 + 速度估计结果。
- `out_dir/summary.csv`：数据集汇总指标，包含方向误差、速度误差、有效帧数和输入特征来源。
- `out_dir/overlays/...`：使用 `--overlay` 时生成的可视化图片。
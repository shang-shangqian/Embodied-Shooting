```text
初始框 x1 y1 x2 y2
  -> OSTrack 跟踪每帧 bbox
  -> bbox 中心点 + 相机内参
  -> yaw / pitch 方向估计
```

## 文件说明

- `ostrack_tracker.py`：封装官方 OSTrack 跟踪器。负责加载外部 OSTrack 仓库、配置文件和模型权重，把输入初始框从 `x1 y1 x2 y2` 转成 OSTrack 使用的 `x y w h`，并逐帧输出跟踪到的 bbox。
- `pipeline.py`：方向估计的核心工具。负责读取 RGB 帧列表、把每帧跟踪框转换成 yaw/pitch、计算角速度、写出估计结果 CSV，也可以生成带 bbox 和角度文字的可视化图片。
- `estimate_video.py`：单个序列的运行入口。输入一段 RGB 图片、初始框、相机内参和 OSTrack 配置，输出该序列的方向估计 CSV，可选输出 overlay 图片。
- `evaluate_direction_dataset.py`：整个数据集的批量评测入口。读取数据集的 `manifest.csv`，对每个完整序列或多目标场景中的每个目标运行 `estimate_video.py`，再和真值 CSV 比较，汇总 yaw/pitch 误差、角度 RMSE、跟踪耗时等指标。

## 依赖说明

需要单独安装官方 OSTrack 仓库，并准备好对应 checkpoint。代码默认使用官方目录结构：

```text
OSTrack/
  experiments/ostrack/*.yaml
  lib/test/tracker/ostrack.py
  lib/config/ostrack/config.py
```


## 单序列运行

```bash
python -m test12.estimate_video \
  --rgb_dir test10_direction_eval_dataset/sequences/seq_000000_env_000/rgb \
  --initial_box_file test10_direction_eval_dataset/sequences/seq_000000_env_000/initial_box.txt \
  --intrinsics_json test10_direction_eval_dataset/sequences/seq_000000_env_000/intrinsics.json \
  --fps 50 \
  --ostrack_repo /path/to/OSTrack \
  --ostrack_checkpoint isaaclab/test12/OSTrack_ep0300.pth.tar \
  --ostrack_config vitb_256_mae_ce_32x4_ep300 \
  --device cuda:0 \
  --out_csv test12_outputs/seq_000000_est.csv \
  --overlay_dir test12_outputs/seq_000000_overlay
```

评估单序列误差：

```bash
python -m test10.evaluate_direction \
  --est_csv test12_outputs/seq_000000_est.csv \
  --gt_csv test10_direction_eval_dataset/sequences/seq_000000_env_000/ground_truth_direction.csv
```

## 数据集批量评测

```bash
python -m test12.evaluate_direction_dataset \
  --dataset_dir test10_direction_eval_dataset \
  --fps 50 \
  --ostrack_repo /path/to/OSTrack \
  --ostrack_checkpoint isaaclab/test12/OSTrack_ep0300.pth.tar \
  --ostrack_config vitb_256_mae_ce_32x4_ep300 \
  --out_dir test12_outputs/direction_dataset_eval \
  --overlay
```

批量评测会在 `out_dir` 下生成各序列估计 CSV、可视化图片、采样图片和 `summary.csv` 汇总结果。
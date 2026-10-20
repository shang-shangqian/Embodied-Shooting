# LK + OSTrack 多目标视觉

| 文件 | 作用 |
| --- | --- |
| `multi_target_tracker.py` | LK、RANSAC、Kalman、目标优先级与精跟踪调度。 |
| `camera_geometry.py` | 相机内参、像素与方向转换。 |
| `ostrack_tracker.py`、`batched_ostrack.py`、`online_vision.py` | OSTrack 序列、共享批量和仿真在线适配。 |
| `estimate_video.py` | 图片序列推理入口。 |
| `collect_multi_target_dataset.py`、`collect_velocity_text_dataset.py` | 多目标数据/主目标文本采集入口。 |
| `multi_target_collect_env.py` | 多目标采集环境，复用本仓库 `test11` 的基础环境。 |
| `evaluate_multi_target_dataset.py`、`evaluate_direction_dataset.py` | 多目标和单目标方向评估。 |
| `isaacsim_dataset.py`、`check_multi_target_dataset.py` | 数据写入与完整性检查。 |
| `multi_target_io.py`、`pipeline.py` | 初始框读取、CSV、叠加图及方向后处理。 |
| `rgb_visibility.py` | 初始化目标的渲染可见性检查。 |
| `__init__.py` | 公共接口。 |
| `tests/` | 本地已有的跟踪、数据和指标测试。 |
| `ostrack/OSTrack-main/` | 本地已有的 OSTrack 推理依赖子集；不是完整训练工程。 |
| `lite3/`、`turret/`、`*.usd`、`*.urdf` | 本地模型资产原文件。 |

从仓库根目录进行纯 LK 多目标图片推理：

```bash
python source/standalone/test_lk_ostrack/estimate_video.py \
  --rgb_dir /path/to/rgb --initial_boxes_file /path/to/initial_targets.json \
  --intrinsics_json /path/to/intrinsics.json --out_csv outputs/tracks.csv --disable_fine_tracking
```

启用 OSTrack 时将禁用参数换为 `--ostrack_repo /path/to/OSTrack --ostrack_checkpoint /path/to/OSTrack_ep0300.pth.tar`。
相机内参和初始框需来自实际场景，LK 不需要训练；此子集不包含 OSTrack 独立训练入口。

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p source/standalone/test_lk_ostrack/collect_multi_target_dataset.py \
  --headless --out_dir outputs/vision_dataset
python source/standalone/test_lk_ostrack/check_multi_target_dataset.py outputs/vision_dataset
python source/standalone/test_lk_ostrack/evaluate_multi_target_dataset.py \
  --dataset_dir outputs/vision_dataset --out_dir outputs/vision_eval --disable_fine_tracking
```

采集依赖本仓库 `test11` 及与你原项目一致的 SDK/资产环境。
本次没有加入 `demo.py`、`ray_caster_compat.py`、`recoil/` 或环境副本等本地该目录不存在的代码。

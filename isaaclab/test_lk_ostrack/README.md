# LK + OSTrack 多目标视觉

输入第一帧目标框，输出各目标的框、方向、置信度和主目标 ID。
多目标共享 LK 前后向光流，每目标使用 RANSAC/Kalman，主目标可由 OSTrack 精跟踪。
支持序列推理、Isaac Lab 数据采集、离线评估及固定/自适应精跟踪调度。

## 文件说明

| 文件或目录 | 作用 |
| --- | --- |
| `__init__.py` | 导出跟踪器、配置和公共数据结构。 |
| `demo.py` | 生成双目标合成图片，并调用真实序列推理入口。 |
| `estimate_video.py` | 单目标/多目标图片序列推理入口。 |
| `multi_target_tracker.py` | LK、前后向检查、RANSAC、Kalman、目标优先级及 OSTrack 调度。 |
| `camera_geometry.py` | 相机内参、目标框和像素/方向转换。 |
| `ostrack_tracker.py` | 官方 OSTrack 的序列和逐帧跟踪适配。 |
| `batched_ostrack.py` | 多环境共享一个 OSTrack 网络，分别维护目标状态。 |
| `online_vision.py` | 仿真中串行/批量在线 OSTrack 及失跟恢复。 |
| `multi_target_io.py` | 初始框读取、跟踪结果 CSV 和叠加图输出。 |
| `pipeline.py` | 图片加载、单目标方向/角速度计算、结果和计时输出。 |
| `collect_multi_target_dataset.py` | 多目标 RGB、视觉方向、RayCaster 测距和监督标签采集入口。 |
| `collect_velocity_text_dataset.py` | 静止机器人下的主目标文本采集入口，不保存 RGB。 |
| `check_multi_target_dataset.py` | 采集目录的结构和完整性检查入口。 |
| `isaacsim_dataset.py` | 序列数据写入、manifest 和完整性校验。 |
| `evaluate_multi_target_dataset.py` | 多目标 IoU、方向、测距融合位置误差和运行速度评估入口。 |
| `evaluate_direction_dataset.py` | 将数据集中各目标分别交给单目标 OSTrack，评估方向误差和耗时。 |
| `multi_target_collect_env.py` | 多目标场景、目标监督和按视觉方向进行测距的采集环境。 |
| `turret_aim_env.py` | 机器人、炮塔、相机及传感器的基础仿真环境。 |
| `turret_aim_env_collect.py` | 关闭训练奖励、处理预热和序列采集状态的采集环境。 |
| `ray_caster_compat.py` | 局部修正多 mesh RayCaster 的 mesh ID 维度，不修改 SDK。 |
| `rgb_visibility.py` | 检查目标框中实际渲染的目标颜色，辅助初始化可见性检查。 |
| `target_motion.py` | 目标在视场和距离约束内连续运动及碰撞反射。 |
| [recoil/](recoil/README.md) | 后坐载荷模型及仿真注入。 |
| [tests/](tests/README.md) | 跟踪、数据、评估和命令行回归测试。 |
| [lite3/](lite3/README.md)、[turret/](turret/README.md) | 本地机器人和炮塔 USD 资产。 |
| [ostrack/](ostrack/README.md) | OSTrack 推理源码和配套配置。 |
| `lite3_with_turret.usd`、`turret_with_edite.usd`、`turret.urdf` | 组合场景和炮塔描述。 |

## 环境与输入

安装方式见[总 README](../../README.md#安装)。纯 LK 演示不需要权重和仿真器；
OSTrack 推理额外需要 CUDA PyTorch 和 `.[ostrack]` 依赖。
从 [OSTrack 官方模型目录](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing)
准备与 `vitb_256_mae_ce_32x4_ep300` 对应的 `OSTrack_ep0300.pth.tar`，
放到 `checkpoints/OSTrack_ep0300.pth.tar`。同名其他配置权重不能混用，源码默认自动定位。

自己的图片需按时间排序、文件名补零；初始框使用第一帧像素 `xyxy` 坐标：

```json
{"targets": [{"id": 0, "bbox": [100, 80, 150, 130]}]}
```

相机内参 JSON 示例（替换为真实标定值）：

```json
{"fx": 600, "fy": 600, "cx": 320, "cy": 180, "width": 640, "height": 360}
```

## 运行命令

以下均从仓库根目录执行。CPU 演示与测试：

```bash
python -m test_lk_ostrack.demo
python -m pytest -q
```

演示输出到 `outputs/lk_ostrack_demo/`；重复运行需用 `--out_dir` 换一个空目录。
演示启用 OSTrack 时增加 `--ostrack_checkpoint checkpoints/OSTrack_ep0300.pth.tar`。

自己的多目标序列（纯 LK）：

```bash
python -m test_lk_ostrack.estimate_video \
  --rgb_dir /path/to/rgb \
  --initial_boxes_file /path/to/initial_targets.json \
  --intrinsics_json /path/to/intrinsics.json \
  --out_csv outputs/tracks.csv --disable_fine_tracking
```

默认帧率为 30 FPS，其他帧率用 `--fps` 指定。保存叠加图增加 `--overlay_dir outputs/overlay`。
启用 OSTrack 时，将 `--disable_fine_tracking` 替换为
`--ostrack_checkpoint checkpoints/OSTrack_ep0300.pth.tar`。
单目标入口用 `--initial_box x1 y1 x2 y2` 替代 `--initial_boxes_file`，且需要 OSTrack 权重。

仿真多目标数据采集：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test_lk_ostrack/collect_multi_target_dataset.py \
  --headless --target_height_constraint --out_dir outputs/vision_dataset
python -m test_lk_ostrack.check_multi_target_dataset outputs/vision_dataset
```

默认 4 个环境、每环境 3 个目标，收集 20 条 120 帧序列；未提供 checkpoint 时使用 LK。
示例打开非默认的高度约束，适合不加载行走策略的静止采集。
移动机器人采集增加 `--locomotion_policy_path /path/to/policy.pt`。
需要精跟踪时增加 `--ostrack_checkpoint checkpoints/OSTrack_ep0300.pth.tar`。

只采集主目标文本：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test_lk_ostrack/collect_velocity_text_dataset.py \
  --headless --target_height_constraint --out_dir outputs/velocity_text
```

该入口不保存 RGB，不能用于需要图片的离线跟踪评估，也不自动生成速度训练用 TAQ 数据块。

多目标离线评估：

```bash
python -m test_lk_ostrack.evaluate_multi_target_dataset \
  --dataset_dir outputs/vision_dataset \
  --out_dir outputs/vision_eval --disable_fine_tracking
```

启用 OSTrack 同样将禁用参数替换为 checkpoint 参数。
默认使用 GT 距离/速度计算目标优先级，纯视觉回退实验增加 `--kinematics_source vision`。
位置误差融合的是采集时记录的测距，不会离线重新模拟雷达。

单目标 OSTrack 方向评估：

```bash
python -m test_lk_ostrack.evaluate_direction_dataset \
  --dataset_dir outputs/vision_dataset \
  --ostrack_checkpoint checkpoints/OSTrack_ep0300.pth.tar
```

该入口默认输出到 `test12_outputs/direction_dataset_eval/`，可用 `--out_dir` 更改。

## 训练与补充文档

LK/RANSAC/Kalman 无需训练。本分支只保留 OSTrack 推理源码，不提供 OSTrack 训练入口。
决策网络训练见[决策模块](../test7_2_weight/README.md#训练)。
自适应调度见 [ADAPTIVE_OSTRACK_DESIGN.md](ADAPTIVE_OSTRACK_DESIGN.md)，指标见
[TRACKING_SPEED_EVALUATION.md](TRACKING_SPEED_EVALUATION.md)，第三方来源见
[THIRD_PARTY.md](THIRD_PARTY.md)，历史验证记录见 [VALIDATION.md](VALIDATION.md)。

# 速度误差来源验证工具

本目录已经实现两组配对消融、逐帧 audit、流式 checkpoint 回放、episode 级统计和异常事件绘图。核心方案现已改为“固定相机 / 机身挂载相机”对照，详细实验依据见 `VELOCITY_ERROR_SOURCE_VALIDATION.md`。

> **实现状态：可以开始 pilot。** A/B 已迁移为固定 proxy 相机，四组均保留相同机器人物理过程；静态编译和单元测试已通过。正式大规模采集仍需等待 `pilot_fixed_camera_v1` 的相机、root 和目标轨迹门限全部通过。

## 条件定义

第一组用于拆分上游跟踪与相机运动。四组机器狗均加载相同 locomotion policy、执行相同零速度指令并保留真实物理运动：

| 条件 | 方向观测 | 实验相机 | 机器狗 |
| --- | --- | --- | --- |
| A | GT 投影 | 固定世界位姿 proxy | 零指令物理站立 |
| B | LK + OSTrack | 固定世界位姿 proxy | 零指令物理站立 |
| C | GT 投影 | torso 挂载相机 | 零指令物理站立 |
| D | LK + OSTrack | torso 挂载相机 | 零指令物理站立 |

第二组在物理站立条件下拆分方向与距离：

| 条件 | 方向 | 距离 |
| --- | --- | --- |
| P0 | 真实 | 真实 |
| P1 | GT | 真实路径（raycast 使用 GT 方向） |
| P2 | 真实 | GT |
| P3 | GT | GT |

P0/P1/P2/P3 默认使用机身挂载相机；它们不参与固定相机/移动相机交互项。

验证模式使用所有条件共享的名义参考相机扇区和固定目标锚点生成目标轨迹，并关闭依赖实时相机位姿的目标距离壳。这样既保证初始目标位于公共视场，也不会让相机模式改变目标世界轨迹。逐帧 audit 必须同时保存目标世界坐标、实验相机位姿、真实挂载相机位姿和 robot root 状态；最终校验同时拒绝目标轨迹不一致、机器人轨迹不一致、固定相机漂移或挂载相机同步错误的数据。

核心验证 episode 使用共同固定时长。单个相机条件下的 out-of-view、tracker lost 或无效测量只记录为观测事件，不得触发物理环境 reset；否则 A/B/C/D 会从该时刻起失去轨迹配对。分析同时报告全时段结果和共同有效帧结果，不能通过删除 lost 帧隐藏相机运动造成的失败。

同一条配对轨迹的四个条件共用最早出现有效方向和距离观测的时刻作为离线速度误差计分起点；单独采集的 stream 则从自身首次有效观测开始。起点前的初始化帧仍保存在 stream 和 `frame_metrics.csv`，但 `valid=0`，不进入汇总、配对效应及误差事件。共同起点之后，即使某个条件仍然出画面或 tracker lost，该帧也照常计入。逐帧字段 `initialization_frame` 和 `evaluation_started` 标明这个边界，`summary.json` 记录被排除且原本有速度标签的初始化帧数。

A/B 只固定用于渲染、GT 投影、可见性、range 起点和坐标转换的 proxy 相机，不固定机器狗，也不把本体速度或速度标签置零。旧的固定 root 实验只保留为平台运动总效应的补充消融。

## 1. 生成与执行采集任务

在仓库根目录先只生成 manifest 并检查命令：

```bash
/home/jby/miniconda3/envs/isaac_lab_232/bin/python \
  source/standalone/test11/velocity_error_validation/tools/collect_paired_conditions.py \
  --config source/standalone/test11/velocity_error_validation/configs/pilot.yaml
```

确认 checkpoint、相机模式和设备参数正确后，加 `--run` 执行。固定相机 pilot 写入新的 `outputs/pilot_fixed_camera_v1`。`outputs/pilot`、`pilot_v2`、`pilot_v3`、`pilot_v4` 都属于旧固定机身方案或其调试数据，不得混入新的核心因果分析。正式 A/B/C/D 使用 `configs/formal.yaml`，写入 `outputs/formal_fixed_camera_v1`；P0/P1/P2/P3 使用 `configs/perception.yaml`。输出目录非空时启动器默认拒绝覆盖；只有确认要继续同一实验定义的已有目录时才使用 `--allow-existing`。

```bash
/home/jby/miniconda3/envs/isaac_lab_232/bin/python \
  source/standalone/test11/velocity_error_validation/tools/collect_paired_conditions.py \
  --config source/standalone/test11/velocity_error_validation/configs/pilot.yaml \
  --run
```

manifest 记录完整命令、seed、pair ID、配置 SHA-256 和 Git commit。每个 episode 的 stream metadata 必须包含 `condition`、`pair_id`、`trajectory_seed` 和 `trajectory_uid`，否则执行后校验失败。pilot 还要求至少 12 个完整 A/B/C/D trajectory pair、每个条件在完整 pair 上至少 3,000 个非 INVALID 帧；不完整 pair 会被明确列出并从所有配对效应中排除。

每个 episode 额外写入 `paired_state_audit.jsonl`，覆盖包括 tracker lost/out-of-view 在内的所有物理帧。最终校验要求四组时间轴完全一致，并检查目标世界轨迹、robot root 位姿/速度、A/B 固定相机漂移和背景光流、C/D 挂载相机同步误差。任一门限失败都会拒绝整批数据。

## 2. 主分析

```bash
/home/jby/miniconda3/envs/isaac_lab_232/bin/python \
  source/standalone/test11/velocity_error_validation/tools/analyze_error_sources.py \
  --data-root source/standalone/test11/velocity_error_validation/outputs/formal_fixed_camera_v1 \
  --checkpoint source/standalone/test11/speed_prediction/stage1_v2_runs/09-17-17-16/best_macro_p95.pt \
  --out-dir source/standalone/test11/velocity_error_validation/reports/formal_fixed_camera_v1
```

采集器把 audit 写在每个 stream 同目录的 `velocity_error_audit.jsonl`，所以通常不必指定 `--audit-root`；分析器默认从 `--data-root` 自动发现。GT/audit 字段只在模型推理完成后按 `(episode_uid, frame_index)` 合并，模型输入严格限定为 stream 中的 24 维特征、source ID 和 valid mask。

主要输出：

- `frame_metrics.csv`：预测、GT、baseline、sigma、gate、方向/range/位置误差、机身运动和运动学残差；
- `episode_metrics.csv`：每个 episode 的误差统计；
- `grouped_metrics.csv`：condition、scenario、source、距离、目标速度/加速度、跟踪误差和抖动分位组，并同时给出 GRU、加权物理基线、零速度基线和 GT 位置有限差分 oracle；
- `paired_effects.csv`：A/B/C/D 与 P0/P1/P2/P3 的 episode 配对差值及 95% bootstrap CI；
- `event_catalog.csv`：误差阈值、top 1%、episode max，以及跟踪/range/角加速度/背景光流尾部事件；
- `lagged_correlations.csv`：候选原因相对速度误差的 `-5～+10` 帧相关；
- `time_offset_scan.csv`：按总体、condition、source 的 `-100～+100 ms` 标签偏移扫描；
- `summary.json`：总体指标、checkpoint 哈希、配对覆盖和目标轨迹一致性结果。

旧数据没有 audit 时仍可分析模型误差，但 `trajectory_validation.status` 会明确标记为未校验，不能据此宣称因果配对成立。

## 3. 异常事件图

```bash
/home/jby/miniconda3/envs/isaac_lab_232/bin/python \
  source/standalone/test11/velocity_error_validation/tools/plot_error_events.py \
  --report-dir source/standalone/test11/velocity_error_validation/reports/formal \
  --max-events 50
```

图片写入 `figures/`，包含预测/GT、误差跳变、跟踪质量、测距/位置、机身抖动、sigma/gate 和 source/reset。

## 4. 自检

```bash
/home/jby/miniconda3/envs/isaac_lab_232/bin/python -m unittest discover \
  -s source/standalone/test11/velocity_error_validation/tests -v
```

单元测试覆盖 JSONL audit、向量导数、两组 factorial 差分方向和目标轨迹不一致检测。由于采集端依赖 Isaac Sim，完整 A/B/C/D 采集仍需在可启动仿真的 GPU 环境中先跑 pilot；静态测试不能替代该检查。

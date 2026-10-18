# 完整感知与控制链

连接 RGB 跟踪、测距、速度估计和决策网络，将结果写回仿真环境。
单目标链负责坐标转换与观测组装，SS 多目标模式增加目标调度、重试和清场诊断。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 导出集成模块接口。 |
| `shooting_chain.py` | 定义机器人状态、相机外参和控制链；组装 28 维 Actor 观测并生成瞄准/开火命令。 |
| `factory.py` | `build_chain()`：组合视觉跟踪器、速度估计器和调用方提供的决策策略。 |
| `model_loaders.py` | 加载 OSTrack、普通 GRU/TAQ 速度模型及 TorchScript 决策策略。 |
| `policy_adapters.py` | 将 RL-Games player 接入控制链并保留其循环状态。 |
| `legacy_gru_adapter.py` | 将普通 GRU 的滚动推理接口适配为控制链的速度估计接口。 |
| `isaac_adapter.py` | 把控制链的目标预测写回 Isaac Lab 炮塔环境。 |
| `isaac_training_bridge.py` | 训练/评估时从相机和测距生成目标状态，保持 PPO 对决策前向和循环状态的管理。 |
| `static_salvo_perception.py` | SS 目标池的一次性框初始化及后续 RGB 多目标跟踪。 |
| `salvo_engagement.py` | 按视觉 ID 和距离调度交战，处理超时、重试及切换。 |
| `salvo_ablation.py` | A/B/C/D 消融开关、固定布局读写和目标切换时的 Actor 状态重置。 |
| `salvo_control_snapshot.py` | 只读控制/GT 快照，供诊断使用，不作为策略输入。 |
| `salvo_diagnostics.py` | 逐步记录 Actor 输入、开火门控和阻止开火的原因。 |
| `episode_video.py` | 按 episode 录像，叠加感知相机与命中后的短暂停留画面。 |
| `video_scene_view.py` | 录像用观察视角、轨迹投影和弹道标注。 |

## 训练与测试

本目录是集成库，没有独立训练入口。决策训练使用
[test7_2_weight 的完整感知微调命令](../test7_2_weight/README.md#训练)。
先按[总 README](../../README.md#安装)准备环境和默认权重，以下从仓库根目录执行。

完整感知评估（单环境以降低显存占用，默认 SD 场景）：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test7_2_weight/eval_with_track_gt.py \
  --headless --num_envs 1 --full_chain_perception \
  --checkpoint /path/to/run_or_checkpoint \
  --locomotion_policy /path/to/policy.pt
```

需要录像时增加 `--record_video`；默认保存到本目录的 `video/<时间戳>/`，
可用 `--video_dir` 改到其他新目录。需要静态单目标时增加 `--case ss`。

SS 多目标连续清场：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test7_2_weight/eval_with_track_gt.py \
  --headless --num_envs 1 --episodes 1 --static_salvo \
  --checkpoint /path/to/run_or_checkpoint \
  --locomotion_policy /path/to/policy.pt
```

`--static_salvo` 自动启用完整感知和 SS，默认 100 个目标、D 方案；一个 episode 对应整个目标池。
未指定报告路径时，JSON 和逐步 CSV 写到 `outputs/static_salvo/`；启用录像后默认写入录像目录。
`--salvo_report` 可指定报告文件。详细诊断字段见 [SS_CONTROL_DIAGNOSTICS.md](SS_CONTROL_DIAGNOSTICS.md)。

完整感知会使用重置后的初始框；SS 对整个目标池做一次框初始化，后续由 RGB 跟踪。
它不是一个自动检测器，也不能把结果解释为完全不使用初始化标注的目标发现能力。

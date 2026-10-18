# 普通 13 维 GRU

输入为位置、位置增量、机器人线/角速度和运动标志，输出三维速度；保持原网络结构。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 子包定义。 |
| `speed_prediction_model.py` | 输入归一化、GRU 和速度输出头。 |
| `taq_dataset.py` | TAQ-v2 转为 13 维窗口，按 episode 划分、因果平滑和变长批处理。 |
| `train_gru_velocity.py` | 历史数据/TAQ-v2 短窗训练、归一化、模型选择及可选留出测试。 |
| `train_gru_velocity_episodes.py` | 完整回合 BPTT、旧权重初始化及独立测试。 |
| `evaluate_gru_velocity.py` | TAQ-v2 权重的独立窗口/完整回合测试，读取指定清单划分并检查泄漏。 |
| `test_gru_velocity.py` | 历史 26 维 chunk 的外部测试，不能用于 TAQ-v2。 |
| `online_gru_velocity.py` | 权重校验及 rolling/full 在线推理。 |
| `evaluate_vision_gru.py` | 仿真中实际视觉/测距接入冻结 GRU，记录逐帧误差。 |
| `compare_vision_gru_initialization.py` | 在相同已保存观测上比较 first-valid 与 zero-filled 初始化。 |

## 训练与离线测试

从仓库根目录执行，安装见[上级说明](../README.md#环境)：

```bash
python -m test11.speed_prediction.legacy_gru.train_gru_velocity_episodes --data_dirs /path/to/episodes
python -m test11.speed_prediction.evaluate_gru_velocity \
  --checkpoint /path/to/best_model.pt --manifest /path/to/data_manifest.json
```

完整回合默认按 80/10/10 划分；`test_metrics.json` 在模型选择完成后生成。
短窗训练及 `--context window` 测试见[上级训练说明](../README.md#训练)。
预处理和归一化从 checkpoint 读取；窗口与完整回合的误差统计口径不同。

## 实时视觉测试

准备 Isaac Lab、行走策略、默认位置的速度和 OSTrack 权重，从仓库根目录执行：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test11/speed_prediction/legacy_gru/evaluate_vision_gru.py \
  --headless --locomotion_policy /path/to/policy.pt
```

默认速度权重为 `checkpoints/velocity/best_model.pt`，OSTrack 为 `checkpoints/OSTrack_ep0300.pth.tar`，
可用 `--checkpoint`、`--ostrack_checkpoint` 更改。默认 rolling 历史；评估完整历史增加 `--context full`。
结果写入 `outputs/velocity/vision_eval/<时间戳>/`，包含逐帧 CSV、分组指标和初始化延迟。
此测试使用一次 GT 框初始化，后续输入来自视觉和测距，GT 速度仅用于评分。

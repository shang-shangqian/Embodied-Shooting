# TAQ-GRU

将 24 维连续特征、跟踪来源、时间间隔和运动补偿基线结合，预测速度与不确定度。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 子包定义。 |
| `time_aware_velocity_model.py` | 网络、质量门控、归一化和速度基线。 |
| `online_taq_velocity.py` | 观测/输出结构和按目标维护历史的在线推理。 |
| `taq_velocity_training.py` | 掩码损失、批处理搬运和误差统计。 |
| `train_taq_velocity.py` | TAQ-v2 短窗训练，保存验证集选择的权重。 |
| `train_taq_velocity_episodes.py` | 完整回合训练，复用参考清单的 train/val/test。 |
| `train_stage2_balanced.py` | ideal 与退化数据按受控比例采样训练。 |
| `evaluate_taq_velocity.py` | 短窗口离线评估。 |
| `evaluate_taq_velocity_streaming.py` | 按完整回合时间顺序评估，按保存的状态重置标志处理历史。 |

## 训练与测试

安装和采集见[上级说明](../README.md)。从仓库根目录执行短窗训练：

```bash
python -m test11.speed_prediction.train_taq_velocity \
  --train_dirs /path/to/train_chunks --val_dirs /path/to/val_chunks
python -m test11.speed_prediction.evaluate_taq_velocity \
  --data_dirs /path/to/test_chunks --checkpoint /path/to/taq_model.pt
```

短窗训练的 train/val 两组分别必须同时包含 SS 和 SD，并保持 episode 隔离。
仅 SD 数据使用下面的完整回合入口。

完整回合训练复用普通 GRU 已保存的 episode 划分：

```bash
python -m test11.speed_prediction.taq_gru.train_taq_velocity_episodes \
  --reference_manifest /path/to/data_manifest.json
```

模型选择完成后自动保存 `test_metrics.json`。独立完整流评估见[上级测试命令](../README.md#测试)。
新训练默认使用 odometry-aligned v2 基线；旧权重沿用其声明的 v1 基线，版本和归一化不能混搭。

# 速度模块测试

| 文件 | 验证内容 |
| --- | --- |
| `test_legacy_gru_taq_data.py` | 13 维适配、episode 隔离、归一化及短窗训练 smoke。 |
| `test_legacy_gru_episodes.py` | 完整回合训练、梯度、旧权重初始化和部署兼容。 |
| `test_evaluate_gru_velocity.py` | 独立 GRU 测试入口的清单隔离、评分及格式检查。 |
| `test_online_legacy_gru.py` | rolling/full 在线推理、初始化与实时视觉评分。 |
| `test_time_aware_velocity.py` | TAQ 数据、特征、网络、损失、采样和流评估。 |
| `test_aligned_velocity_baseline.py` | 运动补偿基线、因果性及在线/离线一致性。 |
| `test_taq_full_episode.py` | TAQ 完整回合输出、梯度连续性和输入隔离。 |
| `test_stage2_degradation.py` | 退化采样、数据协议和随机性控制。 |

安装 `.[velocity,dev]`，从仓库根目录执行：

```bash
python -m pytest -q isaaclab/test11/speed_prediction/tests
```

测试使用临时合成数据，包含短训练，不需要仿真器、CUDA 或真实模型权重；不代表真实数据精度。
完整采集、训练和独立评估见[上级说明](../README.md)。

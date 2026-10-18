# TAQ-GRU 速度模型

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | TAQ-GRU 子包定义。 |
| `time_aware_velocity_model.py` | 时间与质量感知 GRU；融合连续特征、来源编码和运动补偿基线，输出速度、对角不确定度及可靠性门控。 |
| `online_taq_velocity.py` | 定义 `TAQObservation` 和在线估计结果；为每个目标维护历史与隐藏状态，构建特征并执行因果推理。 |

本分支没有 TAQ 独立训练脚本。使用 TAQ-v2 权重，且基线版本、归一化与输入变换元数据必须匹配。

从仓库根目录加载自己的 TAQ 权重：

```bash
python -c "from integrated_shooting_chain.model_loaders import load_velocity_estimator; print(load_velocity_estimator('/path/to/taq_model.pt', backend='taq_gru'))"
```

该命令只检查加载。整链测试使用[上级命令](../README.md#测试)，增加
`--velocity_backend taq_gru --full_chain_velocity_checkpoint /path/to/taq_model.pt`。

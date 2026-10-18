# 普通 GRU 速度模型

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 普通 GRU 子包定义。 |
| `speed_prediction_model.py` | `VelocityPredictorGRU`：输入归一化、GRU 和三维速度输出头，输入为 `[B,T,13]`。 |
| `taq_dataset.py` | 将 TAQ-v2 数据转为普通 GRU 窗口，提供按 episode 划分、因果平滑、批处理和窗口推理。 |
| `online_gru_velocity.py` | 校验并加载权重，按目标维护滚动历史，将在线位置序列转为速度预测。 |

本分支不含该网络的训练入口。可从仓库根目录运行无权重的前向检查：

```bash
python -c "import torch; from test11.speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU; print(VelocityPredictorGRU()(torch.zeros(1, 8, 13)).shape)"
```

输出应为 `torch.Size([1, 3])`。该检查只验证网络可运行。
真实权重加载和整链测试见[上级说明](../README.md#测试)，默认后端就是 `legacy_gru`。

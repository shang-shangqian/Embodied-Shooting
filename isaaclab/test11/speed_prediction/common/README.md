# 共享数据与特征工具

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 共享工具包定义。 |
| `time_aware_velocity_data.py` | 定义 TAQ-v2 标签、跟踪来源和 24 维特征；实现坐标变换、因果差分、运动补偿速度基线、数据块读写、数据集和归一化统计。 |

本目录是函数库，没有独立训练/评估入口。数据标签表示相对平移速度，不能直接用旋转坐标系中的位置差分代替。

从仓库根目录检查数据格式常量：

```bash
python -c "from test11.speed_prediction.common.time_aware_velocity_data import TAQ_SCHEMA_VERSION, CONTINUOUS_FEATURE_DIM; print(TAQ_SCHEMA_VERSION, CONTINUOUS_FEATURE_DIM)"
```

应输出 `taq_velocity_v2 24`。权重加载和在线联调见[上级说明](../README.md#测试)。

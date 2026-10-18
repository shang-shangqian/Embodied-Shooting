# 共享数据与物理接口

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 共享工具包定义。 |
| `time_aware_velocity_data.py` | TAQ-v2 数据/标签、24 维因果特征、速度基线、数据集和归一化。 |
| `stage2_degradation.py` | 图像/跟踪/测距/时序/本体观测退化配置、模拟和计数。 |
| `sensor_time_contract.py` | 传感器包、物理采样门和每回合预热协议。 |
| `filter_state_output.py` | 统一滤波状态输出及速度相关接口。 |
| `target_motion.py` | 固定视场平面和距离壳内连续运动、碰撞时间反射及初始化。 |

本目录由[采集、训练和测试入口](../README.md)调用，没有独立训练入口。
验证相关数值和数据协议，从仓库根目录执行：

```bash
python -m pytest -q isaaclab/test11/speed_prediction/tests
```

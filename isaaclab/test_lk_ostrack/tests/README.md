# 视觉模块测试

| 文件 | 验证内容 |
| --- | --- |
| `test_multi_target_tracker.py` | 光流跟踪、遮挡/预测、优先级切换、精跟踪调度及结果读写。 |
| `test_isaacsim_dataset.py` | 序列写入、manifest、完整性校验和不完整序列处理。 |
| `test_evaluate_multi_target_dataset.py` | 框 IoU、方向/位置误差及评估统计。 |
| `test_rgb_visibility.py` | 框内目标颜色、遮挡与输入形状检查。 |
| `test_cli.py` | 从其他工作目录运行已安装的演示入口，以及缺少可选依赖时提前报错。 |

先按[总 README](../../../README.md#安装)安装 `.[dev]`，从仓库根目录执行：

```bash
python -m pytest -q
```

仅测跟踪器可运行 `python -m pytest -q isaaclab/test_lk_ostrack/tests/test_multi_target_tracker.py`。
测试无需仿真器、CUDA 或模型权重，不代表完整仿真训练/评估已通过；本目录没有训练入口。

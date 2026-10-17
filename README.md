# 具身射击系统

本分支新增可安装的 [LK + OSTrack 多目标视觉模块](isaaclab/test_lk_ostrack/README.md)，
包含 CPU 演示、OSTrack 精跟踪、Isaac Lab 采集和离线评估。环境配置和权重准备完成后，
可直接运行现成入口，无需另写脚本。

```bash
python -m pip install -e '.[dev]'
python -m test_lk_ostrack.demo
```

## 项目结构

| 目录 | 说明 |
|------|------|
| `test12/` | 方向预测模块（OSTrack） |
| `test4/` | 方向预测模块（CNN） |
| `test7_1/` | 完整代码 |
| `test7_2_weight/` | 强化学习决策网络(GT值) |
| `test_temp/` | 速度预测模块 |

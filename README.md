# LK + OSTrack 多目标视觉模块

本分支仅保留从 `source/standalone/test_lk_ostrack` 整理的必要代码和资源。
[完整运行说明](isaaclab/test_lk_ostrack/README.md)
包含 CPU 演示、OSTrack 精跟踪、Isaac Lab 采集和离线评估。环境配置和权重准备完成后，
可直接运行现成入口，无需另写脚本。

```bash
python -m pip install -e '.[dev]'
python -m test_lk_ostrack.demo
```

代码、模型资源及第三方推理源码位于 `isaaclab/test_lk_ostrack/`。
安装依赖由根目录 `pyproject.toml` 管理。数据集、运行输出和模型权重不随仓库提供。

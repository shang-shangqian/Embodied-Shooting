# LK + OSTrack 与炮塔决策网络

本分支包含 LK + OSTrack 视觉模块，以及从 `source/standalone/test7_2_weight`
整理的炮塔决策训练、评估代码和必要运行依赖。
[完整运行说明](isaaclab/test_lk_ostrack/README.md)
包含 CPU 演示、OSTrack 精跟踪、Isaac Lab 采集和离线评估。环境配置和权重准备完成后，
可直接运行现成入口，无需另写脚本。

```bash
python -m pip install -e '.[dev]'
python -m test_lk_ostrack.demo
```

代码、模型资源及第三方推理源码位于 `isaaclab/test_lk_ostrack/`。
安装依赖由根目录 `pyproject.toml` 管理。数据集、运行输出和模型权重不随仓库提供。

炮塔决策模块位于 `isaaclab/test7_2_weight/`，见其
[安装与运行说明](isaaclab/test7_2_weight/README.md)。
`isaaclab/integrated_shooting_chain/` 和 `isaaclab/test11/speed_prediction/` 仅包含决策模块调用的控制链、
录像和速度估计运行代码，不包含其他实验目录、训练记录或权重。

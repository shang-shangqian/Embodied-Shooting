# LK + OSTrack 感知与炮塔决策

`lk-ostrack-standalone` 分支包含视觉跟踪、速度估计、感知控制链和炮塔决策代码。
整体流程：RGB 图像 → LK/OSTrack 跟踪 → 方向与测距融合 → GRU 速度估计 → PPO 决策 → 瞄准与开火。

## 目录说明

| 目录 | 对应模块 |
| --- | --- |
| [isaaclab/test_lk_ostrack/](isaaclab/test_lk_ostrack/README.md) | 多目标视觉跟踪、仿真数据采集和离线评估；含模型资产、后坐力和测试子目录。 |
| [isaaclab/test11/speed_prediction/](isaaclab/test11/speed_prediction/README.md) | 普通 GRU、TAQ-GRU 速度模型及在线推理；本分支不含独立训练脚本。 |
| [isaaclab/integrated_shooting_chain/](isaaclab/integrated_shooting_chain/README.md) | 连接视觉、测距、速度和决策，处理多目标调度、诊断及录像。 |
| [isaaclab/test7_2_weight/](isaaclab/test7_2_weight/README.md) | 炮塔决策网络、PPO 训练、GT/完整感知评估和 SS 多目标清场。 |
| `pyproject.toml` | Python 依赖、模块安装和命令行入口。 |

`/path/to/...` 替换为实际路径。


## 权重与数据

本分支不附带模型权重、训练数据和历史输出。完整感知模式默认读取：

| 路径或参数 | 内容 |
| --- | --- |
| `--locomotion_policy /path/to/policy.pt` | 兼容 Lite3 观测和关节顺序的行走 TorchScript 策略。 |
| `--checkpoint /path/to/run_or_checkpoint` | 决策 `.pth` 或包含 `nn/` 的训练目录。 |
| `checkpoints/velocity/best_model.pt` | 默认的 13 维普通 GRU 速度权重。 |
| `checkpoints/OSTrack_ep0300.pth.tar` | 与 `vitb_256_mae_ce_32x4_ep300` 配套的 OSTrack 权重。 |


## 常用入口

- [视觉演示、序列推理、采集与评估](isaaclab/test_lk_ostrack/README.md#运行命令)
- [决策网络训练与评估](isaaclab/test7_2_weight/README.md#训练)
- [完整感知链测试、SS 清场与录像](isaaclab/integrated_shooting_chain/README.md#训练与测试)
- [速度模型加载和后端选择](isaaclab/test11/speed_prediction/README.md#测试)

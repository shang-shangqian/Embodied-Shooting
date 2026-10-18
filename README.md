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

各目录 README 列出代码文件用途和运行命令。所有命令均从**本仓库根目录**执行；
`/path/to/...` 替换为实际路径。命令省略默认参数，更多选项使用对应入口的 `--help` 查看。

## 安装

仅运行 CPU 视觉演示和回归测试，使用 Python 3.10 或 3.11：

```bash
python -m pip install -e '.[dev]'
python -m test_lk_ostrack.demo
python -m pytest -q
```

演示结果位于 `outputs/lk_ostrack_demo/`；重复运行时用 `--out_dir` 指定新的空目录。

仿真采集、决策训练和评估使用已配置好的 **Isaac Lab v2.3.2 + Isaac Sim 5.1.0 + Python 3.11**
环境及 NVIDIA RTX GPU。在该环境中执行：

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab-2.3.2
python -m pip install -e '.[decision,ostrack,dev]'
```

`ISAACLAB_PATH` 指向外部 Isaac Lab SDK，需已配置其 `_isaac_sim` 和 Python 环境。
本仓库的 `isaaclab/` 是实验代码容器，不是完整 SDK；无需将代码复制进 SDK。
已有 PyTorch/torchvision 应与仿真环境匹配；基础依赖使用 OpenCV headless。

## 权重与数据

本分支不附带模型权重、训练数据和历史输出。完整感知模式默认读取：

| 路径或参数 | 内容 |
| --- | --- |
| `--locomotion_policy /path/to/policy.pt` | 兼容 Lite3 观测和关节顺序的行走 TorchScript 策略，决策训练/评估必需。 |
| `--checkpoint /path/to/run_or_checkpoint` | 决策 `.pth` 或包含 `nn/` 的训练目录，评估和恢复训练时使用。 |
| `checkpoints/velocity/best_model.pt` | 默认的 13 维普通 GRU 速度权重，需包含 TAQ-v2 标签及预处理元数据。 |
| `checkpoints/OSTrack_ep0300.pth.tar` | 与 `vitb_256_mae_ce_32x4_ep300` 配套的 OSTrack 权重。 |

自定义速度权重路径：训练用 `--velocity_model`，完整感知评估用 `--full_chain_velocity_checkpoint`。
使用 TAQ-v2 网络时增加 `--velocity_backend taq_gru`，并提供该网络对应的权重。
OSTrack 权重来源及输入格式见[视觉模块](isaaclab/test_lk_ostrack/README.md)。

## 常用入口

- [视觉演示、序列推理、采集与评估](isaaclab/test_lk_ostrack/README.md#运行命令)
- [决策网络训练与评估](isaaclab/test7_2_weight/README.md#训练)
- [完整感知链测试、SS 清场与录像](isaaclab/integrated_shooting_chain/README.md#训练与测试)
- [速度模型加载和后端选择](isaaclab/test11/speed_prediction/README.md#测试)

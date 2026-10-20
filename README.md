# LK/OSTrack、炮塔决策与速度估计

`lk-ostrack-standalone` 当前发布下列四个本地目录中的必要代码、测试及运行所需模型资产。
仓库路径与本地路径一致，已登记的代码文件按原内容复制，不改变算法、物理参数或导入路径。

| 目录 | 作用 |
| --- | --- |
| [source/standalone/test_lk_ostrack/](source/standalone/test_lk_ostrack/README.md) | LK/OSTrack 跟踪、图片序列推理、多目标采集和离线评估。 |
| [source/standalone/test7_2_weight/](source/standalone/test7_2_weight/README.md) | PPO 决策网络、GT 训练/评估及原有感知调用接口。 |
| [source/standalone/test11/](source/standalone/test11/README.md) | 速度数据采集环境、普通 GRU/TAQ 训练、测试及数据工具。 |
| [source/standalone/integrated_shooting_chain/](source/standalone/integrated_shooting_chain/README.md) | 完整感知决策桥、SS 连续清场调度、诊断、录像与相关测试。 |

不包含 `test4`、`model_with_recoil` 等其他实验目录，也不复制它们的文件到上述目录中。
历史数据、权重、日志、测试结果和视频不上传。第三方 OSTrack 仅保留已有推理依赖子集及原许可证。

## 环境与命令

命令均从本仓库根目录执行。已有数据的速度训练/评估使用 Python 3.10/3.11 和 PyTorch；
仿真使用与原项目匹配的 Isaac Lab v2.3.2 / Isaac Sim 环境。

```bash
python -m pip install -e '.[velocity,dev]'
export ISAACLAB_PATH=/absolute/path/to/IsaacLab-2.3.2
```

仿真/PPO 依赖使用 `.[decision,ostrack,dev]`。安装本仓库不会安装或修改 Isaac Lab SDK。
行走策略、决策权重、速度权重和 OSTrack 权重需自行准备，原代码中的默认实验路径按实际情况覆盖。

## 当前发布范围的限制

- GT 决策训练/评估的命令见决策模块 README。
- 完整感知决策、SS 连续清场、诊断和录像所需的 `integrated_shooting_chain` 源码已包含；模型权重与实验布局文件需自行准备。
- 旧 CNN 感知分支引用 `test4`；可选后坐力功能引用 `model_with_recoil`，此次也不包含。
- 各模块对 SDK、本地资产和环境配置的要求沿用原项目；本分支不是自带全部外部依赖的 SDK。

## SS 连续清场调度

`--case ss --static_salvo` 默认采用 D 流程：切换目标时清空 Actor GRU，等待新目标的感知与速度就绪后再决策。
每次切换从本轮尚未尝试的可用目标中动态选择视觉估计距离最近的目标；失败目标进入下一轮重试池，
本轮其余可用目标处理完后才参与重试。连续五枪未命中或选中后满 3 个仿真秒未命中即切出，
暂缓至少 1 秒，每目标最多尝试 3 次。完整参数与报告字段见射击链 README。

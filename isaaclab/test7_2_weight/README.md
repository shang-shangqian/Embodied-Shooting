# 炮塔决策网络：训练与评估

从本地 `source/standalone/test7_2_weight` 整理，包含全部 8 个原有 Python 文件、
机器人与炮台 USD/URDF，以及训练和评估实际导入的依赖。
支持 GT 训练、LK/OSTrack + 测距 + 速度 GRU 的完整感知训练/评估、SS 多目标清场及录像。

## 环境与权重

在已配置 Isaac Sim 5.1.0、Isaac Lab v2.3.2 和 Python 3.11 的环境中，
从本仓库根目录执行：

```bash
conda activate isaac_lab_232
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
python -m pip install -e '.[decision,ostrack]'
```

`ISAACLAB_PATH` 指向自己的 Isaac Lab SDK 根目录；按照 SDK 安装流程配置其
`_isaac_sim` 链接和环境。无须复制本仓库文件进 SDK。

权重不上传，按模式自行准备：

| 权重 | 用途 |
| --- | --- |
| Lite3 导出的 TorchScript `policy.pt` | 所有模式的腿部控制；须匹配本环境 Lite3 的观测和 12 关节顺序 |
| 决策网络 `.pth` checkpoint 或包含 `nn/` 的训练目录 | 评估或恢复训练；从头训练时不需要 |
| `checkpoints/velocity/best_model.pt` | 完整感知模式的速度估计，默认使用带有对应预处理元数据的 legacy GRU 权重 |
| `checkpoints/OSTrack_ep0300.pth.tar` | OSTrack 精跟踪；配置为 `vitb_256_mae_ce_32x4_ep300` |

所有路径均可通过命令行覆盖；OSTrack 权重下载方式见
[视觉模块说明](../test_lk_ostrack/README.md)。关闭精跟踪使用 `--no-fine_tracking`。
如使用 TAQ-v2 速度权重，传入 `--velocity_backend taq_gru`，不能混用两种网络的权重。
本次上传未包含这些自训权重的下载服务。

## 从头训练（GT）

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test7_2_weight/train_turret_mixed_gt.py \
  --headless --device cuda:0 --training_mode gt \
  --locomotion_policy /path/to/policy.pt
```

训练输出写入 `runs/test7_2_weight/`。恢复训练时增加
`--checkpoint /path/to/run_or_checkpoint --which best`。
PPO 参数由 `ppo_training_config.py` 提供，入口 `--help` 可查看全部参数。

## GT 评估

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test7_2_weight/eval_with_track_gt.py \
  --headless --num_envs 1 \
  --checkpoint /path/to/run_or_checkpoint --which best \
  --locomotion_policy /path/to/policy.pt \
  --case ss --episodes 10
```

场景：`ss` 静止机器人/静止目标，`sd` 静止机器人/动态目标，
`ds` 动态机器人/静止目标，`dd` 动态机器人/动态目标，`mixed` 混合。

## 完整感知链

训练将 GT 命令的 `--training_mode gt` 改为 `--training_mode full_chain`，并增加：

```bash
--velocity_model /path/to/velocity/best_model.pt \
--ostrack_checkpoint /path/to/OSTrack_ep0300.pth.tar
```

评估在 GT 评估命令上增加：

```bash
--full_chain_perception \
--full_chain_velocity_checkpoint /path/to/velocity/best_model.pt \
--ostrack_checkpoint /path/to/OSTrack_ep0300.pth.tar
```

默认自动定位本仓库已包含的 OSTrack 推理源码，无须另行编写感知桥。
启用 `--static_salvo --salvo_targets 100 --episodes 1` 可运行 SS 多目标连续清场；
该模式自动启用完整感知链，仍需提供速度和 OSTrack 权重。
加 `--record_video` 可录像。控制链的诊断说明见
[SS_CONTROL_DIAGNOSTICS.md](../integrated_shooting_chain/SS_CONTROL_DIAGNOSTICS.md)。

## 文件与依赖

- 本目录保留训练、评估、环境、双 GRU 网络、PPO 配置、目标跟随、清场和旧版感知桥。
- `vision_model1.py` 是旧版可选 CNN 感知桥需要的模型定义，来自原 `test4`，只复制模型文件。
- `../integrated_shooting_chain/` 包含本模块调用的感知桥、调度、诊断及录像运行代码。
- `../speed_prediction/` 包含速度 GRU/TAQ 模型、在线推理和必要预处理，不包含历史训练数据或实验输出。
- `../test_lk_ostrack/` 复用分支已有的视觉模块及测距传感器兼容代码。
- USD 组合资产复用同源的本地相机版本，消除远程相机引用，其余机器人资产随目录提供。

本次仅做代码整理和基本语法/导入路径检查，未加载模型权重或启动训练、评估和仿真。

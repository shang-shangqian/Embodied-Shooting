# 速度估计模块

根据目标位置历史、机器人运动和跟踪质量，估计 yaw-only base 坐标系下的目标相对平移速度。
支持默认的 13 维普通 GRU 和 24 维连续特征的 TAQ-GRU；两种网络权重不能互换。
环境和默认权重位置见[总 README](../../../README.md#权重与数据)。

## 文件说明

| 文件或目录 | 作用 |
| --- | --- |
| `__init__.py` | 速度模块包定义。 |
| `_compat.py` | 将兼容入口转发到实际实现，保持模块对象一致。 |
| `online_taq_velocity.py` | TAQ 在线推理的兼容入口，实际实现位于 `taq_gru/`。 |
| `time_aware_velocity_data.py` | 共享特征与数据格式的兼容入口，实际实现位于 `common/`。 |
| `time_aware_velocity_model.py` | TAQ 网络的兼容入口，实际实现位于 `taq_gru/`。 |
| [common/](common/README.md) | 数据格式、因果特征、速度基线和归一化工具。 |
| [legacy_gru/](legacy_gru/README.md) | 普通 GRU、数据窗口预处理和在线推理。 |
| [taq_gru/](taq_gru/README.md) | 时间与质量感知 GRU，以及按目标维护状态的在线估计器。 |

## 训练

本分支未包含 `train_gru_velocity.py`、`train_taq_velocity.py` 或独立的速度离线评估脚本，
需要使用已有训练权重。决策模块的 `--training_mode full_chain` 训练的是 PPO 决策网络，
速度网络保持冻结，不会训练速度 GRU。

## 测试

在已安装本仓库且具备 PyTorch 的环境中，先检查默认 GRU 权重能否加载：

```bash
python -c "from integrated_shooting_chain.model_loaders import load_velocity_estimator; print(load_velocity_estimator('checkpoints/velocity/best_model.pt'))"
```

该命令只检查模型和元数据，不代表精度评估。在线效果通过完整感知评估检查：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test7_2_weight/eval_with_track_gt.py \
  --headless --num_envs 1 --full_chain_perception \
  --checkpoint /path/to/run_or_checkpoint \
  --locomotion_policy /path/to/policy.pt
```

示例使用单环境降低显存占用；默认完整感知场景为 SD。
切换 TAQ 时增加 `--velocity_backend taq_gru`，并用
`--full_chain_velocity_checkpoint /path/to/taq_model.pt` 指定对应权重。
这里统计的是整条控制链的任务指标；独立速度误差评估需要完整项目中的评估脚本和标注数据。

# 炮塔决策网络

使用非对称 Actor/Critic 双 GRU 和 PPO 训练瞄准、提前量与开火决策。
支持 GT 输入和 LK/OSTrack + 测距 + 冻结速度模型的完整感知输入。

## 文件说明

| 文件或目录 | 作用 |
| --- | --- |
| `__init__.py` | 决策模块包定义。 |
| `train_turret_mixed_gt.py` | 混合场景 PPO 训练入口，支持 GT/完整感知、恢复训练及多 GPU。 |
| `eval_with_track_gt.py` | 加载决策权重，运行各类场景评估、轨迹显示、SS 清场和录像。 |
| `turret_aim_env_gt.py` | 机器人与炮塔仿真环境、观测、动作、奖励、弹道及命中判定。 |
| `asym_central_value_gru_network_gt.py` | 注册 RL-Games 使用的非对称 Actor/Critic 双 GRU 网络。 |
| `ppo_training_config.py` | PPO 参数、GT/完整感知默认配置和恢复训练时的学习率处理。 |
| `target_follow_motion_gt.py` | 为动态机器人场景生成带随机扰动的目标跟随行走指令。 |
| `static_salvo_eval.py` | 管理 SS 多目标池、命中移除、清场计数和结束条件。 |
| `rl_perception_bridge_gt.py` | 兼容旧 CNN 方向网络与普通 GRU 的感知桥；完整感知链使用独立的集成模块。 |
| `vision_model1.py` | 旧感知桥使用的图像编码器和方向网络定义。 |
| `lite3_with_turret.usd`、`turret_with_edite.usd`、`turret.urdf` | 机器人/炮塔组合场景及炮塔模型描述。 |
| [lite3/](lite3/README.md)、[turret/](turret/README.md) | 机器人和炮塔的本地 USD 资产。 |

## 环境与权重

先按[总 README](../../README.md#安装)配置仿真环境并安装 `.[decision,ostrack,dev]`。
行走策略通过必需参数 `--locomotion_policy` 提供；评估还需要决策 `.pth` 或训练目录。
完整感知的默认速度和 OSTrack 权重位置见[权重表](../../README.md#权重与数据)。
关闭 OSTrack 可增加 `--no-fine_tracking`；使用 TAQ 增加 `--velocity_backend taq_gru` 并提供匹配权重。

## 训练

以下命令均从仓库根目录执行。GT 从头训练：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test7_2_weight/train_turret_mixed_gt.py \
  --headless --locomotion_policy /path/to/policy.pt
```

训练输出位于 `runs/test7_2_weight/`。恢复训练增加 `--checkpoint /path/to/run_or_checkpoint`，
目录默认选择 best 权重。完整感知微调：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test7_2_weight/train_turret_mixed_gt.py \
  --headless --training_mode full_chain \
  --checkpoint /path/to/run_or_checkpoint \
  --locomotion_policy /path/to/policy.pt
```

GT 默认 64 个环境；完整感知默认 16 个环境、SS/SD 混合，速度网络保持冻结。
显存不足时用 `--num_envs` 减少环境数。PPO 默认参数不必重复写入命令。

## 测试

GT 混合场景评估：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test7_2_weight/eval_with_track_gt.py \
  --headless --checkpoint /path/to/run_or_checkpoint \
  --locomotion_policy /path/to/policy.pt
```

默认 16 个环境、10 个 episode，输出命中、开火及分场景统计。
指定场景时增加 `--case ss`、`sd`、`ds` 或 `dd`：第一个字母表示机器人静/动，第二个表示目标静/动。

完整感知评估增加 `--full_chain_perception`，当前仅支持 SS/SD，未指定场景时使用 SD。
清场、录像和完整感知命令见[集成模块](../integrated_shooting_chain/README.md#训练与测试)。

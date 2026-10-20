# 炮塔决策网络

| 文件 | 作用 |
| --- | --- |
| `train_turret_mixed_gt.py` | GT/完整感知模式的 PPO 训练入口。 |
| `eval_with_track_gt.py` | 决策权重评估、场景统计及原有扩展接口。 |
| `turret_aim_env_gt.py` | 机器人/炮塔环境、观测、奖励、弹道及命中判定。 |
| `asym_central_value_gru_network_gt.py` | 非对称 Actor/Critic 双 GRU。 |
| `ppo_training_config.py` | PPO 参数和恢复训练配置。 |
| `target_follow_motion_gt.py` | 动态机器人的目标跟随运动。 |
| `static_salvo_eval.py` | SS 目标池与清场状态管理。 |
| `rl_perception_bridge_gt.py` | 原有 CNN/速度感知桥接口，CNN 分支依赖此次未发布的 `test4`。 |
| `lite3/`、`turret/`、USD/URDF | 原始本地模型资产。 |

从仓库根目录，在已配置好的 Isaac Lab 环境执行 GT 训练和评估：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p source/standalone/test7_2_weight/train_turret_mixed_gt.py \
  --headless --locomotion_policy /path/to/policy.pt
"$ISAACLAB_PATH/isaaclab.sh" -p source/standalone/test7_2_weight/eval_with_track_gt.py \
  --headless --checkpoint /path/to/run_or_checkpoint --locomotion_policy /path/to/policy.pt
```

GT 为默认模式，评估默认 mixed；可增加 `--case ss`、`sd`、`ds`、`dd` 选择场景。
训练输出为 `runs/test7_2_weight/`；恢复训练增加 `--checkpoint /path/to/run_or_checkpoint`。

`--training_mode full_chain`、`--full_chain_perception`、`--static_salvo` 及全流程录像需要
此次未发布的 `integrated_shooting_chain`。保留原源码中的调用，不复制该目录或裁剪这些功能代码。
本次不加入本地该目录没有的 `vision_model1.py` 或 `__init__.py`。

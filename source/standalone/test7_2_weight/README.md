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

`--training_mode full_chain`、`--full_chain_perception`、`--static_salvo` 及全流程录像使用
已发布的相邻目录 [integrated_shooting_chain](../integrated_shooting_chain/README.md)。
权重、采集数据、实验布局文件和录像需自行准备或生成。
本次不加入本地该目录没有的 `vision_model1.py` 或 `__init__.py`。

## SS 连续清场

`--case ss --static_salvo --salvo_targets 100` 使用 100 个静止目标，预热后只提供一帧全部目标的 GT 框初始化视觉跟踪。
每次切换在本轮尚未尝试的可用目标中，按最新视觉估计距离动态选择最近目标；失败目标进入下一轮重试池。
本轮其他可用目标处理完后才重试，不固定首次排序，也不会在暂缓期结束后立即选回本轮失败目标。

默认 `--salvo_switch_policy both --salvo_miss_limit 5 --salvo_stall_timeout_s 3`，
连续五枪未命中或单次尝试满 3 个仿真秒未命中即切出，命中优先。保持至少 1 秒暂缓、最多 3 次尝试。
D 组默认清空切换后的 Actor GRU，并等待感知与速度就绪；等待期间保持控制目标角并禁火。
报告的 `selection=dynamic_nearest_per_round_deferred_retries`、`selection_round`、
`round_pending_target_ids` 与 `next_round_target_ids` 用于核对轮次及重试顺序。

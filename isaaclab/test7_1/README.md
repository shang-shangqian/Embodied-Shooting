## 代码文件说明

- `turret_aim_env.py`：核心强化学习环境。定义 `TurretAimEnvCfg` 和 `TurretAimEnv`，负责加载机器人与炮塔 USD、创建相机和 raycast 传感器、采样远距离目标、更新目标运动、计算视觉监督标签、生成 actor/critic 观测、处理炮塔控制和开火动作、计算弹道命中与奖励、统计评估指标。
- `train_turret_mixed.py`：混合场景训练入口。启动 Isaac Sim，创建 `TurretAimEnv`，安装静止/运动组合场景逻辑，接入感知桥接，可加载已有 checkpoint 继续训练，并保存 rl-games 训练结果。
- `eval_with_track.py`：带轨迹可视化的评估入口。加载训练好的策略，支持视觉模型、速度模型和 raycast 组成的感知闭环评估，并输出命中率、开火次数、准确率、SS/SD 响应时间等指标。
- `rl_perception_bridge.py`：感知到 RL 观测的桥接模块。负责加载视觉方向模型和速度 GRU 模型，把相机角度、raycast 距离、目标速度序列等转换到机器人 yaw-only base 坐标系，并写入环境中的 RL 预测量。
- `target_follow_motion.py`：目标跟随运动控制补丁。给环境安装移动底盘跟随目标的命令生成逻辑；在混合场景中会根据当前 case 决定是否让底盘运动。
- `asym_central_value_gru_network.py`：rl-games 自定义网络之一。使用非对称 actor/critic 结构，actor 使用策略观测，critic 使用特权观测，并用 GRU 处理时序信息。


## 资产文件说明

- `lite3_with_turret.usd`：组合后的机器人 + 炮塔主资产，环境默认从这里加载。
- `lite3/`：Lite3 机器人本体相关 USD 资产和配置。
- `turret/`：炮塔相关 USD 资产和配置。
- `lite3/configuration/`、`turret/configuration/`：分别存放机器人和炮塔的 base、physics、sensor 等 USD 配置文件。


### 训练指令示例
```bash
./isaaclab.sh -p source/standalone/test7_1/train_turret_mixed.py \
  --locomotion_policy logs/rsl_rl/deeprobotics_lite3_turret_flat/2026-03-10_20-28-54/exported/policy.pt \
  --vision_model source/standalone/test4/vision_model/vision_runs/vision_runs_cnn_only/04-27-17-54/vision_best.pth \
  --velocity_model source/standalone/test11/speed_prediction/gru_runs/04-28-13-59/best_model.pt \
  --num_envs 4 \
  --max_epochs 25000 \
  --checkpoint runs/turret_aim_mixed_04-29-00-59-06 \
  --which best
  --headless
```

### 测试指令示例
```bash
./isaaclab.sh -p source/standalone/test7_1/eval_with_track.py \
  --checkpoint runs/turret_aim_mixed_04-29-00-59-06/nn/turret_aim.pth \
  --which best \
  --locomotion_policy logs/rsl_rl/deeprobotics_lite3_turret_flat/2026-03-10_20-28-54/exported/policy.pt \
  --vision_model source/standalone/test4/vision_model/vision_runs/vision_runs_cnn_only/04-27-17-54/vision_best.pth \
  --velocity_model source/standalone/test11/speed_prediction/gru_runs/04-28-13-59/best_model.pt \
  --num_envs 1 \
  --episodes 10 \
  --case ss \
  --headless
```



本文件夹是远距离炮塔瞄准任务的 GT 版本，和 `test7_1` 类似，但使用真值信息或 GT fallback。

## 代码文件说明

- `turret_aim_env_gt.py`：核心强化学习环境。定义 `TurretAimEnvCfg` 和 `TurretAimEnv`，负责加载机器人与炮塔资产、创建相机和 raycast 传感器、采样远距离目标、计算 GT 位置/速度/距离/角度标签、生成 actor/critic 观测、处理炮塔控制与开火动作、计算弹道命中和奖励。
- `train_turret_mixed_gt.py`：GT 版本的混合场景训练入口。启动 Isaac Sim，创建 `TurretAimEnv`，安装静止/运动组合场景逻辑、目标跟随逻辑和感知桥接，可加载 checkpoint 继续训练。
- `eval_with_track_gt.py`：GT 版本的策略评估入口。加载训练好的策略进行评估，支持轨迹绘制、开火/命中统计、命中率、超时率、平均开火次数等指标，也可以接入视觉/速度模型桥接做对比。
- `rl_perception_bridge_gt.py`：GT 版本的感知桥接模块。负责加载视觉方向模型和速度 GRU 模型，将相机角度、距离和速度序列转换成 RL 需要的目标状态；默认允许 GT distance fallback 和速度缓存未就绪时的 GT velocity fallback，方便联调。
- `target_follow_motion_gt.py`：移动底盘跟随目标的控制补丁。给环境安装目标跟随命令生成逻辑，使移动场景下底盘可以按目标方向和速度范围运动。
- `asym_central_value_gru_network_gt.py`：GT 版本训练使用的 rl-games 自定义网络。采用非对称 actor/critic 结构，actor 使用策略观测，critic 使用特权观测，并用 GRU 处理时序信息。



### 测试指令示例
```bash
./isaaclab.sh -p source/standalone/test7_gt/eval_with_track_gt.py --headless --checkpoint runs/turret_aim_mixed_06-12-00-38-31/nn --which last --locomotion_policy logs/rsl_rl/deeprobotics_lite3_turret_flat/2026-03-10_20-28-54/exported/policy.pt --num_envs 32 --episodes 500 --case ss
```

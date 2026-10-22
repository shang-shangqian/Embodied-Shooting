# MuJoCo 全链路射击测试

入口：`tools/play_mujoco_shooting.py`。本次已接入实际模型推理，并完成首轮运行。
epoch 114000 基线没有触发开火；五发微调模型站立时发射两次、行走时发射一次，
均未命中。选配相机旋转补偿后，本段站立和行走测试均保持跟踪。接口链路已
运行到弹道评分，但命中性能尚未通过迁移验证。

[本轮视频与完整结果](../../../../outputs/mujoco_shooting_validation_20261022/README.md)。

## 已接通的链路

```text
MuJoCo tracking_camera RGB（1024×576）
  → 首帧红色球检测框
  → 原 MultiTargetVisionTracker：LK + OSTrack
  → 原参数阵面激光：120734 条射线
  → 原冻结速度 GRU
  → 原射击 Actor：28 维观测、归一化、GRU 状态、tanh(mu)
  → yaw/pitch 位置增量、开火确认与冷却
  → 原环境的同步预测弹道命中判定
```

初始化框只在第一帧从 RGB 颜色连通域获得。后续没有逐帧颜色检测、真值框或
目标位置回填。这个检测器仅适用于测试场景中的红球，不代表通用目标检测能力。
OSTrack 原始构造器写死了 CUDA；本目录的设备适配器只替换设备分配，保留
原模型权重和原 `initialize/track` 方法，使 CPU 也可以运行。

`shooting.py` 区分两种坐标系：Actor 的机器人速度和重力使用完整机身系；
目标位置、目标速度和 GRU 自身运动特征使用 yaw-only base 系。炮口点速度
通过 Jacobian 计算，包含机身与炮台运动，不使用静止炮口假设。

## 权重与训练配置

默认配置：[shooting_config.json](shooting_config.json)，沿用最近连射报告里的：

- 射击：`turret_aim_mixed_09-08-16-46-24` 的 epoch 114000 checkpoint。
- 速度：`gru_runs_v2/20261009_152810_116165/best_model.pt`。
- OSTrack：`OSTrack_ep0300.pth.tar`。
- 行走：此前已经使用的 `2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt`。

射击模型按原 RL-Games 网络严格加载全部状态，包括 59 维输入的运行统计量。
前 28 维是 Actor 观测，独立 Critic 的 31 维输入填零，其 GRU 不影响 Actor。
站立预热期间使用机器人状态和零目标占位更新 Actor 循环状态，但禁止炮台增量
和开火；首次获取目标保留预热状态，与原连射评测一致。预热结束后缺少预测时
跳过 Actor 推理、保持炮台指令并禁止开火，不推进 Actor 循环状态。

传入 `--decision-checkpoint` 时，会读取该训练目录的 `training_settings.json`，
匹配其速度权重、开火阈值、确认步数和每回合射击预算。旧 checkpoint 没有
该文件时使用原环境默认参数。五发微调模型的阈值为 0.5、确认 1 步，与默认
epoch 114000 的阈值 0.65、确认 2 步不同，不能混用。

## 运行

从仓库根目录运行；使用现有 Python 环境，不需要启动 Isaac Sim：

```bash
conda activate isaac_lab_232
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=4

# 站立，100 m 处的目标横向以 1 m/s 移动
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_shooting.py \
  --duration 12 --target-velocity 0 1 0 --video \
  --output outputs/my_mujoco_shooting/standing_moving

# 行走与射击：先建立感知，4 s 后在 1 s 内将前进指令升到 0.5 m/s
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_shooting.py \
  --duration 12 --walk-speed 0.5 --target-velocity 0 1 0 \
  --stabilize-camera --video --output outputs/my_mujoco_shooting/walking_moving
```

输出目录必须是新目录，避免覆盖旧结果。默认使用 CPU 推理与 MuJoCo 3.16 的
`discrete` 积分器。相机补偿是可选实验：只使用相机内参与自身姿态，将 RGB
旋转到首帧相机朝向，跟踪后再把像素和角度转回当前相机系；不读目标真值。
省略 `--stabilize-camera` 可以复现原始跟踪器输入的对照。

替换射击权重的方式：

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_shooting.py \
  --decision-checkpoint runs/test7_2_weight/full_chain_5shots_20261017_201341/nn/turret_aim.pth \
  --duration 12 --target-velocity 0 1 0 --video \
  --output outputs/my_mujoco_shooting/finetuned
```

## 输出与判读

- `shooting.mp4`：25 FPS，相机、跟踪框、真实炮口指向、机身视角及状态。
- `first_camera.png`、`last_camera.png`：原始相机图像。
- `report.json`：权重哈希、配置、实际发射、命中、感知有效帧数和跌倒统计。
- `steps.jsonl`：逐步动作、跟踪来源、测距、估计误差和开火门控状态。
- `trajectory.npz`：200 Hz 物理状态、50 Hz Actor 观测与动作。
- `last_lidar_scan.npz`：最后一次完整阵面扫描。

时长包含前 2 s 站立预热；视频从开始感知后录制，因此通常比仿真短 2 s。
绿色标记是跟踪目标，黄色标记是炮口前方 100 m 点的投影；黄色不是制导或
瞄准控制输入。`ballistic_would_hit` 只用于离线诊断，不能算作实际发射命中。

可以独立分析已有轨迹，不重新跑策略或推进物理：

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/analyze_mujoco_shooting.py \
  outputs/my_mujoco_shooting/walking_moving
```

它生成 `analysis.json`，检查目标是否仍在相机视野、跟踪误差、连续高开火意图
步数和开火阻塞原因。目标真值仅用于场景生成、弹道评分和这些误差统计。

## 物理与验证范围

炮台在内存中改用带原 USD 刚度、阻尼及力矩上限的位置驱动；腿关节仍由行走
策略控制。磁盘上的原始 USD、MJCF 机器人和权重均不修改。PhysX 的驱动隐式
使用步末位置与速度，MuJoCo 的 `discrete` 能隐式积分位置反馈；这是迁移的
数值选择，不代表两引擎的接触、约束及力限求解完全等价。
参见 [PhysX articulation drives](https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/docs/Articulations.html)
和 [MuJoCo numerical integration](https://mujoco.readthedocs.io/en/stable/computation/)。

`--integrator implicitfast` 作为对照，通过 `kd + h*kp` 表达步末位置预测的
线性驱动近似。旧行走验证默认的积分设置不因此改变。

命中判定沿用原环境：初速 882 m/s、重力 9.81 m/s²、半径 0.35 m、0.4 s 内
401 个采样点，按弹丸和目标同一时刻的轨迹段计算最近距离。默认没有空气阻力；
实现同时保留 Mach-Cd/RK2 分支。它是同步预测评分，不是 MuJoCo 高速弹丸接触。
单目标诊断中目标命中后保留，便于持续观察；尚未移植百目标选择、摧毁和重试流程。
两类测试不应直接比较清场时间或总命中率。

检查命令：

```bash
python source/standalone/integrated_shooting_chain/mujoco_sim/tests/test_mujoco_shooting.py
```

检查覆盖原网络的归一化与循环状态、Critic 输入隔离、炮口速度、坐标系、
增量/限位/力限、预热与开火节奏、相机旋转补偿、原函数的重力和阻力弹道数值对照，
以及原生隐式位置驱动的一维解析解。它们证明接口与公式，不证明策略迁移性能。

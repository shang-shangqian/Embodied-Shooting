# test7_2_weight：GT 炮塔决策网络训练与评估

本目录用于训练和评估安装在 Lite3 四足机器人上的炮塔决策网络。当前训练入口是
GT-only 模式：目标相对位置、速度和距离直接来自仿真真值，不启动相机，也不安装
感知桥。预训练 locomotion policy 负责腿部关节，RL 策略只负责炮塔瞄准和开火。

```text
GT 目标状态 + 机器人本体状态
              |
              v
非对称 Actor/Critic 双 GRU
              |
              v
yaw 增量 + pitch 增量 + fire intent
              |
              v
弹道命中、瞄准、开火效率和稳定性奖励
```

## 环境要求

- 从 IsaacLab 仓库根目录 `/home/jby/IsaacLab-2.3.2` 执行命令。
- 使用带有 `torch`、`rl_games` 和 IsaacLab 的 `isaac_lab_232` Conda 环境。
- 训练和评估需要 NVIDIA GPU 以及可用的 NVIDIA 驱动。
- 必须提供 Lite3 locomotion TorchScript 策略 `policy.pt`。
- 运行资产是本目录中的 `lite3_with_turret.usd`，其引用的资源位于 `lite3/` 和
  `turret/`。

当前机器建议先初始化 Isaac Sim 环境：

```bash
cd /home/jby/IsaacLab-2.3.2
source _isaac_sim/setup_conda_env.sh

ISAAC_PYTHON=/home/jby/miniconda3/envs/isaac_lab_232/bin/python
LOCOMOTION_POLICY=logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt
```

后续命令默认已经执行以上准备步骤，并继续使用同一个终端。

## 文件说明

| 文件或目录 | 作用 |
| --- | --- |
| `train_turret_mixed_gt.py` | GT-only 训练入口。创建环境、安装四场景混合逻辑和目标跟随逻辑，配置 RL-Games PPO、非对称 GRU、日志及 checkpoint。 |
| `eval_with_track_gt.py` | GT-only 评估入口。加载训练 checkpoint，按指定场景运行确定性策略，输出命中率、射击次数、弹道最小距离和响应时间等统计；GUI 模式可以绘制机器人、目标和弹道轨迹。 |
| `turret_aim_env_gt.py` | IsaacLab `DirectRLEnv`。负责资产加载、目标生成与运动、炮塔和腿部动作、观测、奖励、终止条件、弹道计算、命中判断以及评估统计。 |
| `asym_central_value_gru_network_gt.py` | 注册 RL-Games 网络 `asym_central_value_gru`。Actor 和 critic 分别编码目标与机器人状态，并使用独立 GRU；处理 rollout、序列训练和 episode done 后的隐藏状态重置。 |
| `target_follow_motion_gt.py` | 运行时扩展环境的 locomotion command。动态机器人场景中生成朝向目标的随机化速度和转向命令；静态机器人场景中强制命令为零。 |
| `rl_perception_bridge_gt.py` | 可选的视觉方向网络、距离来源和速度 GRU 桥接模块。当前 GT 训练与 GT 评估入口会强制关闭该模块，保留它是为了后续感知输入联调。 |
| `lite3_with_turret.usd` | 训练和评估实际加载的 Lite3 与炮塔组合资产。 |
| `lite3/` | Lite3 USD、网格和材质依赖。 |
| `turret_with_edite.usd`、`turret/` | 炮塔 USD 及其资源。 |
| `turret.urdf` | 炮塔的 URDF 源定义，包含 `yaw_joint`、`pitch_joint` 和 `gun_tip`。 |

## 策略接口

决策网络动作维度为 3：

1. yaw 关节位置增量，归一化动作范围 `[-1, 1]`。
2. pitch 关节位置增量，归一化动作范围 `[-1, 1]`。
3. 开火意图，环境将其从 `[-1, 1]` 映射到 `[0, 1]` 后与阈值比较。

Actor 使用 28 维观测，其中目标分支 9 维、机器人分支 19 维。Critic 使用 31 维
特权观测，其中目标分支 12 维、机器人分支 19 维。GT-only 模式下 actor 的目标
位置、速度和距离也由仿真真值提供；critic 额外获得目标加速度。

腿部动作不属于上述 3 维决策动作。它由外部 `policy.pt` 输出 12 维关节动作，环境
按该 locomotion policy 的训练范围将其限制在 `[-100, 100]`。

## 场景定义

每个 episode 从以下四种场景之一采样：

| 场景 | 机器人 | 目标 |
| --- | --- | --- |
| `SS` | 静止 | 静止 |
| `SD` | 静止 | 动态 |
| `DS` | 动态 | 静止 |
| `DD` | 动态 | 动态 |

当前训练默认权重为：

```text
SS / SD / DS / DD = 0.5 / 0.5 / 0.0 / 0.0
```

因此默认训练只覆盖静止机器人。需要训练移动底盘时，必须显式给 `DS` 或 `DD`
非零权重。训练入口会将四个非负权重归一化为概率。

## 从头训练

### 使用当前默认场景权重

```bash
"./isaaclab.sh -p source/standalone/test7_2_weight/train_turret_mixed_gt.py \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --max_epochs 15000 \
  --locomotion_policy "$LOCOMOTION_POLICY"
```

脚本内部会强制关闭相机和感知桥，因此无需额外传递相机参数。

### 训练全部四种场景

以下示例使用 `SS/SD/DS/DD = 0.2/0.2/0.3/0.3`：

```bash
"./isaaclab.sh -p source/standalone/test7_2_weight/train_turret_mixed_gt.py \
  --headless \
  --device cuda:0 \
  --num_envs 64 \
  --max_epochs 15000 \
  --p_ss 0.2 \
  --p_sd 0.2 \
  --p_ds 0.3 \
  --p_dd 0.3 \
  --locomotion_policy "$LOCOMOTION_POLICY"
```


## 输出目录

不传 `--experiment_name` 时，每次训练自动创建时间戳目录：

```text
runs/test7_2_weight/
  turret_aim_mixed_MM-DD-HH-MM-SS/
    nn/
      *.pth
    summaries/
    best_checkpoint_epoch.txt
```

模型 checkpoint 位于 `nn/`。默认从 epoch 11500 开始比较并保存最佳模型，且当前
`save_frequency` 为 0，不会定期保存 checkpoint。需要更早保存最佳模型时可以传入：

```text
--save_best_after 0
```

如需固定目录名，可传入：

```text
--experiment_name my_gt_run
```

对应输出目录为 `runs/test7_2_weight/my_gt_run/`。重复使用同一名称可能混合或覆盖
已有输出，常规训练建议保留自动时间戳。

## 断点续训

`--checkpoint` 可以接收具体 `.pth` 文件，也可以接收包含 `nn/` 的训练目录。
先将 `RUN_DIR` 改成实际目录：

```bash
RUN_DIR=runs/test7_2_weight/turret_aim_mixed_08-02-12-00-00

"$ISAAC_PYTHON" \
  source/standalone/test7_2_weight/train_turret_mixed_gt.py \
  --headless \
  --device cuda:0 \
  --num_envs 64 \
  --max_epochs 15000 \
  --checkpoint "$RUN_DIR" \
  --which latest \
  --locomotion_policy "$LOCOMOTION_POLICY"
```

checkpoint 选择规则：

- `--which best`：优先选择 `turret_aim.pth` 或最新的非 `last_` 模型。
- `--which last`：选择最新的 `last_*.pth`，不存在时直接报错。
- `--which latest`：选择目录中修改时间最新的 `.pth`。

不指定 `--experiment_name` 时，续训结果会写入新的时间戳目录，不会写回原目录。



## 模型评估

### Headless 批量评估

先把 `RUN_DIR` 指向要测试的训练目录：

```bash
RUN_DIR=runs/test7_2_weight/turret_aim_mixed_08-02-12-00-00

"$ISAAC_PYTHON" \
  source/standalone/test7_2_weight/eval_with_track_gt.py \
  --headless \
  --num_envs 16 \
  --checkpoint "$RUN_DIR" \
  --which best \
  --locomotion_policy "$LOCOMOTION_POLICY" \
  --case ss \
  --episodes 100
```

### GUI 轨迹评估

不传 `--headless` 时会打开 Isaac Sim，并为指定环境绘制机器人轨迹、目标轨迹和
最近若干次开火的弹道：

```bash
RUN_DIR=runs/test7_2_weight/turret_aim_mixed_08-02-12-00-00

"$ISAAC_PYTHON" \
  source/standalone/test7_2_weight/eval_with_track_gt.py \
  --num_envs 1 \
  --checkpoint "$RUN_DIR" \
  --which best \
  --locomotion_policy "$LOCOMOTION_POLICY" \
  --case sd \
  --episodes 10 \
  --draw_tracks \
  --track_env_id 0 \
  --pause_between_episodes \
  --pause_seconds 2.0
```

`--case` 支持 `ss`、`sd`、`ds`、`dd` 


## 资产文件说明

- `lite3_with_turret.usd`：组合后的机器人 + 炮塔主资产，环境默认从这里加载。
- `lite3/`：Lite3 机器人本体相关 USD 资产和配置。
- `turret/`：炮塔相关 USD 资产和配置。
- `lite3/configuration/`、`turret/configuration/`：分别存放机器人和炮塔的 base、physics、sensor 等 USD 配置文件。
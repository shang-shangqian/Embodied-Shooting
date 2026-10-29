# test7_2_weight：炮塔决策网络训练与评估

本目录用于训练和评估安装在 Lite3 四足机器人上的炮塔决策网络。训练入口
支持 `--training_mode gt` 和 `--training_mode full_chain`：前者使用仿真真值，后者用
RGB 跟踪、动态射线测距和 Stage1 TAQ-v2 在线生成 actor 观测。预训练 locomotion
policy 负责腿部关节，RL 策略只负责炮塔瞄准和开火。

2026-10-21 起，训练入口默认采用混合炮塔初始姿态：40% 零位、40% 一般随机姿态、
20% 限位附近，预热期间保持采样控制目标。默认联合训练 SS/SD，比例 40%/60%；
测评环境仍保持零位初始化。微调命令与参数见
[SS/SD 混合姿态微调](SS_SD_MIXED_POSE_FINETUNE_20261021.md)。

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
| `train_turret_mixed_gt.py` | GT/full-chain 双模式训练入口。创建环境、安装场景混合逻辑，配置 RL-Games PPO、非对称 GRU、日志及 checkpoint。 |
| `eval_with_track_gt.py` | GT-only 评估入口。加载训练 checkpoint，按指定场景运行确定性策略，输出命中率、射击次数、弹道最小距离和响应时间等统计；GUI 模式可以绘制机器人、目标和弹道轨迹。 |
| `turret_aim_env_gt.py` | IsaacLab `DirectRLEnv`。负责资产加载、目标生成与运动、炮塔和腿部动作、观测、奖励、终止条件、弹道计算、命中判断以及评估统计。 |
| `asym_central_value_gru_network_gt.py` | 注册 RL-Games 网络 `asym_central_value_gru`。Actor 和 critic 分别编码目标与机器人状态，并使用独立 GRU；处理 rollout、序列训练和 episode done 后的隐藏状态重置。 |
| `target_follow_motion_gt.py` | 运行时扩展环境的 locomotion command。动态机器人场景中生成朝向目标的随机化速度和转向命令；静态机器人场景中强制命令为零。 |
| `rl_perception_bridge_gt.py` | 旧版 CNN/GRU 感知桥，新 full-chain 训练不再使用它。新桥位于 `integrated_shooting_chain/isaac_training_bridge.py`。 |
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

2026-10-21 起，训练和测评默认恢复 `--yaw_error_mode wrapped_los`：Actor 索引 7 使用
`wrap_to_pi(目标 yaw − 实际枪口 yaw)`，表示圆周上的最短转向，与历史权重输入一致。
可以继续使用原 checkpoint 测试。观测仍为 28 维，pitch、Critic 和奖励保持原定义。
可达关节角方案保留为显式选项 `--yaw_error_mode joint_reachable`，需要适配后的权重。
新 checkpoint 和测评报告记录观测协议，协议不匹配会打印提示。两种模式的说明见
[yaw 观测修改说明](YAW_REACHABLE_OBSERVATION_20261020.md)。

腿部动作不属于上述 3 维决策动作。它由外部 `policy.pt` 输出 12 维关节动作，环境
按该 locomotion policy 的训练范围将其限制在 `[-100, 100]`。

## 场景定义

新增 SS 连续清场评估（`--static_salvo`）：一局同时生成 100 个静止目标，位置在
站立预热完成时的相机扇区内随机采样，之后保持世界坐标不变。机器人保持 SS 的
零速度指令站立策略。每轮从尚未尝试的可用目标中，按最新跟踪结果估计的相机距离动态选择最近目标，持续打击到
命中、跟踪丢失、对当前目标连续实际射击 5 枪均未命中，或选中后满 3 个仿真秒仍未命中，
任一条件成立后重新选择本轮尚未尝试的最近可用目标；未命中而切出的目标放入下一轮重试池，
不会因为距离近或暂缓期结束而在本轮再次被选中。已击毁或尝试次数耗尽的目标移出队列。
本轮其余可用目标都获得尝试后，才将重试池加入新一轮；每轮均动态按距离选择，不固定首次排序。
跟踪不可用的目标不阻塞换轮，恢复后可继续参与；本轮仍可见但在暂缓期的目标会等待冷却，
不会因此提前启动再下一轮。新出现的可用目标加入本轮待处理池。
中间不重置机器人、炮塔、射击冷却或整局计时。第五枪若命中仍按正常命中移除目标。
失败切出后至少暂缓 1 个仿真秒，且需进入下一轮才允许重试；每个目标最多尝试 3 次（包括首次），每次重选计数归零。
暂缓不删除视觉轨迹，也不计为击毁。等待感知、冷却和没有实际开火的步不计枪数，
默认采用 `--salvo_switch_policy both`，五枪与 3 秒条件并列，先达到哪个就触发哪个。
3 秒从每次选中目标开始，包含等待感知、瞄准和冷却，开火不重新计时；边界上命中优先。
持续测距无效或策略持续不开火时也会在 3 秒后暂缓该目标；整体 `--salvo_timeout_s` 仍生效。
用 `--salvo_miss_limit 5` 与 `--salvo_stall_timeout_s 3` 配置阈值。
`--salvo_switch_policy misses` 只用枪数；`--salvo_switch_policy timeout` 只用旧时限规则。
之前命令若显式写了 `--salvo_switch_policy misses`，需改成 `both` 才能同时启用。
报告 `selection=dynamic_nearest_per_round_deferred_retries` 标记此策略；每局记录
`selection_round`（从 1 开始）、`round_pending_target_ids`（本轮未尝试）与
`next_round_target_ids`（等待下一轮，按失败入池顺序记录；换轮后仍动态按距离选择）。
`pending_target_ids` 合并本轮和下一轮目标，不包含当前目标，不能视为固定的未来射击顺序。

在已有评估命令上增加以下参数即可运行一局 100 目标测试：

增加 `--salvo_return_home` 可测试“每次目标尝试前物理回零”的对照组，默认关闭。
回零期间暂停 Actor、禁止开火；实际 yaw/pitch 在零位附近稳定后，清空 Actor 一次并等待
速度就绪再射击。回零时间计入整局、从三秒尝试时限中单独排除；回零超时明确结束。
完整命令和指标见 [回零对照说明](../integrated_shooting_chain/SS_RETURN_HOME_20261021.md)。

```bash
./isaaclab.sh -p source/standalone/test7_2_weight/eval_with_track_gt.py \
  --headless --num_envs 1 \
  --checkpoint runs/test7_2_weight/YOUR_RUN_DIR --which best \
  --locomotion_policy logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt \
  --case ss --static_salvo --salvo_targets 100 --episodes 1 \
  --salvo_timeout_s 600 --salvo_report outputs/static_salvo/ss_100.json
```

替换 `YOUR_RUN_DIR` 为需要评估的策略目录。`--static_salvo` 自动启用完整感知链，
使用现有 LK/OSTrack → GRU → 决策网络；普通单目标评估的 GT/full-chain 开关不变。
预热后，用同一帧 RGB 和全部 100 个目标的 GT 框一次性初始化多目标跟踪。
此后每帧由 LK 更新全部剩余目标，OSTrack（启用时）精修当前选中的目标。
目标距离由跟踪框大小和已知的 0.70 m 球径估计，不用 GT 位置排序。
命中后仅删除对应跟踪 ID，其他目标的跟踪历史保留；切换时只重置当前目标的速度
估计历史，不再次提供 GT 框，也不重新进行机器人的 2 s 站立预热。
丢失目标不会通过 GT 重新初始化，仍计入未击毁数；如果所有未击毁目标都无法继续跟踪，
以 `tracking_lost` 提前结束；所有可跟踪目标耗尽尝试次数时，以 `attempts_exhausted`
结束。暂时处于重试等待期时继续跟踪并等待，不提前结束。首次 GT 初始化失败不会在后续
帧重试，以 `bootstrap_failed` 结束。只有全部击毁才记为清场成功。
GT 目标 ID 与实体位置的映射仅用于渲染、所选目标射线测量和弹道评分。
当前尚未模拟其他球对射线测距的遮挡、误击或飞行弹丸的到达时间。
该场景验证单次 GT 初始化后的持续多目标跟踪与清场，不包含自主检测。

测试并录制一局 100 球清场过程（使用当前已保存的策略）：

```bash
cd /home/jby/IsaacLab-2.3.2
conda activate isaac_lab_232
source _isaac_sim/setup_conda_env.sh

python source/standalone/test7_2_weight/eval_with_track_gt.py \
  --headless --num_envs 1 \
  --checkpoint runs/test7_2_weight/full_chain_ss_sd_gru_finetune_20261010_125055 \
  --yaw_error_mode wrapped_los \
  --which best \
  --locomotion_policy logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt \
  --case ss --static_salvo --salvo_targets 100 --episodes 1 \
  --salvo_timeout_s 600 \
  --record_video --video_episodes 1 --video_env_id 0 --video_hit_hold_s 0
```

每局生成一个 1280×720、50 FPS 的 MP4，包含预热到清场、超时或机器人失败的完整过程。
主画面是机身相机，叠加最近可用的跟踪框：青色为普通跟踪、黄色为当前目标、灰色为
丢失目标，命中时当前框变绿。右上角为覆盖机器人和全部原始小球位置的全场概览，
其中保留机器狗特写，命中时显示黄色评估弹道。字幕显示仿真秒数、枪数、累计命中数、
剩余数、命中率和感知就绪状态。叠加只写入输出视频，不写回视觉模块输入。
录像在每次物理步评分后、目标切换前采集，保留中途命中帧，整局结束才关闭文件。
录像增加渲染和编码耗时，不额外推进物理步；报告中的秒数仍为仿真时间。

默认输出到 `source/standalone/integrated_shooting_chain/video/<timestamp>/`：
`env0_episode_001.mp4` 为录像，`manifest.json` 为录像状态，`salvo_report.json` 为清场统计。
`salvo_report.steps.csv` 为逐步诊断，记录动作第 3 维、开火意愿、开火阈值、感知 ready、
动作执行前的冷却、目标 ID/尝试次数、框尺寸、跟踪置信度、预测距离/速度及 Actor 使用的
预测方向与当前枪口方向之差。`blocked_reason` 直接列出阻止开火的条件；多个条件用 `|`
分隔。瞄准误差是预测方向误差，不使用 GT 来驱动或修正开火。
诊断 CSV 版本 2 还记录 `decision_*`（决策前）和 `post_*`（物理步与 reward 完成后、
reset 前）的绝对预测/GT/枪口方向、测距有效性、yaw/pitch 动作、关节控制目标与实际角度、
速度、限位和缓存弹道最小距离。`actor_observation` 保存环境实际输出的完整 28 维观测副本，
并标注其生成时刻和目标 ID；GT 仅用于诊断，不输入跟踪器或 Actor。
完整字段、坐标约定和无效值规则见
[SS 控制链诊断说明](../integrated_shooting_chain/SS_CONTROL_DIAGNOSTICS.md)。
控制/循环状态对照使用 `--salvo_ablation A|B|C|D`，完整四组顺序运行命令、固定布局与
seed 规则也见该说明。`--static_salvo` 默认采用 D：感知未就绪时保持炮塔控制目标角，
切换到不同目标时清空一次 Actor GRU 状态，等待新目标的有效测距、速度估计及历史跨度
要求满足后再推理。等待期间不调用策略网络，保持循环状态和控制目标角，并禁止开火。
视觉、速度估计及机器人站立控制继续运行；同一目标中途感知失效时也暂停推理。
无需额外参数；用 `--no-salvo_wait_for_velocity` 可复现之前等待期间仍推理的 D 流程。
CSV 版本 3 起的 `velocity_ready`、`actor_inference_skipped` 和 `actor_wait_step_count`
记录速度就绪与暂停情况，JSON 记录 `wait_for_velocity` 和每局累计等待时间。
速度估计历史继续按原流程在目标切换时重建；首次获取目标保留 warmup 历史。
需要复现原逻辑时显式传入 `--salvo_ablation A`；普通单目标评估和训练入口不安装这些开关。
终端默认每 10 个实际秒输出一次进度，包含仿真时间、射击/命中/剩余数及开火阻塞原因；
暂缓、重试和次数耗尽会即时打印事件。
用 `--video_dir <新目录>` 可以指定保存位置；该目录必须尚不存在。
显式指定 `--salvo_report` 时，统计报告保存到指定路径。上述命令关闭末帧定格；
如需在全部清场后停留 1.5 秒，改为 `--video_hit_hold_s 1.5`，仅增加视频播放帧。
移除 `--record_video` 即可进行不录像的测试，报告默认写到 `outputs/static_salvo/`。
测试运行阶段按一次 Ctrl+C，会先关闭录像和 CSV，再保存 `interrupted` 的未完成统计，
随后退出 Isaac；应等待终端退出后再下载 MP4。初始化阶段中断和强制杀进程无法保证文件收尾。

初次验证可先把目标数改为 10、整局时限改为 30 个仿真秒：

```bash
# 替换上面完整命令中的同名参数，并追加以下重试/日志配置。
--salvo_targets 10 --salvo_timeout_s 30 \
--salvo_switch_policy both --salvo_miss_limit 5 --salvo_stall_timeout_s 3 \
--salvo_retry_delay_s 1 --salvo_max_attempts 3 \
--salvo_log_interval_s 10
```

`--salvo_diagnostics <新CSV路径>` 可单独指定诊断文件；默认使用 JSON 报告同名的
`.steps.csv`，文件已存在时拒绝覆盖。以上 retry/timeout 均为仿真秒，日志间隔为
实际秒。评估器步数上限会随整局时限扩展，避免 600 秒测试被默认 27,000 步提前截断。
LK 框中心按相似变换更新，宽高只按缩放更新，避免旋转导致外接矩形累计膨胀；SS 清场
还限制框宽高为各自首次视觉框尺寸的 0.5～2 倍，超出时拒绝观测。该累计尺度约束只在
静止 SS 清场中启用，普通运动目标跟踪不启用。

每局输出并在 JSON 的 `episodes` 中保存：

| 字段 | 含义 |
| --- | --- |
| `shots` / `destroyed` / `remaining` | 实际触发枪数、命中移除数、剩余数。 |
| `accuracy` | 命中移除数 / 实际触发枪数；无射击时为 `null`。 |
| `clearance_sim_s` | 预热后到全部命中的仿真秒数；未清场为 `null`。 |
| `engagement_sim_s` | 预热后的实际测试仿真秒数，包含等待感知就绪和目标切换。 |
| `warmup_sim_s` / `total_sim_s` | 初始预热秒数 / 包含预热的总仿真秒数。 |
| `mean_firing_hz` | `(shots - 1) / (末枪时刻 - 首枪时刻)`；不足两枪为 `null`。 |
| `shots_per_engagement_s` | 枪数 / 测试仿真秒数，包含首枪前等待。 |
| `cleared` / `reason` | 是否清场；原因包括 `cleared`、`timeout`、`failure`、`tracking_lost`、`attempts_exhausted`、`bootstrap_failed`、`interrupted`。 |
| `gt_bootstrap_frames` / `gt_bootstrap_targets` | 本局使用 GT 框初始化的帧数 / 目标数；正常为 `1` / `100`。 |
| `attempts` / `deferred_count` / `exhausted_target_ids` | 各目标尝试次数、暂缓次数、失败且耗尽尝试次数的未击毁目标。 |
| `remaining_target_ids` / `engagement_events` | 未击毁目标列表 / 选择、暂缓、重试耗尽和移除事件。 |

连续射击保留策略的开火意图阈值与确认步数，不强制盲射。用
`--salvo_fire_cooldown_steps N` 调整射击上限，实际最高频率为
`1 / ((N + 1) * step_dt)`，具体 `step_dt` 和上限保存在 JSON 中。默认沿用现有
5 步冷却。命中率需要由测试结果验证；这版不自动搜索满足某个命中率阈值的最大频率。
比较频率时，应使用同一策略和随机种子，并分别保存不同冷却设置的报告。
`--episodes` 在这个模式下表示清场测试局数，每局重新生成一个目标池；建议
`--num_envs 1`，便于明确统计一局的枪数与秒数。

每个 episode 从以下四种场景之一采样：

| 场景 | 机器人 | 目标 |
| --- | --- | --- |
| `SS` | 静止 | 静止 |
| `SD` | 静止 | 动态 |
| `DS` | 动态 | 静止 |
| `DD` | 动态 | 动态 |

当前训练默认权重为：

```text
SS / SD / DS / DD = 0.4 / 0.6 / 0.0 / 0.0
```

这是混合初始姿态训练及 full-chain 模式的默认值。显式 `--p_ss/--p_sd/--p_ds/--p_dd`
可覆盖场景权重。GT 模式选择 `--turret_init_mode zero` 且不显式指定权重时，恢复原来的
`0.3 / 0.3 / 0.2 / 0.2`；full-chain 仍只支持 SS/SD。

## 从头训练

### 使用当前默认场景权重

```bash
./isaaclab.sh -p source/standalone/test7_2_weight/train_turret_mixed_gt.py \
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
./isaaclab.sh -p source/standalone/test7_2_weight/train_turret_mixed_gt.py \
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

模型 checkpoint 位于 `nn/`。GT 模式默认从 epoch 11500 开始比较并保存最佳模型，
`save_frequency` 为 0，不定期保存。全链路模式默认 `save_best_after=0`，并每 50 个
绝对 epoch 额外保存检查点。可用 `--save_frequency` 修改周期，设为 0 则关闭定期保存。
需要更早保存最佳模型时可以传入：

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

全链路微调默认使用 16 环境、128 步 rollout、1024 minibatch、2 次 mini-epoch、
8 步 GRU 训练序列和固定 `3e-6` 学习率，熵系数仍为 0.003。可通过
`--horizon_length`、`--minibatch_size`、`--mini_epochs`、`--seq_length`、
`--learning_rate`、`--lr_schedule none|adaptive`、`--e_clip`、`--grad_norm` 和
`--kl_threshold` 覆盖设置。GT 模式保留原默认值。

全链路模式在恢复 checkpoint 后默认同时重设优化器学习率和调度器状态，避免恢复的
优化器学习率与配置不一致。`--no-reset_learning_rate` 则保留 checkpoint 学习率，
并将调度器同步到该值。其他权重、归一化、优化器动量和 epoch/frame 计数仍正常恢复。
每次运行保存 `training_settings.json`，记录本次解析后的参数与 PPO 配置。

`--max_epochs` 是绝对轮次上限：从 epoch 114000 恢复后追加 200 轮，应设置为
114200，而不是 200。全链路示例见 `../integrated_shooting_chain/README.md`。



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

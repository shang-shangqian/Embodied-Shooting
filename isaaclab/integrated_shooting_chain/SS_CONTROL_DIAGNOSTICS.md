# SS 清场控制链诊断

`--static_salvo` 自动生成逐步 CSV；版本 2 增加“预测方向 → 决策动作 → 关节执行 → 枪口方向”
的完整记录。无需开启 `--debug_diagnostics`，也不受前 50 步调试打印限制。
版本 3 增加速度链就绪状态和 Actor 等待期间是否跳过推理；当前版本 4 增加本次尝试的
实际射击数与连续未命中枪数。
默认与 JSON 报告同目录、同名，扩展名为 `.steps.csv`；用
`--salvo_diagnostics <新CSV路径>` 指定其他位置。已有文件不会被覆盖。

## 一行对应一个控制步

- `decision_*`：调用原 `_pre_physics_step` 前采样，此时动作尚未改变控制目标。
- `*_after_action_*`、`*_command_change_rad`：原 `_pre_physics_step` 返回后采样，物理步尚未运行。
- `post_*`：物理步、命中统计和 reward 完成后，环境 reset 和目标切换前采样。
- `decision_sim_s` / `post_sim_s`：各自的 episode 仿真时刻。
  `decision_frame_step` / `post_frame_step`：环境全局 `common_step_counter`。
- `target_id` 是本步决策前选中的目标；`post_target_id` 用于检查执行期间是否发生身份变化。
  `episode_index` 区分同一环境的不同局。命中帧仍对命中目标计算 GT 对照，不拿下一目标代替。
- `tracker_timestamp_s` 是跟踪结果的全局仿真时刻，`tracker_age_s` 是相对决策时刻的年龄。
  框和跟踪质量记录于决策前，与 `decision_gt_*` 对照；不要拿旧框对照物理步后的相机姿态。

CSV 写出不会推进物理、调用渲染、重新跟踪或测距。弹道字段读取 reward 已计算的缓存，
不会再积分一次弹道。诊断仍有少量张量计算、GPU→CPU 复制和磁盘写入开销。

## 字段与判读

以下带 `decision_` / `post_` 的字段，均可比较同一步执行前后的数值。

| 字段 | 含义 |
| --- | --- |
| `pred_valid` | ready 和 training_ready 均成立，预测位置/方向/速度有限，距离有限且为正 |
| `gt_valid` | 目标池已初始化，选中 ID 有效；命中当步仍有效 |
| `pred_yaw_base_rad` / `pred_pitch_base_rad` | bridge 写回的原始预测角，原点为机器人 root |
| `gt_yaw_base_rad` / `gt_pitch_base_rad` | 同一 root 原点的 GT 目标方向 |
| `pred_gt_*_error_rad` / `pred_gt_position_error_m` | 同原点预测与 GT 的误差，仅在二者有效时填写 |
| `muzzle_yaw_base_rad` / `muzzle_pitch_base_rad` | 由枪口连杆真实 forward 计算的方向角 |
| `gt_muzzle_*_base_rad` / `pred_muzzle_*_base_rad` | 将 GT / 预测目标都改用枪口点为原点得到的方向 |
| `gt_muzzle_*_error_rad` / `pred_muzzle_*_error_rad` | 上述目标方向减去真实枪口方向，消除 root/枪口原点差异 |
| `raycast_valid` / `raycast_distance_m` | 已有测距结果与有效性；无效距离为空 |
| `gt_center_x_px` / `gt_center_y_px` | 当前所选 GT 球心在机身相机内的投影 |
| `gt_projection_valid` | GT 有效、投影有限且位于相机前方；不等同于无遮挡或处于画面内 |
| `yaw_joint_pos_rad` / `pitch_joint_pos_rad` | 实际关节角；对应 `*_joint_vel_rps` 为实际角速度 |
| `yaw_command_rad` / `pitch_command_rad` | 写给关节的位置控制目标 |
| `*_command_error_rad` | 控制目标角减去实际关节角，不做周期折返，保留限位附近的差值 |
| `*_lower_limit_rad` / `*_upper_limit_rad` / `*_at_limit` | USD 关节限位，以及实际角是否距任一限位不超过 0.001 rad |
| `pred_pos_base_*` / `gt_pos_base_*` | root 原点、yaw-only base 坐标系中的位置 |
| `gt_pos_w_*` / `root_pos_w_*` / `muzzle_pos_w_*` / `camera_pos_w_*` | 世界坐标位置 |
| `root_quat_w_*` / `camera_quat_w_*` / `muzzle_forward_w_*` | root / ROS optical 相机姿态（wxyz），以及世界坐标下的枪口 forward |

其他字段：

| 字段 | 含义 |
| --- | --- |
| `action_yaw_raw` / `action_pitch_raw` | 环境收到的原始动作 |
| `action_yaw` / `action_pitch` / `action_fire` | 环境裁剪后保存的三维动作 |
| `*_delta_requested_rad` | 原控制逻辑计算的关节目标增量，包含预热门控 |
| `*_command_after_action_rad` / `*_command_change_rad` | 施加限位后的控制目标及本步实际目标增量 |
| `*_command_clamped` | 实际目标增量与请求增量相差超过 1e-6 rad |
| `actor_observation` | 本动作之前最后一次环境输出的完整 28 维 policy 观测，JSON 数组 |
| `actor_obs_distance` / `actor_obs_yaw_error` / `actor_obs_pitch_error` | 该数组索引 6/7/8，已经环境归一化、裁剪；不是米或弧度 |
| `actor_obs_*ready` / `actor_obs_target_id` / `actor_obs_frame_step` / `actor_obs_sim_s` | 上述观测生成时的状态；可检查观测与动作是否属于同一目标/时刻 |
| `actor_obs_available` | 是否实际捕获到观测；没有时留空，不用重算的观测冒充 |
| `bbox_center_*_px` / `tracker_yaw_cam_rad` / `tracker_pitch_cam_rad` / `tracker_distance_m` | 决策时所选轨迹的框中心、相机方向和框尺寸估计距离 |
| `tracker_gt_center_error_px` / `tracker_gt_*_error_rad` | 同时刻框中心/方向与 GT 投影的误差 |
| `ballistic_cache_valid` / `ballistic_min_dist_m` / `ballistic_hit` / `ballistic_closest_time_s` | 本步评分缓存中的弹道最小距离、命中判定和最近点飞行时间，不是实际飞行弹丸 |
| `hit_radius_m` | 判定命中使用的半径 |
| `velocity_ready` | 决策时速度桥是否达到有效测距、观测次数和历史跨度要求 |
| `actor_inference_skipped` | 本帧是否等待感知就绪而未调用策略网络；动作是保持/禁火占位 |
| `actor_wait_step_count` | 本局该环境累计跳过策略推理的控制步数 |
| `attempt_shots` / `consecutive_misses` | 本次尝试已评分的实际射击数 / 连续未命中数；在该步评分后、下一次调度前采样 |

`actor_observation` 的布局为：位置方向 0–2、速度 3–5、距离 6、yaw/pitch 误差 7–8、
机器人分支 9–27。这是环境 sanitize/clip 后、RL-Games 预处理及网络内部运行统计归一化前的
观测副本，不会修改返回给 Actor 的张量，也不记录或重置循环状态。
`actor_inference_skipped=True` 时仍保存环境输出观测以便诊断，但该观测没有送入策略网络。

角度单位均为弧度。Yaw-only base 中 yaw=`atan2(y,x)`，pitch=`atan2(-z,hypot(x,y))`，
所以抬枪方向的 pitch 为负。相机字段使用 ROS optical 坐标（x 右、y 下、z 前）。
yaw 差值折返到 [-π,π]；位置控制目标角差保留原值。
上述枪口方向误差是直视 LOS 误差，未加入重力补偿。判断能否命中应同时查看弹道字段，
不能要求 LOS 误差必须为零。

无效或非有限的派生值留空。原始预测字段在 ready=False 时可能仍是上次有效预测；
此时只把它当作原始状态，不解释为本目标的有效输入。
旧列 `yaw_error_rad` / `pitch_error_rad` 保留“post 原始预测角减枪口角”的原有语义，
并非 Actor 决策时的精确输入；定位控制问题应使用 `actor_obs_*` 与明确分阶段的新字段。

诊断使用的 GT 只流向诊断计算和 CSV，不输入视觉模块或 Actor，也不增加 GT 初始化次数。
诊断记录本身不改变动作、循环状态、开火奖励或阈值；控制保持和 Actor 状态清空由下述
射击流程配置决定。

JSON 报告记录 `diagnostics_schema_version=4`、采样协议，以及决策模型、速度模型、
OSTrack 权重的实际路径和随机种子配置。`control_diagnostics_config` 记录枪口连杆/forward 轴/
偏移、yaw/pitch 最大动作增量及误差观测缩放/裁剪参数，方便核对控制约定。
旧报告没有这些字段，不能依赖当前默认模型来复现。
需要复现旧问题时保留 100 目标、明确指定原决策与速度权重，先将时限缩为 30 个仿真秒。

## A/B/C/D 对照实验

常规 `--static_salvo` 默认以 `--salvo_switch_policy both --salvo_miss_limit 5
--salvo_stall_timeout_s 3` 调度。
只累计当前目标实际触发且已评分的未命中射击；连续五枪均未命中后，下一次感知调度
选择其他可尝试目标；同时保留单次尝试 3 秒无命中的退出条件，两者先触发者生效。
3 秒从每次选中开始，包含感知等待、瞄准、冷却；射击不重置时钟，边界上命中优先。
等待、冷却、低开火意图不消耗枪数，但仍消耗尝试时间，因此一直不开火时也能退出。
跟踪丢失和整局时限仍生效。暂缓 1 秒和最多 3 次尝试保持，每次重选计数及计时归零。
目标首次按视觉估计距离从近到远排队，随后按队列顺序选择。未命中而切出的目标追加到队尾，
即使 1 秒暂缓期结束且距离更近，也不会越过队列前方仍可尝试的目标。
击毁或次数耗尽的目标移出队列，跟踪不可用和冷却中的目标暂时跳过；新可用 ID 追加队尾。
报告 `selection=initial_nearest_fifo_requeue` 标记此策略，单局 `pending_target_ids` 保存待处理顺序（不含当前目标）。
JSON 的 `switch_policy` / `miss_limit` 标注规则；事件 reason 为 `consecutive_misses`，
记录触发时的 `attempt_shots` / `consecutive_misses`。旧 `--salvo_stall_timeout_s 3` 参数
在 misses 模式下不参与判定。`both` 同时启用两项，`timeout` 只用旧时限，`misses` 只用枪数。
旧命令显式选择了 misses 时需改为 both，不能只添加 stall_timeout 参数。

`--static_salvo` 未指定 `--salvo_ablation` 时默认采用 D 流程；普通单目标评估保持原逻辑。
D 默认同时启用 `--salvo_wait_for_velocity`：目标切换后清空 Actor 状态，等待新目标的
预测、桥接与速度 ready 均成立后再推理。等待期间不调用策略网络，返回 `[0,0,-1]`
保持控制目标角并禁止开火；Actor 和 Critic 隐状态都不演化。视觉、Raycast、速度历史、
机器人站立控制和仿真继续运行。warmup 内保留原推理行为；warmup 后即使尚无目标也暂停。
单个环境恢复 ready 后立即正常推理；多环境只对 ready 环境组成子批次，不阻塞其他环境。
同一目标中途感知失效时也暂停并保留现有隐状态，恢复后继续，不额外重复清空。
此门控使用速度模块的实际就绪条件，不额外添加固定 sleep，也不要求攒满 0.48 秒窗口。
用 `--no-salvo_wait_for_velocity` 可以复现此前 D 的“等待时仍推理”行为。
报告中的 `wait_for_velocity`、`actor_wait_protocol` 与每局 `actor_wait_steps`/`actor_wait_sim_s`
用于区分新旧流程；等待时环境保存的上一动作来自保持/禁火占位，不再写入未执行的策略提议。

`--salvo_ablation A|B|C|D` 可显式选择对照组：A 保留原控制和历史，B 在感知未就绪时保持炮塔控制目标角，
C 在有效目标 ID 切换后只重置决策 Actor GRU，D 同时启用两项。首次获取目标保留共同
预热历史；之后即使中间经过无目标等待，只要新 ID 与上次有效 ID 不同，就在第一次推理
前重置一次，同一目标后续帧不重复清空。Critic GRU、速度估计 GRU、冷却、机器人及炮塔位置
不因该开关额外重置；速度估计模块仍按原流程在目标身份变化时独立重建历史。

B/D 保留网络提出的原动作，但撤销未就绪步的控制目标角增量；已存在的位置控制继续
保持该目标角，不把关节转回零位。CSV 的 `hold_applied`、`actor_gru_reset`、
`actor_gru_reset_count` 和 `*_delta_before_hold_rad` 区分提出的动作与实际施加的增量。
每局报告记录实际保持步数、Actor 重置次数及事件。

`--salvo_seed 61` 同时指定环境、RL-Games 和 OpenCV seed。A 用 `--salvo_layout_out`
保存相对环境原点的完整目标布局；其余组用 `--salvo_layout_in` 加载同一布局，报告记录
`layout_sha256`。固定布局模式只支持一个环境、一局；文件不可覆盖，半径/目标数/校验值
不匹配时拒绝运行。布局只用于仿真场景生成，不给视觉增加 GT 初始化。

从已设置 Isaac 环境的终端顺序运行四组（默认不录像、每组 30 个仿真秒）：
该四组脚本显式采用旧 3 秒切换规则并关闭等待速度后推理，保持原 A/B/C/D 的两因素定义；
常规测试入口默认启用新 D 与五枪未命中切换流程。

```bash
cd /home/jby/IsaacLab-2.3.2
conda activate isaac_lab_232
source _isaac_sim/setup_conda_env.sh
python source/standalone/integrated_shooting_chain/tools/run_salvo_ablation.py \
  --gpu 1 --timeout_s 30 --seed 61
```

默认输出在 `source/standalone/integrated_shooting_chain/video/ablation_<timestamp>/`。
其中 `layout.json` 为四组共用布局，A/B/C/D 子目录各有报告、诊断 CSV 和 `run.log`；
`commands.json` 保存原始命令，`status.json` 保存进度，完成后生成 `comparison.json`
和 `REPORT.md`。每组单独进程、顺序运行；某组失败时停止，保留日志，不继续浪费 GPU。
加 `--record_video` 可让所有组都录像，勿只给某一组录像。用 `--dry_run` 只保存命令。
评估入口新增 `--device cuda:1`，统一设置渲染、物理、感知和决策模型设备。
此首轮只针对一个固定布局，不能据此得出总体统计结论。

中断后可在上述命令中加 `--resume_from <原实验目录>`，并用 `--output_dir <新目录>`
指定续跑位置。脚本复用已完成组及布局，只重跑未完成组；原目录和中断文件保留。
seed、时限、模型及录像模式必须与原实验相同，否则拒绝续跑。

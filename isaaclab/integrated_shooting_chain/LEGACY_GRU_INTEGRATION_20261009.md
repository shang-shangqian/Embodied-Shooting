# 全链路速度模块切换为普通 GRU

日期：2026-10-09。没有修改网络结构、重新训练速度/决策权重或变更仿真资产。

## 当前连接

```text
test_lk_ostrack：LK + 自适应 OSTrack
    → 跟踪方向驱动的 RayCaster 测距
    → test11/speed_prediction/legacy_gru：冻结的普通 13 维 GRU
    → test7_2_weight：原 28 维观测、3 维动作的强化学习决策网络
```

默认速度权重：
`source/standalone/test11/speed_prediction/gru_runs_v2/20261009_152810_116165/best_model.pt`。
SHA-256：`6a90484fe28728e624b3ff712222371d712fd2b97f888dc500d5ccebef86e689`。

`source/standalone/model_with_recoil/` 当前只有 USD/URDF 和建模计划。
实际 Python 衔接位于 `integrated_shooting_chain/`，训练/评估入口在 `test7_2_weight/`。
本次没有把 test7 默认资产改为 model_with_recoil 的资产；也不声称验证了新的后坐力动力学。

## 接口与启动规则

- 新增 `legacy_gru_adapter.py`，复用前一步已验证的 `OnlineLegacyGRU`，而非另写一套预处理。
- 输入为位置 3、位置增量 3（单位 m，不除 dt）、机器人线速度 3、角速度 3 和运动标志 1。
- 使用 checkpoint 的归一化和平滑参数，默认 0.48 秒滑窗、最多 32 个原始帧。
- 输出是 yaw-only base 坐标中的目标相对平移速度，直接写入原 actor 桥接缓冲区。
- 机器人预热结束后才初始化视觉跟踪器；首次直接有效位置只建立 GRU 历史。
- 后续测距缺失和跟踪丢失不额外重置 GRU；无效位置按原规则填零，但该帧不标记 actor 预测就绪。
- 每个目标/环境拥有独立历史，新 episode 清理对应状态；模型/协议异常仍硬重置以避免传播错误。
- 原 actor 端就绪条件保留：默认至少 3 次直接有效观测、首次有效位置后至少 0.10 秒。
- 不使用 GT 目标位置/速度补齐输入；仿真 bbox 仅用于每 episode 的初次检测代理。
- 普通 GRU 没有不确定度头，兼容字段 `reliability_gate` 仅表示本帧是否直接有效观测。

`factory.build_chain()`、Isaac 桥接配置和 test7 全链路训练/评估均默认选择 `legacy_gru`。
原 TAQ loader 保留，需明确选择 `taq_gru` 并提供匹配权重；不自动猜测模型类型。
GT-only 训练/评估模式、视觉模型和 RL 网络结构保持不变。

## 验证

### 回归测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  source/standalone/integrated_shooting_chain/tests \
  source/standalone/test11/speed_prediction/tests \
  source/standalone/test11/velocity_error_validation/tests -q
```

结果：**146 passed、1 skipped、36 subtests passed**。
覆盖 checkpoint 校验/预处理、13 维特征和 yaw-rate 对齐、预热跳过、首帧无速度输出、
目标/环境隔离、后续缺测、选择性重置、actor 写回及工厂默认后端。Python 编译检查、
`git diff --check` 和启动 shell 语法检查通过。

### CUDA0 真实全链路冒烟

RTX 3090 / CUDA0，1 个环境，SD，现有 LK/OSTrack 和 RayCaster。
速度使用上述冻结 GRU；决策权重来自
`runs/test7_2_weight_four/turret_aim_mixed_09-14-14-45-45/nn/turret_aim.pth`。
没有执行 PPO 更新，也没有启用 actor GT fallback。

机器人预热为 test7 默认的 2 秒。第 100 步视觉初始化成功、测距有效，GRU 首帧仅建立历史。
日志随后完成了 1 个 episode：

| 指标 | 实测 |
|---|---:|
| 完成 episode | 1 |
| 平均 episode 步数 | 114 |
| 发射次数 | 1 |
| 命中次数 | 1 |
| 超时 / 跌倒 / 无效状态 | 0 / 0 / 0 |
| 射击轨迹最小距离 | 0.1260 m |
| 预热结束后首次发射 / 首次命中响应 | 0.2600 / 0.2600 s |

这只是连接验证，不能把单次命中率推广为正式性能，也未采样 GRU 的逐帧速度误差。

实际诊断日志：`/tmp/full_chain_legacy_gru_debug_20261009.log`。
加入预热门控后的普通运行日志：`/tmp/full_chain_legacy_gru_smoke_20261009_v2.log`。
原普通运行重定向输出有缓冲，诊断改用无缓冲输出以确认 episode 的完整指标。

**关闭阶段限制**：episode 已结束并打印指标，但 Isaac Sim 在 `simulation_app.close()`
中的 SyntheticData/Replicator 渲染回调停滞。诊断调用栈指向仿真关闭流程，而非 GRU 推理；
已终止此次由助手启动的测试进程以释放资源，没有把正常退出也标为通过，也未修改该关闭问题。

## 使用

激活 `isaac_lab_232` 并加载 Isaac Sim 环境后，评估命令：

```bash
CUDA_VISIBLE_DEVICES=0 python source/standalone/test7_2_weight/eval_with_track_gt.py \
  --headless --num_envs 1 \
  --checkpoint runs/test7_2_weight_four/turret_aim_mixed_09-14-14-45-45 --which best \
  --locomotion_policy logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt \
  --full_chain_perception --velocity_backend legacy_gru --case sd --episodes 10 \
  --bridge_debug --no-draw_tracks --no-pause_between_episodes
```

速度权重可通过 `--full_chain_velocity_checkpoint` 或 `--gru_velocity_checkpoint` 覆盖。
旧 `--taq_velocity_checkpoint` 名字保留为同一参数的别名，使用旧 TAQ 权重时还须
指定 `--velocity_backend taq_gru`。

全链路 PPO 训练仍使用 `train_full_chain.sh`，默认普通 GRU；`--velocity_model` 覆盖速度权重。
当前速度训练数据主要为机器人站立 SS/SD，因此原 DS/DD 限制保留。

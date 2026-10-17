# OStrack 自适应触发改进设计

> 实施状态：同步自适应策略、固定策略回退、质量指标、失败退避、结果门控、CSV
> 诊断和离线评测汇总已经实现；第 11 节的异步扩展尚未实现。

## 1. 文档目标

本文定义 `test_lk_ostrack` 中 OStrack 从“固定周期执行”改为“按跟踪质量自适应执行”的设计方案，作为后续实现、测试和性能验收的依据。

本次改进只改变 OStrack 的调度方式，不改变以下既有原则：

- 所有目标继续由 LK 光流、RANSAC 和 Kalman 每帧维护。
- 仍然只允许当前主目标使用 OStrack 精跟踪。
- 主目标切换时必须为新目标重新建立 OStrack 状态。
- OStrack 是校正和恢复模块，不取代轻量主跟踪链路。
- 跟踪结果和目标优先级仍由 `MultiTargetVisionTracker` 统一输出。

第一阶段实现同步自适应调度；第二阶段再考虑异步推理。这样可以先单独验证触发逻辑，避免把调度错误和异步时序错误混在一起。

## 2. 当前实现与问题

当前 `_run_fine_tracker()` 在以下情况下执行 OStrack：

1. 主目标发生变化时，重新初始化 OStrack。
2. 主目标状态为 `predicted` 时立即执行。
3. 其他情况下，每隔 `fine_tracker_interval_frames` 帧执行一次。

默认间隔为 3 帧。这一设计简单、可预测，但有三个问题：

- LK 稳定时仍固定执行 OStrack，产生不必要的 GPU 计算。
- OStrack 推理集中出现在固定帧上，形成周期性延迟尖峰。
- 固定间隔不区分“稳定跟踪”“开始漂移”和“已经丢失”等不同风险状态。

现有正式评测中，跟踪器平均耗时约为 7～9 ms，但 P95 为 24～29 ms，超过 50 Hz 控制循环的 20 ms 单帧预算。因此改进目标不只是提高平均 FPS，更重要的是降低 OStrack 调用比例、20 ms 超时率和延迟抖动。

## 3. 设计目标与非目标

### 3.1 设计目标

- 稳定阶段减少至少 40% 的 OStrack 调用。
- 在 LK 开始恶化但尚未丢失时及时校正。
- `predicted`、主目标切换等高风险事件可以快速响应。
- 使用最小间隔、连续帧确认和失败退避避免频繁触发。
- 保留最大校正间隔，防止长时间完全依赖 LK 后发生慢性漂移。
- 记录完整触发原因和耗时，支持离线回放、阈值调优和消融实验。
- 固定周期策略继续保留，作为回退方案和 A/B 基线。

### 3.2 非目标

- 本阶段不替换 OStrack 主干网络。
- 本阶段不对所有目标运行 OStrack。
- 本阶段不修改目标优先级公式。
- 本阶段不解决目标检测、长期重识别或全画面搜索问题。
- 自适应触发不能替代独立的失效恢复模块；它只负责决定何时调用现有精跟踪器。

## 4. 总体方案

每帧先完成 LK、RANSAC 和 Kalman 更新，再评估当前主目标质量，由调度器决定是否调用 OStrack。

```text
RGB frame
   |
   v
LK forward/backward + RANSAC + Kalman
   |
   +--> 主目标选择
            |
            v
       质量指标计算
            |
            v
  硬触发 / 软触发 / 安全兜底
            |
       +----+----+
       |         |
      跳过      OStrack
                   |
                   v
              Kalman 校正
                   |
                   v
             重新提取 LK 角点
```

调度决策分为四种来源：

1. **状态硬触发**：已经出现明确失效信号，优先执行。
2. **质量软触发**：多个质量指标持续变差后执行。
3. **最大间隔兜底**：长期稳定时也进行低频校正。
4. **主目标生命周期事件**：切换、初始化和恢复时管理 OStrack 状态。

## 5. 所需质量指标

### 5.1 可直接复用的指标

`_TargetState` 已经包含：

- `confidence`
- `status`
- `missed_frames`
- `ransac_inliers`
- `median_fb_error_px`
- `points`
- `fine_score`
- `source`

### 5.2 建议新增的指标

为了识别“LK 仍有结果但框已经开始漂移”的情况，建议新增：

- `last_motion_confidence`：当前帧未经时间平滑的 LK 运动置信度。
- `last_inlier_ratio`：RANSAC 内点数除以有效 LK 点数。
- `last_point_retention`：前后向过滤后保留点数除以原始点数。
- `last_bbox_innovation`：LK 测量框相对 Kalman 预测框的归一化变化。
- `last_bbox_iou_with_prediction`：LK 测量框与当前帧 Kalman 先验框的 IoU。

其中框创新可以定义为：

```text
center_error = ||center_lk - center_kalman|| / max(diagonal_lk, 1)
scale_error  = max(|log(w_lk / w_kalman)|, |log(h_lk / h_kalman)|)
bbox_innovation = center_error + scale_error
```

归一化后阈值不会过度依赖图像分辨率和目标框尺寸。目标只有十几个像素时，应同时观察创新值与 IoU，避免单像素误差被过度放大。

## 6. 触发规则

### 6.1 主目标切换

当 `selected_target_id` 改变时：

1. 重置旧目标对应的 OStrack 状态。
2. 用新主目标当前帧 bbox 初始化 OStrack。
3. 初始化当帧不再额外调用 `track()`。
4. 重置调度器的坏帧计数、失败计数和运行帧计数。

初始化不是一次普通的 OStrack 推理，应单独统计 `fine_init_count` 和 `fine_init_latency_ms`。

### 6.2 硬触发

满足以下任意条件时产生硬触发请求：

- `state.status == "predicted"`。
- `state.missed_frames > 0`。
- LK/RANSAC 当前帧完全没有有效测量。
- 主目标 bbox 非法、越界严重或尺寸突变。
- 主目标刚从丢失状态恢复，需要精跟踪确认。

硬触发可以绕过软触发的连续坏帧要求，但仍需服从失败退避；否则 OStrack 连续失败时会每帧阻塞主循环。

### 6.3 软触发

建议先计算 0～1 的风险分数：

```text
confidence_risk = clip((warning_confidence - confidence)
                       / (warning_confidence - critical_confidence), 0, 1)

point_risk = 1 - clip(ransac_inliers / desired_inliers, 0, 1)

fb_risk = clip(median_fb_error_px / fb_max_error_px, 0, 1)

innovation_risk = clip(bbox_innovation / innovation_critical, 0, 1)

risk = 0.35 * confidence_risk
     + 0.20 * point_risk
     + 0.20 * fb_risk
     + 0.25 * innovation_risk
```

推荐规则：

- `risk >= adaptive_risk_threshold` 记为一个坏帧。
- 连续 `adaptive_bad_frames` 个坏帧后触发 OStrack。
- 风险恢复到 `adaptive_risk_clear_threshold` 以下才清除坏帧状态。
- 清除阈值必须低于触发阈值，形成迟滞区间。

不建议只用平滑后的 `state.confidence` 触发，因为它已经混合了点保留率、内点率和 FB 误差，可能掩盖单项指标的突然恶化。

### 6.4 最大间隔兜底

即使质量始终良好，如果距离上次成功校正超过 `adaptive_max_interval_frames`，仍执行一次 OStrack。它用于控制 LK 的慢性累计漂移。

最大间隔应按“上次成功应用 OStrack 结果”计算；失败退避按“上次尝试 OStrack”计算。两个时间必须分开，防止失败后因为一直超过最大间隔而每帧重试。

### 6.5 最小间隔

除主目标初始化外，两次 OStrack 推理之间至少间隔 `adaptive_min_interval_frames`。软触发和最大间隔必须遵守该限制。

硬触发原则上也遵守最小间隔。只有在确认上一帧 OStrack 成功、但当前 bbox 出现严重非法结果时，才允许配置为绕过最小间隔；默认不建议开启这一例外。

## 7. 推荐初始配置

以下参数作为第一轮实验起点，不应直接视为最终值：

```python
fine_tracker_policy: str = "fixed"  # fixed | adaptive

adaptive_min_interval_frames: int = 2
adaptive_max_interval_frames: int = 12
adaptive_bad_frames: int = 2

adaptive_warning_confidence: float = 0.75
adaptive_critical_confidence: float = 0.55
adaptive_risk_threshold: float = 0.55
adaptive_critical_risk_threshold: float = 0.85
adaptive_risk_clear_threshold: float = 0.35

adaptive_desired_inliers: int = 9
adaptive_innovation_critical: float = 0.75
adaptive_prediction_iou_threshold: float = 0.55

adaptive_failure_backoff_initial_frames: int = 1
adaptive_failure_backoff_max_frames: int = 8
```

选择 12 帧作为最大间隔时，在 50 Hz 下约为 240 ms。它比当前 3 帧一次明显降低调用率，同时仍保留定期校正。对于高速运动或更强相机抖动，可以降到 8 帧；对于稳定平台可以测试 15～20 帧。

开发阶段默认策略保持 `fixed`，便于兼容现有结果。自适应策略通过完整验收后，再考虑将默认值切换为 `adaptive`。

## 8. 决策优先级与伪代码

触发原因按以下优先级记录：

```text
target_switch
hard_tracking_failure
critical_quality
soft_quality_streak
max_interval
none
```

调度器伪代码：

```python
def decide_fine_action(state, scheduler, frame_index):
    if selected_target_changed:
        return INITIALIZE, "target_switch"

    if fine_tracker_not_initialized:
        return INITIALIZE, "missing_state"

    frames_since_attempt = frame_index - scheduler.last_attempt_frame
    frames_since_success = frame_index - scheduler.last_success_frame

    if frames_since_attempt < scheduler.current_backoff_frames:
        return SKIP, "failure_backoff"

    hard_failure = (
        state.status == "predicted"
        or state.missed_frames > 0
        or state.last_motion_confidence <= critical_confidence
    )

    risk = compute_tracking_risk(state)
    update_bad_frame_streak_with_hysteresis(risk)

    if frames_since_attempt < adaptive_min_interval_frames:
        return SKIP, "min_interval"

    if hard_failure:
        return TRACK, "hard_tracking_failure"

    if risk >= critical_risk_threshold:
        return TRACK, "critical_quality"

    if scheduler.bad_frame_streak >= adaptive_bad_frames:
        return TRACK, "soft_quality_streak"

    if frames_since_success >= adaptive_max_interval_frames:
        return TRACK, "max_interval"

    return SKIP, "stable"
```

OSTrack 成功后：

- 更新 `last_attempt_frame` 和 `last_success_frame`。
- 清零连续坏帧数和失败次数。
- 使用较低测量噪声校正 Kalman。
- 丢弃旧 LK 点并在校正后的 bbox 内重新检测角点。
- 记录校正前后 bbox 的中心位移、尺度变化和 IoU。

OSTrack 失败、抛出异常或分数不合格后：

- 只更新 `last_attempt_frame`。
- 增加连续失败次数。
- 退避帧数按 `1, 2, 4, 8` 增加，并限制最大值。
- 丢弃 OStrack 内部的可疑位置状态；退避结束后从可信的 LK/Kalman 框重新初始化。
- 保留 LK/Kalman 当前输出，不应用无效 bbox。
- 不把失败本身解释为 LK 已经丢失。

## 9. 防振荡与安全约束

### 9.1 连续坏帧确认

小目标通常只有 8～17 px，单帧 FB 误差、角点数量和 bbox 尺度容易波动。软触发要求连续两帧恶化，可以过滤孤立噪声。

### 9.2 风险迟滞

使用不同的触发阈值和清除阈值。例如风险达到 0.55 开始累计坏帧，回落到 0.35 以下才完全恢复稳定，避免风险在阈值附近反复开关。

### 9.3 OStrack 结果融合门控

不能因为 OStrack 被触发，就无条件相信它的结果。应用前至少检查：

- `result.ok` 为真。
- bbox 有限、面积为正并位于合理图像范围。
- score 存在时满足 `fine_tracker_min_score`。
- 尺度变化没有超过单帧物理上限。
- 与 Kalman/LK 先验差异过大时，进入“恢复候选”而不是立即覆盖。

对于明显大幅校正，可以要求下一次 OStrack 或下一帧 LK 再次确认。否则 OStrack 的一次错误跳转可能直接造成 ID 混淆。

### 9.4 主目标选择稳定性

自适应 OStrack 不应直接修改主目标切换规则。主目标仍由现有优先级和 `priority_switch_margin` 决定。

## 10. 同步实现方案

第一阶段保持 OStrack 在 `update()` 内同步执行，主要改动位置如下。

### 10.1 `MultiTargetConfig`

新增第 7 节中的策略和阈值，并在 `__post_init__()` 中校验：

- 策略只能为 `fixed` 或 `adaptive`。
- 最小间隔必须小于等于最大间隔。
- 风险清除阈值必须小于触发阈值。
- 所有比例和置信度必须在 `[0, 1]`。
- 退避帧数必须为正。

### 10.2 `_TargetState`

增加原始 LK 质量与创新指标。调度器自身状态可以先保存在 `MultiTargetVisionTracker` 中，因为当前只有一个共享 OStrack 实例，只维护当前主目标。

建议的调度状态：

```python
_fine_last_attempt_frame
_fine_last_success_frame
_fine_bad_frame_streak
_fine_failure_count
_fine_backoff_until_frame
_fine_last_trigger_reason
```

### 10.3 `_fit_motion()`

除现有返回值外，返回点保留率、内点率和原始运动置信度。调用处保存到 `_TargetState`。

### 10.4 `_run_fine_tracker()`

拆分为三个职责清晰的私有函数：

```python
_compute_fine_tracking_risk(state)
_decide_fine_tracker_action(state)
_apply_fine_tracker_result(state, result)
```

固定周期策略继续走原逻辑；自适应策略走新的决策函数。两种策略应共用结果校验和 Kalman 校正代码，避免行为分叉。

## 11. 异步扩展方案

同步自适应通过验收后，可以把 OStrack 放入独立 CUDA stream 或工作线程。异步实现必须解决结果过期问题，不能只把同步调用包进线程。

每个任务至少携带：

```text
frame_index
selected_target_id
target_generation
input_bbox
image_timestamp
```

结果应用条件：

- 当前主目标 ID 与任务 ID 相同。
- `target_generation` 没有因重置或重绑定发生变化。
- 结果帧龄不超过 `max_async_result_age_frames`。
- 结果与当前 Kalman 预测通过一致性门控。

异步队列最多保留一个待处理任务；有任务执行时只更新触发需求，不继续堆积图像。主目标切换时旧任务结果必须作废。

第一版异步融合可以只接受不超过 2 帧的结果。更旧结果需要先传播到当前时刻再校正，复杂度和错误风险明显上升，不建议作为初始实现。

## 12. 日志与可观测性

建议为每帧主目标增加以下字段：

- `fine_policy`
- `fine_action`
- `fine_trigger_reason`
- `fine_risk`
- `fine_bad_frame_streak`
- `frames_since_fine_attempt`
- `frames_since_fine_success`
- `fine_attempted`
- `fine_applied`
- `fine_score`
- `fine_latency_ms`
- `fine_failure_count`
- `fine_correction_center_px`
- `fine_correction_iou`
- `tracker_total_latency_ms`
- `deadline_20ms_missed`

汇总报告至少增加：

- OStrack 初始化次数、尝试次数、成功次数和应用次数。
- 各触发原因占比。
- OStrack 调用帧比例。
- 单次 OStrack 延迟分布。
- 总延迟 mean/P50/P95/P99。
- 20 ms 超时帧比例。
- 触发前后 1、3、5 帧的 IoU 和角度误差变化。

只有记录“为什么触发”和“触发是否改善结果”，才能判断阈值是否合理。

## 13. 测试计划

### 13.1 单元测试

使用 stub fine tracker 验证：

1. 稳定质量下，在最大间隔前不执行。
2. 达到最大间隔后执行一次。
3. 单个坏帧不触发软校正。
4. 连续坏帧达到阈值后触发。
5. `predicted` 状态产生硬触发。
6. 最小间隔能够阻止连续执行。
7. 风险迟滞不会在阈值附近振荡。
8. 主目标切换只初始化，不在同帧额外调用 `track()`。
9. OStrack 异常不会破坏 LK/Kalman 输出。
10. OStrack 连续失败遵守指数退避。
11. 低 score、非法框和异常尺度不会被应用。
12. 成功校正后重新提取 LK 角点。
13. `fixed` 策略调用周期与改动前一致。

### 13.2 序列级测试

至少覆盖：

- 稳定匀速目标。
- 快速横向运动。
- 深度方向尺度变化。
- 相机抖动和机器人运动。
- 低纹理目标。
- 两目标交叉和短时重叠。
- 短时遮挡后恢复。
- OStrack 返回错误候选框。
- 主目标频繁接近切换阈值。

### 13.3 消融实验

统一数据集上比较：

```text
LK only
固定间隔 1
固定间隔 3（当前基线）
固定间隔 5
仅硬触发
硬触发 + 最大间隔
完整自适应策略
```

## 14. 验收标准

相对固定间隔 3 帧基线，建议采用以下第一阶段标准：

| 指标 | 目标 |
| --- | --- |
| OStrack 调用次数 | 降低至少 40% |
| mean IoU | 下降不超过 0.005 |
| IoU ≥ 0.5 比例 | 下降不超过 0.5 个百分点 |
| 角度误差 P95 | 增加不超过 0.05° |
| 跟踪可用率 | 下降不超过 0.2 个百分点 |
| 身份混淆率 | 增加不超过 0.5 个百分点 |
| 20 ms 超时率 | 降低至少 50% |
| `fixed` 回退模式 | 与原行为一致 |

平均 FPS 只作为辅助指标。必须同时检查 P95/P99、20 ms 超时率和最差序列，避免通过减少调用得到更高平均速度，却把漂移风险推迟到少数序列中。

对于遮挡恢复数据集，还应单独报告：

- 从首次失效到成功校正的帧数。
- 恢复成功率。
- 恢复后的 ID 是否正确。
- 恢复后 5 帧内的平均 IoU。

## 15. 实施阶段

### 阶段 A：补齐指标与日志

- 不改变固定周期行为。
- 增加原始 LK 质量、bbox 创新、OSTrack 耗时和调用原因日志。
- 在现有正式数据集上重新得到可比较基线。

### 阶段 B：同步自适应触发

- 实现配置、风险计算、硬/软触发、迟滞和失败退避。
- 保留 `fixed` 策略。
- 完成单元测试和固定数据集 A/B 评测。

### 阶段 C：困难场景调参

- 增加遮挡、交叉、快速运动、低纹理和相机运动数据。
- 基于触发原因统计调整阈值。
- 确认减少调用后没有放大最差序列问题。

### 阶段 D：异步化

- 在同步自适应已经稳定后实现单任务异步队列。
- 加入帧龄、目标 generation 和一致性门控。
- 重新评估端到端延迟，而不是只统计网络推理时间。

## 16. 推荐结论

建议采用“固定策略保留 + 同步自适应先行 + 异步作为第二阶段”的路线。自适应调度的核心不是简单地把周期从 3 改成更大的数字，而是建立以下闭环：

```text
跟踪质量下降 -> 连续确认 -> OStrack 校正 -> 质量恢复
      ^                                  |
      +------ 失败退避与最大间隔兜底 -----+
```

第一版应优先保证可解释、可回放和可回退。完成日志与 A/B 基线后，再根据真实触发分布调整风险权重和阈值。这样既能降低 OStrack 的平均调用量，也能避免在最需要精跟踪时错过校正时机。

# LK + OSTrack 多目标视觉模块

该模块接受第一帧中的多个 `xyxy` 矩形框，为每个 ID 输出框、中心、
当前/预测方向、置信度和目标优先级。

```text
每框 9-16 个角点
  -> 所有 ID 合并为一次前向/后向 LK 光流
  -> 前后向误差过滤
  -> 每个 ID 独立 RANSAC 仿射运动估计
  -> 每个 ID 独立卡尔曼速度与提前量预测
  -> 跟踪质量、距离、速度和画面位置联合优先级
  -> 带滞回的主目标切换
  -> 仅主目标进入 OSTrack 精跟踪
```

OSTrack 不再为每个目标复制一份模型。非主目标使用轻量光流维持，主目标发生
切换时重新建立 OSTrack 模板。

将固定周期 OStrack 改为按跟踪质量自适应触发的设计、阈值、测试与验收方案，见
[`ADAPTIVE_OSTRACK_DESIGN.md`](ADAPTIVE_OSTRACK_DESIGN.md)。

## 代码文件说明

下面列出本目录自行维护的 Python 文件。`ostrack/OSTrack-main/` 是随项目放置的
第三方 OSTrack 源码，不在此逐文件展开。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 定义包的公共 API，导出多目标跟踪器和 OSTrack 适配器等常用类。 |
| `camera_geometry.py` | 相机内参、像素与 yaw/pitch 的转换、三维投影/反投影及边界框辅助函数。 |
| `multi_target_tracker.py` | 多目标跟踪核心：LK 光流、前后向误差过滤、RANSAC、卡尔曼预测和目标选择。 |
| `multi_target_io.py` | 读取初始目标框，写入多目标结果 CSV，并生成带跟踪框的叠加图。 |
| `ostrack_tracker.py` | 对官方 OSTrack 做适配，提供目录图片单目标跟踪和逐帧精跟踪接口。 |
| `batched_ostrack.py` | 多个 Isaac Lab 环境共享同一 OSTrack 网络并进行批量推理，分别保存各环境的模板和状态。 |
| `online_vision.py` | 将 OSTrack 接入 Isaac Lab 在线环境，负责逐环境初始化、串行/批量推理、丢失判定和搜索恢复。 |
| `pipeline.py` | 单目标结果后处理：由目标框计算方向和角速度，并写入 CSV 或可视化图片。 |
| `estimate_video.py` | 图片序列推理入口；支持单目标 OSTrack，以及多目标 LK + 可选 OSTrack 精跟踪。 |
| `multi_target_collect_env.py` | Isaac Sim 多目标采集环境，生成目标、控制运动、提供监督标签，并支持外部视觉方向驱动 RayCaster。 |
| `collect_multi_target_dataset.py` | Isaac Sim 数据采集入口，在线运行 LK/可选 OSTrack 获取方向，再用 RayCaster 获取距离并写成序列数据集。 |
| `isaacsim_dataset.py` | 数据集写入与校验工具，负责序列目录、manifest、元数据和 `ground_truth.csv`。 |
| `check_multi_target_dataset.py` | 已采集数据集的完整性检查入口。 |
| `evaluate_multi_target_dataset.py` | 在整个 Isaac Sim 数据集上评估多目标管线，统计 IoU、角度误差、视觉方向与记录测距融合后的三维坐标误差、丢失率、ID 错误和速度。 |
| `evaluate_direction_dataset.py` | 将数据集中的目标逐个交给单目标 OSTrack，评估方向误差和推理耗时。 |
| `tests/test_multi_target_tracker.py` | 测试多目标光流、遮挡、主目标切换、文件 I/O 和 OSTrack 网络复用。 |
| `tests/test_isaacsim_dataset.py` | 测试数据集写入、读取、完整性检查及未完成序列清理。 |
| `tests/test_evaluate_multi_target_dataset.py` | 测试 IoU 和角度误差等基础评估函数。 |

相关的非 Python 资源中，`lite3_with_turret.usd` 是采集脚本实际加载的机器人与炮台
场景；`lite3/`、`turret/`、`turret.urdf` 和其他 USD 文件是模型资源；
`ostrack/OSTrack.zip` 与 `ostrack/OSTrack-main/` 分别是 OSTrack 源码归档和展开目录。

以下命令均假设当前目录是 IsaacLab 仓库根目录。可先用 `--help` 查看某个入口的
完整参数，例如：

```bash
PYTHONPATH=source/standalone python3 -m test_lk_ostrack.estimate_video --help
PYTHONPATH=source/standalone python3 -m test_lk_ostrack.evaluate_multi_target_dataset --help
./isaaclab.sh -p source/standalone/test_lk_ostrack/collect_multi_target_dataset.py --help
```

## Python API

```python
from test_lk_ostrack import MultiTargetConfig, MultiTargetVisionTracker, TargetKinematics
from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.ostrack_tracker import OSTrackFrameTracker

intrinsics = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
fine_tracker = OSTrackFrameTracker(
    repo_path="/path/to/OSTrack",
    checkpoint="/path/to/OSTrack_ep0300.pth.tar",
    device="cuda:0",
)
tracker = MultiTargetVisionTracker(
    intrinsics,
    config=MultiTargetConfig(prediction_horizon_s=0.08),
    fine_tracker=fine_tracker,
)

result = tracker.initialize(
    first_rgb,
    {3: (100, 80, 150, 130), 7: (420, 90, 475, 145)},
    priority_biases={3: 0.2, 7: 0.0},
    target_kinematics={
        3: TargetKinematics(distance_m=8.5, speed_m_s=2.1, source="radar"),
        7: TargetKinematics(distance_m=12.0, speed_m_s=1.4, source="radar"),
    },
)
result = tracker.update(
    next_rgb,
    dt_s=1 / 60,
    target_kinematics={3: (8.4, 2.2), 7: (11.9, 1.5)},
)

primary_id = result.selected_target_id
for target in result.targets:
    print(target.target_id, target.predicted_yaw_rad, target.predicted_pitch_rad)
```

输入图像必须是 `RGB uint8 [H,W,3]`。调用 `set_priority_bias(id, value)` 可在运行时
加入敌我识别、装甲板类别、剩余血量或比赛策略产生的优先级；视觉模块本身不会猜测
这些语义。

## 主目标优先级

默认优先级权重为：置信度 `0.35`、画面中心度 `0.20`、框面积 `0.05`、
跟踪稳定性 `0.10`、距离 `0.20`、速度 `0.10`。各分量按权重绝对值之和
归一化，再叠加每个 ID 的 `priority_bias`。距离越近得分越高，速度越快得分
越高；将 `priority_speed_weight` 设为负数可改为优先较慢、较容易命中的目标。

`initialize()` 和 `update()` 的 `target_kinematics` 接受每个 ID 的外部距离与速度，
格式可以是 `TargetKinematics` 或 `(distance_m, speed_m_s)`。当前数据集评估默认从
`ground_truth.csv` 注入真值，并在结果中记录 `kinematics_source=ground_truth`；未来
雷达、深度相机、PnP 或状态估计模块可以直接替换该输入。

缺少某个 ID 的外部输入时，才回退到视觉估计：距离由已知目标宽高与像素框尺寸计算
`z ~= f * size / pixel_size`；速度使用卡尔曼框的短时三维变化计算。正式评估如需测试
这条回退链路，可加 `--kinematics_source vision`。

## 多目标视频

`initial_targets.json`：

```json
{
  "targets": [
    {"id": 3, "bbox": [100, 80, 150, 130], "priority": 0.2},
    {"id": 7, "bbox": [420, 90, 475, 145], "priority": 0.0}
  ]
}
```

也可使用文本，每行格式为 `id x1 y1 x2 y2 [priority]`。

```bash
PYTHONPATH=source/standalone python3 -m test_lk_ostrack.estimate_video \
  --rgb_dir /path/to/rgb \
  --initial_boxes_file initial_targets.json \
  --intrinsics_json intrinsics.json \
  --fps 60 \
  --ostrack_repo source/standalone/test_lk_ostrack/ostrack/OSTrack-main \
  --ostrack_checkpoint /path/to/OSTrack_ep0300.pth.tar \
  --out_csv outputs/multi_tracks.csv \
  --overlay_dir outputs/overlay
```

在没有 checkpoint 的 CPU 环境中，可用 `--disable_fine_tracking` 验证
LK/RANSAC/Kalman 链路，此时不需要两个 OSTrack 路径参数。

### OStrack 自适应触发

多目标模式保留 `fixed` 固定周期策略，并新增 `adaptive` 自适应策略。开发和 A/B
评测阶段默认仍为 `fixed`；启用自适应触发时增加：

```bash
--fine_tracker_policy adaptive \
--adaptive_min_interval_frames 2 \
--adaptive_max_interval_frames 12 \
--adaptive_bad_frames 2
```

自适应策略在 LK/RANSAC 质量稳定时跳过 OStrack；在跟踪进入 `predicted`、原始运动
置信度过低、风险连续超限或达到最大校正间隔时触发。低分、非法或跳变过大的 OStrack
结果不会写入 Kalman，连续失败按 `1, 2, 4, 8` 帧退避。

输出 CSV 会记录 `motion_confidence`、`point_retention`、
`ransac_inlier_ratio`、`bbox_innovation`、`fine_trigger_reason`、`fine_risk`、
`fine_attempted`、`fine_applied` 和 `fine_latency_ms` 等诊断字段。完整设计与调参说明见
[`ADAPTIVE_OSTRACK_DESIGN.md`](ADAPTIVE_OSTRACK_DESIGN.md)。

CSV 每帧每 ID 一行，关键字段包括：

- `yaw_rad/pitch_rad`：当前框中心方向。
- `predicted_yaw_rad/predicted_pitch_rad`：考虑 `prediction_horizon_s` 的提前方向。
- `confidence`：有效点保留率、RANSAC 内点率和前后向误差的联合置信度。
- `selected`：当前主目标。
- `source`：`flow`、`prediction` 或 `ostrack`。

## 主目标切换

目标优先级默认由置信度、距光轴距离、框面积和跟踪稳定性组成。切换必须超过
`priority_switch_margin`，用于避免两个目标分数接近时主目标来回跳变。

## 单目标兼容

原有 `--initial_box` 或 `--initial_box_file` 单目标模式仍保留，继续使用全程 OSTrack。

## Isaac Sim 多目标数据采集

采集器复用 `test11` 的机器人、炮台和 TiledCamera，在每个并行环境的同一画面中
生成多个独立运动的彩色目标。建议先用小规模参数验证场景和相机路径：

默认靶标带黑白高对比特征点；所有特征点都位于标注球半径内。高对比特征只有在
目标占据足够像素时才能帮助 LK。首个序列开始时会输出实际框尺寸，小于 `12 px` 时会
提示无法可靠获得 9～16 个角点。若只想使用纯色球体，可加
`--no-target_features`。
采集器加载本目录中的 `lite3_with_turret.usd`。`--target_depth_speed_scale` 控制
相机光轴方向速度相对横向速度的比例：默认 `0.15`，设为 `0` 表示目标距离不主动变化，
设为 `1` 表示不抑制前后运动。
相机成像参数与 `test7_1/turret_aim_env.py` 保持一致：分辨率 `1024x576`、焦距
`10.0`、水平/垂直孔径 `5.76/3.24`，对应约 `32.13°x18.40°` 视场。在该分辨率下，
采集得到的内参应接近 `fx=fy=1777.78`。无论远程 H60YA 相机是否能解析，采集环境都会
应用这套成像参数；若相机资产无法加载，则在相同挂载路径创建本地 pinhole 相机。
Python 异常会同时写入输出目录的 `failure.log`。

```bash
./isaaclab.sh -p \
  source/standalone/test_lk_ostrack/collect_multi_target_dataset.py \
  --headless \
  --num_envs 1 \
  --num_targets 3 \
  --num_sequences 2 \
  --sequence_length 60 \
  --camera_width 1024 \
  --camera_height 576 \
  --out_dir outputs/isaacsim_multi_smoke
```

小规模采集正常后，可以提高并行数和序列数：

```bash
./isaaclab.sh -p \
  source/standalone/test_lk_ostrack/collect_multi_target_dataset.py \
  --headless \
  --num_envs 4 \
  --num_targets 4 \
  --num_sequences 100 \
  --sequence_length 180 \
  --target_distance_min 80 \
  --target_distance_max 150 \
  --target_speed_min 0.4 \
  --target_speed_max 1.5 \
  --out_dir outputs/isaacsim_multi_100
```

采集目录结构：

```text
dataset/
  dataset_meta.json
  manifest.csv
  sequences/seq_000000_env_000/
    rgb/000000.png
    intrinsics.json
    initial_targets.json
    ground_truth.csv
    sequence_meta.json
```

`ground_truth.csv` 每帧每 ID 一行，包含当前/原始投影框、可见状态、yaw/pitch、
MultiMeshRayCaster 测距 `lidar_range_m` 及其有效标记 `lidar_valid`，以及目标相对相机的
位置和速度（`x_cam_m/y_cam_m/z_cam_m`、`vx_cam_mps/vy_cam_mps/vz_cam_mps`），
目标在机器狗 yaw-only 轴下的位置和相对速度（`x_yaw_m/y_yaw_m/z_yaw_m`、
`vx_yaw_mps/vy_yaw_mps/vz_yaw_mps`），以及采集时刻的相机位姿
（`camera_position_yaw_*`、`rotation_yaw_from_camera_*`）。相机光心在相机轴下的线速度
和角速度仍记录在 `camera_v*_cam_mps`、`camera_w*_cam_radps`。速度模块使用
`yaw/pitch + lidar_range_m` 作为观测，先用采集时刻的 yaw-only 位姿把测量从相机系转换
到 yaw-only 系，再进入 EKF；位置和速度真值只用于监督标签及评估。
机器狗 base 在 yaw-only 坐标系下的线速度和角速度记录在
`base_v*_yaw_mps`、`base_w*_yaw_radps`。其中三维线速度同时作为 GRU 输入，并用于训练
阶段筛选机器狗静止片段；base 角速度不直接进入 GRU，GRU 使用同步的相机角速度。
序列首帧要求所有目标完整可见并满足最小中心间距；
序列开始后仍会保留部分出画目标的真值。

### 仅采集速度文本数据

只训练速度模块时，使用主目标 CSV-only 入口。采集器仍会在内存中读取当前 RGB 帧供
LK/OSTrack 使用，但不会编码 PNG、保留图像序列或创建 `rgb/` 目录。机器狗的 `vx/vy/yaw-rate`
指令会被强制设为 0，并且只有实际 base 线速度和角速度低于门限时才会采集。目标运动会被
强制启用，且 `--target_speed_min` 必须大于 0。

每条完成序列只记录初始化时选中的主目标，`ground_truth.csv` 每帧只有一行，行内
`target_id` 保留该目标在视觉模块中的真实 ID。若视觉模块在序列中途切换主目标，该未完成
序列会被删除并重新采集，避免一个 GRU 窗口混入两个目标。`initial_targets.json` 仍保留首帧
全部目标框，用于复现视觉初始化；`manifest.csv` 中的 `recorded_target_count` 为 1：

```bash
./isaaclab.sh -p \
  source/standalone/test_lk_ostrack/collect_velocity_text_dataset.py \
  --headless \
  --num_envs 4 \
  --num_targets 3 \
  --num_sequences 100 \
  --sequence_length 180 \
  --target_distance_min 80 \
  --target_distance_max 150 \
  --target_speed_min 0.6 \
  --target_speed_max 5.0 \
  --raycast_min_range 65 \
  --raycast_max_range 165 \
  --out_dir outputs/velocity_text_100
```

如需用步态策略维持站立姿态，可额外传入 `--locomotion_policy_path /path/to/policy.pt`；
专用入口仍会把该策略的速度指令范围固定为 0。未提供策略时，环境使用默认站立关节目标。
采集期实际运动门限默认是 `--max_static_base_speed_mps 0.15` 和
`--max_static_base_angular_speed_radps 0.15`，用于排除复位瞬态而不把站立物理抖动误判为行走；
需要更严格的数据时可调低门限或在训练数据构建阶段用 `--max_base_speed_mps` 二次筛选。

默认不依赖 OSTrack checkpoint，使用 LK + 前后向检查 + RANSAC + Kalman 选择并跟踪主目标。
若要采集实际 OSTrack 精跟踪主目标的方向，需同时提供两个路径（每个并行环境会维护自己的
OSTrack 状态）；也可以显式加 `--disable_fine_tracking` 强制只用 LK：

```bash
  --ostrack_repo source/standalone/test_lk_ostrack/ostrack/OSTrack-main \
  --ostrack_checkpoint /path/to/OSTrack_ep0300.pth.tar
```

序列起始帧的仿真投影框只用于初始化 tracker 和写入 `initial_targets.json`，该帧不写入
测量行；后续 `yaw_rad/pitch_rad`、bbox 和可见状态均来自 LK/OSTrack 输出。每帧再把这些
视觉角度送入 MultiMeshRayCaster，`lidar_range_m/lidar_valid` 是唯一的距离测量字段。
目标相对位置/速度字段仍是仿真监督标签，供 EKF/GRU 训练和评估使用，不作为观测输入。

采集器会让一个 MultiMeshRayCaster 同时跟踪全部 `TargetSphere_*`，每个仿真帧只更新一次
射线束。每条命中射线先按 MultiMeshRayCaster 返回的目标 mesh slot 关联到 `target_id`，
再用 yaw/pitch 角窗做几何一致性检查，因此目标靠近或交叉时不会串用距离。默认射线束为
水平 `40 deg`、垂直 `24 deg`、
角分辨率 `0.15 deg`，主/回退关联锥为 `0.75/1.5 deg`。这些参数都可通过
`--raycast_*` 参数覆盖；目标距离默认 `80-150 m`，位于默认测距范围 `65-165 m` 内。

采集目标运动且要求目标始终留在相机范围内时，`--keep_targets_in_view` 会在每步检查
目标完整 bbox、相机距离和安全视场边界；`--target_spawn_border_px` 同时作为初始采样和
运动过程中的像素安全边界。世界固定高度区间默认关闭，此时不存在固定高度上限，但球体
最低点始终不得进入地面。采集器还会在初始化前和每个记录帧中检查真值框内是否确实有
彩色目标像素：初始化失败时自动重采样，序列中途失败时删除该序列并重新采集。因此
`--no-target_height_constraint` 表示“无固定高度上限 + 相机视场 + 地面物理可见性约束”，
而不是允许目标进入地面以下。若命令或旧脚本曾启用固定高度区间，可显式关闭：

```bash
  --dynamic_targets \
  --keep_targets_in_view \
  --target_spawn_border_px 96 \
  --no-target_height_constraint
```

对于 Lite3 正式采集，应传入 locomotion TorchScript 策略来维持闭环站立；仅使用默认
关节目标不能保证机身和相机稳定。若要采集行走、平移和偏航数据，则给该策略输入非零
速度指令；相机自运动字段会被保存，用于训练延迟和自运动补偿。

如果希望仍然运行 locomotion policy 的站立闭环，但把输入指令中的前向/侧向速度和 yaw
角速度全部固定为零，传入 policy 的同时增加 `--locomotion_zero_command`。该选项会将
三个指令范围设为 `(0, 0)`，并把实际范围写入 `dataset_meta.json`：

```bash
  --locomotion_policy_path /path/to/policy.pt \
  --locomotion_zero_command
```

采集器在首次初始化以及每次重新采样目标后都会等待 `1.0 s`，再初始化 tracker 并写入
下一条序列；实际时长同时记录在 `dataset_meta.json` 和 `sequence_meta.json` 的
`collection_warmup_s` 字段中。

CSV-only 模式的 yaw/pitch 来自在线 LK/OSTrack，距离来自使用视觉方向查询的
MultiMeshRayCaster，不使用目标位置真值冒充测量。文本数据同时保留相机坐标和 yaw-only
坐标。默认的 EKF/GRU 训练路径使用 yaw-only
字段；在线 `YawOnlyVelocityObserver` 对真实视觉结果执行同样的采集时刻转换。文本序列
可以直接传给 `python -m test_ekf_gru.training_data`，不需要补充图片。

CSV-only 目录结构：

```text
dataset/
  dataset_meta.json
  manifest.csv
  sequences/seq_000000_env_000/
    intrinsics.json
    initial_targets.json
    ground_truth.csv
    sequence_meta.json
```

采集结束后检查文件完整性：

```bash
PYTHONPATH=source/standalone python3 \
  source/standalone/test_lk_ostrack/check_multi_target_dataset.py \
  outputs/isaacsim_multi_100
```

对任意一个采集序列运行多目标跟踪：

```bash
SEQ=outputs/isaacsim_multi_100/sequences/seq_000000_env_000
PYTHONPATH=source/standalone python3 -m test_lk_ostrack.estimate_video \
  --rgb_dir "$SEQ/rgb" \
  --initial_boxes_file "$SEQ/initial_targets.json" \
  --intrinsics_json "$SEQ/intrinsics.json" \
  --fps 50 \
  --disable_fine_tracking \
  --out_csv "$SEQ/lk_estimates.csv" \
  --overlay_dir "$SEQ/lk_overlay"
```

## 数据集评估

先用纯 LK/RANSAC/Kalman 模式快速评估整个多目标数据集，不需要 OSTrack checkpoint：

```bash
PYTHONPATH=source/standalone python3 -m \
  test_lk_ostrack.evaluate_multi_target_dataset \
  --dataset_dir outputs/isaacsim_multi_100 \
  --out_dir outputs/isaacsim_multi_100_eval \
  --disable_fine_tracking
```

只评估少量序列和帧时，可加入 `--max_sequences 2 --max_frames 60`。启用主目标
OSTrack 精跟踪时，去掉 `--disable_fine_tracking` 并提供模型路径：

```bash
PYTHONPATH=source/standalone python3 -m \
  test_lk_ostrack.evaluate_multi_target_dataset \
  --dataset_dir outputs/isaacsim_multi_100 \
  --out_dir outputs/isaacsim_multi_100_eval_ostrack \
  --ostrack_repo source/standalone/test_lk_ostrack/ostrack/OSTrack-main \
  --ostrack_checkpoint /path/to/OSTrack_ep0300.pth.tar \
  --device cuda:0
```

评估结果会写入输出目录中的 `summary.json`、`summary.csv` 和
`per_sequence.csv`。评估器还会按真值相机系相对速度分箱，生成
`speed_by_bin.csv` 和 `speed_range.json`；后者给出满足验收条件的连续速度区间。

如果 `ground_truth.csv` 含有 `lidar_range_m/lidar_valid` 和
`x_cam_m/y_cam_m/z_cam_m`，评估器还会把完整跟踪管线最终输出的
`yaw_rad/pitch_rad` 与同帧已记录测距融合，在 OpenCV 相机系（x 向右、y 向下、z 向前）
计算三维坐标误差。总体和逐序列输出中会增加：

- `position_samples`、`position_range_availability`：有效融合样本数及其占可见真值的比例；
- `position_3d_mae_m`、`position_3d_rmse_m` 和
  `position_3d_p50/p75/p80/p85/p90/p95/p99/max_m`；
- `position_precision_0_25/0_5/1_0/2_0m`；
- 相机 x/y/z 的 bias 和 MAE、径向 bias/MAE、横向误差 MAE/P95；
- `measured_range_bias/mae/p95_m`。

`speed_by_bin.csv` 和 `speed_range.json` 也会输出每个速度区间的有效坐标样本数、
测距可用率，以及三维误差 MAE、RMSE、P50/P75/P80/P85/P90/P95/P99 和最大值。

坐标估计为
`lidar_range_m * [cos(pitch)sin(yaw), -sin(pitch), cos(pitch)cos(yaw)]`，
真值直接使用 `target_pos_cam`。坐标指标只统计真值可见且 `lidar_valid=1` 的样本；
没有已记录有效测距的旧数据集会输出 `position_samples=0`，不会把 `distance_m` 冒充测距。
`--position_range_adjustment_m` 可在融合前给记录距离增加一个固定偏移，默认值为 0；
它主要用于消融实验。例如半径 0.4 m 的数据若要移除采集时的 `2R/3` 中心补偿，使用
`--position_range_adjustment_m -0.2666666667`。该操作反推的是平均球面命中点距离，
不等同于恢复未保存的原始逐射线点云。

### 跟踪适用速度范围

测速度上限时应使用 `--target_motion_mode camera_lateral`。该模式让目标在相机 x 轴
方向以采样速度左右移动，垂直速度和深度速度均为零，因而
`--target_speed_min/max` 就是受控的实际横向速度。默认的 `random` 模式会随机分解
横向、垂直和深度分量，适合综合测试，但不适合单变量扫速。

建议先固定平台与炮塔，并固定帧率、分辨率、目标大小和纹理，只扫描速度；距离使用
窄区间并在报告中注明。以下命令在 `95-105 m`、`50 Hz`、`1024x576` 条件下采集
`0.5-30 m/s` 的粗扫数据：

```bash
./isaaclab.sh -p \
  source/standalone/test_lk_ostrack/collect_multi_target_dataset.py \
  --headless \
  --num_envs 4 \
  --num_targets 4 \
  --num_sequences 100 \
  --sequence_length 300 \
  --target_distance_min 95 \
  --target_distance_max 105 \
  --no-target_height_constraint \
  --target_speed_min 0.5 \
  --target_speed_max 30 \
  --target_motion_mode camera_lateral \
  --locomotion_zero_command \
  --turret_action_scale 0 \
  --disable_fine_tracking \
  --out_dir outputs/tracking_speed_coarse
```

随后用实际要部署的 LK + OSTrack 配置离线评估同一批图像：

```bash
PYTHONPATH=source/standalone python3 -m \
  test_lk_ostrack.evaluate_multi_target_dataset \
  --dataset_dir outputs/tracking_speed_coarse \
  --out_dir outputs/tracking_speed_coarse_eval \
  --ostrack_repo source/standalone/test_lk_ostrack/ostrack/OSTrack-main \
  --ostrack_checkpoint \
    source/standalone/test_lk_ostrack/ostrack/OSTrack-main/output/checkpoints/OSTrack_ep0300.pth.tar \
  --device cuda:0 \
  --speed_bin_width_mps 1.0 \
  --speed_min_observations 1000 \
  --speed_min_tracking_availability 0.95 \
  --speed_min_success_iou_0_5 0.80 \
  --speed_max_angular_p95_deg 1.5
```

一个速度分箱只有同时满足样本数、可用率、`IoU>=0.5` 成功率和角误差 P95 四个条件
才算通过。`speed_range.json` 中的 `maximum_continuous_supported_speed_mps` 是从最低
有效速度开始、不中断地通过所有分箱时得到的保守上限；不能用高速度处偶然通过的孤立
分箱抬高上限。`speed_by_bin.csv` 同时输出角误差 MAE、P50、P75、P80、P85、P90、P95、P99
和最大值，用于观察主体分布与长尾。默认验收线只是基线，应按实际开火角误差和系统
可用率需求调整。

速度上限只对本次实验条件有效。相机横向运动近似满足
`像素位移/帧 = fx * 横向速度 / (距离 * FPS)`；本项目默认 `fx≈1777.8`，在
`100 m, 50 Hz` 时 `1 m/s≈0.356 px/帧`。因此正式报告至少应同时注明 `m/s`、
`px/帧`、距离、FPS、分辨率、目标像素尺寸以及是否启用 OSTrack。粗扫找到首个失败分箱
后，再在临界区间用更窄的速度范围和 `--speed_bin_width_mps 0.25` 重复采集评估。

若要把数据集中的目标逐个作为单目标运行 OSTrack，并评估方向误差：

```bash
PYTHONPATH=source/standalone python3 -m \
  test_lk_ostrack.evaluate_direction_dataset \
  --dataset_dir outputs/isaacsim_multi_100 \
  --out_dir outputs/isaacsim_direction_eval \
  --ostrack_repo source/standalone/test_lk_ostrack/ostrack/OSTrack-main \
  --ostrack_checkpoint /path/to/OSTrack_ep0300.pth.tar \
  --device cuda:0 \
  --max_sequences 2 \
  --overlay
```

该命令会为每个目标保存估计结果，最终汇总到输出目录的 `summary.csv`。已有估计
默认会复用；需要重新运行 OSTrack 时添加 `--force`。

## 测试

运行全部测试：

```bash
PYTHONPATH=source/standalone python3 -m unittest discover \
  -s source/standalone/test_lk_ostrack/tests -v
```

只运行一个测试文件：

```bash
PYTHONPATH=source/standalone python3 -m unittest \
  test_lk_ostrack.tests.test_multi_target_tracker -v
```

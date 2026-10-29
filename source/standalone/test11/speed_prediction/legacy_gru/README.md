# 普通 13 维 GRU 重训

本目录训练的是 `VelocityPredictorGRU`，不是 TAQ-GRU，也不是学习 Q/R 的 CV 滤波器。网络定义 `speed_prediction_model.py` 保持不变：单层 GRU（默认隐藏维度 256）加原 MLP 输出头，输出三维速度。

## 完整 episode 训练（2026-10-13）

`train_gru_velocity_episodes.py` 只读取 `.taqstream`，使用这次通过检查的完整回合数据。旧的 `train_gru_velocity.py` 只读取短窗口；将新数据目录传给旧入口，仍然会训练短窗口。

新入口保持原网络参数和 13 维输入定义：位置、位置增量（不除以 dt）、机器人线速度、机器人角速度和运动标志。对整段 GRU 输出的每个有效监督帧复用原输出头，使用 SmoothL1 损失。每个 batch 包含若干完整回合，回合内进行完整反向传播，没有 TBPTT 截断；隐藏状态仅在回合边界重新初始化。

初始化与现有 `OnlineLegacyGRU(context="full", initialization_policy="first_valid")` 对齐：等待首次直接跟踪且测距有效的位置，以第二个位置帧开始预测。开始预测后的无效位置、预测位置按保存值处理，不用 GT 补齐输入，不因为 tracker lost/reinitialize 清空记忆。13 维模型没有增加观测有效性输入维度；来源掩码用于首次有效观测判定和指标分组。速度 GT 仅用于监督和评分，保存的 `target_valid_mask` 决定监督范围（保留原数据 0.10/0.12 s 的 burn-in 边界差异）。

在项目根目录执行：

```bash
cd /home/jby/IsaacLab-2.3.2
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
source /home/jby/IsaacLab-2.3.2/_isaac_sim/setup_conda_env.sh

python source/standalone/test11/speed_prediction/legacy_gru/train_gru_velocity_episodes.py \
  --data_dirs /mnt/A/hust_jby/isaaclab_storage/test11/data/legacy_gru_full_episode_ideal_toi_20261012/train/sd_clean_seed20261012_20261012_215946 \
  --save_dir source/standalone/test11/speed_prediction/legacy_gru/full_episode_runs \
  --train_ratio 0.8 --test_ratio 0.1 --seed 42 \
  --hidden_dim 256 --num_layers 1 --dropout 0 \
  --pos_smooth_window 1 \
  --batch_size 32 --epochs 100 --lr 3e-4 \
  --early_stop_patience 20 --num_workers 0 --device cuda:0
```

这会从头训练新模型，按 episode 分成 240 个训练、30 个验证、30 个测试回合。数据文件中的 `split=train` 是原采集划分，新入口会在本次目录内部按回合生成并保存独立划分；归一化仅使用本次训练回合。完整流必须有 `episode_complete=True`，不接受不完整回合或相同 UID 的重复流。

每次运行创建新的上海时间戳目录，保存 `data_manifest.json`、`input_norm.json`、`run_config.json`、`history.json`、`best_model.pt`、`test_metrics.json` 和 `summary.json`。根据验证集完整回合逐帧误差范数 Mean 选择最佳权重，再对测试回合评估一次；测试指标包含 Mean、P50/P75/P80/P85/P90/P95/P99、最大误差和分轴 MAE/RMSE，单位 m/s。这些是完整回合逐帧指标，不是短窗口末帧指标。

在同一指令末尾加 `--prepare_only` 可只检查数据并写出划分及归一化统计，不启动训练。默认 `batch_size=32` 指 32 个完整回合，已在 CUDA 0 / RTX 3090 上通过一个真实训练批次验证；不需要安装或启动 Isaac Sim 来训练保存的张量。

## 从旧权重微调完整 episode

使用 `--init_checkpoint` 加载原模型全部参数和归一化 buffer，新建 AdamW 优化器。网络配置和 `pos_smooth_window` 默认继承 checkpoint；显式指定不同配置会报错。微调期间不会用新数据覆盖原归一化统计，也不会修改原权重文件。这是微调入口，不恢复旧优化器状态或旧训练轮次。

```bash
cd /home/jby/IsaacLab-2.3.2
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
source /home/jby/IsaacLab-2.3.2/_isaac_sim/setup_conda_env.sh

python source/standalone/test11/speed_prediction/legacy_gru/train_gru_velocity_episodes.py \
  --data_dirs /mnt/A/hust_jby/isaaclab_storage/test11/data/legacy_gru_full_episode_ideal_toi_20261012/train/sd_clean_seed20261012_20261012_215946 \
  --init_checkpoint source/standalone/test11/speed_prediction/gru_runs_v2/20261009_152810_116165/best_model.pt \
  --save_dir source/standalone/test11/speed_prediction/legacy_gru/full_episode_finetune_runs \
  --train_ratio 0.8 --test_ratio 0.1 --seed 42 \
  --batch_size 32 --epochs 100 --lr 3e-5 \
  --early_stop_patience 20 --num_workers 0 --device cuda:0
```

这套数据仍分为 240/30/30 个回合，保持先前 seed=42 的划分。首先在验证集评估未微调的旧模型，保存 `initial_val_metrics.json` 和 epoch=0 的 `best_model.pt`，随后用验证 Mean 选择权重。如果所有微调轮次均未优于初始权重，最终 `best_epoch=0`，最佳模型保留旧权重。每轮日志输出三个方向的 MAE；`history.json` 保存 Mean/P95/P99 等完整逐帧指标。本实验暂不改变各轴损失权重，以便单独判断旧权重初始化的效果。

输出位于独立的 `full_episode_finetune_runs/<上海时间戳>/`。`run_config.json`、数据清单及 checkpoint 会记录初始化文件的绝对路径和 SHA-256，以及归一化来源。末尾加 `--prepare_only` 可验证旧权重和真实数据的兼容性并保存配置，不开始微调。

## 数据适配

`train_gru_velocity.py` 支持历史 `sequence_data` chunk，以及 TAQ-v2 的 `taq_velocity_chunk_*.pt`。递归发现 episode 内的 chunk，按 scenario 分层、按 episode 整组划分数据，保存划分清单。同一 episode 的所有窗口和 chunk 不会跨训练、验证、测试集。

TAQ-v2 数据由 `taq_dataset.py` 转为原 13 维输入：

- 位置 3 维。
- 因果平滑后的位置增量 3 维，不除以 `dt`，不直接使用 TAQ 有限差分速度。
- 机器人线速度 3 维。
- 机器人角速度 3 维，z 分量沿用 TAQ 的 yaw-only heading 导数。
- 运动标志 1 维，由机器人水平线速度大于 0.05 m/s 推导。

每窗只监督最后一个有效标签，输入也截止该时刻。变长窗口在 batch 中补零，但 GRU 输出头只读取最后真实输入帧的隐藏状态；没有增加网络层或参数。归一化只统计训练集的真实输入帧，不包含 padding，也不读取验证或测试集。

缺测沿用 TAQ 保存的观测位置，不使用 GT 位置补齐模型输入。TAQ-v1、不同标签语义或 gyro-z 语义被明确拒绝；不同 schema 的数据也不混合训练。此适配保留旧网络结构，不声称完全复现旧 checkpoint 的原采样和标签协议。

## 历史短窗口数据集

```text
/mnt/A/hust_jby/isaaclab_storage/test11/data/velocity_train_fov3d_20260930_005113/train
```

包括 4,800 个 SS 和 7,200 个 SD 窗口，来自 3,192 个 episode。主要为机器狗零运动指令站立、clean 观测；不能据此宣称已验证行走和复杂退化性能。

## 短窗口训练指令

在项目根目录执行：

```bash
cd /home/jby/IsaacLab-2.3.2
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232

python source/standalone/test11/speed_prediction/legacy_gru/train_gru_velocity.py \
  --data_dirs /mnt/A/hust_jby/isaaclab_storage/test11/data/velocity_train_fov3d_20260930_005113/train \
  --save_dir /mnt/A/hust_jby/isaaclab_storage/test11/gru_runs/legacy_fov3d \
  --train_ratio 0.8 --test_ratio 0.1 --seed 42 \
  --hidden_dim 256 --num_layers 1 --dropout 0 \
  --pos_smooth_window 1 \
  --batch_size 512 --epochs 40 --lr 1e-3 \
  --num_workers 4 --device cuda:0
```

只想检查数据和划分时，在同一命令末尾增加 `--prepare_only`。该模式只写划分清单和归一化统计，不创建或训练模型。预检查会创建独立时间戳目录；正式运行需去掉该参数。

本次代码适配没有启动这套真实数据的正式训练；单元测试中的短训练仅使用临时生成的合成样本。

## 结果

每次运行在 `--save_dir` 下创建独立的上海时间时间戳目录，包含：

- `data_manifest.json`：文件清单、episode 数、样本数、划分种子和比例。
- `input_norm.json`：仅训练集的输入均值和标准差。
- `best_model.pt`：按验证 loss 选择的权重及预处理、标签语义和划分记录。
- `history.json`：训练和验证历史，含验证速度误差范数 Mean/P95/P99。
- `test_metrics.json`：若 `--test_ratio > 0`，训练结束后加载最佳权重，对预留测试集评估一次。
- `summary.json`：最佳 epoch、样本数和最终测试结果。

测试指标是每个窗口末监督帧的预测误差，不是完整 episode 的逐帧指标。不要直接与不同数据划分或逐帧评分的 CV 滤波结果比较。

`test_gru_velocity.py` 的外部评估入口仍面向旧 26 维 chunk；当前 TAQ-v2 重训使用训练脚本自动保存的独立测试结果，不应把 TAQ-v2 路径直接交给旧外部评估脚本。

## 实时视觉联调

`evaluate_vision_gru.py` 启动现有 test11 仿真场景，使用
`test_lk_ostrack.multi_target_tracker.MultiTargetVisionTracker` 的 LK/自适应 OSTrack，
以及跟踪方向驱动的 RayCaster 测距。普通 GRU 每帧接收真实视觉输出和机器人速度，
不是先采集完再离线回放；不训练模型，不用 GT 位置补齐输入。
复用 `collect_vision_data.main(frame_observer=..., collection_finished=...)` 的可选帧回调，
在仿真关闭前保存报告，避免 Kit 关闭过程直接结束 Python 进程导致指标未落盘。
原采集命令行为不变。

默认 SD clean、机身相机、机器狗零运动指令，距离/速度等场景参数沿用现有 collector。
默认权重为 `gru_runs_v2/20261009_152810_116165/best_model.pt`。
这是 test11 中的单目标球体场景，不是 test_lk_ostrack 自身的多目标环境。
初次和 tracker 重初始化仍使用仿真 bbox 检测代理，因此不声称完成了真实检测器端到端验收。

```bash
cd /home/jby/IsaacLab-2.3.2
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
source _isaac_sim/setup_conda_env.sh

CUDA_VISIBLE_DEVICES=0 python source/standalone/test11/speed_prediction/legacy_gru/evaluate_vision_gru.py \
  --scenario sd --profile clean --condition D \
  --context both --num_envs 4 --num_steps 2000 --device cuda:0 \
  --headless
```

去掉 `--headless` 可查看仿真画面。`--scenario ss` 测静止目标，`--profile mixed` 测退化场景。
`--context rolling` 用原训练匹配的 0.48 秒滑动窗口，`full` 连续保留状态，`both` 在同一视觉流上同时评分。
默认 `--initialization_policy first_valid`：预热及首次直接有效观测之前不更新 GRU 历史或隐藏状态；
首次有效位置仅建立历史，从第二个位置帧开始预测。相机帧仍完整保存在观测流和逐帧报告中。
只在新的仿真 episode 重置状态，开始接收输入后 tracker 丢失或重初始化不额外清除 GRU 状态。
开始接收输入后的缺测沿用视觉预测位置；无效位置按 TAQ 存储规则填零，不使用 GT 修复，不添加新物理传播。
`--initialization_policy zero_filled` 保留修改前的行为，仅用于对照。
`--measure_latency` 可测预处理加 GRU 的同步耗时，不包含视觉和仿真耗时；同步测量会影响运行吞吐。
非零机器人命令需使用 `--condition D`，避免训练采集条件 `none` 的静止过滤。

默认结果写入本目录 `vision_eval_runs/<上海时间戳>/`；也可通过 `--out_dir` 指定其他盘的新目录。
输出 `frame_metrics.csv`、`summary.json`、`REPORT.md` 和可回放的 `observations/`。
脚本拒绝覆盖已存在的输出目录，失败运行会标记 `completed=false`。

报告包含逐帧误差范数 Mean/P50/P95/P99、最大值、分轴 MAE/RMSE/bias、超阈值比例，
并分为全帧/地上、直接观测/预测/丢失、首次有效观测后的启动 0.5 秒/其余阶段。
只评分已有至少两个位置帧、完成 episode 0.1 秒 burn-in 且已出现首次直接有效观测的时刻。
完全没有直接有效观测的 episode 保留在可用性计数中，不产生速度精度分数。
`summary.json` 的 `episode_timing` 记录首次直接观测、首次预测和首次评分时刻，以及首次输出等待时间；
`initialization_wait_frames` 记录初始化门控等待帧，不把等待时间当成零误差。
GT 速度和目标高度只在预测之后参与评分，地上筛选只作用于次级统计，不改变模型输入。
`condition D` 不启用静止过滤且保留完整异常流，指标不能直接等同于原训练集末帧验证结果。

### 初始化同流对照

无需重启仿真，可在保存的完整视觉流上复用相同权重比较初始化策略：

```bash
python source/standalone/test11/speed_prediction/legacy_gru/compare_vision_gru_initialization.py \
  --data_root /tmp/legacy_gru_vision_smoke_20261009_03/observations \
  --out_dir /tmp/legacy_gru_initialization_comparison --device cpu
```

该工具复用实时适配器，分别保存两种策略的完整逐帧结果、启动和稳定阶段指标、输出等待时间。
`REPORT.md` 和 `comparison.json` 使用两种策略的共同评分时刻，避免因首次有效位置不再输出而产生评分偏差。
后续缺测不跳过，不增加首次有效观测后的 burn-in，不重新训练权重；输出目录必须不存在。

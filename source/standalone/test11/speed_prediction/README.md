# 速度预测代码目录

代码按模型分类；本次只调整组织和导入，不改变网络结构、损失、训练参数、数据格式或已有权重。

## 目录

```text
speed_prediction/
├── legacy_gru/                 早期普通 GRU：13 维输入，窗口末帧速度输出
│   ├── speed_prediction_model.py
│   ├── train_gru_velocity.py
│   ├── taq_dataset.py
│   └── test_gru_velocity.py
├── taq_gru/                    TAQ-GRU：24 维特征，时间感知、质量门控、残差速度
│   ├── time_aware_velocity_model.py
│   ├── online_taq_velocity.py
│   ├── taq_velocity_training.py
│   ├── train_taq_velocity.py
│   ├── train_taq_velocity_episodes.py
│   ├── train_stage2_balanced.py
│   ├── evaluate_taq_velocity.py
│   └── evaluate_taq_velocity_streaming.py
├── common/                     跨模型共享的观测与物理接口
│   ├── time_aware_velocity_data.py
│   ├── stage2_degradation.py
│   ├── sensor_time_contract.py
│   └── filter_state_output.py
├── data_tools/                 数据审计，不负责训练网络
│   ├── audit_stage1_ideal_v2.py
│   └── audit_stage2_data.py
├── tests/                      数值、数据与旧路径兼容性测试
├── gru_runs/                   已有权重，保持原路径
├── stage1_v2_runs/              已有 TAQ-v2 权重，保持原路径
└── 原同名 .py                  兼容转发入口，不再存放实际实现
```

学习 Q/R 的 CV 滤波器、实验 GRU 和 TCN 当前实现仍在 `../velocity_error_validation/tools/optimization_components.py`，完整流 TBPTT 训练仍在该目录的 `train_flow_ablations.py`。它们属于独立优化实验，本次没有迁移，也没有混入 TAQ-GRU 的训练入口。

## 选择 GRU 训练入口

- 重新训练当前 TAQ-v2 模型：`taq_gru/train_taq_velocity.py`。
- 完整 episode 训练 TAQ-v2：`taq_gru/train_taq_velocity_episodes.py`，复用指定 manifest 的 train/val/test episode 划分，支持仅 SD 数据。
- 按 ideal/Stage-2 受控比例混合训练 TAQ-v2：`taq_gru/train_stage2_balanced.py`。
- 复现实验中的旧 13 维模型：`legacy_gru/train_gru_velocity.py`。
- 分别评估窗口和完整 episode：`taq_gru/evaluate_taq_velocity.py`、`taq_gru/evaluate_taq_velocity_streaming.py`。

旧 GRU 的训练入口现在也支持 TAQ-v2 chunk，通过 `legacy_gru/taq_dataset.py` 构造原 13 维输入，处理变长窗口并按 episode 划分训练、验证和可选测试集。历史标签与 TAQ-v2 标签不能混用；TAQ-v1 仍被拒绝。具体指令见 `legacy_gru/README.md`。

从项目根目录查看参数：

```bash
python source/standalone/test11/speed_prediction/taq_gru/train_taq_velocity.py --help
python source/standalone/test11/speed_prediction/legacy_gru/train_gru_velocity.py --help
```

TAQ-v2 的重新训练命令示例（本次没有执行训练）：

```bash
python source/standalone/test11/speed_prediction/taq_gru/train_taq_velocity.py \
  --train_dirs source/standalone/test11/data/stage1_ideal_v2/train/ss \
               source/standalone/test11/data/stage1_ideal_v2/train/sd \
  --val_dirs source/standalone/test11/data/stage1_ideal_v2/val/ss \
             source/standalone/test11/data/stage1_ideal_v2/val/sd \
  --save_dir source/standalone/test11/speed_prediction/stage1_v2_runs/retrain \
  --device cuda:0
```

运行时使用已配置 PyTorch 的 `isaac_lab_232` 环境。此例沿用历史 ideal-v2 数据，不是新采样/时间协议的训练方案；如要训练完整流或改变数据协议，应另行制定实验配置。

完整 episode 重训示例（默认保存到 `taq_gru/full_episode_runs/<时间戳>/`）：

```bash
python source/standalone/test11/speed_prediction/taq_gru/train_taq_velocity_episodes.py \
  --reference_manifest source/standalone/test11/speed_prediction/legacy_gru/full_episode_finetune_runs/20261013_163052_278937/data_manifest.json \
  --save_dir source/standalone/test11/speed_prediction/taq_gru/full_episode_runs \
  --device cuda:1 --epochs 100 --batch_size 32
```

此入口保持 TAQ 网络和默认损失不变，从头初始化权重、仅从训练 episode 计算归一化统计量，并对完整 episode 做 BPTT。按存储的 `state_reset_mask` 清除状态，baseline 同步分段；与 legacy 的仅 episode 边界重置存在区别。监督帧对齐 legacy 的 first-valid 初始化规则。模型按验证集平均速度向量误差选择，独立测试集仅在选择结束后评估。入口拒绝 GT/oracle 方向、距离或 GT 可见性门控数据，检查文件及 episode ID 不交叉。加 `--prepare_only` 可仅检查数据；加 `--old_taq_checkpoint` 可在同一测试划分比较旧 TAQ。仅 clean SD 数据不能验证 SS 泛化或丢帧/时延鲁棒性。

完整 episode 入口可用 `--baseline_window_s 0.24` 做固定窗对照，默认是 `0.48` 秒。
窗口会写入模型配置、训练参数与 checkpoint；加载后在线端使用该模型窗口。

## 兼容性

新建 TAQ 模型和三个训练入口默认使用 `odometry_aligned_relative_velocity_v2`：
积分已有 yaw rate 与基座线速度，将历史观测转到共同参考系拟合，再转回当前 yaw 轴并减去基座当前线速度。
24 维输入及目标相对平移速度标签保持相同。新 baseline 只使用观测和已有机器人运动特征。

checkpoint 的顶层 `baseline_version` 与 `model_config.baseline_version` 必须一致。
旧 checkpoint 自动选择 `physical_relative_velocity_v1`，旧文件不必迁移或重写。
新旧版本均可通过标准评估入口和全链路 loader 加载；冲突或未知版本会被拒绝。
重新训练会生成新版本权重，旧权重的修正头不能直接搭配新 baseline。

训练时可显式加 `--baseline_version odometry_aligned_relative_velocity_v2`，
或用 `--baseline_version physical_relative_velocity_v1` 做原算法对照。
新 baseline 在线缓存以模型的时间窗为准，高帧率时允许超过 `max_history_steps`，避免与离线窗口不一致；
旧 baseline 保留原来的帧数上限。
直接调用底层 `weighted_velocity_baseline()` 的默认版本及历史常量 `TAQ_BASELINE_VERSION` 仍为 v1，
供已有实验工具兼容。新代码应调用模型的 `compute_velocity_baseline()`，或显式传入 `baseline_version`。

旧脚本路径和导入路径继续可用，例如 `speed_prediction.time_aware_velocity_model`、直接导入 `speed_prediction_model`，以及按文件路径动态加载旧模型定义。旧路径转发到新实现，没有复制两份模型代码；已有 checkpoint 路径不变。

新增代码优先使用分类后的路径，例如：

```python
from speed_prediction.taq_gru.time_aware_velocity_model import TimeAwareQualityGatedGRU
from speed_prediction.common.time_aware_velocity_data import build_time_aware_features
from speed_prediction.legacy_gru.speed_prediction_model import VelocityPredictorGRU
```

这些导入需要 `source/standalone/test11` 在 Python 搜索路径中；直接执行新的 CLI 脚本时会自行处理。

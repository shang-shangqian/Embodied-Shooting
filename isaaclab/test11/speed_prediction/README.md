# 速度估计：采集、训练与测试

支持普通 13 维 GRU 和 TAQ-GRU。输入为 LK/OSTrack 位置历史、机器人运动及跟踪质量，
标签为 yaw-only base 坐标系中的目标相对平移速度。两种网络的权重不能互换。

## 文件说明

| 文件或目录 | 作用 |
| --- | --- |
| `collect_vision_data.py` | Isaac Lab 采集入口，输出 TAQ-v2 短窗口，可同时保存完整 episode。 |
| `train_gru_velocity.py` | 普通 GRU 短窗口训练的兼容入口。 |
| `test_gru_velocity.py` | 历史 26 维 chunk 的普通 GRU 测试入口；不是 TAQ-v2 测试入口。 |
| `evaluate_gru_velocity.py` | TAQ-v2 训练的普通 GRU 独立评估入口，支持完整 episode/短窗口和测试清单。 |
| `train_taq_velocity.py` | TAQ-GRU 短窗口训练的兼容入口。 |
| `train_stage2_balanced.py` | ideal/退化数据按受控比例混合训练 TAQ-GRU 的兼容入口。 |
| `evaluate_taq_velocity.py` | TAQ-GRU 短窗口评估的兼容入口。 |
| `evaluate_taq_velocity_streaming.py` | TAQ-GRU 完整流评估的兼容入口。 |
| `online_taq_velocity.py` | TAQ 在线推理兼容入口。 |
| `speed_prediction_model.py`、`time_aware_velocity_model.py` | 普通 GRU/TAQ 网络定义的兼容入口。 |
| `time_aware_velocity_data.py`、`stage2_degradation.py` | 数据格式/特征和退化模拟的兼容入口。 |
| `sensor_time_contract.py`、`filter_state_output.py` | 采样时间协议和滤波输出接口的兼容入口。 |
| `taq_velocity_training.py` | TAQ 损失与指标计算的兼容入口。 |
| `audit_stage1_ideal_v2.py`、`audit_stage2_data.py` | 数据审计兼容入口。 |
| `__init__.py`、`_compat.py` | 包定义和入口转发，不复制模型实现。 |
| [data_collection/](data_collection/README.md) | 速度采集所需的机器人、炮塔和目标运动环境。 |
| [legacy_gru/](legacy_gru/README.md) | 普通 GRU 网络、短窗/完整 episode 训练、离线评估和实时视觉联调。 |
| [taq_gru/](taq_gru/README.md) | TAQ 网络、短窗/完整 episode/均衡训练及评估。 |
| [common/](common/README.md) | 数据协议、因果特征、退化、时间协议及连续目标运动工具。 |
| [data_tools/](data_tools/README.md) | 数据完整性与观测来源审计。 |
| [tests/](tests/README.md) | 数据隔离、数值、训练/评估和在线一致性测试。 |

## 环境

所有命令从本仓库根目录执行，路径占位符替换为实际文件。只训练或评估已有张量数据：

```bash
python -m pip install -e '.[velocity,dev]'
```

采集及实时视觉联调使用已配置的 Isaac Lab v2.3.2、Isaac Sim 5.1.0 和 Python 3.11 环境，
需提供匹配 Lite3 的行走策略。在该环境中执行：

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab-2.3.2
python -m pip install -e '.[decision,ostrack,dev]'
```
短窗训练和离线评估不启动仿真，也不需要 OSTrack 或行走权重。

## 数据采集

采集 SD 动态目标，同时保存完整 episode：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test11/speed_prediction/collect_vision_data.py \
  --headless --scenario sd --out_dir data/velocity/sd \
  --locomotion_policy /path/to/policy.pt --save_episode_stream
```

采集 SS 时改为 `--scenario ss`，并换一个输出目录。未提供 OSTrack 参数时使用 LK；
启用 OSTrack 增加 `--ostrack_repo isaaclab/test_lk_ostrack/ostrack/OSTrack-main`
和 `--ostrack_checkpoint checkpoints/OSTrack_ep0300.pth.tar`，这两个参数需成对提供。
默认 16 个环境、10,000 步、clean 观测、机器人零运动指令，行走策略负责站立。
内置资产复用视觉模块，并在采集环境生成地面；不需要其他实验目录。

输出包括 `taq_velocity_chunk_*.pt`、元数据及审计记录；增加上述流参数后还输出 `.taqstream`。
完整 episode 训练只接受标记完整的流，末尾截断的回合不会作为完整回合训练。
短窗口和完整流是两种评分协议，不应混合比较误差。

## 训练

推荐当前使用的完整 episode 普通 GRU：

```bash
python -m test11.speed_prediction.legacy_gru.train_gru_velocity_episodes \
  --data_dirs data/velocity/sd
```

默认按 episode 划分 train/val/test 为 80%/10%/10%；每个场景至少需要 3 个独立 episode。
输出到 `outputs/velocity/legacy_full/<时间戳>/`，包括 `best_model.pt`、`data_manifest.json`、
`history.json` 和 `test_metrics.json`。模型按验证集选择，训练结束后才评估独立测试集。
微调已有权重增加 `--init_checkpoint /path/to/best_model.pt`；只检查数据增加 `--prepare_only`。

TAQ 完整 episode 训练复用普通 GRU 的同一划分：

```bash
python -m test11.speed_prediction.taq_gru.train_taq_velocity_episodes \
  --reference_manifest outputs/velocity/legacy_full/<时间戳>/data_manifest.json
```

输出到 `outputs/velocity/taq_full/<时间戳>/`，也会在模型选择后保存独立测试指标。
保留清单中的真实路径；换服务器后需同步数据并更新清单，或重新生成划分。

需要短窗口训练时：

```bash
python -m test11.speed_prediction.train_gru_velocity \
  --data_dirs /path/to/window_chunks --train_ratio 0.8 --test_ratio 0.1
python -m test11.speed_prediction.train_taq_velocity \
  --train_dirs /path/to/train_chunks --val_dirs /path/to/val_chunks
```

普通 GRU 短窗默认不留测试集，上例显式留出 10%；TAQ 短窗要求 train/val 两组各自同时包含 SS 和 SD，
且两组来自不同 episode；多个场景目录可跟在同一个参数后。仅 SD 数据可使用上述完整 episode 入口。
单独采集验证或测试数据时，使用不同 `--seed`、`--split val`/`--split test` 和输出目录，
不要只把相同轨迹复制到不同目录。短窗输出分别位于 `outputs/velocity/legacy_window/` 和 `taq_window/`。

## 测试

普通 GRU 完整 episode：默认只评估清单中的 test 划分。

```bash
python -m test11.speed_prediction.evaluate_gru_velocity \
  --checkpoint /path/to/best_model.pt --manifest /path/to/data_manifest.json
```

短窗口权重增加 `--context window`。也可用 `--data_dirs /path/to/heldout_data` 替代清单，
此时该目录须只包含独立测试数据。需要保存指标时增加 `--out_json outputs/velocity/gru_test.json`。
输出速度向量误差 Mean/P95/P99 和分轴 MAE/RMSE；完整回合按监督帧评分，短窗按窗口末监督帧评分。

TAQ 短窗口和完整流分别使用：

```bash
python -m test11.speed_prediction.evaluate_taq_velocity \
  --data_dirs /path/to/test_chunks --checkpoint /path/to/taq_model.pt
python -m test11.speed_prediction.evaluate_taq_velocity_streaming \
  --data_root /path/to/heldout_streams --checkpoint /path/to/taq_model.pt
```

TAQ 流入口会评估目录下所有 `.taqstream`，不要传入包含训练/验证回合的总目录。
完整 episode 训练已按清单自动生成 `test_metrics.json`，不需要为了测试再次训练。

数据审计和单元测试：

```bash
python -m test11.speed_prediction.audit_stage2_data data/velocity
python -m pytest -q isaaclab/test11/speed_prediction/tests
```

实时 LK/OSTrack + 普通 GRU 测试见 [legacy_gru/README.md](legacy_gru/README.md#实时视觉测试)。
模型权重、采集数据、日志和训练结果不随仓库上传。

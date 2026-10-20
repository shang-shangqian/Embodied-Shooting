# 速度模型训练与测试

此目录保留本地原始代码。数据采集器位于上一级 `test11/collect_vision_data.py`，不在此目录重复放置。

| 文件或目录 | 作用 |
| --- | --- |
| `legacy_gru/` | 普通 13 维 GRU、短窗/完整回合训练、离线评估及在线推理。 |
| `taq_gru/` | TAQ-GRU、短窗/完整回合训练、均衡训练和流评估。 |
| `common/` | 数据协议、特征、退化、时间协议和目标运动。 |
| `data_tools/` | 理想/退化数据审计。 |
| `tests/` | 数据、训练、评估和在线一致性测试；不含依赖其他实验目录的布局测试。 |
| 根目录同名 `.py`、`_compat.py`、`__init__.py` | 原有包定义及兼容转发入口。 |

所有命令从仓库根目录执行，已有张量数据只需 PyTorch，不需要仿真器。

## 完整 episode 训练

```bash
python source/standalone/test11/speed_prediction/legacy_gru/train_gru_velocity_episodes.py \
  --data_dirs /path/to/episodes --save_dir outputs/velocity/legacy_full
python source/standalone/test11/speed_prediction/taq_gru/train_taq_velocity_episodes.py \
  --reference_manifest /path/to/data_manifest.json --save_dir outputs/velocity/taq_full
```

普通 GRU 默认按 episode 划分为 80%/10%/10%，每个场景至少 3 个回合；TAQ 复用其清单。
两者均在验证集选择模型后生成独立 `test_metrics.json`。只检查数据可增加 `--prepare_only`，
普通 GRU 微调可增加 `--init_checkpoint /path/to/best_model.pt`。

## 独立测试

```bash
python source/standalone/test11/speed_prediction/evaluate_gru_velocity.py \
  --checkpoint /path/to/best_model.pt --manifest /path/to/data_manifest.json
python source/standalone/test11/speed_prediction/evaluate_taq_velocity_streaming.py \
  --checkpoint /path/to/taq_model.pt --data_root /path/to/heldout_streams
```

普通 GRU 默认读取清单的 test 划分。TAQ 流入口评估目录下全部流，须使用独立测试目录。
不要把训练集和验证集混入测试目录。

## 短窗口训练与测试

```bash
python source/standalone/test11/speed_prediction/train_gru_velocity.py \
  --data_dirs /path/to/window_chunks --train_ratio 0.8 --test_ratio 0.1 --save_dir outputs/velocity/legacy_window
python source/standalone/test11/speed_prediction/train_taq_velocity.py \
  --train_dirs /path/to/train_chunks --val_dirs /path/to/val_chunks --save_dir outputs/velocity/taq_window
python source/standalone/test11/speed_prediction/evaluate_taq_velocity.py \
  --data_dirs /path/to/test_chunks --checkpoint /path/to/taq_model.pt
```

普通 GRU 的短窗独立测试使用 `evaluate_gru_velocity.py` 并增加 `--context window`。
`test_gru_velocity.py` 只用于历史 26 维 chunk，不能直接评估 TAQ-v2 数据。
TAQ 短窗训练要求 train/val 两组各自含 SS 和 SD，且 episode 隔离；仅 SD 使用完整回合入口。
真实权重、数据、训练记录和临时分析脚本不随代码上传。

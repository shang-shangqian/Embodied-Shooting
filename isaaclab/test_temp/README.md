本文件夹主要用于目标的速度预测。整体流程是：在 IsaacLab 环境中采集视觉方向、raycast 距离、目标相对位置序列和速度标签，保存成 `velocity_seq_chunk_*.pt`，再用 GRU 训练目标速度预测模型。

## 代码文件说明

- `turret_aim_env.py`：基础炮塔瞄准环境。负责加载机器人与炮塔资产、创建相机和 raycast 传感器、采样远距离目标、计算相机方向标签、目标速度标签、目标距离、弹道命中和环境状态。
- `turret_aim_env_collect.py`：数据采集专用环境。继承 `TurretAimEnv`，关闭正常奖励，并只在目标真实进入图像、相机姿态和标签都有效时保存样本，适合生成速度预测训练数据。
- `collect_vision_data.py`：速度序列数据采集入口。启动 Isaac Sim，运行采集环境，加载视觉方向模型，结合 raycast 距离生成目标位置序列，并保存 `sequence_data` 和 `target_velocity_base` 到 `velocity_seq_chunk_*.pt`。
- `speed_prediction_model.py`：速度预测模型定义。包含 `VelocityPredictorGRU`，输入一段目标位置/角度/距离/运动状态序列，输出目标在 yaw-only base 坐标系下的速度 `[vx, vy, vz]`。
- `train_gru_velocity.py`：GRU 速度模型训练脚本。读取 `velocity_seq_chunk_*.pt` 数据，构造序列数据集，划分训练/验证文件，训练 `VelocityPredictorGRU`，保存 `best_model.pt`、训练配置和 `history.json`。
- `test_gru_velocity.py`：GRU 速度模型测试脚本。加载训练好的 `best_model.pt`，在指定测试目录上评估速度预测误差，并按距离、角度区域等分组输出统计结果，可选保存 JSON。
- `plot_training_history.py`：训练曲线绘图工具。读取 `train_gru_velocity.py` 生成的 `history.json`，绘制训练/验证 loss 和相关指标曲线。
- `exact_data_from_pt.py`：`.pt` 数据检查工具。读取采集得到的 chunk 文件，打印 `meta`、`sequence_data`、`target_velocity_base` 等字段和样本内容，用于确认数据格式、维度和标签是否正确。


## 数据文件格式

`collect_vision_data.py` 保存的每个 chunk 通常包含：

- `sequence_data`：目标位置、视觉角度、raycast 距离、运动状态等时间序列输入。
- `target_velocity_base`：窗口末帧对应的目标速度标签，坐标系为 yaw-only base。
- `meta`：采集时间、序列长度、采样间隔、数据字段说明等元信息。


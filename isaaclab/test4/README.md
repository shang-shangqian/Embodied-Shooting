
## 顶层文件

- `turret_aim_env.py`：核心强化学习环境。定义 `TurretAimEnvCfg` 和 `TurretAimEnv`，负责加载带炮塔的机器人资产、配置相机、采样目标、处理 yaw/pitch 控制、计算瞄准误差、弹道命中指标、奖励、终止条件和评估指标。

## vision_cnn 子目录

- `vision_cnn/turret_aim_env_collect.py`：专门用于视觉数据采集的环境版本。继承 `TurretAimEnv`，关闭正常奖励，重点保证只在目标真实进入图像、相机和标签有效时保存样本。
- `vision_cnn/collect_vision_data.py`：视觉数据采集脚本。启动 Isaac Sim，创建采集环境，随机控制炮塔扫描目标，并把图像序列、方向标签、速度标签和状态量保存成 `vision_chunk_*.pt` 数据块。
- `vision_cnn/vision_model1.py`：视觉模型定义。包含 CNN 图像编码器、方向预测头和损失函数；当前版本会取图像序列的最后一帧，并拼接 `dog_moving_flag` 这类运动状态标志来预测目标方向 `[yaw_cam, pitch_cam]`。
- `vision_cnn/test_vision1.py`：测试文件。
- `vision_cnn/train_vision1.py`：视觉模型训练脚本。读取 `collect_vision_data.py` 生成的 `vision_chunk_*.pt` 数据集，按 chunk 划分训练集和验证集。


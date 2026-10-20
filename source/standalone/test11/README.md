# 速度数据采集与预测

| 文件或目录 | 作用 |
| --- | --- |
| `collect_vision_data.py` | TAQ-v2 速度数据采集，支持短窗与完整流。 |
| `turret_aim_env.py` | 原始机器人、相机、测距及物理环境。 |
| `turret_aim_env_collect.py` | 采集预热、目标轨迹和验证环境。 |
| [speed_prediction/](speed_prediction/README.md) | 普通 GRU/TAQ 模型、训练、测试、数据工具和已有数值测试。 |
| `lite3/`、`turret/`、USD/URDF | 本地模型资产原文件。 |

代码按本地路径和原内容发布；不再将采集器搬到 `speed_prediction/`，也不生成 `data_collection/` 副本。
未选择历史采集 shell、GT-raycast 旧脚本、独立优化实验、数据和训练输出。
默认不启用后坐力；该可选功能依赖此次未发布的 `model_with_recoil`。

设置外部 Isaac Lab SDK 后，从仓库根目录采集：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p source/standalone/test11/collect_vision_data.py \
  --headless --scenario sd --out_dir data/velocity/sd \
  --locomotion_policy /path/to/policy.pt --save_episode_stream
```

SS 数据改用 `--scenario ss` 并更换输出目录。默认 LK；OSTrack 源码和权重需成对显式提供。
训练和评估不需要启动仿真，命令见速度模块 README。

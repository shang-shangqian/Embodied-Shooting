# 速度数据采集环境

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 采集环境包定义。 |
| `turret_aim_env.py` | 机器人/炮塔、相机、测距、监督、行走策略和基础物理环境。 |
| `turret_aim_env_collect.py` | 无训练奖励的采集环境，处理目标轨迹、视场约束和验证相机。 |

环境来自本地 test11。发布版复用 `test_lk_ostrack` 的资产、RayCaster 兼容层和后坐载荷，
并默认生成地面，不需要复制其他实验目录。采集算法、标签协议和网络结构沿用原实现。

完成[仿真环境安装](../README.md#环境)，从仓库根目录执行：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p isaaclab/test11/speed_prediction/collect_vision_data.py \
  --headless --scenario sd --out_dir data/velocity/sd \
  --locomotion_policy /path/to/policy.pt --save_episode_stream
```

本目录不直接训练网络；采集结果的训练和测试见[上级说明](../README.md)。

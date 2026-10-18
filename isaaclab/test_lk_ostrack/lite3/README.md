# Lite3 机器人资产

| 文件或目录 | 作用 |
| --- | --- |
| `Lite3.usd` | Lite3 机器人 USD 入口。 |
| `configuration/Lite3_base.usd` | 基础模型层。 |
| `configuration/Lite3_robot.usd` | 机器人模型层。 |
| `configuration/Lite3_physics.usd` | 物理属性层。 |
| `configuration/Lite3_sensor.usd` | 传感器层。 |

由上级 `lite3_with_turret.usd` 组合加载。本目录无 Python 代码和独立训练/测试入口，
通过[上级模块](../README.md)的仿真采集或决策训练/评估检查加载结果。

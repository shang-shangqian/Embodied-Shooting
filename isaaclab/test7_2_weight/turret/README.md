# 炮塔资产

| 文件或目录 | 作用 |
| --- | --- |
| `turret.usd` | 炮塔 USD 入口。 |
| `configuration/turret_base.usd` | 基础模型层。 |
| `configuration/turret_physics.usd` | 物理属性层。 |
| `configuration/turret_sensor.usd` | 传感器层。 |

上级目录的 `turret_with_edite.usd`、`lite3_with_turret.usd` 和 `turret.urdf` 提供组合场景与模型描述。
本目录无 Python 代码和独立训练/测试入口，使用[上级模块](../README.md)的仿真命令检查加载。

# 后坐载荷与仿真注入

根据内弹道解生成载荷时程，并按物理步长保冲量地注入机器人。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 导出参数、内弹道和载荷接口，可选导出依赖 PyTorch 的注入接口。 |
| `params.py` | 武器、内弹道、导气室、撞击和载荷合成参数。 |
| `interior_ballistics.py` | 内弹道 RK4 求解、参数标定和起动条件敏感性计算。 |
| `loads.py` | 合成各类载荷、按物理步长重采样和冲量自检。 |
| `injection.py` | 向 articulation 的武器连杆施加局部后坐力并管理发射状态。 |
| `integration.py` | 创建注入器并生成接入炮塔环境的示例代码。 |
| `_compat.py` | 兼容不同 NumPy 版本的梯形积分函数。 |

本目录不是学习模型，无训练入口。从仓库根目录进行离线载荷自检：

```bash
python -c "from test_lk_ostrack.recoil import WeaponParams, generate_recoil_loads; print(generate_recoil_loads(WeaponParams()).audit())"
```

离线载荷计算只需基础依赖；注入仿真需要 PyTorch 和 Isaac Lab。
本分支没有独立 `validate.py` 入口，仿真采集命令见[上级说明](../README.md#运行命令)。

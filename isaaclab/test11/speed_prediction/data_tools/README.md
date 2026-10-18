# 数据审计

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 审计工具包定义。 |
| `audit_stage1_ideal_v2.py` | 检查理想 TAQ-v2 数据的标签、窗口间隔、监督预热及流格式。 |
| `audit_stage2_data.py` | 汇总退化数据的场景、来源、可观测性、窗口与 episode 分布。 |

从仓库根目录执行，无需启动仿真：

```bash
python -m test11.speed_prediction.audit_stage1_ideal_v2 /path/to/clean_data
python -m test11.speed_prediction.audit_stage2_data /path/to/collected_data
```

审计不训练模型；两者面向不同的数据检查需求，不应把非理想数据交给理想条件硬门限做验收。

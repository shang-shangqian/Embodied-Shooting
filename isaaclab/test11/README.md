# test11：速度估计

| 文件或目录 | 作用 |
| --- | --- |
| `__init__.py` | 定义 `test11` 包。 |
| [speed_prediction/](speed_prediction/README.md) | 普通 GRU、TAQ-GRU、共享数据格式及在线速度推理。 |

本分支只保留完整感知链所需的速度运行代码，没有单独的数据采集、训练和离线评估入口。
模型加载检查及联调命令见 [speed_prediction/README.md](speed_prediction/README.md#测试)。

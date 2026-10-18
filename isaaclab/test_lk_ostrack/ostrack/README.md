# OSTrack 第三方推理源码

`OSTrack-main/` 保留本项目推理实际需要的第三方源码子集：

| 路径 | 作用 |
| --- | --- |
| `OSTrack-main/lib/models/ostrack/` | OSTrack 网络、ViT 主干和候选消除实现。 |
| `OSTrack-main/lib/models/layers/` | 注意力、嵌入和预测头等基础层。 |
| `OSTrack-main/lib/config/` | 模型配置定义。 |
| `OSTrack-main/lib/test/` | 跟踪器、模板/搜索区域处理及推理工具。 |
| `OSTrack-main/lib/train/` | 推理时复用的加载和数据处理辅助文件，不是完整训练工程。 |
| `OSTrack-main/experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml` | 随包使用的模型配置。 |
| `OSTrack-main/LICENSE`、`OSTrack-main/README.md` | 保留的原项目许可证和介绍。 |

原 README 中的训练/benchmark 命令面向完整上游仓库，本分支不包含对应入口。
本项目统一通过 `test_lk_ostrack` 使用该源码，无需运行原版安装或路径生成脚本。

准备[上级说明](../README.md#环境与输入)中的权重，从仓库根目录测试：

```bash
python -m test_lk_ostrack.demo --ostrack_checkpoint checkpoints/OSTrack_ep0300.pth.tar
```

输出目录已存在时用 `--out_dir` 指定新的空目录。源码来源与兼容修改见
[THIRD_PARTY.md](../THIRD_PARTY.md)；本分支无 OSTrack 独立训练命令。

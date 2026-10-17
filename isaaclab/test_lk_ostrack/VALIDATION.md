# 验证记录（2026-10-17）

## 已通过

- 在独立 checkout 中安装 `pip install -e '.[ostrack,dev]'`，原实验目录未加入导入路径。
- 37 项 pytest 测试：跟踪、目标切换、文件 I/O、数据集校验、指标计算、RGB 可见性、
  独立目录运行演示及缺失 OSTrack 依赖时的错误传播。
- CPU 演示：生成 60 帧双目标序列，输出 120 行跟踪结果及叠加图片。
- 所有 Python 文件可解析；wheel 构建通过，机器人层、OSTrack 推理源码、YAML 和许可证包含在安装包中。
- 使用 USD API 逐一检查全部模型层，无缺失或远程资源引用；组合场景可解析本地 Camera。
- OSTrack 运行时源码导入、现有 ViT-B/256 权重的严格参数匹配以及一次 CPU 前向计算通过。
  此检查在用户要求停止加载权重前完成。
- Isaac Sim 5.1.0 / Isaac Lab v2.3.2 能加载场景、机器人 articulation、TiledCamera 和
  三目标 MultiMeshRayCaster；相机和测距初始化通过。

## 未完成的验证

- 按用户要求，未继续进行 CUDA 权重加载或 GPU 精跟踪推理。
- 仿真采集尚未获得完整有效序列。诊断图显示上部目标被炮台遮挡，RGB 可见性检查
  导致重新采样。README 的小规模命令加入 `--target_height_constraint`，将目标限制在
  较低高度；这一参数组合尚未完成仿真复测。
- 因未生成有效仿真序列，采集到离线评估的完整集成链路未在本次整理中通过。
  数据集读写和评估基础函数已有独立回归测试。

安装与 CPU 验证环境：Linux x86_64、Python 3.11、NumPy 1.26.4、OpenCV headless
4.10.0.84、Pillow 11.3.0、PyTorch 2.7.0、torchvision 0.22.0、timm 1.0.27。
这些记录仅表示以上检查范围，不构成真实数据跟踪精度或完整仿真可复现性的结论。

# MuJoCo 代码与模型

本目录集中存放 MuJoCo 相关实现。所有命令均从仓库根目录运行。

```text
mujoco_sim/
├── mujoco_camera.py           # 机身相机，标定、RGB、深度和坐标转换
├── mujoco_lidar.py            # 阵面激光与目标距离估计
├── shooting.py                # 炮台执行、坐标转换、弹道和相机补偿
├── shooting_models.py         # 原射击 Actor 与 OSTrack 的设备适配
├── shooting_config.json       # 射击/速度/视觉权重组合
├── mjcf/lite3_with_turret/    # 模型、网格、控制/传感器配置及详细说明
├── tools/                    # 转换、预览、验证、传感器演示和行走工具
└── tests/                    # MuJoCo 传感器和行走接口检查
```

通用射击链、Isaac 适配器及训练入口仍位于上一级目录。`urdf/` 保留在
上一级目录；这里的 MJCF 直接由原始 USD 构建。现有测试结果和视频保留在
仓库根目录的 `outputs/` 下。

目录名使用 `mujoco_sim`，避免与第三方 Python 包 `mujoco` 同名。

## 工具入口

| 脚本（位于 `tools/`） | 用途 | Python 环境 |
| --- | --- | --- |
| `build_lite3_mjcf.py` | 从原始 USD 构建 MJCF | `isaac_lab_232` |
| `validate_lite3_mjcf.py` | 质量、惯量、关节、碰撞和姿态检查 | `mujoco_test` |
| `preview_lite3_mjcf.py` | 模型预览、碰撞显示、站立 PD 检查 | `mujoco_test` |
| `demo_mujoco_camera.py` | 相机采集和可选激光测距演示 | `mujoco_test` |
| `demo_mujoco_lidar.py` | 阵面激光演示 | `mujoco_test` |
| `play_mujoco_lite3.py` | 加载已有行走策略、录制或回放轨迹 | `isaac_lab_232` |
| `analyze_mujoco_walk.py` | 已保存轨迹的接触和航向分析 | `isaac_lab_232` |
| `check_lidar_source_parity.py` | 对照 Isaac 源函数检查扫描模式与后处理 | `isaac_lab_232` |
| `play_mujoco_shooting.py` | 全链路感知、射击策略执行、评分与视频 | `isaac_lab_232` |
| `analyze_mujoco_shooting.py` | 离线检查跟踪、可见性和开火门控 | `isaac_lab_232` |

例如查看行走程序参数：

```bash
conda activate isaac_lab_232
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/play_mujoco_lite3.py --help
```

脚本默认从本目录读取 MJCF 和配置，并从仓库根目录定位原始 USD、策略权重、
Isaac 源码与输出目录。此前行走报告中记录的旧 `mjcf/` 路径，在轨迹回放和
接触分析时会自动映射到新位置；历史报告不需要重写。

## 详细说明

- [模型来源、转换与验证](mjcf/lite3_with_turret/README.md)
- [相机参数与接口](mjcf/lite3_with_turret/CAMERA.md)
- [激光参数与接口](mjcf/lite3_with_turret/LIDAR.md)
- [行走策略接口与运行命令](mjcf/lite3_with_turret/WALKING.md)
- [全链路射击接口、运行与限制](SHOOTING.md)

独立程序可将 `source/standalone/integrated_shooting_chain/mujoco_sim` 加入
`sys.path`，再导入 `mujoco_camera` 或 `mujoco_lidar`。通过父包导入时使用
`integrated_shooting_chain.mujoco_sim.mujoco_camera` 和
`integrated_shooting_chain.mujoco_sim.mujoco_lidar`。

目前已有模型、相机、激光、行走工具及射击执行/弹道接口。射击首轮测试已运行，
选配相机补偿恢复了本段行走跟踪；微调策略已触发发射，但本轮尚未命中。
具体范围、视频和复现方式见 [SHOOTING.md](SHOOTING.md)。

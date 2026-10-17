# LK + OSTrack 多目标视觉模块

输入第一帧的多个 `xyxy` 目标框，输出每个目标的位置、方向、预测方向、置信度及主目标 ID。
全部目标共享 LK 前后向光流；每个目标分别进行 RANSAC 和 Kalman 估计；仅主目标使用 OSTrack 精跟踪。
支持固定周期和自适应触发，以及 Isaac Lab 中的多目标数据采集、RayCaster 测距和离线评估。

此分支的代码位于 `isaaclab/test_lk_ostrack`，由本地
`source/standalone/test_lk_ostrack` 整理。采集基类、机器人 USD、测距兼容代码及
OSTrack 推理源码已包含，不依赖其他实验目录，无需编写接线脚本。

## 1. 获取代码和安装基础环境

下面命令从仓库根目录执行。CPU 演示只需要 Python 3.10 或 3.11；建议使用 3.11。

```bash
git clone --branch lk-ostrack-standalone --single-branch https://github.com/shang-shangqian/Embodied-Shooting.git
cd Embodied-Shooting
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

基础安装自动安装 NumPy、OpenCV headless 和 Pillow。headless 版也会正常输出叠加图片，
不弹出桌面窗口。不要在同一环境中同时安装多个 OpenCV wheel。

## 2. 直接运行自带 CPU 演示

```bash
python -m test_lk_ostrack.demo
python -m pytest -q
```

演示自动生成 60 帧双目标图片、初始框和相机内参，运行真实的多目标推理入口。
输出位于 `outputs/lk_ostrack_demo/`，其中 `tracks.csv` 是跟踪结果，`overlay/` 是叠加图片。
无需 GPU、模型权重或仿真器。再次运行时用 `--out_dir outputs/demo_2` 指定新目录。
合成演示用于验证安装和处理链路，不代表真实场景精度。

## 3. 启用 OSTrack 精跟踪

需要 NVIDIA GPU 和匹配的 CUDA PyTorch。验证环境使用 PyTorch 2.7.0、torchvision 0.22.0、
CUDA 12.8；先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v270)
安装匹配的一对 torch/torchvision，然后执行：

```bash
python -m pip install -e '.[ostrack,dev]'
mkdir -p checkpoints
```

从 [OSTrack 官方模型目录](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing)
下载与 **`vitb_256_mae_ce_32x4_ep300`** 对应的 `OSTrack_ep0300.pth.tar`，
放在 `checkpoints/OSTrack_ep0300.pth.tar`。不同模型可能同名，请按配置选择，
不要使用 384、ViT-L 或 GOT10K 的其他 checkpoint。
推理不需要额外的 MAE 预训练权重，也不需要运行官方路径生成脚本或旧版 `install.sh`。

```bash
python -m test_lk_ostrack.demo \
  --out_dir outputs/ostrack_demo \
  --ostrack_checkpoint checkpoints/OSTrack_ep0300.pth.tar \
  --device cuda:0
```

默认自动定位随包提供的 `ostrack/OSTrack-main`。若要使用外部兼容源码，
可在正式推理入口传入 `--ostrack_repo /path/to/OSTrack`。
官方 tracker 的精跟踪实现使用 CUDA；CPU 模式通过 `--disable_fine_tracking` 运行 LK 链路。
第三方来源、许可证和兼容修改见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 4. 自己的图片序列

将按时间排序、名称补零的 RGB 图片放进一个目录，例如 `000000.png`、`000001.png`。
初始框使用第一帧的像素坐标，JSON 示例：

```json
{"targets": [{"id": 0, "bbox": [100, 80, 150, 130]}, {"id": 1, "bbox": [420, 90, 475, 145]}]}
```

相机内参 JSON 示例（实际运行请填写真实标定值）：

```json
{"fx": 600, "fy": 600, "cx": 320, "cy": 180, "width": 640, "height": 360}
```

```bash
python -m test_lk_ostrack.estimate_video \
  --rgb_dir /path/to/rgb \
  --initial_boxes_file /path/to/initial_targets.json \
  --intrinsics_json /path/to/intrinsics.json \
  --fps 30 \
  --ostrack_checkpoint checkpoints/OSTrack_ep0300.pth.tar \
  --out_csv outputs/my_tracks.csv \
  --overlay_dir outputs/my_overlay
```

仅用 LK 时将 checkpoint 参数替换为 `--disable_fine_tracking`。
自适应策略增加 `--fine_tracker_policy adaptive`；参数及验收方案见
[ADAPTIVE_OSTRACK_DESIGN.md](ADAPTIVE_OSTRACK_DESIGN.md)。
单目标兼容入口使用 `--initial_box x1 y1 x2 y2`，替代 `--initial_boxes_file`。

## 5. Isaac Lab 多目标采集

仿真额外需要 **Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + Python 3.11** 和支持 RTX 的 NVIDIA GPU。
按 [Isaac Lab v2.3.2 官方安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html)
配置仿真环境。Isaac Lab 自身的安装和 `_isaac_sim` 链接属于环境配置，
本仓库不重复提交整个仿真 SDK。

在已配置的 Isaac Lab Conda 环境里安装本仓库，而不是前面的普通 CPU venv：

```bash
conda activate isaac_lab_232
python -m pip install -e '.[ostrack,dev]'
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
```

`ISAACLAB_PATH` 填写自己的 Isaac Lab 根目录。先做小规模采集（未传 checkpoint 时自动使用 LK）：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p \
  isaaclab/test_lk_ostrack/collect_multi_target_dataset.py \
  --headless --device cuda:0 \
  --num_envs 1 --num_targets 3 \
  --target_height_constraint \
  --num_sequences 1 --sequence_length 30 \
  --out_dir outputs/isaacsim_multi_smoke

python -m test_lk_ostrack.check_multi_target_dataset outputs/isaacsim_multi_smoke
```

启用精跟踪时增加 `--ostrack_checkpoint /absolute/path/to/OSTrack_ep0300.pth.tar`。
机器人和炮台 USD 已包含，全部资源引用在本地解析。相机使用本地 pinhole，
保留挂载变换，默认分辨率 1024×576、焦距 10、孔径 5.76×3.24。
测距传感器的 mesh ID 维度兼容处理随包提供，不需要修改 Isaac Lab 文件。

默认不启用行走策略；若需要移动平台采集，下载兼容 Lite3 观测/关节顺序的
RSL-RL TorchScript `policy.pt` 并传入 `--locomotion_policy_path /path/to/policy.pt`。
该行走权重不随包提供，也不是默认采集的前置条件。
默认无行走策略时建议保留上述 `--target_height_constraint`，将目标限制在
0.45–3 m 高度范围；不限制高度时，上部目标可能被炮台遮挡，初始化会反复重采样。
第一次可见性检查失败时会保存 `initialization_preview.png` 和对应 JSON，
用于检查渲染画面和初始化框；采集异常写入输出目录的 `failure.log`。
`collect_velocity_text_dataset.py` 是只记录主目标速度文本的现成入口；
所有入口均支持 `--help` 查看参数。

## 6. 数据集评估

```bash
python -m test_lk_ostrack.evaluate_multi_target_dataset \
  --dataset_dir outputs/isaacsim_multi_smoke \
  --disable_fine_tracking \
  --out_dir outputs/isaacsim_multi_smoke_eval
```

启用 OSTrack 时将 `--disable_fine_tracking` 替换为 checkpoint 参数。
输出包括 IoU、方向误差、视觉方向与记录测距融合的三维误差、丢失率及耗时。
默认从 ground truth 输入距离/速度用于主目标优先级；验证纯视觉回退时增加
`--kinematics_source vision`。离线评估使用采集时记录的测距值，不会重新模拟实时雷达。
更多指标解释见 [TRACKING_SPEED_EVALUATION.md](TRACKING_SPEED_EVALUATION.md)。

## 7. 目录和输出

| 文件 | 用途 |
| --- | --- |
| `multi_target_tracker.py`、`camera_geometry.py` | LK、RANSAC、Kalman、方向和优先级 |
| `ostrack_tracker.py`、`batched_ostrack.py`、`online_vision.py` | 单帧、批量和在线 OSTrack 适配 |
| `estimate_video.py`、`demo.py` | 序列推理和自带演示 |
| `collect_multi_target_dataset.py`、`collect_velocity_text_dataset.py` | 仿真采集入口 |
| `multi_target_collect_env.py`、`turret_aim_env*.py`、`ray_caster_compat.py` | 完整采集环境及 SDK 兼容处理 |
| `isaacsim_dataset.py`、`multi_target_io.py`、`pipeline.py` | 数据读写、校验、结果绘制 |
| `evaluate_*_dataset.py` | 多目标和单目标评估 |
| `lite3/`、`turret/`、`*.usd` | 本地模型资产 |
| `ostrack/OSTrack-main/` | OSTrack 推理所需源码、默认 YAML 和原许可证 |
| `tests/` | 不依赖仿真和权重的回归测试 |

未包含历史数据、输出、缓存、源码 ZIP、论文材料及模型权重。
`outputs/`、`checkpoints/`、`*.pt`、`*.pth.tar` 已加入 Git 忽略规则。

本次整理的实际验证范围见 [VALIDATION.md](VALIDATION.md)。

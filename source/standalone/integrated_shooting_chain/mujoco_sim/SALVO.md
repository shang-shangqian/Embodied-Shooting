# MuJoCo：SS 百目标清场

入口 `tools/eval_mujoco_salvo.py` 默认在同一场景创建 100 个静态球形目标，
由站立机器人逐个射击。与 Isaac 的 `--case ss --static_salvo` 对照，
不是对一个目标重复 100 回合。无需启动 Isaac Sim。

## 运行

在当前项目根目录运行，输出目录必须尚不存在：

```bash
cd /home/jby/IsaacLab-2.3.2
conda activate isaac_lab_232

# 先构建 100 个目标、检查视野、保存布局；不加载模型、不渲染、不射击。
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/eval_mujoco_salvo.py \
  --dry-run --output outputs/mujoco_ss100_preview

# CPU 推理和软件渲染，避免使用其他账户正在训练的 GPU。
export CUDA_VISIBLE_DEVICES=''
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=4
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
python source/standalone/integrated_shooting_chain/mujoco_sim/tools/eval_mujoco_salvo.py \
  --salvo-targets 100 --salvo-timeout-s 600 --device cpu --threads 4 \
  --output outputs/mujoco_ss100
```

加 `--video` 保存视频；加 `--stabilize-camera` 试验已有的相机旋转补偿。
默认权重来自 `shooting_config.json`，包括原射击 Actor、冻结速度 GRU、OSTrack；
站立控制沿用原行走 TorchScript 策略。`--decision-checkpoint` 同时读取配套
`training_settings.json` 中的速度模型、开火阈值和确认步数。
百目标模式将每回合总发射预算设为 0（不限制），防止单目标的五发预算提前结束清场。
默认 `model_root` 为 `/mnt/A/hust_jby/IsaacLab-2.3.2`：决策、速度、OSTrack 权重、
OSTrack 实现目录和运动策略统一从数据盘读取。模型相对路径从该根目录解析；
指向当前系统盘项目的绝对模型路径也会映射到数据盘，包括训练设置中的旧速度路径。
其他绝对模型路径保留原值；缺失文件直接报错，不回退读取系统盘副本。
可通过 `--model-root PATH` 覆盖模型根目录，`--locomotion-policy PATH` 覆盖运动权重。
配置文件、导入布局和输出路径仍从当前项目根目录解析。
本项目沿用现有 `camera_config.json` 相机标定；实际内参和视场记录在报告中，
与另一份项目或 Isaac 做对照时应同时核对相机参数。

## 与 Isaac 共用的规则

- SS：机器人站立，目标生成后世界坐标固定，半径 0.35 m。
- 默认种子 61；相机光学球面扇区距离 80–150 m、yaw ±0.26 rad、
  pitch ±0.12 rad，拒绝与地面相交的目标。相同分布不保证两平台的随机布局相同。
- 预热结束后，一次性为所有目标提供初始真值框。此后 LK 连续跟踪全部目标，
  OSTrack 精修当前目标；不再读取真值框纠正、重识别或重建跟踪器。
- 复用 `salvo_engagement.py`：每轮按 RGB 估计距离选择最近目标；连续 5 发未命中
  或 3 秒无命中则切换，失败目标推迟到下一轮，至少等待 1 秒，每个目标最多尝试 3 次。
  没有开火的时间也计入 3 秒上限。等待期内不会误判为尝试耗尽。
- 所有未摧毁目标继续显示；阵面 LiDAR 只检测选中的目标，与 Isaac 的选中目标射线网格协议一致。
- 采用 Isaac 消融组 D：预测未就绪时保持炮台、不推理 Actor、不射击；切换不同 ID
  只清空 Actor GRU，保留 Critic 与总调用计数。首次目标保留预热状态。
  速度历史在 ID 改变时重置，其他目标的视觉历史保留；发射冷却跨目标保留。
- 只有实际发射才评分。复用同步预测弹道，命中后立即隐藏并移除该目标，切换到下一个。
  仿真不在每次命中后重置。仍未摧毁或丢失的目标计入最初的 100 个目标分母。
- 600 秒总时限包含默认 2 秒站立预热；终止原因为清场、跟踪丢失、尝试耗尽、
  超时、机器人跌倒或仿真异常。

## 布局与输出

`layout.json` 使用 Isaac 的 schema 1、float32 坐标及 SHA256 校验，
`positions_env_local_m` 在 MuJoCo 单环境中就是世界坐标。可用
`--layout-in PATH`（或 `--salvo_layout_in PATH`）重放，包括 Isaac 导出的布局。
导入后会检查数量、半径、高度、校验值以及初始相机视野；不同平台相机姿态可能
使某个布局不可见，此时明确报错。预检查布局基于 stand keyframe，正式随机布局
基于站立预热后的实际相机姿态，因此种子相同也可能不同；需要逐目标对照时导入同一布局。

- `salvo_report.json`：结束原因、摧毁/剩余/发射/命中率、逐发评分、调度与重试事件、
  Actor 等待/重置、初始框使用次数、权重路径和哈希。
- `steps.jsonl`：逐控制步的目标 ID、就绪状态、估计、动作、开火与机器人状态。
- `layout.json`：初始静态布局，可重放。
- `salvo.mp4`：启用 `--video` 时保存，相机图像与目标框。

`--dry-run` 的报告明确为 `mode=dry_run, completed=false`，不表示完成射击。
`completed` 表示评测正常结束，是否全部摧毁请看 `cleared` 和 `destroyed`。
命中判定是原环境的同步预测弹道，不是 MuJoCo 弹丸接触。
规则与接口对齐不代表策略已在 MuJoCo 达到 Isaac 的命中性能；需要正式运行报告评估。

## 检查

```bash
python -m pytest -q source/standalone/integrated_shooting_chain/mujoco_sim/tests/test_mujoco_salvo.py
```

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ============================================================
# Utility
# ============================================================

def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[rl_perception_bridge] CUDA not available, fallback to CPU.")
        return torch.device("cpu")
    return dev


def _nan_to_num_(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算两个角度的最小差值，输出范围 [-pi, pi]
    """
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


# ============================================================
# Load vision model
# Compatible with your vision_model1.py / collect_vision_data.py
# ============================================================

def _build_vision_model_fallback(device: torch.device):
    """
    直接从当前工程里的 vision_model1.py 构建模型。
    """
    import sys
    from pathlib import Path

    this_file = Path(__file__).resolve()
    standalone_dir = this_file.parent.parent   # .../source/standalone

    if str(standalone_dir) not in sys.path:
        sys.path.insert(0, str(standalone_dir))
    try:
        from test4.vision_model.vision_cnn.vision_model1 import build_vision_model
    except ModuleNotFoundError:
        from test4.vision_model.vision_model1 import build_vision_model

    model = build_vision_model(
        use_proprio=True,
        proprio_dim=1,
    )
    return model.to(device)


def load_vision_model(model_path: str | Path, device: str | torch.device):
    """
    加载当前版本视觉模型。

    约定：
    - 模型结构固定为 vision_model1.py 里的 build_vision_model(use_proprio=True, proprio_dim=1)
    - 输入固定包含 dog_moving_flag
    - checkpoint 只支持：
        1) torch.save(model.state_dict(), path)
        2) torch.save({"model_state_dict": ...}, path)
        3) torch.save({"state_dict": ...}, path)
    """
    device = _resolve_device(device)
    model_path = Path(model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Vision model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    model = _build_vision_model_fallback(device=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Unsupported checkpoint type: {type(ckpt)}. "
            "Expected a state_dict-like dict checkpoint."
        )

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    print(f"[rl_perception_bridge][vision] loaded checkpoint from: {model_path}")
    return model


def run_vision_model(
    vision_model,
    images: torch.Tensor,
    dog_moving_flag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
        images: [N, 3, H, W]
        dog_moving_flag: [N, 1]
    输出:
        pred_yaw_cam:   [N]
        pred_pitch_cam: [N]

    这里直接复用你采集脚本里的简化调用思路：
    encoder(images) + direction_head(fused_features)
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise RuntimeError(f"Unexpected image shape for vision model: {tuple(images.shape)}")

    if dog_moving_flag.ndim != 2 or dog_moving_flag.shape[0] != images.shape[0] or dog_moving_flag.shape[1] != 1:
        raise RuntimeError(
            "dog_moving_flag must have shape [N, 1], "
            f"got {tuple(dog_moving_flag.shape)}"
        )

    model_device = next(vision_model.parameters()).device
    images = images.to(device=model_device, dtype=torch.float32)
    dog_moving_flag = dog_moving_flag.to(device=model_device, dtype=torch.float32)

    with torch.no_grad():
        out = vision_model(images, proprio=dog_moving_flag)
        if not hasattr(out, "direction"):
            raise RuntimeError("vision_model(images, proprio=...) returned no '.direction' field")
        direction = out.direction

    if direction.ndim != 2 or direction.shape[1] < 2:
        raise RuntimeError(f"Vision direction shape must be [N, 2], got {tuple(direction.shape)}")

    pred_yaw = _nan_to_num_(direction[:, 0])
    pred_pitch = _nan_to_num_(direction[:, 1])

    return pred_yaw, pred_pitch


# ============================================================
# Load velocity model
# Compatible with speed_prediction_model.py / train_gru_velocity.py
# ============================================================

def load_velocity_model(
    model_path: str | Path,
    device: str | torch.device,
    model_def_path: str | Path | None = None,
):
    import importlib.util
    from pathlib import Path
    device = _resolve_device(device)
    model_path = Path(model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Velocity model not found: {model_path}")

    this_file = Path(__file__).resolve()

    candidate_model_def_paths: list[Path] = []
    if model_def_path is not None and str(model_def_path).strip() != "":
        candidate_model_def_paths.append(Path(model_def_path).expanduser())

    candidate_model_def_paths.extend(
        [
            model_path.parent / "speed_prediction_model.py",
            this_file.parent.parent / "test_11" / "speed_prediction_model.py",
            this_file.parent.parent / "test_temp" / "speed_prediction_model.py",
        ]
    )

    velocity_model_def_path = next((p for p in candidate_model_def_paths if p.exists()), None)
    if velocity_model_def_path is None:
        searched = "\n  - ".join(str(p) for p in candidate_model_def_paths)
        raise FileNotFoundError(
            "Could not find velocity model definition speed_prediction_model.py. "
            "Pass --velocity_model_def explicitly. Searched:\n"
            f"  - {searched}"
        )

    spec = importlib.util.spec_from_file_location(
        "velocity_speed_prediction_model",
        velocity_model_def_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {velocity_model_def_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    VelocityPredictorGRU = module.VelocityPredictorGRU

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        model.eval()
        print(f"[rl_perception_bridge][velocity] loaded full nn.Module from: {model_path}")
        return model

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported velocity checkpoint type: {type(ckpt)}")

    cfg = ckpt.get("config", {})
    model = VelocityPredictorGRU(
        input_dim=int(cfg.get("input_dim", 13)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)

    state_dict = ckpt.get("model_state_dict", None)
    if state_dict is None:
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k[7:] if k.startswith("module.") else k] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"[rl_perception_bridge][velocity] loaded checkpoint from: {model_path}")
    print(f"[rl_perception_bridge][velocity] loaded model definition from: {velocity_model_def_path}")
    print(f"[rl_perception_bridge][velocity] missing keys: {missing}")
    print(f"[rl_perception_bridge][velocity] unexpected keys: {unexpected}")

    model.eval()
    return model


# ============================================================
# Distance
# ============================================================

def estimate_distance_from_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    prefer_key_order: tuple[str, ...] = ("lidar_distance", "pred_distance", "distance"),
) -> torch.Tensor:
    """
    按优先级从 batch 中读取距离。
    默认优先使用 MultiMeshRayCaster 距离；最后允许回退到 GT distance（distance），方便联调。
    """
    for key in prefer_key_order:
        if key in batch:
            pred_distance = batch[key]
            if pred_distance.ndim != 1:
                pred_distance = pred_distance.squeeze(-1)
            pred_distance = pred_distance.to(device=device, dtype=torch.float32)
            pred_distance = _nan_to_num_(pred_distance)
            return pred_distance

    raise KeyError(
        "No distance source found in batch. "
        f"Expected one of: {prefer_key_order}"
    )


# ============================================================
# Geometry helpers
# ============================================================

def _camera_angles_and_distance_to_camera_frame_position(
    yaw_cam: torch.Tensor,
    pitch_cam: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    """
    将相机坐标系下的 yaw/pitch/d 重建为 3D 向量。

    与环境里的 camera label 定义保持一致：
        yaw_cam   = atan2(x, z)
        pitch_cam = atan2(-y, sqrt(x^2 + z^2))
    反解可得：
        z = d * cos(pitch) * cos(yaw)
        x = d * cos(pitch) * sin(yaw)
        y = -d * sin(pitch)
    """
    if distance.ndim != 1:
        distance = distance.squeeze(-1)

    cp = torch.cos(pitch_cam)
    sp = torch.sin(pitch_cam)
    cy = torch.cos(yaw_cam)
    sy = torch.sin(yaw_cam)

    z = distance * cp * cy
    x = distance * cp * sy
    y = -distance * sp

    pos_cam = torch.stack([x, y, z], dim=-1)
    return _nan_to_num_(pos_cam)


def _quat_conjugate(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_conj = quat_wxyz.clone()
    quat_conj[..., 1:] *= -1.0
    return quat_conj


def _rotate_local_to_world(quat_wxyz: torch.Tensor, vec_local: torch.Tensor) -> torch.Tensor:
    """
    这里直接复用 IsaacLab 的 quat_apply 语义：
    quat_apply(q, v_local) => v_world
    """
    from isaaclab.utils.math import quat_apply
    return quat_apply(quat_wxyz, vec_local)


def _camera_frame_position_to_base_frame_position(
    base_env: Any,
    pos_cam: torch.Tensor,
) -> torch.Tensor:
    """
    将相机坐标系下的位置向量 pos_cam 转换为 yaw-only base 坐标系下的位置。

    关键修复：
    1) 优先使用 base_env._get_runtime_camera_pose_w()，不要再直接用
       base_env._vision_camera.data.pos_w / quat_w_ros。
    2) 强制把 env 里的 CPU/GPU tensor 对齐到 pos_cam.device。
       这样 CPU PhysX eval 和 GPU eval 都能跑。
    """
    if base_env._vision_camera is None:
        raise RuntimeError("base_env._vision_camera is None")

    device = pos_cam.device
    dtype = pos_cam.dtype

    # ------------------------------------------------------------
    # 1) 相机位姿：优先使用环境里 runtime 重建的相机位姿
    # ------------------------------------------------------------
    if hasattr(base_env, "_get_runtime_camera_pose_w"):
        cam_pos_w, cam_quat_w = base_env._get_runtime_camera_pose_w()
    else:
        cam_pos_w = base_env._vision_camera.data.pos_w
        cam_quat_w = base_env._vision_camera.data.quat_w_ros

    cam_pos_w = cam_pos_w.to(device=device, dtype=dtype)
    cam_quat_w = cam_quat_w.to(device=device, dtype=dtype)

    # ------------------------------------------------------------
    # 2) 机器狗 root 位姿，同步到 pos_cam 的 device
    # ------------------------------------------------------------
    root_pos_w = base_env.turret.data.root_pos_w.to(device=device, dtype=dtype)
    root_quat_w = base_env.turret.data.root_quat_w.to(device=device, dtype=dtype)

    # ------------------------------------------------------------
    # 3) 相机局部坐标 -> 世界坐标
    # ------------------------------------------------------------
    pos_offset_w = _rotate_local_to_world(cam_quat_w, pos_cam)
    target_pos_w = cam_pos_w + pos_offset_w

    # ------------------------------------------------------------
    # 4) 世界坐标 -> yaw-only base 坐标
    # ------------------------------------------------------------
    to_target_root_w = target_pos_w - root_pos_w

    if hasattr(base_env, "_get_yaw_only_quat"):
        # base_env 的函数可能内部创建 CPU tensor，所以返回后也要搬回 device
        root_quat_yaw = base_env._get_yaw_only_quat(root_quat_w)
        root_quat_yaw = root_quat_yaw.to(device=device, dtype=dtype)
    else:
        basis_x = torch.zeros((root_quat_w.shape[0], 3), device=device, dtype=dtype)
        basis_x[:, 0] = 1.0

        forward_w = _rotate_local_to_world(root_quat_w, basis_x)
        forward_xy = forward_w[:, :2]
        forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp_min(1e-8)

        yaw = torch.atan2(forward_xy[:, 1], forward_xy[:, 0])
        half_yaw = 0.5 * yaw

        root_quat_yaw = torch.zeros_like(root_quat_w)
        root_quat_yaw[:, 0] = torch.cos(half_yaw)
        root_quat_yaw[:, 1] = 0.0
        root_quat_yaw[:, 2] = 0.0
        root_quat_yaw[:, 3] = torch.sin(half_yaw)

    if hasattr(base_env, "_rotate_world_to_local"):
        # base_env 的方法本身用 torch 运算，只要输入 device 一致即可
        pos_base = base_env._rotate_world_to_local(root_quat_yaw, to_target_root_w)
    else:
        quat_conj = _quat_conjugate(root_quat_yaw)
        pos_base = _rotate_local_to_world(quat_conj, to_target_root_w)

    return _nan_to_num_(pos_base)


def _rotate_world_to_yaw_only_base(
    base_env: Any,
    root_quat_w: torch.Tensor,
    vec_w: torch.Tensor,
) -> torch.Tensor:
    device = vec_w.device
    dtype = vec_w.dtype
    root_quat_w = root_quat_w.to(device=device, dtype=dtype)

    if hasattr(base_env, "_rotate_world_to_yaw_only_local"):
        return _nan_to_num_(base_env._rotate_world_to_yaw_only_local(root_quat_w, vec_w))

    basis_x = torch.zeros((root_quat_w.shape[0], 3), device=device, dtype=dtype)
    basis_x[:, 0] = 1.0
    forward_w = _rotate_local_to_world(root_quat_w, basis_x)
    forward_xy = forward_w[:, :2]
    forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp_min(1e-8)
    yaw = torch.atan2(forward_xy[:, 1], forward_xy[:, 0])
    half_yaw = 0.5 * yaw

    root_quat_yaw = torch.zeros_like(root_quat_w)
    root_quat_yaw[:, 0] = torch.cos(half_yaw)
    root_quat_yaw[:, 3] = torch.sin(half_yaw)

    quat_conj = _quat_conjugate(root_quat_yaw)
    return _nan_to_num_(_rotate_local_to_world(quat_conj, vec_w))


def _vector_to_base_yaw_pitch(vec_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = vec_base[:, 0]
    y = vec_base[:, 1]
    z = vec_base[:, 2]
    yaw_base = torch.atan2(y, x)
    pitch_base = torch.atan2(-z, torch.sqrt(x * x + y * y + 1e-8))
    return _nan_to_num_(yaw_base), _nan_to_num_(pitch_base)


# ============================================================
# Per-env temporal buffer for GRU velocity prediction
# ============================================================

@dataclass
class _SequenceState:
    pos_base_hist: list[torch.Tensor]
    robot_lin_vel_hist: list[torch.Tensor]
    robot_ang_vel_hist: list[torch.Tensor]
    dog_flag_hist: list[torch.Tensor]


# ============================================================
# Per-env temporal buffer for GRU velocity prediction
# GPU tensor ring buffer version
# ============================================================

class VelocitySequenceBuffer:
    """
    为每个 env 维护最近 9 帧缓存（纯 GPU tensor 环形缓冲区）。

    组装规则与 train_gru_velocity.py / test_gru_velocity.py 一致：
      x: [M, 8, 13]
      [
        pos_abs(3),
        pos_diff(3),
        robot_lin_vel(3),
        robot_ang_vel(3),
        dog_moving_flag(1),
      ]

    相比原版：
    - 不再使用 Python list.append()/pop(0)
    - 不再按 env 做 Python for-loop
    - update / reset / build 全部走 tensor
    """

    def __init__(self, num_envs: int, device: torch.device, hist_len: int = 9):
        self.num_envs = int(num_envs)
        self.device = device
        self.hist_len = int(hist_len)

        if self.hist_len < 2:
            raise ValueError(f"hist_len must be >= 2, got {self.hist_len}")

        # 环形写指针：表示“下一次写入”的槽位
        self.write_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # 当前每个 env 已累计的有效帧数，最大 hist_len
        self.counts = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # 历史缓存
        self.pos_base_hist = torch.zeros(
            (self.num_envs, self.hist_len, 3), dtype=torch.float32, device=self.device
        )
        self.robot_lin_vel_hist = torch.zeros(
            (self.num_envs, self.hist_len, 3), dtype=torch.float32, device=self.device
        )
        self.robot_ang_vel_hist = torch.zeros(
            (self.num_envs, self.hist_len, 3), dtype=torch.float32, device=self.device
        )
        self.dog_flag_hist = torch.zeros(
            (self.num_envs, self.hist_len, 1), dtype=torch.float32, device=self.device
        )

    def reset_envs(self, env_ids: torch.Tensor | list[int] | None = None):
        if env_ids is None:
            self.write_idx.zero_()
            self.counts.zero_()
            self.pos_base_hist.zero_()
            self.robot_lin_vel_hist.zero_()
            self.robot_ang_vel_hist.zero_()
            self.dog_flag_hist.zero_()
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return

        self.write_idx[env_ids] = 0
        self.counts[env_ids] = 0
        self.pos_base_hist[env_ids] = 0.0
        self.robot_lin_vel_hist[env_ids] = 0.0
        self.robot_ang_vel_hist[env_ids] = 0.0
        self.dog_flag_hist[env_ids] = 0.0

    @torch.no_grad()
    def update(
        self,
        pos_base: torch.Tensor,        # [N, 3]
        robot_lin_vel: torch.Tensor,   # [N, 3]
        robot_ang_vel: torch.Tensor,   # [N, 3]
        dog_flag: torch.Tensor,        # [N, 1]
        valid_mask: torch.Tensor | None = None,   # [N]
        reset_invalid: bool = False,
    ):
        pos_base = torch.nan_to_num(
            pos_base.to(self.device, dtype=torch.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        robot_lin_vel = torch.nan_to_num(
            robot_lin_vel.to(self.device, dtype=torch.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        robot_ang_vel = torch.nan_to_num(
            robot_ang_vel.to(self.device, dtype=torch.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        dog_flag = torch.nan_to_num(
            dog_flag.to(self.device, dtype=torch.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        if pos_base.shape != (self.num_envs, 3):
            raise RuntimeError(f"pos_base must be [{self.num_envs}, 3], got {tuple(pos_base.shape)}")
        if robot_lin_vel.shape != (self.num_envs, 3):
            raise RuntimeError(f"robot_lin_vel must be [{self.num_envs}, 3], got {tuple(robot_lin_vel.shape)}")
        if robot_ang_vel.shape != (self.num_envs, 3):
            raise RuntimeError(f"robot_ang_vel must be [{self.num_envs}, 3], got {tuple(robot_ang_vel.shape)}")
        if dog_flag.shape != (self.num_envs, 1):
            raise RuntimeError(f"dog_flag must be [{self.num_envs}, 1], got {tuple(dog_flag.shape)}")

        if valid_mask is None:
            valid_mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            valid_mask = valid_mask.to(device=self.device, dtype=torch.bool)
            if valid_mask.shape != (self.num_envs,):
                raise RuntimeError(
                    f"valid_mask must be [{self.num_envs}], got {tuple(valid_mask.shape)}"
                )

        if reset_invalid:
            invalid_env_ids = torch.nonzero(~valid_mask, as_tuple=False).squeeze(-1)
            if invalid_env_ids.numel() > 0:
                self.reset_envs(invalid_env_ids)

        valid_env_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if valid_env_ids.numel() == 0:
            return

        slot = self.write_idx[valid_env_ids]

        self.pos_base_hist[valid_env_ids, slot] = pos_base[valid_env_ids]
        self.robot_lin_vel_hist[valid_env_ids, slot] = robot_lin_vel[valid_env_ids]
        self.robot_ang_vel_hist[valid_env_ids, slot] = robot_ang_vel[valid_env_ids]
        self.dog_flag_hist[valid_env_ids, slot] = dog_flag[valid_env_ids]

        self.write_idx[valid_env_ids] = (self.write_idx[valid_env_ids] + 1) % self.hist_len
        self.counts[valid_env_ids] = torch.clamp(self.counts[valid_env_ids] + 1, max=self.hist_len)

    @torch.no_grad()
    def build_model_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          x: [M, 8, 13]
          valid_env_ids: [M]

        只有缓存满 9 帧的 env 才参与 GRU 推理。
        """
        ready_mask = self.counts >= self.hist_len
        valid_env_ids = torch.nonzero(ready_mask, as_tuple=False).squeeze(-1)

        if valid_env_ids.numel() == 0:
            return (
                torch.zeros((0, self.hist_len - 1, 13), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.long, device=self.device),
            )

        # 对于环形缓冲区，write_idx 指向“下一次写入的位置”，
        # 因此它也是“当前最旧帧”的位置。
        # 需要按时间顺序重排为：
        #   oldest, ..., newest
        #
        # 构造每个 ready env 的时间索引：
        #   idx = write_idx + [0,1,...,hist_len-1] mod hist_len
        base = torch.arange(self.hist_len, device=self.device).view(1, self.hist_len)  # [1, 9]
        start = self.write_idx[valid_env_ids].view(-1, 1)                               # [M, 1]
        time_idx = (start + base) % self.hist_len                                       # [M, 9]

        # 高级索引取出按时间排序后的完整历史
        pos = self.pos_base_hist[valid_env_ids.unsqueeze(1), time_idx]                  # [M, 9, 3]
        robot_lin = self.robot_lin_vel_hist[valid_env_ids.unsqueeze(1), time_idx]       # [M, 9, 3]
        robot_ang = self.robot_ang_vel_hist[valid_env_ids.unsqueeze(1), time_idx]       # [M, 9, 3]
        dog_flag = self.dog_flag_hist[valid_env_ids.unsqueeze(1), time_idx]             # [M, 9, 1]

        pos_abs = pos[:, 1:, :]                  # [M, 8, 3]
        pos_diff = pos[:, 1:, :] - pos[:, :-1, :]  # [M, 8, 3]

        x = torch.cat(
            [
                pos_abs,
                pos_diff,
                robot_lin[:, 1:, :],
                robot_ang[:, 1:, :],
                dog_flag[:, 1:, :],
            ],
            dim=-1,
        )  # [M, 8, 13]

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.to(dtype=torch.float32), valid_env_ids


# ============================================================
# Main bridge
# ============================================================

class RLPerceptionBridge:
    """
    在线感知桥接器：
    - 读环境视觉 batch
    - 跑视觉方向
    - 读/估计距离
    - 重建 pred_pos_base
    - 维护 9 帧缓存
    - 跑 GRU 速度预测
    - 写回 env.set_rl_predicted_target_state(...)
    """

    def __init__(
        self,
        base_env: Any,
        vision_model_path: str | Path,
        velocity_model_path: str | Path | None = None,
        velocity_model_def_path: str | Path | None = None,
        device: str | torch.device | None = None,
        use_gt_distance_fallback: bool = True,
        use_gt_velocity_until_buffer_ready: bool = True,
        velocity_sequence_stride: int = 3,
        debug: bool = False,
    ):
        self.base_env = base_env
        self.device = _resolve_device(device if device is not None else base_env.device)
        self.debug = bool(debug)

        self.vision_model = load_vision_model(vision_model_path, self.device)

        self.velocity_model = None
        if velocity_model_path is not None and str(velocity_model_path).strip() != "":
            self.velocity_model = load_velocity_model(
                velocity_model_path,
                self.device,
                model_def_path=velocity_model_def_path,
            )

        self.use_gt_distance_fallback = bool(use_gt_distance_fallback)
        self.use_gt_velocity_until_buffer_ready = bool(use_gt_velocity_until_buffer_ready)
        self.velocity_sequence_stride = max(1, int(velocity_sequence_stride))
        self._velocity_bridge_step = 0

        self.seq_buffer = VelocitySequenceBuffer(
            num_envs=int(base_env.num_envs),
            device=self.device,
        )

    def reset_envs(self, env_ids: torch.Tensor | list[int] | None = None):
        self.seq_buffer.reset_envs(env_ids)

    @torch.no_grad()
    def step(self) -> dict[str, torch.Tensor]:
        """
        在 env.step(action) 之后调用一次。
        返回一个 debug dict。
        """
        # --------------------------------------------------------
        # 0. 读取当前 step 已准备好的视觉缓存（只刷新一次）
        # --------------------------------------------------------
        if hasattr(self.base_env, "_ensure_vision_supervision_ready"):
            self.base_env._ensure_vision_supervision_ready()
        else:
            self.base_env.refresh_vision_supervision()

        batch = self.base_env.get_vision_supervision_batch()

        images_hist = batch["images"]              # [N, T, 3, H, W]
        valid_mask = batch["valid_mask"]           # [N]
        dog_moving_flag = batch["dog_moving_flag"] # [N, 1]

        if images_hist.ndim != 5:
            raise RuntimeError(f"Expected images shape [N, T, 3, H, W], got {tuple(images_hist.shape)}")

        images = images_hist[:, -1, ...].to(self.device, dtype=torch.float32)
        dog_moving_flag = dog_moving_flag.to(self.device, dtype=torch.float32)
        valid_mask = valid_mask.to(self.device, dtype=torch.bool)

        # --------------------------------------------------------
        # 1. vision -> yaw/pitch in camera frame
        # --------------------------------------------------------
        pred_yaw_cam, pred_pitch_cam = run_vision_model(
            self.vision_model,
            images=images,
            dog_moving_flag=dog_moving_flag,
        )
        pred_yaw_cam = _nan_to_num_(pred_yaw_cam)
        pred_pitch_cam = _nan_to_num_(pred_pitch_cam)

        # --------------------------------------------------------
        # 2. 用“当前帧视觉预测”驱动当前帧 MultiMeshRayCaster 重筛
        # --------------------------------------------------------
        if (
            hasattr(self.base_env, "set_dynamic_target_raycast_angles")
            and hasattr(self.base_env, "run_dynamic_target_raycast_for_current_frame")
        ):
            self.base_env.set_dynamic_target_raycast_angles(
                yaw=pred_yaw_cam,
                pitch=pred_pitch_cam,
            )
            self.base_env.run_dynamic_target_raycast_for_current_frame()

            # 重新读取 batch，此时 lidar_distance / refined_yaw / refined_pitch
            # 已经对应当前帧视觉预测重筛后的结果
            batch = self.base_env.get_vision_supervision_batch()

        # --------------------------------------------------------
        # 3. 融合 refined yaw/pitch（有 raycast 命中时优先）
        # --------------------------------------------------------
        fused_yaw_cam = pred_yaw_cam
        fused_pitch_cam = pred_pitch_cam
        lidar_valid_mask = batch.get("lidar_valid_mask", None)
        if lidar_valid_mask is None:
            lidar_valid_mask = torch.ones_like(valid_mask, dtype=torch.bool, device=self.device)
        else:
            lidar_valid_mask = lidar_valid_mask.to(self.device, dtype=torch.bool)

        if (
            "refined_yaw" in batch
            and "refined_pitch" in batch
        ):
            refined_yaw = _nan_to_num_(batch["refined_yaw"].to(self.device, dtype=torch.float32))
            refined_pitch = _nan_to_num_(batch["refined_pitch"].to(self.device, dtype=torch.float32))

            use_refined_mask = valid_mask & lidar_valid_mask
            fused_yaw_cam = torch.where(use_refined_mask, refined_yaw, pred_yaw_cam)
            fused_pitch_cam = torch.where(use_refined_mask, refined_pitch, pred_pitch_cam)

        measurement_valid_mask = valid_mask & lidar_valid_mask

        # --------------------------------------------------------
        # 4. distance
        # --------------------------------------------------------
        if self.use_gt_distance_fallback:
            distance_keys = ("lidar_distance", "pred_distance", "distance")
        else:
            distance_keys = ("lidar_distance", "pred_distance")

        pred_distance = estimate_distance_from_batch(
            batch=batch,
            device=self.device,
            prefer_key_order=distance_keys,
        )
        pred_distance = _nan_to_num_(pred_distance).clamp_min(1e-4)

        # --------------------------------------------------------
        # 5. direction + distance -> pred_pos_base
        # --------------------------------------------------------
        pred_pos_cam = _camera_angles_and_distance_to_camera_frame_position(
            yaw_cam=fused_yaw_cam,
            pitch_cam=fused_pitch_cam,
            distance=pred_distance,
        )

        pred_pos_base = _camera_frame_position_to_base_frame_position(
            base_env=self.base_env,
            pos_cam=pred_pos_cam,
        )

        pred_yaw_base, pred_pitch_base = _vector_to_base_yaw_pitch(pred_pos_base)

        # --------------------------------------------------------
        # 6. update velocity temporal buffer
        #    关键修复：invalid env 直接 reset，不把脏帧塞进 GRU buffer
        # --------------------------------------------------------
        root_quat_w_for_vel = self.base_env.turret.data.root_quat_w.to(self.device, dtype=torch.float32)
        root_lin_vel_b = self.base_env.turret.data.root_lin_vel_b.to(self.device, dtype=torch.float32)
        root_ang_vel_b = self.base_env.turret.data.root_ang_vel_b.to(self.device, dtype=torch.float32)
        robot_lin_vel_w = _rotate_local_to_world(root_quat_w_for_vel, root_lin_vel_b)
        robot_ang_vel_w = _rotate_local_to_world(root_quat_w_for_vel, root_ang_vel_b)
        robot_lin_vel = _rotate_world_to_yaw_only_base(self.base_env, root_quat_w_for_vel, robot_lin_vel_w)
        robot_ang_vel = _rotate_world_to_yaw_only_base(self.base_env, root_quat_w_for_vel, robot_ang_vel_w)

        invalid_env_ids_for_velocity = torch.nonzero(~measurement_valid_mask, as_tuple=False).squeeze(-1)
        if invalid_env_ids_for_velocity.numel() > 0:
            self.seq_buffer.reset_envs(invalid_env_ids_for_velocity)

        sample_velocity_frame = (self._velocity_bridge_step % self.velocity_sequence_stride) == 0
        self._velocity_bridge_step += 1

        if sample_velocity_frame:
            self.seq_buffer.update(
                pos_base=pred_pos_base,
                robot_lin_vel=robot_lin_vel,
                robot_ang_vel=robot_ang_vel,
                dog_flag=dog_moving_flag,
                valid_mask=measurement_valid_mask,
                reset_invalid=False,
            )

        # --------------------------------------------------------
        # 7. velocity GRU
        # --------------------------------------------------------
        num_envs = int(self.base_env.num_envs)
        pred_vel_base = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)

        x_seq, valid_env_ids = self.seq_buffer.build_model_input()

        vel_ready_mask = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        if valid_env_ids.numel() > 0:
            vel_ready_mask[valid_env_ids] = True

        if self.velocity_model is not None and valid_env_ids.numel() > 0:
            vel_pred = self.velocity_model(x_seq)
            vel_pred = _nan_to_num_(vel_pred)
            pred_vel_base[valid_env_ids] = vel_pred

        if self.use_gt_velocity_until_buffer_ready and "velocity_base" in batch:
            gt_vel_base = batch["velocity_base"].to(self.device, dtype=torch.float32)
            pred_vel_base[~vel_ready_mask] = gt_vel_base[~vel_ready_mask]

        if self.velocity_model is None and "velocity_base" in batch:
            pred_vel_base = batch["velocity_base"].to(self.device, dtype=torch.float32)
            vel_ready_mask[:] = True

        pred_vel_base = _nan_to_num_(pred_vel_base)

        self.base_env._rl_bridge_velocity_ready = vel_ready_mask

        # --------------------------------------------------------
        # 8. 写回 env
        # --------------------------------------------------------
        if self.velocity_model is not None:
            write_mask = measurement_valid_mask & vel_ready_mask
            self.base_env._rl_bridge_training_ready = write_mask.clone()
        else:
            write_mask = measurement_valid_mask
            self.base_env._rl_bridge_training_ready = write_mask.clone()

        env_ids_to_write = torch.nonzero(write_mask, as_tuple=False).squeeze(-1)
        invalid_env_ids = torch.nonzero(~write_mask, as_tuple=False).squeeze(-1)

        if env_ids_to_write.numel() > 0:
            self.base_env.set_rl_predicted_target_state(
                pred_pos_base=pred_pos_base,
                pred_vel_base=pred_vel_base,
                pred_distance=pred_distance,
                pred_yaw_base=pred_yaw_base,
                pred_pitch_base=pred_pitch_base,
                env_ids=env_ids_to_write,
            )

        if invalid_env_ids.numel() > 0 and hasattr(self.base_env, "_rl_pred_ready"):
            self.base_env._rl_pred_ready[invalid_env_ids] = False

        if self.debug and int(self.base_env.common_step_counter) % 50 == 0:
            e0 = 0
            print(
                "[rl_perception_bridge]",
                f"step={int(self.base_env.common_step_counter)}",
                f"valid={int(valid_mask.sum().item())}/{num_envs}",
                f"vel_ready={int(vel_ready_mask.sum().item())}/{num_envs}",
                f"vel_stride={self.velocity_sequence_stride}",
                f"vel_sample={int(sample_velocity_frame)}",
                f"raycast_valid={int(batch.get('lidar_valid_mask', torch.zeros_like(valid_mask)).to(self.device, dtype=torch.bool).sum().item()) if 'lidar_valid_mask' in batch else 0}/{num_envs}",
                f"pred_yaw_cam0={float(pred_yaw_cam[e0].item()):.4f}",
                f"pred_pitch_cam0={float(pred_pitch_cam[e0].item()):.4f}",
                f"pred_dist0={float(pred_distance[e0].item()):.4f}",
                f"pred_pos_base0={pred_pos_base[e0].detach().cpu().tolist()}",
                f"pred_vel_base0={pred_vel_base[e0].detach().cpu().tolist()}",
            )

        return {
            "valid_mask": valid_mask,
            "pred_distance": pred_distance,
            "pred_pos_base": pred_pos_base,
            "pred_vel_base": pred_vel_base,
            "pred_yaw_base": pred_yaw_base,
            "pred_pitch_base": pred_pitch_base,
        }


# ============================================================
# Runtime install helper
# ============================================================

def install_rl_perception_bridge(
    base_env: Any,
    vision_model_path: str | Path,
    velocity_model_path: str | Path | None = None,
    velocity_model_def_path: str | Path | None = None,
    device: str | torch.device | None = None,
    use_gt_distance_fallback: bool = True,
    use_gt_velocity_until_buffer_ready: bool = True,
    velocity_sequence_stride: int = 3,
    debug: bool = False,
) -> RLPerceptionBridge:
    """
    创建 bridge，并挂到 env 上，方便外部脚本直接调用：
        base_env._rl_perception_bridge.step()
    """
    bridge = RLPerceptionBridge(
        base_env=base_env,
        vision_model_path=vision_model_path,
        velocity_model_path=velocity_model_path,
        velocity_model_def_path=velocity_model_def_path,
        device=device,
        use_gt_distance_fallback=use_gt_distance_fallback,
        use_gt_velocity_until_buffer_ready=use_gt_velocity_until_buffer_ready,
        velocity_sequence_stride=velocity_sequence_stride,
        debug=debug,
    )
    base_env._rl_perception_bridge = bridge
    print("[rl_perception_bridge] installed.")
    return bridge

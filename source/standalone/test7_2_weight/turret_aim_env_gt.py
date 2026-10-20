from __future__ import annotations
import math
import os
from collections.abc import Sequence
from typing import cast
import traceback
from pathlib import Path
import imageio.v2 as imageio
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
import numpy as np
from isaaclab.sensors import (
    TiledCamera,
    TiledCameraCfg,
    RayCasterCfg,
    MultiMeshRayCaster,
    MultiMeshRayCasterCfg,
    patterns,
)
from isaaclab.utils.math import quat_apply, sample_uniform, wrap_to_pi
from pxr import Gf, UsdGeom


@configclass
class TurretAimEnvCfg(DirectRLEnvCfg):
    """炮台瞄准目标（DirectRL）环境配置

    设计目标：
    - 加载你的资产文件 `lite3_with_turret.usd`（机器狗 + 炮台）。
    - 仅控制炮台 yaw/pitch 两个关节。
    - reset 时随机生成目标点，奖励按瞄准误差。

    注意：需要把下面的关节/连杆名字改成 USD 里真实的名字。
    """

    # env
    episode_length_s = 10.0 #每个episode的时长
    max_shots_per_episode: int = 0  # 0 disables the budget; count actual fires, not intent.
    decimation = 4 # 与 Lite3 locomotion 训练保持一致（0.005 * 4 = 0.02s）
    action_space = 3  # 动作维度

    # Actor obs:
    # pred_pos_base(3)
    # pred_vel_base(3)
    # pred_distance(1)
    # policy_yaw_error_los(1)      # 目标直视 yaw 与真实枪口 yaw 的误差，不含重力补偿
    # policy_pitch_error_los(1)    # 目标直视 pitch 与真实枪口 pitch 的误差，不含重力补偿
    # turret_yaw_pitch(2)
    # turret_yaw_pitch_vel(2)
    # root_lin_vel_b(3)
    # root_ang_vel_b(3)
    # projected_gravity_b(3)
    # dog_moving_flag(1)
    # prev_actions(3)
    # fire_intent(1)
    # fire_cooldown_ratio(1)
    observation_space = 28

    # Critic privileged obs:
    # gt_pos_base(3)
    # gt_vel_base(3)
    # gt_distance(1)
    # gt_yaw_error_gravity(1)      # gravity-only 解析 yaw 误差
    # gt_pitch_error_gravity(1)    # gravity-only 解析 pitch 误差
    # turret_yaw_pitch(2)
    # turret_yaw_pitch_vel(2)
    # root_lin_vel_b(3)
    # root_ang_vel_b(3)
    # projected_gravity_b(3)
    # dog_moving_flag(1)
    # prev_actions(3)
    # fire_intent(1)
    # fire_cooldown_ratio(1)
    # target_acc_base(3)
    state_space = 31

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=500.0, replicate_physics=True)
    
    debug_vis: bool = False     #是否可视化
    debug_diagnostics: bool = False     #是否打印调试诊断信息
    fail_on_invalid_actions: bool = True    #动作无效时是否抛出错误（如包含 NaN 或 Inf）
    spawn_ground: bool = False     #是否生成地面

    # --- asset ---
    usd_path: str = os.path.join(os.path.dirname(__file__), "lite3_with_turret.usd")
    asset_prim_path: str = "/World/envs/env_0/Lite3"  # 只在 env_0 里 spawn，后续 clone
    turret_prim_path_expr: str = "/World/envs/env_.*/Lite3/Lite3/TORSO"
    # --- vision camera / supervision ---
    enable_vision_supervision: bool = True

    # 这里必须改成你 USD 里真实的 camera prim 路径（支持 env_.*/ 正则）
    # 下面这个路径只是按你之前发的路径改成多环境形式后的示例
    # ------------------------------------------------------------------
    # camera paths
    # ------------------------------------------------------------------
    # 真实挂在机器狗上的相机（只用来读取固定外参 / 原始相机参数）
    camera_prim_path_expr: str = (
        "/World/envs/env_.*/Lite3/Lite3/TORSO/"
        "SG2_OX03CC_5200_GMSL2_H60YA/"
        "Camera_SG2_OX03CC_5200_GMSL2_H60YA"
    )

    # 运行时渲染代理相机（真正给 TiledCamera 绑定的相机）
    # 不挂在 articulation 下面，而是放在 /World 下；
    # 每帧把 runtime 重建出来的世界位姿写进去，避免 TiledCamera 读到“死相机”
    runtime_camera_proxy_prim_path_expr: str = "/World/RuntimeRenderCameras/env_.*/Camera"

    camera_width: int = 1024
    camera_height: int = 576
    vision_history_length: int = 4
    vision_update_period: float = 0.0
    vision_normalize_rgb: bool = True

    # --- debug image dump ---
    dump_debug_camera_steps: tuple[int, ...] = ()
    dump_debug_camera_env_id: int = 0
    dump_debug_camera_dir: str = "debug_camera_frames"

    # 注意：TiledCamera 绑定的是 proxy camera，而不是真实 USD 相机
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path=runtime_camera_proxy_prim_path_expr,
        update_period=vision_update_period,
        data_types=["rgb"],
        spawn=None,
        width=camera_width,
        height=camera_height,
        update_latest_camera_pose=True,
    )

    # --- MultiMeshRayCaster: 挂到 TORSO，用多条射线模拟 runtime lidar 的锥形束 ---
    enable_dynamic_target_raycast_debug: bool = False
    dynamic_target_raycast_debug_every: int = 20

    # 直接挂到每个 env 的 TORSO
    dynamic_target_raycast_parent_prim_path_expr: str = "/World/envs/env_.*/Lite3/Lite3/TORSO"

    # 相对 TORSO 的安装位姿：与原 runtime lidar 保持一致
    dynamic_target_raycast_translation: tuple[float, float, float] = (0.23, 0.0, 0.073)
    dynamic_target_raycast_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # 射线束参数：尽量对齐你原来的 runtime lidar
    dynamic_target_raycast_min_range: float = 0.3
    # Must cover the complete 80--150 m target shell. Keep a small margin for
    # camera/base offsets and the rendered target radius.
    dynamic_target_raycast_max_range: float = 165.0
    dynamic_target_raycast_horizontal_fov_deg: float = 180.0
    dynamic_target_raycast_vertical_fov_deg: float = 40.0
    dynamic_target_raycast_horizontal_resolution_deg: float = 0.2
    dynamic_target_raycast_vertical_resolution_deg: float = 0.3

    # 预留一个基础 cfg，真正的 pattern 会在 _create_dynamic_target_multimesh_raycast_sensor() 里按上面参数动态生成
    dynamic_target_multimesh_raycast_cfg: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Lite3/Lite3/TORSO",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.23, 0.0, 0.073),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        attach_yaw_only=False,
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=2,
            vertical_fov_range=(-10.0, 10.0),
            horizontal_fov_range=(-25.0, 25.0),
            horizontal_res=1.0,
        ),
        max_distance=165.0,
        update_mesh_ids=False,
        reference_meshes=True,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/TargetSphere",
                is_shared=False,
                merge_prim_meshes=True,
                track_mesh_transforms=True,
            ),
        ],
    )

    #自动寻找炮台 prim
    auto_resolve_turret_prim: bool = False
    # Prefer articulation roots whose prim-path contains any of these substrings (case-insensitive).
    turret_path_hints: tuple[str, ...] = ("turret", "gimbal", "mount")
    # Search depth (relative to asset root prim) when scanning for articulation roots.
    turret_search_depth: int = 10

    # 控制的joint
    yaw_joint_name: str = "yaw_joint"
    pitch_joint_name: str = "pitch_joint"

    # 可选：用某个连杆（枪口/炮管）来计算朝向与位置，会比仅用 yaw/pitch 解析式更准确。
    # 例：muzzle_body_name_exp
    # r = "barrel_link" 或 "muzzle"（支持正则）
    muzzle_body_name_expr = "pitch_link"
    muzzle_forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # 枪口朝向在该连杆坐标系下的轴
    muzzle_offset_b: tuple[float, float, float] = (0.09, 0.0, 0.0)  # gun_tip 相对 pitch_link 的局部偏移

    # --- control ---
    max_delta_yaw: float = 0.03      # 每个 env step 最大 yaw 增量，单位 rad
    max_delta_pitch: float = 0.02    # 每个 env step 最大 pitch 增量，单位 rad

    # --- observation normalization ---
    # 瞄准误差单独用高增益尺度，不要和关节角共用 pi / pitch_limit。
    # 0.05 rad ≈ 2.86 deg，表示 0.05rad 误差归一化为 1.0。
    # 远距离命中时 0.01~0.05rad 都是关键误差范围。
    aim_error_obs_scale: float = 0.05
    aim_error_obs_clip: float = 5.0

    # --- locomotion policy (for moving-base aiming) ---
    enable_locomotion_policy: bool = False      #是否启用预训练的移动策略来控制腿部关节（否则保持默认站立姿态）
    locomotion_policy_path: str | None = None
    locomotion_command_resample_s: float = 10.0     #每隔多少秒重新采样一次移动指令（线速度 x/y 和角速度 z）
    locomotion_lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    locomotion_lin_vel_y_range: tuple[float, float] = (-0.8, 0.8)
    locomotion_ang_vel_z_range: tuple[float, float] = (-0.8, 0.8)
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.35)

    # --- target sampling (vision-friendly, in camera sector) ---
    # 距离限制在 80~150 m
    target_radius_range: tuple[float, float] = (80.0, 150.0)
    # --- rendered target sphere ---
    render_target_sphere: bool = True
    target_sphere_radius: float = 0.35
    target_sphere_color: tuple[float, float, float] = (1.0, 0.1, 0.1)

    # 视觉模块的有效距离范围（相对相机）
    vision_valid_distance_range: tuple[float, float] = (80.0, 150.0)
    vision_distance_bounce_margin: float = 0.2

    # 初始目标的相对高度采样范围。它默认只控制 reset 时的初始高度，
    # 不限制目标后续运动；目标是否留在画面内由 camera-sector yaw/pitch 约束保证。
    target_height_range: tuple[float, float] = (0.5, 2.0)
    constrain_target_relative_height: bool = False
    target_yaw_range: tuple[float, float] = (-math.pi / 2, math.pi / 2)

    # 按相机 yaw/pitch 视场采样目标。启用后 target_height_range 仅作为
    # 相机不可用时的 fallback，不再把正式视觉训练限制在固定相对高度。
    sample_target_in_camera_sector: bool = True

    # 相机扇区采样：水平/垂直角范围（单位 rad）
    # 建议先取相对保守范围，保证目标更容易在画面中央附近
    # The 1024x576 camera has an approximately +/-0.28 rad horizontal half
    # FOV. Keep a small image-edge margin for the complete rendered sphere.
    target_cam_yaw_range: tuple[float, float] = (-0.26, 0.26)
    target_cam_pitch_range: tuple[float, float] = (-0.12, 0.12)
    # Do not terminate on a single missing render. At 50 Hz this default is a
    # 0.1 s grace period, applied only after all reset/aim warmups finish.
    target_out_of_view_grace_frames: int = 5

    # 如果你想让目标尽量更容易看见，可以给一个额外高度偏移范围
    # 最终高度会由相机位置 + 距离 * sin(pitch) 决定，这里通常不再单独采 z
    # dynamic target (optional)
    enable_dynamic_target: bool = False
    # Three-dimensional speed magnitude; direction is uniform on the unit sphere.
    target_speed_range: tuple[float, float] = (0.6, 5.0)
    # --- 新增：目标加速度配置 ---
    target_acc_range: tuple[float, float] = (-0.6, 0.6)
    target_vertical_acc_range: tuple[float, float] = (-0.2, 0.2)

    # --- termination ---
    failure_height_threshold: float = 0.16
    failure_tilt_cos_threshold: float = 0.4     #炮台根部竖直向上的向量与世界竖直向上的向量的夹角余弦值低于该阈值即判定为失败（对应约 66° 以上的倾倒）

    # --- ballistic (gravity + optional air drag) ---
    muzzle_speed: float = 882.0      #枪口初速，单位 m/s
    gravity_mps2: float = 9.81
    hit_radius: float = 0.35    #命中判定半径
    ballistic_max_time: float = 0.4     
    ballistic_num_samples: int = 401

    # 空气阻力：Mach-Cd 查表二次阻力
    enable_air_drag: bool = False

    air_density: float = 1.204              # kg/m^3
    speed_of_sound_mps: float = 343.0       # m/s，常温近似声速

    projectile_diameter_m: float = 0.0057   # 5.56mm 弹头直径，单位 m
    projectile_mass_kg: float = 0.004       # 约 4.0g，单位 kg

    # fallback：当 use_mach_dependent_cd=False 时使用固定 Cd
    drag_coefficient: float = 0.25

    # 是否使用 Mach-Cd 查表
    use_mach_dependent_cd: bool = True

    # Mach-Cd 表：根据当前弹丸速度对应的 Mach 数插值得到 Cd
    mach_cd_mach: tuple[float, ...] = (
        0.00, 0.50, 0.80, 0.95, 1.05, 1.20, 1.50, 2.00, 2.50, 3.00
    )

    mach_cd_cd: tuple[float, ...] = (
        0.16, 0.16, 0.18, 0.28, 0.36, 0.34, 0.30, 0.26, 0.24, 0.23
    )

    # --- reward weights ---
    w_rate: float = -1    #位置增量惩罚权重
    w_action_change: float = -0.01   #动作变化惩罚权重（鼓励平滑）
    r_hit: float = 100.0
    r_fire_cost: float = -0.8
    r_time_penalty: float = -0.7

    # 开火后冷却步数：控制最小射击间隔
    fire_cooldown_steps: int = 5

    # direct fire intent 参数：
    # action[2] -> fire_intent = (action[2] + 1) / 2
    # 连续 fire_intent > threshold 达到 hold_steps 后，才允许开火
    fire_intent_threshold: float = 0.65
    fire_intent_hold_steps: int = 2
    r_no_fire_timeout_penalty: float = -5.0

    # 第一枪命中奖励：鼓励一次瞄准、一次命中
    r_first_shot_hit_bonus: float = 50

    # 重复开火递增惩罚：第2枪开始，每次 fire 额外扣分越来越多
    r_repeat_fire_penalty: float = -4

    w_yaw_align = 0.1
    w_pitch_align = 0.1

    aim_warmup_s: float = 2.0       #机器狗随机移动时间
    w_ballistic: float = 0.8       #弹道奖励权重（基于弹道预测的命中距离和命中判定）


    # 命中判定距离
    max_hit_distance: float = 170.0

    # Turret actuator limits and gains are inherited from the authored USD drives.


class TurretAimEnv(DirectRLEnv):
    cfg: TurretAimEnvCfg

    def __init__(self, cfg: TurretAimEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.max_shots_per_episode < 0:
            raise ValueError("max_shots_per_episode must be nonnegative (0 disables the budget)")
        # Target spheres are needed by camera/raycast training or interactive visualization only.
        # In headless GT mode both flags are false, so no sphere prim is spawned or updated.
        self._target_sphere_visuals_enabled = bool(
            cfg.render_target_sphere
            and (cfg.enable_vision_supervision or cfg.enable_dynamic_target_raycast_debug or cfg.debug_vis)
        )

        # ----------------------------
        # dynamic target raycast debug (MultiMeshRayCaster)
        # ----------------------------
        self._dynamic_target_raycast_sensor: MultiMeshRayCaster | None = None
        self._dynamic_target_raycast_mount_paths: list[str] = []
        self._dynamic_target_raycast_mount_translate_ops = []
        self._dynamic_target_raycast_mount_orient_ops = []

        # ----------------------------
        # vision camera / runtime proxy
        # ----------------------------
        self._vision_camera: TiledCamera | None = None
        self._runtime_camera_proxy_paths: list[str] = []
        self._vision_camera_bootstrapped: bool = False
        self._vision_supervision_refreshed_this_step = False

        self._debug_camera_dumped_steps: set[int] = set()
        self._debug_camera_dump_dir = Path(cfg.dump_debug_camera_dir).expanduser().resolve()
        self._debug_camera_dump_dir.mkdir(parents=True, exist_ok=True)
        # print(f"[debug image] dump dir = {self._debug_camera_dump_dir}")

        # eval/action debug counter
        self._debug_print_counter = 0

        super().__init__(cfg, render_mode, **kwargs)
        self._target_markers: VisualizationMarkers | None = None

        # resolve joints / bodies
        self._yaw_dof_ids, yaw_names = self.turret.find_joints(self.cfg.yaw_joint_name)
        self._pitch_dof_ids, pitch_names = self.turret.find_joints(self.cfg.pitch_joint_name)
        if len(self._yaw_dof_ids) != 1 or len(self._pitch_dof_ids) != 1:
            raise RuntimeError(
                "Yaw/Pitch joints must resolve to exactly 1 joint each. "
                f"Got yaw={yaw_names}, pitch={pitch_names}. Available joints: {self.turret.joint_names}"
            )

        self._turret_dof_ids = [int(self._yaw_dof_ids[0]), int(self._pitch_dof_ids[0])]
        turret_joint_limits = self.turret.data.joint_pos_limits[:, self._turret_dof_ids, :]
        invalid_joint_limits = (
            ~torch.isfinite(turret_joint_limits).all(dim=(1, 2))
            | (turret_joint_limits[..., 0] >= turret_joint_limits[..., 1]).any(dim=1)
        )
        if torch.any(invalid_joint_limits):
            raise RuntimeError(
                "Yaw/pitch USD joint limits must be finite and ordered as [lower, upper]. "
                f"Got {turret_joint_limits[0].detach().cpu().tolist()}"
            )
        if self.cfg.debug_diagnostics:
            print(
                "[TurretAimEnv] yaw/pitch limits from USD (rad) =",
                turret_joint_limits[0].detach().cpu().tolist(),
            )

        self._muzzle_body_id: int | None = None
        if self.cfg.muzzle_body_name_expr is not None:
            body_ids, body_names = self.turret.find_bodies(self.cfg.muzzle_body_name_expr)
            if len(body_ids) == 0:
                raise RuntimeError(
                    f"muzzle_body_name_expr='{self.cfg.muzzle_body_name_expr}' did not match any bodies. "
                    f"Available bodies: {self.turret.body_names}"
                )
            self._muzzle_body_id = int(body_ids[0])

        self._forward_axis_b = torch.tensor(self.cfg.muzzle_forward_axis, device=self.device, dtype=torch.float32)
        self._forward_axis_b = self._forward_axis_b / (torch.norm(self._forward_axis_b) + 1e-6)
        self._muzzle_offset_b = torch.tensor(self.cfg.muzzle_offset_b, device=self.device, dtype=torch.float32)

        self._leg_joint_names = [
            "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
            "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
            "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
            "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
        ]
        self._leg_dof_ids = self._resolve_leg_joint_ids(self._leg_joint_names)
        self._leg_action_scale = torch.tensor(
            [0.125 if "HipX" in name else 0.25 for name in self._leg_joint_names],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self._default_leg_joint_pos = self.turret.data.default_joint_pos[:, self._leg_dof_ids].clone()

        # buffers(动作,目标关节速度,目标点坐标,连续成功步数)
        self._actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._turret_pos_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, 3), device=self.device)  # 保存上一时刻动作
        self._delta_pos = torch.zeros((self.num_envs, 2), device=self.device)     # 本步 yaw/pitch 位置增量
        self._fire_command = torch.zeros((self.num_envs,), device=self.device)
        self._fire_cooldown = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._fire_intent = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # 连续高 fire intent 计数：
        # 只做去抖确认，不是 rearm；不要求 intent 降下来。
        # 如果 intent 一直高，cooldown 结束后仍然可以再次开火。
        self._fire_intent_high_steps = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )

        self._episode_fire_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._shot_budget_exhausted = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # 每个 episode 内的开火弹道最小距离统计
        # 用于训练日志观察“是不是乱开火 / 开火离目标多远”
        self._episode_fire_min_dist_sum = torch.zeros(
            (self.num_envs,),
            dtype=torch.float32,
            device=self.device,
        )
        self._episode_fire_min_dist_count = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self._episode_fire_min_dist_min = torch.full(
            (self.num_envs,),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        self._aim_warmup_left_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._target_quat_w[:, 0] = 1.0   # identity quaternion: wxyz = [1, 0, 0, 0]
        self._target_acc_w = torch.zeros((self.num_envs, 3), device=self.device) # 加速度
        self._target_anchor_w = torch.zeros((self.num_envs, 3), device=self.device)
        # Per-environment target motion policy.  Standalone environments default
        # to the config-wide dynamic-target behavior; mixed train/eval scripts
        # replace these masks per episode (SS/SD/DS/DD).
        default_target_dynamic = torch.full(
            (self.num_envs,),
            bool(cfg.enable_dynamic_target),
            dtype=torch.bool,
            device=self.device,
        )
        self._target_motion_mask = default_target_dynamic.clone()
        self._target_relative_constraint_mask = default_target_dynamic.clone()
        self._target_constraint_fixed_anchor_mask = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._target_out_of_view_termination_mask = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._target_out_of_view_streak = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._target_out_of_view_terminated = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        # 视觉 / 雷达 GT 缓存
        self._vision_distance_gt = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._vision_pos_base_gt = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # RL bridge 写回的预测状态
        self._rl_pred_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._rl_pred_pos_base = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._rl_pred_vel_base = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._rl_pred_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._rl_pred_yaw_base = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._rl_pred_pitch_base = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # bridge 侧辅助标志
        self._rl_bridge_training_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._rl_bridge_velocity_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # 环境侧是否允许在 RL 预测缺失时回退到 GT
        # 训练脚本可在严格模式下把它改成 False：
        #     base_env._use_gt_fallback_for_missing_rl_pred = False
        self._use_gt_fallback_for_missing_rl_pred = True

        self._episode_start_target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._locomotion_commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._locomotion_last_actions = torch.zeros((self.num_envs, len(self._leg_joint_names)), device=self.device)
        self._locomotion_joint_targets = self._default_leg_joint_pos.clone()
        self._locomotion_command_time_left = torch.zeros((self.num_envs,), device=self.device)
        self._locomotion_policy = self._load_locomotion_policy()
        self._aim_warmup_steps = max(0, int(round(self.cfg.aim_warmup_s / self.step_dt)))
        self._hit_flag = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if self.cfg.debug_diagnostics:
            print("[TurretAimEnv][init] enable_locomotion_policy =", self.cfg.enable_locomotion_policy)
            print("[TurretAimEnv][init] locomotion_policy_path =", self.cfg.locomotion_policy_path)
            print("[TurretAimEnv][init] locomotion_policy_loaded =", self._locomotion_policy is not None)
        self._invalid_debug_print_count = 0

        self._gravity_w = torch.tensor([0.0, 0.0, -self.cfg.gravity_mps2], device=self.device, dtype=torch.float32)
        # ballistic cache：同一个 env step 内 reward / done 共用，避免重复积分弹道
        self._ballistic_cache_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._cached_min_ballistic_dist = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._cached_ballistic_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._cached_closest_time = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # ------------------------------------------------------------------
        # vision buffers
        # ------------------------------------------------------------------
        self._vision_history_length = int(self.cfg.vision_history_length)

        self._vision_rgb_history: torch.Tensor | None = None
        if self.cfg.enable_vision_supervision:
            self._vision_rgb_history = torch.zeros(
                (
                    self.num_envs,
                    self._vision_history_length,
                    3,
                    int(self.cfg.camera_height),
                    int(self.cfg.camera_width),
                ),
                dtype=torch.float32,
                device=self.device,
            )

        # 每一帧是否有效：相机姿态正常、标签有限、目标在画面内、warmup 结束
        self._vision_frame_valid_history = torch.zeros(
            (self.num_envs, self._vision_history_length),
            dtype=torch.bool,
            device=self.device,
        )

        self._vision_history_count = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )

        self._vision_valid_mask = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )

        self._vision_direction_gt = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float32,
            device=self.device,
        )

        self._vision_velocity_gt = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )

        # 目标相对机器狗本体速度（base frame）
        self._vision_velocity_base_gt = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )

        # reset 之后，先等若干个 step，让 camera pose / sensor 输出稳定
        self._vision_collect_warmup_steps_after_reset = 2
        self._vision_collect_warmup_left = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        # reset 后是否还没有完成“正式目标采样”
        # True 表示当前 target 只是占位点，等相机 pose 稳定后再真正按 camera sector 采样
        self._pending_target_resample = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )

        self._target_sphere_prim_paths: list[str] = []
        if self._target_sphere_visuals_enabled:
            self._target_sphere_prim_paths = [
                f"/World/envs/env_{i}/TargetSphere" for i in range(self.num_envs)
            ]

        self._raycast_hit_distance = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self._raycast_hit_valid = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._raycast_expected_surface_distance = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )

        # evaluation metrics (跨回合累计)
        self._eval_total_fires = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_total_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_total_hits = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_total_timeouts = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_total_fallen = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_total_invalid = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_total_target_out_of_view = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_total_shot_budget_exhausted = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_target_spawn_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_target_spawn_sum = torch.zeros((3,), dtype=torch.float32, device=self.device)
        self._eval_fire_min_dist_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._eval_fire_min_dist_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self._eval_fire_min_dist_min = torch.full((), float("inf"), dtype=torch.float32, device=self.device)
        self._eval_fire_min_dist_max = torch.zeros((), dtype=torch.float32, device=self.device)
        self._eval_target_spawn_samples: list[list[float]] = []
        self._eval_fire_min_dist_samples: list[float] = []
        self._eval_episode_fire_counts: list[int] = []
        self._eval_episode_hits: list[int] = []
        self._collect_eval_details: bool = False
        self._eval_samples_limit: int = 50

        self._ballistic_time_samples = torch.linspace(
            0.0,
            self.cfg.ballistic_max_time,
            self.cfg.ballistic_num_samples,
            device=self.device,
            dtype=torch.float32,
        )  # shape (num_samples,)

        # 时间离散步长（长度为 num_samples-1），用于离散积分弹道
        self._ballistic_dt_samples = self._ballistic_time_samples[1:] - self._ballistic_time_samples[:-1]

        # 二次空气阻力参数：
        #     a_drag = -0.5 * rho * Cd(Mach) * A / m * |v| * v
        #     A = pi * (d / 2)^2
        #
        # 注意：
        # - 这里不考虑风速，所以 v 直接使用弹丸世界系速度 bullet_vel。
        # - _drag_accel_coeff 仍保留为固定 Cd fallback，避免其它旧代码引用时报错。
        projectile_radius = 0.5 * float(self.cfg.projectile_diameter_m)
        cross_section_area = math.pi * projectile_radius * projectile_radius
        projectile_mass = max(float(self.cfg.projectile_mass_kg), 1e-6)

        self._projectile_area = torch.tensor(
            cross_section_area,
            device=self.device,
            dtype=torch.float32,
        )

        self._projectile_mass = torch.tensor(
            projectile_mass,
            device=self.device,
            dtype=torch.float32,
        )

        self._air_density = torch.tensor(
            float(self.cfg.air_density),
            device=self.device,
            dtype=torch.float32,
        )

        self._speed_of_sound_mps = torch.tensor(
            max(float(self.cfg.speed_of_sound_mps), 1e-6),
            device=self.device,
            dtype=torch.float32,
        )

        self._mach_table = torch.tensor(
            self.cfg.mach_cd_mach,
            device=self.device,
            dtype=torch.float32,
        )

        self._cd_table = torch.tensor(
            self.cfg.mach_cd_cd,
            device=self.device,
            dtype=torch.float32,
        )

        if self._mach_table.numel() != self._cd_table.numel() or self._mach_table.numel() < 2:
            raise RuntimeError(
                "mach_cd_mach and mach_cd_cd must have the same length and contain at least 2 points."
            )

        if not torch.all(self._mach_table[1:] > self._mach_table[:-1]):
            raise RuntimeError("mach_cd_mach must be strictly increasing.")

        # 固定 Cd fallback，主要用于兼容旧代码
        fallback_drag_accel_coeff = (
            0.5
            * float(self.cfg.air_density)
            * float(self.cfg.drag_coefficient)
            * cross_section_area
            / projectile_mass
        )

        self._drag_accel_coeff = torch.tensor(
            fallback_drag_accel_coeff,
            device=self.device,
            dtype=torch.float32,
        )




    def _ensure_runtime_camera_proxies(self) -> None:
        """Create proxy render cameras under /World/RuntimeRenderCameras.

        真实相机挂在 articulation 下面，某些情况下 Hydra / TiledCamera 不会稳定跟踪其位姿。
        这里创建一组独立的 proxy camera，每帧把 runtime 相机世界位姿同步到 proxy，
        TiledCamera 只读取 proxy，相当于把“渲染相机”和“几何真值相机”强行统一。
        """
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom, Sdf, Gf

        stage = get_current_stage()
        if stage is None:
            raise RuntimeError("USD stage is None, cannot create runtime camera proxies.")

        root_path = "/World/RuntimeRenderCameras"
        root_xf = UsdGeom.Xform.Define(stage, root_path)

        source_camera_paths = sim_utils.find_matching_prim_paths(self.cfg.camera_prim_path_expr)
        if len(source_camera_paths) == 0:
            raise RuntimeError(
                f"No source camera prim matched: {self.cfg.camera_prim_path_expr}"
            )

        self._runtime_camera_proxy_paths: list[str] = []

        for env_id in range(self.num_envs):
            src_path = (
                f"/World/envs/env_{env_id}/Lite3/Lite3/TORSO/"
                "SG2_OX03CC_5200_GMSL2_H60YA/"
                "Camera_SG2_OX03CC_5200_GMSL2_H60YA"
            )
            proxy_env_path = f"{root_path}/env_{env_id}"
            proxy_cam_path = f"{proxy_env_path}/Camera"

            UsdGeom.Xform.Define(stage, proxy_env_path)
            proxy_cam = UsdGeom.Camera.Define(stage, proxy_cam_path)
            self._runtime_camera_proxy_paths.append(proxy_cam_path)

            src_prim = stage.GetPrimAtPath(src_path)
            if not src_prim.IsValid():
                raise RuntimeError(f"Invalid source camera prim: {src_path}")

            src_cam = UsdGeom.Camera(src_prim)

            # 复制关键成像参数
            for attr_name in [
                "projection",
                "focalLength",
                "focusDistance",
                "fStop",
                "horizontalAperture",
                "verticalAperture",
                "horizontalApertureOffset",
                "verticalApertureOffset",
            ]:
                src_attr = src_cam.GetPrim().GetAttribute(attr_name)
                if src_attr and src_attr.IsValid() and src_attr.HasValue():
                    proxy_cam.GetPrim().CreateAttribute(
                        attr_name,
                        src_attr.GetTypeName(),
                        custom=False,
                    ).Set(src_attr.Get())

            # clippingRange 需要单独处理
            try:
                clip_val = src_cam.GetClippingRangeAttr().Get()
                if clip_val is not None:
                    proxy_cam.GetClippingRangeAttr().Set(clip_val)
            except Exception:
                pass

            # 关键：标准化为 [translate, orient, scale]，否则 TiledCamera 初始化会报错
            # "not a xformable prim with standard transform operations [translate, orient, scale]"
            sim_utils.standardize_xform_ops(
                proxy_cam.GetPrim(),
                translation=(0.0, 0.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            )

        if self.cfg.debug_diagnostics:
            print(
                "[vision] created runtime camera proxies:",
                self._runtime_camera_proxy_paths[: min(4, len(self._runtime_camera_proxy_paths))]
            )


    def _ensure_vision_camera_initialized(self) -> bool:
        """确保延迟创建的 TiledCamera 已完成 SensorBase 内部初始化。

        说明：
        - 当 TiledCamera 不走 scene.sensors 的标准生命周期时，
        PLAY 回调可能不会替它补齐 _is_outdated / _timestamp 等内部字段。
        - 此时访问 self._vision_camera.data 会在 SensorBase._update_outdated_buffers() 里报：
        AttributeError: 'TiledCamera' object has no attribute '_is_outdated'
        """
        if self._vision_camera is None:
            return False

        # 已初始化完成
        if getattr(self._vision_camera, "_is_initialized", False) and hasattr(self._vision_camera, "_is_outdated"):
            return True

        try:
            # 优先走底层初始化实现
            if hasattr(self._vision_camera, "_initialize_impl"):
                self._vision_camera._initialize_impl()
                self._vision_camera._is_initialized = True
        except Exception as exc:
            if self.cfg.debug_diagnostics:
                print(f"[vision] manual camera _initialize_impl failed: {exc}")

        ok = getattr(self._vision_camera, "_is_initialized", False) and hasattr(self._vision_camera, "_is_outdated")
        if self.cfg.debug_diagnostics and ok:
            print("[vision] delayed TiledCamera manual initialization succeeded")

        return ok

    def bootstrap_vision_camera(self) -> None:
        """两阶段初始化的第二阶段。

        目的：
        1) 在 super().__init__() 之后延迟创建 TiledCamera；
        2) 先把 runtime pose 推到 proxy camera；
        3) 手动补齐 SensorBase 初始化；
        4) render 几次，让 render product / annotator 稳定挂载。

        注意：
        - 这里创建出来的 TiledCamera 不再注册到 self.scene.sensors。
        - 后续由本类手动调用 self.sim.render() + self._vision_camera.update(dt=0.0)。
        """
        if not self.cfg.enable_vision_supervision:
            return
        if self._vision_camera_bootstrapped:
            return

        if len(self._runtime_camera_proxy_paths) == 0:
            self._ensure_runtime_camera_proxies()

        # 如果之前误注册过，先从 scene.sensors 里移除
        if hasattr(self, "scene") and hasattr(self.scene, "sensors"):
            if "vision_camera" in self.scene.sensors:
                try:
                    self.scene.sensors.pop("vision_camera", None)
                except Exception:
                    pass

        if self._vision_camera is None:
            self._vision_camera = TiledCamera(self.cfg.tiled_camera)

        # 先把当前 runtime pose 写进 proxy，相机首帧就不要错
        try:
            self._push_runtime_camera_pose_to_usd()
        except Exception as exc:
            print(f"[vision] initial proxy pose push failed: {exc}")

        # 手动初始化延迟创建的 TiledCamera
        if not self._ensure_vision_camera_initialized():
            raise RuntimeError(
                "[vision] Delayed TiledCamera manual initialization failed. "
                "The camera object exists, but SensorBase internals were not initialized."
            )

        # warmup 几帧
        last_exc = None
        for _ in range(6):
            try:
                self.sim.render()
            except Exception as exc:
                last_exc = exc
                continue

            try:
                self._vision_camera.update(dt=0.0)
                _ = self._vision_camera.data  # 触发一次 lazy path，确保内部缓存可访问
                last_exc = None
            except Exception as exc:
                last_exc = exc

        if last_exc is not None and self.cfg.debug_diagnostics:
            print(f"[vision] bootstrap warmup last exception: {last_exc}")

        self._vision_camera_bootstrapped = True
        self._vision_supervision_refreshed_this_step = False


    def _set_camera_pose_on_prim(
        self,
        prim_path: str,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
    ) -> None:
        """把 runtime 相机世界位姿写到 proxy camera prim。

        注意：
        - 这里强制只用标准 [translate, orient, scale] xform ops
        - 不叠加额外 transform op，避免 Hydra/TiledCamera 读姿态时错乱
        """
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom, Gf, Usd

        stage = get_current_stage()
        if stage is None:
            return

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        # 强制标准化，确保只有 translate/orient/scale 这套标准 op 顺序
        sim_utils.standardize_xform_ops(
            prim,
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )

        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()

        translate_op = None
        orient_op = None

        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
                orient_op = op

        if translate_op is None or orient_op is None:
            return

        p = pos_w.detach().cpu().tolist()
        q = quat_wxyz.detach().cpu().tolist()

        translate_op.Set(
            Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])),
            Usd.TimeCode.Default(),
        )
        orient_op.Set(
            Gf.Quatd(
                float(q[0]),
                Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])),
            ),
            Usd.TimeCode.Default(),
        )


    def _setup_scene(self):
        # 可选生成地面
        if self.cfg.spawn_ground:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 先在 env_0 下放入 USD 资产，后续 clone
        usd_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.usd_path)
        sim_utils.spawn_from_usd(self.cfg.asset_prim_path, usd_cfg)

        # 在 env_0 下创建“纯 visual”的 target sphere，后续 clone
        if self._target_sphere_visuals_enabled:
            sphere_cfg = sim_utils.SphereCfg(
                radius=self.cfg.target_sphere_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.cfg.target_sphere_color
                ),
                rigid_props=None,
                collision_props=None,
            )
            sim_utils.spawn_sphere(
                prim_path="/World/envs/env_0/TargetSphere",
                cfg=sphere_cfg,
                translation=(0.0, 0.0, 2.0),
            )

        self._vision_camera = None

        # 先 clone 所有环境
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # ----------------------------------------------------------
        # 视觉相机：这里只创建 proxy camera prim，不在这里直接创建 TiledCamera
        # ----------------------------------------------------------
        if self.cfg.enable_vision_supervision:
            camera_matches = sim_utils.find_matching_prim_paths(self.cfg.camera_prim_path_expr)
            if len(camera_matches) == 0:
                raise RuntimeError(
                    f"[vision] No real camera prim matched after clone: {self.cfg.camera_prim_path_expr}"
                )

            print(f"[vision] matched {len(camera_matches)} real camera prim(s) after clone")
            if self.cfg.debug_diagnostics:
                for p in camera_matches[: min(8, len(camera_matches))]:
                    print(f"[vision] real camera prim: {p}")

            # 创建 runtime proxy cameras，后续 TiledCamera 绑定到这些 proxy 上
            self._ensure_runtime_camera_proxies()

            # 这里先不创建 TiledCamera，等 super().__init__() 完成后
            # 在 bootstrap_vision_camera() 里再延迟创建
            self._vision_camera = None

            if self.cfg.debug_diagnostics:
                print(
                    "[vision] runtime proxy cameras created in _setup_scene(), "
                    f"real_camera_count={len(camera_matches)}, "
                    f"proxy_count={len(self._runtime_camera_proxy_paths)}"
                )

        # dynamic target raycast: 直接挂到 TORSO，不再创建额外 mount
        if self.cfg.enable_dynamic_target_raycast_debug:
            self._create_dynamic_target_multimesh_raycast_sensor()

        # bind cloned target spheres
        if self._target_sphere_visuals_enabled:
            self.target_sphere = True
            self._target_sphere_prim_paths = [
                f"/World/envs/env_{env_id}/TargetSphere"
                for env_id in range(self.num_envs)
            ]
        else:
            self.target_sphere = None
            self._target_sphere_prim_paths = []


        # resolve turret prim path expression
        turret_prim_path_expr = self._resolve_turret_prim_path_expr()
        print("[TurretAimEnv] resolved turret_prim_path_expr =", turret_prim_path_expr)

        # attach turret articulation
        turret_cfg = ArticulationCfg(
            prim_path=turret_prim_path_expr,
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.cfg.base_init_pos,
                joint_pos={
                    ".*_HipX_joint": 0.0,
                    ".*_HipY_joint": -0.8,
                    ".*_Knee_joint": 1.6,
                    self.cfg.yaw_joint_name: 0.0,
                    self.cfg.pitch_joint_name: 0.0,
                },
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.99,
            actuators={
                "Hip": DelayedPDActuatorCfg(
                    joint_names_expr=[".*_HipX_joint", ".*_HipY_joint"],
                    effort_limit=24.0,
                    velocity_limit=26.2,
                    stiffness=30.0,
                    damping=1.0,
                    friction=0.0,
                    armature=0.0,
                    min_delay=0,
                    max_delay=5,
                ),
                "Knee": DelayedPDActuatorCfg(
                    joint_names_expr=[".*_Knee_joint"],
                    effort_limit=36.0,
                    velocity_limit=17.3,
                    stiffness=30.0,
                    damping=1.0,
                    friction=0.0,
                    armature=0.0,
                    min_delay=0,
                    max_delay=5,
                ),
                "turret": ImplicitActuatorCfg(
                    joint_names_expr=[self.cfg.yaw_joint_name, self.cfg.pitch_joint_name],
                    effort_limit_sim=None,
                    velocity_limit_sim=None,
                    stiffness=None,
                    damping=None,
                ),
            },
        )
        self.turret = Articulation(turret_cfg)
        self.scene.articulations["turret"] = self.turret

        # 添加全局环境光，避免相机导出的图像发黑
        light_cfg = sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # ---------------------------------------------------------
        # 在物理步之前，处理 reset 后延迟的 target 正式采样
        # 这样后续 render 出来的 rgb、target 状态、label 才是同一时刻的
        # ---------------------------------------------------------
        self._vision_supervision_refreshed_this_step = False
        self._maybe_finalize_pending_target_resample()

        # ----------------------------
        # 1. 检查动作有效性
        # ----------------------------
        invalid_actions = ~torch.isfinite(actions).all(dim=-1)
        if torch.any(invalid_actions):
            raise RuntimeError("Turret policy produced non-finite actions.")

        actions = actions.clamp(-1.0, 1.0)
        self._invalidate_ballistic_cache()

        # 保存上一时刻“策略原始动作”
        # PPO / A2C 训练时，环境内部不要再对动作做 EMA 平滑，
        # 否则实际执行动作与策略计算 log-prob 的动作不一致。
        self._prev_actions.copy_(self._actions)
        self._actions.copy_(actions)

        # warm-up: disable aiming/firing before robot settles
        aiming_enabled = self._aim_warmup_left_steps <= 0
        aiming_scale = aiming_enabled.float()

        # ----------------------------
        # 2. yaw/pitch 位置增量控制
        # ----------------------------
        self._delta_pos[:, 0] = self._actions[:, 0] * self.cfg.max_delta_yaw * aiming_scale
        self._delta_pos[:, 1] = self._actions[:, 1] * self.cfg.max_delta_pitch * aiming_scale
        self._turret_pos_targets[:, 0] += self._delta_pos[:, 0]
        self._turret_pos_targets[:, 1] += self._delta_pos[:, 1]

        # -------------------------------------------------
        # eval action debug：用于分析“炮台为什么不转”
        # 注意：
        # - joint_pos 只作为关节状态打印
        # - 真正的瞄准误差使用真实枪口 forward 在 yaw-only base 系下的角度
        # - reference 使用和 reward 一致的 muzzle-based aim refs
        # -------------------------------------------------
        if self.cfg.debug_diagnostics and getattr(self, "_debug_print_counter", 0) < 50:
            try:
                # 当前真实枪口方向角：yaw-only base frame
                muzzle_yaw_base, muzzle_pitch_base = self._compute_muzzle_actual_yaw_pitch_in_yaw_only_base()

                # reward 里使用的参考角：以枪口点为原点，yaw-only base frame
                yaw_ref, pitch_los_ref, pitch_g_ref = self._compute_muzzle_based_aim_refs()

                # 和 reward 一致的 yaw 误差
                yaw_err = wrap_to_pi(yaw_ref - muzzle_yaw_base)

                # pitch reward 是一个区间奖励，不是单点误差
                pitch_low = torch.minimum(pitch_los_ref, pitch_g_ref) - 0.01
                pitch_high = torch.maximum(pitch_los_ref, pitch_g_ref) + 0.04

                pitch_under = torch.clamp(pitch_low - muzzle_pitch_base, min=0.0)
                pitch_over = torch.clamp(muzzle_pitch_base - pitch_high, min=0.0)
                pitch_band_excess = pitch_under + pitch_over

                # joint angle 仍然打印，但只用于看执行器状态，不再当作瞄准误差
                turret_pos = self.turret.data.joint_pos
                yaw_pos = float(turret_pos[0, int(self._yaw_dof_ids[0])].detach().cpu().item())
                pitch_pos = float(turret_pos[0, int(self._pitch_dof_ids[0])].detach().cpu().item())

                yaw_tgt = float(self._turret_pos_targets[0, 0].detach().cpu().item())
                pitch_tgt = float(self._turret_pos_targets[0, 1].detach().cpu().item())

                yaw_delta = float(self._delta_pos[0, 0].detach().cpu().item())
                pitch_delta = float(self._delta_pos[0, 1].detach().cpu().item())

                act0 = float(self._actions[0, 0].detach().cpu().item())
                act1 = float(self._actions[0, 1].detach().cpu().item())
                act2 = float(self._actions[0, 2].detach().cpu().item())

                muzzle_yaw_0 = float(muzzle_yaw_base[0].detach().cpu().item())
                muzzle_pitch_0 = float(muzzle_pitch_base[0].detach().cpu().item())

                yaw_ref_0 = float(yaw_ref[0].detach().cpu().item())
                pitch_los_ref_0 = float(pitch_los_ref[0].detach().cpu().item())
                pitch_g_ref_0 = float(pitch_g_ref[0].detach().cpu().item())

                pitch_low_0 = float(pitch_low[0].detach().cpu().item())
                pitch_high_0 = float(pitch_high[0].detach().cpu().item())

                yaw_err_0 = float(yaw_err[0].detach().cpu().item())
                pitch_band_excess_0 = float(pitch_band_excess[0].detach().cpu().item())

                pred_ready_0 = bool(self._rl_pred_ready[0].detach().cpu().item())
                warmup_left_0 = int(self._aim_warmup_left_steps[0].detach().cpu().item())
                aiming_enabled_0 = bool((self._aim_warmup_left_steps[0] <= 0).detach().cpu().item())

                print(
                    "[eval action debug]",
                    f"step={int(self.common_step_counter)}",
                    f"actions[0]=[{act0:.4f}, {act1:.4f}, {act2:.4f}]",
                    f"warmup_left={warmup_left_0}",
                    f"aiming_enabled={aiming_enabled_0}",
                    f"pred_ready={pred_ready_0}",
                    f"delta=[{yaw_delta:.4f}, {pitch_delta:.4f}]",
                    f"joint_pos=[{yaw_pos:.4f}, {pitch_pos:.4f}]",
                    f"joint_tgt=[{yaw_tgt:.4f}, {pitch_tgt:.4f}]",
                    f"muzzle_actual=[{muzzle_yaw_0:.4f}, {muzzle_pitch_0:.4f}]",
                    f"muzzle_ref=[yaw={yaw_ref_0:.4f}, pitch_los={pitch_los_ref_0:.4f}, pitch_g={pitch_g_ref_0:.4f}]",
                    f"pitch_band=[{pitch_low_0:.4f}, {pitch_high_0:.4f}]",
                    f"err_muzzle=[yaw={yaw_err_0:.4f}, pitch_band_excess={pitch_band_excess_0:.4f}]",
                )

                self._debug_print_counter = getattr(self, "_debug_print_counter", 0) + 1

            except Exception as e:
                print(f"[eval action debug] print failed: {e}")

        # Clamp against the current per-environment limits parsed from the USD.
        turret_joint_limits = self.turret.data.joint_pos_limits[:, self._turret_dof_ids, :]
        self._turret_pos_targets.copy_(
            torch.maximum(
                torch.minimum(self._turret_pos_targets, turret_joint_limits[..., 1]),
                turret_joint_limits[..., 0],
            )
        )

        # ----------------------------
        # 3. fire intent -> two-step confirmed direct fire
        #
        # 第3维动作直接表示当前开火意愿：
        #   action[2] ∈ [-1, 1]
        #   fire_intent = (action[2] + 1) / 2 ∈ [0, 1]
        #
        # 开火条件：
        #   1) GT-only 下 pred_ready 永远为 True；非 GT 下仍等待 bridge ready
        #   2) warmup 结束
        #   3) cooldown 结束
        #   4) fire_intent 连续 fire_intent_hold_steps 步大于 fire_intent_threshold
        #
        # 注意：
        #   这不是 rearm 机制，不要求 intent 先降下来。
        #   如果 intent 一直高，cooldown 结束后仍然可以再次开火。
        # ----------------------------
        fire_intent = ((self._actions[:, 2] + 1.0) * 0.5).clamp(0.0, 1.0)

        if bool(getattr(self, "_gt_only_mode", False)):
            pred_ready = torch.ones_like(self._rl_pred_ready, dtype=torch.bool)
        else:
            pred_ready = self._rl_pred_ready
            if hasattr(self, "_rl_bridge_training_ready"):
                pred_ready = pred_ready & self._rl_bridge_training_ready

        aiming_enabled = self._aim_warmup_left_steps <= 0
        can_fire = self._fire_cooldown <= 0

        # 连续 intent 确认，过滤单步动作抖动
        #
        # 关键修改：
        #   warmup 没结束时，不允许累计 _fire_intent_high_steps；
        #   pred_ready 没准备好时，也不允许累计。
        #
        # 这样可以避免：
        #   Actor 在 warmup 期间一直输出高 fire_intent，
        #   导致 warmup 结束第一帧瞬间开火。
        #
        # 注意：
        #   不把 can_fire 加进 fire_count_enabled。
        #   否则 cooldown 期间会清空计数，导致每次 cooldown 后还要额外等 hold_steps。
        fire_count_enabled = pred_ready & aiming_enabled
        fire_high = (fire_intent > float(self.cfg.fire_intent_threshold)) & fire_count_enabled

        self._fire_intent_high_steps = torch.where(
            fire_high,
            torch.clamp(
                self._fire_intent_high_steps + 1,
                max=int(self.cfg.fire_intent_hold_steps),
            ),
            torch.zeros_like(self._fire_intent_high_steps),
        )

        intent_confirmed = self._fire_intent_high_steps >= int(self.cfg.fire_intent_hold_steps)

        fire_now = pred_ready & aiming_enabled & can_fire & intent_confirmed

        self._fire_command = fire_now.float()
        self._eval_total_fires += torch.count_nonzero(fire_now)
        self._episode_fire_count += fire_now.long()

        # Keep the current intent in the existing one-dimensional observation slot.
        self._fire_intent.copy_(fire_intent)

        # 更新 cooldown
        self._fire_cooldown = torch.clamp(self._fire_cooldown - 1, min=0)
        self._fire_cooldown[fire_now] = self.cfg.fire_cooldown_steps

        # 更新 warmup 计时
        self._aim_warmup_left_steps = torch.clamp(self._aim_warmup_left_steps - 1, min=0)

        # stage-2/3 locomotion policy drives leg joints (moving-base training)
        if self._locomotion_policy is not None:
            self._update_locomotion_commands()
            locomotion_obs = self._compute_locomotion_observations()
            with torch.no_grad():
                locomotion_actions = self._locomotion_policy(locomotion_obs)
            locomotion_actions = locomotion_actions.to(device=self.device, dtype=torch.float32)
            locomotion_actions = locomotion_actions.clamp(-100.0, 100.0)
            self._locomotion_last_actions.copy_(locomotion_actions)
            self._locomotion_joint_targets.copy_(self._default_leg_joint_pos + self._leg_action_scale * locomotion_actions)

        if self.cfg.enable_dynamic_target:
            target_motion_mask = self._target_motion_mask
            target_motion_ids = torch.nonzero(target_motion_mask, as_tuple=False).squeeze(-1)
            if target_motion_ids.numel() > 0:
                target_vel = self._target_vel_w[target_motion_ids]
                target_acc = self._target_acc_w[target_motion_ids]
                target_vel = target_vel + target_acc * self.step_dt

                self._target_vel_w[target_motion_ids] = target_vel
                self._target_pos_w[target_motion_ids] += target_vel * self.step_dt

                # Apply the configured distance/FOV constraint to the selected
                # moving targets. The mask decides whether the reference anchor
                # is fixed or follows the current robot pose.
                constraint_mask = (
                    self._target_relative_constraint_mask & target_motion_mask
                )
                constraint_ids = torch.nonzero(constraint_mask, as_tuple=False).squeeze(-1)
                if constraint_ids.numel() > 0:
                    fixed_anchor_mask = self._target_constraint_fixed_anchor_mask[constraint_ids]
                    fixed_ids = constraint_ids[fixed_anchor_mask]
                    moving_anchor_ids = constraint_ids[~fixed_anchor_mask]
                    if fixed_ids.numel() > 0:
                        self._handle_target_out_of_bounds(fixed_ids, fixed_anchor=True)
                    if moving_anchor_ids.numel() > 0:
                        self._handle_target_out_of_bounds(moving_anchor_ids, fixed_anchor=False)

                # Ground clearance is the only unconditional height constraint.
                # Camera yaw/pitch constraints above keep selected targets in view.
                self._enforce_target_above_ground(target_motion_ids)

        if self._target_sphere_visuals_enabled:
            self._update_rendered_target_spheres()

        # 注意：
        # 这里不再自动调用 _debug_raycast_dynamic_target()。
        # 当前帧的 raycast 后处理应当只在外部已经拿到“当前帧视觉预测角度”之后，
        # 通过 run_dynamic_target_raycast_for_current_frame() 显式触发，
        # 避免 1 step 内重复调用多次、并避免用旧角度筛当前帧雷达。
        if self._target_markers is not None and self._target_markers.is_visible():
            translations = self._target_pos_w.detach().cpu().numpy()
            self._target_markers.visualize(translations=translations)


    def refresh_vision_supervision(self) -> None:
        """Explicitly refresh vision buffers once after env.step()."""
        if not self.cfg.enable_vision_supervision:
            self._vision_supervision_refreshed_this_step = True
            return
        if self._vision_supervision_refreshed_this_step:
            return

        # 只消耗 reset 后的额外等待步数
        self._vision_collect_warmup_left = torch.clamp(self._vision_collect_warmup_left - 1, min=0)

        # 这里只刷新当前 step 对应的视觉缓存，不在这里做 target resample
        self._refresh_vision_supervision()
        self._vision_supervision_refreshed_this_step = True


    def _ensure_vision_supervision_ready(self) -> None:
        """供 rl_perception_bridge 调用，保证每个 env step 最多刷新一次视觉监督缓存。"""
        if not self._vision_supervision_refreshed_this_step:
            self.refresh_vision_supervision()



    def set_rl_predicted_target_state(
        self,
        pred_pos_base: torch.Tensor,
        pred_vel_base: torch.Tensor,
        pred_distance: torch.Tensor,
        pred_yaw_base: torch.Tensor,
        pred_pitch_base: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """由 rl_perception_bridge 写回当前 step 的预测目标状态。"""
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)

        if env_ids_t.numel() == 0:
            return

        self._rl_pred_pos_base[env_ids_t] = pred_pos_base[env_ids_t].to(self.device, dtype=torch.float32)
        self._rl_pred_vel_base[env_ids_t] = pred_vel_base[env_ids_t].to(self.device, dtype=torch.float32)

        pd = pred_distance
        if pd.ndim != 1:
            pd = pd.squeeze(-1)
        self._rl_pred_distance[env_ids_t] = pd[env_ids_t].to(self.device, dtype=torch.float32)

        py = pred_yaw_base
        if py.ndim != 1:
            py = py.squeeze(-1)
        self._rl_pred_yaw_base[env_ids_t] = py[env_ids_t].to(self.device, dtype=torch.float32)

        pp = pred_pitch_base
        if pp.ndim != 1:
            pp = pp.squeeze(-1)
        self._rl_pred_pitch_base[env_ids_t] = pp[env_ids_t].to(self.device, dtype=torch.float32)

        self._rl_pred_ready[env_ids_t] = True


    def _apply_action(self) -> None:
        dof_ids = [int(self._yaw_dof_ids[0]), int(self._pitch_dof_ids[0])]
        self.turret.set_joint_position_target(self._turret_pos_targets, joint_ids=dof_ids)
        self.turret.set_joint_position_target(self._locomotion_joint_targets, joint_ids=self._leg_dof_ids)


    def _load_locomotion_policy(self):
        if not self.cfg.enable_locomotion_policy:
            return None
        if not self.cfg.locomotion_policy_path:
            raise ValueError(
                "enable_locomotion_policy=True but locomotion_policy_path is not set. "
                "Pass an exported RSL-RL TorchScript policy.pt path."
            )
        if not os.path.exists(self.cfg.locomotion_policy_path):
            raise FileNotFoundError(f"Locomotion policy not found: {self.cfg.locomotion_policy_path}")
        policy = torch.jit.load(self.cfg.locomotion_policy_path, map_location=self.device)
        policy.eval()
        return policy

    def _resolve_leg_joint_ids(self, joint_names: list[str]) -> list[int]:
        joint_ids: list[int] = []
        resolved_names: list[str] = []
        for joint_name in joint_names:
            dof_ids, matched_names = self.turret.find_joints(joint_name)
            if len(dof_ids) != 1:
                raise RuntimeError(
                    f"Leg joint '{joint_name}' must resolve to exactly 1 joint. Got {matched_names}."
                )
            joint_ids.append(int(dof_ids[0]))
            resolved_names.append(matched_names[0])
        if resolved_names != joint_names:
            raise RuntimeError(f"Resolved leg joint order mismatch: expected {joint_names}, got {resolved_names}")
        return joint_ids

    def _update_locomotion_commands(self) -> None:
        self._locomotion_command_time_left -= self.step_dt
        env_ids_t = torch.nonzero(self._locomotion_command_time_left <= 0.0, as_tuple=False).squeeze(-1)
        if env_ids_t.numel() == 0:
            return
        self._resample_locomotion_commands(env_ids_t)

    def _resample_locomotion_commands(self, env_ids_t: torch.Tensor) -> None:
        env_count = int(env_ids_t.numel())
        self._locomotion_commands[env_ids_t, 0] = sample_uniform(
            self.cfg.locomotion_lin_vel_x_range[0],
            self.cfg.locomotion_lin_vel_x_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)
        self._locomotion_commands[env_ids_t, 1] = sample_uniform(
            self.cfg.locomotion_lin_vel_y_range[0],
            self.cfg.locomotion_lin_vel_y_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)
        self._locomotion_commands[env_ids_t, 2] = sample_uniform(
            self.cfg.locomotion_ang_vel_z_range[0],
            self.cfg.locomotion_ang_vel_z_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)
        self._locomotion_command_time_left[env_ids_t] = self.cfg.locomotion_command_resample_s

    def _compute_locomotion_observations(self) -> torch.Tensor:
        leg_joint_pos = self.turret.data.joint_pos[:, self._leg_dof_ids]
        leg_joint_vel = self.turret.data.joint_vel[:, self._leg_dof_ids]
        leg_joint_pos_rel = leg_joint_pos - self.turret.data.default_joint_pos[:, self._leg_dof_ids]
        obs = torch.cat(
            [
                self.turret.data.root_ang_vel_b * 0.25,
                self.turret.data.projected_gravity_b,
                self._locomotion_commands,
                leg_joint_pos_rel,
                leg_joint_vel * 0.05,
                self._locomotion_last_actions,
            ],
            dim=-1,
        )
        # return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _report_invalid_envs(self, stage: str, invalid_mask: torch.Tensor, extra_tensors: dict[str, torch.Tensor]) -> None:
        if not self.cfg.debug_diagnostics:
            return
        invalid_ids = torch.nonzero(invalid_mask, as_tuple=False).squeeze(-1)
        if invalid_ids.numel() == 0:
            return
        sample_id = int(invalid_ids[0].item())
        print(f"[TurretAimEnv][debug] stage={stage} invalid_envs={int(invalid_ids.numel())} sample_env={sample_id}")
        for name, tensor in extra_tensors.items():
            if tensor.ndim == 0:
                value = tensor.detach().cpu().item()
            else:
                value = tensor[sample_id].detach().cpu().tolist()
            print(f"[TurretAimEnv][debug]   {name}={value}")

    def _get_invalid_state_mask(self) -> torch.Tensor:
        checks = [
            self.turret.data.root_pos_w,
            self.turret.data.root_lin_vel_b,
            self.turret.data.root_ang_vel_b,
            self.turret.data.projected_gravity_b,
            self.turret.data.joint_pos,
            self.turret.data.joint_vel,
            self._target_pos_w,
            self._target_vel_w,
            self._locomotion_commands,
            self._locomotion_last_actions,
            self._actions,
            self._prev_actions,
            self._delta_pos,
            self._turret_pos_targets,
            self._locomotion_joint_targets,
            self._fire_command.unsqueeze(-1),
            self._aim_warmup_left_steps.unsqueeze(-1).float(),
        ]
        invalid_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        for tensor in checks:
            invalid_mask |= ~torch.isfinite(tensor).all(dim=-1)
        return invalid_mask
    

    def _compute_muzzle_velocity_w(
        self,
        body_pos_w: torch.Tensor,
        muzzle_pos_w: torch.Tensor,
    ) -> torch.Tensor:
        """Compute muzzle world linear velocity.

        优先使用 muzzle body 原点的线速度 + 角速度叉乘局部偏移；
        如果 body 级速度不可用，则回退到 root 线速度 + root 角速度叉乘项。
        """
        # offset from body origin to muzzle tip in world frame
        offset_w = muzzle_pos_w - body_pos_w

        # 优先使用 body 级速度（如果指定了 muzzle body）
        if self._muzzle_body_id is not None:
            body_lin_vel_w = self.turret.data.body_lin_vel_w[:, self._muzzle_body_id]
            body_ang_vel_w = self.turret.data.body_ang_vel_w[:, self._muzzle_body_id]
            return body_lin_vel_w + torch.cross(body_ang_vel_w, offset_w, dim=-1)

        # 回退：root 线速度 + root 角速度叉乘项
        root_quat_w = self.turret.data.root_quat_w
        root_lin_vel_w = quat_apply(root_quat_w, self.turret.data.root_lin_vel_b)
        root_ang_vel_w = quat_apply(root_quat_w, self.turret.data.root_ang_vel_b)
        return root_lin_vel_w + torch.cross(root_ang_vel_w, offset_w, dim=-1)

    # ------------------------------------------------------------------
    # vision supervision helpers
    # ------------------------------------------------------------------
    def _quat_conjugate(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        quat_conj = quat_wxyz.clone()
        quat_conj[..., 1:] *= -1.0
        return quat_conj

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiply in wxyz convention."""
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def _rotate_world_to_local(self, quat_wxyz: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
        """Rotate world-frame vector into local frame using inverse quaternion."""
        quat_conj = self._quat_conjugate(quat_wxyz)
        return quat_apply(quat_conj, vec_w)

    def _rotate_local_to_world(self, quat_wxyz: torch.Tensor, vec_local: torch.Tensor) -> torch.Tensor:
        """Rotate local-frame vector into world frame."""
        return quat_apply(quat_wxyz, vec_local)

    def _get_yaw_only_quat(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        """
        从完整 root 姿态中提取 yaw-only 四元数：
        - z 轴始终和世界竖直方向一致
        - x/y 只跟随 yaw 旋转
        - 不包含 roll / pitch
        """
        basis_x = torch.zeros((quat_wxyz.shape[0], 3), device=quat_wxyz.device, dtype=quat_wxyz.dtype)
        basis_x[:, 0] = 1.0

        forward_w = quat_apply(quat_wxyz, basis_x)  # 机体 x 轴在世界系中的方向
        forward_xy = forward_w[:, :2]
        forward_xy_norm = torch.norm(forward_xy, dim=-1, keepdim=True).clamp_min(1e-8)
        forward_xy = forward_xy / forward_xy_norm

        yaw = torch.atan2(forward_xy[:, 1], forward_xy[:, 0])

        half_yaw = 0.5 * yaw
        q = torch.zeros_like(quat_wxyz)
        q[:, 0] = torch.cos(half_yaw)   # w
        q[:, 1] = 0.0                   # x
        q[:, 2] = 0.0                   # y
        q[:, 3] = torch.sin(half_yaw)   # z
        return q

    def _rotate_world_to_yaw_only_local(self, root_quat_w: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
        yaw_quat_w = self._get_yaw_only_quat(root_quat_w)
        return self._rotate_world_to_local(yaw_quat_w, vec_w)

    def _rotate_yaw_only_local_to_world(self, root_quat_w: torch.Tensor, vec_local: torch.Tensor) -> torch.Tensor:
        yaw_quat_w = self._get_yaw_only_quat(root_quat_w)
        return self._rotate_local_to_world(yaw_quat_w, vec_local)

    def _ensure_camera_extrinsics_from_torso(self):
        """只从 USD 读取一次 camera 相对 TORSO 的固定外参。

        注意：
        - 后续任何 runtime 相机世界位姿，都由
        当前 TORSO body pose + 这里缓存的固定外参
        重建得到。
        - 不再使用 self._vision_camera.data.pos_w / quat_w_ros
        去反推外参，因为那一路数据在当前工程里不可靠。
        """
        if hasattr(self, "_camera_local_pos_in_torso") and hasattr(self, "_camera_local_quat_in_torso"):
            return

        if self._vision_camera is None and not self.cfg.enable_vision_supervision:
            raise RuntimeError("Vision camera is disabled; cannot resolve camera extrinsics.")

        if not hasattr(self, "_torso_body_id") or self._torso_body_id is None:
            torso_body_ids, _ = self.turret.find_bodies("TORSO")
            if len(torso_body_ids) == 0:
                raise RuntimeError(f"Could not resolve TORSO body. Available bodies: {self.turret.body_names}")
            self._torso_body_id = int(torso_body_ids[0])

        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom
        import torch

        stage = get_current_stage()
        if stage is None:
            raise RuntimeError("USD stage is None, cannot resolve camera extrinsics from USD.")

        camera_matches = sim_utils.find_matching_prim_paths(self.cfg.camera_prim_path_expr)
        torso_matches = sim_utils.find_matching_prim_paths(self.cfg.turret_prim_path_expr)

        if len(camera_matches) == 0:
            raise RuntimeError(f"No camera prim matched: {self.cfg.camera_prim_path_expr}")
        if len(torso_matches) == 0:
            raise RuntimeError(f"No TORSO prim matched: {self.cfg.turret_prim_path_expr}")

        camera_prim_path = next((p for p in camera_matches if "/env_0/" in p), camera_matches[0])
        torso_prim_path = next((p for p in torso_matches if "/env_0/" in p), torso_matches[0])

        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        torso_prim = stage.GetPrimAtPath(torso_prim_path)

        if not camera_prim.IsValid():
            raise RuntimeError(f"Invalid camera prim: {camera_prim_path}")
        if not torso_prim.IsValid():
            raise RuntimeError(f"Invalid torso prim: {torso_prim_path}")

        camera_xf = UsdGeom.Xformable(camera_prim)
        torso_xf = UsdGeom.Xformable(torso_prim)

        cam_world_mat = camera_xf.ComputeLocalToWorldTransform(0.0)
        torso_world_mat = torso_xf.ComputeLocalToWorldTransform(0.0)

        # 相对旋转：可直接由相对矩阵提取
        rel_mat = cam_world_mat * torso_world_mat.GetInverse()
        rel_rot = rel_mat.ExtractRotationQuat()

        cam_local_quat_single = torch.tensor(
            [
                float(rel_rot.GetReal()),
                float(rel_rot.GetImaginary()[0]),
                float(rel_rot.GetImaginary()[1]),
                float(rel_rot.GetImaginary()[2]),
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        cam_local_quat_single = cam_local_quat_single / torch.norm(
            cam_local_quat_single, dim=-1, keepdim=True
        ).clamp_min(1e-12)

        # 相对平移不要直接用 rel_mat.ExtractTranslation()
        # 用 R_torso^T * (p_cam - p_torso) 更稳，不会把世界平移项带进来
        cam_world_pos = cam_world_mat.ExtractTranslation()
        torso_world_pos = torso_world_mat.ExtractTranslation()

        cam_world_pos_t = torch.tensor(
            [float(cam_world_pos[0]), float(cam_world_pos[1]), float(cam_world_pos[2])],
            dtype=torch.float32,
            device=self.device,
        )
        torso_world_pos_t = torch.tensor(
            [float(torso_world_pos[0]), float(torso_world_pos[1]), float(torso_world_pos[2])],
            dtype=torch.float32,
            device=self.device,
        )

        torso_world_quat = torso_world_mat.ExtractRotationQuat()
        torso_world_quat_t = torch.tensor(
            [
                float(torso_world_quat.GetReal()),
                float(torso_world_quat.GetImaginary()[0]),
                float(torso_world_quat.GetImaginary()[1]),
                float(torso_world_quat.GetImaginary()[2]),
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        torso_world_quat_t = torso_world_quat_t / torch.norm(
            torso_world_quat_t, dim=-1, keepdim=True
        ).clamp_min(1e-12)

        cam_offset_world_t = (cam_world_pos_t - torso_world_pos_t).unsqueeze(0)
        cam_local_pos_single = self._rotate_world_to_local(
            torso_world_quat_t, cam_offset_world_t
        ).squeeze(0).unsqueeze(0)

        self._camera_local_pos_in_torso = cam_local_pos_single.repeat(self.num_envs, 1)
        self._camera_local_quat_in_torso = cam_local_quat_single.repeat(self.num_envs, 1)

        if self.cfg.debug_diagnostics:
            print(
                "[camera extrinsics init]",
                f"camera_prim_path={camera_prim_path}",
                f"torso_prim_path={torso_prim_path}",
                f"local_pos={self._camera_local_pos_in_torso[0].detach().cpu().tolist()}",
                f"local_quat={self._camera_local_quat_in_torso[0].detach().cpu().tolist()}",
            )


    def _get_runtime_camera_pose_w_usd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """由当前 TORSO runtime pose + 固定外参，重建相机世界位姿（USD camera frame）。"""
        self._ensure_camera_extrinsics_from_torso()

        if not hasattr(self, "_torso_body_id") or self._torso_body_id is None:
            torso_body_ids, _ = self.turret.find_bodies("TORSO")
            if len(torso_body_ids) == 0:
                raise RuntimeError(f"Could not resolve TORSO body. Available bodies: {self.turret.body_names}")
            self._torso_body_id = int(torso_body_ids[0])

        if hasattr(self.turret.data, "body_pos_w") and hasattr(self.turret.data, "body_quat_w"):
            torso_pos_w = self.turret.data.body_pos_w[:, self._torso_body_id]
            torso_quat_w = self.turret.data.body_quat_w[:, self._torso_body_id]
        else:
            torso_pos_w = self.turret.data.root_pos_w
            torso_quat_w = self.turret.data.root_quat_w

        cam_local_pos = self._camera_local_pos_in_torso
        cam_local_quat = self._camera_local_quat_in_torso

        if cam_local_pos.shape[0] != torso_pos_w.shape[0]:
            cam_local_pos = cam_local_pos[:1].repeat(torso_pos_w.shape[0], 1)
        if cam_local_quat.shape[0] != torso_quat_w.shape[0]:
            cam_local_quat = cam_local_quat[:1].repeat(torso_quat_w.shape[0], 1)

        cam_pos_w = torso_pos_w + self._rotate_local_to_world(torso_quat_w, cam_local_pos)
        cam_quat_w_usd = self._quat_mul(torso_quat_w, cam_local_quat)
        cam_quat_w_usd = cam_quat_w_usd / torch.norm(cam_quat_w_usd, dim=-1, keepdim=True).clamp_min(1e-12)

        return cam_pos_w, cam_quat_w_usd


    def _get_runtime_camera_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ROS optical frame 下的 runtime 相机世界位姿。

        这个才是几何计算 / 标签 / raycast 应该统一使用的相机姿态。
        """
        cam_pos_w, cam_quat_w_usd = self._get_runtime_camera_pose_w_usd()

        q_usd_to_ros = torch.zeros_like(cam_quat_w_usd)
        q_usd_to_ros[:, 1] = 1.0  # [0, 1, 0, 0]

        cam_quat_w = self._quat_mul(cam_quat_w_usd, q_usd_to_ros)
        cam_quat_w = cam_quat_w / torch.norm(cam_quat_w, dim=-1, keepdim=True).clamp_min(1e-12)

        return cam_pos_w, cam_quat_w


    def _estimate_red_blob_center_single(
        self,
        rgb_chw: torch.Tensor,
    ) -> tuple[bool, float, float, float, int]:
        """Estimate red target center from one RGB frame [3,H,W] in normalized [0,1] space."""
        if rgb_chw.ndim != 3 or rgb_chw.shape[0] < 3:
            return False, 0.0, 0.0, 0.0, 0

        r = rgb_chw[0]
        g = rgb_chw[1]
        b = rgb_chw[2]

        # red dominance score
        score = torch.clamp(r - torch.maximum(g, b), min=0.0)
        mask = (score > 0.08) & (r > 0.20)

        if int(mask.sum().item()) <= 0:
            return False, 0.0, 0.0, 0.0, 0

        w_map = score * mask.to(dtype=score.dtype)
        sum_w = float(w_map.sum().item())
        if sum_w <= 1e-8:
            return False, 0.0, 0.0, 0.0, 0

        h, w = int(r.shape[0]), int(r.shape[1])
        u_axis = torch.arange(w, device=rgb_chw.device, dtype=rgb_chw.dtype)
        v_axis = torch.arange(h, device=rgb_chw.device, dtype=rgb_chw.dtype)

        u = float((w_map.sum(dim=0) * u_axis).sum().item() / sum_w)
        v = float((w_map.sum(dim=1) * v_axis).sum().item() / sum_w)
        return True, u, v, sum_w, int(mask.sum().item())


    def _compute_camera_direction_labels(
        self,
        cam_pos_w: torch.Tensor | None = None,
        cam_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [num_envs, 2] -> [yaw_cam, pitch_cam]."""
        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized.")

        if cam_pos_w is None or cam_quat_w is None:
            cam_pos_w, cam_quat_w = self._get_runtime_camera_pose_w()

        to_target_w = self._target_pos_w - cam_pos_w
        to_target_cam = self._rotate_world_to_local(cam_quat_w, to_target_w)

        x = to_target_cam[:, 0]
        y = to_target_cam[:, 1]
        z = to_target_cam[:, 2].clamp_min(1e-6)

        yaw_cam = torch.atan2(x, z)
        pitch_cam = torch.atan2(-y, torch.sqrt(x * x + z * z).clamp_min(1e-6))
        return torch.stack([yaw_cam, pitch_cam], dim=-1)

    def _compute_camera_velocity_labels(
        self,
        cam_pos_w: torch.Tensor | None = None,
        cam_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [num_envs, 3] -> target relative velocity expressed in camera frame."""
        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized.")

        if cam_pos_w is None or cam_quat_w is None:
            cam_pos_w, cam_quat_w = self._get_runtime_camera_pose_w()

        root_quat_w = self.turret.data.root_quat_w
        root_lin_vel_w = quat_apply(root_quat_w, self.turret.data.root_lin_vel_b)
        root_ang_vel_w = quat_apply(root_quat_w, self.turret.data.root_ang_vel_b)

        root_pos_w = self.turret.data.root_pos_w
        cam_offset_w = cam_pos_w - root_pos_w

        cam_lin_vel_w = root_lin_vel_w + torch.cross(root_ang_vel_w, cam_offset_w, dim=-1)
        target_rel_vel_w = self._target_vel_w - cam_lin_vel_w
        target_rel_vel_cam = self._rotate_world_to_local(cam_quat_w, target_rel_vel_w)
        return target_rel_vel_cam

    def _compute_target_projected_in_image_mask(
        self,
        cam_pos_w: torch.Tensor | None = None,
        cam_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return env mask where the rendered target sphere is truly visible in image."""
        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized.")

        if cam_pos_w is None or cam_quat_w is None:
            cam_pos_w, cam_quat_w = self._get_runtime_camera_pose_w()

        to_target_w = self._target_pos_w - cam_pos_w
        to_target_cam = self._rotate_world_to_local(cam_quat_w, to_target_w)

        x = to_target_cam[:, 0]
        y = to_target_cam[:, 1]
        z = to_target_cam[:, 2]

        K = self._vision_camera.data.intrinsic_matrices
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]

        z_safe = z.clamp_min(1e-6)
        u = fx * (x / z_safe) + cx
        v = fy * (y / z_safe) + cy

        sphere_radius_m = float(self.cfg.target_sphere_radius)
        r_pix_x = fx * sphere_radius_m / z_safe
        r_pix_y = fy * sphere_radius_m / z_safe
        r_pix = torch.minimum(r_pix_x, r_pix_y)

        min_radius_pix = 2.0

        finite = (
            torch.isfinite(u)
            & torch.isfinite(v)
            & torch.isfinite(z)
            & torch.isfinite(r_pix)
        )

        in_front = z > 1e-4
        size_ok = r_pix >= min_radius_pix

        inside = (
            (u - r_pix >= 0.0)
            & (u + r_pix < float(self.cfg.camera_width))
            & (v - r_pix >= 0.0)
            & (v + r_pix < float(self.cfg.camera_height))
        )

        return in_front & finite & size_ok & inside

    def _compute_base_relative_velocity_labels(self) -> torch.Tensor:
        """
        yaw-only base frame 下的目标相对物理平移速度 [vx, vy, vz]

        语义与 Stage1 TAQ v2 标签严格一致：
            R_WB.T @ (v_target_world - v_base_world)

        这里不包含仅由 yaw-only 坐标轴旋转引起的
        ``-omega x position`` 视运动项。
        """
        root_quat_w = self.turret.data.root_quat_w
        root_lin_vel_w = quat_apply(root_quat_w, self.turret.data.root_lin_vel_b)
        target_rel_vel_w = self._target_vel_w - root_lin_vel_w
        return self._rotate_world_to_yaw_only_local(root_quat_w, target_rel_vel_w)
    
    def _compute_base_yaw_pitch_from_pos(self, pos_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        yaw-only base frame 下：
            yaw   = atan2(y, x)
            pitch = atan2(-z, sqrt(x^2 + y^2))
        """
        x = pos_base[:, 0]
        y = pos_base[:, 1]
        z = pos_base[:, 2]

        yaw_base = torch.atan2(y, x)
        pitch_base = torch.atan2(-z, torch.sqrt(x * x + y * y).clamp_min(1e-6))
        return yaw_base, pitch_base

    def _get_target_out_of_camera_sector_mask(self) -> torch.Tensor:
        """Return env mask where target is outside configured camera sector."""
        if (not self.cfg.sample_target_in_camera_sector) or (self._vision_camera is None):
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        cam_pos_w, cam_quat_w = self._get_runtime_camera_pose_w()

        to_target_w = self._target_pos_w - cam_pos_w
        to_target_cam = self._rotate_world_to_local(cam_quat_w, to_target_w)

        x = to_target_cam[:, 0]
        y = to_target_cam[:, 1]
        z = to_target_cam[:, 2]
        r = torch.norm(to_target_cam, dim=-1)

        yaw_cam = torch.atan2(x, z.clamp_min(1e-6))
        pitch_cam = torch.atan2(-y, torch.sqrt(x * x + z * z).clamp_min(1e-6))

        r_min, r_max = float(self.cfg.target_radius_range[0]), float(self.cfg.target_radius_range[1])
        yaw_min, yaw_max = float(self.cfg.target_cam_yaw_range[0]), float(self.cfg.target_cam_yaw_range[1])
        pitch_min, pitch_max = float(self.cfg.target_cam_pitch_range[0]), float(self.cfg.target_cam_pitch_range[1])

        outside = (
            (r < r_min) | (r > r_max)
            | (yaw_cam < yaw_min) | (yaw_cam > yaw_max)
            | (pitch_cam < pitch_min) | (pitch_cam > pitch_max)
        )
        return outside
    


    def _push_runtime_camera_pose_to_usd(self, env_ids: torch.Tensor | None = None) -> None:
        """把重建出来的 runtime 相机位姿推到 proxy render cameras。"""
        if len(self._runtime_camera_proxy_paths) == 0:
            return

        cam_pos_w, cam_quat_w_usd = self._get_runtime_camera_pose_w_usd()

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).flatten()

        for env_id_t in env_ids:
            env_id = int(env_id_t.item())
            if env_id < 0 or env_id >= len(self._runtime_camera_proxy_paths):
                continue

            self._set_camera_pose_on_prim(
                self._runtime_camera_proxy_paths[env_id],
                cam_pos_w[env_id],
                cam_quat_w_usd[env_id],
            )

    def _sync_render_camera_before_read(self) -> None:
        """读取 rgb 前，同步 proxy camera -> render -> TiledCamera.update。"""
        if self._vision_camera is None:
            return

        if not self._ensure_vision_camera_initialized():
            if self.cfg.debug_diagnostics:
                print("[vision] camera not initialized before read; skip this frame")
            return

        try:
            self._push_runtime_camera_pose_to_usd()
        except Exception as exc:
            if self.cfg.debug_diagnostics:
                print(f"[vision] proxy pose push failed before read: {exc}")
            return

        try:
            self.sim.render()
        except Exception as exc:
            if self.cfg.debug_diagnostics:
                print(f"[vision] render failed before read: {exc}")
            return

        try:
            self._vision_camera.update(dt=0.0)
        except Exception as exc:
            if self.cfg.debug_diagnostics:
                print(f"[vision] camera.update failed before read: {exc}")


    def _maybe_dump_debug_camera_frame(
        self,
        rgb: torch.Tensor,
        cam_pos_w: torch.Tensor,
        cam_quat_w: torch.Tensor,
        direction_gt: torch.Tensor,
        target_in_view: torch.Tensor,
    ) -> None:
        """在指定 step 导出调试图片，便于排查相机坐标系/投影问题。"""
        step = int(self.common_step_counter)
        wanted_steps = set(int(s) for s in self.cfg.dump_debug_camera_steps)

        if step in wanted_steps:
            print(
                f"[debug image] hit step={step}, "
                f"rgb_shape={tuple(rgb.shape)}, "
                f"dumped={step in self._debug_camera_dumped_steps}, "
                f"dump_dir={self._debug_camera_dump_dir}"
            )

        if step not in wanted_steps:
            return
        if step in self._debug_camera_dumped_steps:
            return
        if rgb.ndim != 4 or rgb.shape[0] <= 0:
            print(f"[debug image] skip save at step={step}, invalid rgb shape: {tuple(rgb.shape)}")
            return

        env_id = int(self.cfg.dump_debug_camera_env_id)
        if env_id < 0 or env_id >= self.num_envs:
            env_id = 0

        try:
            rgb_env = rgb[env_id].detach().cpu().clamp(0.0, 1.0).numpy()  # [3,H,W]
            img = np.transpose(rgb_env, (1, 2, 0))  # [H,W,3]
            img_u8 = (img * 255.0).round().astype(np.uint8)

            h, w = img_u8.shape[0], img_u8.shape[1]

            def draw_cross(arr: np.ndarray, u: int, v: int, color: tuple[int, int, int], size: int = 8, thick: int = 2):
                if not (0 <= u < w and 0 <= v < h):
                    return
                u0 = max(0, u - size)
                u1 = min(w, u + size + 1)
                v0 = max(0, v - size)
                v1 = min(h, v + size + 1)

                vt0 = max(0, v - thick)
                vt1 = min(h, v + thick + 1)
                ut0 = max(0, u - thick)
                ut1 = min(w, u + thick + 1)

                arr[vt0:vt1, u0:u1] = color
                arr[v0:v1, ut0:ut1] = color

            # ---------- 几何投影得到的目标中心 ----------
            gt_u = None
            gt_v = None
            try:
                to_target_w = self._target_pos_w[env_id] - cam_pos_w[env_id]
                to_target_cam = self._rotate_world_to_local(
                    cam_quat_w[env_id : env_id + 1],
                    to_target_w.unsqueeze(0),
                )[0]

                x = float(to_target_cam[0].item())
                y = float(to_target_cam[1].item())
                z = float(to_target_cam[2].item())

                if z > 1e-6:
                    K = self._vision_camera.data.intrinsic_matrices[env_id]
                    fx = float(K[0, 0].item())
                    fy = float(K[1, 1].item())
                    cx = float(K[0, 2].item())
                    cy = float(K[1, 2].item())

                    gt_u = int(round(fx * (x / z) + cx))
                    gt_v = int(round(fy * (y / z) + cy))
                    draw_cross(img_u8, gt_u, gt_v, (0, 255, 0), size=10, thick=2)  # 绿色：几何投影中心
            except Exception as exc:
                print(f"[debug image] step={step} project target failed: {exc}")

            # ---------- 从图里估计红球中心 ----------
            blob_ok, blob_u, blob_v, blob_score, blob_pixels = self._estimate_red_blob_center_single(rgb[env_id])
            if blob_ok:
                draw_cross(
                    img_u8,
                    int(round(blob_u)),
                    int(round(blob_v)),
                    (255, 255, 0),  # 黄色：图像里检测到的红球中心
                    size=10,
                    thick=2,
                )

            # ---------- 保存图片 ----------
            img_path = self._debug_camera_dump_dir / f"step_{step:04d}_env_{env_id}.png"
            imageio.imwrite(img_path, img_u8)

            # ---------- 保存对应元信息 ----------
            meta_path = self._debug_camera_dump_dir / f"step_{step:04d}_env_{env_id}.txt"
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"step={step}\n")
                f.write(f"env_id={env_id}\n")
                f.write(f"target_in_view={bool(target_in_view[env_id].item())}\n")
                f.write(f"cam_pos_w={cam_pos_w[env_id].detach().cpu().tolist()}\n")
                f.write(f"cam_quat_w={cam_quat_w[env_id].detach().cpu().tolist()}\n")
                f.write(f"target_pos_w={self._target_pos_w[env_id].detach().cpu().tolist()}\n")
                f.write(f"direction_gt_yaw_pitch={direction_gt[env_id].detach().cpu().tolist()}\n")
                f.write(f"projected_uv={[gt_u, gt_v]}\n")
                f.write(f"red_blob_found={blob_ok}\n")
                f.write(f"red_blob_uv={[float(blob_u), float(blob_v)] if blob_ok else None}\n")
                f.write(f"red_blob_score={float(blob_score)}\n")
                f.write(f"red_blob_pixels={int(blob_pixels)}\n")

            self._debug_camera_dumped_steps.add(step)
            print(f"[debug image] saved step={step} env={env_id} -> {img_path}")

        except Exception as exc:
            print(f"[debug image] save failed at step={step}: {exc}")

    def _refresh_vision_supervision(self) -> None:
        """只把 TiledCamera 当作 RGB 数据源；所有几何统一使用 runtime pose。"""
        if self._vision_camera is None or self._vision_rgb_history is None:
            self._vision_valid_mask.zero_()
            return

        if not self._ensure_vision_camera_initialized():
            self._vision_valid_mask.zero_()
            return

        # 先把 runtime pose 推到 proxy camera，再 render/update
        self._sync_render_camera_before_read()

        rgb = self._vision_camera.data.output["rgb"]
        if rgb is None:
            self._vision_valid_mask.zero_()
            return

        if rgb.ndim != 4 or rgb.shape[-1] < 3:
            self._vision_valid_mask.zero_()
            return

        rgb = rgb[..., :3].permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)
        if self.cfg.vision_normalize_rgb:
            rgb = rgb / 255.0

        # -----------------------------
        # 关键：
        # 不再使用 self._vision_camera.data.pos_w / quat_w_ros
        # 所有监督真值、投影、raycast 均使用 runtime 重建位姿
        # -----------------------------
        cam_pos_w, cam_quat_w = self._get_runtime_camera_pose_w()

        direction_gt = self._compute_camera_direction_labels(
            cam_pos_w=cam_pos_w,
            cam_quat_w=cam_quat_w,
        )
        velocity_gt = self._compute_camera_velocity_labels(
            cam_pos_w=cam_pos_w,
            cam_quat_w=cam_quat_w,
        )
        target_in_view = self._compute_target_projected_in_image_mask(
            cam_pos_w=cam_pos_w,
            cam_quat_w=cam_quat_w,
        )

        cam_pos_finite = torch.isfinite(cam_pos_w).all(dim=-1)
        cam_quat_finite = torch.isfinite(cam_quat_w).all(dim=-1)
        rgb_finite = torch.isfinite(rgb).all(dim=(1, 2, 3))
        label_finite = torch.isfinite(direction_gt).all(dim=-1) & torch.isfinite(velocity_gt).all(dim=-1)
        warmup_done = self._vision_collect_warmup_left <= 0

        current_frame_valid = (
            cam_pos_finite
            & cam_quat_finite
            & rgb_finite
            & label_finite
            & warmup_done
            & target_in_view
        )

        self._vision_rgb_history = torch.roll(self._vision_rgb_history, shifts=-1, dims=1)
        self._vision_rgb_history[:, -1] = rgb

        # 导出指定 step 的原始渲染图，用来排查相机坐标系/投影问题
        self._maybe_dump_debug_camera_frame(
            rgb=rgb,
            cam_pos_w=cam_pos_w,
            cam_quat_w=cam_quat_w,
            direction_gt=direction_gt,
            target_in_view=target_in_view,
        )

        self._vision_frame_valid_history = torch.roll(self._vision_frame_valid_history, shifts=-1, dims=1)
        self._vision_frame_valid_history[:, -1] = current_frame_valid

        self._vision_history_count = torch.clamp(self._vision_history_count + 1, max=self._vision_history_length)

        full_history = self._vision_history_count >= self._vision_history_length
        history_all_valid = torch.all(self._vision_frame_valid_history, dim=1)
        self._vision_valid_mask = full_history & history_all_valid

        self._vision_direction_gt.copy_(direction_gt)
        self._vision_velocity_gt.copy_(velocity_gt)
        self._vision_velocity_base_gt.copy_(self._compute_base_relative_velocity_labels())

        to_target_cam_w = self._target_pos_w - cam_pos_w
        self._vision_distance_gt[:] = torch.norm(to_target_cam_w, dim=-1)

        root_pos_w = self.turret.data.root_pos_w
        root_quat_w = self.turret.data.root_quat_w
        to_target_root_w = self._target_pos_w - root_pos_w
        self._vision_pos_base_gt[:] = self._rotate_world_to_yaw_only_local(root_quat_w, to_target_root_w)

        if self.cfg.debug_diagnostics and (int(self.common_step_counter) % 50 == 0):
            print(
                f"[vision debug] step={int(self.common_step_counter)} "
                f"cam_pos_finite={int(cam_pos_finite.sum())}/{self.num_envs} "
                f"cam_quat_finite={int(cam_quat_finite.sum())}/{self.num_envs} "
                f"rgb_finite={int(rgb_finite.sum())}/{self.num_envs} "
                f"label_finite={int(label_finite.sum())}/{self.num_envs} "
                f"warmup_done={int(warmup_done.sum())}/{self.num_envs} "
                f"target_in_view={int(target_in_view.sum())}/{self.num_envs} "
                f"valid={int(self._vision_valid_mask.sum())}/{self.num_envs}"
            )

            # 这里只做诊断打印，不参与任何计算
            try:
                sensor_pos = self._vision_camera.data.pos_w
                sensor_quat = self._vision_camera.data.quat_w_ros
                env0 = 0
                print(
                    "[camera sync check] "
                    f"step={int(self.common_step_counter)} "
                    f"sensor_pos0={sensor_pos[env0].detach().cpu().tolist()} "
                    f"runtime_pos0={cam_pos_w[env0].detach().cpu().tolist()} "
                    f"sensor_quat0={sensor_quat[env0].detach().cpu().tolist()} "
                    f"runtime_quat0={cam_quat_w[env0].detach().cpu().tolist()}"
                )
            except Exception as exc:
                print(f"[camera sync check] read sensor pose failed: {exc}")




    def get_vision_supervision_batch(self) -> dict[str, torch.Tensor]:
        """Return already-refreshed vision buffers for the current step."""
        if self._vision_rgb_history is None:
            raise RuntimeError("Vision supervision is disabled; no RGB history buffer was allocated.")

        root_speed_xy = torch.norm(self.turret.data.root_lin_vel_b[:, :2], dim=-1, keepdim=True)
        dog_moving_flag = (root_speed_xy > 0.05).to(dtype=torch.float32)

        lidar_distance = self._vision_distance_gt.clone()
        lidar_valid_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        if self.cfg.enable_dynamic_target_raycast_debug and (self._dynamic_target_raycast_sensor is not None):
            lidar_distance = self._raycast_hit_distance.clone()
            lidar_valid_mask = self._raycast_hit_valid.clone()

            invalid_mask = ~lidar_valid_mask
            lidar_distance[invalid_mask] = self._vision_distance_gt[invalid_mask]

        if not hasattr(self, "_printed_lidar_source_once"):
            self._printed_lidar_source_once = False

        if self.cfg.debug_diagnostics and not self._printed_lidar_source_once:
            if self.cfg.enable_dynamic_target_raycast_debug and (self._dynamic_target_raycast_sensor is not None):
                print("[vision batch] lidar_distance source = MultiMeshRayCaster")
            else:
                print("[vision batch] lidar_distance source = GT fallback")
            self._printed_lidar_source_once = True

        refined_yaw = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        refined_pitch = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        if hasattr(self, "_raycast_refined_yaw"):
            refined_yaw = self._raycast_refined_yaw.clone()
        if hasattr(self, "_raycast_refined_pitch"):
            refined_pitch = self._raycast_refined_pitch.clone()

        return {
            "images": self._vision_rgb_history,
            "direction": self._vision_direction_gt,
            "velocity": self._vision_velocity_gt,
            "velocity_base": self._vision_velocity_base_gt,
            "dog_moving_flag": dog_moving_flag,
            "valid_mask": self._vision_valid_mask,
            "distance": self._vision_distance_gt,
            "lidar_distance": lidar_distance,
            "lidar_valid_mask": lidar_valid_mask,
            "pos_base_gt": self._vision_pos_base_gt,
            "refined_yaw": refined_yaw,
            "refined_pitch": refined_pitch,
        }

    # 炮台朝向计算
    def _compute_turret_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (muzzle_pos_w, quat_w, forward_w)."""
        if self._muzzle_body_id is not None:
            body_pos_w = self.turret.data.body_pos_w[:, self._muzzle_body_id]
            quat_w = self.turret.data.body_quat_w[:, self._muzzle_body_id]
        else:
            body_pos_w = self.turret.data.root_pos_w
            quat_w = self.turret.data.root_quat_w

        forward_axis_b = self._forward_axis_b.unsqueeze(0).expand(quat_w.shape[0], -1)
        forward_w = quat_apply(quat_w, forward_axis_b)
        forward_w = forward_w / (torch.norm(forward_w, dim=-1, keepdim=True) + 1e-6)

        muzzle_offset_b = self._muzzle_offset_b.unsqueeze(0).expand(quat_w.shape[0], -1)
        muzzle_pos_w = body_pos_w + quat_apply(quat_w, muzzle_offset_b)

        return muzzle_pos_w, quat_w, forward_w
    

    def _compute_angle_error(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        turret_pos_w, _, turret_fwd_w = self._compute_turret_frame()

        to_target_w = self._target_pos_w - turret_pos_w
        dist = torch.norm(to_target_w, dim=-1, keepdim=True).clamp_min(1e-6)
        to_target_dir_w = to_target_w / dist

        # 用 atan2(||cross||, dot) 代替 acos(dot)，数值更稳定
        dot = (turret_fwd_w * to_target_dir_w).sum(dim=-1).clamp(-1.0, 1.0)
        cross = torch.cross(turret_fwd_w, to_target_dir_w, dim=-1)
        cross_norm = torch.norm(cross, dim=-1).clamp_min(1e-8)

        ang_err = torch.atan2(cross_norm, dot)

        return ang_err, dist.squeeze(-1), to_target_dir_w
    

    def _compute_muzzle_based_aim_refs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于“枪口点 -> 目标”的几何关系，计算：
            1) yaw_ref        : 枪口参考水平角（base yaw-only frame）
            2) pitch_los_ref  : 不考虑重力时的直瞄 pitch
            3) pitch_g_ref    : 只考虑重力时的补偿 pitch（忽略空气阻力）

        约定与本项目现有 pitch 符号一致：
            yaw   = atan2(y, x)
            pitch = atan2(-z, sqrt(x^2 + y^2))
        即：
            - 目标更高 -> pitch 更负（抬枪）
            - 目标更低 -> pitch 更正（压枪）
        """
        muzzle_pos_w, _, _ = self._compute_turret_frame()
        root_quat_w = self.turret.data.root_quat_w

        rel_w = self._target_pos_w - muzzle_pos_w
        rel_base = self._rotate_world_to_yaw_only_local(root_quat_w, rel_w)

        x = rel_base[:, 0]
        y = rel_base[:, 1]
        z = rel_base[:, 2]

        horiz = torch.sqrt(x * x + y * y).clamp_min(1e-6)

        # 1) 几何参考 yaw / pitch（枪口点参考）
        yaw_ref = torch.atan2(y, x)
        pitch_los_ref = torch.atan2(-z, horiz)

        # 2) 只考虑重力时的抛体补偿角
        #    标准抛体公式先求“上为正”的 elevation，再转成当前项目的 pitch 符号（负号）
        v = float(self.cfg.muzzle_speed)
        g = float(abs(self.cfg.gravity_mps2))

        # 标准公式中的竖直高度差（上为正）
        dz = z

        v2 = v * v
        disc = v2 * v2 - g * (g * horiz * horiz + 2.0 * dz * v2)

        # 无解时回退到 LOS；有解时取低抛解
        valid = disc >= 0.0
        disc_safe = torch.clamp(disc, min=0.0)
        sqrt_disc = torch.sqrt(disc_safe)

        theta_low = torch.atan((v2 - sqrt_disc) / (g * horiz).clamp_min(1e-6))
        pitch_g_ref = -theta_low  # 转回本项目 pitch 符号：抬枪为负

        pitch_g_ref = torch.where(valid, pitch_g_ref, pitch_los_ref)

        return yaw_ref, pitch_los_ref, pitch_g_ref


    def _compute_soft_band_rewards(
        self,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        yaw_ref: torch.Tensor,
        pitch_los_ref: torch.Tensor,
        pitch_g_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        只考虑重力时的精确瞄准奖励：
        - yaw 对准枪口到目标的水平角 yaw_ref
        - pitch 对准只考虑重力的低抛弹道角 pitch_g_ref
        - 不再加入空气阻力补偿项
        """

        # yaw：无 deadband，高斯奖励持续逼近 0 误差
        yaw_err = wrap_to_pi(yaw - yaw_ref)
        yaw_sigma = 0.015
        yaw_reward = torch.exp(-((yaw_err / yaw_sigma) ** 2))

        # pitch：只追 gravity-only 弹道角
        # 注意：pitch_g_ref 已经是本项目符号，pitch 正方向向下，抬枪为负。
        pitch_target = pitch_g_ref

        pitch_err = pitch - pitch_target
        pitch_sigma = 0.015
        pitch_reward = torch.exp(-((pitch_err / pitch_sigma) ** 2))

        return yaw_reward, pitch_reward
    
    
    def _segment_segment_min_distance(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        q0: torch.Tensor,
        q1: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量计算 3D 线段 P(s)=p0+s*(p1-p0), Q(t)=q0+t*(q1-q0) 的最小距离。
        
        输入:
            p0, p1, q0, q1: shape (N, 3)
        返回:
            min_dist: shape (N,)
            s_clamped: shape (N,)   # P 线段上的最近点参数，范围 [0,1]
            t_clamped: shape (N,)   # Q 线段上的最近点参数，范围 [0,1]
        """
        u = p1 - p0                      # (N, 3)
        v = q1 - q0                      # (N, 3)
        w0 = p0 - q0                     # (N, 3)

        a = (u * u).sum(dim=-1)          # (N,)
        b = (u * v).sum(dim=-1)
        c = (v * v).sum(dim=-1)
        d = (u * w0).sum(dim=-1)
        e = (v * w0).sum(dim=-1)

        # 无界直线最优解
        denom = a * c - b * b
        parallel = denom.abs() < eps

        s = torch.zeros_like(a)
        t = torch.zeros_like(a)

        # 非平行情况
        non_parallel = ~parallel
        if torch.any(non_parallel):
            s_np = (b[non_parallel] * e[non_parallel] - c[non_parallel] * d[non_parallel]) / denom[non_parallel]
            t_np = (a[non_parallel] * e[non_parallel] - b[non_parallel] * d[non_parallel]) / denom[non_parallel]
            s[non_parallel] = s_np
            t[non_parallel] = t_np

        # 平行 / 近似平行时，先固定 s=0，再投影到 q 段
        if torch.any(parallel):
            t_p = torch.where(
                c[parallel] > eps,
                e[parallel] / c[parallel],
                torch.zeros_like(c[parallel]),
            )
            s[parallel] = 0.0
            t[parallel] = t_p

        # 第一次裁剪
        s = s.clamp(0.0, 1.0)
        t = t.clamp(0.0, 1.0)

        # 交替投影两次，基本足够稳定
        # 已知 t，更新 s
        s = torch.where(
            a > eps,
            ((b * t) - d) / a,
            torch.zeros_like(a),
        ).clamp(0.0, 1.0)

        # 已知 s，更新 t
        t = torch.where(
            c > eps,
            ((b * s) + e) / c,
            torch.zeros_like(c),
        ).clamp(0.0, 1.0)

        # 再更新一次 s（可提升边界情况下的稳定性）
        s = torch.where(
            a > eps,
            ((b * t) - d) / a,
            torch.zeros_like(a),
        ).clamp(0.0, 1.0)

        closest_p = p0 + s.unsqueeze(-1) * u
        closest_q = q0 + t.unsqueeze(-1) * v
        min_dist = torch.norm(closest_p - closest_q, dim=-1)

        return min_dist, s, t


    def _interp_cd_from_mach(self, mach: torch.Tensor) -> torch.Tensor:
        """
        根据 Mach 数线性插值得到 Cd。

        输入:
            mach: 任意 shape 的 tensor

        输出:
            cd: 与 mach 同 shape 的 tensor
        """
        mach_table = self._mach_table.to(device=mach.device, dtype=mach.dtype)
        cd_table = self._cd_table.to(device=mach.device, dtype=mach.dtype)

        orig_shape = mach.shape
        x = mach.reshape(-1)

        x = x.clamp(
            min=float(mach_table[0].item()),
            max=float(mach_table[-1].item()),
        )

        idx = torch.searchsorted(mach_table, x, right=True) - 1
        idx = idx.clamp(0, mach_table.numel() - 2)

        x0 = mach_table[idx]
        x1 = mach_table[idx + 1]
        y0 = cd_table[idx]
        y1 = cd_table[idx + 1]

        alpha = (x - x0) / (x1 - x0).clamp_min(1e-6)
        cd = y0 + alpha * (y1 - y0)

        return cd.reshape(orig_shape)


    def _compute_air_drag_accel_w(self, bullet_vel_w: torch.Tensor) -> torch.Tensor:
        """
        世界坐标系下的空气阻力加速度。

        不考虑风速时：
            v_rel = bullet_vel_w

        二次阻力：
            a_drag = -0.5 * rho * Cd(Mach) * A / m * |v| * v

        其中：
            Mach = |v| / speed_of_sound
        """
        dtype = bullet_vel_w.dtype
        device = bullet_vel_w.device

        speed = torch.norm(bullet_vel_w, dim=-1, keepdim=True).clamp_min(1e-6)

        if bool(self.cfg.use_mach_dependent_cd):
            mach = speed / self._speed_of_sound_mps.to(device=device, dtype=dtype)
            cd = self._interp_cd_from_mach(mach)
        else:
            cd = torch.full_like(speed, float(self.cfg.drag_coefficient))

        rho = self._air_density.to(device=device, dtype=dtype)
        area = self._projectile_area.to(device=device, dtype=dtype)
        mass = self._projectile_mass.to(device=device, dtype=dtype)

        coeff = 0.5 * rho * cd * area / mass

        drag_acc = -coeff * speed * bullet_vel_w
        return drag_acc


    def _compute_ballistic_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (min_distance_to_target, hit_flag, closest_time_s).

        GT 对比实验用弹道：
        - 弹丸只受重力，不考虑空气阻力；
        - 初始速度 = 枪口点当前速度 + 枪口方向 * muzzle_speed；
        - 命中距离按“同一时刻”的弹丸位置和目标位置比较；
        - 不再使用异步 segment-segment 最近距离，避免物理上不同时间点被误判为命中。
        """
        muzzle_pos_w, _, muzzle_fwd_w = self._compute_turret_frame()

        if self._muzzle_body_id is not None:
            body_pos_w = self.turret.data.body_pos_w[:, self._muzzle_body_id]
        else:
            body_pos_w = self.turret.data.root_pos_w

        muzzle_vel_w = self._compute_muzzle_velocity_w(
            body_pos_w=body_pos_w,
            muzzle_pos_w=muzzle_pos_w,
        )

        # 初始弹速：继承枪口点速度 + 枪口方向初速
        bullet_vel0 = muzzle_vel_w + muzzle_fwd_w * float(self.cfg.muzzle_speed)

        target_pos0 = self._target_pos_w
        target_vel0 = self._target_vel_w
        target_acc0 = getattr(self, "_target_acc_w", torch.zeros_like(target_vel0))

        dtype = muzzle_pos_w.dtype
        device = muzzle_pos_w.device

        # ------------------------------------------------------------
        # 1) 无空气阻力：解析重力弹道
        # ------------------------------------------------------------
        if not bool(self.cfg.enable_air_drag):
            times = self._ballistic_time_samples.to(device=device, dtype=dtype)  # (T,)
            t = times.view(1, -1, 1)                                             # (1, T, 1)

            gravity_w = self._gravity_w.to(device=device, dtype=dtype).view(1, 1, 3)

            bullet_pos_traj = (
                muzzle_pos_w.unsqueeze(1)
                + bullet_vel0.unsqueeze(1) * t
                + 0.5 * gravity_w * t * t
            )

            # GT 对比里建议 target_acc0 强制为 0；
            # 这里保留 0.5*a*t^2，是为了即使之后打开目标加速度，弹道评估仍自洽。
            target_pos_traj = (
                target_pos0.unsqueeze(1)
                + target_vel0.unsqueeze(1) * t
                + 0.5 * target_acc0.unsqueeze(1) * t * t
            )

        # ------------------------------------------------------------
        # 2) 如果以后手动打开空气阻力，则退回数值积分
        #    当前 GT 实验不建议走这个分支。
        # ------------------------------------------------------------
        else:
            bullet_pos = muzzle_pos_w
            bullet_vel = bullet_vel0

            target_pos = target_pos0
            target_vel = target_vel0
            target_acc = target_acc0

            bullet_pos_hist = [bullet_pos]
            target_pos_hist = [target_pos]

            gravity_acc = self._gravity_w.to(device=device, dtype=dtype).unsqueeze(0)

            for dt in self._ballistic_dt_samples:
                dt = dt.to(device=device, dtype=dtype).view(1, 1)

                # ------------------------------------------------------------
                # 弹丸运动：重力 + Mach-Cd 二次空气阻力
                # 使用 midpoint / RK2 积分，比原来的半隐式欧拉更稳定。
                # ------------------------------------------------------------
                acc0 = gravity_acc + self._compute_air_drag_accel_w(bullet_vel)

                vel_mid = bullet_vel + 0.5 * acc0 * dt

                acc_mid = gravity_acc + self._compute_air_drag_accel_w(vel_mid)

                bullet_pos = bullet_pos + vel_mid * dt
                bullet_vel = bullet_vel + acc_mid * dt

                # ------------------------------------------------------------
                # 目标运动：保持原来的恒定加速度模型
                # ------------------------------------------------------------
                target_pos = target_pos + target_vel * dt + 0.5 * target_acc * dt * dt
                target_vel = target_vel + target_acc * dt

                bullet_pos_hist.append(bullet_pos)
                target_pos_hist.append(target_pos)

            bullet_pos_traj = torch.stack(bullet_pos_hist, dim=1)
            target_pos_traj = torch.stack(target_pos_hist, dim=1)
            times = self._ballistic_time_samples.to(device=device, dtype=dtype)

        # ------------------------------------------------------------
        # 3) 同一时间段内的同步最近距离
        #    bullet(alpha) 和 target(alpha) 使用同一个 alpha，
        #    避免不同时间点之间的 segment-segment 假最近距离。
        # ------------------------------------------------------------
        bullet_seg_start = bullet_pos_traj[:, :-1, :]
        bullet_seg_end = bullet_pos_traj[:, 1:, :]

        target_seg_start = target_pos_traj[:, :-1, :]
        target_seg_end = target_pos_traj[:, 1:, :]

        rel0 = bullet_seg_start - target_seg_start
        rel1 = bullet_seg_end - target_seg_end
        rel_delta = rel1 - rel0

        denom = (rel_delta * rel_delta).sum(dim=-1).clamp_min(1e-8)
        alpha = -((rel0 * rel_delta).sum(dim=-1)) / denom
        alpha = alpha.clamp(0.0, 1.0)

        closest_rel = rel0 + alpha.unsqueeze(-1) * rel_delta
        seg_min_dist = torch.norm(closest_rel, dim=-1)

        min_dist, min_seg_idx = torch.min(seg_min_dist, dim=-1)

        t0 = times[:-1]
        dt_seg = times[1:] - times[:-1]

        row_idx = torch.arange(self.num_envs, device=device)
        closest_time = t0[min_seg_idx] + alpha[row_idx, min_seg_idx] * dt_seg[min_seg_idx]

        hit = min_dist < float(self.cfg.hit_radius)

        return min_dist, hit, closest_time
    

    def _invalidate_ballistic_cache(self, env_ids: torch.Tensor | None = None) -> None:
        """将 ballistic cache 标记为失效。
        env_ids=None 表示全部失效。
        """
        if env_ids is None:
            self._ballistic_cache_valid[:] = False
        else:
            if env_ids.numel() > 0:
                self._ballistic_cache_valid[env_ids] = False

    def _get_cached_ballistic_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回当前 step 的 ballistic metrics。
        若 cache 无效，则只计算一次并写入缓存；同一步 reward / done 共用。
        """
        invalid_envs = torch.nonzero(~self._ballistic_cache_valid, as_tuple=False).squeeze(-1)

        if invalid_envs.numel() > 0:
            min_dist, hit, closest_time = self._compute_ballistic_metrics()

            self._cached_min_ballistic_dist.copy_(min_dist)
            self._cached_ballistic_hit.copy_(hit)
            self._cached_closest_time.copy_(closest_time)
            self._ballistic_cache_valid[:] = True

        return (
            self._cached_min_ballistic_dist,
            self._cached_ballistic_hit,
            self._cached_closest_time,
        )


    def _get_observations(self) -> dict:
        joint_pos = self.turret.data.joint_pos
        joint_vel = self.turret.data.joint_vel

        yaw = joint_pos[:, int(self._yaw_dof_ids[0])]
        pitch = joint_pos[:, int(self._pitch_dof_ids[0])]
        yaw_vel = joint_vel[:, int(self._yaw_dof_ids[0])]
        pitch_vel = joint_vel[:, int(self._pitch_dof_ids[0])]

        root_lin_vel_b = self.turret.data.root_lin_vel_b
        root_ang_vel_b = self.turret.data.root_ang_vel_b
        projected_gravity_b = self.turret.data.projected_gravity_b
        # This observation is built after the current action has been applied.
        # Therefore the action history exposed to actor/critic must be the
        # action just executed. Keep _prev_actions for action-change rewards,
        # which compare the current action against the preceding one.
        prev_actions = self._actions
        fire_intent_obs = self._fire_intent.unsqueeze(-1)
        fire_cooldown_ratio = (
            self._fire_cooldown.to(torch.float32) / max(float(self.cfg.fire_cooldown_steps), 1.0)
        ).unsqueeze(-1)

        root_speed_xy = torch.norm(root_lin_vel_b[:, :2], dim=-1, keepdim=True)
        dog_moving_flag = (root_speed_xy > 0.05).to(dtype=torch.float32)

        # -------------------------------------------------
        # GT target state
        # -------------------------------------------------
        root_pos_w = self.turret.data.root_pos_w
        root_quat_w = self.turret.data.root_quat_w

        to_target_root_w = self._target_pos_w - root_pos_w
        gt_pos_base = self._rotate_world_to_yaw_only_local(root_quat_w, to_target_root_w)
        gt_vel_base = self._compute_base_relative_velocity_labels()

        # 距离改成：机器狗本体（root/base）到目标的真实距离
        gt_distance = torch.norm(to_target_root_w, dim=-1)

        gt_yaw_base, gt_pitch_base = self._compute_base_yaw_pitch_from_pos(gt_pos_base)

        # -------------------------------------------------
        # policy 侧：
        #   - 默认模式：预测缺失时回退 GT
        #   - 严格模式：预测缺失时不给 GT，只给 0 占位
        # -------------------------------------------------
        use_gt_fallback = bool(getattr(self, "_use_gt_fallback_for_missing_rl_pred", True))
        pred_ready = self._rl_pred_ready
        pred_ready_f = pred_ready.unsqueeze(-1).to(dtype=torch.float32)

        if use_gt_fallback:
            policy_pos_base = torch.where(pred_ready.unsqueeze(-1), self._rl_pred_pos_base, gt_pos_base)
            policy_vel_base = torch.where(pred_ready.unsqueeze(-1), self._rl_pred_vel_base, gt_vel_base)

            policy_distance = torch.where(
                pred_ready,
                self._rl_pred_distance,
                gt_distance,
            ).unsqueeze(-1)

            policy_yaw_base_raw = torch.where(
                pred_ready,
                self._rl_pred_yaw_base,
                gt_yaw_base,
            )

            policy_pitch_base_raw = torch.where(
                pred_ready,
                self._rl_pred_pitch_base,
                gt_pitch_base,
            )

        else:
            # 严格模块模式：
            # 当 bridge 还没写回有效预测时，不允许 Actor 看到 GT。
            # 这里直接输出 0 占位，保持 observation 维度不变。
            policy_pos_base = self._rl_pred_pos_base * pred_ready_f
            policy_vel_base = self._rl_pred_vel_base * pred_ready_f

            policy_distance = (
                self._rl_pred_distance * pred_ready.to(dtype=torch.float32)
            ).unsqueeze(-1)

            policy_yaw_base_raw = self._rl_pred_yaw_base * pred_ready.to(dtype=torch.float32)
            policy_pitch_base_raw = self._rl_pred_pitch_base * pred_ready.to(dtype=torch.float32)

        # -------------------------------------------------
        # Actor/Critic aim reference
        #
        # Match turret_aim_env_gt_g.py behavior:
        # - Actor fallback uses gravity-compensated pitch reference.
        # - If RL perception prediction is ready, Actor still uses predicted yaw/pitch.
        # - Critic always uses privileged gravity-compensated reference.
        # -------------------------------------------------
        muzzle_yaw_base, muzzle_pitch_base = self._compute_muzzle_actual_yaw_pitch_in_yaw_only_base()

        yaw_ref_obs, pitch_los_ref_obs, pitch_g_ref_obs = self._compute_muzzle_based_aim_refs()

        # Use gravity-compensated pitch as the GT/fallback reference.
        pitch_ref_obs = pitch_g_ref_obs

        if use_gt_fallback:
            policy_yaw_ref_raw = torch.where(
                pred_ready,
                self._rl_pred_yaw_base,
                yaw_ref_obs,
            )

            policy_pitch_ref_raw = torch.where(
                pred_ready,
                self._rl_pred_pitch_base,
                pitch_ref_obs,
            )
        else:
            policy_yaw_ref_raw = self._rl_pred_yaw_base * pred_ready.to(dtype=torch.float32)
            policy_pitch_ref_raw = self._rl_pred_pitch_base * pred_ready.to(dtype=torch.float32)

        policy_yaw_error = wrap_to_pi(policy_yaw_ref_raw - muzzle_yaw_base).unsqueeze(-1)
        policy_pitch_error = (policy_pitch_ref_raw - muzzle_pitch_base).unsqueeze(-1)

        # Critic privileged obs always uses gravity-compensated ballistic reference.
        gt_yaw_error = wrap_to_pi(yaw_ref_obs - muzzle_yaw_base).unsqueeze(-1)
        gt_pitch_error = (pitch_g_ref_obs - muzzle_pitch_base).unsqueeze(-1)



        # -------------------------------------------------
        # Observation normalization + structured obs layout
        #
        # Actor:
        #   actor_target_obs: 9
        #   robot_obs:        19
        #   total:            28
        #
        # Critic:
        #   critic_target_obs: 12
        #   robot_obs:         19
        #   total:             31
        # -------------------------------------------------

        # critic privileged state
        target_acc_base = self._rotate_world_to_yaw_only_local(
            self.turret.data.root_quat_w,
            self._target_acc_w,
        )

        eps = 1e-6

        # -----------------------------
        # 1) Network-input normalization
        # -----------------------------
        # Target position is encoded as unit LOS direction.
        # Target distance is centered to the configured spawn range.
        r_min = float(self.cfg.target_radius_range[0])
        r_max = float(self.cfg.target_radius_range[1])
        dist_mid = 0.5 * (r_min + r_max)
        dist_half = max(0.5 * (r_max - r_min), 1.0)

        # -----------------------------
        # Angle / error normalization scales
        # -----------------------------
        # Normalize joint angles around the center of the current USD limits. This also
        # handles asymmetric limits without introducing another Python-side range.
        turret_joint_limits = self.turret.data.joint_pos_limits[:, self._turret_dof_ids, :]
        turret_joint_center = 0.5 * (turret_joint_limits[..., 0] + turret_joint_limits[..., 1])
        turret_joint_half_range = 0.5 * (turret_joint_limits[..., 1] - turret_joint_limits[..., 0])
        turret_joint_half_range = turret_joint_half_range.clamp_min(eps)

        # 瞄准误差：表示“还差多少才能打中”，需要高增益，不要除以 pi
        # 0.05rad -> 1.0
        # 0.01rad -> 0.2
        aim_error_obs_scale = max(
            float(getattr(self.cfg, "aim_error_obs_scale", 0.05)),
            eps,
        )
        aim_error_obs_clip = max(
            float(getattr(self.cfg, "aim_error_obs_clip", 5.0)),
            1.0,
        )

        joint_vel_scale = 5.0
        root_lin_vel_scale = 2.0
        root_ang_vel_scale = 3.0

        target_speed_scale = max(
            abs(float(self.cfg.target_speed_range[0])),
            abs(float(self.cfg.target_speed_range[1])),
            1.0,
        )
        target_vel_scale = target_speed_scale + root_lin_vel_scale

        target_acc_scale = max(
            abs(float(self.cfg.target_acc_range[0])),
            abs(float(self.cfg.target_acc_range[1])),
            abs(float(self.cfg.target_vertical_acc_range[0])),
            abs(float(self.cfg.target_vertical_acc_range[1])),
            1.0,
        )

        def _soft_norm(x: torch.Tensor, scale: float) -> torch.Tensor:
            return torch.tanh(x / max(float(scale), eps))

        def _aim_error_norm(x: torch.Tensor) -> torch.Tensor:
            return torch.clamp(
                x / aim_error_obs_scale,
                -aim_error_obs_clip,
                aim_error_obs_clip,
            )

        # -----------------------------
        # 2) Target-side normalization
        # -----------------------------
        policy_distance_valid = policy_distance > eps
        policy_distance_safe = policy_distance.clamp_min(eps)

        policy_pos_base_n = torch.where(
            policy_distance_valid,
            policy_pos_base / policy_distance_safe,
            torch.zeros_like(policy_pos_base),
        )
        policy_vel_base_n = _soft_norm(policy_vel_base, target_vel_scale)
        policy_distance_n = torch.where(
            policy_distance_valid,
            (policy_distance - dist_mid) / dist_half,
            torch.zeros_like(policy_distance),
        )

        policy_yaw_error_n = _aim_error_norm(policy_yaw_error)
        policy_pitch_error_n = _aim_error_norm(policy_pitch_error)

        gt_distance_col = gt_distance.unsqueeze(-1)
        gt_distance_safe = gt_distance_col.clamp_min(eps)

        gt_pos_base_n = gt_pos_base / gt_distance_safe
        gt_vel_base_n = _soft_norm(gt_vel_base, target_vel_scale)
        gt_distance_n = (gt_distance_col - dist_mid) / dist_half

        gt_yaw_error_n = _aim_error_norm(gt_yaw_error)
        gt_pitch_error_n = _aim_error_norm(gt_pitch_error)

        target_acc_base_n = _soft_norm(target_acc_base, target_acc_scale)

        # -----------------------------
        # 3) Robot-side normalization
        # -----------------------------
        turret_joint_pos = torch.stack((yaw, pitch), dim=-1)
        turret_joint_pos_n = torch.clamp(
            (turret_joint_pos - turret_joint_center) / turret_joint_half_range,
            -1.0,
            1.0,
        )
        yaw_n = turret_joint_pos_n[:, 0:1]
        pitch_n = turret_joint_pos_n[:, 1:2]

        yaw_vel_n = _soft_norm(yaw_vel.unsqueeze(-1), joint_vel_scale)
        pitch_vel_n = _soft_norm(pitch_vel.unsqueeze(-1), joint_vel_scale)

        root_lin_vel_b_n = _soft_norm(root_lin_vel_b, root_lin_vel_scale)
        root_ang_vel_b_n = _soft_norm(root_ang_vel_b, root_ang_vel_scale)

        projected_gravity_b = torch.clamp(projected_gravity_b, -1.0, 1.0)
        dog_moving_flag = torch.clamp(dog_moving_flag, 0.0, 1.0)
        prev_actions = torch.clamp(prev_actions, -1.0, 1.0)

        fire_intent_n = torch.clamp(fire_intent_obs, 0.0, 1.0)
        fire_cooldown_ratio = torch.clamp(fire_cooldown_ratio, 0.0, 1.0)


        # projected_gravity_b 本身大约就在 [-1, 1]
        # dog_moving_flag 本身是 0/1
        # prev_actions 本身大约是 [-1, 1]
        # fire_cooldown_ratio 本身是 0~1

        # -----------------------------
        # 4) Actor target branch: 9 dims
        # -----------------------------
        actor_target_obs = torch.cat(
            [
                policy_pos_base_n,       # 3
                policy_vel_base_n,       # 3
                policy_distance_n,       # 1
                policy_yaw_error_n,      # 1
                policy_pitch_error_n,    # 1
            ],
            dim=-1,
        )

        # -----------------------------
        # 5) Shared robot branch: 19 dims
        # -----------------------------
        robot_obs = torch.cat(
            [
                yaw_n,                   # 1
                pitch_n,                 # 1
                yaw_vel_n,               # 1
                pitch_vel_n,             # 1
                root_lin_vel_b_n,        # 3
                root_ang_vel_b_n,        # 3
                projected_gravity_b,     # 3
                dog_moving_flag,         # 1
                prev_actions,            # 3
                fire_intent_n,           # 1
                fire_cooldown_ratio,     # 1
            ],
            dim=-1,
        )

        # -----------------------------
        # 6) Critic target branch: 12 dims
        #
        # 注意：
        # target_acc_base 必须放在 critic target branch 里，
        # 这样和 critic_target_dim=12, critic_robot_dim=19 对齐。
        # -----------------------------
        critic_target_obs = torch.cat(
            [
                gt_pos_base_n,           # 3
                gt_vel_base_n,           # 3
                gt_distance_n,           # 1
                gt_yaw_error_n,          # 1
                gt_pitch_error_n,        # 1
                target_acc_base_n,       # 3
            ],
            dim=-1,
        )

        # -----------------------------
        # 7) Final obs
        # -----------------------------
        policy_obs = torch.cat([actor_target_obs, robot_obs], dim=-1)    # 28
        critic_obs = torch.cat([critic_target_obs, robot_obs], dim=-1)   # 31

        # -----------------------------
        # 8) Shape guard
        # -----------------------------
        if policy_obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"policy_obs dim mismatch: got {policy_obs.shape[-1]}, "
                f"expected {self.cfg.observation_space}"
            )

        if critic_obs.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"critic_obs dim mismatch: got {critic_obs.shape[-1]}, "
                f"expected {self.cfg.state_space}"
            )

        # -----------------------------
        # 9) Invalid diagnostics before sanitizing
        # -----------------------------
        invalid_policy = ~torch.isfinite(policy_obs).all(dim=-1)
        invalid_critic = ~torch.isfinite(critic_obs).all(dim=-1)
        invalid_obs = invalid_policy | invalid_critic

        if torch.any(invalid_obs):
            self._report_invalid_envs(
                "observations",
                invalid_obs,
                {
                    "policy_obs": policy_obs,
                    "critic_obs": critic_obs,
                    "gt_pos_base": gt_pos_base,
                    "gt_vel_base": gt_vel_base,
                    "gt_distance": gt_distance,
                    "policy_pos_base": policy_pos_base,
                    "policy_vel_base": policy_vel_base,
                    "policy_distance": policy_distance,
                    "target_acc_base": target_acc_base,
                    "rl_pred_ready": self._rl_pred_ready,
                    "rl_pred_pos_base": self._rl_pred_pos_base,
                    "rl_pred_vel_base": self._rl_pred_vel_base,
                    "rl_pred_distance": self._rl_pred_distance,
                    "root_pos_w": self.turret.data.root_pos_w,
                    "root_lin_vel_b": root_lin_vel_b,
                    "root_ang_vel_b": root_ang_vel_b,
                    "projected_gravity_b": projected_gravity_b,
                    "target_vel_w": self._target_vel_w,
                    "target_acc_w": self._target_acc_w,
                },
            )

        # -----------------------------
        # 10) Safety sanitize + clip
        #
        # 归一化后，正常观测基本不会超过 [-10, 10]。
        # 这里的 clamp 只作为异常保护，不再破坏 80~150m 距离信息。
        # -----------------------------
        policy_obs = torch.nan_to_num(
            policy_obs,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        critic_obs = torch.nan_to_num(
            critic_obs,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

        policy_obs = torch.clamp(policy_obs, -10.0, 10.0)
        critic_obs = torch.clamp(critic_obs, -10.0, 10.0)

        return {
            "policy": policy_obs,
            "critic": critic_obs,
        }


    def _compute_current_step_hit(self) -> torch.Tensor:
        """
        统一计算当前 step 是否发生命中。

        命中条件与 reward / done 保持一致：
        1) 当前 step 真的触发了 fire
        2) 弹道判定命中
        3) 目标距离在允许命中范围内
        4) warmup 已结束

        Returns:
            hit_now: [num_envs] 的 bool tensor
        """
        fire_now = self._fire_command.bool()

        if not torch.any(fire_now):
            return torch.zeros_like(fire_now, dtype=torch.bool)

        _, dist, _ = self._compute_angle_error()
        _, ballistic_hit, _ = self._get_cached_ballistic_metrics()
        aiming_enabled = self._aim_warmup_left_steps <= 0

        hit_now = fire_now & ballistic_hit & (dist < self.cfg.max_hit_distance) & aiming_enabled
        return hit_now

    def _record_fire_ballistic_stats(
        self,
        fire_now: torch.Tensor,
        min_ballistic_dist: torch.Tensor,
    ) -> None:
        """Record fire-distance metrics from the post-physics ballistic state."""
        if not torch.any(fire_now):
            return

        fire_ids = torch.nonzero(fire_now, as_tuple=False).squeeze(-1)
        fire_min_dists = min_ballistic_dist[fire_ids]

        # Eval aggregate statistics.
        self._eval_fire_min_dist_count += fire_min_dists.numel()
        self._eval_fire_min_dist_sum += fire_min_dists.sum()
        self._eval_fire_min_dist_min = torch.minimum(
            self._eval_fire_min_dist_min,
            torch.min(fire_min_dists),
        )
        self._eval_fire_min_dist_max = torch.maximum(
            self._eval_fire_min_dist_max,
            torch.max(fire_min_dists),
        )

        remaining = self._eval_samples_limit - len(self._eval_fire_min_dist_samples)
        if remaining > 0:
            sample_vals = fire_min_dists.detach().cpu().tolist()[:remaining]
            self._eval_fire_min_dist_samples.extend(float(value) for value in sample_vals)

        # Per-episode statistics used by training logs.
        self._episode_fire_min_dist_count[fire_ids] += 1
        self._episode_fire_min_dist_sum[fire_ids] += fire_min_dists
        self._episode_fire_min_dist_min[fire_ids] = torch.minimum(
            self._episode_fire_min_dist_min[fire_ids],
            fire_min_dists,
        )

    def _compute_muzzle_actual_yaw_pitch_in_yaw_only_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算真实枪口 forward 方向在 yaw-only base 坐标系下的 yaw / pitch。

        坐标约定：
            yaw   = atan2(y, x)
            pitch = atan2(-z, sqrt(x^2 + y^2))

        注意：
        - yaw 的 atan2 第二个参数不能 clamp 到正数，否则 x<0 时 yaw 会被算错。
        - pitch 正方向为向下，因此使用 -z。
        """
        _, _, muzzle_fwd_w = self._compute_turret_frame()
        root_quat_w = self.turret.data.root_quat_w

        fwd_base = self._rotate_world_to_yaw_only_local(root_quat_w, muzzle_fwd_w)
        fwd_base = fwd_base / torch.norm(fwd_base, dim=-1, keepdim=True).clamp_min(1e-6)

        x = fwd_base[:, 0]
        y = fwd_base[:, 1]
        z = fwd_base[:, 2]

        yaw = torch.atan2(y, x)
        pitch = torch.atan2(-z, torch.sqrt(x * x + y * y).clamp_min(1e-6))

        return yaw, pitch
        
    def _get_rewards(self) -> torch.Tensor:
        yaw, pitch = self._compute_muzzle_actual_yaw_pitch_in_yaw_only_base()

        # -------------------------------------------------
        # 基于“枪口点 -> 目标”的 yaw / pitch 参考奖励
        # -------------------------------------------------
        yaw_ref, pitch_los_ref, pitch_g_ref = self._compute_muzzle_based_aim_refs()
        yaw_align_reward, pitch_align_reward = self._compute_soft_band_rewards(
            yaw=yaw,
            pitch=pitch,
            yaw_ref=yaw_ref,
            pitch_los_ref=pitch_los_ref,
            pitch_g_ref=pitch_g_ref,
        )


        delta_pen = (self._delta_pos ** 2).sum(dim=-1)
        action_change_pen = ((self._actions[:, :2] - self._prev_actions[:, :2]) ** 2).sum(dim=-1)

        min_ballistic_dist, _, _ = self._get_cached_ballistic_metrics()
        fire_now = self._fire_command.bool()
        self._record_fire_ballistic_stats(fire_now, min_ballistic_dist)

        aiming_enabled = self._aim_warmup_left_steps <= 0
        aiming_mask = aiming_enabled.float()

        hit_now = self._compute_current_step_hit()
        bad_fire = fire_now & (~hit_now) & aiming_enabled

        # latch hit event for done logic
        self._hit_flag |= hit_now

        # -------------------------------------------------
        # Fire efficiency shaping
        #
        # 注意：
        # self._episode_fire_count 在 _pre_physics_step() 里已经包含了当前这次 fire。
        # 所以：
        #   当前 fire 是第1枪 -> _episode_fire_count == 1
        #   当前 fire 是第2枪 -> _episode_fire_count == 2
        # -------------------------------------------------
        shot_index = self._episode_fire_count.to(torch.float32)

        # 第一枪命中：额外奖励
        first_shot_hit_bonus = (
            self.cfg.r_first_shot_hit_bonus
            * (hit_now & (self._episode_fire_count == 1)).float()
        )

        # 重复开火递增惩罚：
        # 第1枪不扣；
        # 第2枪额外扣 1 * r_repeat_fire_penalty；
        # 第3枪额外扣 2 * r_repeat_fire_penalty；
        # 第4枪额外扣 3 * r_repeat_fire_penalty；
        # 只在 fire_now 当步扣，不在普通瞄准步扣。
        repeat_fire_penalty = (
            self.cfg.r_repeat_fire_penalty
            * torch.clamp(shot_index - 1.0, min=0.0)
            * fire_now.float()
            * aiming_mask
        )
        

        # -------------------------------------------------
        # Ballistic shaping:
        # 远距离有梯度 + 近距离精修
        # -------------------------------------------------
        d = min_ballistic_dist

        # 远距离项：6m 仍然有明显信号，不会像 exp 那样直接接近 0
        ballistic_far = 1.0 / (1.0 + (d / 3.0) ** 2)

        # 近距离项随命中半径缩放，保证奖励塑形和实际命中判定一致。
        hit_scale = max(float(self.cfg.hit_radius), 1e-6)
        ballistic_near = torch.exp(-((d / hit_scale).clamp(max=10.0) ** 2))

        ballistic_score = 0.4 * ballistic_far + 1.2 * ballistic_near

        # Small guard against policies that aim forever and never fire.
        near_timeout = self.episode_length_buf >= self.max_episode_length - 2
        no_fire_timeout_penalty = (
            self.cfg.r_no_fire_timeout_penalty
            * (near_timeout & (self._episode_fire_count == 0) & aiming_enabled).float()
        )

        # -------------------------------------------------
        # Bad fire penalty:
        # 错得越远，开火惩罚越大；3m 以后封顶，避免过大
        # -------------------------------------------------
        bad_fire_penalty = bad_fire.float() * (
            -2.0 - 3.0 * torch.clamp(d, 0.0, 3.0)
        )

        reward = (
            self.cfg.w_yaw_align * yaw_align_reward * aiming_mask
            + self.cfg.w_pitch_align * pitch_align_reward * aiming_mask
            + self.cfg.w_ballistic * ballistic_score * aiming_mask
            + self.cfg.r_hit * hit_now.float()
            + first_shot_hit_bonus
            + repeat_fire_penalty
            + no_fire_timeout_penalty
            + bad_fire_penalty
            + self.cfg.r_fire_cost * fire_now.float() * aiming_mask
            + self.cfg.w_rate * delta_pen * aiming_mask
            + self.cfg.w_action_change * action_change_pen * aiming_mask
            + self.cfg.r_time_penalty * aiming_mask
        )

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------------------------------------
        # 基础终止条件
        # ------------------------------------------------------------
        invalid_state = self._get_invalid_state_mask()

        root_pos_w = self.turret.data.root_pos_w
        fallen = root_pos_w[:, 2] < float(self.cfg.failure_height_threshold)

        root_quat_w = self.turret.data.root_quat_w
        up_vec_b = torch.tensor(
            [0.0, 0.0, 1.0],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        up_vec_w = quat_apply(root_quat_w, up_vec_b)
        tilt_cos = up_vec_w[:, 2]
        tilted = tilt_cos < float(self.cfg.failure_tilt_cos_threshold)

        invalid_or_failed = invalid_state | fallen | tilted

        # ------------------------------------------------------------
        # 命中 / 弹道命中
        # ------------------------------------------------------------
        # 安全兜底：若某些流程下 _get_rewards 未先执行，确保这里也能更新命中锁存位。
        current_step_hit = self._compute_current_step_hit()
        self._hit_flag |= current_step_hit

        hit_now = self._hit_flag.clone()

        # The last allowed shot is scored normally before ending the episode.
        # Exhaustion is a task failure (terminated), not a time-limit truncation.
        shot_limit = int(self.cfg.max_shots_per_episode)
        shot_budget_exhausted = (
            (shot_limit > 0) & (self._episode_fire_count >= shot_limit) & (~hit_now)
        )
        self._shot_budget_exhausted.copy_(shot_budget_exhausted)

        # ------------------------------------------------------------
        # timeout
        # ------------------------------------------------------------
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if shot_limit > 0:
            # RL-Games bootstraps timeouts. A simultaneous hit or spent budget
            # is a completed task and must not receive timeout bootstrapping.
            time_out &= ~(hit_now | shot_budget_exhausted)

        # ------------------------------------------------------------
        # 目标可见性 / 相机扇区相关
        # ------------------------------------------------------------
        target_out_of_view = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        if self.cfg.enable_vision_supervision and (self._vision_camera is not None):
            self._ensure_vision_supervision_ready()
            target_in_view = self._compute_target_projected_in_image_mask()
            target_out_of_view = ~target_in_view

        # ------------------------------------------------------------
        # terminated
        # ------------------------------------------------------------
        # Never terminate from the reset placeholder, stale camera frames, or
        # the two-second articulation settling period. Once eligible, require
        # several consecutive projected-image misses instead of reacting to a
        # single renderer/tracker transient.
        out_of_view_eligible = (
            self._target_out_of_view_termination_mask
            & (~self._pending_target_resample)
            & (self._vision_collect_warmup_left <= 0)
            & (self._aim_warmup_left_steps <= 0)
        )
        out_of_view_miss = target_out_of_view & out_of_view_eligible
        self._target_out_of_view_streak = torch.where(
            out_of_view_miss,
            self._target_out_of_view_streak + 1,
            torch.zeros_like(self._target_out_of_view_streak),
        )
        grace_frames = max(1, int(self.cfg.target_out_of_view_grace_frames))
        target_out_of_view_termination = out_of_view_miss & (
            self._target_out_of_view_streak >= grace_frames
        )
        self._target_out_of_view_terminated.copy_(target_out_of_view_termination)
        terminated = hit_now | invalid_or_failed | target_out_of_view_termination | shot_budget_exhausted

        # ------------------------------------------------------------
        # 统计累计（给 eval 汇总用）
        # ------------------------------------------------------------
        done_envs = terminated | time_out
        if torch.any(done_envs):
            self._eval_total_episodes += torch.count_nonzero(done_envs)
            self._eval_total_hits += torch.count_nonzero(hit_now & done_envs)
            self._eval_total_timeouts += torch.count_nonzero(time_out & done_envs)
            self._eval_total_fallen += torch.count_nonzero(fallen & done_envs)
            self._eval_total_invalid += torch.count_nonzero(invalid_state & done_envs)
            self._eval_total_target_out_of_view += torch.count_nonzero(
                target_out_of_view_termination & done_envs
            )
            self._eval_total_shot_budget_exhausted += torch.count_nonzero(shot_budget_exhausted & done_envs)

            if self._collect_eval_details:
                spawn_positions = self._episode_start_target_pos_w[done_envs]
                spawn_count = int(spawn_positions.shape[0])
                self._eval_target_spawn_count += spawn_count
                self._eval_target_spawn_sum += spawn_positions.sum(dim=0)
                remaining = self._eval_samples_limit - len(self._eval_target_spawn_samples)
                if remaining > 0:
                    samples = spawn_positions.detach().cpu().tolist()[:remaining]
                    self._eval_target_spawn_samples.extend(
                        [[float(value) for value in position] for position in samples]
                    )

                done_ids = torch.nonzero(done_envs, as_tuple=False).squeeze(-1).detach().cpu().tolist()
                for env_id in done_ids:
                    fires = int(self._episode_fire_count[env_id].item())
                    hit = bool(hit_now[env_id].item())
                    self._eval_episode_fire_counts.append(fires)
                    self._eval_episode_hits.append(1 if hit else 0)

        return terminated, time_out

    def reset_eval_metrics(self) -> None:
        self._collect_eval_details = True
        self._eval_total_fires.zero_()
        self._eval_total_episodes.zero_()
        self._eval_total_hits.zero_()
        self._eval_total_timeouts.zero_()
        self._eval_total_fallen.zero_()
        self._eval_total_invalid.zero_()
        self._eval_total_target_out_of_view.zero_()
        self._eval_total_shot_budget_exhausted.zero_()
        self._eval_target_spawn_count.zero_()
        self._eval_target_spawn_sum.zero_()
        self._eval_fire_min_dist_count.zero_()
        self._eval_fire_min_dist_sum.zero_()
        self._eval_fire_min_dist_min.fill_(float("inf"))
        self._eval_fire_min_dist_max.zero_()
        self._episode_fire_count.zero_()
        self._fire_intent_high_steps.zero_()
        self._episode_fire_min_dist_sum.zero_()
        self._episode_fire_min_dist_count.zero_()
        self._episode_fire_min_dist_min.fill_(float("inf"))
        self._eval_target_spawn_samples.clear()
        self._eval_fire_min_dist_samples.clear()
        self._eval_episode_fire_counts.clear()
        self._eval_episode_hits.clear()

    def get_eval_metrics(self) -> dict[str, float | list[float] | list[list[float]] | list[int]]:
        episodes = int(self._eval_total_episodes.item())
        fires = int(self._eval_total_fires.item())
        hits = int(self._eval_total_hits.item())
        timeouts = int(self._eval_total_timeouts.item())
        fallen = int(self._eval_total_fallen.item())
        invalid = int(self._eval_total_invalid.item())
        target_out_of_view = int(self._eval_total_target_out_of_view.item())
        shot_budget_exhausted = int(self._eval_total_shot_budget_exhausted.item())
        target_spawn_count = int(self._eval_target_spawn_count.item())
        fire_min_dist_count = int(self._eval_fire_min_dist_count.item())
        denom = max(episodes, 1)

        if target_spawn_count > 0:
            target_spawn_mean = (self._eval_target_spawn_sum / float(target_spawn_count)).detach().cpu().tolist()
        else:
            target_spawn_mean = [0.0, 0.0, 0.0]

        fire_min_dist_mean = (
            float((self._eval_fire_min_dist_sum / float(fire_min_dist_count)).item()) if fire_min_dist_count > 0 else 0.0
        )
        fire_min_dist_min = (
            float(self._eval_fire_min_dist_min.item()) if fire_min_dist_count > 0 else 0.0
        )
        fire_min_dist_max = (
            float(self._eval_fire_min_dist_max.item()) if fire_min_dist_count > 0 else 0.0
        )

        return {
            "episodes": float(episodes),
            "fires": float(fires),
            "hits": float(hits),
            "timeouts": float(timeouts),
            "fallen": float(fallen),
            "invalid": float(invalid),
            "target_out_of_view": float(target_out_of_view),
            "shot_budget_exhausted": float(shot_budget_exhausted),
            "hit_rate": float(hits / denom),
            "avg_fires_per_episode": float(fires / denom),
            "timeout_rate": float(timeouts / denom),
            "fallen_rate": float(fallen / denom),
            "invalid_rate": float(invalid / denom),
            "target_out_of_view_rate": float(target_out_of_view / denom),
            "shot_budget_exhausted_rate": float(shot_budget_exhausted / denom),
            "target_spawn_count": float(target_spawn_count),
            "target_spawn_mean_x": float(target_spawn_mean[0]),
            "target_spawn_mean_y": float(target_spawn_mean[1]),
            "target_spawn_mean_z": float(target_spawn_mean[2]),
            "target_spawn_samples": self._eval_target_spawn_samples,
            "fire_min_dist_count": float(fire_min_dist_count),
            "fire_min_dist_mean": fire_min_dist_mean,
            "fire_min_dist_min": fire_min_dist_min,
            "fire_min_dist_max": fire_min_dist_max,
            "fire_min_dist_samples": self._eval_fire_min_dist_samples,
            "episode_fire_counts": self._eval_episode_fire_counts,
            "episode_hits": self._eval_episode_hits,
        }
    

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        # 注意：
        # runtime lidar 是全局创建的，不应因单个/部分 env reset 而整体清零初始化状态。
        # 否则 rollout 中任意 env done 都会让所有 env 的 lidar 重新 warmup，导致采集不稳定。

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        env_ids_seq = env_ids_t.detach().cpu().tolist()
        episode_count = max(int(env_ids_t.numel()), 1)

        root_height = self.turret.data.root_pos_w[env_ids_t, 2]
        upright_cos = -self.turret.data.projected_gravity_b[env_ids_t, 2]
        fallen = (root_height < self.cfg.failure_height_threshold) | (upright_cos < self.cfg.failure_tilt_cos_threshold)
        invalid_state = self._get_invalid_state_mask()[env_ids_t]
        # Use the exact latched termination reason from _get_dones(). This
        # keeps TensorBoard consistent with the projected-image termination
        # rule instead of reporting the looser configured angular sector.
        target_out_of_view = self._target_out_of_view_terminated[env_ids_t]
        time_out = self.reset_time_outs[env_ids_t]
        hit_flags = self._hit_flag[env_ids_t]
        episode_fires = self._episode_fire_count[env_ids_t]

        hit_rate = torch.count_nonzero(hit_flags).float() / episode_count
        timeout_rate = torch.count_nonzero(time_out).float() / episode_count

        mean_fires = torch.mean(episode_fires.float())
        fired_episode_rate = torch.count_nonzero(episode_fires > 0).float() / episode_count

        # 只开一枪且命中的比例：可近似看成 first-shot hit rate
        one_fire_hit_rate = torch.count_nonzero(
            hit_flags & (episode_fires == 1)
        ).float() / episode_count

        # episode 内 fire_min_dist 统计
        fire_min_counts = self._episode_fire_min_dist_count[env_ids_t]
        total_fire_min_count = torch.sum(fire_min_counts)

        if int(total_fire_min_count.item()) > 0:
            fire_min_dist_mean = (
                torch.sum(self._episode_fire_min_dist_sum[env_ids_t])
                / total_fire_min_count.float()
            )

            valid_fire_min = torch.where(
                fire_min_counts > 0,
                self._episode_fire_min_dist_min[env_ids_t],
                torch.full_like(self._episode_fire_min_dist_min[env_ids_t], float("inf")),
            )
            fire_min_dist_min = torch.min(valid_fire_min)
        else:
            fire_min_dist_mean = torch.zeros((), dtype=torch.float32, device=self.device)
            fire_min_dist_min = torch.zeros((), dtype=torch.float32, device=self.device)

        self.extras["log"] = {
            "Episode_Termination/hit_rate": hit_rate,
            "Episode_Termination/fallen_rate": torch.count_nonzero(fallen).float() / episode_count,
            "Episode_Termination/invalid_rate": torch.count_nonzero(invalid_state).float() / episode_count,
            "Episode_Termination/target_out_of_view_rate": torch.count_nonzero(target_out_of_view).float() / episode_count,
            "Episode_Termination/time_out_rate": timeout_rate,
            "Episode_Termination/shot_budget_exhausted_rate": (
                torch.count_nonzero(self._shot_budget_exhausted[env_ids_t]).float() / episode_count
            ),
            "Episode_Termination/mean_root_height": torch.mean(root_height),
            "Episode_Termination/mean_upright_cos": torch.mean(upright_cos),

            # 训练时重点观察这几个
            "Fire/mean_fires_per_episode": mean_fires,
            "Fire/fired_episode_rate": fired_episode_rate,
            "Fire/one_fire_hit_rate": one_fire_hit_rate,
            "Fire/fire_min_dist_mean": fire_min_dist_mean,
            "Fire/fire_min_dist_min": fire_min_dist_min,
        }

        self._hit_flag[env_ids_t] = False
        self._fire_cooldown[env_ids_t] = 0
        # 所有 stage 在 reset 后都走 warm-up，先让机器狗稳定
        self._aim_warmup_left_steps[env_ids_t] = self._aim_warmup_steps
        self.turret.reset(env_ids_seq)
        super()._reset_idx(env_ids_seq)

        self._actions[env_ids_t] = 0.0
        self._prev_actions[env_ids_t] = 0.0
        self._delta_pos[env_ids_t] = 0.0
        self._fire_command[env_ids_t] = 0.0
        self._fire_intent[env_ids_t] = 0.0
        self._fire_intent_high_steps[env_ids_t] = 0

        self._rl_pred_ready[env_ids_t] = False
        self._rl_pred_pos_base[env_ids_t] = 0.0
        self._rl_pred_vel_base[env_ids_t] = 0.0
        self._rl_pred_distance[env_ids_t] = 0.0
        self._rl_pred_yaw_base[env_ids_t] = 0.0
        self._rl_pred_pitch_base[env_ids_t] = 0.0
        self._rl_bridge_training_ready[env_ids_t] = False
        self._rl_bridge_velocity_ready[env_ids_t] = False
        self._target_out_of_view_streak[env_ids_t] = 0
        self._target_out_of_view_terminated[env_ids_t] = False

        self._episode_fire_count[env_ids_t] = 0
        self._shot_budget_exhausted[env_ids_t] = False
        self._episode_fire_min_dist_sum[env_ids_t] = 0.0
        self._episode_fire_min_dist_count[env_ids_t] = 0
        self._episode_fire_min_dist_min[env_ids_t] = float("inf")
        self._target_vel_w[env_ids_t] = 0.0
        # reset 后 ballistic cache 失效
        # reset vision buffers for these envs
        if self._vision_rgb_history is not None:
            self._vision_rgb_history[env_ids_t] = 0.0
        self._vision_frame_valid_history[env_ids_t] = False
        self._vision_history_count[env_ids_t] = 0
        self._vision_valid_mask[env_ids_t] = False
        self._vision_direction_gt[env_ids_t] = 0.0
        self._vision_velocity_gt[env_ids_t] = 0.0
        self._vision_velocity_base_gt[env_ids_t] = 0.0
        self._vision_distance_gt[env_ids_t] = 0.0
        self._vision_pos_base_gt[env_ids_t] = 0.0
        self._vision_collect_warmup_left[env_ids_t] = self._vision_collect_warmup_steps_after_reset
        self._pending_target_resample[env_ids_t] = False
        self._invalidate_ballistic_cache(env_ids_t)
        self._cached_min_ballistic_dist[env_ids_t] = 0.0
        self._cached_ballistic_hit[env_ids_t] = False
        self._cached_closest_time[env_ids_t] = 0.0

        self._locomotion_last_actions[env_ids_t] = 0.0
        self._locomotion_joint_targets[env_ids_t] = self.turret.data.default_joint_pos[env_ids_t][:, self._leg_dof_ids]
        if self._locomotion_policy is not None:
            self._resample_locomotion_commands(env_ids_t)
        else:
            self._locomotion_commands[env_ids_t] = 0.0
            self._locomotion_command_time_left[env_ids_t] = 0.0

        default_root_state = self.turret.data.default_root_state[env_ids_t].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids_t]
        default_root_state[:, 0] = self.scene.env_origins[env_ids_t, 0] + self.cfg.base_init_pos[0]
        default_root_state[:, 1] = self.scene.env_origins[env_ids_t, 1] + self.cfg.base_init_pos[1]
        default_root_state[:, 2] = self.scene.env_origins[env_ids_t, 2] + self.cfg.base_init_pos[2]
        self._target_anchor_w[env_ids_t, 0] = default_root_state[:, 0]
        self._target_anchor_w[env_ids_t, 1] = default_root_state[:, 1]
        self._target_anchor_w[env_ids_t, 2] = default_root_state[:, 2]

        joint_pos = self.turret.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.turret.data.default_joint_vel[env_ids_t].clone()
        joint_pos[:, int(self._yaw_dof_ids[0])] = 0.0
        joint_pos[:, int(self._pitch_dof_ids[0])] = 0.0
        joint_vel[:, int(self._yaw_dof_ids[0])] = 0.0
        joint_vel[:, int(self._pitch_dof_ids[0])] = 0.0

        self._turret_pos_targets[env_ids_t, 0] = joint_pos[:, int(self._yaw_dof_ids[0])]
        self._turret_pos_targets[env_ids_t, 1] = joint_pos[:, int(self._pitch_dof_ids[0])]
        dof_ids = [int(self._yaw_dof_ids[0]), int(self._pitch_dof_ids[0])]
        env_ids_sim = cast(Sequence[int], env_ids_t)
        self.turret.set_joint_position_target(
            self._turret_pos_targets[env_ids_t],
            joint_ids=dof_ids,
            env_ids=env_ids_sim,
        )

        # ===== reset debug: 写入仿真前检查数据是否正常 =====
        # if self.cfg.debug_diagnostics:
        #     print("[TurretAimEnv][reset debug] default_root_state finite =", torch.isfinite(default_root_state).all().item())
        #     print("[TurretAimEnv][reset debug] joint_pos finite =", torch.isfinite(joint_pos).all().item())
        #     print("[TurretAimEnv][reset debug] joint_vel finite =", torch.isfinite(joint_vel).all().item())
        #     print("[TurretAimEnv][reset debug] default_root_state[0] =", default_root_state[0].detach().cpu().tolist())
        #     print("[TurretAimEnv][reset debug] joint_pos[0] =", joint_pos[0].detach().cpu().tolist())
        #     print("[TurretAimEnv][reset debug] joint_vel[0] =", joint_vel[0].detach().cpu().tolist())

        self.turret.write_root_pose_to_sim(default_root_state[:, :7], env_ids_sim)
        self.turret.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_sim)
        self.turret.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_sim)
        # self.turret.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        # self.turret.set_joint_velocity_target(joint_vel, env_ids=env_ids_t)

        # sample target position/velocity
        self._resample_targets(env_ids_t, sample_position=True, sample_velocity=True)

        if self._target_sphere_visuals_enabled:
            self._update_rendered_target_spheres(env_ids_t)

        # 记录该 episode 的起始目标位置（在 done 时统一计入评估统计）
        self._episode_start_target_pos_w[env_ids_t] = self._target_pos_w[env_ids_t]

        # visualize target
        if self._target_markers is not None and self._target_markers.is_visible():
            # build full translations array (num_envs, 3)
            translations = self._target_pos_w.detach().cpu().numpy()
            self._target_markers.visualize(translations=translations)

        # reset 后立刻把 proxy render camera 同步到最新 runtime pose，
        # 避免新 episode 前几帧仍然读到上一回合/默认位姿画面
        if self.cfg.enable_vision_supervision and env_ids_t.numel() > 0:
            try:
                self._push_runtime_camera_pose_to_usd(env_ids_t)
            except Exception as exc:
                if self.cfg.debug_diagnostics:
                    print(f"[vision] proxy pose push failed in _reset_idx: {exc}")


    def _update_rendered_target_spheres(self, env_ids_t: torch.Tensor | None = None) -> None:
        """Move pure-visual target spheres to self._target_pos_w.

        注意：
        - 这里不要用 XformCommonAPI.SetTranslate()
        - 直接查找/创建 xformOp:translate
        - 写入的是 env-local 坐标，而不是 world 坐标
        """
        if not self._target_sphere_visuals_enabled:
            return

        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom, Gf, Usd

        stage = get_current_stage()
        if stage is None:
            print("[target sphere visual update] stage is None")
            return

        if env_ids_t is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if env_ids_t.numel() == 0:
            return

        env_ids_list = env_ids_t.detach().cpu().tolist()
        target_pos_cpu = self._target_pos_w[env_ids_t].detach().cpu()
        env_origins_cpu = self.scene.env_origins[env_ids_t].detach().cpu()

        moved_count = 0
        invalid_count = 0

        for local_idx, env_id in enumerate(env_ids_list):
            prim_path = self._target_sphere_prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                invalid_count += 1
                print(f"[target sphere visual update] invalid prim: {prim_path}")
                continue

            xform = UsdGeom.Xformable(prim)

            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            if translate_op is None:
                translate_op = xform.AddTranslateOp(
                    UsdGeom.XformOp.PrecisionDouble,
                    "",
                    False,
                )

            # world -> env local
            p_w = target_pos_cpu[local_idx]
            env_origin = env_origins_cpu[local_idx]
            p_local = p_w - env_origin

            translate_op.Set(
                Gf.Vec3d(
                    float(p_local[0]),
                    float(p_local[1]),
                    float(p_local[2]),
                ),
                Usd.TimeCode.Default(),
            )
            moved_count += 1

        if int(self.common_step_counter) % 50 == 0 and len(env_ids_list) > 0 and self.cfg.debug_diagnostics:
            debug_local_idx = 0
            debug_id = int(env_ids_list[debug_local_idx])
            if hasattr(self, "_mix_robot_dynamic"):
                dynamic_local_ids = torch.nonzero(
                    self._mix_robot_dynamic[env_ids_t],
                    as_tuple=False,
                ).squeeze(-1)
                if dynamic_local_ids.numel() > 0:
                    debug_local_idx = int(dynamic_local_ids[0].detach().cpu().item())
                    debug_id = int(env_ids_list[debug_local_idx])

            target_pos_w = target_pos_cpu[debug_local_idx].tolist()
            target_pos_local = (
                target_pos_cpu[debug_local_idx] - env_origins_cpu[debug_local_idx]
            ).tolist()
            root_pos_w = self.turret.data.root_pos_w[debug_id].detach().cpu().tolist()
            root_lin_vel_b = self.turret.data.root_lin_vel_b[debug_id].detach().cpu().tolist()
            locomotion_cmd = self._locomotion_commands[debug_id].detach().cpu().tolist()
            robot_dynamic = (
                bool(self._mix_robot_dynamic[debug_id].detach().cpu().item())
                if hasattr(self, "_mix_robot_dynamic")
                else None
            )
            print(
                "[target sphere visual update]",
                f"step={int(self.common_step_counter)}",
                f"moved={moved_count}",
                f"invalid={invalid_count}",
                f"debug_env={debug_id}",
                f"robot_dynamic={robot_dynamic}",
                f"target_path={self._target_sphere_prim_paths[debug_id]}",
                f"target_pos_w={target_pos_w}",
                f"target_pos_local={target_pos_local}",
                f"root_pos_w={root_pos_w}",
                f"root_lin_vel_b={root_lin_vel_b}",
                f"locomotion_cmd={locomotion_cmd}",
            )


    def _enforce_target_above_ground(self, env_ids_t: torch.Tensor | None = None) -> None:
        """Keep the target sphere above ground without imposing a flight-height ceiling."""
        if env_ids_t is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long).flatten()
        if env_ids_t.numel() == 0:
            return

        eps = 1e-3
        ground_z = self.scene.env_origins[env_ids_t, 2]
        min_center_z = ground_z + float(self.cfg.target_sphere_radius) + eps
        below_ground = self._target_pos_w[env_ids_t, 2] < min_center_z
        if torch.any(below_ground):
            below_ids = env_ids_t[below_ground]
            self._target_pos_w[below_ids, 2] = min_center_z[below_ground]
            # Reflect downward motion so the target does not remain pinned to the floor.
            self._target_vel_w[below_ids, 2] = torch.abs(self._target_vel_w[below_ids, 2])
            self._target_acc_w[below_ids, 2] = torch.abs(self._target_acc_w[below_ids, 2])
    
    
    def _create_dynamic_target_multimesh_raycast_sensor(self) -> None:
        """创建挂在 TORSO 上的多射线 MultiMeshRayCaster。"""

        h_fov = float(self.cfg.dynamic_target_raycast_horizontal_fov_deg)
        v_fov = float(self.cfg.dynamic_target_raycast_vertical_fov_deg)
        h_res = float(self.cfg.dynamic_target_raycast_horizontal_resolution_deg)
        v_res = float(self.cfg.dynamic_target_raycast_vertical_resolution_deg)

        # 用垂直 FOV 和垂直角分辨率估算通道数
        if v_res <= 0.0:
            channels = 1
        else:
            channels = max(1, int(round(v_fov / v_res)) + 1)

        sensor_cfg = self.cfg.dynamic_target_multimesh_raycast_cfg.replace(
            prim_path=self.cfg.dynamic_target_raycast_parent_prim_path_expr,
            offset=MultiMeshRayCasterCfg.OffsetCfg(
                pos=self.cfg.dynamic_target_raycast_translation,
                rot=self.cfg.dynamic_target_raycast_orientation_wxyz,
            ),
            pattern_cfg=patterns.LidarPatternCfg(
                channels=channels,
                vertical_fov_range=(-0.5 * v_fov, 0.5 * v_fov),
                horizontal_fov_range=(-0.5 * h_fov, 0.5 * h_fov),
                horizontal_res=h_res,
            ),
            max_distance=float(self.cfg.dynamic_target_raycast_max_range),
        )

        self._dynamic_target_raycast_sensor = MultiMeshRayCaster(sensor_cfg)
        self.scene.sensors["dynamic_target_raycast"] = self._dynamic_target_raycast_sensor

        if self.cfg.debug_diagnostics:
            print(
                "[raycast dbg] MultiMeshRayCaster initialized on TORSO:",
                f"prim_path={self.cfg.dynamic_target_raycast_parent_prim_path_expr}",
                f"h_fov={h_fov}",
                f"v_fov={v_fov}",
                f"h_res={h_res}",
                f"v_res={v_res}",
                f"channels={channels}",
            )

    def set_dynamic_target_raycast_angles(self, yaw: torch.Tensor, pitch: torch.Tensor) -> None:
        """由外部视觉模块设置目标方向角（相对于相机光心）。

        注意：
        1) 这里只记录“要查询的方向”，不旋转传感器本体；
        2) 这里的数据源必须是视觉预测，不能用 GT 覆盖。
        """
        yaw = torch.as_tensor(yaw, device=self.device, dtype=torch.float32)
        pitch = torch.as_tensor(pitch, device=self.device, dtype=torch.float32)

        if yaw.ndim > 1:
            yaw = yaw.squeeze(-1)
        if pitch.ndim > 1:
            pitch = pitch.squeeze(-1)

        if yaw.numel() != self.num_envs or pitch.numel() != self.num_envs:
            raise RuntimeError(
                "set_dynamic_target_raycast_angles got wrong shape: "
                f"yaw={tuple(yaw.shape)}, pitch={tuple(pitch.shape)}, num_envs={self.num_envs}"
            )

        yaw = torch.nan_to_num(yaw, nan=0.0, posinf=0.0, neginf=0.0).clone()
        pitch = torch.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0).clone()

        self._external_raycast_yaw = yaw
        self._external_raycast_pitch = pitch


    def run_dynamic_target_raycast_for_current_frame(self) -> None:
        """Use the *already-set* external yaw/pitch to re-filter the *current-frame* ray hits immediately.

        这个方法的作用是：
        1) env.step() 之后，当前帧传感器数据已经在 scene / sensor 中；
        2) 外部采集脚本拿当前帧 RGB 做视觉预测；
        3) 通过 set_dynamic_target_raycast_angles() 写入当前帧视觉角度；
        4) 立刻调用本函数，在“当前帧”上重新执行一次 raycast 筛选；
        5) 然后再读取 lidar_distance / refined_yaw / refined_pitch。

        这样可避免“上一帧视觉预测去筛当前帧雷达”的一拍时延。
        """
        if not self.cfg.enable_dynamic_target_raycast_debug:
            return
        if self._dynamic_target_raycast_sensor is None:
            return

        self._debug_raycast_dynamic_target()

            

    def get_dynamic_target_raycast_distance(self) -> torch.Tensor:
        """返回当前每个 env 的 raycast 命中距离。
        
        返回:
            distance: [num_envs]
                - 命中目标球：返回从炮口到命中点的距离
                - 未命中：返回 inf
        """
        if not self.cfg.enable_dynamic_target_raycast_debug:
            raise RuntimeError("Dynamic target raycast is disabled.")
        if self._dynamic_target_raycast_sensor is None:
            raise RuntimeError("Dynamic target raycast sensor is not initialized.")
        return self._raycast_hit_distance.clone()

    def get_dynamic_target_raycast_hit_mask(self) -> torch.Tensor:
        """返回当前每个 env 是否命中目标球。"""
        if not self.cfg.enable_dynamic_target_raycast_debug:
            raise RuntimeError("Dynamic target raycast is disabled.")
        if self._dynamic_target_raycast_sensor is None:
            raise RuntimeError("Dynamic target raycast sensor is not initialized.")
        return self._raycast_hit_valid.clone()


    def _debug_raycast_dynamic_target(self) -> None:
        """用视觉方向先验，在 MultiMeshRayCaster 的 hit 结果中做角窗筛选与稳健平均。

        输出:
            self._raycast_hit_distance   : [N]  相机到目标中心的估计距离（不是表面距离）
            self._raycast_hit_valid      : [N]  是否成功得到有效估计
            self._raycast_expected_surface_distance : [N] 调试用，保存估计表面距离
            self._raycast_refined_yaw    : [N]  可选：修正后的相机系 yaw
            self._raycast_refined_pitch  : [N]  可选：修正后的相机系 pitch
        """
        if not self.cfg.enable_dynamic_target_raycast_debug:
            return
        if self._dynamic_target_raycast_sensor is None:
            return
        if self._vision_camera is None:
            return

        try:
            self._dynamic_target_raycast_sensor.update(dt=self.physics_dt)
            data = self._dynamic_target_raycast_sensor.data

            sensor_pos_w = data.pos_w            # [N, 3]
            ray_hits_w = data.ray_hits_w         # [N, R, 3]

            num_envs = ray_hits_w.shape[0]
            env_ids = torch.arange(num_envs, device=self.device)

            # ------------------------------------------------------------
            # lazy init 调试缓冲
            # ------------------------------------------------------------
            if (not hasattr(self, "_raycast_refined_yaw")) or (self._raycast_refined_yaw.shape[0] != num_envs):
                self._raycast_refined_yaw = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)
                self._raycast_refined_pitch = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)

            # ------------------------------------------------------------
            # 1) 当前想查询的目标方向（相机系）
            # ------------------------------------------------------------
            if hasattr(self, "_external_raycast_yaw") and hasattr(self, "_external_raycast_pitch"):
                yaw_cam = self._external_raycast_yaw
                pitch_cam = self._external_raycast_pitch
                angle_src = "external"
            else:
                yaw_cam = self._vision_direction_gt[:, 0]
                pitch_cam = self._vision_direction_gt[:, 1]
                angle_src = "gt_fallback"

            yaw_cam = yaw_cam.to(device=self.device, dtype=torch.float32).view(-1)
            pitch_cam = pitch_cam.to(device=self.device, dtype=torch.float32).view(-1)

            # 先全部置为“无有效 hit”
            # 但 refined 默认回退到当前送给 raycast 的角度，而不是 0
            self._raycast_hit_valid[:] = False
            self._raycast_hit_distance[:] = float("inf")
            self._raycast_expected_surface_distance[:] = float("inf")
            self._raycast_refined_yaw[:] = yaw_cam
            self._raycast_refined_pitch[:] = pitch_cam

            # ROS optical frame:
            # x-right, y-down, z-forward
            dir_cam_x = torch.sin(yaw_cam) * torch.cos(pitch_cam)
            dir_cam_y = -torch.sin(pitch_cam)
            dir_cam_z = torch.cos(yaw_cam) * torch.cos(pitch_cam)
            desired_dir_cam = torch.stack([dir_cam_x, dir_cam_y, dir_cam_z], dim=-1)  # [N, 3]
            desired_dir_cam = desired_dir_cam / torch.norm(desired_dir_cam, dim=-1, keepdim=True).clamp_min(1e-6)

            cam_pos_w, cam_quat_w = self._get_runtime_camera_pose_w()

            desired_dir_w = quat_apply(cam_quat_w, desired_dir_cam)
            desired_dir_w = desired_dir_w / torch.norm(desired_dir_w, dim=-1, keepdim=True).clamp_min(1e-6)

            # ------------------------------------------------------------
            # 2) ray hit 几何
            #    注意：IsaacLab 2.3.2 里没有 ray_directions_w，
            #    这里只能从 hit 点反推“命中方向”
            #    但由于当前 MultiMeshRayCaster 只打 TargetSphere，
            #    用 hit 点筛角窗是可行的。
            # ------------------------------------------------------------
            valid_hits_mask = torch.isfinite(ray_hits_w).all(dim=-1)  # [N, R]

            vec_to_hit_w = ray_hits_w - sensor_pos_w.unsqueeze(1)     # [N, R, 3]
            hit_dist_sensor = torch.norm(vec_to_hit_w, dim=-1)        # [N, R]

            hit_dirs_w = vec_to_hit_w / hit_dist_sensor.unsqueeze(-1).clamp_min(1e-6)

            # ------------------------------------------------------------
            # 3) 角窗筛选（张量化）
            # ------------------------------------------------------------
            h_res_deg = float(self.cfg.dynamic_target_raycast_horizontal_resolution_deg)
            v_res_deg = float(self.cfg.dynamic_target_raycast_vertical_resolution_deg)

            # 放宽角窗
            yaw_half_window_deg_1 = max(4.0 * h_res_deg, 2.5)
            pitch_half_window_deg_1 = max(4.0 * v_res_deg, 2.5)

            yaw_half_window_deg_2 = max(8.0 * h_res_deg, 5.0)
            pitch_half_window_deg_2 = max(8.0 * v_res_deg, 5.0)

            cone_deg_1 = max(yaw_half_window_deg_1, pitch_half_window_deg_1)
            cone_deg_2 = max(yaw_half_window_deg_2, pitch_half_window_deg_2)

            cos_thr_1 = math.cos(math.radians(cone_deg_1))
            cos_thr_2 = math.cos(math.radians(cone_deg_2))

            dir_cos = torch.sum(hit_dirs_w * desired_dir_w.unsqueeze(1), dim=-1)  # [N, R]

            cand_mask_1 = valid_hits_mask & (dir_cos >= cos_thr_1)  # [N, R]
            cand_mask_2 = valid_hits_mask & (dir_cos >= cos_thr_2)  # [N, R]

            # 第一层有候选就优先用第一层，否则 fallback 到第二层
            cand_count_1 = cand_mask_1.sum(dim=1)  # [N]
            use_mask_1 = cand_count_1 >= 1
            final_cand_mask = torch.where(use_mask_1.unsqueeze(1), cand_mask_1, cand_mask_2)  # [N, R]

            cand_count = final_cand_mask.sum(dim=1)  # [N]
            valid_env_mask = cand_count >= 1  # [N]

            target_radius = float(self.cfg.target_sphere_radius)

            # ------------------------------------------------------------
            # 4) 表面距离稳健聚类（张量化）
            # ------------------------------------------------------------
            # 用 +inf 屏蔽无效候选，再做按行排序求“有效候选集合”的中位数
            inf_tensor = torch.full_like(hit_dist_sensor, float("inf"))
            masked_d = torch.where(final_cand_mask, hit_dist_sensor, inf_tensor)  # [N, R]
            sorted_d, _ = torch.sort(masked_d, dim=1)  # [N, R]

            # 中位数索引：这里取 lower median，避免偶数个候选时再做额外 gather 两次
            median_idx = torch.clamp((cand_count - 1) // 2, min=0)  # [N]
            med = sorted_d.gather(1, median_idx.unsqueeze(1)).squeeze(1)  # [N]

            # 对无有效候选的 env，med 设为 0，后面会用 valid_env_mask 屏蔽
            med = torch.where(valid_env_mask, med, torch.zeros_like(med))

            tol = torch.maximum(
                torch.full_like(med, 0.35),
                0.10 * med,
            )  # [N]

            cluster_mask = final_cand_mask & (torch.abs(hit_dist_sensor - med.unsqueeze(1)) <= tol.unsqueeze(1))  # [N, R]

            # 如果某个 env 聚类后空了，则回退到 final_cand_mask
            cluster_count = cluster_mask.sum(dim=1)  # [N]
            use_cluster = cluster_count >= 1
            final_cluster_mask = torch.where(use_cluster.unsqueeze(1), cluster_mask, final_cand_mask)  # [N, R]

            final_cluster_count = final_cluster_mask.sum(dim=1)  # [N]
            final_valid_mask = valid_env_mask & (final_cluster_count >= 1)  # [N]

            # ------------------------------------------------------------
            # 5) 对候选束做加权均值（张量化）
            #    注意：不能直接用 ray_hits_w * weights。
            #    因为无效 hit 往往是 inf/nan，哪怕 weight=0，inf*0 仍会传播成 nan。
            #    所以必须先把无效项清零，再参与加权求和。
            # ------------------------------------------------------------
            weights = final_cluster_mask.to(ray_hits_w.dtype)  # [N, R]
            denom = final_cluster_count.to(ray_hits_w.dtype).clamp_min(1.0).unsqueeze(1)  # [N, 1]

            zero_hits = torch.zeros_like(ray_hits_w)
            safe_ray_hits_w = torch.where(
                valid_hits_mask.unsqueeze(-1),
                ray_hits_w,
                zero_hits,
            )  # [N, R, 3]

            zero_dirs = torch.zeros_like(hit_dirs_w)
            safe_hit_dirs_w = torch.where(
                valid_hits_mask.unsqueeze(-1),
                hit_dirs_w,
                zero_dirs,
            )  # [N, R, 3]

            zero_dists = torch.zeros_like(hit_dist_sensor)
            safe_hit_dist_sensor = torch.where(
                valid_hits_mask,
                hit_dist_sensor,
                zero_dists,
            )  # [N, R]

            mean_hit_w = (safe_ray_hits_w * weights.unsqueeze(-1)).sum(dim=1) / denom  # [N, 3]
            mean_dir_w_sensor = (safe_hit_dirs_w * weights.unsqueeze(-1)).sum(dim=1) / denom  # [N, 3]
            mean_surface_dist_sensor = (
                (safe_hit_dist_sensor * weights).sum(dim=1)
                / final_cluster_count.to(hit_dist_sensor.dtype).clamp_min(1.0)
            )  # [N]

            mean_dir_norm = torch.norm(mean_dir_w_sensor, dim=-1, keepdim=True)
            mean_dir_w_sensor = mean_dir_w_sensor / mean_dir_norm.clamp_min(1e-6)

            # ------------------------------------------------------------
            # 6) 从“表面 hit 点”恢复“目标中心点”（张量化）
            # ------------------------------------------------------------
            cam_to_hit_w = mean_hit_w - cam_pos_w  # [N, 3]
            cam_to_hit_norm = torch.norm(cam_to_hit_w, dim=-1, keepdim=True)
            cam_to_hit_dir_w = cam_to_hit_w / cam_to_hit_norm.clamp_min(1e-6)

            est_center_w = mean_hit_w + cam_to_hit_dir_w * target_radius  # [N, 3]

            center_vec_cam_w = est_center_w - cam_pos_w  # [N, 3]
            center_dist_cam = torch.norm(center_vec_cam_w, dim=-1)  # [N]

            geom_valid_mask = (
                torch.isfinite(mean_hit_w).all(dim=-1)
                & torch.isfinite(mean_dir_w_sensor).all(dim=-1)
                & torch.isfinite(mean_surface_dist_sensor)
                & torch.isfinite(est_center_w).all(dim=-1)
                & torch.isfinite(center_vec_cam_w).all(dim=-1)
                & torch.isfinite(center_dist_cam)
                & (center_dist_cam > 1e-6)
            )
            final_valid_mask = final_valid_mask & geom_valid_mask

            # ------------------------------------------------------------
            # 7) 反算 refined yaw/pitch（相机系，张量化）
            # ------------------------------------------------------------
            center_vec_cam_local = self._rotate_world_to_local(
                cam_quat_w,
                center_vec_cam_w,
            )  # [N, 3]

            x = center_vec_cam_local[:, 0]
            y = center_vec_cam_local[:, 1]
            z = center_vec_cam_local[:, 2].clamp_min(1e-6)

            refined_yaw = torch.atan2(x, z)  # [N]
            refined_pitch = torch.atan2(-y, torch.sqrt(x * x + z * z).clamp_min(1e-6))  # [N]

            # ------------------------------------------------------------
            # 8) 写回（张量化）
            # ------------------------------------------------------------
            self._raycast_hit_valid[:] = final_valid_mask
            self._raycast_hit_distance[:] = torch.where(
                final_valid_mask,
                center_dist_cam,
                self._raycast_hit_distance,
            )
            self._raycast_expected_surface_distance[:] = torch.where(
                final_valid_mask,
                mean_surface_dist_sensor,
                self._raycast_expected_surface_distance,
            )
            self._raycast_refined_yaw[:] = torch.where(
                final_valid_mask,
                refined_yaw,
                self._raycast_refined_yaw,
            )
            self._raycast_refined_pitch[:] = torch.where(
                final_valid_mask,
                refined_pitch,
                self._raycast_refined_pitch,
            )

            # ------------------------------------------------------------
            # 9) 调试打印（增强版）
            #    输出：
            #      - TORSO 世界位姿（仿真数据）
            #      - camera prim 的 USD 世界位姿
            #      - _vision_camera.data 的世界位姿
            #      - runtime 重建相机位姿
            #      - pred / gt / refined yaw/pitch
            #      - raycast 距离与有效性
            # ------------------------------------------------------------
            if not hasattr(self, "_raycast_debug_counter"):
                self._raycast_debug_counter = 0
            self._raycast_debug_counter += 1

            every = max(1, int(self.cfg.dynamic_target_raycast_debug_every))
            if self.cfg.debug_diagnostics and (self._raycast_debug_counter % every == 0) and (num_envs > 0):
                try:
                    from isaacsim.core.utils.stage import get_current_stage

                    def _usd_world_pose_from_prim_path(prim_path: str):
                        """读取某个 USD prim 的世界位姿，返回 (pos_xyz, quat_wxyz)；失败时返回 (None, None)。"""
                        stage = get_current_stage()
                        prim = stage.GetPrimAtPath(prim_path)
                        if not prim.IsValid():
                            return None, None

                        xformable = UsdGeom.Xformable(prim)
                        world_mat = xformable.ComputeLocalToWorldTransform(0.0)

                        t = world_mat.ExtractTranslation()
                        pos = [float(t[0]), float(t[1]), float(t[2])]

                        rot = world_mat.ExtractRotationQuat()
                        quat = [
                            float(rot.GetReal()),
                            float(rot.GetImaginary()[0]),
                            float(rot.GetImaginary()[1]),
                            float(rot.GetImaginary()[2]),
                        ]
                        return pos, quat

                    # ------------------------------------------------------------------
                    # env_0 prim path
                    # ------------------------------------------------------------------
                    torso_matches = sim_utils.find_matching_prim_paths(
                        self.cfg.dynamic_target_raycast_parent_prim_path_expr
                    )
                    camera_matches = sim_utils.find_matching_prim_paths(
                        self.cfg.camera_prim_path_expr
                    )

                    torso_prim_path_0 = torso_matches[0] if len(torso_matches) > 0 else None
                    camera_prim_path_0 = camera_matches[0] if len(camera_matches) > 0 else None

                    # ------------------------------------------------------------------
                    # 1) TORSO 世界位姿（仿真数据）
                    # ------------------------------------------------------------------
                    torso_pos_w_sim = [float(v) for v in self.turret.data.root_pos_w[0].detach().cpu().tolist()]
                    torso_quat_w_sim = [float(v) for v in self.turret.data.root_quat_w[0].detach().cpu().tolist()]

                    # ------------------------------------------------------------------
                    # 2) TORSO / Camera prim 的 USD 世界位姿
                    # ------------------------------------------------------------------
                    torso_pos_w_usd, torso_quat_w_usd = (None, None)
                    if torso_prim_path_0 is not None:
                        torso_pos_w_usd, torso_quat_w_usd = _usd_world_pose_from_prim_path(torso_prim_path_0)

                    camera_pos_w_usd, camera_quat_w_usd = (None, None)
                    if camera_prim_path_0 is not None:
                        camera_pos_w_usd, camera_quat_w_usd = _usd_world_pose_from_prim_path(camera_prim_path_0)

                    # ------------------------------------------------------------------
                    # 3) _vision_camera.data 世界位姿
                    # ------------------------------------------------------------------
                    cam_pos_w_data = None
                    cam_quat_w_data = None
                    if (self._vision_camera is not None) and hasattr(self._vision_camera, "data"):
                        if hasattr(self._vision_camera.data, "pos_w") and self._vision_camera.data.pos_w is not None:
                            cam_pos_w_data = [float(v) for v in self._vision_camera.data.pos_w[0].detach().cpu().tolist()]
                        if hasattr(self._vision_camera.data, "quat_w_world") and self._vision_camera.data.quat_w_world is not None:
                            cam_quat_w_data = [float(v) for v in self._vision_camera.data.quat_w_world[0].detach().cpu().tolist()]
                        elif hasattr(self._vision_camera.data, "quat_w_ros") and self._vision_camera.data.quat_w_ros is not None:
                            cam_quat_w_data = [float(v) for v in self._vision_camera.data.quat_w_ros[0].detach().cpu().tolist()]
                        elif hasattr(self._vision_camera.data, "quat_w_opengl") and self._vision_camera.data.quat_w_opengl is not None:
                            cam_quat_w_data = [float(v) for v in self._vision_camera.data.quat_w_opengl[0].detach().cpu().tolist()]
                        elif hasattr(self._vision_camera.data, "quat_w") and self._vision_camera.data.quat_w is not None:
                            cam_quat_w_data = [float(v) for v in self._vision_camera.data.quat_w[0].detach().cpu().tolist()]

                    # ------------------------------------------------------------------
                    # 4) runtime 重建相机位姿
                    # ------------------------------------------------------------------
                    cam_pos_w_rt, cam_quat_w_rt = self._get_runtime_camera_pose_w()
                    cam_pos_w_rt_0 = [float(v) for v in cam_pos_w_rt[0].detach().cpu().tolist()]
                    cam_quat_w_rt_0 = [float(v) for v in cam_quat_w_rt[0].detach().cpu().tolist()]

                    # ------------------------------------------------------------------
                    # 5) pred / gt / refined
                    # ------------------------------------------------------------------
                    pred_yaw_0 = float(yaw_cam[0].detach().cpu().item())
                    pred_pitch_0 = float(pitch_cam[0].detach().cpu().item())

                    gt_yaw_0 = float(self._vision_direction_gt[0, 0].detach().cpu().item())
                    gt_pitch_0 = float(self._vision_direction_gt[0, 1].detach().cpu().item())

                    refined_yaw_0 = float(self._raycast_refined_yaw[0].detach().cpu().item())
                    refined_pitch_0 = float(self._raycast_refined_pitch[0].detach().cpu().item())

                    ray_valid_0 = bool(self._raycast_hit_valid[0].detach().cpu().item())
                    ray_dist_0 = float(self._raycast_hit_distance[0].detach().cpu().item())
                    surf_dist_0 = float(self._raycast_expected_surface_distance[0].detach().cpu().item())

                    print(
                        "[camera pose dbg]",
                        f"step={int(self.common_step_counter)}",
                        "NOTE=runtime_pose_used_for_all_geometry",
                        f"torso_prim_path_0={torso_prim_path_0}",
                        f"camera_prim_path_0={camera_prim_path_0}",
                    )

                    print(
                        "[camera pose dbg][runtime]",
                        f"step={int(self.common_step_counter)}",
                        f"camera_pos_w_rt={cam_pos_w_rt_0}",
                        f"camera_quat_w_rt={cam_quat_w_rt_0}",
                    )

                    print(
                        "[camera pose dbg][sim torso]",
                        f"step={int(self.common_step_counter)}",
                        f"torso_pos_w_sim={torso_pos_w_sim}",
                        f"torso_quat_w_sim={torso_quat_w_sim}",
                    )

                    print(
                        "[camera pose dbg][sensor data only for compare]",
                        f"step={int(self.common_step_counter)}",
                        f"camera_pos_w_data={cam_pos_w_data}",
                        f"camera_quat_w_data={cam_quat_w_data}",
                    )

                    print(
                        "[camera pose dbg][usd authored only for compare]",
                        f"step={int(self.common_step_counter)}",
                        f"torso_pos_w_usd={torso_pos_w_usd}",
                        f"torso_quat_w_usd={torso_quat_w_usd}",
                        f"camera_pos_w_usd={camera_pos_w_usd}",
                        f"camera_quat_w_usd={camera_quat_w_usd}",
                    )

                    print(
                        "[raycast dbg pose]",
                        f"step={int(self.common_step_counter)}",
                        f"cam_pos_w_rt={cam_pos_w_rt_0}",
                        f"cam_quat_w_rt={cam_quat_w_rt_0}",
                    )

                    print(
                        "[raycast dbg sensor-data-compare]",
                        f"step={int(self.common_step_counter)}",
                        f"cam_pos_w_data={cam_pos_w_data}",
                        f"cam_quat_w_data={cam_quat_w_data}",
                    )

                    print(
                        "[raycast dbg]",
                        f"step={int(self.common_step_counter)}",
                        f"angle_src={angle_src}",
                        f"pred_yaw[0]={pred_yaw_0:.4f}",
                        f"pred_pitch[0]={pred_pitch_0:.4f}",
                        f"gt_yaw[0]={gt_yaw_0:.4f}",
                        f"gt_pitch[0]={gt_pitch_0:.4f}",
                        f"refined_yaw[0]={refined_yaw_0:.4f}",
                        f"refined_pitch[0]={refined_pitch_0:.4f}",
                        f"ray_valid[0]={ray_valid_0}",
                        f"ray_dist[0]={ray_dist_0:.4f}",
                        f"surface_dist[0]={surf_dist_0:.4f}",
                        f"valid_hits={int(valid_hits_mask[0].sum().item())}/{int(valid_hits_mask.shape[1])}",
                        f"cand_count={int(cand_count[0].item())}",
                        f"cluster_count={int(final_cluster_count[0].item())}",
                    )

                except Exception as e:
                    print(f"[camera pose dbg] failed: {e}")

        except Exception as exc:
            print(f"[raycast dbg] runtime failed: {exc}")
            traceback.print_exc()
            self._raycast_hit_valid[:] = False
            self._raycast_hit_distance[:] = float("inf")
            self._raycast_expected_surface_distance[:] = float("inf")
            if hasattr(self, "_raycast_refined_yaw"):
                self._raycast_refined_yaw[:] = 0.0
            if hasattr(self, "_raycast_refined_pitch"):
                self._raycast_refined_pitch[:] = 0.0
    
    
    def _sample_targets_in_camera_sector(self, env_ids_t: torch.Tensor) -> None:
        """Sample target positions directly in the camera viewing sector.

        逻辑：
        1. 在相机坐标系下采样 yaw / pitch / range
        2. 构造 camera-frame 方向向量
        3. 旋转到 world frame
        4. 加到 camera world position 上，得到 target_pos_w
        """
        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized, cannot sample target in camera sector.")

        env_count = int(env_ids_t.numel())
        if env_count == 0:
            return

        cam_pos_w_all, cam_quat_w_all = self._get_runtime_camera_pose_w()
        cam_pos_w = cam_pos_w_all[env_ids_t]       # [M, 3]
        cam_quat_w = cam_quat_w_all[env_ids_t]     # [M, 4]

        r = sample_uniform(
            self.cfg.target_radius_range[0],
            self.cfg.target_radius_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)

        yaw = sample_uniform(
            self.cfg.target_cam_yaw_range[0],
            self.cfg.target_cam_yaw_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)

        pitch = sample_uniform(
            self.cfg.target_cam_pitch_range[0],
            self.cfg.target_cam_pitch_range[1],
            (env_count, 1),
            self.device,
        ).squeeze(-1)

        # ROS optical frame:
        # x -> right
        # y -> down
        # z -> forward
        #
        # 令 pitch>0 表示目标在图像上方，因此 camera-frame 里 y 要取负
        x_cam = r * torch.sin(yaw) * torch.cos(pitch)
        y_cam = -r * torch.sin(pitch)
        z_cam = r * torch.cos(yaw) * torch.cos(pitch)

        target_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [M, 3]

        # camera -> world
        target_offset_w = quat_apply(cam_quat_w, target_cam)
        target_pos_w = cam_pos_w + target_offset_w

        self._target_pos_w[env_ids_t] = target_pos_w


    def _sample_placeholder_targets(self, env_ids_t: torch.Tensor) -> None:
        """Sample temporary placeholder targets right after reset.

        这里不依赖 camera pose，只是先给一个稳定、有限的占位目标，
        避免 reset 当步就读取 stale camera pose 做正式采样。
        """
        if env_ids_t.numel() == 0:
            return

        env_count = int(env_ids_t.numel())

        # 给一个比较保守的前方占位点
        r = sample_uniform(8.0, 10.0, (env_count, 1), self.device)
        yaw = sample_uniform(-0.08, 0.08, (env_count, 1), self.device)
        z = sample_uniform(0.9, 1.2, (env_count, 1), self.device)

        target_pos_rel = torch.cat([r * torch.cos(yaw), r * torch.sin(yaw), z], dim=-1)
        self._target_pos_w[env_ids_t] = self._target_anchor_w[env_ids_t] + target_pos_rel


    def _sample_target_velocities(self, count: int) -> torch.Tensor:
        """Sample total speed uniformly and an independent isotropic 3-D direction."""
        speed_min, speed_max = (float(value) for value in self.cfg.target_speed_range)
        if not (math.isfinite(speed_min) and math.isfinite(speed_max) and 0.0 <= speed_min <= speed_max):
            raise ValueError("target_speed_range must be finite and satisfy 0 <= min <= max")
        speed = sample_uniform(speed_min, speed_max, (count, 1), self.device)
        azimuth = sample_uniform(-math.pi, math.pi, (count, 1), self.device)
        # Uniform elevation angles would bias directions toward the poles.
        vertical = sample_uniform(-1.0, 1.0, (count, 1), self.device)
        horizontal = torch.sqrt((1.0 - vertical.square()).clamp_min(0.0))
        direction = torch.cat(
            (horizontal * torch.cos(azimuth), horizontal * torch.sin(azimuth), vertical), dim=-1
        )
        return speed * direction

    def _maybe_finalize_pending_target_resample(self) -> None:
        """After reset, once camera pose is stable, finalize target sampling in camera sector."""
        if not self.cfg.sample_target_in_camera_sector:
            return
        if self._vision_camera is None:
            return

        pending_mask = self._pending_target_resample & (self._vision_collect_warmup_left <= 0)
        env_ids_t = torch.nonzero(pending_mask, as_tuple=False).squeeze(-1)
        if env_ids_t.numel() == 0:
            return

        # 现在相机 pose 已经更稳定了，再真正按 camera sector 采样
        self._sample_targets_in_camera_sector(env_ids_t)
        self._enforce_target_above_ground(env_ids_t)

        if self.cfg.enable_dynamic_target:
            env_count = int(env_ids_t.numel())
            self._target_vel_w[env_ids_t] = self._sample_target_velocities(env_count)
            
            # --- 新增：随机采样加速度 ---
            acc_xy_mag = sample_uniform(self.cfg.target_acc_range[0], self.cfg.target_acc_range[1], (env_count, 1), self.device)
            acc_heading = sample_uniform(-math.pi, math.pi, (env_count, 1), self.device)
            acc_xy = torch.cat([acc_xy_mag * torch.cos(acc_heading), acc_xy_mag * torch.sin(acc_heading)], dim=-1)
            acc_z = sample_uniform(self.cfg.target_vertical_acc_range[0], self.cfg.target_vertical_acc_range[1], (env_count, 1), self.device)
            self._target_acc_w[env_ids_t] = torch.cat([acc_xy, acc_z], dim=-1)
        else:
            self._target_vel_w[env_ids_t] = 0.0
            self._target_acc_w[env_ids_t] = 0.0

        self._pending_target_resample[env_ids_t] = False
        self._episode_start_target_pos_w[env_ids_t] = self._target_pos_w[env_ids_t]

        if self._target_sphere_visuals_enabled:
            self._update_rendered_target_spheres(env_ids_t)

        if self.cfg.debug_diagnostics:
            print(f"[TurretAimEnv] finalized delayed target resample for {int(env_ids_t.numel())} envs")


    def _resample_targets(self, env_ids_t: torch.Tensor, sample_position: bool, sample_velocity: bool):
        if env_ids_t.numel() == 0:
            return

        env_count = int(env_ids_t.numel())

        if sample_position:
            if self.cfg.sample_target_in_camera_sector and (self._vision_camera is not None):
                # reset 当下先不要直接读 camera pose 做正式采样
                # 先给占位点，等 reset 后若干步再正式采样
                self._sample_placeholder_targets(env_ids_t)
                self._pending_target_resample[env_ids_t] = True
            else:
                r = sample_uniform(
                    self.cfg.target_radius_range[0],
                    self.cfg.target_radius_range[1],
                    (env_count, 1),
                    self.device,
                )
                yaw = sample_uniform(
                    self.cfg.target_yaw_range[0],
                    self.cfg.target_yaw_range[1],
                    (env_count, 1),
                    self.device,
                )
                z = sample_uniform(
                    self.cfg.target_height_range[0],
                    self.cfg.target_height_range[1],
                    (env_count, 1),
                    self.device,
                )

                target_pos_rel = torch.cat([r * torch.cos(yaw), r * torch.sin(yaw), z], dim=-1)
                self._target_pos_w[env_ids_t] = self._target_anchor_w[env_ids_t] + target_pos_rel
                self._pending_target_resample[env_ids_t] = False

            # 约束：目标球心高度需满足 z - radius > 0，避免穿地
            self._enforce_target_above_ground(env_ids_t)

        if sample_velocity:
            if self.cfg.sample_target_in_camera_sector and (self._vision_camera is not None):
                self._target_vel_w[env_ids_t] = 0.0
                self._target_acc_w[env_ids_t] = 0.0
            else:
                if self.cfg.enable_dynamic_target:
                    self._target_vel_w[env_ids_t] = self._sample_target_velocities(env_count)
                    
                    # --- 新增：随机采样加速度 ---
                    acc_xy_mag = sample_uniform(self.cfg.target_acc_range[0], self.cfg.target_acc_range[1], (env_count, 1), self.device)
                    acc_heading = sample_uniform(-math.pi, math.pi, (env_count, 1), self.device)
                    acc_xy = torch.cat([acc_xy_mag * torch.cos(acc_heading), acc_xy_mag * torch.sin(acc_heading)], dim=-1)
                    acc_z = sample_uniform(self.cfg.target_vertical_acc_range[0], self.cfg.target_vertical_acc_range[1], (env_count, 1), self.device)
                    self._target_acc_w[env_ids_t] = torch.cat([acc_xy, acc_z], dim=-1)
                else:
                    self._target_vel_w[env_ids_t] = 0.0
                    self._target_acc_w[env_ids_t] = 0.0


    def _handle_target_out_of_bounds(
        self,
        env_ids_t: torch.Tensor | None = None,
        *,
        fixed_anchor: bool = False,
    ) -> None:
        """Reflect targets at distance/FOV bounds, with optional legacy height bounds.

        ``fixed_anchor=True`` keeps the reset anchor for a static robot. With a
        moving robot, ``fixed_anchor=False`` updates the constraint frame from
        the current robot pose so the target remains inside the moving camera view.
        """
        eps = 1e-6
        if env_ids_t is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long).flatten()
        if env_ids_t.numel() == 0:
            return

        if fixed_anchor:
            anchor_w = self._target_anchor_w[env_ids_t]
        else:
            anchor_w = self.turret.data.root_pos_w[env_ids_t]
            self._target_anchor_w[env_ids_t] = anchor_w

        target_pos_w = self._target_pos_w[env_ids_t].clone()
        target_vel_w = self._target_vel_w[env_ids_t].clone()
        target_acc_w = self._target_acc_w[env_ids_t].clone()

        rel = target_pos_w - anchor_w
        rel_xy = rel[:, :2]
        vel_xy = target_vel_w[:, :2]
        acc_xy = target_acc_w[:, :2]

        r = torch.norm(rel_xy, dim=-1).clamp_min(eps)
        r_min, r_max = float(self.cfg.target_radius_range[0]), float(self.cfg.target_radius_range[1])
        yaw_min, yaw_max = float(self.cfg.target_yaw_range[0]), float(self.cfg.target_yaw_range[1])

        # Optional legacy relative-height boundary. Disabled by default: vertical
        # movement is instead bounded by the camera pitch sector farther below.
        if self.cfg.constrain_target_relative_height:
            z_min = float(self.cfg.target_height_range[0])
            z_max = float(self.cfg.target_height_range[1])
            if z_min > z_max:
                raise ValueError(
                    "target_height_range must be ordered as (min_relative_z, max_relative_z)."
                )
            below_z = rel[:, 2] < z_min
            above_z = rel[:, 2] > z_max
            if torch.any(below_z):
                rel[below_z, 2] = z_min
                target_vel_w[below_z, 2] = torch.abs(target_vel_w[below_z, 2])
                target_acc_w[below_z, 2] = torch.abs(target_acc_w[below_z, 2])
            if torch.any(above_z):
                rel[above_z, 2] = z_max
                target_vel_w[above_z, 2] = -torch.abs(target_vel_w[above_z, 2])
                target_acc_w[above_z, 2] = -torch.abs(target_acc_w[above_z, 2])

        # Relative radius boundary.
        unit_r = rel_xy / r.unsqueeze(-1)
        radial_speed = torch.sum(vel_xy * unit_r, dim=-1, keepdim=True)
        tangential_speed = vel_xy - radial_speed * unit_r
        radial_acc = torch.sum(acc_xy * unit_r, dim=-1, keepdim=True)
        tangential_acc = acc_xy - radial_acc * unit_r
        radial_hit = (r < r_min) | (r > r_max)
        if torch.any(radial_hit):
            r_clamped = torch.clamp(r, min=r_min, max=r_max)
            rel_xy[radial_hit] = unit_r[radial_hit] * r_clamped[radial_hit].unsqueeze(-1)
            target_vel_w[radial_hit, :2] = (
                tangential_speed - radial_speed * unit_r
            )[radial_hit]
            target_acc_w[radial_hit, :2] = (
                tangential_acc - radial_acc * unit_r
            )[radial_hit]

        # Relative yaw sector boundary.
        def _reflect_about_ray(v_xy: torch.Tensor, ray_yaw: float) -> torch.Tensor:
            ray_dir = torch.tensor(
                [math.cos(ray_yaw), math.sin(ray_yaw)],
                device=self.device,
                dtype=v_xy.dtype,
            )
            proj = torch.sum(v_xy * ray_dir.unsqueeze(0), dim=-1, keepdim=True) * ray_dir.unsqueeze(0)
            return 2.0 * proj - v_xy

        r2 = torch.norm(rel_xy, dim=-1).clamp_min(eps)
        yaw2 = torch.atan2(rel_xy[:, 1], rel_xy[:, 0])
        low_yaw_hit = yaw2 < yaw_min
        high_yaw_hit = yaw2 > yaw_max
        if torch.any(low_yaw_hit):
            rel_xy[low_yaw_hit, 0] = r2[low_yaw_hit] * math.cos(yaw_min)
            rel_xy[low_yaw_hit, 1] = r2[low_yaw_hit] * math.sin(yaw_min)
            target_vel_w[low_yaw_hit, :2] = _reflect_about_ray(target_vel_w[low_yaw_hit, :2], yaw_min)
            target_acc_w[low_yaw_hit, :2] = _reflect_about_ray(target_acc_w[low_yaw_hit, :2], yaw_min)
        if torch.any(high_yaw_hit):
            rel_xy[high_yaw_hit, 0] = r2[high_yaw_hit] * math.cos(yaw_max)
            rel_xy[high_yaw_hit, 1] = r2[high_yaw_hit] * math.sin(yaw_max)
            target_vel_w[high_yaw_hit, :2] = _reflect_about_ray(target_vel_w[high_yaw_hit, :2], yaw_max)
            target_acc_w[high_yaw_hit, :2] = _reflect_about_ray(target_acc_w[high_yaw_hit, :2], yaw_max)

        target_pos_w = anchor_w + rel

        # Use the runtime camera for distance and angular FOV constraints when
        # it is available; GT-only mode uses the proxy branch below.
        if self._vision_camera is not None:
            d_min = float(self.cfg.vision_valid_distance_range[0])
            d_max = float(self.cfg.vision_valid_distance_range[1])
            margin = float(self.cfg.vision_distance_bounce_margin)
            cam_pos_all, cam_quat_all = self._get_runtime_camera_pose_w()
            cam_pos_w = cam_pos_all[env_ids_t]
            cam_quat_w = cam_quat_all[env_ids_t]

            rel_cam = target_pos_w - cam_pos_w
            dist_cam = torch.norm(rel_cam, dim=-1, keepdim=True).clamp_min(eps)
            unit_cam = rel_cam / dist_cam
            radial_speed_cam = torch.sum(target_vel_w * unit_cam, dim=-1, keepdim=True)
            radial_acc_cam = torch.sum(target_acc_w * unit_cam, dim=-1, keepdim=True)

            outer_shell_hit = (dist_cam > d_max) & (radial_speed_cam > 0.0)
            inner_shell_hit = (dist_cam < d_min) & (radial_speed_cam < 0.0)
            shell_hit = outer_shell_hit | inner_shell_hit
            if torch.any(shell_hit):
                target_dist = dist_cam.clone()
                target_dist[outer_shell_hit] = max(d_min + margin, d_max - margin)
                target_dist[inner_shell_hit] = min(d_max - margin, d_min + margin)
                hit_mask = shell_hit.squeeze(-1)
                target_pos_w[hit_mask] = (
                    cam_pos_w[hit_mask]
                    + unit_cam[hit_mask] * target_dist[hit_mask]
                )
                target_vel_w[hit_mask] = target_vel_w[hit_mask] - 2.0 * radial_speed_cam[hit_mask] * unit_cam[hit_mask]
                target_acc_w[hit_mask] = target_acc_w[hit_mask] - 2.0 * radial_acc_cam[hit_mask] * unit_cam[hit_mask]

            # Project back inside the configured camera angular sector while
            # preserving range. This is the actual FOV constraint.
            rel_cam = target_pos_w - cam_pos_w
            rel_cam_local = self._rotate_world_to_local(cam_quat_w, rel_cam)
            cam_range = torch.norm(rel_cam_local, dim=-1).clamp_min(eps)
            cam_horizontal = torch.sqrt(
                rel_cam_local[:, 0] ** 2 + rel_cam_local[:, 2] ** 2
            ).clamp_min(eps)
            cam_yaw = torch.atan2(rel_cam_local[:, 0], rel_cam_local[:, 2].clamp_min(eps))
            cam_pitch = torch.atan2(-rel_cam_local[:, 1], cam_horizontal)
            cam_yaw_min, cam_yaw_max = (
                float(self.cfg.target_cam_yaw_range[0]),
                float(self.cfg.target_cam_yaw_range[1]),
            )
            cam_pitch_min, cam_pitch_max = (
                float(self.cfg.target_cam_pitch_range[0]),
                float(self.cfg.target_cam_pitch_range[1]),
            )
            clamped_yaw = torch.clamp(cam_yaw, min=cam_yaw_min, max=cam_yaw_max)
            clamped_pitch = torch.clamp(cam_pitch, min=cam_pitch_min, max=cam_pitch_max)
            sector_hit = (cam_yaw != clamped_yaw) | (cam_pitch != clamped_pitch)
            if torch.any(sector_hit):
                horizontal = cam_range * torch.cos(clamped_pitch)
                target_cam_local = torch.stack(
                    (
                        horizontal * torch.sin(clamped_yaw),
                        -cam_range * torch.sin(clamped_pitch),
                        horizontal * torch.cos(clamped_yaw),
                    ),
                    dim=-1,
                )
                target_pos_w[sector_hit] = cam_pos_w[sector_hit] + self._rotate_local_to_world(
                    cam_quat_w[sector_hit], target_cam_local[sector_hit]
                )

                # Reflect the outward camera-frame component so the target
                # does not remain pinned to the FOV edge on the next step.
                vel_cam = self._rotate_world_to_local(cam_quat_w, target_vel_w)
                acc_cam = self._rotate_world_to_local(cam_quat_w, target_acc_w)
                low_yaw = cam_yaw < cam_yaw_min
                high_yaw = cam_yaw > cam_yaw_max
                low_pitch = cam_pitch < cam_pitch_min
                high_pitch = cam_pitch > cam_pitch_max
                if torch.any(low_yaw):
                    vel_cam[low_yaw, 0] = torch.abs(vel_cam[low_yaw, 0])
                    acc_cam[low_yaw, 0] = torch.abs(acc_cam[low_yaw, 0])
                if torch.any(high_yaw):
                    vel_cam[high_yaw, 0] = -torch.abs(vel_cam[high_yaw, 0])
                    acc_cam[high_yaw, 0] = -torch.abs(acc_cam[high_yaw, 0])
                if torch.any(low_pitch):
                    vel_cam[low_pitch, 1] = -torch.abs(vel_cam[low_pitch, 1])
                    acc_cam[low_pitch, 1] = -torch.abs(acc_cam[low_pitch, 1])
                if torch.any(high_pitch):
                    vel_cam[high_pitch, 1] = torch.abs(vel_cam[high_pitch, 1])
                    acc_cam[high_pitch, 1] = torch.abs(acc_cam[high_pitch, 1])
                target_vel_w = self._rotate_local_to_world(cam_quat_w, vel_cam)
                target_acc_w = self._rotate_local_to_world(cam_quat_w, acc_cam)
        else:
            # GT-only train/eval deliberately do not spawn a camera. Use the
            # robot yaw-only frame as a deterministic camera-FOV proxy so SD
            # still receives the same angular constraint without requiring
            # camera/USD initialization.
            root_quat_w = self.turret.data.root_quat_w[env_ids_t]
            yaw_quat_w = self._get_yaw_only_quat(root_quat_w)
            rel_local = self._rotate_world_to_local(yaw_quat_w, target_pos_w - anchor_w)
            target_range = torch.norm(rel_local, dim=-1).clamp_min(eps)
            horizontal = torch.sqrt(rel_local[:, 0] ** 2 + rel_local[:, 1] ** 2).clamp_min(eps)
            target_yaw = torch.atan2(rel_local[:, 1], rel_local[:, 0])
            target_pitch = torch.atan2(-rel_local[:, 2], horizontal)
            fov_yaw_min, fov_yaw_max = (
                float(self.cfg.target_cam_yaw_range[0]),
                float(self.cfg.target_cam_yaw_range[1]),
            )
            fov_pitch_min, fov_pitch_max = (
                float(self.cfg.target_cam_pitch_range[0]),
                float(self.cfg.target_cam_pitch_range[1]),
            )
            clamped_yaw = torch.clamp(target_yaw, min=fov_yaw_min, max=fov_yaw_max)
            clamped_pitch = torch.clamp(target_pitch, min=fov_pitch_min, max=fov_pitch_max)
            sector_hit = (target_yaw != clamped_yaw) | (target_pitch != clamped_pitch)
            if torch.any(sector_hit):
                horizontal = target_range * torch.cos(clamped_pitch)
                target_local = torch.stack(
                    (
                        horizontal * torch.cos(clamped_yaw),
                        horizontal * torch.sin(clamped_yaw),
                        -target_range * torch.sin(clamped_pitch),
                    ),
                    dim=-1,
                )
                target_pos_w[sector_hit] = anchor_w[sector_hit] + self._rotate_local_to_world(
                    yaw_quat_w[sector_hit], target_local[sector_hit]
                )

                vel_local = self._rotate_world_to_local(yaw_quat_w, target_vel_w)
                acc_local = self._rotate_world_to_local(yaw_quat_w, target_acc_w)
                low_yaw = target_yaw < fov_yaw_min
                high_yaw = target_yaw > fov_yaw_max
                low_pitch = target_pitch < fov_pitch_min
                high_pitch = target_pitch > fov_pitch_max
                if torch.any(low_yaw):
                    vel_local[low_yaw, 1] = torch.abs(vel_local[low_yaw, 1])
                    acc_local[low_yaw, 1] = torch.abs(acc_local[low_yaw, 1])
                if torch.any(high_yaw):
                    vel_local[high_yaw, 1] = -torch.abs(vel_local[high_yaw, 1])
                    acc_local[high_yaw, 1] = -torch.abs(acc_local[high_yaw, 1])
                if torch.any(low_pitch):
                    vel_local[low_pitch, 2] = -torch.abs(vel_local[low_pitch, 2])
                    acc_local[low_pitch, 2] = -torch.abs(acc_local[low_pitch, 2])
                if torch.any(high_pitch):
                    vel_local[high_pitch, 2] = torch.abs(vel_local[high_pitch, 2])
                    acc_local[high_pitch, 2] = torch.abs(acc_local[high_pitch, 2])
                target_vel_w = self._rotate_local_to_world(yaw_quat_w, vel_local)
                target_acc_w = self._rotate_local_to_world(yaw_quat_w, acc_local)

        self._target_pos_w[env_ids_t] = target_pos_w
        self._target_vel_w[env_ids_t] = target_vel_w
        self._target_acc_w[env_ids_t] = target_acc_w


    def _resolve_turret_prim_path_expr(self) -> str:
        """Resolve the turret articulation prim-path expression.

        If the configured expression doesn't match any prims, tries to discover an articulation root under
        the spawned asset in env_0. This helps when the USD prim hierarchy differs from the guess.
        """

        # Fast path: config matches an articulation root.
        matches = sim_utils.find_matching_prim_paths(self.cfg.turret_prim_path_expr)
        if len(matches) > 0:
            # Validate that at least one match is an articulation root.
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import UsdPhysics

            stage = get_current_stage()
            art_root_api = getattr(UsdPhysics, "ArticulationRootAPI", None)
            for path in matches:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid() and art_root_api is not None and prim.HasAPI(art_root_api):
                    return self.cfg.turret_prim_path_expr
            # If we got matches but none are articulation roots, continue to auto-resolve.

        if not self.cfg.auto_resolve_turret_prim:
            raise RuntimeError(
                f"Could not find prim with path expression: '{self.cfg.turret_prim_path_expr}'. "
                "Set TurretAimEnvCfg.auto_resolve_turret_prim=True to auto-discover candidates."
            )

        # Import USD/Stage helpers only after Kit is running.
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdPhysics

        stage = get_current_stage()
        art_root_api = getattr(UsdPhysics, "ArticulationRootAPI", None)
        asset_root_path = self.cfg.asset_prim_path

        # Validate asset root.
        asset_root_prim = stage.GetPrimAtPath(asset_root_path)
        if not asset_root_prim.IsValid():
            # fallback: try to resolve if asset_root_path itself is an expression
            asset_matches = sim_utils.find_matching_prim_paths(asset_root_path)
            if len(asset_matches) > 0:
                asset_root_path = asset_matches[0]
                asset_root_prim = stage.GetPrimAtPath(asset_root_path)
        if not asset_root_prim.IsValid():
            raise RuntimeError(
                "Asset root prim is not valid. "
                f"Tried '{self.cfg.asset_prim_path}'. "
                "Update TurretAimEnvCfg.asset_prim_path to match your USD spawn location."
            )

        # Find articulation roots under asset.
        try:
            art_root_prims = sim_utils.get_all_matching_child_prims(
                asset_root_path,
                predicate=(lambda prim: art_root_api is not None and prim.HasAPI(art_root_api)),
                depth=self.cfg.turret_search_depth,
                stage=stage,
            )
        except Exception:
            art_root_prims = []

        art_root_paths = [p.GetPath().pathString for p in art_root_prims]

        # If none found, collect substring hits only for diagnostics.
        diag_hint_paths: list[str] = []
        if len(art_root_paths) == 0:
            hint_tokens = tuple(h.lower() for h in self.cfg.turret_path_hints)
            for prim in stage.Traverse():
                p = prim.GetPath().pathString
                if not p.startswith(asset_root_path):
                    continue
                pl = p.lower()
                if any(h in pl for h in hint_tokens):
                    diag_hint_paths.append(p)

        # Rank candidates: prefer ones containing hint tokens and deeper paths (more specific).
        hint_tokens = tuple(h.lower() for h in self.cfg.turret_path_hints)

        def score(path: str) -> tuple[int, int]:
            pl = path.lower()
            hint_score = sum(1 for h in hint_tokens if h in pl)
            depth_score = path.count("/")
            return (hint_score, depth_score)

        art_root_paths = sorted(set(art_root_paths), key=score, reverse=True)

        if len(art_root_paths) == 0:
            msg = (
                "Could not find any prim with UsdPhysics.ArticulationRootAPI under the spawned asset. "
                "This usually means your USD is not authored as an articulation (missing ArticulationRootAPI), "
                "or the asset references inside the USD failed to load (broken file paths), so links/joints are missing. "
                f"asset_prim_path='{self.cfg.asset_prim_path}'. "
                "Fix the USD (apply ArticulationRootAPI on the articulation root prim, and ensure joints have body0/body1), "
                "then set turret_prim_path_expr to that articulation root path."
            )
            if len(diag_hint_paths) > 0:
                msg += "\nHint-matching prim paths (not necessarily articulation roots):\n" + "\n".join(
                    f"  - {p}" for p in diag_hint_paths[:20]
                )
            msg += "\nTip: run ./isaaclab.sh -p source/standalone/test1/inspect_usd.py --usd <your.usd> to list candidates."
            raise RuntimeError(msg)

        chosen_path = art_root_paths[0]

        # Convert env_0 path into env regex expression so it works for cloned envs.
        # Typical prefix after clone: /World/envs/env_0/...
        if "/World/envs/env_0/" in chosen_path:
            turret_expr = chosen_path.replace("/World/envs/env_0/", "/World/envs/env_.*/", 1)
        elif "/World/env_0/" in chosen_path:
            turret_expr = chosen_path.replace("/World/env_0/", "/World/env_.*/", 1)
        else:
            turret_expr = chosen_path

        # Print actionable guidance.
        print("[TurretAimEnv] turret_prim_path_expr did not match anything.")
        print(f"[TurretAimEnv] Config turret_prim_path_expr: {self.cfg.turret_prim_path_expr}")
        print(f"[TurretAimEnv] Auto-discovered candidate (env_0): {chosen_path}")
        print(f"[TurretAimEnv] Suggested turret_prim_path_expr: {turret_expr}")
        print("[TurretAimEnv] Top candidates:")
        for p in art_root_paths[:10]:
            print(f"  - {p}")

        # Extra sanity: ensure chosen prim is an articulation root.
        chosen_prim = stage.GetPrimAtPath(chosen_path)
        if not chosen_prim.IsValid() or art_root_api is None or not chosen_prim.HasAPI(art_root_api):
            raise RuntimeError(
                "Auto-resolve selected a prim that is not an articulation root. "
                f"chosen_path='{chosen_path}'. "
                "Apply ArticulationRootAPI to the correct root prim in your USD, or manually set turret_prim_path_expr "
                "to an existing articulation root prim path."
            )

        # Sanity: ensure suggested expression matches at least one prim.
        suggested_matches = sim_utils.find_matching_prim_paths(turret_expr)
        if len(suggested_matches) == 0:
            print("[TurretAimEnv] WARNING: suggested expression still matches 0 prims. You likely need to copy the")
            print("[TurretAimEnv] exact prim path (no regex) or adjust the env prefix replacement.")

        return turret_expr

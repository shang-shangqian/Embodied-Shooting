from __future__ import annotations
import math
import os
from collections.abc import Sequence
from typing import cast
import traceback
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
from isaaclab.utils.math import quat_apply, sample_uniform
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
    decimation = 4 # 与 Lite3 locomotion 训练保持一致（0.005 * 4 = 0.02s）
    action_space = 3  # 动作维度
    observation_space = 25 #观测维度
    state_space = 0
    training_stage: int = 1 #训练阶段

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
    # 这些相机不挂在 articulation 下面，而是放在 World 下，
    # 每帧由代码把 runtime 世界位姿写进去，避免“相机跟着机器人不刷新”的问题。
    runtime_camera_proxy_prim_path_expr: str = "/World/RuntimeRenderCameras/env_.*/Camera"

    camera_width: int = 1024
    camera_height: int = 576
    camera_projection_type: str = "perspective"
    camera_focal_length: float = 10.0
    camera_horizontal_aperture: float = 5.76
    camera_vertical_aperture: float = 3.24
    camera_clipping_range: tuple[float, float] = (0.1, 1000000.0)
    camera_f_stop: float = 0.0
    vision_history_length: int = 4
    vision_update_period: float = 0.0
    vision_normalize_rgb: bool = True
    # Far-target data: a 0.4 m sphere at 150 m is only about 1 px radius
    # with the current camera, so the old 2 px cutoff rejects valid samples.
    min_visible_target_radius_pix: float = 1.0

    # 注意：TiledCamera 现在绑定的是 proxy camera，不再直接绑 USD 里狗身上的原始相机
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
    enable_dynamic_target_raycast_debug: bool = True
    dynamic_target_raycast_debug_every: int = 20

    # 直接挂到每个 env 的 TORSO
    dynamic_target_raycast_parent_prim_path_expr: str = "/World/envs/env_.*/Lite3/Lite3/TORSO"

    # 相对 TORSO 的安装位姿：与原 runtime lidar 保持一致
    dynamic_target_raycast_translation: tuple[float, float, float] = (0.23, 0.0, 0.073)
    dynamic_target_raycast_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # 射线束参数：尽量对齐你原来的 runtime lidar
    dynamic_target_raycast_min_range: float = 65.0
    dynamic_target_raycast_max_range: float = 165.0
    dynamic_target_raycast_horizontal_fov_deg: float = 40.0
    dynamic_target_raycast_vertical_fov_deg: float = 24.0
    dynamic_target_raycast_horizontal_resolution_deg: float = 0.15
    dynamic_target_raycast_vertical_resolution_deg: float = 0.15
    dynamic_target_raycast_primary_cone_deg: float = 3.0
    dynamic_target_raycast_fallback_cone_deg: float = 8.0
    dynamic_target_raycast_center_offset_radius_scale: float = 2.0 / 3.0

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
        update_mesh_ids=True,
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
    leg_joint_names_expr: tuple[str, ...] = (".*_HipX_joint", ".*_HipY_joint", ".*_Knee_joint")

    # 可选：用某个连杆（枪口/炮管）来计算朝向与位置，会比仅用 yaw/pitch 解析式更准确。
    # 例：muzzle_body_name_exp
    # r = "barrel_link" 或 "muzzle"（支持正则）
    muzzle_body_name_expr = "pitch_link"
    muzzle_forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # 枪口朝向在该连杆坐标系下的轴
    muzzle_offset_b: tuple[float, float, float] = (0.09, 0.0, 0.0)  # gun_tip 相对 pitch_link 的局部偏移

    # --- control ---
    max_delta_yaw: float = 0.06      # 每个 env step 最大 yaw 增量，单位 rad
    max_delta_pitch: float = 0.04    # 每个 env step 最大 pitch 增量，单位 rad
    action_smoothing: float = 0.0  # 动作平滑系数，0.0 表示不平滑，1.0 表示完全平滑（即保持初始动作不变）。建议在训练初期使用一些平滑（如 0.5），等策略稳定后再逐渐降低到 0.0。
    # turret joint limits
    yaw_min: float = -3.14159
    yaw_max: float = 3.14159
    pitch_min: float = -0.524   #-30度
    pitch_max: float = 1.047    #60度

    # --- locomotion policy (for moving-base aiming) ---
    enable_locomotion_policy: bool = False      #是否启用预训练的移动策略来控制腿部关节（否则保持默认站立姿态）
    locomotion_policy_path: str | None = None
    locomotion_command_resample_s: float = 10.0     #每隔多少秒重新采样一次移动指令（线速度 x/y 和角速度 z）
    locomotion_lin_vel_x_range: tuple[float, float] = (0.2, 0.6)
    locomotion_lin_vel_y_range: tuple[float, float] = (-0.2, 0.2)
    locomotion_ang_vel_z_range: tuple[float, float] = (-0.4, 0.4)
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.35)

    # --- target sampling (vision-friendly, in camera sector) ---
    # 距离限制在 80~150 m
    target_radius_range: tuple[float, float] = (80.0, 150.0)
    # --- rendered target sphere ---
    render_target_sphere: bool = True
    target_sphere_radius: float = 0.4
    target_sphere_color: tuple[float, float, float] = (1.0, 0.1, 0.1)

    # 视觉模块的有效距离范围（相对相机）
    vision_valid_distance_range: tuple[float, float] = (80.0, 150.0)
    vision_distance_bounce_margin: float = 0.2

    # 每个 env 下真实球体的 prim 路径
    target_sphere_prim_path_expr: str = "/World/envs/env_.*/TargetSphere"

    # 兼容旧逻辑保留，但视觉采样优先使用 camera-sector 逻辑
    target_height_range: tuple[float, float] = (0.5, 3.0)
    target_yaw_range: tuple[float, float] = (-math.pi / 2, math.pi / 2)

    # 是否按相机扇区采样目标
    sample_target_in_camera_sector: bool = True

    # 相机扇区采样：水平/垂直角范围（单位 rad）
    # 建议先取相对保守范围，保证目标更容易在画面中央附近
    # Camera fov from focal=10.0, aperture=(5.76, 3.24):
    # horizontal half fov ~= 0.281 rad, vertical half fov ~= 0.161 rad.
    # Keep a small margin so the rendered sphere remains inside the image.
    target_cam_yaw_range: tuple[float, float] = (-0.26, 0.26)
    target_cam_pitch_range: tuple[float, float] = (-0.14, 0.14)

    # 如果你想让目标尽量更容易看见，可以给一个额外高度偏移范围
    # 最终高度会由相机位置 + 距离 * sin(pitch) 决定，这里通常不再单独采 z
    # dynamic target (optional)
    enable_dynamic_target: bool = False
    target_speed_range: tuple[float, float] = (0.6, 5.0)
    target_vertical_speed_range: tuple[float, float] = (-0.5, 0.5)
    target_max_speed_xy: float = 5.0
    # --- 新增：目标加速度配置 ---
    target_acc_range: tuple[float, float] = (0.0, 0.0)
    target_vertical_acc_range: tuple[float, float] = (0.0, 0.0)

    # --- termination ---
    success_angle: float = 0.05  # rad
    failure_height_threshold: float = 0.16
    failure_tilt_cos_threshold: float = 0.4     #炮台根部竖直向上的向量与世界竖直向上的向量的夹角余弦值低于该阈值即判定为失败（对应约 66° 以上的倾倒）

    # --- ballistic (gravity + optional air drag) ---
    muzzle_speed: float = 882.0      #枪口初速，单位 m/s
    gravity_mps2: float = 9.81
    hit_radius: float = 0.35    #命中判定半径
    ballistic_max_time: float = 1.5     
    ballistic_num_samples: int = 181

    # 空气阻力（球体二次阻力）
    enable_air_drag: bool = True
    air_density: float = 1.204          # kg/m^3
    drag_coefficient: float = 0.47      # 光滑球体近似 Cd
    projectile_diameter_m: float = 0.017
    projectile_mass_kg: float = 0.0032

    # --- reward weights ---
    w_align: float = 0.04    #瞄准奖励权重
    w_rate: float = -0.00002     #位置增量惩罚权重
    w_action: float = -0.0002    #动作幅度惩罚权重
    w_base_ang_vel: float = -0.003       #底盘角速度惩罚权重（鼓励稳定）
    w_action_change: float = -0.0005     #动作变化惩罚权重（鼓励平滑）
    r_hit: float = 35.0     #成功击中目标奖励
    r_bad_fire: float = -1.5    #不良射击惩罚（开火但未击中）
    r_fire_bonus_near: float = 0.0      #开火奖励（目标在一定距离内才有奖励，鼓励近距离射击）
    r_fire_cost: float = -0.12        #每次开火的固定惩罚（鼓励节约弹药）
    r_time_penalty: float = -0.03      #每步的时间惩罚（鼓励快速完成任务）
    fire_cooldown_steps: int = 5    #开火后需要冷却的步数（即最小射速），单位为 env step 数

    aim_warmup_s: float = 1.0       #机器狗随机移动时间
    w_ballistic: float = 0.18       #弹道奖励权重（基于弹道预测的命中距离和命中判定）

    # 第3维动作不再直接 >0.5 开火，而是作为连续 fire intent
    fire_charge_max: float = 1.25       # charge 的最大值，超过该值不再增长
    fire_charge_threshold: float = 0.55
    fire_charge_gain: float = 2.2      # charge 增长速度（每秒）
    fire_charge_leak: float = 0.18      # charge 泄放速度（每秒）

    # 命中判定距离
    max_hit_distance: float = 170.0

    # --- actuator settings (覆盖/补充 USD joint drive) ---
    yaw_effort_limit: float = 50.0
    pitch_effort_limit: float = 50.0
    stiffness: float = 20.0
    damping: float = 2.0


class TurretAimEnv(DirectRLEnv):
    cfg: TurretAimEnvCfg

    def __init__(self, cfg: TurretAimEnvCfg, render_mode: str | None = None, **kwargs):
        # ----------------------------
        # dynamic target raycast debug (MultiMeshRayCaster)
        # 这些要放在 super().__init__ 前，避免 _setup_scene() 创建后又被覆盖
        # ----------------------------
        self._dynamic_target_raycast_sensor: MultiMeshRayCaster | None = None
        self._dynamic_target_raycast_mount_paths: list[str] = []
        self._dynamic_target_raycast_mount_translate_ops = []
        self._dynamic_target_raycast_mount_orient_ops = []

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
        self._fire_charge = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._episode_fire_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._aim_warmup_left_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._target_quat_w[:, 0] = 1.0   # identity quaternion: wxyz = [1, 0, 0, 0]
        self._target_acc_w = torch.zeros((self.num_envs, 3), device=self.device) # 加速度
        self._target_anchor_w = torch.zeros((self.num_envs, 3), device=self.device)
        #视觉与雷达的真值缓存
        self._vision_distance_gt = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._vision_pos_base_gt = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
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

        self._target_sphere_prim_paths: list[str] = [
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

        # 二次空气阻力加速度系数: a_drag = -k * |v| * v
        # k = 0.5 * rho * Cd * A / m, A = pi * (d/2)^2
        projectile_radius = 0.5 * self.cfg.projectile_diameter_m
        cross_section_area = math.pi * projectile_radius * projectile_radius
        drag_accel_coeff = 0.5 * self.cfg.air_density * self.cfg.drag_coefficient * cross_section_area / max(
            self.cfg.projectile_mass_kg,
            1e-6,
        )
        self._drag_accel_coeff = torch.tensor(drag_accel_coeff, device=self.device, dtype=torch.float32)


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

            # Force the collection camera intrinsics to the requested pinhole setup.
            proxy_cam.GetProjectionAttr().Set(str(self.cfg.camera_projection_type))
            proxy_cam.GetFocalLengthAttr().Set(float(self.cfg.camera_focal_length))
            proxy_cam.GetHorizontalApertureAttr().Set(float(self.cfg.camera_horizontal_aperture))
            proxy_cam.GetVerticalApertureAttr().Set(float(self.cfg.camera_vertical_aperture))
            proxy_cam.GetFStopAttr().Set(float(self.cfg.camera_f_stop))
            proxy_cam.GetClippingRangeAttr().Set(
                Gf.Vec2f(
                    float(self.cfg.camera_clipping_range[0]),
                    float(self.cfg.camera_clipping_range[1]),
                )
            )

            # clippingRange 需要单独处理
            try:
                clip_val = src_cam.GetClippingRangeAttr().Get()
                if clip_val is not None:
                    proxy_cam.GetClippingRangeAttr().Set(
                        Gf.Vec2f(
                            float(self.cfg.camera_clipping_range[0]),
                            float(self.cfg.camera_clipping_range[1]),
                        )
                    )
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


    def _set_camera_pose_on_prim(
        self,
        prim_path: str,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
    ) -> None:
        """Write world pose to a proxy camera prim."""
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom, Gf, Usd

        stage = get_current_stage()
        if stage is None:
            return

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()

        translate_op = None
        orient_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
                orient_op = op

        if translate_op is None:
            translate_op = xf.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble,
                "runtime",
                False,
            )
        if orient_op is None:
            orient_op = xf.AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble,
                "runtime",
                False,
            )

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
        # 注意：这里不再给它刚体/质量/碰撞属性，只保留可见 mesh
        if self.cfg.render_target_sphere:
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

        # Collection subclasses use this hook to add all scene prims that
        # must be cloned into every environment (for example TargetSphere_01).
        spawn_extra_prims = getattr(
            self, "_spawn_extra_scene_prims_before_clone", None
        )
        if callable(spawn_extra_prims):
            spawn_extra_prims()

        # vision camera：一定要在 clone 之后再初始化
        # 否则 TiledCamera 可能只绑定到 clone 前的 source env / 旧 prim，
        # 后续各 env 的相机不会正确进入 render graph，导致画面冻结或始终读到旧帧。
        self._vision_camera: TiledCamera | None = None

        # 先 clone 所有环境
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # clone 完成后，再创建 runtime proxy cameras，并让 TiledCamera 绑定到 proxy
        if self.cfg.enable_vision_supervision:
            camera_matches = sim_utils.find_matching_prim_paths(self.cfg.camera_prim_path_expr)
            if len(camera_matches) == 0:
                raise RuntimeError(
                    f"[vision] No camera prim matched after clone: {self.cfg.camera_prim_path_expr}"
                )

            print(f"[vision] matched {len(camera_matches)} source camera prim(s) after clone")
            if self.cfg.debug_diagnostics:
                for p in camera_matches[: min(8, len(camera_matches))]:
                    print(f"[vision] source camera prim: {p}")

            self._ensure_runtime_camera_proxies()
            self._vision_camera = TiledCamera(self.cfg.tiled_camera)

        # dynamic target raycast: 直接挂到 TORSO，不再创建额外 mount
        if self.cfg.enable_dynamic_target_raycast_debug:
            self._create_dynamic_target_multimesh_raycast_sensor()

        # bind cloned target spheres
        # 这里不再把 TargetSphere 绑定成 RigidObject，而是只记录每个 env 的 prim path，
        # 后续直接通过 USD/Xform 改它的位置。
        if self.cfg.render_target_sphere:
            self.target_sphere = True
            self._target_sphere_prim_paths = [
                f"/World/envs/env_{env_id}/TargetSphere"
                for env_id in range(self.num_envs)
            ]
        else:
            self.target_sphere = None
            self._target_sphere_prim_paths = []

        # add sensors to scene
        if self._vision_camera is not None:
            self.scene.sensors["vision_camera"] = self._vision_camera

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
                    "FL_HipX_joint": 0.0,
                    "FL_HipY_joint": -0.8,
                    "FL_Knee_joint": 1.6,
                    "FR_HipX_joint": 0.0,
                    "FR_HipY_joint": -0.8,
                    "FR_Knee_joint": 1.6,
                    "HL_HipX_joint": 0.0,
                    "HL_HipY_joint": -0.8,
                    "HL_Knee_joint": 1.6,
                    "HR_HipX_joint": 0.0,
                    "HR_HipY_joint": -0.8,
                    "HR_Knee_joint": 1.6,
                },
            ),
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
                    effort_limit_sim=max(self.cfg.yaw_effort_limit, self.cfg.pitch_effort_limit),
                    stiffness=self.cfg.stiffness,
                    damping=self.cfg.damping,
                ),
            },
        )
        self.turret = Articulation(turret_cfg)
        self.scene.articulations["turret"] = self.turret

        # 灯光和目标可视化 marker
        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._target_markers = None
        if self.cfg.debug_vis and (self.sim.has_gui() or self.sim.has_rtx_sensors()):
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/target",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.1,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.2, 0.2)
                        ),
                    )
                },
            )
            self._target_markers = VisualizationMarkers(marker_cfg)


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

        # 保存上一时刻动作
        self._prev_actions.copy_(self._actions)

        # 动作平滑
        if self.cfg.action_smoothing > 0.0:
            filtered_actions = self.cfg.action_smoothing * self._actions + (1.0 - self.cfg.action_smoothing) * actions
            self._actions.copy_(filtered_actions)
        else:
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

        # Clamp joint limits
        self._turret_pos_targets[:, 0] = torch.clamp(self._turret_pos_targets[:, 0], self.cfg.yaw_min, self.cfg.yaw_max)
        self._turret_pos_targets[:, 1] = torch.clamp(self._turret_pos_targets[:, 1], self.cfg.pitch_min, self.cfg.pitch_max)

        # ----------------------------
        # 3. fire intent -> charge -> fire
        # ----------------------------
        # 第3维动作 a2 ∈ [-1, 1]，映射成 fire_intent ∈ [0, 1]
        fire_intent = ((self._actions[:, 2] + 1.0) * 0.5).clamp(0.0, 1.0)

        # 给 fire intent 一个轻微迟滞，降低移动底盘下的抖动影响
        # 上一帧已有一定 charge 时，允许更容易继续蓄力
        intent_mask = fire_intent > 0.35
        sustain_mask = (self._fire_charge > 0.15) & (fire_intent > 0.20)
        effective_fire_intent = torch.where(intent_mask | sustain_mask, fire_intent, torch.zeros_like(fire_intent))

        charge_up = self.cfg.fire_charge_gain * self.step_dt * effective_fire_intent
        charge_down = self.cfg.fire_charge_leak * self.step_dt * (1.0 - effective_fire_intent)

        self._fire_charge = self._fire_charge + charge_up - charge_down
        self._fire_charge = self._fire_charge.clamp(0.0, self.cfg.fire_charge_max)

        can_fire = self._fire_cooldown <= 0
        fire_now = aiming_enabled & can_fire & (self._fire_charge >= self.cfg.fire_charge_threshold)

        self._fire_command = fire_now.float()
        self._eval_total_fires += torch.count_nonzero(fire_now)
        self._episode_fire_count += fire_now.long()

        # 统计：开火时弹道-目标最小距离
        if torch.any(fire_now):
            min_ballistic_dist, _, _ = self._get_cached_ballistic_metrics()
            fire_min_dists = min_ballistic_dist[fire_now]
            self._eval_fire_min_dist_count += fire_min_dists.numel()
            self._eval_fire_min_dist_sum += fire_min_dists.sum()
            self._eval_fire_min_dist_min = torch.minimum(self._eval_fire_min_dist_min, torch.min(fire_min_dists))
            self._eval_fire_min_dist_max = torch.maximum(self._eval_fire_min_dist_max, torch.max(fire_min_dists))

            remaining = self._eval_samples_limit - len(self._eval_fire_min_dist_samples)
            if remaining > 0:
                sample_vals = fire_min_dists.detach().cpu().tolist()[:remaining]
                self._eval_fire_min_dist_samples.extend(float(v) for v in sample_vals)

        # 开火后不要清零，保留一部分 charge，避免每次从 0 重来
        self._fire_charge[fire_now] = 0.2

        # 更新 cooldown buffer
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
            locomotion_actions = locomotion_actions.clamp(-1.0, 1.0)
            self._locomotion_last_actions.copy_(locomotion_actions)
            self._locomotion_joint_targets.copy_(self._default_leg_joint_pos + self._leg_action_scale * locomotion_actions)

        if self.cfg.enable_dynamic_target:
            self._target_vel_w += self._target_acc_w * self.step_dt

            speed_xy = torch.norm(self._target_vel_w[:, :2], dim=-1, keepdim=True)
            max_speed = float(self.cfg.target_max_speed_xy)
            scale = torch.clamp(max_speed / speed_xy.clamp_min(1e-6), max=1.0)
            self._target_vel_w[:, :2] *= scale

            self._target_pos_w += self._target_vel_w * self.step_dt

            self._handle_target_out_of_bounds()
            self._enforce_target_above_ground()

        # 不管目标是否动态，都把渲染球同步到当前位置
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
        """Explicitly refresh vision buffers once after env.step().

        重要：
        - 这里只读取“当前已经渲染好的相机输出”和“当前时刻对应的标签”
        - 不再在这里触发 target resample，避免:
            新 target 状态 + 旧 rgb 图像
        的时序错位
        """
        # 只消耗 reset 后的额外等待步数
        self._vision_collect_warmup_left = torch.clamp(self._vision_collect_warmup_left - 1, min=0)

        # 不要在这里调用 _maybe_finalize_pending_target_resample()
        # 否则可能出现：
        #   - target_pos / label 已更新到新目标
        #   - rgb 还是上一帧已渲染的旧图
        # 导致图像与标签错位

        # 直接刷新视觉缓存与标签
        self._refresh_vision_supervision()


    def _apply_action(self) -> None:
        dof_ids = [int(self._yaw_dof_ids[0]), int(self._pitch_dof_ids[0])]
        self.turret.set_joint_position_target(self._turret_pos_targets, joint_ids=dof_ids)
        self.turret.set_joint_position_target(self._locomotion_joint_targets, joint_ids=self._leg_dof_ids)


    def _load_locomotion_policy(self):
        # 只要显式启用了 locomotion policy，就允许加载；
        # 不再因为 training_stage == 1 而强行跳过。
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

    def _get_yaw_rate_world(
        self,
        quat_wxyz: torch.Tensor,
        angular_velocity_world: torch.Tensor,
    ) -> torch.Tensor:
        """Return the derivative of the extracted yaw angle in world axes.

        The yaw-only frame is defined from the horizontal projection of the
        body's forward axis.  Therefore its angular rate is the derivative of
        ``atan2(forward_y, forward_x)``; it is not generally the world-frame
        z component of angular velocity when roll or pitch is non-zero.
        """
        basis_x = torch.zeros_like(angular_velocity_world)
        basis_x[..., 0] = 1.0
        forward_world = quat_apply(quat_wxyz, basis_x)
        forward_dot_world = torch.cross(
            angular_velocity_world, forward_world, dim=-1
        )
        horizontal_squared = (
            forward_world[..., 0].square() + forward_world[..., 1].square()
        ).clamp_min(1.0e-8)
        return (
            forward_world[..., 0] * forward_dot_world[..., 1]
            - forward_world[..., 1] * forward_dot_world[..., 0]
        ) / horizontal_squared

    def _rotate_world_to_yaw_only_local(self, root_quat_w: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
        yaw_quat_w = self._get_yaw_only_quat(root_quat_w)
        return self._rotate_world_to_local(yaw_quat_w, vec_w)

    def _rotate_yaw_only_local_to_world(self, root_quat_w: torch.Tensor, vec_local: torch.Tensor) -> torch.Tensor:
        yaw_quat_w = self._get_yaw_only_quat(root_quat_w)
        return self._rotate_local_to_world(yaw_quat_w, vec_local)

    def _ensure_camera_extrinsics_from_torso(self):
        """Lazy-init camera extrinsics relative to TORSO body.

        关键修正：
        不再用 self._vision_camera.data.pos_w / quat_w_ros 反推相机外参，
        因为从日志看这路 data pose 可能是冻结的/错误的。

        这里改为：
        1) 直接从 USD 读取 camera prim 与 TORSO prim 的相对变换（固定外参）
        2) 保存成 camera 相对 TORSO 的局部平移 / 局部旋转
        3) 后续 runtime 世界位姿由当前 TORSO body pose 重建
        """
        if hasattr(self, "_camera_local_pos_in_torso") and hasattr(self, "_camera_local_quat_in_torso"):
            return

        if self._vision_camera is None:
            raise RuntimeError("Vision camera is not initialized.")

        # ------------------------------------------------------------
        # 1) 解析 TORSO body id（runtime 重建时要用）
        # ------------------------------------------------------------
        if not hasattr(self, "_torso_body_id") or self._torso_body_id is None:
            torso_body_ids, torso_body_names = self.turret.find_bodies("TORSO")
            if len(torso_body_ids) == 0:
                raise RuntimeError(
                    f"Could not resolve TORSO body. Available bodies: {self.turret.body_names}"
                )
            self._torso_body_id = int(torso_body_ids[0])

        # ------------------------------------------------------------
        # 2) 从 USD 读取 camera 相对 TORSO 的固定外参
        # ------------------------------------------------------------
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom
        import torch

        stage = get_current_stage()
        if stage is None:
            raise RuntimeError("USD stage is None, cannot resolve camera extrinsics from USD.")

        camera_matches = sim_utils.find_matching_prim_paths(self.cfg.camera_prim_path_expr)
        torso_matches = sim_utils.find_matching_prim_paths(self.cfg.turret_prim_path_expr)

        if len(camera_matches) == 0:
            raise RuntimeError(
                f"No camera prim matched: {self.cfg.camera_prim_path_expr}"
            )
        if len(torso_matches) == 0:
            raise RuntimeError(
                f"No TORSO prim matched: {self.cfg.turret_prim_path_expr}"
            )

        # 优先取 env_0，若没有则取第一个
        camera_prim_path = None
        for p in camera_matches:
            if "/env_0/" in p:
                camera_prim_path = p
                break
        if camera_prim_path is None:
            camera_prim_path = camera_matches[0]

        torso_prim_path = None
        for p in torso_matches:
            if "/env_0/" in p:
                torso_prim_path = p
                break
        if torso_prim_path is None:
            torso_prim_path = torso_matches[0]

        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        torso_prim = stage.GetPrimAtPath(torso_prim_path)

        if not camera_prim.IsValid():
            raise RuntimeError(f"Invalid camera prim: {camera_prim_path}")
        if not torso_prim.IsValid():
            raise RuntimeError(f"Invalid TORSO prim: {torso_prim_path}")

        camera_xf = UsdGeom.Xformable(camera_prim)
        torso_xf = UsdGeom.Xformable(torso_prim)

        cam_world_mat = camera_xf.ComputeLocalToWorldTransform(0.0)
        torso_world_mat = torso_xf.ComputeLocalToWorldTransform(0.0)

        # 注意：
        # 在这里直接用矩阵乘法提相对平移时，Gf.Matrix4d 的乘法约定很容易把“世界平移”带进来，
        # 你现在日志里 camera_pos_w_rt 跳到 [200, 100, ...] 基本就是这个问题。
        #
        # 这里改成：
        # 1) 旋转仍然从相对矩阵提取
        # 2) 平移不用 rel_mat.ExtractTranslation()，而是直接用
        #    cam_local_pos = R_torso^T * (cam_world - torso_world)
        #    这样不会把世界原点项错误混进去

        rel_mat = cam_world_mat * torso_world_mat.GetInverse()

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

        # 归一化四元数，避免数值误差
        cam_local_quat_single = cam_local_quat_single / torch.norm(
            cam_local_quat_single, dim=-1, keepdim=True
        ).clamp_min(1e-12)

        # 注意：
        # 这里不要再用 rel_mat.ExtractTranslation() 去覆盖 cam_local_pos_single。
        # 平移必须保留前面用
        #   cam_local_pos = R_torso^T * (cam_world - torso_world)
        # 算出来的结果，否则会把世界平移项重新带回来。

        # ------------------------------------------------------------
        # 3) 扩展到所有 env
        # ------------------------------------------------------------
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
        """Reconstruct runtime camera world pose in USD camera frame.

        Returns:
            cam_pos_w:      [N, 3]
            cam_quat_w_usd: [N, 4] (wxyz), 供渲染 / proxy camera 使用
        """
        self._ensure_camera_extrinsics_from_torso()

        if not hasattr(self, "_torso_body_id") or self._torso_body_id is None:
            torso_body_ids, torso_body_names = self.turret.find_bodies("TORSO")
            if len(torso_body_ids) == 0:
                raise RuntimeError(
                    f"Could not resolve TORSO body. Available bodies: {self.turret.body_names}"
                )
            self._torso_body_id = int(torso_body_ids[0])

        if not hasattr(self.turret.data, "body_pos_w") or not hasattr(self.turret.data, "body_quat_w"):
            root_pos_w = self.turret.data.root_pos_w
            root_quat_w = self.turret.data.root_quat_w

            cam_local_pos = self._camera_local_pos_in_torso
            cam_local_quat = self._camera_local_quat_in_torso

            if cam_local_pos.shape[0] != root_pos_w.shape[0]:
                cam_local_pos = cam_local_pos[:1].repeat(root_pos_w.shape[0], 1)
            if cam_local_quat.shape[0] != root_quat_w.shape[0]:
                cam_local_quat = cam_local_quat[:1].repeat(root_quat_w.shape[0], 1)

            cam_pos_w = root_pos_w + self._rotate_local_to_world(root_quat_w, cam_local_pos)
            cam_quat_w_usd = self._quat_mul(root_quat_w, cam_local_quat)
            cam_quat_w_usd = cam_quat_w_usd / torch.norm(
                cam_quat_w_usd, dim=-1, keepdim=True
            ).clamp_min(1e-12)
            return cam_pos_w, cam_quat_w_usd

        torso_pos_w = self.turret.data.body_pos_w[:, self._torso_body_id]
        torso_quat_w = self.turret.data.body_quat_w[:, self._torso_body_id]

        cam_local_pos = self._camera_local_pos_in_torso
        cam_local_quat = self._camera_local_quat_in_torso

        if cam_local_pos.shape[0] != torso_pos_w.shape[0]:
            cam_local_pos = cam_local_pos[:1].repeat(torso_pos_w.shape[0], 1)
        if cam_local_quat.shape[0] != torso_quat_w.shape[0]:
            cam_local_quat = cam_local_quat[:1].repeat(torso_quat_w.shape[0], 1)

        cam_pos_w = torso_pos_w + self._rotate_local_to_world(
            torso_quat_w, cam_local_pos
        )

        cam_quat_w_usd = self._quat_mul(torso_quat_w, cam_local_quat)
        cam_quat_w_usd = cam_quat_w_usd / torch.norm(
            cam_quat_w_usd, dim=-1, keepdim=True
        ).clamp_min(1e-12)

        return cam_pos_w, cam_quat_w_usd


    def _get_runtime_camera_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct runtime camera world pose in ROS optical frame.

        Returns:
            cam_pos_w:  [N, 3]
            cam_quat_w: [N, 4] (wxyz), 供几何/标签/雷达计算使用
        """
        cam_pos_w, cam_quat_w_usd = self._get_runtime_camera_pose_w_usd()

        # USD camera frame -> ROS optical frame
        q_usd_to_ros = torch.zeros_like(cam_quat_w_usd)
        q_usd_to_ros[:, 1] = 1.0  # [0, 1, 0, 0]

        cam_quat_w = self._quat_mul(cam_quat_w_usd, q_usd_to_ros)
        cam_quat_w = cam_quat_w / torch.norm(
            cam_quat_w, dim=-1, keepdim=True
        ).clamp_min(1e-12)

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

        min_radius_pix = float(self.cfg.min_visible_target_radius_pix)

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
        yaw-only base frame 下的目标相对速度 [vx, vy, vz]

        坐标定义：
        - z 轴始终与世界竖直方向一致
        - x/y 只跟随机器狗 yaw 旋转
        - 不跟随 roll/pitch 倾斜
        """
        root_pos_w = self.turret.data.root_pos_w
        root_quat_w = self.turret.data.root_quat_w

        # 先把 body frame 速度转回 world frame
        root_lin_vel_w = quat_apply(root_quat_w, self.turret.data.root_lin_vel_b)
        root_ang_vel_w = quat_apply(root_quat_w, self.turret.data.root_ang_vel_b)

        rel_pos_w = self._target_pos_w - root_pos_w

        # The yaw-only frame follows the extracted heading.  Its rate must be
        # the derivative of that heading, including roll/pitch coupling.
        yaw_rate_world = self._get_yaw_rate_world(root_quat_w, root_ang_vel_w)
        yaw_only_ang_vel_w = torch.zeros_like(root_ang_vel_w)
        yaw_only_ang_vel_w[:, 2] = yaw_rate_world

        # yaw-only 旋转坐标系的相对速度项
        base_frame_point_vel_w = root_lin_vel_w + torch.cross(yaw_only_ang_vel_w, rel_pos_w, dim=-1)

        target_rel_vel_w = self._target_vel_w - base_frame_point_vel_w
        target_rel_vel_base = self._rotate_world_to_yaw_only_local(root_quat_w, target_rel_vel_w)
        return target_rel_vel_base

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
    


    def _get_camera_prim_paths(self) -> list[str]:
        return [
            f"/World/envs/env_{i}/Lite3/Lite3/TORSO/"
            "SG2_OX03CC_5200_GMSL2_H60YA/"
            "Camera_SG2_OX03CC_5200_GMSL2_H60YA"
            for i in range(self.num_envs)
        ]


    def _push_runtime_camera_pose_to_usd(self, env_ids: torch.Tensor | None = None) -> None:
        """Push reconstructed runtime camera world pose to proxy render cameras."""
        if self._vision_camera is None:
            return
        if not hasattr(self, "_runtime_camera_proxy_paths"):
            return

        # 注意：
        # 给渲染 proxy camera 写的是 USD 相机坐标系下的姿态，
        # 不能用经过 ROS 修正后的 cam_quat_w。
        cam_pos_w, cam_quat_w_usd = self._get_runtime_camera_pose_w_usd()

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)

        for env_id_t in env_ids_t:
            env_id = int(env_id_t.item())
            self._set_camera_pose_on_prim(
                self._runtime_camera_proxy_paths[env_id],
                cam_pos_w[env_id],
                cam_quat_w_usd[env_id],
            )

        if self.cfg.debug_diagnostics and (int(self.common_step_counter) % 50 == 0):
            env0 = 0
            print(
                "[proxy camera push] "
                f"step={int(self.common_step_counter)} "
                f"proxy_path={self._runtime_camera_proxy_paths[env0]} "
                f"pos0={cam_pos_w[env0].detach().cpu().tolist()} "
                f"quat0_usd={cam_quat_w_usd[env0].detach().cpu().tolist()}"
            )


    def _sync_render_camera_before_read(self) -> None:
        """把 runtime 重建相机位姿写到 proxy camera，然后强制渲染，再刷新 TiledCamera。"""
        if self._vision_camera is None:
            return

        # 1) 先把当前 physics 下的相机世界位姿推到 proxy cameras
        self._push_runtime_camera_pose_to_usd()

        # 2) 强制触发一次渲染，让 USD 树的新位姿真正进入渲染结果
        try:
            self.sim.render()
        except Exception:
            pass

        # 3) 再刷新 TiledCamera 数据
        try:
            self._vision_camera.update(dt=0.0)
        except Exception:
            pass


    def _refresh_vision_supervision(self) -> None:
        if self._vision_camera is None:
            self._vision_valid_mask.zero_()
            return

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

        self._vision_frame_valid_history = torch.roll(self._vision_frame_valid_history, shifts=-1, dims=1)
        self._vision_frame_valid_history[:, -1] = current_frame_valid

        self._vision_history_count = torch.clamp(self._vision_history_count + 1, max=self._vision_history_length)

        full_history = self._vision_history_count >= self._vision_history_length
        history_all_valid = torch.all(self._vision_frame_valid_history, dim=1)
        self._vision_valid_mask = full_history & history_all_valid

        self._vision_direction_gt.copy_(direction_gt)
        self._vision_velocity_gt.copy_(velocity_gt)
        self._vision_velocity_base_gt.copy_(self._compute_base_relative_velocity_labels())

        # runtime camera -> target 的真实距离
        to_target_cam_w = self._target_pos_w - cam_pos_w
        self._vision_distance_gt[:] = torch.norm(to_target_cam_w, dim=-1)

        # GT 位置转到 yaw-only base 坐标系
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
        root_speed_xy = torch.norm(self.turret.data.root_lin_vel_b[:, :2], dim=-1, keepdim=True)
        dog_moving_flag = (root_speed_xy > 0.05).to(dtype=torch.float32)

        lidar_distance = self._vision_distance_gt.clone()
        lidar_valid_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        if self.cfg.enable_dynamic_target_raycast_debug and (self._dynamic_target_raycast_sensor is not None):
            lidar_distance = self._raycast_hit_distance.clone()
            lidar_valid_mask = self._raycast_hit_valid.clone()

        if not hasattr(self, "_printed_lidar_source_once"):
            self._printed_lidar_source_once = False

        if not self._printed_lidar_source_once:
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


    def _compute_ballistic_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (min_distance_to_target, hit_flag, closest_time_s).

        使用离散积分模拟弹道，但命中判定不再使用“采样点到采样点”的最小距离，
        而是使用“相邻两帧形成的线段”之间的最小距离：
        - 弹丸轨迹：bullet_pos[k] -> bullet_pos[k+1]
        - 目标轨迹：target_pos[k] -> target_pos[k+1]
        这样即使真实最近点落在两个采样时刻之间，也不会漏判。
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

        # 初始弹速 = 枪口点速度 + 枪口前向初速
        bullet_pos = muzzle_pos_w
        bullet_vel = muzzle_vel_w + muzzle_fwd_w * self.cfg.muzzle_speed

        target_pos = self._target_pos_w
        target_vel = self._target_vel_w

        bullet_pos_hist = [bullet_pos]
        target_pos_hist = [target_pos]

        for dt in self._ballistic_dt_samples:
            dt = dt.view(1, 1)

            if self.cfg.enable_air_drag:
                speed = torch.norm(bullet_vel, dim=-1, keepdim=True).clamp_min(1e-6)
                drag_acc = -self._drag_accel_coeff * speed * bullet_vel
            else:
                drag_acc = torch.zeros_like(bullet_vel)

            total_acc = self._gravity_w.unsqueeze(0) + drag_acc

            # 半隐式欧拉：先更新速度，再更新位置
            bullet_vel = bullet_vel + total_acc * dt
            bullet_pos = bullet_pos + bullet_vel * dt
            target_pos = target_pos + target_vel * dt

            bullet_pos_hist.append(bullet_pos)
            target_pos_hist.append(target_pos)

        # shape: (N, T, 3)
        bullet_pos_traj = torch.stack(bullet_pos_hist, dim=1)
        target_pos_traj = torch.stack(target_pos_hist, dim=1)

        # 每一段的起点和终点
        # shape: (N, T-1, 3)
        bullet_seg_start = bullet_pos_traj[:, :-1, :]
        bullet_seg_end   = bullet_pos_traj[:, 1:, :]
        target_seg_start = target_pos_traj[:, :-1, :]
        target_seg_end   = target_pos_traj[:, 1:, :]

        num_envs = bullet_pos_traj.shape[0]
        num_segs = bullet_seg_start.shape[1]

        # 拉平成 (N*(T-1), 3)，做批量线段-线段最短距离
        p0 = bullet_seg_start.reshape(-1, 3)
        p1 = bullet_seg_end.reshape(-1, 3)
        q0 = target_seg_start.reshape(-1, 3)
        q1 = target_seg_end.reshape(-1, 3)

        seg_min_dist_flat, seg_s_flat, _ = self._segment_segment_min_distance(p0, p1, q0, q1)

        # reshape 回 (N, T-1)
        seg_min_dist = seg_min_dist_flat.view(num_envs, num_segs)
        seg_s = seg_s_flat.view(num_envs, num_segs)

        # 找到每个 env 的最小距离所在的那一段
        min_dist, min_seg_idx = torch.min(seg_min_dist, dim=-1)

        # 最近时刻：段起始时刻 + 段内参数 * dt
        t0 = self._ballistic_time_samples[:-1]                  # (T-1,)
        dt_seg = self._ballistic_dt_samples                     # (T-1,)

        row_idx = torch.arange(num_envs, device=self.device)
        closest_time = t0[min_seg_idx] + seg_s[row_idx, min_seg_idx] * dt_seg[min_seg_idx]

        hit = min_dist < self.cfg.hit_radius

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

        ang_err, dist, to_target_dir_w = self._compute_angle_error()
        _, turret_quat_w, _ = self._compute_turret_frame()

        quat_conj = turret_quat_w.clone()
        quat_conj[:, 1:] *= -1.0
        to_target_dir_b = quat_apply(quat_conj, to_target_dir_w)

        yaw = joint_pos[:, int(self._yaw_dof_ids[0])]
        pitch = joint_pos[:, int(self._pitch_dof_ids[0])]
        yaw_vel = joint_vel[:, int(self._yaw_dof_ids[0])]
        pitch_vel = joint_vel[:, int(self._pitch_dof_ids[0])]

        projected_gravity_b = self.turret.data.projected_gravity_b
        prev_actions = self._prev_actions
        fire_charge_obs = self._fire_charge.unsqueeze(-1)

        obs = torch.cat(
            [
                to_target_dir_b,                          # 3
                ang_err.unsqueeze(-1),                   # 1
                dist.unsqueeze(-1),                      # 1
                yaw.unsqueeze(-1),                       # 1
                pitch.unsqueeze(-1),                     # 1
                yaw_vel.unsqueeze(-1),                   # 1
                pitch_vel.unsqueeze(-1),                 # 1
                self.turret.data.root_lin_vel_b,         # 3
                self.turret.data.root_ang_vel_b,         # 3
                self._locomotion_commands,               # 3
                projected_gravity_b,                     # 3
                prev_actions,                            # 3
                fire_charge_obs,                         # 1 
            ],
            dim=-1,
        )

        invalid_obs = ~torch.isfinite(obs).all(dim=-1)
        if torch.any(invalid_obs):
            self._report_invalid_envs(
                "observations",
                invalid_obs,
                {
                    "obs": obs,
                    "root_pos_w": self.turret.data.root_pos_w,
                    "root_lin_vel_b": self.turret.data.root_lin_vel_b,
                    "root_ang_vel_b": self.turret.data.root_ang_vel_b,
                    "projected_gravity_b": self.turret.data.projected_gravity_b,
                    "locomotion_commands": self._locomotion_commands,
                    "prev_actions": self._prev_actions,
                    "target_vel_w": self._target_vel_w,
                },
            )

        obs = obs.clamp(-10.0, 10.0)

        return {"policy": obs}


    def _get_rewards(self) -> torch.Tensor:
        ang_err, dist, _ = self._compute_angle_error()
        align = torch.exp(-5.0 * ang_err)

        delta_pen = (self._delta_pos ** 2).sum(dim=-1)
        act_pen = (self._actions[:, :2] ** 2).sum(dim=-1)
        action_change_pen = ((self._actions[:, :2] - self._prev_actions[:, :2]) ** 2).sum(dim=-1)

        min_ballistic_dist, ballistic_hit, _ = self._get_cached_ballistic_metrics()
        ballistic_score = torch.exp(-6.0 * min_ballistic_dist)
        fire_now = self._fire_command.bool()

        aiming_enabled = self._aim_warmup_left_steps <= 0
        aiming_mask = aiming_enabled.float()

        hit_now = fire_now & ballistic_hit & (dist < self.cfg.max_hit_distance) & aiming_enabled
        bad_fire = fire_now & (~hit_now) & aiming_enabled

        # 开火质量奖励：鼓励“敢开火”，但要在弹道接近时才有明显收益
        fire_quality_reward = fire_now.float() * aiming_mask * (8.0 * torch.exp(-20.0 * min_ballistic_dist))

        # 近失奖励：差一点也给正反馈，避免全靠稀疏命中奖励
        near_fire = fire_now & (~hit_now) & aiming_enabled & (min_ballistic_dist < 0.12)
        near_fire_reward = 1.0 * near_fire.float()

        reward = (
            self.cfg.w_align * align * aiming_mask
            + self.cfg.w_ballistic * ballistic_score * aiming_mask
            + fire_quality_reward
            + self.cfg.r_hit * hit_now.float()
            + near_fire_reward
            + self.cfg.r_bad_fire * bad_fire.float()
            + self.cfg.r_fire_cost * fire_now.float() * aiming_mask
            + self.cfg.w_rate * delta_pen
            + self.cfg.w_action * act_pen * aiming_mask
            + self.cfg.w_action_change * action_change_pen * aiming_mask
            + self.cfg.w_base_ang_vel
            * (self.turret.data.root_ang_vel_b ** 2).sum(dim=-1).clamp(max=100.0)
            * aiming_mask
            + self.cfg.r_time_penalty * aiming_mask
        )

        reward = reward.clamp(-20.0, 40.0)
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, dist, _ = self._compute_angle_error()
        _, ballistic_hit, _ = self._get_cached_ballistic_metrics()
        fire_now = self._fire_command.bool()

        warmup_done = self._aim_warmup_left_steps <= 0
        hit_now = fire_now & warmup_done & ballistic_hit & (dist < self.cfg.max_hit_distance)

        self._hit_flag |= hit_now

        root_height = self.turret.data.root_pos_w[:, 2]
        upright_cos = -self.turret.data.projected_gravity_b[:, 2]
        fallen = (root_height < self.cfg.failure_height_threshold) | (
            upright_cos < self.cfg.failure_tilt_cos_threshold
        )
        fallen_effective = fallen & warmup_done
        invalid_state = self._get_invalid_state_mask()
        target_out_of_view = self._get_target_out_of_camera_sector_mask() & warmup_done
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = hit_now | fallen_effective | invalid_state | target_out_of_view

        done_mask = terminated | time_out
        self._eval_total_episodes += torch.count_nonzero(done_mask)
        self._eval_total_hits += torch.count_nonzero(hit_now)
        self._eval_total_timeouts += torch.count_nonzero(time_out)
        self._eval_total_fallen += torch.count_nonzero(fallen_effective)
        self._eval_total_invalid += torch.count_nonzero(invalid_state)
        self._eval_total_target_out_of_view += torch.count_nonzero(target_out_of_view)

        # 统计：在 episode 结束时记录该回合起始目标位置
        done_env_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            if self._collect_eval_details:
                episode_fires = self._episode_fire_count[done_env_ids].detach().cpu().tolist()
                episode_hits = self._hit_flag[done_env_ids].to(dtype=torch.int64).detach().cpu().tolist()
                self._eval_episode_fire_counts.extend(int(v) for v in episode_fires)
                self._eval_episode_hits.extend(int(v) for v in episode_hits)

            finished_start_targets = self._episode_start_target_pos_w[done_env_ids]
            self._eval_target_spawn_count += finished_start_targets.shape[0]
            self._eval_target_spawn_sum += finished_start_targets.sum(dim=0)

            remaining = self._eval_samples_limit - len(self._eval_target_spawn_samples)
            if remaining > 0:
                sample_pos = finished_start_targets.detach().cpu().tolist()[:remaining]
                self._eval_target_spawn_samples.extend(
                    [[float(p[0]), float(p[1]), float(p[2])] for p in sample_pos]
                )

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
        self._eval_target_spawn_count.zero_()
        self._eval_target_spawn_sum.zero_()
        self._eval_fire_min_dist_count.zero_()
        self._eval_fire_min_dist_sum.zero_()
        self._eval_fire_min_dist_min.fill_(float("inf"))
        self._eval_fire_min_dist_max.zero_()
        self._episode_fire_count.zero_()
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
            "hit_rate": float(hits / denom),
            "avg_fires_per_episode": float(fires / denom),
            "timeout_rate": float(timeouts / denom),
            "fallen_rate": float(fallen / denom),
            "invalid_rate": float(invalid / denom),
            "target_out_of_view_rate": float(target_out_of_view / denom),
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
        target_out_of_view = self._get_target_out_of_camera_sector_mask()[env_ids_t]
        time_out = self.reset_time_outs[env_ids_t]
        hit_rate = torch.count_nonzero(self._hit_flag[env_ids_t]).float() / episode_count

        self.extras["log"] = {
            "Episode_Termination/hit_rate": hit_rate,
            "Episode_Termination/fallen_rate": torch.count_nonzero(fallen).float() / episode_count,
            "Episode_Termination/invalid_rate": torch.count_nonzero(invalid_state).float() / episode_count,
            "Episode_Termination/target_out_of_view_rate": torch.count_nonzero(target_out_of_view).float() / episode_count,
            "Episode_Termination/time_out_rate": torch.count_nonzero(time_out).float() / episode_count,
            "Episode_Termination/mean_root_height": torch.mean(root_height),
            "Episode_Termination/mean_upright_cos": torch.mean(upright_cos),
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
        self._fire_charge[env_ids_t] = 0.0
        self._episode_fire_count[env_ids_t] = 0
        self._target_vel_w[env_ids_t] = 0.0
        # reset 后 ballistic cache 失效
        # reset vision buffers for these envs
        self._vision_rgb_history[env_ids_t] = 0.0
        self._vision_frame_valid_history[env_ids_t] = False
        self._vision_history_count[env_ids_t] = 0
        self._vision_valid_mask[env_ids_t] = False
        self._vision_direction_gt[env_ids_t] = 0.0
        self._vision_velocity_gt[env_ids_t] = 0.0
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
        if self.cfg.debug_diagnostics:
            print("[TurretAimEnv][reset debug] default_root_state finite =", torch.isfinite(default_root_state).all().item())
            print("[TurretAimEnv][reset debug] joint_pos finite =", torch.isfinite(joint_pos).all().item())
            print("[TurretAimEnv][reset debug] joint_vel finite =", torch.isfinite(joint_vel).all().item())
            print("[TurretAimEnv][reset debug] default_root_state[0] =", default_root_state[0].detach().cpu().tolist())
            print("[TurretAimEnv][reset debug] joint_pos[0] =", joint_pos[0].detach().cpu().tolist())
            print("[TurretAimEnv][reset debug] joint_vel[0] =", joint_vel[0].detach().cpu().tolist())

        self.turret.write_root_pose_to_sim(default_root_state[:, :7], env_ids_sim)
        self.turret.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_sim)
        self.turret.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_sim)
        # self.turret.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        # self.turret.set_joint_velocity_target(joint_vel, env_ids=env_ids_t)

        # sample target position/velocity
        self._resample_targets(env_ids_t, sample_position=True, sample_velocity=True)

        # sync rendered target sphere
        self._update_rendered_target_spheres(env_ids_t)

        # 记录该 episode 的起始目标位置（在 done 时统一计入评估统计）
        self._episode_start_target_pos_w[env_ids_t] = self._target_pos_w[env_ids_t]

        # visualize target
        if self._target_markers is not None and self._target_markers.is_visible():
            # build full translations array (num_envs, 3)
            translations = self._target_pos_w.detach().cpu().numpy()
            self._target_markers.visualize(translations=translations)


    def _update_rendered_target_spheres(self, env_ids_t: torch.Tensor | None = None) -> None:
        """Move pure-visual target spheres to self._target_pos_w.

        注意：
        - 这里不要用 XformCommonAPI.SetTranslate()
        - 直接查找/创建 xformOp:translate
        - 写入的是 env-local 坐标，而不是 world 坐标
        """
        if not self.cfg.render_target_sphere:
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

        if self.cfg.debug_diagnostics and int(self.common_step_counter) % 50 == 0 and len(env_ids_list) > 0:
            env0 = env_ids_list[0]
            p0_w = target_pos_cpu[0].tolist()
            p0_local = (target_pos_cpu[0] - env_origins_cpu[0]).tolist()
            print(
                "[target sphere visual update]",
                f"step={int(self.common_step_counter)}",
                f"moved={moved_count}",
                f"invalid={invalid_count}",
                f"env0_path={self._target_sphere_prim_paths[env0]}",
                f"target_pos0_w={p0_w}",
                f"target_pos0_local={p0_local}",
            )


    def _enforce_target_above_ground(self, env_ids_t: torch.Tensor | None = None) -> None:
        """Enforce both constraints on target z:
        1) z in [target_height_range[0], target_height_range[1]]
        2) z - radius > 0
        """
        if not bool(getattr(self.cfg, "constrain_target_height", True)):
            return
        if env_ids_t is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        eps = 1e-3
        z_min_cfg = float(self.cfg.target_height_range[0])
        z_max_cfg = float(self.cfg.target_height_range[1])
        min_center_z = max(z_min_cfg, float(self.cfg.target_sphere_radius) + eps)
        self._target_pos_w[env_ids_t, 2] = torch.clamp(self._target_pos_w[env_ids_t, 2], min=min_center_z, max=z_max_cfg)
    
    
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

        target_count = max(1, int(getattr(self.cfg, "num_visual_targets", 1)))
        # Each explicit TargetSphere_* entry is merged into one per-environment
        # mesh, so the flattened sensor slot is stable and equals target ID.
        self._dynamic_target_raycast_mesh_slots = tuple(range(target_count))
        raycast_targets = []
        for target_id in range(target_count):
            target_name = "TargetSphere" if target_id == 0 else f"TargetSphere_{target_id:02d}"
            raycast_targets.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr=f"{{ENV_REGEX_NS}}/{target_name}",
                    is_shared=False,
                    merge_prim_meshes=True,
                    track_mesh_transforms=True,
                )
            )

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
            update_mesh_ids=True,
            mesh_prim_paths=raycast_targets,
        )

        self._dynamic_target_raycast_sensor = MultiMeshRayCaster(sensor_cfg)
        self.scene.sensors["dynamic_target_raycast"] = self._dynamic_target_raycast_sensor

        print(
            "[raycast dbg] MultiMeshRayCaster initialized on TORSO:",
            f"prim_path={self.cfg.dynamic_target_raycast_parent_prim_path_expr}",
            f"h_fov={h_fov}",
            f"v_fov={v_fov}",
            f"h_res={h_res}",
            f"v_res={v_res}",
            f"channels={channels}",
            f"targets={target_count}",
            f"mesh_slots={self._dynamic_target_raycast_mesh_slots}",
            f"min_range={self.cfg.dynamic_target_raycast_min_range}",
            f"max_range={self.cfg.dynamic_target_raycast_max_range}",
        )

    def set_dynamic_target_raycast_angles(
        self,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        *,
        target_mesh_id: int | None = None,
    ) -> None:
        """由外部视觉模块设置目标方向角和可选 mesh slot。

        注意：
        1) 这里只记录“要查询的方向”，不旋转传感器本体；
        2) ``target_mesh_id`` 是当前传感器配置中的目标 mesh slot，
           不是 Warp 全局 mesh object id；
        3) 这里的数据源必须是视觉预测，不能用 GT 覆盖。
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
        self._external_raycast_target_mesh_id = (
            None if target_mesh_id is None else int(target_mesh_id)
        )


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


    def _debug_raycast_dynamic_target(self, *, update_sensor: bool = True) -> None:
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
            if update_sensor:
                self._dynamic_target_raycast_sensor.update(dt=self.physics_dt)
            data = self._dynamic_target_raycast_sensor.data

            sensor_pos_w = data.pos_w            # [N, 3]
            ray_hits_w = data.ray_hits_w         # [N, R, 3]
            ray_mesh_ids = getattr(data, "ray_mesh_ids", None)

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
            #    Mesh IDs identify the configured TargetSphere_* slot. The
            #    angle window remains as a geometric consistency check.
            # ------------------------------------------------------------
            vec_to_hit_w = ray_hits_w - sensor_pos_w.unsqueeze(1)     # [N, R, 3]
            hit_dist_sensor = torch.norm(vec_to_hit_w, dim=-1)        # [N, R]

            finite_hits_mask = torch.isfinite(ray_hits_w).all(dim=-1)  # [N, R]
            min_hit_dist = max(0.0, float(self.cfg.dynamic_target_raycast_min_range))
            max_hit_dist = max(min_hit_dist, float(self.cfg.dynamic_target_raycast_max_range))
            range_hits_mask = (hit_dist_sensor >= min_hit_dist) & (hit_dist_sensor <= max_hit_dist)
            valid_hits_mask = finite_hits_mask & range_hits_mask

            # Each explicit TargetSphere_* entry is one mesh slot in the
            # per-environment MultiMeshRayCaster list. Use that slot to keep
            # nearby targets from stealing each other's range measurement.
            target_mesh_id = getattr(self, "_external_raycast_target_mesh_id", None)
            if target_mesh_id is not None:
                if ray_mesh_ids is None:
                    raise RuntimeError(
                        "MultiMeshRayCaster mesh IDs are required for target association"
                    )
                if ray_mesh_ids.ndim == 3 and ray_mesh_ids.shape[-1] == 1:
                    ray_mesh_ids = ray_mesh_ids.squeeze(-1)
                if ray_mesh_ids.shape != hit_dist_sensor.shape:
                    raise RuntimeError(
                        "Unexpected ray_mesh_ids shape: "
                        f"expected {tuple(hit_dist_sensor.shape)}, "
                        f"got {tuple(ray_mesh_ids.shape)}"
                    )
                valid_hits_mask &= ray_mesh_ids == int(target_mesh_id)

            hit_dirs_w = vec_to_hit_w / hit_dist_sensor.unsqueeze(-1).clamp_min(1e-6)

            # ------------------------------------------------------------
            # 3) 角窗筛选（张量化）
            # ------------------------------------------------------------
            h_res_deg = float(self.cfg.dynamic_target_raycast_horizontal_resolution_deg)
            v_res_deg = float(self.cfg.dynamic_target_raycast_vertical_resolution_deg)

            # The configured floor can be tightened for exact multi-target
            # bearings while preserving the wider legacy visual-debug window.
            primary_floor_deg = max(
                0.0, float(self.cfg.dynamic_target_raycast_primary_cone_deg)
            )
            fallback_floor_deg = max(
                primary_floor_deg,
                float(self.cfg.dynamic_target_raycast_fallback_cone_deg),
            )
            yaw_half_window_deg_1 = max(4.0 * h_res_deg, primary_floor_deg)
            pitch_half_window_deg_1 = max(4.0 * v_res_deg, primary_floor_deg)

            yaw_half_window_deg_2 = max(8.0 * h_res_deg, fallback_floor_deg)
            pitch_half_window_deg_2 = max(8.0 * v_res_deg, fallback_floor_deg)

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
            center_offset_radius_scale = max(
                0.0,
                float(self.cfg.dynamic_target_raycast_center_offset_radius_scale),
            )

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

            # The mean of visible sphere-surface hits lies closer to the center than
            # the nearest surface point; using a full radius creates a positive range bias.
            est_center_w = mean_hit_w + cam_to_hit_dir_w * (target_radius * center_offset_radius_scale)  # [N, 3]

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
            speed = sample_uniform(self.cfg.target_speed_range[0], self.cfg.target_speed_range[1], (env_count, 1), self.device)
            heading = sample_uniform(-math.pi, math.pi, (env_count, 1), self.device)
            vel_xy = torch.cat([speed * torch.cos(heading), speed * torch.sin(heading)], dim=-1)
            vel_z = sample_uniform(self.cfg.target_vertical_speed_range[0], self.cfg.target_vertical_speed_range[1], (env_count, 1), self.device)
            self._target_vel_w[env_ids_t] = torch.cat([vel_xy, vel_z], dim=-1)
            
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

        # 同步可视化球体
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
                    speed = sample_uniform(self.cfg.target_speed_range[0], self.cfg.target_speed_range[1], (env_count, 1), self.device)
                    heading = sample_uniform(-math.pi, math.pi, (env_count, 1), self.device)
                    vel_xy = torch.cat([speed * torch.cos(heading), speed * torch.sin(heading)], dim=-1)
                    vel_z = sample_uniform(self.cfg.target_vertical_speed_range[0], self.cfg.target_vertical_speed_range[1], (env_count, 1), self.device)
                    self._target_vel_w[env_ids_t] = torch.cat([vel_xy, vel_z], dim=-1)
                    
                    # --- 新增：随机采样加速度 ---
                    acc_xy_mag = sample_uniform(self.cfg.target_acc_range[0], self.cfg.target_acc_range[1], (env_count, 1), self.device)
                    acc_heading = sample_uniform(-math.pi, math.pi, (env_count, 1), self.device)
                    acc_xy = torch.cat([acc_xy_mag * torch.cos(acc_heading), acc_xy_mag * torch.sin(acc_heading)], dim=-1)
                    acc_z = sample_uniform(self.cfg.target_vertical_acc_range[0], self.cfg.target_vertical_acc_range[1], (env_count, 1), self.device)
                    self._target_acc_w[env_ids_t] = torch.cat([acc_xy, acc_z], dim=-1)
                else:
                    self._target_vel_w[env_ids_t] = 0.0
                    self._target_acc_w[env_ids_t] = 0.0


    def _handle_target_out_of_bounds(self):
        """目标越界处理：
        1) 相对“机器狗当前 root 位置”的扇形 / 高度边界反弹
        2) 相对相机的距离壳 [d_min, d_max] 反弹

        关键修改：
        - anchor 不再固定为 reset 时的出生点
        - 每一帧都以机器狗当前 root_pos_w 作为目标运动边界的锚点
        - 反弹时同时处理速度和加速度，避免“速度反了、加速度还继续往墙外推”
        """
        eps = 1e-6

        # ------------------------------------------------------------
        # A. 动态 anchor：使用机器狗当前 root 位置作为锚点
        # ------------------------------------------------------------
        current_anchor_w = self.turret.data.root_pos_w
        self._target_anchor_w[:] = current_anchor_w

        rel = self._target_pos_w - current_anchor_w  # (N, 3)
        rel_xy = rel[:, :2]
        vel_xy = self._target_vel_w[:, :2]
        acc_xy = self._target_acc_w[:, :2]

        r = torch.norm(rel_xy, dim=-1).clamp_min(eps)
        yaw = torch.atan2(rel_xy[:, 1], rel_xy[:, 0])

        r_min, r_max = float(self.cfg.target_radius_range[0]), float(self.cfg.target_radius_range[1])
        yaw_min, yaw_max = float(self.cfg.target_yaw_range[0]), float(self.cfg.target_yaw_range[1])
        if bool(getattr(self.cfg, "constrain_target_height", True)):
            z_min, z_max = (
                float(self.cfg.target_height_range[0]),
                float(self.cfg.target_height_range[1]),
            )

            # ---- Z 边界反弹 ----
            below_z = rel[:, 2] < z_min
            above_z = rel[:, 2] > z_max
            if torch.any(below_z):
                rel[below_z, 2] = z_min
                self._target_vel_w[below_z, 2] = torch.abs(self._target_vel_w[below_z, 2])
                self._target_acc_w[below_z, 2] = torch.abs(self._target_acc_w[below_z, 2])
            if torch.any(above_z):
                rel[above_z, 2] = z_max
                self._target_vel_w[above_z, 2] = -torch.abs(self._target_vel_w[above_z, 2])
                self._target_acc_w[above_z, 2] = -torch.abs(self._target_acc_w[above_z, 2])

        # ---- 半径边界反弹（相对当前 root anchor）----
        unit_r = rel_xy / r.unsqueeze(-1)
        radial_speed = torch.sum(vel_xy * unit_r, dim=-1, keepdim=True)
        tangential_speed = vel_xy - radial_speed * unit_r

        radial_acc = torch.sum(acc_xy * unit_r, dim=-1, keepdim=True)
        tangential_acc = acc_xy - radial_acc * unit_r

        inner_hit = r < r_min
        outer_hit = r > r_max
        radial_hit = inner_hit | outer_hit
        if torch.any(radial_hit):
            r_clamped = torch.clamp(r, min=r_min, max=r_max)
            rel_xy[radial_hit] = unit_r[radial_hit] * r_clamped[radial_hit].unsqueeze(-1)

            vel_xy_reflect = tangential_speed - radial_speed * unit_r
            acc_xy_reflect = tangential_acc - radial_acc * unit_r

            self._target_vel_w[radial_hit, :2] = vel_xy_reflect[radial_hit]
            self._target_acc_w[radial_hit, :2] = acc_xy_reflect[radial_hit]

        # ---- Yaw 边界反弹（相对边界射线镜像）----
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
            self._target_vel_w[low_yaw_hit, :2] = _reflect_about_ray(
                self._target_vel_w[low_yaw_hit, :2], yaw_min
            )
            self._target_acc_w[low_yaw_hit, :2] = _reflect_about_ray(
                self._target_acc_w[low_yaw_hit, :2], yaw_min
            )

        if torch.any(high_yaw_hit):
            rel_xy[high_yaw_hit, 0] = r2[high_yaw_hit] * math.cos(yaw_max)
            rel_xy[high_yaw_hit, 1] = r2[high_yaw_hit] * math.sin(yaw_max)
            self._target_vel_w[high_yaw_hit, :2] = _reflect_about_ray(
                self._target_vel_w[high_yaw_hit, :2], yaw_max
            )
            self._target_acc_w[high_yaw_hit, :2] = _reflect_about_ray(
                self._target_acc_w[high_yaw_hit, :2], yaw_max
            )

        # 先写回“相对当前 root anchor”的世界坐标
        rel[:, :2] = rel_xy
        self._target_pos_w = current_anchor_w + rel

        # ------------------------------------------------------------
        # B. 相对相机的视觉有效距离壳 [d_min, d_max]
        # ------------------------------------------------------------
        if self._vision_camera is None:
            return

        d_min = float(self.cfg.vision_valid_distance_range[0])
        d_max = float(self.cfg.vision_valid_distance_range[1])
        margin = float(self.cfg.vision_distance_bounce_margin)

        cam_pos_w, _ = self._get_runtime_camera_pose_w()
        target_pos_w = self._target_pos_w
        target_vel_w = self._target_vel_w
        target_acc_w = self._target_acc_w

        rel_cam = target_pos_w - cam_pos_w
        dist_cam = torch.norm(rel_cam, dim=-1, keepdim=True).clamp_min(eps)
        unit_cam = rel_cam / dist_cam

        # >0 表示越来越远，<0 表示越来越近
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
            hit_cam_pos_w = cam_pos_w[hit_mask]
            hit_unit_cam = unit_cam[hit_mask]
            hit_target_dist = target_dist[hit_mask]
            hit_target_vel_w = target_vel_w[hit_mask]
            hit_target_acc_w = target_acc_w[hit_mask]
            hit_radial_speed = radial_speed_cam[hit_mask]
            hit_radial_acc = radial_acc_cam[hit_mask]

            # 位置拉回到有效壳面
            self._target_pos_w[hit_mask] = hit_cam_pos_w + hit_unit_cam * hit_target_dist

            # 只反转径向分量，保留切向分量
            self._target_vel_w[hit_mask] = hit_target_vel_w - 2.0 * hit_radial_speed * hit_unit_cam
            self._target_acc_w[hit_mask] = hit_target_acc_w - 2.0 * hit_radial_acc * hit_unit_cam

            # 轻微阻尼
            self._target_vel_w[hit_mask] *= 0.98
            self._target_acc_w[hit_mask] *= 0.98


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

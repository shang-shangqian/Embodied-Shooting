from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import asym_central_value_gru_network  # noqa: F401
from isaaclab.app import AppLauncher
from target_follow_motion import install_target_follow_motion
import types

from rl_perception_bridge import install_rl_perception_bridge


def _install_runtime_track_draw(
    base_env: Any,
    track_env_id: int,
    max_points: int,
    shot_history: int,
    ballistic_draw_stride: int,
    robot_color: tuple[float, float, float, float],
    target_color: tuple[float, float, float, float],
    bullet_color: tuple[float, float, float, float],
    robot_width: float,
    target_width: float,
    bullet_width: float,
    pause_between_episodes: bool,
    pause_seconds: float,
) -> bool:
    """在不修改环境源码的前提下，为 TurretAimEnv 动态挂接轨迹画线逻辑。"""
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    except ModuleNotFoundError:
        print("[eval] Debug draw extension unavailable. Skip trajectory lines.")
        return False

    if not hasattr(base_env, "_pre_physics_step") or not hasattr(base_env, "_reset_idx"):
        print("[eval] Env does not expose expected hooks. Skip trajectory lines.")
        return False

    draw = omni_debug_draw.acquire_debug_draw_interface()
    if draw is None:
        print("[eval] Failed to acquire debug draw interface. Skip trajectory lines.")
        return False

    base_env._track_draw_interface = draw
    base_env._track_env_id = int(track_env_id)
    base_env._track_max_points = max(2, int(max_points))
    base_env._track_shot_history = max(1, int(shot_history))
    base_env._track_ballistic_stride = max(1, int(ballistic_draw_stride))
    base_env._track_robot_color = [float(v) for v in robot_color]
    base_env._track_target_color = [float(v) for v in target_color]
    base_env._track_bullet_color = [float(v) for v in bullet_color]
    base_env._track_robot_width = float(robot_width)
    base_env._track_target_width = float(target_width)
    base_env._track_bullet_width = float(bullet_width)
    base_env._track_robot_points = []
    base_env._track_target_points = []
    base_env._track_bullet_polylines = []
    base_env._pause_between_episodes = bool(pause_between_episodes)
    base_env._pause_seconds = max(0.0, float(pause_seconds))

    # 新增：记录 tracked env 上一帧是否仍处于“占位目标阶段”
    base_env._track_prev_target_pending = False

    original_pre_physics_step = base_env._pre_physics_step
    original_reset_idx = base_env._reset_idx

    def _append_point(
        point_list: list[list[float]],
        point: list[float],
        keep: int,
        reset_jump_threshold: float = 1.0,   # 原来 3.0，改小一点更稳
    ):
        if len(point_list) > 0:
            last = point_list[-1]
            dx = float(point[0] - last[0])
            dy = float(point[1] - last[1])
            dz = float(point[2] - last[2])
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist > reset_jump_threshold:
                point_list.clear()

        point_list.append(point)
        if len(point_list) > keep:
            del point_list[: len(point_list) - keep]

    def _append_polyline(polyline: list[list[float]]):
        if len(polyline) < 2:
            return
        base_env._track_bullet_polylines.append(polyline)
        if len(base_env._track_bullet_polylines) > base_env._track_shot_history:
            del base_env._track_bullet_polylines[: len(base_env._track_bullet_polylines) - base_env._track_shot_history]

    def _build_bullet_polyline(env_id: int) -> list[list[float]]:
        muzzle_pos_w, _, muzzle_fwd_w = base_env._compute_turret_frame()
        if base_env._muzzle_body_id is not None:
            body_pos_w = base_env.turret.data.body_pos_w[:, base_env._muzzle_body_id]
        else:
            body_pos_w = base_env.turret.data.root_pos_w

        muzzle_vel_w = base_env._compute_muzzle_velocity_w(body_pos_w=body_pos_w, muzzle_pos_w=muzzle_pos_w)
        bullet_pos = muzzle_pos_w[env_id : env_id + 1].clone()
        bullet_vel = muzzle_vel_w[env_id : env_id + 1] + muzzle_fwd_w[env_id : env_id + 1] * base_env.cfg.muzzle_speed

        polyline: list[list[float]] = [bullet_pos.squeeze(0).detach().cpu().tolist()]
        for i, dt in enumerate(base_env._ballistic_dt_samples):
            dt = dt.view(1, 1)
            if base_env.cfg.enable_air_drag:
                speed = bullet_vel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                drag_acc = -base_env._drag_accel_coeff * speed * bullet_vel
            else:
                drag_acc = bullet_vel * 0.0
            total_acc = base_env._gravity_w.unsqueeze(0) + drag_acc
            bullet_vel = bullet_vel + total_acc * dt
            bullet_pos = bullet_pos + bullet_vel * dt
            if (i % base_env._track_ballistic_stride) == 0:
                polyline.append(bullet_pos.squeeze(0).detach().cpu().tolist())

        end_point = bullet_pos.squeeze(0).detach().cpu().tolist()
        if polyline[-1] != end_point:
            polyline.append(end_point)
        return polyline

    def _polyline_to_segments(points: list[list[float]], color: list[float], width: float, starts, ends, colors, widths):
        if len(points) < 2:
            return
        for p0, p1 in zip(points[:-1], points[1:]):
            starts.append(p0)
            ends.append(p1)
            colors.append(color)
            widths.append(width)

    def _draw_all_tracks():
        env_id = base_env._track_env_id
        if env_id < 0 or env_id >= int(base_env.num_envs):
            return

        robot_pos = base_env.turret.data.root_pos_w[env_id, :3].detach().cpu().tolist()
        _append_point(base_env._track_robot_points, [float(v) for v in robot_pos], base_env._track_max_points)

        # -----------------------------
        # 关键修复：占位目标阶段不记录 target 轨迹
        # True -> False 的那个瞬间先 clear，再开始记录正式目标
        # -----------------------------
        target_pending = False
        if hasattr(base_env, "_pending_target_resample"):
            target_pending = bool(base_env._pending_target_resample[env_id].item())

        prev_pending = bool(getattr(base_env, "_track_prev_target_pending", False))

        if prev_pending and (not target_pending):
            # 刚从占位目标切到正式目标，清掉旧 target 轨迹，避免连线
            base_env._track_target_points.clear()

        base_env._track_prev_target_pending = target_pending

        if not target_pending:
            target_pos = base_env._target_pos_w[env_id, :3].detach().cpu().tolist()
            _append_point(
                base_env._track_target_points,
                [float(v) for v in target_pos],
                base_env._track_max_points,
            )

        fire_now = base_env._fire_command.bool()
        if bool(fire_now[env_id].item()):
            import torch

            is_hit_fire = False
            if hasattr(base_env, "_get_cached_ballistic_metrics"):
                _, ballistic_hit, _ = base_env._get_cached_ballistic_metrics()
                dist = torch.norm(base_env._target_pos_w - base_env.turret.data.root_pos_w[:, :3], dim=-1)
                if hasattr(base_env, "_aim_warmup_left_steps"):
                    aiming_enabled = base_env._aim_warmup_left_steps <= 0
                else:
                    aiming_enabled = torch.ones_like(fire_now, dtype=torch.bool)
                hit_now = ballistic_hit & (dist < float(base_env.cfg.max_hit_distance)) & aiming_enabled
                is_hit_fire = bool(hit_now[env_id].item())

            if is_hit_fire:
                base_env._track_bullet_polylines.clear()
                _append_polyline(_build_bullet_polyline(env_id))

        starts: list[list[float]] = []
        ends: list[list[float]] = []
        colors: list[list[float]] = []
        widths: list[float] = []

        _polyline_to_segments(
            base_env._track_robot_points,
            base_env._track_robot_color,
            base_env._track_robot_width,
            starts,
            ends,
            colors,
            widths,
        )
        _polyline_to_segments(
            base_env._track_target_points,
            base_env._track_target_color,
            base_env._track_target_width,
            starts,
            ends,
            colors,
            widths,
        )
        for line in base_env._track_bullet_polylines:
            _polyline_to_segments(
                line,
                base_env._track_bullet_color,
                base_env._track_bullet_width,
                starts,
                ends,
                colors,
                widths,
            )

        base_env._track_draw_interface.clear_lines()
        if len(starts) > 0:
            base_env._track_draw_interface.draw_lines(starts, ends, colors, widths)

    def patched_pre_physics_step(self, actions):
        original_pre_physics_step(actions)
        try:
            _draw_all_tracks()
        except Exception as exc:
            if not hasattr(self, "_track_draw_warned"):
                print(f"[eval] trajectory draw disabled due to runtime error: {exc}")
                self._track_draw_warned = True

    def patched_reset_idx(self, env_ids):
        should_pause = False
        hit_on_track_env = False
        try:
            if self._pause_between_episodes:
                import torch

                if env_ids is None:
                    env_ids_t = torch.arange(int(self.num_envs), device=self.device, dtype=torch.long)
                else:
                    env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

                contains_track_env = bool(torch.any(env_ids_t == self._track_env_id).item())
                if contains_track_env:
                    if hasattr(self, "_hit_flag"):
                        hit_on_track_env = bool(self._hit_flag[self._track_env_id].item())
                    if hasattr(self, "episode_length_buf"):
                        should_pause = bool(torch.any(self.episode_length_buf[env_ids_t] > 0).item())
                    else:
                        should_pause = bool(getattr(self, "common_step_counter", 0) > 0)
        except Exception:
            should_pause = False
            hit_on_track_env = False

        if should_pause and self._pause_seconds > 0.0:
            try:
                if hit_on_track_env:
                    self._track_bullet_polylines.clear()
                    _append_polyline(_build_bullet_polyline(self._track_env_id))
                _draw_all_tracks()
                if hasattr(self, "sim"):
                    self.sim.render()
            except Exception:
                pass
            print(f"[eval] Episode finished. Hold frame {self._pause_seconds:.2f}s before reset...")
            start_t = time.time()
            while (time.time() - start_t) < self._pause_seconds:
                try:
                    if hasattr(self, "sim"):
                        self.sim.render()
                except Exception:
                    pass
                time.sleep(1.0 / 60.0)

        result = original_reset_idx(env_ids)
        try:
            env_ids_t = env_ids
            if env_ids_t is None:
                reset_all = True
            else:
                import torch
                env_ids_t = torch.as_tensor(env_ids_t, device=self.device, dtype=torch.long)
                reset_all = bool(torch.any(env_ids_t == self._track_env_id).item())

            if reset_all:
                self._track_robot_points.clear()
                self._track_target_points.clear()
                self._track_bullet_polylines.clear()
                self._track_prev_target_pending = False
                self._track_draw_interface.clear_lines()
        except Exception:
            pass
        return result

    import types
    base_env._pre_physics_step = types.MethodType(patched_pre_physics_step, base_env)
    base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)
    print(f"[eval] Trajectory line drawing enabled (track env id: {base_env._track_env_id}).")
    return True


def _sync_articulation_to_usd_for_viewport(base_env: Any) -> None:
    """
    Eval-only real articulation link USD sync.

    用途：
    - fabric=False 时，Viewport 不一定自动显示 PhysX articulation 的最新 link 姿态；
    - CPU PhysX 下，允许把真实 articulation link 的 USD xform 写成当前物理 pose；
    - 这样 Viewport 里显示的是“真实机器狗”，不是 visual clone，腿部不会悬空错位。

    注意：
    - 必须配合 --cpu_physx_view 使用；
    - 不要在 GPU PhysX / Direct GPU API 下使用，否则会触发：
      PxArticulationLink::setGlobalPose(): illegal if eENABLE_DIRECT_GPU_API is enabled.
    - 不要放到训练脚本里。
    """
    if not bool(getattr(base_env, "_viewport_real_link_sync_enabled", False)):
        return

    if not hasattr(base_env, "turret") or base_env.turret is None:
        return

    try:
        import re
        import torch
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom, Gf, Usd
    except Exception as exc:
        if not hasattr(base_env, "_warned_real_link_sync_import_failed"):
            base_env._warned_real_link_sync_import_failed = True
            print(f"[eval real link sync] import failed: {exc}")
        return

    # 保护：如果不是 CPU PhysX，就不要碰真实 articulation link
    sim_device = ""
    try:
        sim_device = str(getattr(base_env.cfg.sim, "device", "")).lower()
    except Exception:
        sim_device = ""

    if sim_device != "cpu":
        if not hasattr(base_env, "_warned_real_link_sync_not_cpu"):
            base_env._warned_real_link_sync_not_cpu = True
            print(
                "[eval real link sync] skip real link sync because sim.device is not cpu. "
                f"current sim.device={sim_device!r}. "
                "Run with --cpu_physx_view."
            )
        return

    stage = get_current_stage()
    if stage is None:
        return

    data = base_env.turret.data

    if not hasattr(data, "body_pos_w") or not hasattr(data, "body_quat_w"):
        if not hasattr(base_env, "_warned_no_body_pose_for_real_link_sync"):
            base_env._warned_no_body_pose_for_real_link_sync = True
            print("[eval real link sync] turret.data has no body_pos_w/body_quat_w.")
        return

    body_pos_w = data.body_pos_w.detach()
    body_quat_w = data.body_quat_w.detach()

    if body_pos_w.ndim != 3 or body_quat_w.ndim != 3:
        if not hasattr(base_env, "_warned_bad_real_link_body_pose_shape"):
            base_env._warned_bad_real_link_body_pose_shape = True
            print(
                "[eval real link sync] unexpected body pose shape:",
                f"body_pos_w={tuple(body_pos_w.shape)}",
                f"body_quat_w={tuple(body_quat_w.shape)}",
            )
        return

    if not torch.isfinite(body_pos_w).all() or not torch.isfinite(body_quat_w).all():
        if not hasattr(base_env, "_warned_real_link_sync_nonfinite"):
            base_env._warned_real_link_sync_nonfinite = True
            print("[eval real link sync] non-finite body pose, skip.")
        return

    num_envs = int(base_env.num_envs)
    body_names = list(base_env.turret.body_names)
    num_bodies = len(body_names)

    sync_all_envs = bool(getattr(base_env, "_viewport_sync_all_envs", False))
    if sync_all_envs:
        sync_env_ids = list(range(num_envs))
    else:
        track_env_id = int(getattr(base_env, "_track_env_id", 0))
        track_env_id = max(0, min(track_env_id, num_envs - 1))
        sync_env_ids = [track_env_id]

    def _extract_env_id(path: str):
        m = re.search(r"/env_(\d+)/", path)
        if m is None:
            return None
        return int(m.group(1))

    # ------------------------------------------------------------
    # 1) 第一次运行时，建立 env_id -> body_id -> 真实 USD prim path
    # ------------------------------------------------------------
    if not hasattr(base_env, "_viewport_real_body_prim_paths"):
        real_root_paths = [None for _ in range(num_envs)]

        try:
            matched = sim_utils.find_matching_prim_paths(base_env.cfg.turret_prim_path_expr)
            for p in matched:
                env_id = _extract_env_id(p)
                if env_id is not None and 0 <= env_id < num_envs:
                    real_root_paths[env_id] = p
        except Exception as exc:
            print(f"[eval real link sync] find root paths failed: {exc}")

        for env_id in range(num_envs):
            if real_root_paths[env_id] is None:
                expr = str(base_env.cfg.turret_prim_path_expr)
                real_root_paths[env_id] = expr.replace("env_.*", f"env_{env_id}")

        body_name_to_id = {name: i for i, name in enumerate(body_names)}
        real_body_paths: list[list[str | None]] = [
            [None for _ in range(num_bodies)] for _ in range(num_envs)
        ]

        for env_id in sync_env_ids:
            root_path = real_root_paths[env_id]

            # root_path 通常是：
            # /World/envs/env_i/Lite3/Lite3/TORSO
            # 资产根通常是：
            # /World/envs/env_i/Lite3/Lite3
            asset_root_path = root_path.rsplit("/", 1)[0]
            asset_root_prim = stage.GetPrimAtPath(asset_root_path)

            if not asset_root_prim.IsValid():
                print(f"[eval real link sync] invalid asset root prim: {asset_root_path}")
                continue

            try:
                for prim in Usd.PrimRange(asset_root_prim):
                    prim_name = prim.GetName()
                    if prim_name in body_name_to_id:
                        body_id = body_name_to_id[prim_name]
                        real_body_paths[env_id][body_id] = str(prim.GetPath())
            except Exception as exc:
                print(f"[eval real link sync] scan real body prims failed: {exc}")

            # 兜底：TORSO/root body
            if "TORSO" in body_name_to_id:
                torso_id = body_name_to_id["TORSO"]
                if real_body_paths[env_id][torso_id] is None:
                    real_body_paths[env_id][torso_id] = root_path

        base_env._viewport_real_body_prim_paths = real_body_paths
        base_env._viewport_real_xform_ops = {}

        debug_env_id = sync_env_ids[0] if len(sync_env_ids) > 0 else 0
        matched_pairs = []
        for body_id, name in enumerate(body_names):
            p = real_body_paths[debug_env_id][body_id]
            if p is not None:
                matched_pairs.append((body_id, name, p))

        print(
            "[eval real link sync] real body prim cache built | "
            f"sync_env_ids={sync_env_ids}, "
            f"num_bodies={num_bodies}, "
            f"matched_env{debug_env_id}={len(matched_pairs)}/{num_bodies}"
        )
        for body_id, name, path in matched_pairs[: min(24, len(matched_pairs))]:
            print(f"  body_id={body_id:02d} name={name} path={path}")

        if len(matched_pairs) < num_bodies:
            missing = [
                name for body_id, name in enumerate(body_names)
                if real_body_paths[debug_env_id][body_id] is None
            ]
            print(f"[eval real link sync] WARNING: missing real body prims: {missing}")

    # ------------------------------------------------------------
    # 2) 每帧把真实 articulation link 的 USD xform 写成 PhysX body pose
    # ------------------------------------------------------------
    for env_id in sync_env_ids:
        for body_id in range(num_bodies):
            prim_path = base_env._viewport_real_body_prim_paths[env_id][body_id]
            if prim_path is None:
                continue

            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            if prim_path not in base_env._viewport_real_xform_ops:
                # CPU PhysX 下允许标准化真实 articulation link 的 xform ops。
                # GPU Direct API 下这一步会有风险，所以前面已经强制 sim.device == cpu。
                try:
                    sim_utils.standardize_xform_ops(
                        prim,
                        translation=(0.0, 0.0, 0.0),
                        orientation=(1.0, 0.0, 0.0, 0.0),
                        scale=(1.0, 1.0, 1.0),
                    )
                except Exception as exc:
                    if not hasattr(base_env, "_warned_real_link_standardize_failed"):
                        base_env._warned_real_link_standardize_failed = True
                        print(f"[eval real link sync] standardize_xform_ops failed: {exc}")

                xf = UsdGeom.Xformable(prim)

                # 关键：
                # 直接写 world pose，所以断开父级 xform 继承。
                # 这是 eval viewport 专用，不建议训练时使用。
                try:
                    xf.SetResetXformStack(True)
                except Exception:
                    pass

                translate_op = None
                orient_op = None

                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                        translate_op = op
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
                        orient_op = op

                if translate_op is None:
                    translate_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "", False)
                if orient_op is None:
                    orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "", False)

                base_env._viewport_real_xform_ops[prim_path] = (translate_op, orient_op)

            translate_op, orient_op = base_env._viewport_real_xform_ops[prim_path]

            p = body_pos_w[env_id, body_id].detach().cpu().tolist()
            q = body_quat_w[env_id, body_id].detach().cpu().tolist()

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

    if not hasattr(base_env, "_viewport_real_sync_debug_count"):
        base_env._viewport_real_sync_debug_count = 0

    if base_env._viewport_real_sync_debug_count < 5:
        base_env._viewport_real_sync_debug_count += 1
        env0 = sync_env_ids[0] if len(sync_env_ids) > 0 else 0
        print(
            "[eval real link sync] synced real articulation links | "
            f"env={env0}, body_count={num_bodies}, "
            f"root_pos={body_pos_w[env0, 0].detach().cpu().tolist()}"
        )



def _pick_checkpoint_from_dir(run_dir: Path, which: str) -> Path:
    nn_dir = run_dir / "nn"
    search_dirs = [nn_dir, run_dir]

    all_pths: list[Path] = []
    for directory in search_dirs:
        if directory.is_dir():
            all_pths.extend(sorted(directory.glob("*.pth")))

    if len(all_pths) == 0:
        raise FileNotFoundError(
            f"No .pth checkpoints found under: {run_dir} "
            f"(looked in '{nn_dir}' and the directory root)"
        )

    best_candidates = [
        path
        for path in all_pths
        if (
            path.name == "turret_aim.pth"
            or (path.parent.name == "nn" and not path.name.startswith("last_"))
        )
    ]
    last_candidates = [path for path in all_pths if path.name.startswith("last_")]

    def newest(paths: list[Path]) -> Path:
        return max(paths, key=lambda path: path.stat().st_mtime)

    which = which.lower().strip()
    if which == "best":
        if len(best_candidates) > 0:
            return newest(best_candidates)
        non_last = [path for path in all_pths if not path.name.startswith("last_")]
        return newest(non_last) if len(non_last) > 0 else newest(all_pths)
    if which == "last":
        if len(last_candidates) == 0:
            raise FileNotFoundError(
                f"No 'last_*.pth' checkpoints found under: {run_dir}. "
                f"Available: {[path.name for path in all_pths]}"
            )
        return newest(last_candidates)
    if which == "latest":
        return newest(all_pths)

    raise ValueError("--which must be one of: best, last, latest")


def _resolve_checkpoint_path(path_str: str, which: str = "best") -> Path:
    ckpt_path = Path(path_str).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if ckpt_path.is_dir():
        picked = _pick_checkpoint_from_dir(ckpt_path, which=which)
        print(f"[eval] Using checkpoint ({which}): {picked}")
        ckpt_path = picked

    if ckpt_path.suffix != ".pth":
        raise ValueError(f"Expected a .pth checkpoint file, got: {ckpt_path}")

    return ckpt_path


def _install_mixed_training_logic(
    base_env: Any,
    case_probs: tuple[float, float, float, float],
    sd_fire_delay_s: float = 0.0,
):
    import torch
    import types

    prob = torch.tensor(case_probs, device=base_env.device, dtype=torch.float32)
    if torch.any(prob < 0.0):
        raise ValueError("case probabilities must be >= 0")
    prob_sum = float(prob.sum().item())
    if prob_sum <= 0.0:
        raise ValueError("case probabilities sum must be > 0")
    prob = prob / prob.sum()

    # case id mapping
    # 0: robot_static + target_static
    # 1: robot_static + target_dynamic
    # 2: robot_dynamic + target_static
    # 3: robot_dynamic + target_dynamic
    base_env._mix_case_prob = prob
    base_env._mix_case_id = torch.zeros((base_env.num_envs,), dtype=torch.long, device=base_env.device)
    base_env._mix_robot_dynamic = torch.zeros((base_env.num_envs,), dtype=torch.bool, device=base_env.device)
    base_env._mix_target_dynamic = torch.zeros((base_env.num_envs,), dtype=torch.bool, device=base_env.device)
    base_env._mix_case_counts = torch.zeros((4,), dtype=torch.long, device=base_env.device)
    base_env._mix_sd_fire_delay_steps = max(0, int(round(float(sd_fire_delay_s) / float(base_env.step_dt))))
    base_env._mix_fire_delay_left_steps = torch.zeros(
        (base_env.num_envs,),
        dtype=torch.long,
        device=base_env.device,
    )

    original_reset_idx = base_env._reset_idx
    original_pre_physics_step = base_env._pre_physics_step

    def _sample_case_ids(count: int) -> torch.Tensor:
        if count <= 0:
            return torch.zeros((0,), dtype=torch.long, device=base_env.device)
        return torch.multinomial(base_env._mix_case_prob, num_samples=count, replacement=True)

    def _apply_case_masks(env_ids_t: torch.Tensor):
        if env_ids_t.numel() == 0:
            return

        case_ids = base_env._mix_case_id[env_ids_t]
        robot_dynamic = case_ids >= 2
        target_dynamic = (case_ids % 2) == 1
        robot_static = ~robot_dynamic
        target_static = ~target_dynamic

        base_env._mix_robot_dynamic[env_ids_t] = robot_dynamic
        base_env._mix_target_dynamic[env_ids_t] = target_dynamic

        sd_mask = case_ids == 1
        base_env._mix_fire_delay_left_steps[env_ids_t] = torch.where(
            sd_mask,
            torch.full_like(case_ids, int(base_env._mix_sd_fire_delay_steps)),
            torch.zeros_like(case_ids),
        )

        if torch.any(robot_static):
            static_ids = env_ids_t[robot_static]
            base_env._aim_warmup_left_steps[static_ids] = 0
            base_env._locomotion_commands[static_ids] = 0.0
            base_env._locomotion_last_actions[static_ids] = 0.0
            base_env._locomotion_command_time_left[static_ids] = base_env.cfg.locomotion_command_resample_s

        if torch.any(target_static):
            static_target_ids = env_ids_t[target_static]
            base_env._target_vel_w[static_target_ids] = 0.0
            if hasattr(base_env, "_target_acc_w"):
                base_env._target_acc_w[static_target_ids] = 0.0

    def _resample_cases(env_ids_t: torch.Tensor):
        if env_ids_t.numel() == 0:
            return
        sampled = _sample_case_ids(int(env_ids_t.numel()))
        base_env._mix_case_id[env_ids_t] = sampled
        ones = torch.ones_like(sampled, dtype=torch.long)
        base_env._mix_case_counts.scatter_add_(0, sampled, ones)

    def patched_reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        result = original_reset_idx(env_ids_t)
        _resample_cases(env_ids_t)
        _apply_case_masks(env_ids_t)
        if hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None:
            self._rl_perception_bridge.reset_envs(env_ids_t)
        return result

    def patched_pre_physics_step(self, actions):
        robot_static_mask = ~self._mix_robot_dynamic
        if torch.any(robot_static_mask):
            self._aim_warmup_left_steps[robot_static_mask] = 0
            self._locomotion_commands[robot_static_mask] = 0.0
            self._locomotion_command_time_left[robot_static_mask] = self.cfg.locomotion_command_resample_s

        actions_for_step = actions
        sd_fire_delay_mask = self._mix_fire_delay_left_steps > 0
        if torch.any(sd_fire_delay_mask):
            actions_for_step = actions.clone()
            actions_for_step[sd_fire_delay_mask, 2] = -1.0

        original_pre_physics_step(actions_for_step)

        if torch.any(sd_fire_delay_mask):
            self._mix_fire_delay_left_steps[sd_fire_delay_mask] = torch.clamp(
                self._mix_fire_delay_left_steps[sd_fire_delay_mask] - 1,
                min=0,
            )

        target_static_mask = ~self._mix_target_dynamic
        if torch.any(target_static_mask):
            self._target_vel_w[target_static_mask] = 0.0
            if hasattr(self, "_target_acc_w"):
                self._target_acc_w[target_static_mask] = 0.0

    def get_mix_case_stats(self):
        counts = self._mix_case_counts.detach().cpu().to(dtype=torch.float32)
        total = float(counts.sum().item())
        if total <= 0.0:
            return {"ss": 0.0, "sd": 0.0, "ds": 0.0, "dd": 0.0, "total": 0.0}
        return {
            "ss": float((counts[0] / total).item()),
            "sd": float((counts[1] / total).item()),
            "ds": float((counts[2] / total).item()),
            "dd": float((counts[3] / total).item()),
            "total": total,
        }

    base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)
    base_env._pre_physics_step = types.MethodType(patched_pre_physics_step, base_env)
    base_env.get_mix_case_stats = types.MethodType(get_mix_case_stats, base_env)
    print(
        "[eval] mixed-case logic installed | "
        f"sd_fire_delay_s={float(sd_fire_delay_s):.3f} "
        f"steps={int(base_env._mix_sd_fire_delay_steps)}"
    )


def _case_to_probs(case_name: str) -> tuple[float, float, float, float]:
    case_name = case_name.lower().strip()
    if case_name == "mixed":
        return (0.25, 0.25, 0.25, 0.25)
    if case_name == "ss":
        return (1.0, 0.0, 0.0, 0.0)
    if case_name == "sd":
        return (0.0, 1.0, 0.0, 0.0)
    if case_name == "ds":
        return (0.0, 0.0, 1.0, 0.0)
    if case_name == "dd":
        return (0.0, 0.0, 0.0, 1.0)
    raise ValueError("--case must be one of: ss, sd, ds, dd, mixed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_envs", type=int, default=16)

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pth checkpoint or a run directory.",
    )
    parser.add_argument("--which", type=str, default="best", choices=["best", "last", "latest"])

    parser.add_argument(
        "--locomotion_policy",
        type=str,
        required=True,
        help="Path to exported locomotion TorchScript policy (policy.pt).",
    )

    parser.add_argument(
        "--vision_model",
        type=str,
        default="source/standalone/test4/vision_model/vision_runs/vision_runs_cnn_only/04-10-00-24/vision_best.pth",
        help="Path to vision model checkpoint used by rl_perception_bridge.",
    )
    parser.add_argument(
        "--velocity_model",
        type=str,
        default="",
        help="Path to velocity GRU checkpoint used by rl_perception_bridge. Required when bridge is enabled.",
    )
    parser.add_argument(
        "--enable_rl_perception_bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to install rl_perception_bridge into the environment.",
    )
    parser.add_argument(
        "--bridge_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable periodic debug print from rl_perception_bridge.",
    )
    parser.add_argument(
        "--bridge_use_gt_distance_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow bridge to fallback to GT distance when predicted/lidar distance is unavailable. Disabled by default for strict camera+raycast eval.",
    )
    parser.add_argument(
        "--bridge_use_gt_velocity_until_buffer_ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow bridge to fallback to GT velocity until GRU sequence buffer is ready. Disabled by default for strict velocity-module eval.",
    )
    parser.add_argument(
        "--bridge_velocity_sequence_stride",
        type=int,
        default=3,
        help="Env-step stride for velocity-GRU sequence samples. Default matches test_temp collection.",
    )

    parser.add_argument(
        "--case",
        type=str,
        default="mixed",
        choices=["ss", "sd", "ds", "dd", "mixed"],
        help="Evaluation scenario.",
    )

    parser.add_argument("--target_speed_min", type=float, default=0.6)
    parser.add_argument("--target_speed_max", type=float, default=5)
    parser.add_argument("--target_vz_min", type=float, default=-0.5)
    parser.add_argument("--target_vz_max", type=float, default=0.5)
    parser.add_argument(
        "--sd_fire_delay_s",
        type=float,
        default=2.0,
        help="In SD episodes, let the target move for this many seconds before fire is allowed.",
    )

    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--print_case_stats", action="store_true", default=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--draw_tracks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track_env_id", type=int, default=0)
    parser.add_argument("--track_max_points", type=int, default=600)
    parser.add_argument("--track_shot_history", type=int, default=25)
    parser.add_argument("--track_ballistic_stride", type=int, default=3)
    parser.add_argument("--robot_track_width", type=float, default=8.0)
    parser.add_argument("--target_track_width", type=float, default=8.0)
    parser.add_argument("--bullet_track_width", type=float, default=6.0)
    parser.add_argument("--pause_between_episodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pause_seconds", type=float, default=2.0)
    parser.add_argument(
        "--cpu_physx_view",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use CPU PhysX for viewport eval. "
            "Required if you want to sync real PhysX articulation link USD xforms "
            "without Direct GPU API setGlobalPose errors."
        ),
    )
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, which=args.which)
    print(f"[eval] checkpoint = {checkpoint_path}")

    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=bool(args.enable_cameras),
    )
    sim_app = app_launcher.app

    try:
        import torch
        import gymnasium as gym

        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner

        from turret_aim_env import TurretAimEnv, TurretAimEnvCfg
    except ModuleNotFoundError as exc:
        if "omni." in str(exc) or "omni" in str(exc):
            raise ModuleNotFoundError(
                "Isaac Sim modules were not found/initialized. Run this script via IsaacLab launcher."
            ) from exc
        raise

    env_cfg = TurretAimEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)

    # -----------------------------
    # 关键 1：继续关闭 Fabric
    # 避免 TiledCamera 初始化走 Fabric/Warp 路径
    # -----------------------------
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "use_fabric"):
        env_cfg.sim.use_fabric = False
        print("[eval] env_cfg.sim.use_fabric = False")

    # -----------------------------
    # 关键 2：可视化 eval 时使用 CPU PhysX
    # 这样后面同步真实 articulation link 的 USD xform
    # 不会触发 Direct GPU API 下 setGlobalPose 非法的问题。
    # -----------------------------
    if bool(args.cpu_physx_view):
        if int(args.num_envs) > 1:
            print(
                "[eval] WARNING: --cpu_physx_view is intended for viewport visualization. "
                "CPU PhysX with many envs will be slow. Recommend --num_envs 1."
            )

        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
            env_cfg.sim.device = "cpu"
        # 关键：eval 可视化时不要让 DirectRLEnv 自动频繁 render。
        # 否则 env.step() 内部可能先渲染一次“未同步真实 link”的画面，
        # 后面我们再同步 link + render，就会出现一闪一闪。
        #
        # 相机读取仍然会在 _sync_render_camera_before_read() 里手动 self.sim.render()，
        # 所以这里不会影响视觉模型拿图像。
        if hasattr(env_cfg.sim, "render_interval"):
            env_cfg.sim.render_interval = 10**9
            print("[eval] env_cfg.sim.render_interval = 1e9  # disable auto viewport render")

        # 兼容不同 IsaacLab / Isaac Sim 版本的字段；有就关，没有就跳过
        if hasattr(env_cfg.sim, "use_gpu_pipeline"):
            env_cfg.sim.use_gpu_pipeline = False
            print("[eval] env_cfg.sim.use_gpu_pipeline = False")

        if hasattr(env_cfg.sim, "physx"):
            if hasattr(env_cfg.sim.physx, "use_gpu"):
                env_cfg.sim.physx.use_gpu = False
                print("[eval] env_cfg.sim.physx.use_gpu = False")
            if hasattr(env_cfg.sim.physx, "use_gpu_pipeline"):
                env_cfg.sim.physx.use_gpu_pipeline = False
                print("[eval] env_cfg.sim.physx.use_gpu_pipeline = False")

    # 与训练时保持一致：底层环境统一支持 moving base + dynamic target，
    # 再通过 case mask 在每个 episode 约束成 ss/sd/ds/dd。
    env_cfg.training_stage = 2
    env_cfg.enable_locomotion_policy = True
    env_cfg.locomotion_policy_path = args.locomotion_policy
    env_cfg.enable_dynamic_target = True
    env_cfg.target_speed_range = (float(args.target_speed_min), float(args.target_speed_max))
    env_cfg.target_vertical_speed_range = (float(args.target_vz_min), float(args.target_vz_max))
    env_cfg.enable_air_drag = False
    env_cfg.target_acc_range = (0.0, 0.0)
    env_cfg.target_vertical_acc_range = (0.0, 0.0)
    env_cfg.debug_vis = not args.headless
    env_cfg.debug_diagnostics = True

    if args.enable_rl_perception_bridge:
        env_cfg.enable_vision_supervision = True
        env_cfg.vision_history_length = 1
        env_cfg.enable_dynamic_target_raycast_debug = True
        print("[eval] perception bridge sensors enabled | camera=True raycast=True")

    task_id = "Standalone-TurretAim-Mixed-Eval-v0"
    if task_id not in gym.registry:
        gym.register(id=task_id, entry_point=TurretAimEnv)

    env = gym.make(task_id, cfg=env_cfg, disable_env_checker=True)
    base_env: Any = env.unwrapped

    # CPU PhysX + 真实 articulation link 同步开关
    # 只建议用于 eval 可视化，不建议训练时使用。
    base_env._viewport_real_link_sync_enabled = bool(args.cpu_physx_view)

    # 只同步 track_env_id 对应的那只狗，避免 CPU eval 太慢
    base_env._viewport_sync_all_envs = False

    # 现在不再使用 visual clone，所以不要隐藏真实机器狗
    base_env._viewport_hide_real_robot = False

    if base_env._viewport_real_link_sync_enabled:
        print("[eval] real articulation link USD sync enabled.")
    else:
        print("[eval] real articulation link USD sync disabled. Use --cpu_physx_view to enable.")

    # ---------------------------------------------------------
    # one-shot bootstrap for delayed TiledCamera initialization
    # 必须在 RL perception bridge 安装之前完成
    # ---------------------------------------------------------
    if hasattr(base_env, "bootstrap_vision_camera"):
        base_env.bootstrap_vision_camera()
        print("[eval] bootstrap_vision_camera() done.")

    _install_mixed_training_logic(
        base_env,
        case_probs=_case_to_probs(args.case),
        sd_fire_delay_s=float(args.sd_fire_delay_s),
    )

    install_target_follow_motion(
        base_env,
        follow_speed=0.35,
        stop_speed_threshold=0.05,
        yaw_follow_gain=0.8,
        max_yaw_rate=0.35,
        blend=0.7,
        pos_gain=0.75,
        vel_gain=0.25,
        close_distance=1.5,
        enable_debug=False,
    )

    # ---------------------------------------------------------
    # Install RL perception bridge for eval
    # 这里只负责安装 bridge 和 reset hook，不要在这里 patch step
    # ---------------------------------------------------------
    if args.enable_rl_perception_bridge:
        if not args.vision_model.strip():
            raise ValueError(
                "--enable_rl_perception_bridge is True, but --vision_model is empty."
            )
        if not args.velocity_model.strip():
            raise ValueError(
                "--enable_rl_perception_bridge is True, but --velocity_model is empty. "
                "Pass the velocity GRU checkpoint trained with test_temp/train_gru_velocity.py."
            )

        bridge_device = "cpu" if bool(args.cpu_physx_view) else ("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"[eval] rl_perception_bridge device = {bridge_device}")

        bridge = install_rl_perception_bridge(
            base_env=base_env,
            vision_model_path=args.vision_model,
            velocity_model_path=args.velocity_model if args.velocity_model.strip() else None,
            device=bridge_device,
            use_gt_distance_fallback=args.bridge_use_gt_distance_fallback,
            use_gt_velocity_until_buffer_ready=args.bridge_use_gt_velocity_until_buffer_ready,
            velocity_sequence_stride=args.bridge_velocity_sequence_stride,
            debug=args.bridge_debug,
        )

        # reset 时同步清空 bridge 的时序缓存
        orig_reset_idx = base_env._reset_idx

        def patched_reset_idx(self, env_ids):
            out = orig_reset_idx(env_ids)
            if hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None:
                self._rl_perception_bridge.reset_envs(env_ids)
            return out

        base_env._reset_idx = types.MethodType(patched_reset_idx, base_env)

        print("[eval] rl_perception_bridge installed and reset hook installed.")

    # ---------------------------------------------------------
    # 一定要安装：Viewport / USD 同步 step hook
    # 注意：这个必须放在 enable_rl_perception_bridge 的 if 外面
    # 这样即使不开 bridge，也会同步物理层和渲染层
    # ---------------------------------------------------------
    orig_step_for_eval = base_env.step

    def patched_step_for_eval(self, actions):
        out = orig_step_for_eval(actions)

        use_fabric = True
        try:
            use_fabric = bool(getattr(self.cfg.sim, "use_fabric", True))
        except Exception:
            use_fabric = True

        if isinstance(out, tuple):
            out_len = len(out)
            if out_len == 5:
                obs_old, rew, terminated, truncated, extras = out
            elif out_len == 4:
                obs_old, rew, dones, extras = out
            else:
                raise RuntimeError(
                    f"[eval] Unexpected env.step() return format: type={type(out)}, len={out_len}"
                )
        else:
            raise RuntimeError(
                f"[eval] Unexpected env.step() return format: type={type(out)}"
            )

        # 1) env.step() 后，先同步真实 articulation link
        did_real_link_sync = False
        if (
            (not use_fabric)
            and bool(getattr(self, "_viewport_real_link_sync_enabled", False))
        ):
            _sync_articulation_to_usd_for_viewport(self)
            did_real_link_sync = True

        # 2) 再跑 perception bridge
        # bridge 内部可能会为 TiledCamera 调用 self.sim.render()，
        # 但不要把它当成主 Viewport 的最终刷新。
        if hasattr(self, "_rl_perception_bridge") and self._rl_perception_bridge is not None:
            self._rl_perception_bridge.step()
            obs = self._get_observations()
        else:
            obs = obs_old

        # 3) 最后只要做了真实 link 同步，就强制刷新主 Viewport 一次
        if did_real_link_sync:
            try:
                if hasattr(self, "sim"):
                    self.sim.render()
            except Exception as exc:
                if getattr(self.cfg, "debug_diagnostics", False):
                    print(f"[eval] final viewport render failed: {exc}")

        if out_len == 5:
            return obs, rew, terminated, truncated, extras
        else:
            return obs, rew, dones, extras

    base_env.step = types.MethodType(patched_step_for_eval, base_env)
    print("[eval] viewport USD sync hooked into step().")

    if args.print_case_stats:
        print(f"[eval] case = {args.case}")

    if env_cfg.debug_vis and bool(args.draw_tracks):
        _install_runtime_track_draw(
            base_env=base_env,
            track_env_id=int(args.track_env_id),
            max_points=int(args.track_max_points),
            shot_history=int(args.track_shot_history),
            ballistic_draw_stride=int(args.track_ballistic_stride),
            robot_color=(0.1, 0.9, 0.1, 1.0),
            target_color=(1.0, 0.2, 0.2, 1.0),
            bullet_color=(1.0, 0.9, 0.1, 1.0),
            robot_width=float(args.robot_track_width),
            target_width=float(args.target_track_width),
            bullet_width=float(args.bullet_track_width),
            pause_between_episodes=bool(args.pause_between_episodes),
            pause_seconds=float(args.pause_seconds),
        )

    if hasattr(base_env, "reset_eval_metrics"):
        base_env.reset_eval_metrics()

    rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = RlGamesVecEnvWrapper(
        base_env,
        rl_device,
        clip_obs=10.0,
        clip_actions=1.0,
        obs_groups={"obs": ["policy", "critic"], "states": ["critic"]},
        concate_obs_group=True,
    )

    experiment_name = f"turret_aim_eval_{datetime.now().strftime('%m-%d-%H-%M-%S')}"

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
    )

    config = {
        "params": {
            "seed": 61,
            "algo": {
                "name": "a2c_continuous",
            },
            "model": {
                "name": "continuous_a2c_tanh",
            },
            "network": {
                "name": "asym_central_value_gru",
                "actor_target_dim": 9,
                "actor_robot_dim": 19,
                "critic_target_dim": 12,
                "critic_robot_dim": 19,
                "branch_hidden_dim": 64,
                "actor_rnn_hidden_dim": 128,
                "critic_rnn_hidden_dim": 128,
                "head_hidden_dim": 64,
                "fixed_sigma": True,
            },
            "load_checkpoint": True,
            "load_path": str(checkpoint_path),
            "config": {
                "name": "turret_aim",
                "full_experiment_name": experiment_name,
                "env_name": "rlgpu",
                "has_central_value": False,
                "num_actors": env.unwrapped.num_envs,
                "env_config": {},
                "device": rl_device,
                "device_name": rl_device,
                "ppo": True,
                "reward_shaper": {
                    "scale_value": 1.0,
                    "shift_value": 0.0,
                },
                "bounds_loss_coef": 0.0001,
                "horizon_length": 32,
                "minibatch_size": 1024,
                "mini_epochs": 2,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 5e-5,
                "clip_value": True,
                "entropy_coef": 0.0,
                "critic_coef": 2.0,
                "e_clip": 0.1,
                "max_epochs": 1,
                "save_best_after": 0,
                "save_frequency": 0,
                "score_to_win": 100000000,
                "lr_schedule": "adaptive",
                "kl_threshold": 0.01,
                "normalize_advantage": True,
                "grad_norm": 0.5,
                "truncate_grads": True,
                "normalize_input": True,
                "normalize_value": True,
                "seq_length": 8,
                "zero_rnn_on_done": True,
                "player": {
                    "deterministic": bool(args.deterministic),
                    "games_num": int(args.episodes),
                    "print_stats": True,
                },
            },
        }
    }

    runner = Runner(IsaacAlgoObserver())
    runner.load(config)

    run_args: dict[str, Any] = {
        "train": False,
        "play": True,
        "checkpoint": str(checkpoint_path),
    }

    print("[eval] starting play mode ...")
    runner.run(run_args)

    if hasattr(base_env, "get_eval_metrics"):
        metrics = base_env.get_eval_metrics()
        print("\n[eval] ===== Aggregated Metrics =====")
        print(f"[eval] episodes={int(metrics['episodes'])}")
        print(f"[eval] fires={int(metrics['fires'])}")
        print(f"[eval] hits={int(metrics['hits'])}")
        print(f"[eval] hit_rate={metrics['hit_rate']:.4f}")
        print(f"[eval] avg_fires_per_episode={metrics['avg_fires_per_episode']:.4f}")
        print(f"[eval] timeouts={int(metrics['timeouts'])} timeout_rate={metrics['timeout_rate']:.4f}")
        print(f"[eval] fallen={int(metrics['fallen'])} fallen_rate={metrics['fallen_rate']:.4f}")
        print(f"[eval] invalid={int(metrics['invalid'])} invalid_rate={metrics['invalid_rate']:.4f}")
        print(
            f"[eval] target_spawn_count={int(metrics['target_spawn_count'])} "
            f"target_spawn_mean=({metrics['target_spawn_mean_x']:.3f}, {metrics['target_spawn_mean_y']:.3f}, {metrics['target_spawn_mean_z']:.3f})"
        )
        print(
            f"[eval] fire_min_dist_count={int(metrics['fire_min_dist_count'])} "
            f"fire_min_dist_mean={metrics['fire_min_dist_mean']:.4f} "
            f"fire_min_dist_min={metrics['fire_min_dist_min']:.4f} "
            f"fire_min_dist_max={metrics['fire_min_dist_max']:.4f}"
        )

        target_samples = metrics.get("target_spawn_samples", [])
        if isinstance(target_samples, list) and len(target_samples) > 0:
            print("[eval] target_spawn_samples(first up to 50):")
            for idx, pos in enumerate(target_samples):
                if isinstance(pos, list) and len(pos) >= 3:
                    print(f"[eval]   #{idx:02d}: ({float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f})")

        dist_samples = metrics.get("fire_min_dist_samples", [])
        if isinstance(dist_samples, list) and len(dist_samples) > 0:
            print("[eval] fire_min_dist_samples(first up to 50):")
            for idx, dist in enumerate(dist_samples):
                print(f"[eval]   #{idx:02d}: {float(dist):.4f}")

        episode_fire_counts = metrics.get("episode_fire_counts", [])
        episode_hits = metrics.get("episode_hits", [])
        if (
            isinstance(episode_fire_counts, list)
            and isinstance(episode_hits, list)
            and len(episode_fire_counts) > 0
            and len(episode_fire_counts) == len(episode_hits)
        ):
            print("[eval] per-episode fire/hit:")
            for idx, (fire_count, hit_flag) in enumerate(zip(episode_fire_counts, episode_hits)):
                hit_text = "HIT" if int(hit_flag) > 0 else "MISS"
                print(f"[eval]   ep#{idx + 1:02d}: fires={int(fire_count)} result={hit_text}")

    sim_app.close()


if __name__ == "__main__":
    main()

"""Read-only control/GT snapshots for SS evaluation, never policy inputs.

Angles use the environment's yaw-only base frame: yaw=atan2(y,x),
pitch=atan2(-z,hypot(x,y)). GT directions and prediction comparisons use
matching origins. No render, raycast, tracker or ballistic update runs here.
"""
from __future__ import annotations

import math

import torch


SCALAR_FIELDS = (
    "gt_valid", "pred_valid", "raycast_valid", "raycast_distance_m",
    "pred_yaw_base_rad", "pred_pitch_base_rad", "pred_distance_m", "pred_speed_mps",
    "gt_yaw_base_rad", "gt_pitch_base_rad", "gt_distance_m",
    "pred_gt_yaw_error_rad", "pred_gt_pitch_error_rad", "pred_gt_position_error_m",
    "muzzle_yaw_base_rad", "muzzle_pitch_base_rad",
    "gt_muzzle_yaw_base_rad", "gt_muzzle_pitch_base_rad",
    "gt_muzzle_yaw_error_rad", "gt_muzzle_pitch_error_rad",
    "pred_muzzle_yaw_base_rad", "pred_muzzle_pitch_base_rad",
    "pred_muzzle_yaw_error_rad", "pred_muzzle_pitch_error_rad",
    "gt_projection_valid", "gt_center_x_px", "gt_center_y_px", "gt_yaw_cam_rad", "gt_pitch_cam_rad",
    "camera_fx", "camera_fy", "camera_cx", "camera_cy",
    "yaw_joint_pos_rad", "pitch_joint_pos_rad",
    "yaw_joint_vel_rps", "pitch_joint_vel_rps",
    "yaw_command_rad", "pitch_command_rad",
    "yaw_command_error_rad", "pitch_command_error_rad",
    "yaw_lower_limit_rad", "yaw_upper_limit_rad", "pitch_lower_limit_rad", "pitch_upper_limit_rad",
    "yaw_at_limit", "pitch_at_limit",
)
VECTOR_FIELDS = {
    "pred_pos_base": "xyz", "gt_pos_base": "xyz", "gt_pos_w": "xyz",
    "root_pos_w": "xyz", "root_quat_w": "wxyz",
    "muzzle_pos_w": "xyz", "muzzle_forward_w": "xyz",
    "camera_pos_w": "xyz", "camera_quat_w": "wxyz",
}
SNAPSHOT_FIELDS = SCALAR_FIELDS + tuple(
    f"{name}_{axis}" for name, axes in VECTOR_FIELDS.items() for axis in axes
)


def _angles(v):
    return torch.atan2(v[:, 1], v[:, 0]), torch.atan2(-v[:, 2], v[:, :2].norm(dim=-1))


def _wrap(v):
    return torch.atan2(torch.sin(v), torch.cos(v))


def _finite_where(valid, value):
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def control_snapshot(env, target_ids):
    """Snapshot one selected ID per env; hit targets remain valid until reset.

    GPU values are packed into one CPU transfer. Non-finite/invalid derived
    values become empty CSV cells; stale raw predictions retain pred_valid=False.
    """
    pool, data = env._static_salvo, env.turret.data
    ids = torch.as_tensor(target_ids, device=pool.positions.device, dtype=torch.long)
    batch_ids = torch.arange(env.num_envs, device=ids.device)
    gt_valid = pool.ready & (ids >= 0) & (ids < pool.positions.shape[1])
    gt_w = pool.positions[batch_ids, ids.clamp(0, pool.positions.shape[1]-1)]
    gt_valid = gt_valid & torch.isfinite(gt_w).all(dim=-1)
    root_w, root_q = data.root_pos_w, data.root_quat_w
    gt_base = env._rotate_world_to_yaw_only_local(root_q, gt_w-root_w)
    muzzle_w, _, forward_w = env._compute_turret_frame()
    forward_base = env._rotate_world_to_yaw_only_local(root_q, forward_w)
    muzzle_yaw, muzzle_pitch = _angles(forward_base)
    muzzle_offset_base = env._rotate_world_to_yaw_only_local(root_q, muzzle_w-root_w)
    gt_muzzle = env._rotate_world_to_yaw_only_local(root_q, gt_w-muzzle_w)
    gt_yaw, gt_pitch = _angles(gt_base)
    gt_myaw, gt_mpitch = _angles(gt_muzzle)
    pred_base = env._rl_pred_pos_base
    pred_myaw, pred_mpitch = _angles(pred_base-muzzle_offset_base)
    pred_valid = (env._rl_pred_ready & env._rl_bridge_training_ready
                  & torch.isfinite(pred_base).all(dim=-1)
                  & torch.isfinite(env._rl_pred_yaw_base) & torch.isfinite(env._rl_pred_pitch_base)
                  & torch.isfinite(env._rl_pred_distance) & (env._rl_pred_distance > 0)
                  & torch.isfinite(env._rl_pred_vel_base).all(dim=-1))
    comparable = gt_valid & pred_valid

    camera_w, camera_q = env._get_runtime_camera_pose_w()
    gt_cam = env._rotate_world_to_local(camera_q, gt_w-camera_w)
    # Read initialized intrinsics without triggering the sensor's lazy image update.
    camera_data = getattr(env._vision_camera, "_data", None)
    if camera_data is None:
        camera_data = env._vision_camera.data
    intrinsics = camera_data.intrinsic_matrices
    fx, fy, cx, cy = (intrinsics[:, 0, 0], intrinsics[:, 1, 1],
                      intrinsics[:, 0, 2], intrinsics[:, 1, 2])
    projection_valid = (gt_valid & torch.isfinite(gt_cam).all(dim=-1) & (gt_cam[:, 2] > 1e-6)
                        & torch.isfinite(intrinsics).all(dim=-1).all(dim=-1) & (fx > 0) & (fy > 0))
    depth = gt_cam[:, 2].clamp_min(1e-6)
    u, v = cx+fx*gt_cam[:, 0]/depth, cy+fy*gt_cam[:, 1]/depth

    joint_ids = [int(env._yaw_dof_ids[0]), int(env._pitch_dof_ids[0])]
    pos, vel = data.joint_pos[:, joint_ids], data.joint_vel[:, joint_ids]
    limits, command = data.joint_pos_limits[:, joint_ids, :], env._turret_pos_targets
    at_limit = ((pos-limits[..., 0]).abs() <= 1e-3) | ((pos-limits[..., 1]).abs() <= 1e-3)
    ray_valid = getattr(env, "_raycast_hit_valid", torch.zeros_like(gt_valid)) & gt_valid
    ray_distance = getattr(env, "_raycast_hit_distance", torch.full_like(depth, float("nan")))
    ray_valid = ray_valid & torch.isfinite(ray_distance) & (ray_distance > 0)
    values = {
        "gt_valid": gt_valid, "pred_valid": pred_valid, "raycast_valid": ray_valid,
        "raycast_distance_m": _finite_where(ray_valid, ray_distance),
        "pred_yaw_base_rad": env._rl_pred_yaw_base, "pred_pitch_base_rad": env._rl_pred_pitch_base,
        "pred_distance_m": env._rl_pred_distance, "pred_speed_mps": env._rl_pred_vel_base.norm(dim=-1),
        "gt_yaw_base_rad": _finite_where(gt_valid, gt_yaw),
        "gt_pitch_base_rad": _finite_where(gt_valid, gt_pitch),
        "gt_distance_m": _finite_where(gt_valid, gt_base.norm(dim=-1)),
        "pred_gt_yaw_error_rad": _finite_where(comparable, _wrap(env._rl_pred_yaw_base-gt_yaw)),
        "pred_gt_pitch_error_rad": _finite_where(comparable, env._rl_pred_pitch_base-gt_pitch),
        "pred_gt_position_error_m": _finite_where(comparable, (pred_base-gt_base).norm(dim=-1)),
        "muzzle_yaw_base_rad": muzzle_yaw, "muzzle_pitch_base_rad": muzzle_pitch,
        "gt_muzzle_yaw_base_rad": _finite_where(gt_valid, gt_myaw),
        "gt_muzzle_pitch_base_rad": _finite_where(gt_valid, gt_mpitch),
        "gt_muzzle_yaw_error_rad": _finite_where(gt_valid, _wrap(gt_myaw-muzzle_yaw)),
        "gt_muzzle_pitch_error_rad": _finite_where(gt_valid, gt_mpitch-muzzle_pitch),
        "pred_muzzle_yaw_base_rad": _finite_where(pred_valid, pred_myaw),
        "pred_muzzle_pitch_base_rad": _finite_where(pred_valid, pred_mpitch),
        "pred_muzzle_yaw_error_rad": _finite_where(pred_valid, _wrap(pred_myaw-muzzle_yaw)),
        "pred_muzzle_pitch_error_rad": _finite_where(pred_valid, pred_mpitch-muzzle_pitch),
        "gt_projection_valid": projection_valid,
        "gt_center_x_px": _finite_where(projection_valid, u),
        "gt_center_y_px": _finite_where(projection_valid, v),
        "gt_yaw_cam_rad": _finite_where(projection_valid, torch.atan2(gt_cam[:, 0], gt_cam[:, 2])),
        "gt_pitch_cam_rad": _finite_where(projection_valid, torch.atan2(-gt_cam[:, 1], gt_cam[:, (0, 2)].norm(dim=-1))),
        "camera_fx": fx, "camera_fy": fy, "camera_cx": cx, "camera_cy": cy,
        "yaw_joint_pos_rad": pos[:, 0], "pitch_joint_pos_rad": pos[:, 1],
        "yaw_joint_vel_rps": vel[:, 0], "pitch_joint_vel_rps": vel[:, 1],
        "yaw_command_rad": command[:, 0], "pitch_command_rad": command[:, 1],
        "yaw_command_error_rad": command[:, 0]-pos[:, 0],
        "pitch_command_error_rad": command[:, 1]-pos[:, 1],
        "yaw_lower_limit_rad": limits[:, 0, 0], "yaw_upper_limit_rad": limits[:, 0, 1],
        "pitch_lower_limit_rad": limits[:, 1, 0], "pitch_upper_limit_rad": limits[:, 1, 1],
        "yaw_at_limit": at_limit[:, 0], "pitch_at_limit": at_limit[:, 1],
    }
    vectors = dict(pred_pos_base=pred_base, gt_pos_base=gt_base, gt_pos_w=gt_w,
                   root_pos_w=root_w, root_quat_w=root_q, muzzle_pos_w=muzzle_w,
                   muzzle_forward_w=forward_w, camera_pos_w=camera_w, camera_quat_w=camera_q)
    for name, axes in VECTOR_FIELDS.items():
        for j, axis in enumerate(axes):
            value = vectors[name][:, j]
            values[f"{name}_{axis}"] = _finite_where(gt_valid, value) if name.startswith("gt_") else value
    # Masks remain booleans in CSV, although a common float buffer is used for transfer.
    boolean_fields = {"gt_valid", "pred_valid", "raycast_valid", "gt_projection_valid", "yaw_at_limit", "pitch_at_limit"}
    packed = torch.stack([values[k].to(torch.float64) for k in SNAPSHOT_FIELDS], dim=-1).detach().cpu().tolist()
    rows = []
    for row in packed:
        rows.append({k: (bool(v) if k in boolean_fields else v if math.isfinite(v) else "")
                     for k, v in zip(SNAPSHOT_FIELDS, row)})
    return rows

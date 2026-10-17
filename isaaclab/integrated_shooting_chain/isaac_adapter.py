"""Small IsaacLab adapter for writing a chain prediction into test7_2_weight."""

from __future__ import annotations

from typing import Any


def write_prediction_to_turret_env(base_env: Any, command: Any, *, env_id: int = 0) -> None:
    """Write one ``DecisionCommand`` prediction into ``TurretAimEnv``.

    Call this before the next observation/policy evaluation.  The existing
    environment remains responsible for applying the action, cooldown and
    ballistic hit test.  This helper only fills the environment's perception
    bridge buffers and readiness flags.
    """
    import torch

    index = int(env_id)
    if index < 0 or index >= int(base_env.num_envs):
        raise IndexError(f"env_id {index} outside [0, {int(base_env.num_envs)})")
    if not hasattr(base_env, "set_rl_predicted_target_state"):
        raise TypeError("base_env does not expose TurretAimEnv.set_rl_predicted_target_state")

    device = base_env.device
    pos = torch.zeros((base_env.num_envs, 3), dtype=torch.float32, device=device)
    vel = torch.zeros((base_env.num_envs, 3), dtype=torch.float32, device=device)
    distance = torch.zeros((base_env.num_envs,), dtype=torch.float32, device=device)
    yaw = torch.zeros((base_env.num_envs,), dtype=torch.float32, device=device)
    pitch = torch.zeros((base_env.num_envs,), dtype=torch.float32, device=device)
    env_ids = torch.as_tensor([index], dtype=torch.long, device=device)
    ready = bool(command.prediction_ready)
    if ready:
        pos[index] = torch.as_tensor(command.target_position_base_m, dtype=torch.float32, device=device)
        vel[index] = torch.as_tensor(command.target_velocity_base_mps, dtype=torch.float32, device=device)
        distance[index] = torch.linalg.vector_norm(pos[index])
        yaw[index] = torch.atan2(pos[index, 1], pos[index, 0])
        pitch[index] = torch.atan2(-pos[index, 2], torch.linalg.vector_norm(pos[index, :2]).clamp_min(1e-6))
        base_env.set_rl_predicted_target_state(
            pos, vel, distance, yaw, pitch, env_ids=env_ids
        )
    elif hasattr(base_env, "_rl_pred_ready"):
        base_env._rl_pred_ready[index] = False
    if hasattr(base_env, "_rl_bridge_training_ready"):
        base_env._rl_bridge_training_ready[index] = ready
    if hasattr(base_env, "_rl_bridge_velocity_ready"):
        base_env._rl_bridge_velocity_ready[index] = ready

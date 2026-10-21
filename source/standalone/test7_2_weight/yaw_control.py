"""Mechanical yaw goals and versioned Actor observation semantics, without Isaac."""
from __future__ import annotations

import copy
import math
from typing import NamedTuple

import torch


YAW_ERROR_MODES = ("joint_reachable", "wrapped_los")
CONTRACT_KEY = "aim_observation_contract"


class YawGoal(NamedTuple):
    angle: torch.Tensor
    error: torch.Tensor
    exact_reachable: torch.Tensor


def reachable_yaw_goal(yaw, forward, axis, target_from_pivot, muzzle_from_pivot, limits):
    """Solve yaw around the actual joint axis, holding pitch fixed.

    Vectors share a world frame; ``yaw`` and ``limits`` are joint coordinates.
    Includes lateral muzzle offset, arbitrary mounting rotation and axis sign.
    Select the nearest equivalent joint angle inside the limits. If the exact
    heading is outside the range, return its best boundary and mark it inexact.
    A direction along the yaw axis has no yaw solution and holds the joint.
    The returned error is a linear joint displacement and MUST NOT be wrapped.
    """
    eps = 1e-7
    tolerance = 1e-5  # USD limits commonly round pi to 3.14159.
    axis_norm = axis.norm(dim=-1, keepdim=True)
    unit_axis = axis / axis_norm.clamp_min(eps)

    def project(vector):
        return vector - (vector * unit_axis).sum(-1, keepdim=True) * unit_axis

    f, t, r = map(project, (forward, target_from_pivot, muzzle_from_pivot))
    denominator = f.norm(dim=-1) * t.norm(dim=-1)
    lateral = (unit_axis * torch.cross(f, r, dim=-1)).sum(-1)
    sine_offset = lateral / denominator.clamp_min(eps)
    phase = torch.atan2((unit_axis * torch.cross(f, t, dim=-1)).sum(-1), (f * t).sum(-1))
    delta = phase - torch.asin(sine_offset.clamp(-1., 1.))
    unbounded_goal = yaw + delta
    lower, upper = limits.unbind(-1)
    period = 2. * math.pi
    k_low = torch.ceil((lower - unbounded_goal - tolerance) / period)
    k_high = torch.floor((upper - unbounded_goal + tolerance) / period)
    k_nearest = torch.round((yaw - unbounded_goal) / period)
    k = torch.maximum(torch.minimum(k_nearest, k_high), k_low)
    equivalent = (unbounded_goal + period * k).clamp(min=lower, max=upper)

    # If no equivalent fits, compare angular residuals at the two boundaries.
    def residual(bound):
        diff = bound - unbounded_goal
        return torch.atan2(torch.sin(diff), torch.cos(diff)).abs()

    low_error, high_error = residual(lower), residual(upper)
    prefer_low = (low_error < high_error - tolerance) | (
        (low_error - high_error).abs().le(tolerance) & ((yaw - lower).abs() <= (yaw - upper).abs()))
    boundary = torch.where(prefer_low, lower, upper)
    geometry_valid = (axis_norm.squeeze(-1) > eps) & (denominator > eps) & sine_offset.abs().le(1.)
    # Pick the forward-facing root of the line-of-sight equation.
    geometry_valid &= (torch.sqrt((denominator.square() - lateral.square()).clamp_min(0.))
                       - (f * r).sum(-1)) > eps
    geometry_valid &= (torch.isfinite(yaw) & torch.isfinite(delta) & torch.isfinite(limits).all(-1)
                       & (lower < upper))
    exact = geometry_valid & (k_low <= k_high)
    goal = torch.where(exact, equivalent, boundary)
    goal = torch.where(geometry_valid, goal, yaw.clamp(min=lower, max=upper))
    return YawGoal(goal, goal - yaw, exact)


def observation_contract(mode, scale=.05, clip=5.):
    if mode not in YAW_ERROR_MODES:
        raise ValueError(f"Unknown yaw_error_mode: {mode}")
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(clip) or clip < 1:
        raise ValueError("Yaw observation scale must be positive and clip must be >= 1")
    return dict(version="joint_reachable_yaw_v2" if mode == "joint_reachable" else "wrapped_los_v1",
                yaw_error_mode=mode, actor_dim=28, critic_dim=31,
                actor_yaw_index=7, normalization_scale_rad=scale, normalization_clip=clip,
                pitch_error="existing_los_or_gravity_fallback", critic_yaw_error="wrapped_los",
                yaw_reference="predicted_target_from_usd_yaw_pivot" if mode == "joint_reachable" else "legacy_base_los")


def checkpoint_contract(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    contract = state.get(CONTRACT_KEY)
    if contract is None:
        contract = observation_contract("wrapped_los")
        contract["inferred_from_untagged_checkpoint"] = True
    return contract


def contracts_match(source, expected):
    return all(source.get(key) == expected.get(key) for key in
               ("version", "yaw_error_mode", "normalization_scale_rad", "normalization_clip"))


def make_observation_contract_agent(agent_class, contract):
    """Tag every RL-Games saved checkpoint; allow an explicit logged warm start."""
    class ObservationContractAgent(agent_class):
        def get_full_state_weights(self):
            state = super().get_full_state_weights()
            state[CONTRACT_KEY] = copy.deepcopy(contract)
            return state

        def set_full_state_weights(self, weights, set_epoch=True):
            previous = weights.get(CONTRACT_KEY, observation_contract("wrapped_los"))
            if not contracts_match(previous, contract):
                print(f"[aim-contract] Warm-start observation change: {previous['version']} -> "
                      f"{contract['version']}; resumed weights require adaptation to the new yaw input.", flush=True)
            return super().set_full_state_weights(weights, set_epoch=set_epoch)

    return ObservationContractAgent

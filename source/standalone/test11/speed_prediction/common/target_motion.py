"""Continuous specular motion inside fixed planes and a spherical range shell.

Planes use unit inward normals: n dot position + offset >= 0. Positions are
projected only when initializing a trajectory; time integration never teleports
an out-of-bounds target back into the domain.
"""
from __future__ import annotations

import math
import torch


def _check_domain(position, normals, offsets, center, radius_min, radius_max, tolerance):
    signed = (normals * position[:, None, :]).sum(-1) + offsets
    radius = torch.linalg.vector_norm(position - center, dim=-1)
    invalid = ((signed < -tolerance).any(-1)
               | (radius < radius_min - tolerance) | (radius > radius_max + tolerance)
               | ~torch.isfinite(position).all(-1))
    if bool(invalid.any()):
        ids = invalid.nonzero().flatten().tolist()
        raise RuntimeError(f"Target outside fixed motion domain: env_ids={ids}")


def initialize_in_domain(position, normals, offsets, center, radius_min, radius_max):
    """Place reset-time samples in the domain, before any trajectory is recorded."""
    dtype = position.dtype
    p, n, b, c = [v.to(torch.float64) for v in (position, normals, offsets, center)]
    for _ in range(64):
        for plane in range(n.shape[1]):
            signed = (n[:, plane] * p).sum(-1) + b[:, plane]
            p = p + torch.clamp(-signed, min=0.)[:, None] * n[:, plane]
        delta = p - c
        radius = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-12)
        p = c + delta * (radius.clamp(radius_min, radius_max) / radius)[:, None]
        signed = (n * p[:, None, :]).sum(-1) + b
        if bool((signed >= -1e-9).all()):
            break
    _check_domain(p, n, b, c, radius_min, radius_max, 1e-7)
    return p.to(dtype)


def advance_reflecting_targets(position, velocity, dt_s, normals, offsets, center,
                               radius_min, radius_max, max_collisions=64):
    """Move to each earliest collision, reflect, then consume remaining time.

    The output velocity is instantaneous at the end of the step. Across a
    collision it need not equal the average velocity between saved positions.
    Invalid input or exhausted collision iterations fails explicitly rather
    than introducing an unlabelled position correction.
    """
    if not math.isfinite(dt_s) or dt_s <= 0 or not 0 < radius_min < radius_max:
        raise ValueError("Invalid target motion time/range configuration")
    dtype = position.dtype
    p, v, n, b, c = [x.to(torch.float64) for x in (position, velocity, normals, offsets, center)]
    _check_domain(p, n, b, c, radius_min, radius_max, 3e-5)
    if not bool(torch.isfinite(v).all()):
        raise ValueError("Target velocity must be finite")
    remaining = torch.full_like(p[:, 0], dt_s)
    counts = torch.zeros(len(p), dtype=torch.long, device=p.device)
    inf = torch.full_like(remaining, float("inf"))
    for _ in range(max_collisions):
        active = remaining > 1e-12
        if not bool(active.any()):
            break
        signed = (n * p[:, None, :]).sum(-1) + b
        rate = (n * v[:, None, :]).sum(-1)
        plane_times = torch.where(rate < -1e-12,
                                  signed.clamp_min(0.) / (-rate).clamp_min(1e-30),
                                  torch.full_like(rate, float("inf")))
        q = p - c
        speed_sq = (v * v).sum(-1)
        radial_rate = (q * v).sum(-1)
        q_sq = (q * q).sum(-1)
        denom = speed_sq.clamp_min(1e-30)
        outer_disc = radial_rate.square() - speed_sq * (q_sq - radius_max**2)
        outer_time = (-radial_rate + outer_disc.clamp_min(0.).sqrt()) / denom
        # Float32 storage may leave a center a few micrometres outside a
        # sphere. Reflect at t=0 without correcting its position; a slow normal
        # velocity can otherwise turn that tiny distance into a large negative t.
        outer_time = torch.where((outer_time < 0.) & (radial_rate > 0.)
                                 & (q_sq <= (radius_max+3e-5)**2),
                                 torch.zeros_like(outer_time), outer_time)
        outer_time = torch.where((speed_sq > 1e-24) & (outer_disc >= 0.)
                                 & (outer_time >= 0.), outer_time, inf)
        inner_disc = radial_rate.square() - speed_sq * (q_sq - radius_min**2)
        inner_time = (-radial_rate - inner_disc.clamp_min(0.).sqrt()) / denom
        inner_time = torch.where((inner_time < 0.) & (radial_rate < 0.)
                                 & (q_sq >= (radius_min-3e-5)**2),
                                 torch.zeros_like(inner_time), inner_time)
        inner_time = torch.where((speed_sq > 1e-24) & (inner_disc > 1e-20)
                                 & (radial_rate < 0.) & (inner_time >= 0.), inner_time, inf)
        times = torch.cat((plane_times, outer_time[:, None], inner_time[:, None]), dim=1)
        event_time, event = times.min(dim=1)
        hit = active & (event_time <= remaining)
        travel = torch.where(active, torch.minimum(event_time, remaining), torch.zeros_like(remaining))
        p = p + v * travel[:, None]
        remaining = (remaining - travel).clamp_min(0.)
        if not bool(hit.any()):
            continue
        plane_index = event.clamp(max=n.shape[1]-1)
        normal = n[torch.arange(len(p), device=p.device), plane_index]
        radial_normal = p - c
        radial_normal = radial_normal / torch.linalg.vector_norm(radial_normal, dim=-1, keepdim=True).clamp_min(1e-30)
        normal = torch.where((event >= n.shape[1])[:, None], radial_normal, normal)
        reflected = v - 2. * (v * normal).sum(-1)[:, None] * normal
        v = torch.where(hit[:, None], reflected, v)
        counts += hit.long()
    if bool((remaining > 1e-12).any()):
        raise RuntimeError("Target collision iteration limit reached; no position correction applied")
    _check_domain(p, n, b, c, radius_min, radius_max, 3e-5)
    return p.to(dtype), v.to(velocity.dtype), counts

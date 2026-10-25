"""Joint-limit-aware reset sampling for turret recovery training, without Isaac."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
import numpy as np


@dataclass(frozen=True)
class TurretInitConfig:
    mode: str = "zero"
    probabilities: tuple[float, float, float] = (.4, .4, .2)  # zero, general, near-limit
    general_yaw_deg: float = 90.
    general_pitch_deg: float = 20.
    limit_margin_deg: float = 5.
    limit_yaw_band_deg: float = 25.
    limit_pitch_band_deg: float = 15.
    random_yaw_deg: float = 90.
    random_pitch_deg: float = 60.
    random_seed: int = 61

    def __post_init__(self):
        if self.mode not in ("zero", "mixed", "random"):
            raise ValueError("turret_init_mode must be zero, mixed or random")
        if any(not math.isfinite(v) or v < 0 for v in (self.random_yaw_deg, self.random_pitch_deg)):
            raise ValueError("Random turret init extents must be finite and nonnegative")
        if not isinstance(self.random_seed, int) or not 0 <= self.random_seed < 2**31:
            raise ValueError("Turret init seed must be an integer in [0, 2**31)")
        if (len(self.probabilities) != 3
                or any(not math.isfinite(p) or p < 0 for p in self.probabilities)
                or not math.isclose(sum(self.probabilities), 1., abs_tol=1e-6)):
            raise ValueError("Turret init probabilities must be three finite nonnegative values summing to 1")
        if any(not math.isfinite(v) or v < 0 for v in (self.general_yaw_deg, self.general_pitch_deg)):
            raise ValueError("General turret init extents must be finite and nonnegative")
        if any(not math.isfinite(v) or v <= 0 for v in (
            self.limit_margin_deg, self.limit_yaw_band_deg, self.limit_pitch_band_deg
        )):
            raise ValueError("Turret init margin and near-limit bands must be finite and positive")


class TurretInitSample(NamedTuple):
    positions: torch.Tensor
    groups: torch.Tensor  # 0=zero, 1=general, 2=near-limit
    limit_axes: torch.Tensor  # bitmask: 0=none, 1=yaw, 2=pitch, 3=both


def sample_turret_initialization(limits, config=TurretInitConfig(), *, env_ids=None, episode_indices=None):
    """Sample absolute joint angles using each environment's authored limits.

    Zero mode consumes no random numbers, preserving evaluation initialization.
    Near-limit samples evenly mix yaw-only, pitch-only and both-axis states;
    signs are independent. The other axis uses its general sampling range.
    Random mode uses full uniform evaluation extents and private NumPy streams
    keyed by seed, environment ID and episode index; it leaves global training
    and scene RNG state unchanged.
    """
    if limits.ndim != 3 or limits.shape[1:] != (2, 2) or not limits.is_floating_point():
        raise ValueError("Expected floating yaw/pitch limits [env, 2, lower/upper]")
    lower, upper = limits.unbind(-1)
    if (not torch.isfinite(limits).all() or not (lower < upper).all()
            or not ((lower <= 0) & (upper >= 0)).all()):
        raise ValueError("Turret init requires finite ordered joint limits containing zero")
    count = len(limits)
    positions = torch.zeros_like(lower)
    groups = torch.zeros(count, device=limits.device, dtype=torch.long)
    axes = torch.zeros_like(groups)
    if config.mode == 'zero' or count == 0:
        return TurretInitSample(positions, groups, axes)
    if config.mode == 'random':
        # Use the complete requested interval, without the mixed training
        # mode's five-degree margin or a probability mass at zero.
        extent = np.deg2rad([config.random_yaw_deg, config.random_pitch_deg])
        authored = limits.detach().cpu().double().numpy()
        lo = np.maximum(authored[:, :, 0], -extent)
        hi = np.minimum(authored[:, :, 1], extent)
        ids = list(range(count)) if env_ids is None else env_ids.detach().cpu().tolist()
        episodes = [0]*count if episode_indices is None else episode_indices.detach().cpu().tolist()
        if len(ids) != count or len(episodes) != count or any(value < 0 for value in ids+episodes):
            raise ValueError("Expected one nonnegative environment ID and episode index per reset")
        values = []
        for row, (env_id, episode) in enumerate(zip(ids, episodes)):
            # env 0 follows MuJoCo's SeedSequence([seed, episode]); other
            # parallel environments have independent reset streams.
            entropy = [config.random_seed, episode] + ([env_id] if env_id else [])
            rng = np.random.default_rng(np.random.SeedSequence(entropy))
            values.append(rng.uniform(lo[row], hi[row]))
        positions = torch.as_tensor(np.asarray(values), device=limits.device, dtype=limits.dtype)
        groups.fill_(1)
        return TurretInitSample(positions, groups, axes)
    margin = math.radians(config.limit_margin_deg)
    safe_lower, safe_upper = lower+margin, upper-margin
    if not ((safe_lower < safe_upper) & (safe_lower <= 0) & (safe_upper >= 0)).all():
        raise ValueError("Turret init margin leaves no nonempty zero-containing safe interval")
    extent = limits.new_tensor([config.general_yaw_deg, config.general_pitch_deg]).deg2rad()
    general_lower = torch.maximum(safe_lower, -extent)
    general_upper = torch.minimum(safe_upper, extent)
    positions = general_lower + torch.rand_like(lower)*(general_upper-general_lower)
    groups = torch.multinomial(limits.new_tensor(config.probabilities), count, replacement=True)
    positions[groups == 0] = 0
    axes = torch.where(groups == 2, torch.randint(1, 4, (count,), device=limits.device), axes)
    band = limits.new_tensor([config.limit_yaw_band_deg, config.limit_pitch_band_deg]).deg2rad()
    band = torch.minimum(band, (safe_upper-safe_lower)*.5)
    offset = torch.rand_like(lower)*band
    near = torch.where(torch.rand_like(lower) < .5, safe_lower+offset, safe_upper-offset)
    axis_mask = torch.stack(((axes & 1) != 0, (axes & 2) != 0), -1)
    return TurretInitSample(torch.where(axis_mask, near, positions), groups, axes)


def resolve_training_cases(args):
    """Default recovery training to SS/SD, keeping explicit case mixes available."""
    names = ('p_ss', 'p_sd', 'p_ds', 'p_dd')
    full_chain = args.training_mode == 'full_chain'
    defaults = (.4, .6, 0., 0.) if full_chain or args.turret_init_mode == 'mixed' else (.3, .3, .2, .2)
    values = tuple(default if getattr(args, name) is None else float(getattr(args, name))
                   for name, default in zip(names, defaults))
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("Case probabilities must be finite, nonnegative and have a positive sum")
    if full_chain and (values[2] > 0 or values[3] > 0):
        raise ValueError("Full-chain velocity checkpoints support SS/SD only; set --p_ds 0 --p_dd 0")
    total = sum(values)
    for name, value in zip(names, values):
        setattr(args, name, value/total)
    return tuple(getattr(args, name) for name in names)

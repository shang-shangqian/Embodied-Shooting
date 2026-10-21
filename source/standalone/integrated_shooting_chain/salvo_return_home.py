"""Optional physical homing before each SS engagement attempt, without Isaac imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class SalvoReturnHomeConfig:
    angle_tolerance_deg: float = .5
    velocity_tolerance_rps: float = .05
    settle_frames: int = 5
    timeout_s: float = 8.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (
            self.angle_tolerance_deg, self.velocity_tolerance_rps, self.timeout_s
        )):
            raise ValueError("Return-home tolerances and timeout must be finite and positive")
        if isinstance(self.settle_frames, bool) or not isinstance(self.settle_frames, int) or self.settle_frames < 1:
            raise ValueError("Return-home settle_frames must be a positive integer")


class SalvoReturnHome:
    def __init__(self, env, config=SalvoReturnHomeConfig()):
        self.env, self.config = env, config
        self.joint_ids = [int(env._yaw_dof_ids[0]), int(env._pitch_dof_ids[0])]
        limits = env.turret.data.joint_pos_limits[:, self.joint_ids]
        if not torch.isfinite(limits).all() or not ((limits[..., 0] <= 0) & (limits[..., 1] >= 0)).all():
            raise ValueError("Return-home yaw=pitch=0 must lie inside every USD joint range")
        self.max_delta = torch.tensor([env.cfg.max_delta_yaw, env.cfg.max_delta_pitch], device=env.device)
        if not torch.isfinite(self.max_delta).all() or not (self.max_delta > 0).all():
            raise ValueError("Return-home requires positive finite yaw/pitch action increments")
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.failed = torch.zeros_like(self.active)
        self.control_active = torch.zeros_like(self.active)
        self.release_pending = torch.zeros_like(self.active)
        self.keys = [None] * env.num_envs
        self.started_s = [None] * env.num_envs
        self.last_update_step = [None] * env.num_envs
        self.settled = [0] * env.num_envs
        self.steps = [0] * env.num_envs
        self.completed = [0] * env.num_envs
        self.events = [[] for _ in range(env.num_envs)]

    def reset_envs(self, ids):
        for i in ids:
            self.active[i] = self.failed[i] = self.control_active[i] = self.release_pending[i] = False
            self.keys[i] = self.started_s[i] = self.last_update_step[i] = None
            self.settled[i] = self.steps[i] = self.completed[i] = 0
            self.events[i].clear()

    def _event(self, i, name, now, **details):
        key = self.keys[i]
        self.events[i].append(dict(event=name, sim_s=now,
                                   target_id=key[0], attempt=key[1], **details))

    def prepare(self):
        """Sample actual articulation state once per control step, before inference."""
        env = self.env
        pool = env._static_salvo
        positions = env.turret.data.joint_pos[:, self.joint_ids]
        velocities = env.turret.data.joint_vel[:, self.joint_ids]
        for i in range(env.num_envs):
            step = int(env.episode_length_buf[i])
            if self.last_update_step[i] == step:
                continue
            self.last_update_step[i] = step
            if self.failed[i] or int(env._aim_warmup_left_steps[i]) > 0:
                continue
            target = int(pool.active[i])
            scheduler = pool.engagement_schedulers[i]
            now = step * env.step_dt
            if target < 0:
                if self.active[i]:
                    self._event(i, "return_home_cancelled", now, duration_s=now-self.started_s[i])
                    self.active[i] = self.release_pending[i] = False
                    self.keys[i] = None
                    self.settled[i] = 0
                # An idle gap must not repeat a completed home for the same attempt.
                continue
            key = (target, scheduler.attempts[target])
            if key != self.keys[i]:
                if self.active[i]:
                    self._event(i, "return_home_cancelled", now, duration_s=now-self.started_s[i])
                self.active[i] = self.release_pending[i] = False
                self.keys[i] = key
                self.settled[i] = 0
                if key is not None:
                    self.active[i] = True
                    self.started_s[i] = now
                    # Align the command with the actual pose once. This removes
                    # the previous command's lead without teleporting any joint.
                    env._turret_pos_targets[i] = positions[i]
                    self._event(i, "return_home_started", now,
                                initial_joint_pos_rad=positions[i].detach().cpu().tolist())
            if not self.active[i]:
                continue
            stable = (torch.isfinite(positions[i]).all() and torch.isfinite(velocities[i]).all()
                      and (positions[i].abs() <= math.radians(self.config.angle_tolerance_deg)).all()
                      and (velocities[i].abs() <= self.config.velocity_tolerance_rps).all())
            self.settled[i] = self.settled[i]+1 if stable else 0
            if self.settled[i] >= self.config.settle_frames:
                env._turret_pos_targets[i] = 0
                self.active[i] = False
                self.release_pending[i] = True
                self.completed[i] += 1
                self._event(i, "return_home_completed", now, duration_s=now-self.started_s[i],
                            final_joint_pos_rad=positions[i].detach().cpu().tolist(),
                            final_joint_vel_rps=velocities[i].detach().cpu().tolist())
            elif now-self.started_s[i] >= self.config.timeout_s-1e-9:
                self.active[i] = False
                self.failed[i] = True
                self._event(i, "return_home_timeout", now, duration_s=now-self.started_s[i])

    def override_actions(self, actions):
        """Use the normal incremental-position/PD control path; never write joint state."""
        self.control_active.copy_(self.active)
        mask = self.active | self.failed
        if not mask.any():
            return actions
        result = actions.clone()
        result[mask] = 0
        result[mask, 2] = -1.
        result[self.active, :2] = (-self.env._turret_pos_targets[self.active] / self.max_delta).clamp(-1., 1.)
        for i in self.active.nonzero(as_tuple=False).flatten().tolist():
            scheduler = self.env._static_salvo.engagement_schedulers[i]
            scheduler.exclude_attempt_time(self.env.step_dt)
            self.steps[i] += 1
        return result

    def summary(self, i):
        return dict(enabled=True, target_joint_angles_rad=[0., 0.], config=asdict(self.config),
                    control="rate_limited_joint_position_targets_no_state_reset",
                    homing_steps=self.steps[i], homing_sim_s=self.steps[i]*self.env.step_dt,
                    completed=self.completed[i], failed=bool(self.failed[i]),
                    events=list(self.events[i]))


def install_salvo_return_home(env, config=SalvoReturnHomeConfig()):
    controller = SalvoReturnHome(env, config)
    env._static_salvo.return_home = controller
    return controller

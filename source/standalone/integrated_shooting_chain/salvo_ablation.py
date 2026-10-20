"""SS evaluation controls; default to D and retain A/B/C comparisons."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import types

import torch


@dataclass(frozen=True)
class SalvoAblationConfig:
    group: str = "D"
    wait_for_velocity: bool | None = None

    def __post_init__(self):
        if self.group not in "ABCD" or len(self.group) != 1:
            raise ValueError("salvo ablation group must be A, B, C or D")

    @property
    def hold_when_unready(self):
        return self.group in ("B", "D") or self.pause_actor_when_unready

    @property
    def reset_actor_on_switch(self):
        return self.group in ("C", "D")

    @property
    def pause_actor_when_unready(self):
        return self.group == "D" if self.wait_for_velocity is None else self.wait_for_velocity


def layout_digest(positions):
    return hashlib.sha256(json.dumps(positions, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_layout(path, target_count, radius):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    points = record["positions_env_local_m"]
    tensor = torch.tensor(points, dtype=torch.float32)
    if record.get("schema_version") != 1 or tensor.shape != (target_count, 3) or not torch.isfinite(tensor).all():
        raise ValueError("Invalid static salvo layout schema/count/positions")
    if abs(float(record["target_sphere_radius_m"])-float(radius)) > 1e-6:
        raise ValueError("Layout sphere radius differs from evaluation configuration")
    if (tensor[:, 2] < radius).any() or layout_digest(points) != record["sha256"]:
        raise ValueError("Layout is below ground or its checksum does not match")
    return tensor, record["sha256"]


def save_layout(path, positions, radius):
    points = positions.detach().cpu().tolist()
    digest = layout_digest(points)
    record = dict(schema_version=1, positions_env_local_m=points,
                  target_sphere_radius_m=float(radius), sha256=digest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return digest


class SalvoAblation:
    def __init__(self, env, group, *, wait_for_velocity=None):
        self.config = SalvoAblationConfig(group, wait_for_velocity)
        self.env = env
        self.last_target = [None]*env.num_envs
        self.hold_applied = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reset_applied = torch.zeros_like(self.hold_applied)
        self.actor_inference_skipped = torch.zeros_like(self.hold_applied)
        self.proposed_delta = torch.zeros((env.num_envs, 2), device=env.device)
        self.hold_steps = [0]*env.num_envs
        self.actor_resets = [0]*env.num_envs
        self.actor_wait_steps = [0]*env.num_envs
        self.events = [[] for _ in range(env.num_envs)]

    def reset_envs(self, ids):
        for i in ids:
            self.last_target[i] = None
            self.hold_steps[i] = self.actor_resets[i] = 0
            self.actor_wait_steps[i] = 0
            self.events[i].clear()
            self.hold_applied[i] = self.reset_applied[i] = False
            self.actor_inference_skipped[i] = False

    def prediction_ready(self):
        env = self.env
        ready = env._rl_pred_ready & env._rl_bridge_training_ready & (env._static_salvo.active >= 0)
        if self.config.pause_actor_when_unready:
            ready = ready & env._rl_bridge_velocity_ready
        return ready

    def paused_envs(self):
        paused = (~self.prediction_ready() & (self.env._aim_warmup_left_steps <= 0)
                  if self.config.pause_actor_when_unready else torch.zeros_like(self.hold_applied))
        self.actor_inference_skipped.copy_(paused)
        for i in paused.nonzero(as_tuple=False).flatten().tolist():
            self.actor_wait_steps[i] += 1
        return paused

    def before_inference(self, player):
        self.reset_applied[:] = False
        for i, target in enumerate(self.env._static_salvo.active.tolist()):
            if target < 0:
                continue  # A temporary no-target gap does not erase the previous identity.
            old = self.last_target[i]
            self.last_target[i] = target
            # Preserve the common warmup history at the first acquisition.
            if old is None or old == target or not self.config.reset_actor_on_switch:
                continue
            states = player.states
            if not isinstance(states, (tuple, list)) or len(states) != 2 or states[0].ndim != 3 or states[0].shape[1] != self.env.num_envs:
                raise RuntimeError("Expected separate Actor/Critic GRU states [layers, envs, hidden]")
            states[0][:, i, :] = 0  # Never reset the Critic or velocity-estimator GRU.
            self.reset_applied[i] = True
            self.actor_resets[i] += 1
            self.events[i].append(dict(sim_s=int(self.env.episode_length_buf[i])*self.env.step_dt,
                                       old_target_id=old, target_id=target, event="actor_gru_reset"))

    def summary(self, env_id):
        return dict(group=self.config.group, hold_when_unready=self.config.hold_when_unready,
                    reset_actor_on_switch=self.config.reset_actor_on_switch,
                    wait_for_velocity=self.config.pause_actor_when_unready,
                    actor_wait_steps=self.actor_wait_steps[env_id],
                    actor_wait_sim_s=self.actor_wait_steps[env_id]*self.env.step_dt,
                    hold_steps=self.hold_steps[env_id], actor_gru_resets=self.actor_resets[env_id],
                    events=list(self.events[env_id]))


def install_salvo_ablation(env, group="D", *, wait_for_velocity=None):
    experiment = SalvoAblation(env, group, wait_for_velocity=wait_for_velocity)
    env._static_salvo.ablation = experiment
    original_pre, original_reset = env._pre_physics_step, env._reset_idx

    def pre(self, actions):
        ready = experiment.prediction_ready()
        hold = ~ready & (self._aim_warmup_left_steps <= 0) & (
            experiment.config.hold_when_unready or experiment.config.pause_actor_when_unready)
        command = self._turret_pos_targets.clone() if hold.any() else None
        result = original_pre(actions)
        experiment.proposed_delta.copy_(self._delta_pos)
        experiment.hold_applied.copy_(hold)
        if command is not None:
            self._turret_pos_targets[hold] = command[hold]
            self._delta_pos[hold] = 0
            for i in hold.nonzero(as_tuple=False).flatten().tolist():
                experiment.hold_steps[i] += 1
        return result

    def reset(self, ids):
        result = original_reset(ids)
        experiment.reset_envs(range(self.num_envs) if ids is None else [int(i) for i in ids])
        return result

    env._pre_physics_step = types.MethodType(pre, env)
    env._reset_idx = types.MethodType(reset, env)
    return experiment


def install_actor_switch_reset(player, experiment):
    """Reset on identity change, then infer only for environments with valid velocity.

    Warmup retains its existing inference behavior. Waiting environments receive
    [0, 0, -1] (hold/no fire), and neither recurrent state consumes placeholder
    observations. Ready environments continue independently in a compact batch.
    """
    original = player.get_action

    def get_action(self, obs, *args, **kwargs):
        experiment.before_inference(self)
        paused = experiment.paused_envs()
        if not paused.any():
            return original(obs, *args, **kwargs)

        env = experiment.env
        actions = torch.zeros((env.num_envs, 3), device=env.device)
        actions[:, 2] = -1.0
        batched = self.has_batch_dimension
        ready_ids = (~paused).nonzero(as_tuple=False).flatten()
        if ready_ids.numel() == 0:
            return actions if batched else actions[0]

        def select_observations(value):
            if isinstance(value, dict):
                return {key: select_observations(item) for key, item in value.items()}
            if not isinstance(value, torch.Tensor):
                raise TypeError("Expected tensor or dictionary observations for salvo inference")
            value = value if batched else value.unsqueeze(0)
            return value.index_select(0, ready_ids.to(value.device))

        full_states = self.states
        self.states = [state.index_select(1, ready_ids.to(state.device)) for state in full_states]
        self.has_batch_dimension = True
        try:
            ready_actions = original(select_observations(obs), *args, **kwargs)
            updated_states = self.states
        finally:
            self.states = full_states
            self.has_batch_dimension = batched
        for state, updated in zip(full_states, updated_states):
            state.index_copy_(1, ready_ids.to(state.device), updated)
        actions.index_copy_(0, ready_ids.to(actions.device), ready_actions.to(actions.device))
        return actions if batched else actions[0]

    player.get_action = types.MethodType(get_action, player)

"""SS multi-target evaluation; imports Isaac only when rendering the target pool."""
from __future__ import annotations

import types

import torch


class StaticSalvoState:
    """One fixed target pool and independent counters per parallel environment."""

    def __init__(self, num_envs, target_count, device, step_dt):
        if target_count < 1 or step_dt <= 0:
            raise ValueError("target_count and step_dt must be positive")
        self.positions = torch.zeros((num_envs, target_count, 3), device=device)
        self.alive = torch.zeros((num_envs, target_count), dtype=torch.bool, device=device)
        self.active = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.shots = torch.zeros_like(self.steps)
        self.hits = torch.zeros_like(self.steps)
        self.first_shot = torch.full_like(self.steps, -1)
        self.last_shot = torch.full_like(self.steps, -1)
        self.bootstrap_frames = torch.zeros_like(self.steps)
        self.bootstrap_targets = torch.zeros_like(self.steps)
        self.last_hit = torch.zeros_like(self.ready)
        self.finished = torch.zeros_like(self.ready)
        self.step_dt = float(step_dt)
        self.records = []

    def reset(self, ids):
        self.ready[ids] = False
        self.alive[ids] = False
        self.active[ids] = -1
        self.steps[ids] = self.shots[ids] = self.hits[ids] = 0
        self.first_shot[ids] = self.last_shot[ids] = -1
        self.bootstrap_frames[ids] = self.bootstrap_targets[ids] = 0
        self.last_hit[ids] = False
        self.finished[ids] = False

    def advance(self, fire, hit):
        fire = fire.bool() & self.ready
        hit = hit.bool() & fire & (self.active >= 0)
        self.last_hit.copy_(hit)
        schedulers = getattr(self, "engagement_schedulers", None)
        if schedulers is not None:
            for env_id in (fire & (self.active >= 0)).nonzero(as_tuple=False).flatten().tolist():
                schedulers[env_id].record_shot(int(self.active[env_id]), hit=bool(hit[env_id]))
        self.steps += self.ready.long()
        self.shots += fire.long()
        self.first_shot = torch.where(fire & (self.first_shot < 0), self.steps, self.first_shot)
        self.last_shot = torch.where(fire, self.steps, self.last_shot)
        ids = hit.nonzero(as_tuple=False).flatten()
        self.alive[ids, self.active[ids]] = False
        self.hits += hit.long()
        return hit, self.ready & ~self.alive.any(dim=-1)

    def finish(self, env_id, reason, warmup_s):
        if bool(self.finished[env_id]):
            return next(record for record in reversed(self.records) if record["env_id"] == env_id)
        elapsed = int(self.steps[env_id]) * self.step_dt
        shots, hits = int(self.shots[env_id]), int(self.hits[env_id])
        span = int(self.last_shot[env_id] - self.first_shot[env_id]) * self.step_dt
        record = {
            "env_id": env_id, "targets": self.positions.shape[1],
            "destroyed": hits, "remaining": self.positions.shape[1] - hits,
            "cleared": bool(self.ready[env_id] and not self.alive[env_id].any()),
            "reason": reason, "shots": shots,
            "accuracy": hits / shots if shots else None,
            "engagement_sim_s": elapsed, "warmup_sim_s": float(warmup_s),
            "total_sim_s": elapsed + float(warmup_s),
            "clearance_sim_s": elapsed if hits == self.positions.shape[1] else None,
            "shots_per_engagement_s": shots / elapsed if elapsed else None,
            "mean_firing_hz": (shots - 1) / span if shots > 1 and span > 0 else None,
            "gt_bootstrap_frames": int(self.bootstrap_frames[env_id]),
            "gt_bootstrap_targets": int(self.bootstrap_targets[env_id]),
            "remaining_target_ids": (torch.nonzero(self.alive[env_id], as_tuple=False).flatten().tolist()
                                     if bool(self.ready[env_id]) else list(range(self.positions.shape[1]))),
        }
        schedulers = getattr(self, "engagement_schedulers", None)
        if schedulers is not None:
            scheduler = schedulers[env_id]
            scheduler.removed.update(torch.nonzero(~self.alive[env_id], as_tuple=False).flatten().tolist())
            record.update(scheduler.summary())
        if hasattr(self, "ablation"):
            record["ablation"] = self.ablation.summary(env_id)
        if hasattr(self, "return_home"):
            record["return_home"] = self.return_home.summary(env_id)
        record["layout_sha256"] = getattr(self, "layout_sha256", None)
        self.records.append(record)
        self.finished[env_id] = True
        return record


def install_static_salvo(base_env, *, target_count=100, stall_timeout_s=3.0, retry_delay_s=1.0, max_attempts=3,
                         miss_limit=5, switch_policy="both",
                         layout_in=None, layout_out=None):
    """Install after the mixed-case hooks and before a perception bridge.

    All GT boxes are provided once to the multi-target bridge after warmup.
    Later selection uses tracker estimates. The selected ID is mapped to its
    simulation sphere only for rendering, raycast measurement and hit scoring;
    other pool spheres are not raycast occluders.
    """
    state = StaticSalvoState(base_env.num_envs, target_count, base_env.device, base_env.step_dt)
    base_env._static_salvo = state
    state.engagement_config = dict(stall_timeout_s=stall_timeout_s, retry_delay_s=retry_delay_s,
                                   max_attempts=max_attempts, miss_limit=miss_limit, switch_policy=switch_policy)
    original_reset = base_env._reset_idx
    original_observations = base_env._get_observations
    original_dones = base_env._get_dones
    original_rewards = base_env._get_rewards
    pending_switch = torch.zeros_like(state.ready)
    warmup_elapsed = torch.zeros(base_env.num_envs, device=base_env.device)
    fixed_layout = None
    if layout_in and layout_out:
        raise ValueError("layout_in and layout_out are mutually exclusive")
    if layout_in:
        from integrated_shooting_chain.salvo_ablation import load_layout
        fixed_layout, state.layout_sha256 = load_layout(layout_in, target_count, base_env.cfg.target_sphere_radius)

    def sync_visuals(ids):
        if not base_env._target_sphere_visuals_enabled:
            return
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom

        stage = get_current_stage()
        for env_id in ids.tolist():
            points = (state.positions[env_id] - base_env.scene.env_origins[env_id]).cpu().tolist()
            for target_id, point in enumerate(points):
                path = f"/World/envs/env_{env_id}/SalvoTargets/target_{target_id}"
                sphere = UsdGeom.Sphere.Get(stage, path)
                if not sphere:
                    sphere = UsdGeom.Sphere.Define(stage, path)
                    sphere.CreateRadiusAttr(float(base_env.cfg.target_sphere_radius))
                    sphere.CreateDisplayColorAttr([Gf.Vec3f(*base_env.cfg.target_sphere_color)])
                    sphere.AddTranslateOp().Set(Gf.Vec3d(*point))
                else:
                    UsdGeom.Xformable(sphere).GetOrderedXformOps()[0].Set(Gf.Vec3d(*point))
                # TargetSphere represents the selected target for raycasting.
                visible = bool(state.alive[env_id, target_id]) and target_id != int(state.active[env_id])
                sphere.GetVisibilityAttr().Set("inherited" if visible else "invisible")

    def activate(ids):
        active_ids = ids[state.active[ids] >= 0]
        inactive_ids = ids[state.active[ids] < 0]
        base_env._target_pos_w[active_ids] = state.positions[active_ids, state.active[active_ids]]
        if len(inactive_ids):
            base_env._target_pos_w[inactive_ids] = base_env.scene.env_origins[inactive_ids]
            base_env._target_pos_w[inactive_ids, 2] -= 1000.0
        base_env._target_vel_w[ids] = 0
        base_env._target_acc_w[ids] = 0
        base_env._episode_start_target_pos_w[active_ids] = base_env._target_pos_w[active_ids]
        base_env._hit_flag[ids] = False
        base_env._fire_command[ids] = 0
        base_env._fire_intent_high_steps[ids] = 0
        base_env._invalidate_ballistic_cache()
        base_env._vision_supervision_refreshed_this_step = False
        for flag in ("_rl_pred_ready", "_rl_bridge_training_ready", "_rl_bridge_velocity_ready"):
            getattr(base_env, flag)[ids] = False
        base_env._update_rendered_target_spheres(ids)
        sync_visuals(ids)
        # The bridge must read RGB of the newly selected target, not the
        # pre-switch render cached earlier in the same physics step.
        if base_env.cfg.enable_vision_supervision:
            base_env._sync_render_camera_before_read()

    def select_tracked_target(self, env_id, target_id):
        target_id = -1 if target_id is None else int(target_id)
        if target_id >= 0 and (target_id >= target_count or not bool(state.alive[env_id, target_id])):
            raise ValueError(f"Tracker selected an unavailable salvo target: {target_id}")
        if int(state.active[env_id]) == target_id:
            return
        state.active[env_id] = target_id
        activate(torch.tensor([env_id], device=self.device, dtype=torch.long))

    def initialize(ids):
        # Fixed world positions sampled once, after the standing warmup.
        if fixed_layout is not None:
            world = fixed_layout.to(base_env.device)[None]+base_env.scene.env_origins[ids, None]
        else:
            cam_pos, cam_quat = base_env._get_runtime_camera_pose_w()
            count = len(ids) * target_count
            def uniform(bounds):
                return torch.empty((len(ids), target_count), device=base_env.device).uniform_(*bounds)
            quats = cam_quat[ids, None].expand(-1, target_count, -1).reshape(count, 4)
            world = torch.zeros_like(state.positions[ids])
            missing = torch.ones_like(state.alive[ids])
            for _ in range(256):
                radius = uniform(base_env.cfg.target_radius_range)
                yaw = uniform(base_env.cfg.target_cam_yaw_range)
                pitch = uniform(base_env.cfg.target_cam_pitch_range)
                local = torch.stack((radius * yaw.sin() * pitch.cos(), -radius * pitch.sin(),
                                     radius * yaw.cos() * pitch.cos()), dim=-1)
                candidates = base_env._rotate_local_to_world(quats, local.reshape(count, 3)).reshape(len(ids), target_count, 3)
                candidates += cam_pos[ids, None]
                valid = candidates[..., 2] >= (base_env.scene.env_origins[ids, None, 2] + float(base_env.cfg.target_sphere_radius))
                accept = missing & valid
                world[accept] = candidates[accept]
                missing &= ~accept
                if not missing.any():
                    break
            if missing.any():
                raise ValueError("Cannot sample above-ground salvo targets in the camera sector.")
        if layout_out:
            from integrated_shooting_chain.salvo_ablation import save_layout
            state.layout_sha256 = save_layout(layout_out, world[0]-base_env.scene.env_origins[ids[0]],
                                              base_env.cfg.target_sphere_radius)
        state.positions[ids] = world
        state.alive[ids] = True
        state.ready[ids] = True
        warmup_elapsed[ids] = base_env.episode_length_buf[ids] * base_env.step_dt
        activate(ids)

    def observations(self):
        ids = pending_switch.nonzero(as_tuple=False).flatten()
        if len(ids):
            state.active[ids] = -1
            activate(ids)
            pending_switch[ids] = False
        ready = (~state.ready) & (~self._pending_target_resample) & (self._aim_warmup_left_steps <= 0)
        ids = ready.nonzero(as_tuple=False).flatten()
        if len(ids):
            initialize(ids)
        return original_observations()

    def dones(self):
        current_hit = self._compute_current_step_hit()
        hit, cleared = state.advance(self._fire_command, current_hit)
        # Suppress single-target termination while retaining the base failure,
        # timeout and evaluation logic. Rewards still score the old target.
        self._hit_flag.copy_(cleared)
        original_hit = self._compute_current_step_hit
        self._compute_current_step_hit = lambda: current_hit & cleared
        try:
            terminated, truncated = original_dones()
        finally:
            self._compute_current_step_hit = original_hit
        reasons = list(getattr(state, "unengageable_reasons", [None] * self.num_envs))
        if hasattr(state, "return_home"):
            for env_id in state.return_home.failed.nonzero(as_tuple=False).flatten().tolist():
                reasons[env_id] = "return_home_timeout"
        stalled = torch.tensor([reason is not None for reason in reasons], device=self.device) & state.ready & ~cleared
        base_failure = terminated & ~cleared
        terminated = terminated | stalled
        done = terminated | truncated
        pending_switch.copy_(hit & ~done)
        for env_id in done.nonzero(as_tuple=False).flatten().tolist():
            reason = ("cleared" if bool(cleared[env_id]) else
                      "timeout" if bool(truncated[env_id]) else
                      "failure" if bool(base_failure[env_id]) else reasons[env_id] or "failure")
            warmup_s = float(warmup_elapsed[env_id]) if bool(state.ready[env_id]) else float(self.episode_length_buf[env_id]) * self.step_dt
            record = state.finish(env_id, reason, warmup_s)
            print(f"[salvo] env={env_id} {reason} destroyed={record['destroyed']}/{target_count} "
                  f"shots={record['shots']} accuracy={record['accuracy']} "
                  f"sim_s={record['engagement_sim_s']:.3f} firing_hz={record['mean_firing_hz']}")
        return terminated, truncated

    def rewards(self):
        result = original_rewards()
        # The reward function latches individual hits again after _get_dones.
        self._hit_flag[pending_switch] = False
        return result

    def reset(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        state.reset(ids)
        pending_switch[ids] = False
        warmup_elapsed[ids] = 0
        result = original_reset(env_ids)
        self._target_motion_mask[ids] = False
        self._target_out_of_view_termination_mask[ids] = False
        sync_visuals(ids)
        return result

    base_env._get_observations = types.MethodType(observations, base_env)
    base_env.set_static_salvo_selected_target = types.MethodType(select_tracked_target, base_env)
    base_env._get_dones = types.MethodType(dones, base_env)
    base_env._get_rewards = types.MethodType(rewards, base_env)
    base_env._reset_idx = types.MethodType(reset, base_env)
    return state

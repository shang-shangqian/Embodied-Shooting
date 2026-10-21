"""Per-step firing gates and actor-input diagnostics for static salvo evaluation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import signal
import time
import types

from .salvo_control_snapshot import SNAPSHOT_FIELDS, control_snapshot


def install_salvo_interrupt_handler():
    """Allow Python writer cleanup before Kit's native shutdown is invoked."""
    previous = signal.getsignal(signal.SIGINT)

    def stop_salvo(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_salvo)
    return previous


def firing_block_reason(*, target_id, warmup, ready, cooldown, intent, threshold,
                        confirmed, fired):
    if fired:
        return "fired"
    blocked = []
    if target_id < 0:
        blocked.append("no_target")
    if warmup:
        blocked.append("warmup")
    if not ready:
        blocked.append("perception_not_ready")
    if cooldown > 0:
        blocked.append("cooldown")
    if intent <= threshold:
        blocked.append("intent_below_threshold")
    elif not confirmed:
        blocked.append("intent_unconfirmed")
    return "|".join(blocked) or "unexpected_fire_block"


class SalvoDiagnostics:
    schema_version = 6
    fields = (
        "env_id", "step", "sim_s", "wall_s", "target_id", "attempt", "shots", "hits", "remaining",
        "action_fire", "fire_intent", "fire_threshold", "intent_confirmed", "pred_ready",
        "training_ready", "warmup_left", "cooldown_before", "fired", "hit", "blocked_reason",
        "pred_distance_m", "pred_speed_mps", "yaw_error_rad", "pitch_error_rad",
        "bbox_width_px", "bbox_height_px", "tracker_confidence", "tracker_status", "tracker_source",
        "episode_index", "decision_frame_step", "decision_sim_s", "post_frame_step", "post_sim_s",
        "post_target_id", "action_yaw", "action_pitch", "action_yaw_raw", "action_pitch_raw",
        "yaw_delta_requested_rad", "pitch_delta_requested_rad",
        "yaw_command_after_action_rad", "pitch_command_after_action_rad",
        "yaw_command_change_rad", "pitch_command_change_rad", "yaw_command_clamped", "pitch_command_clamped",
        "actor_obs_available", "actor_obs_target_id", "actor_obs_frame_step", "actor_obs_sim_s",
        "actor_obs_pred_ready", "actor_obs_training_ready", "actor_observation",
        "actor_obs_distance", "actor_obs_yaw_error", "actor_obs_pitch_error",
        "bbox_center_x_px", "bbox_center_y_px", "tracker_timestamp_s", "tracker_age_s",
        "tracker_yaw_cam_rad", "tracker_pitch_cam_rad", "tracker_distance_m",
        "tracker_gt_center_error_px", "tracker_gt_yaw_error_rad", "tracker_gt_pitch_error_rad",
        "ballistic_cache_valid", "ballistic_min_dist_m", "ballistic_hit", "ballistic_closest_time_s", "hit_radius_m",
        "ablation_group", "hold_applied", "actor_gru_reset", "actor_gru_reset_count",
        "yaw_delta_before_hold_rad", "pitch_delta_before_hold_rad",
        "velocity_ready", "actor_inference_skipped", "actor_wait_step_count",
        "attempt_shots", "consecutive_misses",
        "actor_yaw_goal_rad", "actor_yaw_joint_error_rad", "actor_yaw_goal_reachable",
        "return_home_active", "return_home_failed", "return_home_settle_count",
        "return_home_step_count", "attempt_excluded_sim_s",
    ) + tuple(f"{phase}_{name}" for phase in ("decision", "post") for name in SNAPSHOT_FIELDS)

    def __init__(self, path, *, log_interval_s=10.0):
        if not math.isfinite(log_interval_s) or log_interval_s <= 0:
            raise ValueError("log_interval_s must be finite and positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._file, fieldnames=self.fields)
        self._csv.writeheader()
        self._file.flush()
        self._started = time.monotonic()
        self._last_print = -math.inf
        self._interval = log_interval_s
        self._pending = []
        self._actor_rows = {}
        self._episode_indices = {}

    def reset_envs(self, env_ids):
        for i in env_ids:
            self._actor_rows.pop(i, None)
            self._episode_indices[i] = self._episode_indices.get(i, 0) + 1

    def capture_observations(self, env, observations):
        """Copy the actual env policy output, after sanitize/clip, before RL-Games preprocessing."""
        policy = observations["policy"].detach().cpu().tolist()
        joint_goal_fields = {}
        for name in ("actor_yaw_goal_rad", "actor_yaw_joint_error_rad", "actor_yaw_goal_reachable"):
            value = getattr(env, "_" + name, None)
            if value is not None:
                joint_goal_fields[name] = value.detach().cpu().tolist()
        for i, obs in enumerate(policy):
            self._actor_rows[i] = dict(
                actor_obs_available=True, actor_obs_target_id=int(env._static_salvo.active[i]),
                actor_obs_frame_step=int(env.common_step_counter),
                actor_obs_sim_s=int(env.episode_length_buf[i]) * env.step_dt,
                actor_obs_pred_ready=bool(env._rl_pred_ready[i]),
                actor_obs_training_ready=bool(env._rl_bridge_training_ready[i]),
                actor_observation=json.dumps(obs, separators=(",", ":"), allow_nan=False),
                actor_obs_distance=obs[6], actor_obs_yaw_error=obs[7], actor_obs_pitch_error=obs[8],
                **{name: values[i] for name, values in joint_goal_fields.items()},
            )

    def before_action(self, env, actions):
        self._pending = [dict(
            target_id=int(env._static_salvo.active[i]),
            pred_ready=bool(env._rl_pred_ready[i]),
            training_ready=bool(env._rl_bridge_training_ready[i]),
            velocity_ready=bool(getattr(env, "_rl_bridge_velocity_ready", env._rl_pred_ready)[i]),
            warmup_left=int(env._aim_warmup_left_steps[i]),
            cooldown_before=int(env._fire_cooldown[i]),
        ) for i in range(env.num_envs)]
        raw_actions = actions.detach().cpu().tolist()
        snapshots = control_snapshot(env, [row["target_id"] for row in self._pending])
        bridge = env._full_chain_perception_bridge
        for i, (row, snapshot) in enumerate(zip(self._pending, snapshots)):
            row.update({f"decision_{k}": v for k, v in snapshot.items()})
            row.update(episode_index=self._episode_indices.get(i, 0),
                       decision_frame_step=int(env.common_step_counter),
                       decision_sim_s=int(env.episode_length_buf[i]) * env.step_dt,
                       action_yaw_raw=raw_actions[i][0], action_pitch_raw=raw_actions[i][1],
                       actor_obs_available=False)
            row.update(self._actor_rows.get(i, {}))
            result = bridge._latest_results[i]
            estimate = None if result is None else next(
                (t for t in result.targets if t.target_id == row["target_id"]), None)
            if estimate is None:
                continue
            x1, y1, x2, y2 = estimate.bbox_xyxy
            center_x, center_y = (x1+x2)/2, (y1+y2)/2
            row.update(bbox_width_px=x2-x1, bbox_height_px=y2-y1,
                       bbox_center_x_px=center_x, bbox_center_y_px=center_y,
                       tracker_confidence=estimate.confidence,
                       tracker_status=estimate.status, tracker_source=estimate.source,
                       tracker_timestamp_s=result.timestamp_s,
                       tracker_age_s=int(env.common_step_counter)*env.step_dt-result.timestamp_s,
                       tracker_yaw_cam_rad=estimate.yaw_rad, tracker_pitch_cam_rad=estimate.pitch_rad,
                       tracker_distance_m=estimate.distance_m)
            if snapshot["gt_projection_valid"]:
                row.update(tracker_gt_center_error_px=math.hypot(
                    center_x-snapshot["gt_center_x_px"], center_y-snapshot["gt_center_y_px"]),
                    tracker_gt_yaw_error_rad=math.atan2(
                        math.sin(estimate.yaw_rad-snapshot["gt_yaw_cam_rad"]),
                        math.cos(estimate.yaw_rad-snapshot["gt_yaw_cam_rad"])),
                    tracker_gt_pitch_error_rad=estimate.pitch_rad-snapshot["gt_pitch_cam_rad"])

    def after_action(self, env):
        for i, row in enumerate(self._pending):
            home = getattr(env._static_salvo, "return_home", None)
            if home is not None:
                row.update(return_home_active=bool(home.control_active[i]),
                           return_home_failed=bool(home.failed[i]),
                           return_home_settle_count=home.settled[i], return_home_step_count=home.steps[i],
                           attempt_excluded_sim_s=env._static_salvo.engagement_schedulers[i].excluded_attempt_s)
            ablation = getattr(env._static_salvo, "ablation", None)
            if ablation is not None:
                row.update(ablation_group=ablation.config.group, hold_applied=bool(ablation.hold_applied[i]),
                           actor_gru_reset=bool(ablation.reset_applied[i]), actor_gru_reset_count=ablation.actor_resets[i],
                           actor_inference_skipped=bool(ablation.actor_inference_skipped[i]),
                           actor_wait_step_count=ablation.actor_wait_steps[i],
                           yaw_delta_before_hold_rad=float(ablation.proposed_delta[i, 0]),
                           pitch_delta_before_hold_rad=float(ablation.proposed_delta[i, 1]))
            row.update(action_fire=float(env._actions[i, 2]), fire_intent=float(env._fire_intent[i]),
                       fire_threshold=float(env.cfg.fire_intent_threshold),
                       intent_confirmed=bool(env._fire_intent_high_steps[i] >= env.cfg.fire_intent_hold_steps),
                       fired=bool(env._fire_command[i]))
            yaw_change = float(env._turret_pos_targets[i, 0])-row["decision_yaw_command_rad"]
            pitch_change = float(env._turret_pos_targets[i, 1])-row["decision_pitch_command_rad"]
            yaw_delta, pitch_delta = float(env._delta_pos[i, 0]), float(env._delta_pos[i, 1])
            row.update(action_yaw=float(env._actions[i, 0]), action_pitch=float(env._actions[i, 1]),
                       yaw_delta_requested_rad=yaw_delta, pitch_delta_requested_rad=pitch_delta,
                       yaw_command_after_action_rad=float(env._turret_pos_targets[i, 0]),
                       pitch_command_after_action_rad=float(env._turret_pos_targets[i, 1]),
                       yaw_command_change_rad=yaw_change, pitch_command_change_rad=pitch_change,
                       yaw_command_clamped=abs(yaw_change-yaw_delta)>1e-6,
                       pitch_command_clamped=abs(pitch_change-pitch_delta)>1e-6)

    def record(self, env):
        wall = time.monotonic() - self._started
        print_progress = wall - self._last_print >= self._interval
        pool = env._static_salvo
        snapshots = control_snapshot(env, [row["target_id"] for row in self._pending]) if self._pending else []
        for i, pending in enumerate(self._pending):
            row = dict(pending)
            snapshot = snapshots[i]
            row.update({f"post_{k}": v for k, v in snapshot.items()})
            yaw_error = (snapshot["pred_yaw_base_rad"]-snapshot["muzzle_yaw_base_rad"]
                         if snapshot["pred_yaw_base_rad"] != "" and snapshot["muzzle_yaw_base_rad"] != "" else None)
            pitch_error = (snapshot["pred_pitch_base_rad"]-snapshot["muzzle_pitch_base_rad"]
                           if snapshot["pred_pitch_base_rad"] != "" and snapshot["muzzle_pitch_base_rad"] != "" else None)
            scheduler = pool.engagement_schedulers[i]
            row.update(env_id=i, step=int(env.episode_length_buf[i]),
                       sim_s=int(env.episode_length_buf[i]) * env.step_dt, wall_s=wall,
                       attempt=scheduler.attempts.get(row["target_id"], 0),
                       attempt_shots=scheduler.attempt_shots,
                       consecutive_misses=scheduler.consecutive_misses,
                       shots=int(pool.shots[i]), hits=int(pool.hits[i]),
                       remaining=pool.positions.shape[1]-int(pool.hits[i]), hit=bool(pool.last_hit[i]),
                       pred_distance_m=float(env._rl_pred_distance[i]),
                       pred_speed_mps=float(env._rl_pred_vel_base[i].norm()),
                       yaw_error_rad=math.atan2(math.sin(yaw_error), math.cos(yaw_error)) if yaw_error is not None else "",
                       pitch_error_rad=pitch_error if pitch_error is not None else "")
            row.update(post_frame_step=int(env.common_step_counter),
                       post_sim_s=int(env.episode_length_buf[i])*env.step_dt,
                       post_target_id=int(pool.active[i]), hit_radius_m=float(env.cfg.hit_radius))
            cache_valid = getattr(env, "_ballistic_cache_valid", None)
            valid = (cache_valid is not None and bool(cache_valid[i]) and snapshot["gt_valid"]
                     and row["post_target_id"] == row["target_id"])
            row["ballistic_cache_valid"] = bool(valid)
            if valid:
                row.update(ballistic_min_dist_m=float(env._cached_min_ballistic_dist[i]),
                           ballistic_hit=bool(env._cached_ballistic_hit[i]),
                           ballistic_closest_time_s=float(env._cached_closest_time[i]))
            row["blocked_reason"] = firing_block_reason(
                target_id=row["target_id"], warmup=row["warmup_left"] > 0,
                ready=row["pred_ready"] and row["training_ready"] and not row.get("actor_inference_skipped", False),
                cooldown=row["cooldown_before"], intent=row["fire_intent"],
                threshold=row["fire_threshold"], confirmed=row["intent_confirmed"], fired=row["fired"],
            )
            if not row["fired"] and row.get("return_home_failed"):
                row["blocked_reason"] = "return_home_timeout"
            elif not row["fired"] and row.get("return_home_active"):
                row["blocked_reason"] = "return_home"
            row = {k: "" if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
            self._csv.writerow(row)
            if print_progress:
                yaw_text = f"{row['yaw_error_rad']:.4f}" if row["yaw_error_rad"] != "" else "invalid"
                pitch_text = f"{row['pitch_error_rad']:.4f}" if row["pitch_error_rad"] != "" else "invalid"
                print(f"[salvo][progress] env={i} wall_s={wall:.1f} sim_s={row['sim_s']:.2f} "
                      f"target={row['target_id']} attempt={row['attempt']} shots={row['shots']} "
                      f"hits={row['hits']} remaining={row['remaining']} intent={row['fire_intent']:.3f} "
                      f"gate={row['blocked_reason']} yaw_err={yaw_text} "
                      f"pitch_err={pitch_text}", flush=True)
        if print_progress or any(bool(v) for v in pool.last_hit):
            self._file.flush()
        if print_progress:
            self._last_print = wall
        self._pending = []

    def close(self):
        self._file.close()


def install_salvo_diagnostics(base_env, diagnostics):
    original_pre = base_env._pre_physics_step
    original_rewards = base_env._get_rewards
    original_observations = base_env._get_observations
    original_reset = base_env._reset_idx

    def pre(self, actions):
        diagnostics.before_action(self, actions)
        result = original_pre(actions)
        diagnostics.after_action(self)
        return result

    def rewards(self):
        # Reward evaluation already populated ballistic caches. Record before
        # reset/target switching, without causing another ballistic integration.
        result = original_rewards()
        diagnostics.record(self)
        return result

    def observations(self):
        result = original_observations()
        diagnostics.capture_observations(self, result)
        return result

    def reset(self, env_ids):
        result = original_reset(env_ids)
        diagnostics.reset_envs(range(self.num_envs) if env_ids is None else [int(i) for i in env_ids])
        return result

    base_env._pre_physics_step = types.MethodType(pre, base_env)
    base_env._get_rewards = types.MethodType(rewards, base_env)
    base_env._get_observations = types.MethodType(observations, base_env)
    base_env._reset_idx = types.MethodType(reset, base_env)

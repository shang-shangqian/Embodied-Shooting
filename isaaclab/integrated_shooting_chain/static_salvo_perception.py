"""One RGB/GT-box bootstrap of the entire pool, followed by RGB-only tracking."""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from .isaac_training_bridge import IsaacFullChainPerceptionBridge, _bootstrap_box
from .salvo_engagement import SalvoEngagementConfig, SalvoEngagementScheduler


class IsaacStaticSalvoPerceptionBridge(IsaacFullChainPerceptionBridge):
    """Track all pool IDs continuously and feed the selected track to the actor.

    LK updates all target boxes. The existing single OSTrack instance refines
    the selected target using its current tracked box, never a new GT box.
    Velocity history is reset on an identity switch, independently of vision.
    """

    def __init__(self, base_env, config):
        super().__init__(base_env, config)
        self._pool_timestamps = [None] * self.num_envs
        self._selected_ids = [None] * self.num_envs
        self._removed_ids = [set() for _ in range(self.num_envs)]
        self._latest_results = [None] * self.num_envs
        pool = base_env._static_salvo
        engagement_config = SalvoEngagementConfig(**getattr(pool, "engagement_config", {}))
        self._schedulers = [SalvoEngagementScheduler(engagement_config) for _ in range(self.num_envs)]
        pool.engagement_schedulers = self._schedulers
        pool.unengageable_reasons = [None] * self.num_envs

    def _make_tracker(self, *args, **kwargs):
        tracker = super()._make_tracker(*args, **kwargs)
        tracker.config = replace(tracker.config, selection_policy="nearest_until_removed",
                                 max_initial_scale_ratio=2.0)
        return tracker

    def reset_envs(self, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else [int(i) for i in env_ids]
        super().reset_envs(ids)
        for env_id in ids:
            self._pool_timestamps[env_id] = None
            self._selected_ids[env_id] = None
            self._removed_ids[env_id].clear()
            self._latest_results[env_id] = None
            self._schedulers[env_id].reset()
            self.base_env._static_salvo.unengageable_reasons[env_id] = None

    def _invalidate_env(self, env_id, *, reset_tracker=False, reset_velocity=False):
        # One failed/lost selected track must not erase the other 99 tracks or
        # restore permission to consume GT. Keep the pool's frame clock too.
        if reset_tracker or reset_velocity:
            self.velocity_estimator.reset(env_id)
            self._valid_observations[env_id] = 0
            self._first_valid_timestamp[env_id] = None

    def _bootstrap_inputs(self, batch):
        # Disable the parent bridge's single-target GT label path entirely.
        return None

    def _initial_pool_boxes(self, env_id, image, camera):
        """The only method that reads GT positions for visual initialization."""
        env = self.base_env
        pool = env._static_salvo
        cam_pos, cam_quat = env._get_runtime_camera_pose_w()
        world = pool.positions[env_id] - cam_pos[env_id]
        rotations = cam_quat[env_id].expand(len(world), -1)
        local = env._rotate_world_to_local(rotations, world).detach().cpu()
        height, width = image.shape[:2]
        boxes = {}
        for target_id, (x, y, z) in enumerate(local.tolist()):
            distance = math.sqrt(x*x + y*y + z*z)
            box = None if z <= 0 else _bootstrap_box(
                math.atan2(x, z), math.atan2(-y, math.hypot(x, z)), distance,
                **camera, width=width, height=height,
                target_radius_m=self.config.target_radius_m,
                min_box_px=self.config.min_bootstrap_box_px,
            )
            if box is None:
                raise RuntimeError(f"Salvo target {target_id} is outside the bootstrap image")
            boxes[target_id] = box
        return boxes

    def _track_frame(self, env_id, image, timestamp, camera, bootstrap):
        env = self.base_env
        pool = env._static_salvo
        if not bool(pool.ready[env_id]):
            return None
        tracker = self._trackers[env_id]
        scheduler = self._schedulers[env_id]
        event_start = len(scheduler.events)
        if tracker is None:
            tracker = self._make_tracker(image.shape[1], image.shape[0], **camera)
            self._trackers[env_id] = tracker

        if not self._bootstrap_consumed[env_id]:
            # Consume exactly one frame even if initialization fails. A failed
            # pool bootstrap cannot be repaired with GT from a later frame.
            self._bootstrap_consumed[env_id] = True
            pool.bootstrap_frames[env_id] += 1
            boxes = self._initial_pool_boxes(env_id, image, camera)
            pool.bootstrap_targets[env_id] = len(boxes)
            result = tracker.initialize(image, boxes, timestamp_s=timestamp)
            selected = scheduler.choose(timestamp, tracker.selectable_target_distances())
            tracker.set_selection_candidates([] if selected is None else [selected])
        else:
            if not tracker.initialized:
                env.set_static_salvo_selected_target(env_id, None)
                pool.unengageable_reasons[env_id] = "bootstrap_failed"
                return None
            removed = set(torch.nonzero(~pool.alive[env_id], as_tuple=False).flatten().tolist())
            newly_removed = removed - self._removed_ids[env_id]
            tracker.remove_targets(newly_removed)
            self._removed_ids[env_id] = removed
            if not tracker.initialized:
                env.set_static_salvo_selected_target(env_id, None)
                self._pool_timestamps[env_id] = timestamp
                return None
            selected = scheduler.choose(timestamp, tracker.selectable_target_distances(), removed_ids=removed)
            tracker.set_selection_candidates([] if selected is None else [selected])
            previous = self._pool_timestamps[env_id]
            dt_s = float(env.step_dt) if previous is None else timestamp - previous
            if dt_s <= 0:
                raise RuntimeError("Salvo tracking requires increasing frame timestamps")
            # No GT labels, positions, ranges, or boxes enter this update.
            result = tracker.update(image, dt_s=dt_s, timestamp_s=timestamp)
        for event in scheduler.events[event_start:]:
            print(f"[salvo][engagement] env={env_id} {event}", flush=True)
        pool.unengageable_reasons[env_id] = scheduler.terminal_reason(tracker.selectable_target_distances())
        self._pool_timestamps[env_id] = timestamp
        self._latest_results[env_id] = result
        estimate = result.selected_target
        selected_id = None if estimate is None or str(estimate.status) == "lost" else int(estimate.target_id)
        if selected_id != self._selected_ids[env_id]:
            self._invalidate_env(env_id, reset_velocity=True)
            self._selected_ids[env_id] = selected_id
        env.set_static_salvo_selected_target(env_id, selected_id)
        return estimate if selected_id is not None else None

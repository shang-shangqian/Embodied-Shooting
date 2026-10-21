"""Choose the nearest visual target per round; defer failures to the next round."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SalvoEngagementConfig:
    stall_timeout_s: float | None = 3.0
    retry_delay_s: float = 1.0
    max_attempts: int = 3
    miss_limit: int = 5
    switch_policy: str = "both"

    def __post_init__(self):
        if self.switch_policy not in ("both", "misses", "timeout"):
            raise ValueError("switch_policy must be both, misses or timeout")
        if self.switch_policy != "misses" and self.stall_timeout_s is None:
            raise ValueError("stall_timeout_s is required for both/timeout policies")
        if self.stall_timeout_s is not None and (not math.isfinite(self.stall_timeout_s) or self.stall_timeout_s <= 0):
            raise ValueError("stall_timeout_s must be finite and positive")
        if not math.isfinite(self.retry_delay_s) or self.retry_delay_s <= 0:
            raise ValueError("retry_delay_s must be finite and positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if isinstance(self.miss_limit, bool) or not isinstance(self.miss_limit, int) or self.miss_limit < 1:
            raise ValueError("miss_limit must be a positive integer")

class SalvoEngagementScheduler:
    selection_policy = "dynamic_nearest_per_round_deferred_retries"

    def __init__(self, config=SalvoEngagementConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.current_id = None
        self.started_s = None
        self.attempts = {}
        self.deferred_until = {}
        self.removed = set()
        self.events = []
        self.pending_ids = []
        self.retry_ids = []
        self.selection_round = 1
        self.attempt_shots = 0
        self.consecutive_misses = 0
        self.excluded_attempt_s = 0.

    def exclude_attempt_time(self, dt_s):
        """Exclude an external homing control step from only the attempt timeout."""
        if not math.isfinite(dt_s) or dt_s < 0:
            raise ValueError("Excluded attempt duration must be finite and nonnegative")
        if self.current_id is not None:
            self.excluded_attempt_s += dt_s

    def record_shot(self, target_id, *, hit):
        """Called once after an actual shot is scored, before choosing the next ID.

        Time spent aiming/waiting/cooling down does not count. A stale shot for
        another ID cannot consume the current target's budget.
        """
        if self.current_id is None or int(target_id) != self.current_id:
            return
        self.attempt_shots += 1
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1

    def choose(self, now_s, visual_distances, *, removed_ids=()):
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        self.removed.update(int(i) for i in removed_ids)
        candidates = {int(i): float(d) for i, d in visual_distances.items()
                      if int(i) not in self.removed and math.isfinite(d) and d > 0}
        # Round membership is stable, while distance priority is refreshed on
        # every frame. Failed IDs must never rejoin this round after cooldown.
        self.pending_ids = [i for i in self.pending_ids
                            if i not in self.removed
                            and self.attempts.get(i, 0) < self.config.max_attempts]
        self.retry_ids = [i for i in self.retry_ids
                          if i not in self.removed
                          and self.attempts.get(i, 0) < self.config.max_attempts]
        queued = set(self.pending_ids) | set(self.retry_ids)
        self.pending_ids.extend(i for i in candidates if i != self.current_id and i not in queued
                                and self.attempts.get(i, 0) < self.config.max_attempts)
        self.pending_ids.sort(key=lambda i: (candidates.get(i, math.inf), i))
        current = self.current_id
        if current is not None:
            miss_limit_reached = (self.config.switch_policy in ("misses", "both")
                                  and self.consecutive_misses >= self.config.miss_limit)
            timeout_reached = (self.config.switch_policy in ("timeout", "both")
                               and now_s - self.started_s - self.excluded_attempt_s >= self.config.stall_timeout_s - 1e-9)
            if current in self.removed:
                self.events.append(dict(sim_s=now_s, target_id=current, event="destroyed",
                                        selection_round=self.selection_round))
                self.current_id = None
            elif (current not in candidates
                  or miss_limit_reached or timeout_reached):
                exhausted = self.attempts[current] >= self.config.max_attempts
                self.deferred_until[current] = now_s + self.config.retry_delay_s
                self.events.append(dict(sim_s=now_s, target_id=current,
                                        event="exhausted" if exhausted else "deferred",
                                        reason=("tracking_lost" if current not in candidates else
                                                "consecutive_misses" if miss_limit_reached else "no_hit"),
                                        attempt=self.attempts[current], attempt_shots=self.attempt_shots,
                                        consecutive_misses=self.consecutive_misses,
                                        selection_round=self.selection_round))
                if not exhausted:
                    self.retry_ids.append(current)
                self.current_id = None
            else:
                return current
            if self.excluded_attempt_s:
                self.events[-1].update(attempt_excluded_sim_s=self.excluded_attempt_s,
                                       attempt_elapsed_sim_s=max(0., now_s-self.started_s-self.excluded_attempt_s))
        if self.current_id is None:
            self.attempt_shots = self.consecutive_misses = 0
            self.excluded_attempt_s = 0.
        # Unavailable tracks do not block a new round. A visible current-round
        # target still cooling down does block it: wait instead of retrying a
        # target that already failed this round.
        if (not any(i in candidates for i in self.pending_ids)
                and any(i in candidates for i in self.retry_ids)):
            self.pending_ids.extend(self.retry_ids)
            self.retry_ids.clear()
            self.selection_round += 1
            self.pending_ids.sort(key=lambda i: (candidates.get(i, math.inf), i))
        eligible = {i: candidates[i] for i in self.pending_ids if i in candidates
                    and now_s >= self.deferred_until.get(i, -math.inf) - 1e-9}
        if not eligible:
            return None
        selected = min(eligible, key=lambda i: (eligible[i], i))
        self.pending_ids.remove(selected)
        self.current_id = selected
        self.started_s = now_s
        self.excluded_attempt_s = 0.
        self.attempt_shots = self.consecutive_misses = 0
        self.attempts[selected] = self.attempts.get(selected, 0) + 1
        self.events.append(dict(sim_s=now_s, target_id=selected, event="selected",
                                attempt=self.attempts[selected], visual_distance_m=eligible[selected],
                                selection_round=self.selection_round))
        return selected

    def terminal_reason(self, visual_distances):
        if self.current_id is not None:
            return None
        remaining = {i for i, d in visual_distances.items()
                     if i not in self.removed and math.isfinite(d) and d > 0}
        if not remaining:
            return "tracking_lost"
        if all(self.attempts.get(i, 0) >= self.config.max_attempts for i in remaining):
            return "attempts_exhausted"
        # Targets with an unexpired deferral still have a future attempt.
        return None

    def summary(self):
        return {
            "selection": self.selection_policy,
            "selection_round": self.selection_round,
            "pending_target_ids": list(self.pending_ids) + list(self.retry_ids),
            "round_pending_target_ids": list(self.pending_ids),
            "next_round_target_ids": list(self.retry_ids),
            "switch_policy": self.config.switch_policy,
            "miss_limit": self.config.miss_limit,
            "active_target_id": self.current_id,
            "active_attempt_shots": self.attempt_shots,
            "active_consecutive_misses": self.consecutive_misses,
            "active_attempt_excluded_sim_s": self.excluded_attempt_s,
            "attempts": dict(self.attempts),
            "deferred_count": sum(e["event"] == "deferred" for e in self.events),
            "exhausted_target_ids": sorted({e["target_id"] for e in self.events
                                             if e["event"] == "exhausted"} - self.removed),
            "engagement_events": list(self.events),
        }


def required_player_steps(timeout_s, step_dt, episodes):
    """Avoid RL-Games' fixed 27,000-step cap cutting off a longer salvo."""
    return (math.ceil(timeout_s / step_dt) + 1) * max(1, episodes)

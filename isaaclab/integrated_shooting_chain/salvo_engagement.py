"""Bounded engagement attempts driven exclusively by visual track IDs/ranges."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SalvoEngagementConfig:
    stall_timeout_s: float = 3.0
    retry_delay_s: float = 1.0
    max_attempts: int = 3

    def __post_init__(self):
        if not math.isfinite(self.stall_timeout_s) or self.stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be finite and positive")
        if not math.isfinite(self.retry_delay_s) or self.retry_delay_s <= 0:
            raise ValueError("retry_delay_s must be finite and positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")


class SalvoEngagementScheduler:
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

    def choose(self, now_s, visual_distances, *, removed_ids=()):
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        self.removed.update(int(i) for i in removed_ids)
        candidates = {int(i): float(d) for i, d in visual_distances.items()
                      if int(i) not in self.removed and math.isfinite(d) and d > 0}
        current = self.current_id
        if current is not None:
            if current in self.removed:
                self.events.append(dict(sim_s=now_s, target_id=current, event="destroyed"))
                self.current_id = None
            elif current not in candidates or now_s - self.started_s >= self.config.stall_timeout_s - 1e-9:
                exhausted = self.attempts[current] >= self.config.max_attempts
                self.deferred_until[current] = now_s + self.config.retry_delay_s
                self.events.append(dict(sim_s=now_s, target_id=current,
                                        event="exhausted" if exhausted else "deferred",
                                        reason="tracking_lost" if current not in candidates else "no_hit",
                                        attempt=self.attempts[current]))
                self.current_id = None
            else:
                return current
        eligible = {i: d for i, d in candidates.items()
                    if self.attempts.get(i, 0) < self.config.max_attempts
                    and now_s >= self.deferred_until.get(i, -math.inf) - 1e-9}
        if not eligible:
            return None
        selected = min(eligible, key=lambda i: (eligible[i], i))
        self.current_id = selected
        self.started_s = now_s
        self.attempts[selected] = self.attempts.get(selected, 0) + 1
        self.events.append(dict(sim_s=now_s, target_id=selected, event="selected",
                                attempt=self.attempts[selected], visual_distance_m=eligible[selected]))
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
            "attempts": dict(self.attempts),
            "deferred_count": sum(e["event"] == "deferred" for e in self.events),
            "exhausted_target_ids": sorted({e["target_id"] for e in self.events
                                             if e["event"] == "exhausted"} - self.removed),
            "engagement_events": list(self.events),
        }


def required_player_steps(timeout_s, step_dt, episodes):
    """Avoid RL-Games' fixed 27,000-step cap cutting off a longer salvo."""
    return (math.ceil(timeout_s / step_dt) + 1) * max(1, episodes)

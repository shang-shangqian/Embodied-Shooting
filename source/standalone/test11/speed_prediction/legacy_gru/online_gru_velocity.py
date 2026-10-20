"""Frozen legacy GRU adapter for causal measured-position streams."""

from collections import deque
from pathlib import Path
import time

import numpy as np
import torch

from ..common.time_aware_velocity_data import TAQ_LABEL_SEMANTICS, TAQ_SCHEMA_VERSION, load_torch_payload
from .speed_prediction_model import VelocityPredictorGRU
from .taq_dataset import causal_smooth


def load_legacy_checkpoint(path):
    checkpoint = load_torch_payload(Path(path))
    if checkpoint.get("data_schema") != TAQ_SCHEMA_VERSION:
        raise ValueError("Live evaluation requires a TAQ-v2-trained legacy checkpoint")
    if checkpoint.get("label_semantics") != TAQ_LABEL_SEMANTICS:
        raise ValueError("Checkpoint does not declare physical relative translation velocity")
    model = VelocityPredictorGRU(**checkpoint["config"]).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for name in ("mean", "std"):
        torch.testing.assert_close(getattr(model, f"input_{name}").flatten(),
                                   checkpoint["input_norm"][name].flatten(), rtol=0, atol=0)
    return model, checkpoint


def window_inputs(rows, smooth_window):
    values = torch.tensor(np.stack([row[1] for row in rows]), dtype=torch.float32)
    position = causal_smooth(values[:, :3], smooth_window)
    moving = (values[1:, 3:5].norm(dim=-1) > 0.05).float()[:, None]
    return torch.cat((position[1:], position[1:] - position[:-1], values[1:, 3:9], moving), dim=-1)


class OnlineLegacyGRU:
    def __init__(self, model, *, device="cpu", context="rolling", window_s=0.48,
                 smooth_window=1, max_history_frames=32, measure_latency=False,
                 initialization_policy="first_valid"):
        if context not in {"rolling", "full"} or window_s <= 0 or smooth_window < 1 or max_history_frames < 2:
            raise ValueError("Invalid online history configuration")
        if initialization_policy not in {"first_valid", "zero_filled"}:
            raise ValueError("Invalid initialization policy")
        self.model = model
        self.device = torch.device(device)
        self.context = context
        self.window_s = float(window_s)
        self.smooth_window = int(smooth_window)
        self.max_history_frames = int(max_history_frames)
        self.measure_latency = bool(measure_latency)
        self.initialization_policy = initialization_policy
        self.reset()

    def reset(self):
        self.rows = deque()
        self.hidden = None
        self.previous_smoothed = None
        self.previous_time = None
        self.episode_key = None
        self.first_time = None
        self.frame_count = 0
        self.started = False

    @torch.inference_mode()
    def update(self, *, timestamp_s, position_base_m, robot_linear_velocity_base_mps,
               robot_angular_velocity_base_rps, episode_key, direct_observation=None):
        if episode_key != self.episode_key:
            self.reset()
            self.episode_key = episode_key
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s) or (self.previous_time is not None and timestamp_s <= self.previous_time):
            raise ValueError("Observation timestamps must be finite and strictly increasing")
        position = np.asarray(position_base_m, dtype=np.float32)
        linear = np.asarray(robot_linear_velocity_base_mps, dtype=np.float32)
        angular = np.asarray(robot_angular_velocity_base_rps, dtype=np.float32)
        if any(vector.shape != (3,) for vector in (position, linear, angular)):
            raise ValueError("Position and robot velocities must be three-vectors")
        if not np.isfinite(linear).all() or not np.isfinite(angular).all():
            raise ValueError("Robot velocities must be finite")
        self.previous_time = timestamp_s
        if not self.started:
            valid_position = bool(np.isfinite(position).all())
            first_valid = valid_position and (direct_observation is None or bool(direct_observation))
            if self.initialization_policy == "first_valid" and not first_valid:
                return None
            self.started = True
        values = np.concatenate((np.nan_to_num(position, nan=0., posinf=0., neginf=0.), linear, angular))
        self.rows.append((timestamp_s, values))
        self.frame_count += 1
        if self.first_time is None:
            self.first_time = timestamp_s
        if self.context == "rolling":
            while self.rows and self.rows[0][0] < timestamp_s - self.window_s - 1e-9:
                self.rows.popleft()
            while len(self.rows) > self.max_history_frames:
                self.rows.popleft()
        else:
            while len(self.rows) > max(2, self.smooth_window):
                self.rows.popleft()
        if self.measure_latency and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        if self.context == "rolling":
            if len(self.rows) < 2:
                return None
            inputs = window_inputs(list(self.rows), self.smooth_window).to(self.device)
            prediction = self.model(inputs[None])[0]
        else:
            position_history = torch.tensor(np.stack([row[1][:3] for row in self.rows]), dtype=torch.float32)
            smoothed = position_history[-self.smooth_window:].mean(dim=0)
            if self.previous_smoothed is None:
                self.previous_smoothed = smoothed
                return None
            moving = (torch.tensor(linear[:2]).norm() > 0.05).float()[None]
            inputs = torch.cat((smoothed, smoothed - self.previous_smoothed,
                                torch.tensor(linear), torch.tensor(angular), moving)).to(self.device)
            output, self.hidden = self.model.gru(self.model.normalize_input(inputs[None, None]), self.hidden)
            prediction = self.model.head(output[0, 0])
            self.previous_smoothed = smoothed
        velocity = prediction.cpu().numpy()
        if not np.isfinite(velocity).all():
            raise ValueError("GRU produced nonfinite velocity")
        elapsed_ms = (time.perf_counter() - started) * 1000 if self.measure_latency else None
        return dict(velocity=velocity, history_frames=len(self.rows) if self.context == "rolling" else self.frame_count,
                    history_span_s=self.rows[-1][0] - self.rows[0][0] if self.context == "rolling" else timestamp_s - self.first_time,
                    inference_ms=elapsed_ms)

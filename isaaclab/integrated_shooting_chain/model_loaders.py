"""Factories for the existing test_lk_ostrack and test11 runtime modules."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def _standalone_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_lk_ostrack_tracker(intrinsics: Any, *, config: Any | None = None, fine_tracker: Any | None = None) -> Any:
    """Create the existing multi-target LK/OSTrack tracker."""
    root = _standalone_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from test_lk_ostrack.multi_target_tracker import MultiTargetVisionTracker

    return MultiTargetVisionTracker(intrinsics, config=config, fine_tracker=fine_tracker)


def make_lk_ostrack_tracker_with_checkpoint(
    intrinsics: Any,
    *,
    ostrack_repo: str | Path | None = None,
    ostrack_checkpoint: str | Path | None = None,
    ostrack_config: str = "vitb_256_mae_ce_32x4_ep300",
    device: str = "cuda:0",
    config: Any | None = None,
) -> Any:
    """Build the LK tracker and, when supplied, its selected-target OSTrack refiner."""
    root = _standalone_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    fine_tracker = None
    if ostrack_repo is not None or ostrack_checkpoint is not None:
        if ostrack_repo is None or ostrack_checkpoint is None:
            raise ValueError("ostrack_repo and ostrack_checkpoint must be provided together")
        from test_lk_ostrack.ostrack_tracker import OSTrackFrameTracker

        fine_tracker = OSTrackFrameTracker(
            repo_path=Path(ostrack_repo).expanduser(),
            checkpoint=Path(ostrack_checkpoint).expanduser(),
            config=ostrack_config,
            device=device,
        )
        # Avoid first-frame model construction in the control loop.
        fine_tracker.prepare()
    return make_lk_ostrack_tracker(intrinsics, config=config, fine_tracker=fine_tracker)


def load_taq_velocity_estimator(checkpoint_path: str | Path, *, device: str | Any = "cpu", history_window_s: float = 0.48, max_history_steps: int = 32) -> Any:
    """Load a schema-v2 TAQ checkpoint and return its online estimator.

    Old ``test6``/TAQ-v1 checkpoints are rejected intentionally; their labels and
    feature semantics are incompatible with the Stage 1 velocity module.
    """
    import torch

    root = _standalone_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from test11.speed_prediction.online_taq_velocity import OnlineTAQVelocityEstimator
    from test11.speed_prediction.time_aware_velocity_data import (
        TAQ_LABEL_SEMANTICS,
        TAQ_SCHEMA_VERSION,
        taq_model_config_from_checkpoint,
    )
    from test11.speed_prediction.time_aware_velocity_model import TimeAwareQualityGatedGRU

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"TAQ velocity checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError("TAQ checkpoint must be a dictionary payload")
    found = payload.get("schema_version")
    if found != TAQ_SCHEMA_VERSION:
        raise RuntimeError(
            f"Incompatible velocity checkpoint schema {found!r}; expected {TAQ_SCHEMA_VERSION!r}. "
            "Use test11/speed_prediction/train_taq_velocity.py to produce a v2 checkpoint."
        )
    if payload.get("label_semantics") != TAQ_LABEL_SEMANTICS:
        raise RuntimeError("TAQ checkpoint has incompatible label_semantics")
    try:
        model_cfg = taq_model_config_from_checkpoint(payload)
    except ValueError as exc:
        raise RuntimeError(f"TAQ checkpoint has incompatible baseline metadata: {exc}") from exc
    model = TimeAwareQualityGatedGRU(**model_cfg).to(device)
    state_dict = payload.get("model_state_dict") or payload.get("state_dict")
    if state_dict is None:
        raise RuntimeError("TAQ checkpoint has no model_state_dict/state_dict")
    cleaned = {str(k)[7:] if str(k).startswith("module.") else str(k): value for k, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    norm = payload.get("input_norm")
    if not isinstance(norm, dict) or "mean" not in norm or "std" not in norm:
        raise RuntimeError("TAQ v2 checkpoint is missing input_norm mean/std")
    model.set_input_norm_stats(norm["mean"], norm["std"])
    model.eval()
    return OnlineTAQVelocityEstimator(
        model,
        history_window_s=history_window_s,
        max_history_steps=max_history_steps,
        device=device,
    )


def load_legacy_gru_velocity_estimator(checkpoint_path: str | Path, *, device: str | Any = "cpu",
                                       history_window_s: float = 0.48, max_history_steps: int = 32) -> Any:
    """Load the TAQ-v2-trained 13-input ordinary GRU and its saved preprocessing."""
    root = _standalone_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from test11.speed_prediction.legacy_gru.online_gru_velocity import load_legacy_checkpoint
    from .legacy_gru_adapter import OnlineLegacyGRUEstimator

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Legacy GRU velocity checkpoint not found: {path}")
    model, checkpoint = load_legacy_checkpoint(path)
    return OnlineLegacyGRUEstimator(
        model, device=device, history_window_s=history_window_s,
        max_history_steps=max_history_steps,
        smooth_window=int(checkpoint["preprocessing"]["pos_smooth_window"]),
    )


def load_velocity_estimator(checkpoint_path: str | Path, *, backend: str = "legacy_gru",
                            device: str | Any = "cpu", history_window_s: float = 0.48,
                            max_history_steps: int = 32) -> Any:
    loaders = {"legacy_gru": load_legacy_gru_velocity_estimator, "taq_gru": load_taq_velocity_estimator}
    if backend not in loaders:
        raise ValueError(f"Unknown velocity backend: {backend}")
    return loaders[backend](checkpoint_path, device=device, history_window_s=history_window_s,
                            max_history_steps=max_history_steps)


def load_torchscript_policy(path: str | Path, *, device: str | Any = "cpu"):
    """Return a numpy-in/numpy-out wrapper for a scripted 28->3 policy."""
    import numpy as np
    import torch

    policy_path = Path(path).expanduser()
    if not policy_path.exists():
        raise FileNotFoundError(f"TorchScript decision policy not found: {policy_path}")
    module = torch.jit.load(str(policy_path), map_location=device)
    module.eval()

    def _policy(observation: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, 28)
        with torch.no_grad():
            output = module(tensor)
        if isinstance(output, dict):
            output = output.get("actions", output.get("action"))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output is None:
            raise RuntimeError("TorchScript policy returned no action tensor")
        return output.detach().cpu().numpy().reshape(-1)

    return _policy

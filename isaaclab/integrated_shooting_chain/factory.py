"""Convenience constructor for a real three-module chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .model_loaders import (
    make_lk_ostrack_tracker,
    make_lk_ostrack_tracker_with_checkpoint,
    load_velocity_estimator,
    load_torchscript_policy,
)
from .shooting_chain import CameraExtrinsics, ShootingChain, ShootingChainConfig


def build_chain(
    *,
    intrinsics_json: str | Path,
    rotation_base_from_camera: Any,
    camera_position_base_m: Any,
    velocity_checkpoint: str | Path,
    velocity_backend: str = "legacy_gru",
    ostrack_repo: str | Path | None = None,
    ostrack_checkpoint: str | Path | None = None,
    ostrack_config: str = "vitb_256_mae_ce_32x4_ep300",
    decision_policy: Callable | None = None,
    decision_torchscript: str | Path | None = None,
    tracker_config: Any | None = None,
    fine_tracker: Any | None = None,
    device: str = "cpu",
    history_window_s: float = 0.48,
    max_history_steps: int = 32,
    chain_config: ShootingChainConfig | None = None,
) -> ShootingChain:
    """Construct vision + ordinary GRU speed + decision policy as one chain.

    ``decision_policy`` is preferred for RL-Games because it can retain the
    player's recurrent state.  ``decision_torchscript`` is convenient for a
    stateless exported 28-to-3 policy. Exactly one must be supplied.
    """
    if (decision_policy is None) == (decision_torchscript is None):
        raise ValueError("Provide exactly one of decision_policy or decision_torchscript")

    from test_lk_ostrack.camera_geometry import CameraIntrinsics

    intrinsics = CameraIntrinsics.from_json(intrinsics_json)
    if fine_tracker is not None:
        if ostrack_repo is not None or ostrack_checkpoint is not None:
            raise ValueError("Pass either fine_tracker or OSTrack repo/checkpoint, not both")
        vision_tracker = make_lk_ostrack_tracker(
            intrinsics, config=tracker_config, fine_tracker=fine_tracker
        )
    else:
        vision_tracker = make_lk_ostrack_tracker_with_checkpoint(
            intrinsics,
            ostrack_repo=ostrack_repo,
            ostrack_checkpoint=ostrack_checkpoint,
            ostrack_config=ostrack_config,
            device=device,
            config=tracker_config,
        )
    velocity_estimator = load_velocity_estimator(
        velocity_checkpoint,
        backend=velocity_backend,
        device=device,
        history_window_s=history_window_s,
        max_history_steps=max_history_steps,
    )
    policy = decision_policy if decision_policy is not None else load_torchscript_policy(decision_torchscript, device=device)
    return ShootingChain(
        vision_tracker,
        velocity_estimator,
        policy,
        camera_extrinsics=CameraExtrinsics(
            rotation_base_from_camera=rotation_base_from_camera,
            camera_position_base_m=camera_position_base_m,
        ),
        config=chain_config,
    )

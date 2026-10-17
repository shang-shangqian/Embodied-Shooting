"""Vision -> velocity -> RL shooting-chain integration.

The package deliberately keeps the orchestration layer independent from IsaacLab.
IsaacLab-specific write-back helpers live in :mod:`isaac_adapter` and are imported
only when they are needed.
"""

from .shooting_chain import (
    CameraExtrinsics,
    DecisionCommand,
    RobotState,
    ShootingChain,
    ShootingChainConfig,
    build_policy_observation,
    tracker_quality_features,
)
from .factory import build_chain

__all__ = [
    "CameraExtrinsics",
    "DecisionCommand",
    "RobotState",
    "ShootingChain",
    "ShootingChainConfig",
    "build_policy_observation",
    "tracker_quality_features",
    "build_chain",
]

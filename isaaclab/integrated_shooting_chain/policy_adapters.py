"""Adapters for a caller-owned RL-Games player or a scripted policy."""

from __future__ import annotations

from typing import Any

import numpy as np


class RLGamesPlayerPolicy:
    """Turn an existing deterministic RL-Games player into a chain policy.

    The player must already be constructed with the test7_2_weight network
    configuration and checkpoint. This preserves RL-Games running observation
    normalization and recurrent state, which cannot be recovered from a raw
    ``.pth`` file by the numpy-only chain.
    """

    def __init__(self, player: Any, *, deterministic: bool = True) -> None:
        if not hasattr(player, "get_action"):
            raise TypeError("player must expose get_action(obs, is_deterministic=...)")
        self.player = player
        self.deterministic = bool(deterministic)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(28)
        # Vector-env players expect an explicit batch dimension. Non-batched
        # players add it in get_action(), so keep those inputs one-dimensional.
        if bool(getattr(self.player, "has_batch_dimension", False)):
            observation = observation[None, :]
        if hasattr(self.player, "obs_to_torch"):
            policy_observation = self.player.obs_to_torch(observation)
        else:
            import torch

            device = getattr(self.player, "device", "cpu")
            policy_observation = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            )
        action = self.player.get_action(
            policy_observation, is_deterministic=self.deterministic
        )
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        reset = getattr(self.player, "reset", None)
        if callable(reset):
            reset()
            return
        init_rnn = getattr(self.player, "init_rnn", None)
        if callable(init_rnn):
            self.player.states = init_rnn()

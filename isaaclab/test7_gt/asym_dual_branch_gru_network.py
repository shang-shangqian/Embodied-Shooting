from __future__ import annotations

import torch
import torch.nn as nn

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import NetworkBuilder


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AsymDualBranchGRUNet(nn.Module):
    """
    Actor:
        branch A: predicted target features
        branch B: proprio / robot features
        concat -> actor GRU -> actor head -> action
    Critic:
        branch A': GT target features
        branch B': proprio / robot features
        concat -> critic GRU -> critic head -> value
    """

    def __init__(
        self,
        actions_num: int,
        actor_target_dim: int = 9,
        actor_robot_dim: int = 19,
        critic_target_dim: int = 12,
        critic_robot_dim: int = 19,
        branch_hidden_dim: int = 64,
        rnn_hidden_dim: int = 128,
        head_hidden_dim: int = 64,
        fixed_sigma: bool = True,
    ):
        super().__init__()

        self.actions_num = int(actions_num)
        self.fixed_sigma = bool(fixed_sigma)

        # actor
        self.actions_num = int(actions_num)
        self.fixed_sigma = bool(fixed_sigma)

        self.actor_target_dim = int(actor_target_dim)
        self.actor_robot_dim = int(actor_robot_dim)
        self.critic_target_dim = int(critic_target_dim)
        self.critic_robot_dim = int(critic_robot_dim)

        self.branch_hidden_dim = int(branch_hidden_dim)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.rnn_num_layers = 1

        # actor
        self.actor_target_enc = MLPBlock(self.actor_target_dim, self.branch_hidden_dim)
        self.actor_robot_enc = MLPBlock(self.actor_robot_dim, self.branch_hidden_dim)
        self.actor_rnn = nn.GRU(
            input_size=self.branch_hidden_dim * 2,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )
        self.actor_head = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, self.head_hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(self.head_hidden_dim, self.actions_num)

        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(self.actions_num))
        else:
            self.sigma = nn.Linear(self.head_hidden_dim, self.actions_num)

        # critic
        self.critic_target_enc = MLPBlock(self.critic_target_dim, self.branch_hidden_dim)
        self.critic_robot_enc = MLPBlock(self.critic_robot_dim, self.branch_hidden_dim)
        self.critic_rnn = nn.GRU(
            input_size=self.branch_hidden_dim * 2,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )
        self.critic_head = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, self.head_hidden_dim),
            nn.ELU(),
            nn.Linear(self.head_hidden_dim, 1),
        )

        self._actor_h = None
        self._critic_h = None

    def is_rnn(self):
        return True

    def get_default_rnn_state(self):
        """
        rl-games 期望返回一个可迭代对象，每个元素都是一个 RNN hidden state tensor。
        对于当前双 GRU 结构，返回:
        [actor_h, critic_h]
        形状均为 [num_layers, batch, hidden_size]
        这里 batch 先给 1，rl-games 后续会按需要扩展/复制。
        """
        actor_h = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, dtype=torch.float32)
        critic_h = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, dtype=torch.float32)
        return [actor_h, critic_h]

    def reset_rnn_states(self):
        self._actor_h = None
        self._critic_h = None

    def _split_actor_obs(self, obs: torch.Tensor):
        # actor obs = 28 = target(9) + robot(19)
        actor_target = obs[:, :9]
        actor_robot = obs[:, 9:]
        return actor_target, actor_robot

    def _split_critic_states(self, states: torch.Tensor):
        # critic states = 31 = target(12) + robot(19)
        critic_target = states[:, :12]
        critic_robot = states[:, 12:]
        return critic_target, critic_robot

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        rnn_states = obs_dict.get("rnn_states", None)

        # 1. 切分出 Actor 和 Critic 各自专属的特征
        # obs 总维度 = 59 (28 维 policy_obs + 31 维 critic_obs)
        actor_full_obs = obs[:, :28]
        critic_full_obs = obs[:, 28:]

        # ------------------------------------------------------------
        # 1. 推断 batch_size / seq_len
        # rl_games 训练时通常传入展平后的 obs: [B*T, C]
        # ------------------------------------------------------------
        if obs.dim() != 2:
            raise RuntimeError(f"Expected obs [B*T, C], got {tuple(obs.shape)}")

        if rnn_states is not None:
            actor_h = rnn_states[0]
            critic_h = rnn_states[1]

            batch_size = actor_h.shape[1]
            seq_len = obs.shape[0] // batch_size
            if obs.shape[0] % batch_size != 0:
                raise RuntimeError(
                    f"[RNN shape error] obs.shape[0]={obs.shape[0]} is not divisible by batch_size={batch_size}"
                )

            if obs.shape[1] != 59:
                raise RuntimeError(
                    f"[Obs dim error] expected obs dim 59, got {obs.shape[1]}"
                )

            if actor_full_obs.shape[1] != 28:
                raise RuntimeError(
                    f"[Actor obs dim error] expected 28, got {actor_full_obs.shape[1]}"
                )

            if critic_full_obs.shape[1] != 31:
                raise RuntimeError(
                    f"[Critic obs dim error] expected 31, got {critic_full_obs.shape[1]}"
                )
        else:
            # 单步推理 fallback
            batch_size = obs.shape[0]
            seq_len = 1
            actor_h = torch.zeros(
                self.rnn_num_layers, batch_size, self.rnn_hidden_dim,
                device=obs.device, dtype=obs.dtype
            )
            critic_h = torch.zeros(
                self.rnn_num_layers, batch_size, self.rnn_hidden_dim,
                device=obs.device, dtype=obs.dtype
            )

        # ------------------------------------------------------------
        # 2. Actor branch
        # ------------------------------------------------------------
        actor_target, actor_robot = self._split_actor_obs(actor_full_obs)

        actor_feat = torch.cat(
            [
                self.actor_target_enc(actor_target),
                self.actor_robot_enc(actor_robot),
            ],
            dim=-1,
        )  # [B*T, 128]

        actor_feat_seq = actor_feat.contiguous().view(batch_size, seq_len, -1)  # [B, T, 128]
        actor_out_seq, actor_h_new = self.actor_rnn(actor_feat_seq, actor_h)
        actor_out_flat = actor_out_seq.contiguous().view(batch_size * seq_len, -1)

        actor_latent = self.actor_head(actor_out_flat)
        mu = self.mu(actor_latent)

        if self.fixed_sigma:
            sigma = self.sigma.expand(batch_size * seq_len, -1)
        else:
            sigma = self.sigma(actor_latent)

        # ------------------------------------------------------------
        # 3. Critic branch
        # ------------------------------------------------------------
        critic_target, critic_robot = self._split_critic_states(critic_full_obs)

        critic_feat = torch.cat(
            [
                self.critic_target_enc(critic_target),
                self.critic_robot_enc(critic_robot),
            ],
            dim=-1,
        )  # [B*T, 128]

        critic_feat_seq = critic_feat.contiguous().view(batch_size, seq_len, -1)  # [B, T, 128]
        critic_out_seq, critic_h_new = self.critic_rnn(critic_feat_seq, critic_h)
        critic_out_flat = critic_out_seq.contiguous().view(batch_size * seq_len, -1)

        value = self.critic_head(critic_out_flat)

        return mu, sigma, value, [actor_h_new, critic_h_new]


class AsymDualBranchGRUBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        actions_num = kwargs["actions_num"]
        net = AsymDualBranchGRUNet(
            actions_num=actions_num,
            actor_target_dim=self.params.get("actor_target_dim", 9),
            actor_robot_dim=self.params.get("actor_robot_dim", 19),
            critic_target_dim=self.params.get("critic_target_dim", 12),
            critic_robot_dim=self.params.get("critic_robot_dim", 19),
            branch_hidden_dim=self.params.get("branch_hidden_dim", 64),
            rnn_hidden_dim=self.params.get("rnn_hidden_dim", 128),
            head_hidden_dim=self.params.get("head_hidden_dim", 64),
            fixed_sigma=self.params.get("fixed_sigma", True),
        )
        return net


model_builder.register_network("asym_dual_branch_gru", AsymDualBranchGRUBuilder)
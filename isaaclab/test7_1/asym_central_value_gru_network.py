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


class AsymDualGRUNet(nn.Module):
    """
    Actor 和 Critic 都使用 GRU

    Actor:
        obs = actor_target_dim + actor_robot_dim

    Critic:
        states = critic_target_dim + critic_robot_dim

    返回:
        mu, sigma, value, [actor_h_new, critic_h_new]
    """

    def __init__(
        self,
        actions_num: int,
        actor_target_dim: int = 9,
        actor_robot_dim: int = 19,
        critic_target_dim: int = 12,
        critic_robot_dim: int = 19,
        branch_hidden_dim: int = 64,
        actor_rnn_hidden_dim: int = 128,
        critic_rnn_hidden_dim: int = 128,
        head_hidden_dim: int = 64,
        fixed_sigma: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.actions_num = int(actions_num)
        self.fixed_sigma = bool(fixed_sigma)

        self.actor_target_dim = int(actor_target_dim)
        self.actor_robot_dim = int(actor_robot_dim)
        self.critic_target_dim = int(critic_target_dim)
        self.critic_robot_dim = int(critic_robot_dim)

        self.branch_hidden_dim = int(branch_hidden_dim)
        self.actor_rnn_hidden_dim = int(actor_rnn_hidden_dim)
        self.critic_rnn_hidden_dim = int(critic_rnn_hidden_dim)
        self.head_hidden_dim = int(head_hidden_dim)

        self.rnn_num_layers = 1

        # -------------------------
        # Actor encoder + GRU
        # -------------------------
        self.actor_target_enc = MLPBlock(self.actor_target_dim, self.branch_hidden_dim)
        self.actor_robot_enc = MLPBlock(self.actor_robot_dim, self.branch_hidden_dim)

        self.actor_rnn = nn.GRU(
            input_size=self.branch_hidden_dim * 2,
            hidden_size=self.actor_rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )

        self.actor_head = nn.Sequential(
            nn.Linear(self.actor_rnn_hidden_dim, self.head_hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(self.head_hidden_dim, self.actions_num)

        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(self.actions_num))
        else:
            self.sigma = nn.Linear(self.head_hidden_dim, self.actions_num)

        # -------------------------
        # Critic encoder + GRU
        # -------------------------
        self.critic_target_enc = MLPBlock(self.critic_target_dim, self.branch_hidden_dim)
        self.critic_robot_enc = MLPBlock(self.critic_robot_dim, self.branch_hidden_dim)

        self.critic_rnn = nn.GRU(
            input_size=self.branch_hidden_dim * 2,
            hidden_size=self.critic_rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.critic_rnn_hidden_dim, self.head_hidden_dim),
            nn.ELU(),
            nn.Linear(self.head_hidden_dim, 1),
        )

    def is_rnn(self):
        return True

    def get_aux_loss(self):
        # rl-games 这里期望返回 dict[str, Tensor]
        # 当前网络没有额外辅助损失，返回空字典即可
        return {}


    def get_default_rnn_state(self):
        actor_h = torch.zeros(
            self.rnn_num_layers, 1, self.actor_rnn_hidden_dim, dtype=torch.float32
        )
        critic_h = torch.zeros(
            self.rnn_num_layers, 1, self.critic_rnn_hidden_dim, dtype=torch.float32
        )
        return [actor_h, critic_h]

    def _split_actor_obs(self, obs: torch.Tensor):
        actor_target = obs[:, : self.actor_target_dim]
        actor_robot = obs[:, self.actor_target_dim : self.actor_target_dim + self.actor_robot_dim]
        return actor_target, actor_robot

    def _split_critic_states(self, states: torch.Tensor):
        critic_target = states[:, : self.critic_target_dim]
        critic_robot = states[:, self.critic_target_dim : self.critic_target_dim + self.critic_robot_dim]
        return critic_target, critic_robot

    def forward(self, obs_dict):
        obs = obs_dict["obs"]   # [B*T, actor+critic]
        rnn_states = obs_dict.get("rnn_states", None)

        if obs.dim() != 2:
            raise RuntimeError(f"Expected obs [B*T, C], got {tuple(obs.shape)}")

        actor_obs_dim = self.actor_target_dim + self.actor_robot_dim
        critic_obs_dim = self.critic_target_dim + self.critic_robot_dim

        expected_total_obs_dim = (
            actor_obs_dim + critic_obs_dim
        )
        if obs.shape[1] != expected_total_obs_dim:
            raise RuntimeError(
                f"Expected total obs dim {expected_total_obs_dim}, got {obs.shape[1]}"
            )

        actor_full_obs = obs[:, :actor_obs_dim]
        critic_full_obs = obs[:, actor_obs_dim : actor_obs_dim + critic_obs_dim]

        # -------------------------
        # RNN state
        # rl-games 在 rollout 单步采样时，常会传入 shape=[num_layers, 1, hidden]
        # 的初始 state 模板；此时 obs.shape[0] 实际对应的是并行 env 数，
        # 不能把 hidden 的 batch 维 1 误当成真实 batch_size。
        # -------------------------
        if rnn_states is not None:
            if not isinstance(rnn_states, (list, tuple)) or len(rnn_states) != 2:
                raise RuntimeError(
                    f"Expected rnn_states as list/tuple of len 2, got {type(rnn_states)} "
                    f"with len={len(rnn_states) if isinstance(rnn_states, (list, tuple)) else 'N/A'}"
                )

            actor_h = rnn_states[0]
            critic_h = rnn_states[1]

            if actor_h.dim() != 3 or critic_h.dim() != 3:
                raise RuntimeError(
                    f"Expected actor_h/critic_h [num_layers, batch, hidden], "
                    f"got actor_h={tuple(actor_h.shape)}, critic_h={tuple(critic_h.shape)}"
                )

            if critic_h.shape[1] != actor_h.shape[1]:
                raise RuntimeError(
                    f"Actor/Critic hidden batch mismatch: actor={actor_h.shape[1]}, critic={critic_h.shape[1]}"
                )

            # 情况 A：
            # rollout 单步采样时，obs=[N, C]，hidden 可能还是 [L,1,H] 模板。
            # 这时应理解为 batch_size=N, seq_len=1，并将 hidden 扩展到 N。
            if actor_h.shape[1] == 1 and obs.shape[0] > 1:
                batch_size = obs.shape[0]
                seq_len = 1

                actor_h = actor_h.expand(actor_h.shape[0], batch_size, actor_h.shape[2]).contiguous()
                critic_h = critic_h.expand(critic_h.shape[0], batch_size, critic_h.shape[2]).contiguous()

            # 情况 B：
            # 训练阶段真正的 RNN batch，obs=[B*T, C]，hidden batch 维应等于 B
            else:
                batch_size = actor_h.shape[1]

                if batch_size <= 0:
                    raise RuntimeError(f"Invalid batch_size inferred from actor_h: {batch_size}")

                if obs.shape[0] % batch_size != 0:
                    raise RuntimeError(
                        f"obs first dim ({obs.shape[0]}) is not divisible by batch_size ({batch_size})"
                    )

                seq_len = obs.shape[0] // batch_size
        else:
            batch_size = obs.shape[0]
            seq_len = 1
            actor_h = torch.zeros(
                self.rnn_num_layers,
                batch_size,
                self.actor_rnn_hidden_dim,
                device=obs.device,
                dtype=obs.dtype,
            )
            critic_h = torch.zeros(
                self.rnn_num_layers,
                batch_size,
                self.critic_rnn_hidden_dim,
                device=obs.device,
                dtype=obs.dtype,
            )

        # -------------------------
        # Actor forward
        # -------------------------
        actor_target, actor_robot = self._split_actor_obs(actor_full_obs)
        actor_feat = torch.cat(
            [
                self.actor_target_enc(actor_target),
                self.actor_robot_enc(actor_robot),
            ],
            dim=-1,
        )  # [B*T, 2H]

        actor_feat_seq = actor_feat.contiguous().view(batch_size, seq_len, -1)
        actor_out_seq, actor_h_new = self.actor_rnn(actor_feat_seq, actor_h)
        actor_out_flat = actor_out_seq.contiguous().view(batch_size * seq_len, -1)

        actor_latent = self.actor_head(actor_out_flat)
        mu = self.mu(actor_latent)

        if self.fixed_sigma:
            sigma = self.sigma.unsqueeze(0).expand(batch_size * seq_len, -1)
        else:
            sigma = self.sigma(actor_latent)

        # -------------------------
        # Critic forward
        # -------------------------
        critic_target, critic_robot = self._split_critic_states(critic_full_obs)
        critic_feat = torch.cat(
            [
                self.critic_target_enc(critic_target),
                self.critic_robot_enc(critic_robot),
            ],
            dim=-1,
        )  # [B*T, 2H]

        critic_feat_seq = critic_feat.contiguous().view(batch_size, seq_len, -1)
        critic_out_seq, critic_h_new = self.critic_rnn(critic_feat_seq, critic_h)
        critic_out_flat = critic_out_seq.contiguous().view(batch_size * seq_len, -1)

        # 训练时默认不打印前向 shape/debug，避免刷屏。
        # 若后面确实需要排查网络维度问题，可临时改成 True。
        enable_network_debug = False

        if enable_network_debug and (not hasattr(self, "_debug_once")):
            print("[debug] obs.shape =", tuple(obs.shape))
            print("[debug] actor_full_obs.shape =", tuple(actor_full_obs.shape))
            print("[debug] critic_full_obs.shape =", tuple(critic_full_obs.shape))
            print("[debug] batch_size =", batch_size)
            print("[debug] seq_len =", seq_len)

            if rnn_states is not None:
                print("[debug] actor_h.shape =", tuple(actor_h.shape))
                print("[debug] critic_h.shape =", tuple(critic_h.shape))

            print("[debug] critic_feat.shape =", tuple(critic_feat.shape))
            print("[debug] critic_feat_seq.shape =", tuple(critic_feat_seq.shape))
            print("[debug] critic_out_seq.shape =", tuple(critic_out_seq.shape))
            print("[debug] critic_out_flat.shape =", tuple(critic_out_flat.shape))

            print("[debug] critic_full_obs finite =", torch.isfinite(critic_full_obs).all().item())
            print("[debug] critic_feat finite =", torch.isfinite(critic_feat).all().item())
            print("[debug] critic_out_seq finite =", torch.isfinite(critic_out_seq).all().item())
            print("[debug] critic_out_flat finite =", torch.isfinite(critic_out_flat).all().item())

            self._debug_once = True
        value = self.value_head(critic_out_flat)  # [B*T, 1]

        return mu, sigma, value, [actor_h_new, critic_h_new]


class AsymCentralValueGRUBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        actions_num = kwargs["actions_num"]
        net = AsymDualGRUNet(
            actions_num=actions_num,
            actor_target_dim=self.params.get("actor_target_dim", 9),
            actor_robot_dim=self.params.get("actor_robot_dim", 19),
            critic_target_dim=self.params.get("critic_target_dim", 12),
            critic_robot_dim=self.params.get("critic_robot_dim", 19),
            branch_hidden_dim=self.params.get("branch_hidden_dim", 64),
            actor_rnn_hidden_dim=self.params.get("actor_rnn_hidden_dim", 128),
            critic_rnn_hidden_dim=self.params.get("critic_rnn_hidden_dim", 128),
            head_hidden_dim=self.params.get("head_hidden_dim", 64),
            fixed_sigma=self.params.get("fixed_sigma", True),
        )
        return net


model_builder.register_network("asym_central_value_gru", AsymCentralValueGRUBuilder)
import torch
import torch.nn as nn


class VelocityPredictorGRU(nn.Module):
    """
    单个 GRU + 单个输出头 + 输入归一化

    输入 x: [B, T, 13]
      [
        refined_pos_abs(3),
        refined_pos_diff(3),
        robot_lin_vel(3),
        robot_ang_vel(3),
        dog_moving_flag(1),
      ]

    输出:
      [vx, vy, vz]
    """

    def __init__(self, input_dim=13, hidden_dim=256, num_layers=1, dropout=0.0):
        super().__init__()

        if input_dim != 13:
            raise ValueError(f"Expected input_dim=13, got {input_dim}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 3),
        )

        # 输入归一化统计量
        self.register_buffer(
            "input_mean",
            torch.zeros(1, 1, self.input_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "input_std",
            torch.ones(1, 1, self.input_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "norm_enabled",
            torch.tensor(True, dtype=torch.bool),
        )

    @torch.no_grad()
    def set_input_norm_stats(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        """
        mean/std shape: [input_dim] 或 [1, input_dim]
        """
        if mean.ndim == 2:
            mean = mean.squeeze(0)
        if std.ndim == 2:
            std = std.squeeze(0)

        if mean.ndim != 1 or mean.shape[0] != self.input_dim:
            raise RuntimeError(f"Expected mean shape [{self.input_dim}], got {tuple(mean.shape)}")
        if std.ndim != 1 or std.shape[0] != self.input_dim:
            raise RuntimeError(f"Expected std shape [{self.input_dim}], got {tuple(std.shape)}")

        std = std.clamp_min(eps)

        self.input_mean.copy_(
            mean.view(1, 1, -1).to(device=self.input_mean.device, dtype=self.input_mean.dtype)
        )
        self.input_std.copy_(
            std.view(1, 1, -1).to(device=self.input_std.device, dtype=self.input_std.dtype)
        )
        self.norm_enabled.fill_(True)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.norm_enabled.item()):
            return (x - self.input_mean) / self.input_std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(f"Expected x shape [B, T, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(f"Expected input feature dim {self.input_dim}, got {x.shape[-1]}")

        x = self.normalize_input(x)

        out, _ = self.gru(x)           # [B, T, H]
        last_hidden = out[:, -1, :]    # [B, H]
        pred = self.head(last_hidden)  # [B, 3]
        return pred

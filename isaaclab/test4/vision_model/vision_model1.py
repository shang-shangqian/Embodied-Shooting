# vision_model.py

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionModelOutput(NamedTuple):
    direction: torch.Tensor   # [B, 2] -> [yaw_cam, pitch_cam]
    velocity: torch.Tensor    # [B, 3] -> [vx_body, vy_body, vz_body]
    features: torch.Tensor    # [B, H]
    sequence_features: torch.Tensor  # [B, T, F]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ImageEncoder(nn.Module):
    """Per-frame CNN encoder for RGB images with flexible resolution."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()

        # Input: [B, 3, H, W]
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=8, stride=4),   # -> [B, 32, 31, 31]
            ConvBlock(32, 64, kernel_size=4, stride=2),            # -> [B, 64, 14, 14]
            ConvBlock(64, 128, kernel_size=3, stride=2),           # -> [B, 128, 6, 6]
            ConvBlock(128, 128, kernel_size=3, stride=1),          # -> [B, 128, 4, 4]
        )

        # Keep projection dim stable, matching 288x512 aspect ratio (~ 1 : 1.77)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 7))  # 【修改点】宽屏池化

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 7, 512),  # 【修改点】输入维度随之变为 128 * 28
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            feat: [B, feature_dim]
        """
        x = self.backbone(x)
        x = self.spatial_pool(x)
        x = self.proj(x)
        return x


class DirectionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VelocityHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TurretVisionNet(nn.Module):
    """
    图像 + 1维机器狗运动状态 的目标角度预测网络
    dog_moving_flag:
        0 -> static
        1 -> moving
    """
    def __init__(
        self,
        image_channels: int = 3,
        cnn_feature_dim: int = 256,
        proprio_dim: int = 1,   # 这里就表示 dog_moving_flag 的维度
        **kwargs,
    ):
        super().__init__()

        self.encoder = ImageEncoder(in_channels=image_channels, feature_dim=cnn_feature_dim)
        self.proprio_dim = int(proprio_dim)

        # 图像特征 256 + dog_moving_flag 1 = 257
        self.direction_head = DirectionHead(cnn_feature_dim + self.proprio_dim)

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> VisionModelOutput:
        # 兼容两种输入：
        # 1) [B, 3, H, W]
        # 2) [B, T, 3, H, W] -> 取最后一帧
        if images.ndim == 4:
            last_image = images
        elif images.ndim == 5:
            last_image = images[:, -1, :, :, :]
        else:
            raise RuntimeError(f"Unexpected images shape: {tuple(images.shape)}")

        batch_size = last_image.shape[0]
        device = last_image.device
        dtype = last_image.dtype

        # 图像编码
        features = self.encoder(last_image)   # [B, 256]

        # proprio 现在要求传入 dog_moving_flag: [B, 1]
        if proprio is None:
            raise RuntimeError(
                "proprio must be provided and should contain dog_moving_flag with shape [B, 1]."
            )

        if proprio.ndim != 2 or proprio.shape[0] != batch_size or proprio.shape[1] != self.proprio_dim:
            raise RuntimeError(
                f"Expected proprio shape [B, {self.proprio_dim}], got {tuple(proprio.shape)}"
            )

        proprio = proprio.to(device=device, dtype=features.dtype)

        # 拼接图像特征 + 运动状态
        fused_features = torch.cat([features, proprio], dim=-1)   # [B, 257]

        direction = self.direction_head(fused_features)            # [B, 2]

        dummy_velocity = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)

        return VisionModelOutput(
            direction=direction,
            velocity=dummy_velocity,
            features=fused_features,
            sequence_features=fused_features.unsqueeze(1),
        )


def build_vision_model(
    use_proprio: bool = True,
    proprio_dim: int = 1,
) -> TurretVisionNet:
    return TurretVisionNet(
        image_channels=3,
        cnn_feature_dim=256,
        proprio_dim=proprio_dim,
    )

def compute_vision_losses(
    pred_direction: torch.Tensor,
    pred_velocity: torch.Tensor,
    gt_direction: torch.Tensor,
    gt_velocity: torch.Tensor,
    direction_weight: float = 1.0,
    velocity_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    direction: [yaw_cam, pitch_cam]
    velocity:  [vx_body, vy_body, vz_body]
    """
    dir_loss = F.smooth_l1_loss(pred_direction, gt_direction)
    vel_loss = F.smooth_l1_loss(pred_velocity, gt_velocity)
    total_loss = direction_weight * dir_loss + velocity_weight * vel_loss

    return {
        "loss": total_loss,
        "direction_loss": dir_loss,
        "velocity_loss": vel_loss,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_vision_model(use_proprio=True, proprio_dim=1).to(device)

    dummy_images = torch.randn(8, 4, 3, 288, 512, device=device)
    dummy_dog_moving_flag = torch.randint(0, 2, (8, 1), device=device).float()

    out = model(dummy_images, proprio=dummy_dog_moving_flag)

    print("direction shape:", out.direction.shape)         # [8, 2]
    print("velocity shape:", out.velocity.shape)           # [8, 3]
    print("feature shape:", out.features.shape)            # [8, 257]
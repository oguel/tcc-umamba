"""U-Net compacta usada como baseline do TCC."""

from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvBlock(nn.Module):
    """Bloco convolucional duplo com GroupNorm, estável para batch pequeno."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net 2D para segmentação binária."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.bottleneck(self.pool(s3))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))
        return self.head(x)

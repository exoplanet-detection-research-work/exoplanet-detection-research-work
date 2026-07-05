"""Residual 1-D CNN local transit morphology encoder (Module 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from exodet.models.config import ModelArchitectureConfig

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor

__all__ = ["DepthwiseSeparableConv1d", "ResidualConvBlock", "LocalCNNEncoder"]


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    return nn.GELU()


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1-D convolution (efficient wide kernels)."""

    def __init__(self, channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(self, x: "Tensor") -> "Tensor":
        x = self.depthwise(x)
        return self.pointwise(x)


class ResidualConvBlock(nn.Module):
    """Residual block with optional depthwise-separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        activation: str,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        conv: nn.Module
        if depthwise and in_channels == out_channels:
            conv = DepthwiseSeparableConv1d(out_channels, kernel_size, padding)
        else:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.block = nn.Sequential(
            conv,
            nn.BatchNorm1d(out_channels),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.proj = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = _activation(activation)

    def forward(self, x: "Tensor") -> "Tensor":
        return self.act(self.block(x) + self.proj(x))


class LocalCNNEncoder(nn.Module):
    """Encodes the local folded view ``(B, 1, L)`` into an embedding vector.

    Uses stacked residual blocks with multi-scale kernels, batch
    normalisation, GELU, dropout, and adaptive global pooling — following
    AstroNet / ExoMiner local-branch design principles.
    """

    def __init__(self, config: ModelArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        channels = list(config.cnn_channels)
        kernels = list(config.cnn_kernel_sizes)
        depthwise = set(config.cnn_depthwise_stages)

        layers: list[nn.Module] = [
            nn.Conv1d(1, channels[0], kernels[0], padding=kernels[0] // 2, bias=False),
            nn.BatchNorm1d(channels[0]),
            _activation(config.activation),
        ]
        in_ch = channels[0]
        for stage, (out_ch, kernel) in enumerate(zip(channels, kernels, strict=True)):
            layers.append(
                ResidualConvBlock(
                    in_ch,
                    out_ch,
                    kernel,
                    config.cnn_dropout,
                    config.activation,
                    depthwise=stage in depthwise,
                )
            )
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch, config.embed_dim)

    def forward(self, local_view: "Tensor") -> tuple["Tensor", "Tensor"]:
        """Encodes local views.

        Args:
            local_view: Shape ``(B, L)`` or ``(B, 1, L)``.

        Returns:
            Tuple of (embedding ``(B, D)``, activation map ``(B, C, L')``).
        """
        if local_view.dim() == 2:
            local_view = local_view.unsqueeze(1)
        activations = self.encoder(local_view)
        pooled = self.pool(activations).squeeze(-1)
        return self.proj(pooled), activations

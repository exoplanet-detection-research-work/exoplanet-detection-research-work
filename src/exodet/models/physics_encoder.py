"""Physics feature MLP encoder (Module 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from exodet.models.config import ModelArchitectureConfig

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor

__all__ = ["PhysicsFeatureEncoder"]


class PhysicsFeatureEncoder(nn.Module):
    """Encodes engineered astrophysical features into a dense embedding.

    Injects explicit ephemeris, detection, shape, photometry, and quality
    priors via a small MLP with layer normalisation, GELU, and dropout.
    """

    def __init__(self, config: ModelArchitectureConfig) -> None:
        super().__init__()
        if config.n_physics_features <= 0:
            raise ValueError("n_physics_features must be > 0 for PhysicsFeatureEncoder.")
        layers: list[nn.Module] = []
        in_dim = config.n_physics_features
        act = nn.GELU() if config.activation == "gelu" else nn.ReLU(inplace=True)
        for hidden in config.physics_hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    act,
                    nn.Dropout(config.physics_dropout),
                ]
            )
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.embed_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: "Tensor") -> "Tensor":
        """Projects physics features to ``(B, embed_dim)``."""
        return self.mlp(features)

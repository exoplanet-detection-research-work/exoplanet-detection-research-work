"""Global phase-folded light curve transformer encoder (Module 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from exodet.models.config import ModelArchitectureConfig

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor

__all__ = ["TransformerEncoderBlock", "GlobalTransformerEncoder"]


def _activation_module(name: str) -> nn.Module:
    return nn.GELU() if name == "gelu" else nn.ReLU(inplace=True)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            _activation_module(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _forward_impl(self, x: "Tensor") -> tuple["Tensor", "Tensor"]:
        normed = self.norm1(x)
        attn_out, weights = self.attn(normed, normed, normed, need_weights=True)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights

    def forward(self, x: "Tensor", use_checkpoint: bool = False) -> tuple["Tensor", "Tensor"]:
        if use_checkpoint and self.training:
            out, weights = checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
            return out, weights
        return self._forward_impl(x)


class GlobalTransformerEncoder(nn.Module):
    """Encodes the global folded curve with a CLS-token transformer.

    Learns long-range periodic structure via learnable positional
    embeddings, pre-norm blocks, and a dedicated CLS readout — inspired
    by Vision Transformer and time-series transformer designs used in
    exoplanet detection literature.
    """

    def __init__(self, config: ModelArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        seq_len = config.global_bins
        self.input_proj = nn.Linear(1, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.embed_dim,
                    config.hidden_dim,
                    config.transformer_heads,
                    config.transformer_dropout,
                    config.activation,
                )
                for _ in range(config.transformer_depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, global_view: "Tensor") -> tuple["Tensor", "Tensor", "Tensor"]:
        """Encodes global views.

        Args:
            global_view: Shape ``(B, G)``.

        Returns:
            Tuple of (CLS embedding ``(B, D)``, sequence ``(B, G+1, D)``,
            CLS attention weights from the last layer ``(B, G+1)``).
        """
        batch = global_view.shape[0]
        tokens = self.input_proj(global_view.unsqueeze(-1))
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : global_view.shape[1] + 1]

        attn_weights = torch.zeros(batch, x.shape[1], device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, weights = block(x, use_checkpoint=self.config.transformer_checkpoint)
            attn_weights = weights[:, 0, :]

        x = self.norm(x)
        return x[:, 0], x, attn_weights

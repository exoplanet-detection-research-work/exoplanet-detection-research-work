"""Cross-attention multi-branch fusion (Module 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from exodet.models.config import ModelArchitectureConfig

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor

__all__ = ["CrossAttentionFusion"]


class CrossAttentionFusion(nn.Module):
    """Fuses CNN, transformer, and physics embeddings.

    Supports cross-attention over branch tokens, feature-wise gating,
    and residual fusion so complementary morphology, periodicity, and
    physics information are preserved.
    """

    def __init__(self, config: ModelArchitectureConfig, n_branches: int) -> None:
        super().__init__()
        self.config = config
        self.n_branches = n_branches
        dim = config.embed_dim

        self.branch_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            config.fusion_heads,
            dropout=config.fusion_dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * n_branches, n_branches),
            nn.Sigmoid(),
        )
        self.res_proj = nn.Linear(dim * n_branches, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.fusion_dropout)

    def forward(self, embeddings: "Tensor") -> tuple["Tensor", "Tensor"]:
        """Fuses branch embeddings.

        Args:
            embeddings: Stacked branch vectors ``(B, N, D)``.

        Returns:
            Tuple of (fused representation ``(B, D)``, attention weights).
        """
        batch, n_branches, dim = embeddings.shape
        normed = self.branch_norm(embeddings)

        if self.config.fusion_strategy == "residual":
            flat = embeddings.reshape(batch, n_branches * dim)
            fused = self.out_norm(self.res_proj(flat))
            return fused, torch.ones(batch, n_branches, device=embeddings.device)

        if self.config.fusion_strategy == "gated":
            flat = embeddings.reshape(batch, n_branches * dim)
            gate = self.gate(flat).unsqueeze(-1)
            weighted = (normed * gate).sum(dim=1)
            fused = self.out_norm(weighted + self.res_proj(flat))
            return self.dropout(fused), gate.squeeze(-1)

        # cross_attention (default): mean branch query attends to all branches
        query = self.query(normed.mean(dim=1, keepdim=True))
        attn_out, weights = self.cross_attn(query, normed, normed, need_weights=True)
        flat = embeddings.reshape(batch, n_branches * dim)
        gate = self.gate(flat).unsqueeze(-1)
        residual = (normed * gate).sum(dim=1)
        fused = self.out_norm(attn_out.squeeze(1) + residual + self.res_proj(flat))
        return self.dropout(fused), weights.squeeze(1)

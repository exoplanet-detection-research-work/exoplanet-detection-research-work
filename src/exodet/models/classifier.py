"""Classification and confidence heads plus full hybrid network (Modules 5–7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from exodet.models.cnn import LocalCNNEncoder
from exodet.models.config import CLASS_LABELS, ModelArchitectureConfig
from exodet.models.fusion import CrossAttentionFusion
from exodet.models.physics_encoder import PhysicsFeatureEncoder
from exodet.models.transformer import GlobalTransformerEncoder

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor

__all__ = [
    "ClassificationHead",
    "ConfidenceHead",
    "ForwardOutput",
    "HybridExoplanetNetwork",
]


@dataclass
class ForwardOutput:
    """Cached forward pass artefacts for interpretability and inference.

    Attributes:
        class_logits: Unnormalised class logits ``(B, C)``.
        class_probs: Softmax probabilities ``(B, C)``.
        confidence: Confidence scores ``(B,)`` in ``[0, 1]``.
        fused: Unified representation ``(B, D)``.
        local_embedding: CNN branch embedding or ``None``.
        global_embedding: Transformer CLS embedding or ``None``.
        physics_embedding: Physics branch embedding or ``None``.
        cnn_activations: Last CNN feature map or ``None``.
        cls_attention: CLS self-attention weights or ``None``.
        fusion_attention: Fusion attention / gate weights or ``None``.
    """

    class_logits: "Tensor"
    class_probs: "Tensor"
    confidence: "Tensor"
    fused: "Tensor"
    local_embedding: "Tensor | None" = None
    global_embedding: "Tensor | None" = None
    physics_embedding: "Tensor | None" = None
    cnn_activations: "Tensor | None" = None
    cls_attention: "Tensor | None" = None
    fusion_attention: "Tensor | None" = None
    extras: dict[str, Any] = field(default_factory=dict)


class ClassificationHead(nn.Module):
    """Multi-class softmax head (cross-entropy compatible)."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, fused: "Tensor") -> "Tensor":
        return self.net(fused)


class ConfidenceHead(nn.Module):
    """Predicts scalar confidence for later temperature scaling / calibration."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused: "Tensor") -> "Tensor":
        return torch.sigmoid(self.net(fused).squeeze(-1))


class HybridExoplanetNetwork(nn.Module):
    """Hybrid CNN + Transformer + physics exoplanet classifier.

    Exposes ``forward``, ``forward_features``, ``extract_features``,
    ``predict``, and ``predict_proba`` without redundant tensor
    recomputation when a forward cache is populated.
    """

    def __init__(self, config: ModelArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self._cache: ForwardOutput | None = None

        self.local_cnn = LocalCNNEncoder(config) if config.use_cnn else None
        self.global_transformer = (
            GlobalTransformerEncoder(config) if config.use_transformer else None
        )
        self.physics_encoder = (
            PhysicsFeatureEncoder(config)
            if config.use_physics and config.n_physics_features > 0
            else None
        )

        n_branches = sum(
            [
                config.use_cnn,
                config.use_transformer,
                config.use_physics and config.n_physics_features > 0,
            ]
        )
        if n_branches == 0:
            raise ValueError(f"No active branches for mode '{config.branch_mode}'.")

        if n_branches == 1:
            self.fusion: nn.Module | None = None
            self.single_proj = nn.Identity()
        else:
            self.fusion = CrossAttentionFusion(config, n_branches)
            self.single_proj = None

        self.classifier = ClassificationHead(
            config.embed_dim, config.num_classes, config.classifier_dropout
        )
        self.confidence_head = ConfidenceHead(
            config.embed_dim, config.confidence_hidden_dim, config.classifier_dropout
        )

        if config.compile_model:
            self._maybe_compile()

    def _maybe_compile(self) -> None:
        if not hasattr(torch, "compile"):
            return
        try:
            if self.local_cnn is not None:
                self.local_cnn = torch.compile(self.local_cnn)  # type: ignore[assignment]
            if self.global_transformer is not None:
                self.global_transformer = torch.compile(self.global_transformer)  # type: ignore[assignment]
        except Exception:
            pass

    def clear_cache(self) -> None:
        """Clears the forward cache."""
        self._cache = None

    def _encode_branches(
        self,
        global_view: "Tensor | None",
        local_view: "Tensor | None",
        physics: "Tensor | None",
    ) -> tuple[list["Tensor"], dict[str, "Tensor | None"]]:
        embeddings: list["Tensor"] = []
        artefacts: dict[str, "Tensor | None"] = {
            "local_embedding": None,
            "global_embedding": None,
            "physics_embedding": None,
            "cnn_activations": None,
            "cls_attention": None,
        }

        if self.local_cnn is not None:
            if local_view is None:
                raise ValueError("local_view required for CNN branch.")
            local_emb, acts = self.local_cnn(local_view)
            embeddings.append(local_emb)
            artefacts["local_embedding"] = local_emb
            artefacts["cnn_activations"] = acts

        if self.global_transformer is not None:
            if global_view is None:
                raise ValueError("global_view required for transformer branch.")
            cls_emb, _seq, cls_attn = self.global_transformer(global_view)
            embeddings.append(cls_emb)
            artefacts["global_embedding"] = cls_emb
            artefacts["cls_attention"] = cls_attn

        if self.physics_encoder is not None:
            if physics is None:
                raise ValueError("physics features required for physics branch.")
            phys_emb = self.physics_encoder(physics)
            embeddings.append(phys_emb)
            artefacts["physics_embedding"] = phys_emb

        return embeddings, artefacts

    def _fuse(self, embeddings: list["Tensor"]) -> tuple["Tensor", "Tensor | None"]:
        if len(embeddings) == 1:
            return embeddings[0], None
        stacked = torch.stack(embeddings, dim=1)
        assert self.fusion is not None
        return self.fusion(stacked)

    def forward(
        self,
        global_view: "Tensor | None" = None,
        local_view: "Tensor | None" = None,
        physics: "Tensor | None" = None,
    ) -> ForwardOutput:
        """Full forward pass with cached artefacts.

        Args:
            global_view: ``(B, G)`` global folded view.
            local_view: ``(B, L)`` local view.
            physics: ``(B, F)`` physics features.

        Returns:
            :class:`ForwardOutput` with logits, probabilities, and embeddings.
        """
        embeddings, artefacts = self._encode_branches(global_view, local_view, physics)
        fused, fusion_attn = self._fuse(embeddings)
        class_logits = self.classifier(fused)
        class_probs = F.softmax(class_logits, dim=-1)
        confidence = self.confidence_head(fused)

        output = ForwardOutput(
            class_logits=class_logits,
            class_probs=class_probs,
            confidence=confidence,
            fused=fused,
            local_embedding=artefacts["local_embedding"],
            global_embedding=artefacts["global_embedding"],
            physics_embedding=artefacts["physics_embedding"],
            cnn_activations=artefacts["cnn_activations"],
            cls_attention=artefacts["cls_attention"],
            fusion_attention=fusion_attn,
            extras={"branch_mode": self.config.branch_mode},
        )
        self._cache = output
        return output

    def forward_features(self) -> ForwardOutput:
        """Returns cached forward output; raises if :meth:`forward` was not run."""
        if self._cache is None:
            raise RuntimeError("forward_features() called before forward().")
        return self._cache

    def extract_features(self) -> dict[str, "Tensor | None"]:
        """Returns embedding dict without recomputing (uses cache)."""
        cached = self.forward_features()
        return {
            "fused": cached.fused,
            "local_embedding": cached.local_embedding,
            "global_embedding": cached.global_embedding,
            "physics_embedding": cached.physics_embedding,
            "confidence": cached.confidence,
        }

    def predict(self, global_view: "Tensor | None" = None, **kwargs: "Tensor | None") -> "Tensor":
        """Predicts class indices via argmax."""
        out = self.forward(global_view=global_view, **kwargs)
        return out.class_logits.argmax(dim=-1)

    def predict_proba(
        self,
        global_view: "Tensor | None" = None,
        **kwargs: "Tensor | None",
    ) -> "Tensor":
        """Predicts class probabilities via softmax."""
        return self.forward(global_view=global_view, **kwargs).class_probs

    def trainer_logits(self, output: ForwardOutput) -> "Tensor":
        """Maps multi-class logits to the format expected by the trainer.

        For ``binary_transit`` mode (default), returns a single logit per
        sample compatible with BCE training (transit vs rest). For
        ``multiclass`` mode, returns full ``(B, C)`` logits.
        """
        logits = output.class_logits
        if self.config.trainer_output == "multiclass":
            return logits
        if logits.shape[-1] == 2:
            return logits[:, 0] - logits[:, 1]
        rest = torch.logsumexp(logits[:, 1:], dim=-1)
        return logits[:, 0] - rest

    @staticmethod
    def class_names(num_classes: int) -> tuple[str, ...]:
        """Returns human-readable class names up to ``num_classes``."""
        if num_classes <= len(CLASS_LABELS):
            return CLASS_LABELS[:num_classes]
        extra = tuple(f"class_{i}" for i in range(len(CLASS_LABELS), num_classes))
        return CLASS_LABELS + extra

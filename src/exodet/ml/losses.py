"""Loss function registry (Module 3)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from exodet.registry import Registry

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["LOSS_FUNCTIONS", "build_loss"]

logger = logging.getLogger(__name__)

LOSS_FUNCTIONS: Registry[Callable[..., "torch.nn.Module"]] = Registry("loss function")


def _require_torch():
    import torch

    return torch


@LOSS_FUNCTIONS.register("bce")
def _bce(**_: object) -> "torch.nn.Module":
    torch = _require_torch()
    return torch.nn.BCEWithLogitsLoss()


@LOSS_FUNCTIONS.register("weighted_bce")
def _weighted_bce(pos_weight: float = 1.0, **_: object) -> "torch.nn.Module":
    torch = _require_torch()
    weight = torch.tensor([float(pos_weight)])
    return torch.nn.BCEWithLogitsLoss(pos_weight=weight)


@LOSS_FUNCTIONS.register("label_smooth_bce")
def _label_smooth_bce(smoothing: float = 0.1, **_: object) -> "torch.nn.Module":
    torch = _require_torch()

    class LabelSmoothBCE(torch.nn.Module):
        def __init__(self, smoothing: float) -> None:
            super().__init__()
            self.smoothing = smoothing
            self.bce = torch.nn.BCEWithLogitsLoss()

        def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
            smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
            return self.bce(logits, smooth)

    return LabelSmoothBCE(smoothing)


@LOSS_FUNCTIONS.register("focal")
def _focal(
    alpha: float = 0.25, gamma: float = 2.0, **_: object
) -> "torch.nn.Module":
    torch = _require_torch()

    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha: float, gamma: float) -> None:
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
            bce = self.bce(logits, targets)
            probs = torch.sigmoid(logits)
            pt = probs * targets + (1.0 - probs) * (1.0 - targets)
            focal_weight = self.alpha * (1.0 - pt) ** self.gamma
            return (focal_weight * bce).mean()

    return FocalLoss(alpha, gamma)


def sklearn_log_loss(
    y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    """Numpy log-loss for sklearn-backend training.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        sample_weight: Optional per-sample weights.

    Returns:
        Mean weighted log loss.
    """
    eps = 1e-7
    prob = np.clip(y_prob, eps, 1.0 - eps)
    loss = -(y_true * np.log(prob) + (1.0 - y_true) * np.log(1.0 - prob))
    if sample_weight is not None:
        return float(np.average(loss, weights=sample_weight))
    return float(loss.mean())


def build_loss(name: str, **params: object) -> object:
    """Instantiates a loss from the registry.

    Args:
        name: Registered loss name.
        **params: Loss hyperparameters.

    Returns:
        A ``torch.nn.Module`` loss (torch backend).
    """
    builder = LOSS_FUNCTIONS.get(name)
    return builder(**params)  # type: ignore[operator]

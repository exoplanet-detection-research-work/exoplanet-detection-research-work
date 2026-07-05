"""Probability calibration (Module 8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from exodet.ml.metrics import expected_calibration_error
from exodet.utils.io import write_json

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "TemperatureScaler",
    "fit_temperature_scaling",
    "reliability_bins",
    "plot_reliability_diagram",
]

logger = logging.getLogger(__name__)


@dataclass
class TemperatureScaler:
    """Post-hoc temperature scaling for binary logits.

    Attributes:
        temperature: Learned scalar temperature (``>= 1`` typical).
    """

    temperature: float = 1.0

    def calibrate(self, logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Applies temperature scaling and returns probabilities."""
        scaled = logits / self.temperature
        return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float64)

    def save(self, path: Path) -> Path:
        """Persists temperature to JSON."""
        return write_json({"temperature": self.temperature}, path)

    @classmethod
    def load(cls, path: Path) -> TemperatureScaler:
        """Loads temperature from JSON."""
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(temperature=float(data["temperature"]))


def fit_temperature_scaling(
    logits: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    max_iter: int = 50,
    lr: float = 0.01,
) -> TemperatureScaler:
    """Fits temperature on a validation set via LBFGS.

    Args:
        logits: Unscaled model logits.
        labels: Binary ground-truth labels.
        max_iter: Optimisation iterations.
        lr: Learning rate.

    Returns:
        Fitted :class:`TemperatureScaler`.
    """
    import torch
    import torch.nn.functional as F

    logit_t = torch.tensor(logits, dtype=torch.float32, requires_grad=False)
    label_t = torch.tensor(labels, dtype=torch.float32)
    temp = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logit_t / temp, label_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(temp.detach().clamp(min=0.1))
    logger.info("Fitted temperature scaling: T=%.3f", temperature)
    return TemperatureScaler(temperature=temperature)


def reliability_bins(
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Computes reliability diagram bin statistics."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences: list[float] = []
    accuracies: list[float] = []
    counts: list[int] = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= low) & (probabilities < high)
        if not np.any(mask):
            confidences.append((low + high) / 2)
            accuracies.append(0.0)
            counts.append(0)
            continue
        confidences.append(float(probabilities[mask].mean()))
        accuracies.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))
    ece = expected_calibration_error(labels, probabilities, n_bins)
    return {
        "bin_confidence": confidences,
        "bin_accuracy": accuracies,
        "bin_counts": counts,
        "ece": [ece],
    }


def plot_reliability_diagram(
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    figure_dir: Path,
    name: str = "reliability",
    n_bins: int = 10,
) -> list[Path]:
    """Exports reliability diagram figure."""
    import matplotlib.pyplot as plt

    from exodet.visualization.style import apply_publication_style, save_figure

    apply_publication_style()
    stats = reliability_bins(labels, probabilities, n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.bar(
        stats["bin_confidence"],
        stats["bin_accuracy"],
        width=1.0 / n_bins,
        alpha=0.7,
        label="model",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability (ECE={stats['ece'][0]:.3f})")
    ax.legend()
    return save_figure(fig, figure_dir, name)

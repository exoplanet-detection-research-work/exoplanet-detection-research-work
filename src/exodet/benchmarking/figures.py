"""Publication-quality figure generation for benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.training.evaluation import ResearchEvaluator
from exodet.utils.io import ensure_dir

__all__ = [
    "PublicationFigures",
    "plot_ablation_summary",
    "plot_sensitivity_curves",
    "plot_learning_curve",
]


class PublicationFigures:
    """Wrapper around :class:`~exodet.training.evaluation.ResearchEvaluator`."""

    def __init__(self, figure_dir: Path, n_bins: int = 10) -> None:
        self.figure_dir = Path(figure_dir)
        ensure_dir(self.figure_dir)
        self.evaluator = ResearchEvaluator(self.figure_dir, n_bins=n_bins)

    def roc_pr_confusion(
        self,
        experiment_name: str,
        split: str,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        *,
        threshold: float = 0.5,
    ) -> list[str]:
        report = self.evaluator.evaluate(
            experiment_name, split, labels, probabilities, threshold=threshold
        )
        return list(report.figure_paths)

    def confusion_matrix(
        self,
        labels: npt.NDArray[np.int_],
        predictions: npt.NDArray[np.int_],
        *,
        name: str = "model",
    ) -> list[str]:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        from exodet.visualization.style import apply_publication_style, save_figure

        apply_publication_style()
        cm = confusion_matrix(labels, predictions)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix — {name}")
        paths = [str(p) for p in save_figure(fig, self.figure_dir, f"{name}_confusion")]
        plt.close(fig)
        return paths


def plot_ablation_summary(
    table: dict[str, dict[str, float]],
    figure_dir: Path,
    *,
    metric: str = "roc_auc",
) -> list[str]:
    """Bar chart comparing ablation architectures."""
    import matplotlib.pyplot as plt

    from exodet.visualization.style import apply_publication_style, save_figure

    apply_publication_style()
    ensure_dir(figure_dir)
    names = list(table.keys())
    values = [float(table[n].get(metric, float("nan"))) for n in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(names, values)
    ax.set_ylabel(metric)
    ax.set_title("Ablation summary")
    ax.tick_params(axis="x", rotation=35)
    paths = [str(p) for p in save_figure(fig, figure_dir, "ablation_summary")]
    plt.close(fig)
    return paths


def plot_sensitivity_curves(
    curves: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
    *,
    metric: str = "roc_auc",
) -> list[str]:
    """Plot performance vs perturbation level."""
    import matplotlib.pyplot as plt

    from exodet.visualization.style import apply_publication_style, save_figure

    apply_publication_style()
    ensure_dir(figure_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for perturbation, points in sorted(curves.items()):
        levels = [float(p["level"]) for p in points]
        scores = [float(p["metrics"].get(metric, float("nan"))) for p in points]
        ax.plot(levels, scores, marker="o", label=perturbation)
    ax.set_xlabel("Perturbation level")
    ax.set_ylabel(metric)
    ax.set_title("Sensitivity analysis")
    ax.legend(fontsize=8)
    paths = [str(p) for p in save_figure(fig, figure_dir, "sensitivity_curves")]
    plt.close(fig)
    return paths


def plot_learning_curve(
    history: dict[str, list[float]],
    figure_dir: Path,
    *,
    name: str = "model",
) -> list[str]:
    """Plot training/validation loss curves."""
    import matplotlib.pyplot as plt

    from exodet.visualization.style import apply_publication_style, save_figure

    apply_publication_style()
    ensure_dir(figure_dir)
    fig, ax = plt.subplots()
    for key, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Learning curve — {name}")
    ax.legend()
    paths = [str(p) for p in save_figure(fig, figure_dir, f"{name}_learning_curve")]
    plt.close(fig)
    return paths

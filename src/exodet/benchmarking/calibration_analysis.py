"""Calibration analysis for benchmarking reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.ml.metrics import expected_calibration_error
from exodet.training.calibration import plot_reliability_diagram, reliability_bins
from exodet.utils.io import ensure_dir, write_json

__all__ = ["CalibrationReport", "analyze_calibration", "maximum_calibration_error"]


def maximum_calibration_error(
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Maximum calibration error (MCE) across reliability bins."""
    labels = np.asarray(labels, dtype=np.int_)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    bins_raw = reliability_bins(labels, probabilities, n_bins)
    gaps = [
        abs(acc - conf)
        for acc, conf, count in zip(
            bins_raw["bin_accuracy"],
            bins_raw["bin_confidence"],
            bins_raw["bin_counts"],
            strict=True,
        )
        if count > 0
    ]
    return float(max(gaps)) if gaps else float("nan")


@dataclass
class CalibrationReport:
    """Calibration diagnostics for one model."""

    model_name: str
    ece: float
    mce: float
    brier_score: float
    n_bins: int
    bins: list[dict[str, float]]
    figure_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "n_bins": self.n_bins,
            "bins": self.bins,
            "figure_paths": self.figure_paths,
        }


def analyze_calibration(
    model_name: str,
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    figure_dir: Path,
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute ECE/MCE/Brier and export reliability + confidence histograms."""
    import matplotlib.pyplot as plt

    from exodet.visualization.style import apply_publication_style, save_figure

    apply_publication_style()
    ensure_dir(figure_dir)
    labels = np.asarray(labels, dtype=np.int_)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ece = expected_calibration_error(labels, probabilities, n_bins)
    mce = maximum_calibration_error(labels, probabilities, n_bins)
    brier = float(np.mean((probabilities - labels) ** 2))
    bins_raw = reliability_bins(labels, probabilities, n_bins)
    bins = [
        {
            "confidence": bins_raw["bin_confidence"][i],
            "accuracy": bins_raw["bin_accuracy"][i],
            "count": bins_raw["bin_counts"][i],
        }
        for i in range(len(bins_raw["bin_confidence"]))
    ]
    figure_paths: list[str] = []
    rel_paths = plot_reliability_diagram(
        labels, probabilities, figure_dir, name=f"{model_name}_reliability", n_bins=n_bins
    )
    figure_paths.extend(str(p) for p in rel_paths)

    fig, ax = plt.subplots()
    ax.hist(probabilities, bins=n_bins, range=(0.0, 1.0), alpha=0.8, edgecolor="black")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence histogram — {model_name}")
    figure_paths.extend(str(p) for p in save_figure(fig, figure_dir, f"{model_name}_confidence_hist"))
    plt.close(fig)

    report = CalibrationReport(
        model_name=model_name,
        ece=ece,
        mce=mce,
        brier_score=brier,
        n_bins=n_bins,
        bins=bins,
        figure_paths=figure_paths,
    )
    write_json(report.to_dict(), figure_dir / f"{model_name}_calibration.json")
    return report

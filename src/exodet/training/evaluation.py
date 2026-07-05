"""Advanced and scientific evaluation (Modules 9 & 12)."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from exodet.ml.metrics import compute_all_metrics, expected_calibration_error
from exodet.representation.containers import DatasetSample, RepresentationDataset
from exodet.training.calibration import plot_reliability_diagram
from exodet.training.curriculum import sample_snr
from exodet.utils.io import ensure_dir, write_json

__all__ = ["ResearchEvaluationReport", "ResearchEvaluator", "ScientificValidator"]

logger = logging.getLogger(__name__)


@dataclass
class ResearchEvaluationReport:
    """Extended evaluation report with stratified metrics and figure paths."""

    experiment_name: str
    split: str
    scores: dict[str, float]
    per_class: dict[str, float] = field(default_factory=dict)
    strata: dict[str, dict[str, float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    figure_paths: list[str] = field(default_factory=list)

    def save(self, path: Path) -> Path:
        return write_json(
            {
                "experiment_name": self.experiment_name,
                "split": self.split,
                "scores": self.scores,
                "per_class": self.per_class,
                "strata": self.strata,
                "extra": self.extra,
                "figure_paths": self.figure_paths,
            },
            path,
        )


class ResearchEvaluator:
    """Publication-quality evaluation with ROC/PR/calibration plots."""

    def __init__(self, figure_dir: Path, n_bins: int = 10) -> None:
        self.figure_dir = Path(figure_dir)
        ensure_dir(self.figure_dir)
        self.n_bins = n_bins

    def evaluate(
        self,
        experiment_name: str,
        split: str,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        metric_specs: tuple[Any, ...] = (),
        threshold: float = 0.5,
    ) -> ResearchEvaluationReport:
        """Computes metrics and exports diagnostic figures."""
        import matplotlib.pyplot as plt

        from exodet.visualization.style import apply_publication_style, save_figure

        apply_publication_style()
        scores, extra = compute_all_metrics(metric_specs, labels, probabilities, threshold)
        scores["ece"] = expected_calibration_error(labels, probabilities, self.n_bins)
        figure_paths: list[str] = []

        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        scores["roc_auc_curve"] = float(roc_auc)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC")
        ax.legend()
        figure_paths.extend(str(p) for p in save_figure(fig, self.figure_dir, f"{split}_roc"))
        plt.close(fig)

        prec, rec, _ = precision_recall_curve(labels, probabilities)
        pr_auc = auc(rec, prec)
        scores["pr_auc_curve"] = float(pr_auc)
        fig, ax = plt.subplots()
        ax.plot(rec, prec, label=f"AP={pr_auc:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall")
        ax.legend()
        figure_paths.extend(str(p) for p in save_figure(fig, self.figure_dir, f"{split}_pr"))
        plt.close(fig)

        cm = confusion_matrix(labels, (probabilities >= threshold).astype(int), labels=[0, 1])
        extra["confusion_matrix"] = cm.tolist()
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")
        figure_paths.extend(str(p) for p in save_figure(fig, self.figure_dir, f"{split}_confusion"))
        plt.close(fig)

        figure_paths.extend(
            str(p)
            for p in plot_reliability_diagram(
                labels, probabilities, self.figure_dir, f"{split}_reliability", self.n_bins
            )
        )

        return ResearchEvaluationReport(
            experiment_name=experiment_name,
            split=split,
            scores=scores,
            extra=extra,
            figure_paths=figure_paths,
        )


class ScientificValidator:
    """Stratified performance for astrophysical regimes (Module 12)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.short_period_threshold = float(self.config.get("short_period_days", 10.0))
        self.shallow_depth_threshold = float(self.config.get("shallow_depth", 0.002))
        self.noise_threshold = float(self.config.get("high_noise_rms", 1e-3))

    def _strata_for_sample(self, sample: DatasetSample) -> list[str]:
        tags: list[str] = []
        period = sample.candidate.period_days
        depth = sample.candidate.depth
        n_transits = sample.candidate.n_transits
        snr = sample_snr(sample)
        names = sample.feature_names
        rms = float(sample.features[names.index("global_rms")]) if "global_rms" in names else 0.0

        tags.append("short_period" if period < self.short_period_threshold else "long_period")
        tags.append("single_transit" if n_transits <= 1 else "multi_transit")
        tags.append("shallow" if depth < self.shallow_depth_threshold else "deep")
        tags.append("high_noise" if rms > self.noise_threshold else "quiet")
        tags.append("high_snr" if snr >= 7.0 else "low_snr")
        return tags

    def summarize(
        self,
        dataset: RepresentationDataset,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> dict[str, dict[str, float]]:
        """Per-stratum accuracy and counts."""
        strata_scores: dict[str, dict[str, float]] = {}
        preds = (probabilities >= threshold).astype(int)
        for index, sample in enumerate(dataset.samples):
            if sample.label < 0:
                continue
            for tag in self._strata_for_sample(sample):
                strata_scores.setdefault(tag, {"correct": 0.0, "total": 0.0, "pos_rate": 0.0})
                strata_scores[tag]["total"] += 1
                if preds[index] == labels[index]:
                    strata_scores[tag]["correct"] += 1
                if labels[index] == 1:
                    strata_scores[tag]["pos_rate"] += 1
        for tag, stats in strata_scores.items():
            total = max(stats["total"], 1.0)
            stats["accuracy"] = stats["correct"] / total
            stats["positive_fraction"] = stats["pos_rate"] / total
        return strata_scores

    def export_table(
        self,
        strata: dict[str, dict[str, float]],
        path: Path,
    ) -> Path:
        """Writes CSV summary table."""
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["stratum", "accuracy", "total", "positive_fraction"],
            )
            writer.writeheader()
            for name, stats in sorted(strata.items()):
                writer.writerow(
                    {
                        "stratum": name,
                        "accuracy": stats.get("accuracy", 0.0),
                        "total": stats.get("total", 0.0),
                        "positive_fraction": stats.get("positive_fraction", 0.0),
                    }
                )
        return path

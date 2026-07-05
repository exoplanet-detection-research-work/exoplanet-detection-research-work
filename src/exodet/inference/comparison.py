"""Multi-model comparison with statistical tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from exodet.config.schema import ExperimentConfig
from exodet.inference.pipeline import ScientificInferencePipeline
from exodet.inference.config import InferenceStageConfig
from exodet.representation.containers import RepresentationDataset
from exodet.utils.io import ensure_dir, write_json

__all__ = ["ModelComparisonReport", "ModelComparator"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelComparisonReport:
    """Comparison metrics and figure paths across models."""

    model_names: tuple[str, ...]
    metrics: dict[str, dict[str, float]]
    mcnemar: dict[str, Any]
    agreement_matrix: list[list[float]]
    figure_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_names": list(self.model_names),
            "metrics": self.metrics,
            "mcnemar": self.mcnemar,
            "agreement_matrix": self.agreement_matrix,
            "figure_paths": dict(self.figure_paths),
        }

    def save(self, path: Path | str) -> Path:
        return write_json(self.to_dict(), path)


class ModelComparator:
    """Compares multiple trained models on the same dataset."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        inference_settings: InferenceStageConfig,
        model_checkpoints: dict[str, str],
    ) -> None:
        self.experiment = experiment
        self.inference_settings = inference_settings
        self.model_checkpoints = model_checkpoints

    def compare(
        self,
        dataset: RepresentationDataset,
        output_dir: Path | str,
        threshold: float = 0.5,
    ) -> ModelComparisonReport:
        """Runs inference for each model and generates comparison artefacts."""
        out = Path(output_dir)
        ensure_dir(out)
        labels = np.array([s.label for s in dataset.samples], dtype=np.int_)
        mask = labels >= 0
        labels = labels[mask]

        predictions: dict[str, npt.NDArray[np.float64]] = {}
        for name, ckpt in self.model_checkpoints.items():
            settings = replace(self.inference_settings, checkpoint_path=ckpt)
            pipeline = ScientificInferencePipeline(self.experiment, settings)
            batch = pipeline.predict_batch(dataset)
            probs = np.array([r.probability for r in batch.results], dtype=np.float64)
            predictions[name] = probs[mask] if len(probs) == len(mask) else probs

        model_names = tuple(predictions.keys())
        metrics: dict[str, dict[str, float]] = {}
        for name, probs in predictions.items():
            preds = (probs >= threshold).astype(np.int_)
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            metrics[name] = {
                "accuracy": float((tp + tn) / max(len(labels), 1)),
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            }

        agreement = self._agreement_matrix(predictions, threshold)
        mcnemar = self._mcnemar(predictions, labels, threshold)
        figures = self._figures(predictions, labels, out)

        report = ModelComparisonReport(
            model_names=model_names,
            metrics=metrics,
            mcnemar=mcnemar,
            agreement_matrix=agreement,
            figure_paths=figures,
        )
        report.save(out / "model_comparison.json")
        return report

    def _agreement_matrix(
        self,
        predictions: dict[str, npt.NDArray[np.float64]],
        threshold: float,
    ) -> list[list[float]]:
        names = list(predictions.keys())
        n = len(names)
        matrix = [[0.0] * n for _ in range(n)]
        hard = {name: (pred >= threshold).astype(np.int_) for name, pred in predictions.items()}
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                matrix[i][j] = float(np.mean(hard[ni] == hard[nj]))
        return matrix

    def _mcnemar(
        self,
        predictions: dict[str, npt.NDArray[np.float64]],
        labels: npt.NDArray[np.int_],
        threshold: float,
    ) -> dict[str, Any]:
        if len(predictions) < 2:
            return {}
        names = list(predictions.keys())
        a_name, b_name = names[0], names[1]
        a = (predictions[a_name] >= threshold).astype(np.int_)
        b = (predictions[b_name] >= threshold).astype(np.int_)
        truth = labels.astype(np.int_)
        a_correct = a == truth
        b_correct = b == truth
        b_only = int(np.sum(a_correct & ~b_correct))
        c_only = int(np.sum(~a_correct & b_correct))
        if b_only + c_only == 0:
            stat, p_value = 0.0, 1.0
        else:
            stat = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
            p_value = float(1.0 - stats.chi2.cdf(stat, 1))
        return {
            "model_a": a_name,
            "model_b": b_name,
            "b": b_only,
            "c": c_only,
            "statistic": float(stat),
            "p_value": p_value,
        }

    def _figures(
        self,
        predictions: dict[str, npt.NDArray[np.float64]],
        labels: npt.NDArray[np.int_],
        output_dir: Path,
    ) -> dict[str, str]:
        paths: dict[str, str] = {}
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, probs in predictions.items():
            fpr, tpr, _ = roc_curve(labels, probs)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC comparison")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = output_dir / "roc_comparison.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        paths["roc"] = str(roc_path)

        fig, ax = plt.subplots(figsize=(6, 5))
        for name, probs in predictions.items():
            precision, recall, _ = precision_recall_curve(labels, probs)
            ax.plot(recall, precision, label=name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR comparison")
        ax.legend(loc="best")
        fig.tight_layout()
        pr_path = output_dir / "pr_comparison.png"
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)
        paths["pr"] = str(pr_path)

        return paths

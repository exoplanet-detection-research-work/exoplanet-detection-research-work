"""Classification metrics framework (Module 6)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from exodet.evaluation.base import METRICS, BaseMetric, EvaluationReport
from exodet.registry import Registry

__all__ = [
    "CLASSIFICATION_METRICS",
    "compute_all_metrics",
    "compute_metric",
    "expected_calibration_error",
]

logger = logging.getLogger(__name__)

CLASSIFICATION_METRICS: Registry[type[BaseMetric]] = Registry("classification metric")


def _binary_labels(
    labels: npt.NDArray[np.int_], probabilities: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    mask = labels >= 0
    return labels[mask], probabilities[mask]


def expected_calibration_error(
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Computes expected calibration error (ECE).

    Args:
        labels: Binary labels.
        probabilities: Predicted positive-class probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Weighted mean absolute calibration gap.
    """
    labels, probabilities = _binary_labels(labels, probabilities)
    if len(labels) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= low) & (probabilities < high)
        if not np.any(mask):
            continue
        acc = labels[mask].mean()
        conf = probabilities[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


@CLASSIFICATION_METRICS.register("accuracy")
@METRICS.register("accuracy")
class AccuracyMetric(BaseMetric):
    """Classification accuracy."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(labels) == 0:
            return float("nan")
        preds = (probabilities >= threshold).astype(int)
        return float(accuracy_score(labels, preds))


@CLASSIFICATION_METRICS.register("precision")
@METRICS.register("precision")
class PrecisionMetric(BaseMetric):
    """Positive-class precision."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(labels) == 0:
            return float("nan")
        preds = (probabilities >= threshold).astype(int)
        return float(precision_score(labels, preds, zero_division=0))


@CLASSIFICATION_METRICS.register("recall")
@METRICS.register("recall")
class RecallMetric(BaseMetric):
    """Positive-class recall."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(labels) == 0:
            return float("nan")
        preds = (probabilities >= threshold).astype(int)
        return float(recall_score(labels, preds, zero_division=0))


@CLASSIFICATION_METRICS.register("f1")
@METRICS.register("f1")
class F1Metric(BaseMetric):
    """F1 score."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(labels) == 0:
            return float("nan")
        preds = (probabilities >= threshold).astype(int)
        return float(f1_score(labels, preds, zero_division=0))


@CLASSIFICATION_METRICS.register("roc_auc")
@METRICS.register("roc_auc")
class RocAucMetric(BaseMetric):
    """Area under the ROC curve."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        del threshold
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, probabilities))


@CLASSIFICATION_METRICS.register("pr_auc")
@METRICS.register("pr_auc")
class PrAucMetric(BaseMetric):
    """Area under the precision-recall curve (average precision)."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        del threshold
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(average_precision_score(labels, probabilities))


@CLASSIFICATION_METRICS.register("mcc")
@METRICS.register("mcc")
class MccMetric(BaseMetric):
    """Matthews correlation coefficient."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(labels) == 0:
            return float("nan")
        preds = (probabilities >= threshold).astype(int)
        return float(matthews_corrcoef(labels, preds))


@CLASSIFICATION_METRICS.register("balanced_accuracy")
@METRICS.register("balanced_accuracy")
class BalancedAccuracyMetric(BaseMetric):
    """Balanced accuracy."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(labels) == 0:
            return float("nan")
        preds = (probabilities >= threshold).astype(int)
        return float(balanced_accuracy_score(labels, preds))


@CLASSIFICATION_METRICS.register("calibration_error")
@METRICS.register("calibration_error")
class CalibrationErrorMetric(BaseMetric):
    """Expected calibration error."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        del threshold
        return expected_calibration_error(labels, probabilities, self.n_bins)


@CLASSIFICATION_METRICS.register("confusion_matrix")
@METRICS.register("confusion_matrix")
class ConfusionMatrixMetric(BaseMetric):
    """Stores confusion matrix counts in ``extra``; returns accuracy."""

    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        labels, probabilities = _binary_labels(labels, probabilities)
        if len(labels) == 0:
            return float("nan")
        preds = (probabilities >= threshold).astype(int)
        self._matrix = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
        return float(accuracy_score(labels, preds))


def compute_metric(
    name: str,
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    threshold: float = 0.5,
    **params: object,
) -> float:
    """Computes one registered metric.

    Args:
        name: Metric registry name.
        labels: Ground-truth labels.
        probabilities: Predicted probabilities.
        threshold: Decision threshold.
        **params: Metric constructor params.

    Returns:
        Scalar metric value.
    """
    metric = CLASSIFICATION_METRICS.build(name, **params)
    return metric.compute(labels, probabilities, threshold=threshold)


def compute_all_metrics(
    metric_specs: tuple[Any, ...],
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    threshold: float = 0.5,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Computes all configured metrics.

    Args:
        metric_specs: Evaluation metric component configs.
        labels: Ground-truth labels.
        probabilities: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        Tuple of (scores dict, extra artifacts dict).
    """
    scores: dict[str, float] = {}
    extra: dict[str, Any] = {}
    for spec in metric_specs:
        metric = CLASSIFICATION_METRICS.build(spec.name, **spec.params)
        scores[spec.name] = metric.compute(labels, probabilities, threshold=threshold)
        if isinstance(metric, ConfusionMatrixMetric) and hasattr(metric, "_matrix"):
            extra["confusion_matrix"] = metric._matrix
    return scores, extra


def build_evaluation_report(
    experiment_name: str,
    split: str,
    metric_specs: tuple[Any, ...],
    labels: npt.NDArray[np.int_],
    probabilities: npt.NDArray[np.float64],
    threshold: float = 0.5,
) -> EvaluationReport:
    """Builds an :class:`~exodet.evaluation.base.EvaluationReport`.

    Args:
        experiment_name: Experiment identifier.
        split: Dataset split name.
        metric_specs: Metric component configs.
        labels: Ground-truth labels.
        probabilities: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        A complete evaluation report.
    """
    scores, extra = compute_all_metrics(
        metric_specs, labels, probabilities, threshold=threshold
    )
    return EvaluationReport(
        experiment_name=experiment_name,
        split=split,
        scores=scores,
        extra=extra,
    )

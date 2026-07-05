"""Abstract evaluation metrics and report container.

Metrics compare predicted probabilities against ground-truth labels.
Implementations (precision/recall, ROC AUC, average precision, ...)
register with :data:`METRICS` and are selected via the YAML
``evaluation.metrics`` list.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.registry import Registry
from exodet.utils.io import write_json

__all__ = ["BaseMetric", "EvaluationReport", "METRICS"]

METRICS: Registry["BaseMetric"] = Registry("metric")


class BaseMetric(abc.ABC):
    """Abstract scalar metric over predictions and labels."""

    @property
    def name(self) -> str:
        """Metric name used as the key in evaluation reports."""
        return type(self).__name__.lower()

    @abc.abstractmethod
    def compute(
        self,
        labels: npt.NDArray[np.int_],
        probabilities: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> float:
        """Computes the metric value.

        Args:
            labels: Ground-truth integer labels, shape ``(n_samples,)``.
            probabilities: Predicted positive-class probabilities,
                shape ``(n_samples,)``.
            threshold: Decision threshold for threshold-dependent
                metrics; ignored by ranking metrics such as AUC.

        Returns:
            The scalar metric value.
        """


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Results of evaluating a model on one dataset split.

    Attributes:
        experiment_name: Name of the evaluated experiment.
        split: Dataset split name (``"val"``, ``"test"``...).
        scores: Metric name to value mapping.
        extra: Additional artifacts (confusion matrix, per-target
            predictions summary, ...), JSON-serializable.
    """

    experiment_name: str
    split: str
    scores: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        """Writes the report to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            The written file path.
        """
        return write_json(
            {
                "experiment_name": self.experiment_name,
                "split": self.split,
                "scores": self.scores,
                "extra": self.extra,
            },
            path,
        )

"""Curriculum learning and class-imbalance handling (Modules 1–2)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from exodet.exceptions import PipelineError
from exodet.representation.containers import DatasetSample, RepresentationDataset

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.data import Sampler

__all__ = [
    "CurriculumScheduler",
    "ClassImbalanceHandler",
    "effective_number_weights",
    "sample_snr",
    "build_train_sampler",
]

logger = logging.getLogger(__name__)


def sample_snr(sample: DatasetSample) -> float:
    """Estimates detection SNR from candidate record or physics features."""
    snr = sample.candidate.snr
    if math.isfinite(snr) and snr > 0:
        return float(snr)
    names = sample.feature_names
    if "snr" in names:
        return float(sample.features[names.index("snr")])
    if "log_snr" in names:
        return float(10 ** sample.features[names.index("log_snr")])
    return 0.0


@dataclass
class CurriculumScheduler:
    """Progressive SNR curriculum: high → medium → low → edge cases.

    Attributes:
        enabled: Whether curriculum filtering is active.
        stages: Ordered list of stage dicts with ``name``, ``min_snr``,
            ``max_snr``, ``start_epoch``, ``end_epoch``.
        schedule: ``linear`` epoch progression or ``step`` per stage.
    """

    enabled: bool = False
    stages: list[dict[str, Any]] | None = None
    schedule: str = "step"

    def __post_init__(self) -> None:
        if self.stages is None:
            self.stages = [
                {"name": "high_snr", "min_snr": 12.0, "max_snr": 1e9, "epochs": 0.25},
                {"name": "medium_snr", "min_snr": 7.0, "max_snr": 12.0, "epochs": 0.25},
                {"name": "low_snr", "min_snr": 4.0, "max_snr": 7.0, "epochs": 0.25},
                {"name": "edge_cases", "min_snr": 0.0, "max_snr": 4.0, "epochs": 0.25},
            ]

    def active_stage(self, epoch: int, total_epochs: int) -> dict[str, Any]:
        """Returns the curriculum stage for ``epoch`` (1-based)."""
        if not self.enabled or not self.stages:
            return {"name": "all", "min_snr": 0.0, "max_snr": 1e9}
        progress = (epoch - 1) / max(total_epochs, 1)
        cumulative = 0.0
        for stage in self.stages:
            frac = float(stage.get("epochs", 0.25))
            cumulative += frac
            if progress <= cumulative or stage is self.stages[-1]:
                return stage
        return self.stages[-1]

    def allowed_indices(
        self,
        dataset: RepresentationDataset,
        epoch: int,
        total_epochs: int,
    ) -> np.ndarray:
        """Indices of samples permitted in the current curriculum stage."""
        stage = self.active_stage(epoch, total_epochs)
        min_snr = float(stage.get("min_snr", 0.0))
        max_snr = float(stage.get("max_snr", 1e9))
        indices = []
        for index, sample in enumerate(dataset.samples):
            if sample.label < 0:
                continue
            snr = sample_snr(sample)
            if min_snr <= snr < max_snr:
                indices.append(index)
        if not indices:
            return np.arange(len(dataset.samples))
        return np.array(indices, dtype=np.int64)


def effective_number_weights(
    class_counts: np.ndarray, beta: float = 0.9999
) -> np.ndarray:
    """Class weights from effective number of samples (Cui et al. 2019).

    Args:
        class_counts: Per-class sample counts.
        beta: Hyperparameter in ``(0, 1)``.

    Returns:
        Weights per class (unnormalised).
    """
    counts = np.maximum(class_counts.astype(np.float64), 1.0)
    effective = (1.0 - beta**counts) / (1.0 - beta)
    weights = 1.0 / effective
    return weights / weights.sum() * len(weights)


@dataclass
class ClassImbalanceHandler:
    """Weighted / balanced sampling and dynamic class weights.

    Attributes:
        strategy: ``none``, ``weighted_sampler``, ``balanced_batch``,
            ``effective_number``, or ``dynamic_weights``.
        beta: Beta for effective-number weighting.
        oversample_positive: Extra weight on transit class (label 1).
    """

    enabled: bool = False
    strategy: str = "none"
    beta: float = 0.9999
    oversample_positive: float = 1.0

    _STRATEGIES = frozenset(
        {"none", "weighted_sampler", "balanced_batch", "effective_number", "dynamic_weights"}
    )

    def __post_init__(self) -> None:
        if self.strategy not in self._STRATEGIES:
            raise PipelineError(
                f"Unknown imbalance strategy '{self.strategy}'. "
                f"Choose from {sorted(self._STRATEGIES)}."
            )

    def per_sample_weights(
        self,
        dataset: RepresentationDataset,
        epoch: int = 1,
    ) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler."""
        del epoch
        labels = np.array([s.label for s in dataset.samples], dtype=np.int64)
        weights = np.ones(len(labels), dtype=np.float64)
        if not self.enabled or self.strategy == "none":
            return weights

        labeled = labels >= 0
        if not labeled.any():
            return weights

        classes, counts = np.unique(labels[labeled], return_counts=True)
        if self.strategy == "effective_number":
            class_w = effective_number_weights(counts, self.beta)
        else:
            class_w = 1.0 / np.maximum(counts.astype(np.float64), 1.0)
            class_w /= class_w.sum() * len(class_w)

        class_map = {int(c): float(w) for c, w in zip(classes, class_w, strict=True)}
        for index, label in enumerate(labels):
            if label >= 0:
                weights[index] = class_map.get(int(label), 1.0)
        if self.oversample_positive > 1.0:
            weights[labels == 1] *= self.oversample_positive
        return weights

    def make_sampler(
        self,
        dataset: RepresentationDataset,
        epoch: int,
        num_samples: int | None = None,
        hard_weights: np.ndarray | None = None,
    ) -> Sampler[int] | None:
        """Builds a PyTorch sampler when imbalance handling is enabled."""
        if not self.enabled or self.strategy not in (
            "weighted_sampler",
            "balanced_batch",
            "effective_number",
            "dynamic_weights",
        ):
            return None
        import torch
        from torch.utils.data import WeightedRandomSampler

        weights = self.per_sample_weights(dataset, epoch)
        if hard_weights is not None:
            weights = weights * hard_weights
        weights = np.maximum(weights, 1e-8)
        n = num_samples or len(dataset)
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights.astype(np.float64)),
            num_samples=n,
            replacement=True,
        )


def build_train_sampler(
    dataset: RepresentationDataset,
    curriculum: CurriculumScheduler,
    imbalance: ClassImbalanceHandler,
    epoch: int,
    total_epochs: int,
    hard_weights: np.ndarray | None = None,
) -> tuple[Sampler[int] | None, np.ndarray]:
    """Combines curriculum filtering with imbalance sampling.

    Returns:
        Tuple of (optional sampler, allowed index array).
    """
    allowed = curriculum.allowed_indices(dataset, epoch, total_epochs)
    if curriculum.enabled:
        logger.debug(
            "Curriculum epoch %d stage %s: %d/%d samples",
            epoch,
            curriculum.active_stage(epoch, total_epochs).get("name"),
            len(allowed),
            len(dataset),
        )
    sampler = imbalance.make_sampler(dataset, epoch, hard_weights=hard_weights)
    return sampler, allowed

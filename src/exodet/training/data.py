"""Research-grade DataLoader construction with curriculum and sampling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from exodet.ml.data import MlBatch, collate_ml_batch
from exodet.representation.containers import RepresentationDataset
from exodet.training.augmentation import (
    AugmentedTorchDataset,
    TrainingAugmentationPipeline,
)
from exodet.training.config import ResearchTrainingConfig
from exodet.training.curriculum import ClassImbalanceHandler, CurriculumScheduler

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.data import DataLoader

__all__ = ["ResearchDataModule", "HardExampleTracker"]

logger = logging.getLogger(__name__)


class HardExampleTracker:
    """Tracks per-sample loss for hard-example mining (Module 4)."""

    def __init__(self, enabled: bool = False, boost: float = 2.0) -> None:
        self.enabled = enabled
        self.boost = boost
        self._losses: dict[str, float] = {}
        self._confidence: dict[str, float] = {}

    def update(
        self,
        sample_ids: tuple[str, ...],
        losses: list[float],
        confidences: list[float] | None = None,
    ) -> None:
        if not self.enabled:
            return
        for index, sid in enumerate(sample_ids):
            self._losses[sid] = float(losses[index])
            if confidences is not None:
                self._confidence[sid] = float(confidences[index])

    def weights_for_dataset(self, dataset: RepresentationDataset) -> np.ndarray:
        """Oversampling weights emphasising hard / low-confidence samples."""
        if not self.enabled or not self._losses:
            return np.ones(len(dataset), dtype=np.float64)
        weights = np.ones(len(dataset), dtype=np.float64)
        if not self._losses:
            return weights
        max_loss = max(self._losses.values()) or 1.0
        for index, sample in enumerate(dataset.samples):
            loss = self._losses.get(sample.sample_id, 0.0)
            conf = self._confidence.get(sample.sample_id, 1.0)
            hardness = loss / max_loss + (1.0 - conf)
            if hardness > 0.5:
                weights[index] = 1.0 + self.boost * hardness
        return weights


class ResearchDataModule:
    """DataModule with curriculum, imbalance sampling, and augmentation."""

    def __init__(
        self,
        train: RepresentationDataset,
        research: ResearchTrainingConfig,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        use_views: str = "both",
        epoch: int = 1,
        total_epochs: int = 50,
        hard_tracker: HardExampleTracker | None = None,
        seed: int = 0,
    ) -> None:
        self.train_data = train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_views = use_views
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.seed = seed

        self.curriculum = CurriculumScheduler(
            enabled=bool(research.curriculum.get("enabled", False)),
            stages=list(research.curriculum.get("stages", [])) or None,
            schedule=str(research.curriculum.get("schedule", "step")),
        )
        self.imbalance = ClassImbalanceHandler(
            enabled=bool(research.imbalance.get("enabled", False)),
            strategy=str(research.imbalance.get("strategy", "none")),
            beta=float(research.imbalance.get("beta", 0.9999)),
            oversample_positive=float(research.imbalance.get("oversample_positive", 1.0)),
        )
        self.augmentation = TrainingAugmentationPipeline(
            enabled=bool(research.augmentation.get("enabled", False)),
            probability=float(research.augmentation.get("probability", 0.5)),
            steps=list(research.augmentation.get("steps", [])),
            seed=seed,
        )
        self.hard_tracker = hard_tracker or HardExampleTracker(
            enabled=bool(research.hard_mining.get("enabled", False)),
            boost=float(research.hard_mining.get("boost", 2.0)),
        )

    def train_dataloader(self) -> DataLoader:
        from torch.utils.data import DataLoader

        hard_w = self.hard_tracker.weights_for_dataset(self.train_data)
        sampler, allowed = __import__(
            "exodet.training.curriculum", fromlist=["build_train_sampler"]
        ).build_train_sampler(
            self.train_data,
            self.curriculum,
            self.imbalance,
            self.epoch,
            self.total_epochs,
            hard_weights=hard_w,
        )
        dataset: Any = AugmentedTorchDataset(
            self.train_data,
            self.augmentation,
            allowed_indices=allowed,
            seed=self.seed + self.epoch,
        )
        use_views = self.use_views

        def _collate(batch: list[dict[str, object]]) -> MlBatch:
            return collate_ml_batch(batch, use_views=use_views)

        shuffle = sampler is None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate,
        )

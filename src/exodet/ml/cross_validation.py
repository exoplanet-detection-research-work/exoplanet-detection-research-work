"""Cross-validation utilities (Module 11)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from exodet.exceptions import PipelineError
from exodet.ml.trainer import SupervisedTrainer
from exodet.representation.containers import RepresentationDataset
from exodet.training.base import TrainingResult
from exodet.utils.io import write_json

__all__ = ["CvSplit", "CrossValidationRunner"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CvSplit:
    """One cross-validation fold.

    Attributes:
        fold_index: Zero-based fold number.
        train: Training subset.
        validation: Validation subset (may be empty for nested outer folds).
        test: Optional held-out test subset.
        groups: Group ids used for group/stratified splits.
    """

    fold_index: int
    train: RepresentationDataset
    validation: RepresentationDataset
    test: RepresentationDataset | None = None
    groups: tuple[str, ...] = ()


class CrossValidationRunner:
    """K-fold, repeated, group, star-level, and nested cross-validation."""

    def __init__(self, cv_config: dict[str, Any]) -> None:
        """Initializes the CV runner from YAML config.

        Args:
            cv_config: ``training.trainer.params.cross_validation`` block.
        """
        self.enabled = bool(cv_config.get("enabled", False))
        self.strategy = str(cv_config.get("strategy", "kfold")).lower()
        self.n_splits = int(cv_config.get("n_splits", 5))
        self.n_repeats = int(cv_config.get("n_repeats", 1))
        self.nested = bool(cv_config.get("nested", False))
        self.inner_splits = int(cv_config.get("inner_splits", 3))
        self.shuffle = bool(cv_config.get("shuffle", True))
        self.random_state = int(cv_config.get("random_state", 42))
        self.group_key = str(cv_config.get("group_key", "target_id"))

    def splits(self, dataset: RepresentationDataset) -> Iterator[CvSplit]:
        """Yields train/validation folds over the dataset.

        Args:
            dataset: Full labeled dataset.

        Yields:
            :class:`CvSplit` instances.

        Raises:
            PipelineError: If the strategy is unknown or data is insufficient.
        """
        self._dataset = dataset
        if not self.enabled:
            yield CvSplit(
                fold_index=0,
                train=dataset,
                validation=RepresentationDataset([]),
            )
            return

        arrays = dataset.to_numpy()
        n_samples = len(dataset)
        if n_samples < self.n_splits:
            raise PipelineError(
                f"CV requires at least {self.n_splits} samples, got {n_samples}."
            )

        if self.strategy in ("group", "star", "star_level", "group_kfold"):
            yield from self._group_kfold(arrays)
        elif self.strategy == "repeated_kfold":
            yield from self._repeated_kfold(n_samples)
        elif self.strategy == "nested":
            yield from self._nested_kfold(arrays)
        elif self.strategy in ("kfold", "stratified"):
            yield from self._kfold(
                n_samples,
                stratified=self.strategy == "stratified",
                labels=arrays["labels"],
            )
        else:
            raise PipelineError(f"Unknown CV strategy '{self.strategy}'.")

    def _kfold(
        self, n_samples: int, stratified: bool = False, labels: np.ndarray | None = None
    ) -> Iterator[CvSplit]:
        from sklearn.model_selection import KFold, StratifiedKFold

        if stratified and labels is not None:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            split_iter = splitter.split(np.zeros(n_samples), labels)
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            split_iter = splitter.split(np.zeros(n_samples))

        for fold_index, (train_idx, val_idx) in enumerate(split_iter):
            yield CvSplit(
                fold_index=fold_index,
                train=self._subset_by_indices(train_idx),
                validation=self._subset_by_indices(val_idx),
            )

    def _repeated_kfold(self, n_samples: int) -> Iterator[CvSplit]:
        from sklearn.model_selection import RepeatedKFold

        splitter = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(n_samples))):
            yield CvSplit(
                fold_index=fold_index,
                train=self._subset_by_indices(train_idx),
                validation=self._subset_by_indices(val_idx),
            )

    def _group_kfold(self, arrays: dict[str, np.ndarray]) -> Iterator[CvSplit]:
        from sklearn.model_selection import GroupKFold

        dataset = self._dataset
        if self.group_key == "target_id":
            groups = arrays["target_ids"]
        else:
            groups = np.array(
                [s.meta.get(self.group_key, s.target_id) for s in dataset.samples],
                dtype=object,
            )

        unique_groups = np.unique(groups)
        if len(unique_groups) < self.n_splits:
            raise PipelineError(
                f"GroupKFold requires at least {self.n_splits} groups, "
                f"got {len(unique_groups)}."
            )

        splitter = GroupKFold(n_splits=self.n_splits)
        for fold_index, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(len(dataset)), groups=groups)
        ):
            yield CvSplit(
                fold_index=fold_index,
                train=self._subset_by_indices(train_idx),
                validation=self._subset_by_indices(val_idx),
                groups=tuple(str(g) for g in unique_groups),
            )

    def _nested_kfold(self, arrays: dict[str, np.ndarray]) -> Iterator[CvSplit]:
        from sklearn.model_selection import KFold

        dataset = self._dataset
        outer = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        for fold_index, (train_val_idx, test_idx) in enumerate(
            outer.split(np.zeros(len(dataset)))
        ):
            train_val = self._subset_by_indices(train_val_idx)
            inner_runner = CrossValidationRunner(
                {
                    "enabled": True,
                    "strategy": "kfold",
                    "n_splits": self.inner_splits,
                    "shuffle": self.shuffle,
                    "random_state": self.random_state,
                }
            )
            inner = next(inner_runner.splits(train_val))
            yield CvSplit(
                fold_index=fold_index,
                train=inner.train,
                validation=inner.validation,
                test=self._subset_by_indices(test_idx),
            )

    def _subset_by_indices(self, indices: np.ndarray | list[int]) -> RepresentationDataset:
        ds = self._dataset
        samples = [ds.samples[int(i)] for i in indices]
        return RepresentationDataset(samples, version=ds.version, meta=ds.meta)

    def run(
        self,
        dataset: RepresentationDataset,
        model_factory: Any,
        trainer: SupervisedTrainer,
        checkpoint_root: Path | None = None,
    ) -> list[TrainingResult]:
        """Runs full cross-validation training.

        Args:
            dataset: Full dataset.
            model_factory: Callable returning a fresh :class:`BaseModel`.
            trainer: Configured trainer.
            checkpoint_root: Optional per-fold checkpoint directory.

        Returns:
            One :class:`~exodet.training.base.TrainingResult` per fold.
        """
        self._dataset = dataset
        results: list[TrainingResult] = []
        fold_metrics: list[dict[str, float]] = []

        for split in self.splits(dataset):
            logger.info("CV fold %d: train=%d val=%d", split.fold_index, len(split.train), len(split.validation))
            model = model_factory()
            ckpt_dir = None
            if checkpoint_root is not None:
                ckpt_dir = checkpoint_root / f"fold_{split.fold_index:02d}"
            result = trainer.train(
                model=model,
                train_data=split.train,
                val_data=split.validation if len(split.validation) > 0 else None,
                checkpoint_dir=ckpt_dir,
            )
            results.append(result)
            if result.history.get("val_loss"):
                fold_metrics.append(
                    {"fold": split.fold_index, "val_loss": result.history["val_loss"][-1]}
                )

        if checkpoint_root is not None:
            write_json({"fold_metrics": fold_metrics}, checkpoint_root / "cv_summary.json")
        return results

"""RepresentationDataset → PyTorch DataLoader (training data module)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np

from exodet.exceptions import DataError, PipelineError
from exodet.representation.containers import RepresentationDataset

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch.utils.data import DataLoader, Dataset

__all__ = ["MlBatch", "RepresentationDataModule", "collate_ml_batch"]

logger = logging.getLogger(__name__)


def _require_torch():
    import torch

    return torch


@dataclass(frozen=True, slots=True)
class MlBatch:
    """One mini-batch of representation data.

    Attributes:
        global_view: Global views, shape ``(batch, bins_g)`` or ``None``.
        local_view: Local views, shape ``(batch, bins_l)`` or ``None``.
        features: Physics features, shape ``(batch, n_features)`` or ``None``.
        labels: Integer labels, shape ``(batch,)``.
        weights: Sample weights, shape ``(batch,)``.
        sample_ids: Sample identifiers.
        target_ids: Host star identifiers.
    """

    global_view: "torch.Tensor | None"
    local_view: "torch.Tensor | None"
    features: "torch.Tensor | None"
    labels: "torch.Tensor"
    weights: "torch.Tensor"
    sample_ids: tuple[str, ...]
    target_ids: tuple[str, ...]


class _RepresentationTorchDataset:
    """Thin PyTorch Dataset over a :class:`RepresentationDataset`."""

    def __init__(self, dataset: RepresentationDataset) -> None:
        self._dataset = dataset
        arrays = dataset.to_numpy()
        self._global = arrays["global_view"]
        self._local = arrays["local_view"]
        self._features = arrays["features"]
        self._labels = arrays["labels"]
        self._weights = arrays["weights"]
        self._sample_ids = arrays["sample_ids"]
        self._target_ids = arrays["target_ids"]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "global_view": self._global[index],
            "local_view": self._local[index],
            "features": self._features[index],
            "labels": self._labels[index],
            "weights": self._weights[index],
            "sample_id": str(self._sample_ids[index]),
            "target_id": str(self._target_ids[index]),
        }


def collate_ml_batch(
    items: list[dict[str, object]],
    use_views: str = "both",
) -> MlBatch:
    """Collates raw dataset items into an :class:`MlBatch`.

    Args:
        items: List of per-sample dicts from the dataset.
        use_views: Which inputs to include (``global``, ``local``, ``both``,
            ``features_only``).

    Returns:
        A batched tensor container on CPU (trainer moves to device).
    """
    torch = _require_torch()
    labels = torch.tensor([int(i["labels"]) for i in items], dtype=torch.long)
    weights = torch.tensor(
        [float(i["weights"]) for i in items], dtype=torch.float32
    )
    sample_ids = tuple(str(i["sample_id"]) for i in items)
    target_ids = tuple(str(i["target_id"]) for i in items)

    global_view = local_view = features = None
    if use_views in ("global", "both"):
        global_view = torch.tensor(
            np.stack([i["global_view"] for i in items]), dtype=torch.float32
        )
    if use_views in ("local", "both"):
        local_view = torch.tensor(
            np.stack([i["local_view"] for i in items]), dtype=torch.float32
        )
    if use_views in ("global", "local", "both", "features_only"):
        features = torch.tensor(
            np.stack([i["features"] for i in items]), dtype=torch.float32
        )

    return MlBatch(
        global_view=global_view,
        local_view=local_view,
        features=features,
        labels=labels,
        weights=weights,
        sample_ids=sample_ids,
        target_ids=target_ids,
    )


class RepresentationDataModule:
    """Builds train/val/test DataLoaders from saved representation splits."""

    def __init__(
        self,
        train: RepresentationDataset,
        validation: RepresentationDataset | None = None,
        test: RepresentationDataset | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        use_views: str = "both",
        shuffle_train: bool = True,
    ) -> None:
        """Initializes the data module.

        Args:
            train: Training split.
            validation: Optional validation split.
            test: Optional test split.
            batch_size: Mini-batch size.
            num_workers: DataLoader worker count.
            pin_memory: Pin memory for GPU transfer.
            use_views: Input channel selection.
            shuffle_train: Shuffle training batches.
        """
        if len(train) == 0:
            raise DataError("Training dataset is empty.")
        self.train_data = train
        self.val_data = validation
        self.test_data = test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_views = use_views
        self.shuffle_train = shuffle_train

    @classmethod
    def from_split_dir(
        cls,
        split_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        use_views: str = "both",
    ) -> "RepresentationDataModule":
        """Loads splits from ``{split_dir}/{train,validation,test}.npz``.

        Args:
            split_dir: Directory containing saved splits.
            batch_size: Mini-batch size.
            num_workers: DataLoader worker count.
            pin_memory: Pin memory for GPU transfer.
            use_views: Input channel selection.

        Returns:
            A data module with all available splits loaded.
        """
        from pathlib import Path

        root = Path(split_dir)
        train_path = root / "train.npz"
        if not train_path.is_file():
            raise PipelineError(f"Training split not found: {train_path}")

        train = RepresentationDataset.load(train_path)
        val = (
            RepresentationDataset.load(root / "validation.npz")
            if (root / "validation.npz").is_file()
            else None
        )
        test = (
            RepresentationDataset.load(root / "test.npz")
            if (root / "test.npz").is_file()
            else None
        )
        return cls(
            train=train,
            validation=val,
            test=test,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_views=use_views,
        )

    def _make_loader(
        self,
        dataset: RepresentationDataset,
        shuffle: bool,
    ) -> "DataLoader":
        torch = _require_torch()
        from torch.utils.data import DataLoader

        torch_ds: Dataset = _RepresentationTorchDataset(dataset)
        use_views = self.use_views

        def _collate(batch: list[dict[str, object]]) -> MlBatch:
            return collate_ml_batch(batch, use_views=use_views)

        return DataLoader(
            torch_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate,
        )

    def train_dataloader(self) -> "DataLoader":
        """Returns the training DataLoader."""
        return self._make_loader(self.train_data, shuffle=self.shuffle_train)

    def val_dataloader(self) -> "DataLoader | None":
        """Returns the validation DataLoader, or ``None``."""
        if self.val_data is None or len(self.val_data) == 0:
            return None
        return self._make_loader(self.val_data, shuffle=False)

    def test_dataloader(self) -> "DataLoader | None":
        """Returns the test DataLoader, or ``None``."""
        if self.test_data is None or len(self.test_data) == 0:
            return None
        return self._make_loader(self.test_data, shuffle=False)

    def labeled_indices(self, split: str = "train") -> Iterator[int]:
        """Yields indices of labeled samples in a split.

        Args:
            split: ``train``, ``validation``, or ``test``.

        Yields:
            Sample indices with ``label >= 0``.
        """
        ds = {"train": self.train_data, "validation": self.val_data, "test": self.test_data}[
            split
        ]
        if ds is None:
            return
        for index, sample in enumerate(ds.samples):
            if sample.label >= 0:
                yield index

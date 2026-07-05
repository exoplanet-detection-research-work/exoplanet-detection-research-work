"""Immutable value objects of the representation stage.

Flow: ``LightCurve + TransitCandidate`` → :class:`PhaseFoldedCurve` →
:class:`View` (global and local) + :class:`FeatureVector` →
:class:`DatasetSample` → :class:`RepresentationDataset`, which is what
the deep-learning stage consumes (as NumPy, PyTorch, Pandas, or Arrow).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from exodet.exceptions import DataError
from exodet.tce.candidate import TransitCandidate, _jsonify
from exodet.utils.io import ensure_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas
    import pyarrow
    import torch

__all__ = [
    "PhaseFoldedCurve",
    "View",
    "FeatureVector",
    "DatasetSample",
    "RepresentationDataset",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PhaseFoldedCurve:
    """A light curve folded on a candidate ephemeris.

    Attributes:
        candidate_id: The candidate whose ephemeris was used.
        target_id: Host target identifier.
        phase: Orbital phase in ``[-0.5, 0.5)``, sorted ascending, with
            the transit centered at phase 0.
        flux: Flux values aligned with ``phase``.
        flux_err: Optional flux uncertainties aligned with ``phase``.
        period_days: Fold period.
        epoch_days: Aligned mid-transit epoch actually used.
        duration_days: Transit duration (for window computations).
        epoch_correction_days: Offset applied by transit alignment.
        meta: Diagnostics (cadence counts, duplicates removed, ...).
        history: Provenance trail (light curve + folding stages).
    """

    candidate_id: str
    target_id: str
    phase: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]
    flux_err: npt.NDArray[np.float64] | None
    period_days: float
    epoch_days: float
    duration_days: float
    epoch_correction_days: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    history: tuple[str, ...] = ()

    @property
    def duty_cycle(self) -> float:
        """Transit duration as a fraction of the period."""
        return self.duration_days / self.period_days

    def __len__(self) -> int:
        return len(self.phase)


@dataclass(frozen=True, slots=True)
class View:
    """A binned, fixed-length representation of a folded curve.

    Attributes:
        kind: ``"global"`` or ``"local"``.
        values: Binned (and optionally normalized) flux, one per bin.
        bin_centers: Phase of each bin center.
        n_empty_bins: Bins with no data before interpolation.
        interpolation: Method used to fill empty bins.
        meta: Binning diagnostics (counts, normalization statistics).
    """

    kind: str
    values: npt.NDArray[np.float64]
    bin_centers: npt.NDArray[np.float64]
    n_empty_bins: int
    interpolation: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        """Number of bins in the view."""
        return len(self.values)

    @property
    def empty_fraction(self) -> float:
        """Fraction of bins that contained no data."""
        return self.n_empty_bins / self.n_bins

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """A named physics feature vector.

    Attributes:
        names: Feature names, aligned with ``values``.
        values: Feature values as float64.
    """

    names: tuple[str, ...]
    values: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise DataError(
                f"Feature names ({len(self.names)}) and values "
                f"({len(self.values)}) are misaligned."
            )

    def as_dict(self) -> dict[str, float]:
        """Returns the features as an ordered name → value mapping."""
        return {name: float(value) for name, value in zip(self.names, self.values)}

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One ML-ready sample: views + physics features + label.

    Attributes:
        sample_id: Unique identifier (candidate id + version).
        target_id: Host star identifier (used for star-level splits).
        candidate: The originating transit candidate (full record).
        global_view: Global orbital representation.
        local_view: Local transit representation.
        feature_names: Physics feature names.
        features: Physics feature values (possibly scaled).
        label: Integer class label (−1 when unlabeled).
        weight: Sample weight for training.
        dataset_version: Version tag of the generating configuration.
        meta: Additional metadata (alignment, view diagnostics, ...).
        history: Full provenance trail.
    """

    sample_id: str
    target_id: str
    candidate: TransitCandidate
    global_view: npt.NDArray[np.float64]
    local_view: npt.NDArray[np.float64]
    feature_names: tuple[str, ...]
    features: npt.NDArray[np.float64]
    label: int = -1
    weight: float = 1.0
    dataset_version: str = "v1"
    meta: dict[str, Any] = field(default_factory=dict)
    history: tuple[str, ...] = ()

    def with_features(
        self, features: npt.NDArray[np.float64], stage: str
    ) -> "DatasetSample":
        """Returns a copy with replaced feature values (e.g. scaled).

        Args:
            features: New feature values, same length and order.
            stage: Provenance entry describing the transformation.

        Returns:
            The updated sample; the original is unchanged.
        """
        if len(features) != len(self.features):
            raise DataError(
                f"Replacement features have length {len(features)}, "
                f"expected {len(self.features)}."
            )
        return replace(
            self,
            features=np.asarray(features, dtype=np.float64),
            history=(*self.history, stage),
        )

    def with_views(
        self,
        global_view: npt.NDArray[np.float64],
        local_view: npt.NDArray[np.float64],
        stage: str,
    ) -> "DatasetSample":
        """Returns a copy with replaced views (e.g. augmented).

        Args:
            global_view: New global view, same length.
            local_view: New local view, same length.
            stage: Provenance entry describing the transformation.

        Returns:
            The updated sample; the original is unchanged.
        """
        if len(global_view) != len(self.global_view) or len(local_view) != len(
            self.local_view
        ):
            raise DataError("Augmented views must preserve bin counts.")
        return replace(
            self,
            global_view=np.asarray(global_view, dtype=np.float64),
            local_view=np.asarray(local_view, dtype=np.float64),
            history=(*self.history, stage),
        )


class RepresentationDataset:
    """An ordered collection of :class:`DatasetSample` objects.

    Provides conversion to every downstream consumer format and
    lossless NPZ+JSON persistence.

    Attributes:
        samples: The samples, in insertion order.
        version: Dataset version tag.
        meta: Dataset-level metadata (split name, scaler stats path...).
    """

    def __init__(
        self,
        samples: list[DatasetSample],
        version: str = "v1",
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            samples: ML-ready samples.
            version: Dataset version tag.
            meta: Optional dataset-level metadata.
        """
        self.samples = list(samples)
        self.version = version
        self.meta = dict(meta or {})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names shared by all samples (empty when no samples)."""
        return self.samples[0].feature_names if self.samples else ()

    def _require_samples(self) -> None:
        if not self.samples:
            raise DataError("Dataset is empty; nothing to convert.")

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Stacks the dataset into NumPy arrays.

        Returns:
            A dictionary with keys ``global_view`` (n, bins_g),
            ``local_view`` (n, bins_l), ``features`` (n, f),
            ``labels`` (n,), ``weights`` (n,), ``sample_ids`` and
            ``target_ids`` (object arrays).
        """
        self._require_samples()
        return {
            "global_view": np.stack([s.global_view for s in self.samples]),
            "local_view": np.stack([s.local_view for s in self.samples]),
            "features": np.stack([s.features for s in self.samples]),
            "labels": np.array([s.label for s in self.samples], dtype=np.int64),
            "weights": np.array([s.weight for s in self.samples], dtype=np.float64),
            "sample_ids": np.array([s.sample_id for s in self.samples], dtype=object),
            "target_ids": np.array([s.target_id for s in self.samples], dtype=object),
        }

    def to_torch(self) -> dict[str, "torch.Tensor"]:
        """Converts the dataset to PyTorch tensors.

        Returns:
            A dictionary of float32 tensors (``labels`` as int64).

        Raises:
            DataError: If PyTorch is not installed.
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dep
            raise DataError(
                "PyTorch is required for to_torch(); install 'torch'."
            ) from exc
        arrays = self.to_numpy()
        return {
            "global_view": torch.from_numpy(
                arrays["global_view"].astype(np.float32)
            ),
            "local_view": torch.from_numpy(arrays["local_view"].astype(np.float32)),
            "features": torch.from_numpy(arrays["features"].astype(np.float32)),
            "labels": torch.from_numpy(arrays["labels"]),
            "weights": torch.from_numpy(arrays["weights"].astype(np.float32)),
        }

    def to_pandas(self) -> "pandas.DataFrame":
        """Converts the dataset to a pandas DataFrame.

        Scalar physics features become one column each; views are
        stored as per-row NumPy arrays in ``global_view``/``local_view``.

        Returns:
            The assembled DataFrame, one row per sample.
        """
        import pandas as pd

        self._require_samples()
        records: list[dict[str, Any]] = []
        for sample in self.samples:
            record: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "target_id": sample.target_id,
                "candidate_id": sample.candidate.candidate_id,
                "label": sample.label,
                "weight": sample.weight,
                "dataset_version": sample.dataset_version,
                "global_view": sample.global_view,
                "local_view": sample.local_view,
            }
            record.update(
                zip(sample.feature_names, (float(v) for v in sample.features))
            )
            records.append(record)
        return pd.DataFrame.from_records(records)

    def to_arrow(self) -> "pyarrow.Table":
        """Converts the dataset to an Apache Arrow table.

        Views become fixed-size list columns; features one column each.

        Returns:
            The Arrow table.

        Raises:
            DataError: If pyarrow is not installed.
        """
        try:
            import pyarrow as pa
        except ImportError as exc:  # pragma: no cover - optional dep
            raise DataError(
                "pyarrow is required for to_arrow(); install 'pyarrow'."
            ) from exc
        arrays = self.to_numpy()
        columns: dict[str, Any] = {
            "sample_id": pa.array([str(v) for v in arrays["sample_ids"]]),
            "target_id": pa.array([str(v) for v in arrays["target_ids"]]),
            "label": pa.array(arrays["labels"]),
            "weight": pa.array(arrays["weights"]),
            "global_view": pa.array(list(arrays["global_view"])),
            "local_view": pa.array(list(arrays["local_view"])),
        }
        for index, name in enumerate(self.feature_names):
            columns[f"feature_{name}"] = pa.array(arrays["features"][:, index])
        return pa.table(columns)

    def save(self, path: Path | str) -> Path:
        """Persists the dataset losslessly as compressed NPZ + JSON.

        Args:
            path: Destination ``.npz`` file (a ``.json`` sidecar with
                candidates and metadata is written next to it).

        Returns:
            The NPZ file path.
        """
        path = Path(path)
        ensure_dir(path.parent)
        arrays = self.to_numpy() if self.samples else {}
        np.savez_compressed(
            path,
            **{
                key: value
                for key, value in arrays.items()
                if key not in ("sample_ids", "target_ids")
            },
        )
        sidecar = {
            "version": self.version,
            "meta": _jsonify(self.meta),
            "feature_names": list(self.feature_names),
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "target_id": s.target_id,
                    "label": s.label,
                    "weight": s.weight,
                    "dataset_version": s.dataset_version,
                    "meta": _jsonify(s.meta),
                    "history": list(s.history),
                    "candidate": s.candidate.to_dict(),
                }
                for s in self.samples
            ],
        }
        json_path = path.with_suffix(".json")
        json_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Saved dataset (%d samples) to %s", len(self), path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "RepresentationDataset":
        """Loads a dataset written by :meth:`save`.

        Args:
            path: The ``.npz`` file path.

        Returns:
            The reconstructed dataset.

        Raises:
            DataError: If files are missing or inconsistent.
        """
        path = Path(path)
        json_path = path.with_suffix(".json")
        if not path.is_file() or not json_path.is_file():
            raise DataError(f"Dataset files not found: {path} / {json_path}")
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        records = sidecar["samples"]
        samples: list[DatasetSample] = []
        if records:
            with np.load(path, allow_pickle=False) as arrays:
                global_views = arrays["global_view"]
                local_views = arrays["local_view"]
                features = arrays["features"]
            if len(records) != len(global_views):
                raise DataError(
                    f"Dataset sidecar lists {len(records)} samples but NPZ "
                    f"holds {len(global_views)}."
                )
            names = tuple(sidecar["feature_names"])
            for index, record in enumerate(records):
                samples.append(
                    DatasetSample(
                        sample_id=record["sample_id"],
                        target_id=record["target_id"],
                        candidate=TransitCandidate.from_dict(record["candidate"]),
                        global_view=global_views[index],
                        local_view=local_views[index],
                        feature_names=names,
                        features=features[index],
                        label=int(record["label"]),
                        weight=float(record["weight"]),
                        dataset_version=record["dataset_version"],
                        meta=record["meta"],
                        history=tuple(record["history"]),
                    )
                )
        return cls(samples, version=sidecar["version"], meta=sidecar["meta"])

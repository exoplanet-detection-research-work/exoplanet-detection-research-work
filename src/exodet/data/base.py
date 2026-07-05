"""Core data container and abstract data interfaces.

Defines the :class:`LightCurve` value object exchanged between all
pipeline stages, plus the abstract base classes that concrete data
sources (e.g. MAST, Kaggle CSV archives) and datasets must implement.
Concrete implementations register themselves in :data:`DATA_SOURCES`
and :data:`DATASETS` to become addressable from YAML configs.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import numpy.typing as npt

from exodet.registry import Registry
from exodet.utils.validation import require_same_length

__all__ = [
    "LightCurve",
    "BaseDataSource",
    "BaseDataset",
    "DATA_SOURCES",
    "DATASETS",
]

logger = logging.getLogger(__name__)

DATA_SOURCES: Registry["BaseDataSource"] = Registry("data source")
DATASETS: Registry["BaseDataset"] = Registry("dataset")


@dataclass(slots=True)
class LightCurve:
    """A photometric time series for a single target.

    This is the canonical unit of data flowing through the pipeline.
    Preprocessing steps consume and produce ``LightCurve`` instances;
    provenance of applied transformations is accumulated in ``history``.

    Attributes:
        target_id: Mission catalog identifier (e.g. ``"KIC 8462852"``).
        time: Observation times in days (mission time system).
        flux: Normalized or raw flux values, same length as ``time``.
        flux_err: Optional 1-sigma flux uncertainties.
        label: Optional ground-truth class label (see ``constants``).
        mission: Lowercase mission key (e.g. ``"kepler"``).
        meta: Free-form metadata (stellar parameters, sector, quarter...).
        history: Names of preprocessing steps applied, in order.
    """

    target_id: str
    time: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]
    flux_err: npt.NDArray[np.float64] | None = None
    label: int | None = None
    mission: str = "kepler"
    meta: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validates array shapes on construction.

        Raises:
            DataError: If ``time``/``flux``/``flux_err`` lengths differ.
        """
        require_same_length(self.time, self.flux, ("time", "flux"))
        if self.flux_err is not None:
            require_same_length(self.flux, self.flux_err, ("flux", "flux_err"))

    def __len__(self) -> int:
        return len(self.time)

    def replace_flux(
        self,
        flux: npt.NDArray[np.float64],
        *,
        step_name: str,
        time: npt.NDArray[np.float64] | None = None,
        flux_err: npt.NDArray[np.float64] | None = None,
    ) -> "LightCurve":
        """Returns a copy with new flux (and optionally time) arrays.

        This is the canonical way for preprocessing steps to produce
        their output: the original curve is never mutated, and the
        step name is appended to the provenance history.

        Args:
            flux: New flux array.
            step_name: Name of the transformation, recorded in history.
            time: New time array; the existing one is kept if omitted.
            flux_err: New uncertainty array; dropped if omitted and the
                new arrays changed length, kept otherwise.

        Returns:
            A new :class:`LightCurve` with updated arrays and history.
        """
        new_time = self.time if time is None else time
        if flux_err is None and self.flux_err is not None:
            flux_err = self.flux_err if len(flux) == len(self.flux) else None
        return LightCurve(
            target_id=self.target_id,
            time=new_time,
            flux=flux,
            flux_err=flux_err,
            label=self.label,
            mission=self.mission,
            meta=dict(self.meta),
            history=[*self.history, step_name],
        )


class BaseDataSource(abc.ABC):
    """Abstract acquirer of raw data from an external archive.

    Implementations download (or locate) raw files for a set of targets
    and materialize them under a local directory. They must be
    idempotent: already-downloaded files are not fetched again.
    """

    @abc.abstractmethod
    def download(self, destination: Path) -> Path:
        """Fetches raw data into a local directory.

        Args:
            destination: Directory that will contain the raw files.

        Returns:
            The directory containing the downloaded data.

        Raises:
            DataError: If the download fails or integrity checks fail.
        """

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """Summarizes the source for logging and provenance records.

        Returns:
            A JSON-serializable description (archive name, query, ...).
        """


class BaseDataset(abc.ABC, Sequence[LightCurve]):
    """Abstract collection of labelled light curves.

    Implementations parse raw files into :class:`LightCurve` objects
    and expose them through the standard sequence protocol, which keeps
    them directly compatible with ML framework data loaders.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of light curves in the dataset."""

    @abc.abstractmethod
    def __getitem__(self, index: int) -> LightCurve:  # type: ignore[override]
        """Returns the light curve at ``index``.

        Args:
            index: Zero-based sample index.

        Returns:
            The requested light curve.
        """

    def __iter__(self) -> Iterator[LightCurve]:
        for index in range(len(self)):
            yield self[index]

    @property
    @abc.abstractmethod
    def labels(self) -> npt.NDArray[np.int_]:
        """Class labels for all samples, aligned with indexing order."""

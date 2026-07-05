"""Immutable value objects of the TCE stage.

Three containers flow through the TCE pipeline:

* :class:`SearchGrid` — the trial periods and durations plus the full
  provenance of how they were derived.
* :class:`Periodogram` — the BLS spectrum with per-period best-fit
  parameters.
* :class:`TransitCandidate` — one detected periodic transit-like
  signal with its detection statistics, quality flags, status, and
  provenance. Candidates are frozen; state transitions (validation,
  harmonic rejection, ranking) produce new instances.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.exceptions import DataError
from exodet.utils.io import ensure_dir

__all__ = [
    "SearchGrid",
    "Periodogram",
    "TransitCandidate",
    "STATUS_CANDIDATE",
    "STATUS_REJECTED_VALIDATION",
    "STATUS_REJECTED_HARMONIC",
    "save_candidates",
    "load_candidates",
]

logger = logging.getLogger(__name__)

STATUS_CANDIDATE = "candidate"
STATUS_REJECTED_VALIDATION = "rejected_validation"
STATUS_REJECTED_HARMONIC = "rejected_harmonic"


@dataclass(frozen=True, slots=True)
class SearchGrid:
    """Trial periods and durations for a BLS search.

    Attributes:
        periods: Trial periods in days, uniformly spaced in frequency
            and ordered by increasing frequency (decreasing period).
        durations: Trial transit durations in days.
        provenance: Every parameter and derived quantity used to build
            the grid (baseline, cadence, spacing, clamping decisions).
    """

    periods: npt.NDArray[np.float64]
    durations: npt.NDArray[np.float64]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def frequencies(self) -> npt.NDArray[np.float64]:
        """Trial frequencies in cycles/day, increasing."""
        return 1.0 / self.periods

    @property
    def min_period(self) -> float:
        """Shortest trial period in days."""
        return float(self.periods.min())

    @property
    def max_period(self) -> float:
        """Longest trial period in days."""
        return float(self.periods.max())

    def __len__(self) -> int:
        return len(self.periods)


@dataclass(frozen=True, slots=True)
class Periodogram:
    """BLS spectrum with per-period best-fit transit parameters.

    All arrays are aligned with ``periods`` (increasing frequency).

    Attributes:
        periods: Trial periods in days.
        power: BLS objective value at each period.
        depth: Best-fit transit depth at each period (relative flux).
        depth_snr: Depth-over-uncertainty at each period.
        duration: Best-fit duration at each period in days.
        transit_time: Best-fit mid-transit epoch at each period in days.
        log_likelihood: Log-likelihood of the best-fit box model.
        objective: BLS objective that ``power`` maximizes
            (``"likelihood"`` or ``"snr"``).
        meta: Provenance (grid parameters, engine settings, counts).
    """

    periods: npt.NDArray[np.float64]
    power: npt.NDArray[np.float64]
    depth: npt.NDArray[np.float64]
    depth_snr: npt.NDArray[np.float64]
    duration: npt.NDArray[np.float64]
    transit_time: npt.NDArray[np.float64]
    log_likelihood: npt.NDArray[np.float64]
    objective: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def frequencies(self) -> npt.NDArray[np.float64]:
        """Trial frequencies in cycles/day."""
        return 1.0 / self.periods

    @property
    def best_index(self) -> int:
        """Index of the highest-power period."""
        return int(np.nanargmax(self.power))

    def __len__(self) -> int:
        return len(self.periods)


@dataclass(frozen=True, slots=True)
class TransitCandidate:
    """One periodic transit-like detection (a TCE).

    Instances are immutable: validation, harmonic analysis, and
    ranking use :meth:`with_status` / :meth:`with_meta` to derive
    updated copies, preserving the original.

    Attributes:
        candidate_id: Unique identifier, ``"<target>-NN"``.
        target_id: Host target identifier (e.g. ``"TIC 123456789"``).
        sectors: TESS sectors contributing to the detection.
        period_days: Orbital period of the candidate in days.
        epoch_days: Mid-transit time of the first transit, in the time
            system of the light curve.
        duration_days: Transit duration in days.
        depth: Fractional transit depth (positive for a dip).
        depth_err: 1-sigma uncertainty on the depth.
        n_transits: Number of transit windows containing data.
        n_expected_transits: Transit windows within the baseline.
        snr: Detection signal-to-noise ratio (depth / depth error).
        sde: Signal Detection Efficiency of the periodogram peak.
        power: BLS objective value at the peak.
        fap: Approximate false-alarm probability (NaN when disabled).
        quality_flags: Names of triggered diagnostic flags.
        status: One of the ``STATUS_*`` constants.
        rejection_reason: Why the candidate was rejected, if it was.
        meta: Additional diagnostics (odd/even depths, rank, ...).
        history: Provenance trail inherited from the light curve plus
            every TCE stage that touched this candidate.
    """

    candidate_id: str
    target_id: str
    sectors: tuple[int, ...]
    period_days: float
    epoch_days: float
    duration_days: float
    depth: float
    depth_err: float
    n_transits: int
    n_expected_transits: int
    snr: float
    sde: float
    power: float
    fap: float
    quality_flags: tuple[str, ...] = ()
    status: str = STATUS_CANDIDATE
    rejection_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    history: tuple[str, ...] = ()

    def with_status(self, status: str, reason: str, stage: str) -> "TransitCandidate":
        """Returns a copy with an updated status and provenance entry.

        Args:
            status: New ``STATUS_*`` value.
            reason: Human-readable explanation.
            stage: Provenance entry describing the deciding stage.

        Returns:
            The updated candidate; the original is unchanged.
        """
        return replace(
            self,
            status=status,
            rejection_reason=reason if status != STATUS_CANDIDATE else None,
            history=(*self.history, stage),
        )

    def with_meta(self, stage: str | None = None, **entries: Any) -> "TransitCandidate":
        """Returns a copy with additional metadata entries.

        Args:
            stage: Optional provenance entry to append.
            **entries: Key/value pairs merged into ``meta``.

        Returns:
            The updated candidate; the original is unchanged.
        """
        history = (*self.history, stage) if stage else self.history
        return replace(self, meta={**self.meta, **entries}, history=history)

    def to_dict(self) -> dict[str, Any]:
        """Converts the candidate to JSON-native types.

        Returns:
            A dictionary safe for ``json.dump``.
        """
        raw = asdict(self)
        raw["sectors"] = [int(s) for s in self.sectors]
        raw["quality_flags"] = list(self.quality_flags)
        raw["history"] = list(self.history)
        raw["meta"] = _jsonify(self.meta)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TransitCandidate":
        """Reconstructs a candidate from :meth:`to_dict` output.

        Args:
            raw: The serialized dictionary.

        Returns:
            The reconstructed candidate.
        """
        data = dict(raw)
        data["sectors"] = tuple(int(s) for s in data["sectors"])
        data["quality_flags"] = tuple(data["quality_flags"])
        data["history"] = tuple(data["history"])
        return cls(**data)


def _jsonify(value: Any) -> Any:
    """Recursively converts NumPy containers to JSON-native types.

    Args:
        value: Arbitrary metadata value.

    Returns:
        A JSON-serializable equivalent.
    """
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_candidates(
    candidates: list[TransitCandidate], path: Path | str
) -> Path:
    """Writes candidates (accepted and rejected) to a JSON file.

    Args:
        candidates: Candidates to persist.
        path: Destination JSON file.

    Returns:
        The written file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            [candidate.to_dict() for candidate in candidates],
            handle,
            indent=2,
            sort_keys=True,
        )
    logger.debug("Saved %d candidate(s) to %s", len(candidates), path)
    return path


def load_candidates(path: Path | str) -> list[TransitCandidate]:
    """Reads candidates written by :func:`save_candidates`.

    Args:
        path: Source JSON file.

    Returns:
        The reconstructed candidates.

    Raises:
        DataError: If the file is missing or malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Candidate file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return [TransitCandidate.from_dict(entry) for entry in raw]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataError(f"Malformed candidate file {path}: {exc}") from exc

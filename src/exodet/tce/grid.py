"""Automatic BLS search-grid generation with physical validation.

The trial-frequency spacing follows the standard BLS criterion (as in
astropy's ``BoxLeastSquares.autoperiod``): the phase drift of a transit
over the observing baseline must stay below a fraction of the shortest
trial duration, giving ``df = min_duration / (oversample * baseline^2)``.
Durations are laid out geometrically between configurable bounds.

Every derived quantity — baseline, cadence, spacing, any clamping of
the requested period range — is recorded in the grid provenance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.preprocessing.common import median_cadence
from exodet.registry import Registry
from exodet.tce.candidate import SearchGrid

__all__ = ["GRID_GENERATORS", "BLSGridGenerator"]

logger = logging.getLogger(__name__)

GRID_GENERATORS: Registry["BLSGridGenerator"] = Registry("TCE grid generator")

_MAX_GRID_SIZE = 5_000_000


@GRID_GENERATORS.register("bls_auto")
class BLSGridGenerator:
    """Builds a validated period/duration grid for one light curve.

    Attributes:
        min_period_days: Shortest trial period; must exceed both the
            Nyquist-like cadence limit and every trial duration.
        max_period_days: Longest trial period, or ``None`` to derive it
            from the baseline as ``baseline / min_n_transits``.
        n_frequencies: Explicit number of trial frequencies; when
            ``None`` the count follows from the spacing criterion.
        oversample: Frequency oversampling factor (>= 1); larger values
            refine the grid beyond the minimal BLS spacing.
        min_duration_days: Shortest trial duration.
        max_duration_days: Longest trial duration.
        n_durations: Number of geometrically spaced trial durations.
        min_n_transits: Minimum transits that must fit in the baseline
            at the longest trial period.
    """

    def __init__(
        self,
        min_period_days: float = 0.5,
        max_period_days: float | None = None,
        n_frequencies: int | None = None,
        oversample: float = 3.0,
        min_duration_days: float = 0.02,
        max_duration_days: float = 0.3,
        n_durations: int = 5,
        min_n_transits: int = 2,
    ) -> None:
        """Initializes and statically validates the grid parameters.

        Args:
            min_period_days: Shortest trial period in days.
            max_period_days: Longest trial period in days, or ``None``
                for automatic derivation from the baseline.
            n_frequencies: Explicit frequency count, or ``None``.
            oversample: Frequency oversampling factor.
            min_duration_days: Shortest trial duration in days.
            max_duration_days: Longest trial duration in days.
            n_durations: Number of trial durations.
            min_n_transits: Required transits at the longest period.

        Raises:
            PipelineError: If any parameter is out of range or the
                parameters are mutually inconsistent.
        """
        if min_period_days <= 0:
            raise PipelineError(f"min_period_days must be > 0, got {min_period_days}.")
        if max_period_days is not None and max_period_days <= min_period_days:
            raise PipelineError(
                f"max_period_days ({max_period_days}) must exceed "
                f"min_period_days ({min_period_days})."
            )
        if oversample < 1.0:
            raise PipelineError(f"oversample must be >= 1, got {oversample}.")
        if not 0 < min_duration_days <= max_duration_days:
            raise PipelineError(
                "Require 0 < min_duration_days <= max_duration_days, got "
                f"{min_duration_days} and {max_duration_days}."
            )
        if n_durations < 1:
            raise PipelineError(f"n_durations must be >= 1, got {n_durations}.")
        if min_n_transits < 1:
            raise PipelineError(f"min_n_transits must be >= 1, got {min_n_transits}.")
        if max_duration_days >= min_period_days:
            raise PipelineError(
                f"max_duration_days ({max_duration_days}) must be shorter than "
                f"min_period_days ({min_period_days}); a transit cannot outlast "
                "its orbit."
            )
        if n_frequencies is not None and n_frequencies < 2:
            raise PipelineError(f"n_frequencies must be >= 2, got {n_frequencies}.")

        self.min_period_days = float(min_period_days)
        self.max_period_days = None if max_period_days is None else float(max_period_days)
        self.n_frequencies = n_frequencies
        self.oversample = float(oversample)
        self.min_duration_days = float(min_duration_days)
        self.max_duration_days = float(max_duration_days)
        self.n_durations = int(n_durations)
        self.min_n_transits = int(min_n_transits)

    def generate(self, light_curve: LightCurve) -> SearchGrid:
        """Builds the search grid for a specific light curve.

        Args:
            light_curve: The (preprocessed) light curve to be searched.

        Returns:
            A validated :class:`SearchGrid` with full provenance.

        Raises:
            PipelineError: If the observations cannot support the
                requested grid (baseline too short, cadence too coarse,
                or an absurdly large grid).
        """
        time = light_curve.time[np.isfinite(light_curve.time)]
        if time.size < 3:
            raise PipelineError(
                f"Target {light_curve.target_id}: need >= 3 finite cadences "
                "to build a search grid."
            )
        baseline = float(time[-1] - time[0])
        cadence = median_cadence(time)
        notes: list[str] = []

        # Nyquist-like cadence limit: at least two samples per period.
        nyquist_period = 2.0 * cadence
        if self.min_period_days < nyquist_period:
            raise PipelineError(
                f"min_period_days ({self.min_period_days:.4f} d) violates the "
                f"cadence limit 2 x {cadence:.4f} d; periods below "
                f"{nyquist_period:.4f} d are unsampleable."
            )
        if self.min_duration_days < 2.0 * cadence:
            notes.append(
                f"min_duration ({self.min_duration_days:.4f} d) spans fewer "
                f"than 2 cadences ({cadence:.4f} d); shortest transits will "
                "be poorly resolved."
            )
            logger.warning("Target %s: %s", light_curve.target_id, notes[-1])

        # Baseline limit: the longest period must still show
        # min_n_transits transits.
        baseline_max_period = baseline / self.min_n_transits
        max_period = self.max_period_days
        if max_period is None:
            max_period = baseline_max_period
            notes.append(
                f"max_period auto-derived as baseline/{self.min_n_transits} "
                f"= {max_period:.4f} d."
            )
        elif max_period > baseline_max_period:
            notes.append(
                f"requested max_period {max_period:.4f} d clamped to "
                f"{baseline_max_period:.4f} d (baseline {baseline:.4f} d, "
                f"min_n_transits {self.min_n_transits})."
            )
            logger.warning("Target %s: %s", light_curve.target_id, notes[-1])
            max_period = baseline_max_period
        if max_period <= self.min_period_days:
            raise PipelineError(
                f"Target {light_curve.target_id}: baseline {baseline:.4f} d "
                f"cannot fit {self.min_n_transits} transits above the minimum "
                f"period {self.min_period_days:.4f} d."
            )

        f_min = 1.0 / max_period
        f_max = 1.0 / self.min_period_days
        if self.n_frequencies is not None:
            n_freq = self.n_frequencies
            df = (f_max - f_min) / (n_freq - 1)
        else:
            df = self.min_duration_days / (self.oversample * baseline**2)
            # Floor (not ceil) so no trial frequency exceeds f_max,
            # which would silently violate the requested minimum period.
            n_freq = int(np.floor((f_max - f_min) / df)) + 1
        if n_freq > _MAX_GRID_SIZE:
            raise PipelineError(
                f"Grid of {n_freq} frequencies exceeds the safety limit "
                f"{_MAX_GRID_SIZE}; increase min_period/min_duration or "
                "reduce oversample."
            )

        frequencies = np.linspace(f_min, f_min + df * (n_freq - 1), n_freq)
        periods = 1.0 / frequencies

        if self.n_durations == 1:
            durations = np.array([self.min_duration_days])
        else:
            durations = np.geomspace(
                self.min_duration_days, self.max_duration_days, self.n_durations
            )

        provenance: dict[str, Any] = {
            "generator": "bls_auto",
            "baseline_days": baseline,
            "median_cadence_days": cadence,
            "nyquist_period_days": nyquist_period,
            "min_period_days": self.min_period_days,
            "max_period_days": float(max_period),
            "frequency_spacing_per_day": float(df),
            "n_frequencies": int(n_freq),
            "oversample": self.oversample,
            "durations_days": durations.tolist(),
            "min_n_transits": self.min_n_transits,
            "n_points": int(time.size),
            "notes": notes,
        }
        logger.info(
            "Target %s: search grid with %d frequencies "
            "(P in [%.3f, %.3f] d) x %d durations.",
            light_curve.target_id,
            n_freq,
            self.min_period_days,
            max_period,
            len(durations),
        )
        return SearchGrid(periods=periods, durations=durations, provenance=provenance)

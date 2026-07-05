"""Vectorized Box Least Squares search engine (astropy backend).

The engine wraps :class:`astropy.timeseries.BoxLeastSquares`, whose
periodogram evaluation is fully vectorized in compiled code over the
period x duration grid — no Python loops touch individual cadences.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from astropy.timeseries import BoxLeastSquares

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.tce.candidate import Periodogram, SearchGrid

__all__ = ["SEARCH_ENGINES", "AstropyBLSEngine"]

logger = logging.getLogger(__name__)

SEARCH_ENGINES: Registry["AstropyBLSEngine"] = Registry("TCE search engine")


@SEARCH_ENGINES.register("astropy_bls")
class AstropyBLSEngine:
    """Box Least Squares periodogram search built on astropy.

    Attributes:
        objective: BLS objective, ``"snr"`` (signal residue analogue,
            recommended for SDE computation) or ``"likelihood"``.
        phase_oversample: Oversampling of the transit duration when
            binning in phase (astropy ``oversample``).
        use_flux_err: Whether to weight cadences by their photometric
            uncertainties when available.
    """

    _OBJECTIVES = ("snr", "likelihood")

    def __init__(
        self,
        objective: str = "snr",
        phase_oversample: int = 10,
        use_flux_err: bool = True,
    ) -> None:
        """Initializes the engine.

        Args:
            objective: Quantity maximized by the periodogram.
            phase_oversample: Phase-bin oversampling factor (>= 1).
            use_flux_err: Use ``flux_err`` as per-point weights.

        Raises:
            PipelineError: If the objective or oversampling is invalid.
        """
        if objective not in self._OBJECTIVES:
            raise PipelineError(
                f"Unknown BLS objective '{objective}'. Available: {self._OBJECTIVES}."
            )
        if phase_oversample < 1:
            raise PipelineError(
                f"phase_oversample must be >= 1, got {phase_oversample}."
            )
        self.objective = objective
        self.phase_oversample = int(phase_oversample)
        self.use_flux_err = use_flux_err

    def _model(self, light_curve: LightCurve) -> tuple[BoxLeastSquares, int]:
        """Builds the astropy BLS model from finite cadences.

        Args:
            light_curve: The input light curve.

        Returns:
            A tuple of (model, number of finite cadences used).

        Raises:
            PipelineError: If fewer than 10 finite cadences remain.
        """
        finite = np.isfinite(light_curve.time) & np.isfinite(light_curve.flux)
        n_dropped = int(np.count_nonzero(~finite))
        if n_dropped:
            logger.warning(
                "Target %s: ignoring %d non-finite cadence(s) in BLS search.",
                light_curve.target_id,
                n_dropped,
            )
        time = light_curve.time[finite]
        flux = light_curve.flux[finite]
        if time.size < 10:
            raise PipelineError(
                f"Target {light_curve.target_id}: only {time.size} finite "
                "cadences; BLS search requires at least 10."
            )
        dy = None
        if self.use_flux_err and light_curve.flux_err is not None:
            dy = light_curve.flux_err[finite]
            if not (np.isfinite(dy).all() and (dy > 0).all()):
                logger.warning(
                    "Target %s: invalid flux_err values; searching unweighted.",
                    light_curve.target_id,
                )
                dy = None
        return BoxLeastSquares(time, flux, dy=dy), int(time.size)

    def search(self, light_curve: LightCurve, grid: SearchGrid) -> Periodogram:
        """Computes the BLS periodogram over the search grid.

        Args:
            light_curve: The (preprocessed) light curve.
            grid: The trial periods and durations.

        Returns:
            The periodogram with per-period best-fit parameters and
            full provenance in ``meta``.

        Raises:
            PipelineError: If astropy rejects the inputs or the entire
                spectrum is non-finite.
        """
        model, n_points = self._model(light_curve)
        try:
            result = model.power(
                grid.periods,
                grid.durations,
                objective=self.objective,
                oversample=self.phase_oversample,
            )
        except ValueError as exc:
            raise PipelineError(
                f"BLS search failed for target {light_curve.target_id}: {exc}"
            ) from exc

        power = np.asarray(result.power, dtype=np.float64)
        bad = ~np.isfinite(power)
        if bad.all():
            raise PipelineError(
                f"Target {light_curve.target_id}: BLS produced no finite power "
                "values (constant or degenerate flux?)."
            )
        if bad.any():
            logger.warning(
                "Target %s: %d/%d non-finite periodogram values set to 0.",
                light_curve.target_id,
                int(bad.sum()),
                power.size,
            )
            power = np.where(bad, 0.0, power)

        meta: dict[str, Any] = {
            "engine": "astropy_bls",
            "objective": self.objective,
            "phase_oversample": self.phase_oversample,
            "weighted": bool(
                self.use_flux_err and light_curve.flux_err is not None
            ),
            "n_points": n_points,
            "grid": dict(grid.provenance),
            "target_id": light_curve.target_id,
        }
        logger.info(
            "Target %s: BLS periodogram over %d periods x %d durations "
            "(objective=%s).",
            light_curve.target_id,
            len(grid.periods),
            len(grid.durations),
            self.objective,
        )
        return Periodogram(
            periods=np.asarray(result.period, dtype=np.float64),
            power=power,
            depth=np.asarray(result.depth, dtype=np.float64),
            depth_snr=np.asarray(result.depth_snr, dtype=np.float64),
            duration=np.asarray(result.duration, dtype=np.float64),
            transit_time=np.asarray(result.transit_time, dtype=np.float64),
            log_likelihood=np.asarray(result.log_likelihood, dtype=np.float64),
            objective=self.objective,
            meta=meta,
        )

    def compute_stats(
        self,
        light_curve: LightCurve,
        period: float,
        duration: float,
        transit_time: float,
    ) -> dict[str, Any]:
        """Computes detailed diagnostics for one candidate solution.

        Wraps astropy ``BoxLeastSquares.compute_stats`` (per-transit
        point counts, refined depth, odd/even depths, harmonic model
        comparison).

        Args:
            light_curve: The searched light curve.
            period: Candidate period in days.
            duration: Candidate duration in days.
            transit_time: Candidate mid-transit epoch in days.

        Returns:
            The astropy statistics dictionary.
        """
        model, _ = self._model(light_curve)
        return model.compute_stats(period, duration, transit_time)

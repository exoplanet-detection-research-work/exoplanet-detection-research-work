"""Observational gap detection and interpolation.

TESS light curves contain gaps from data downlinks, momentum dumps,
and quality filtering. :class:`GapDetector` records gap metadata
without altering the data; :class:`GapInterpolator` optionally fills
gaps with synthetic cadences so downstream methods that assume near-
uniform sampling behave well. Synthetic cadences are flagged in the
per-cadence ``meta["interpolated"]`` mask so they can always be
identified or excluded later.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.preprocessing.base import PREPROCESSORS, BasePreprocessor
from exodet.preprocessing.common import find_gaps, median_cadence

__all__ = ["GapDetector", "GapInterpolator"]

logger = logging.getLogger(__name__)


@PREPROCESSORS.register("gap_detect")
class GapDetector(BasePreprocessor):
    """Detects observational gaps and stores them as metadata.

    The flux is untouched; the step only appends provenance and writes
    ``meta["gaps"]`` (list of gap records) plus
    ``meta["gap_threshold_days"]``.

    Attributes:
        factor: Gap threshold as a multiple of the median cadence.
        min_gap_days: Absolute lower bound of the threshold in days.
    """

    def __init__(self, factor: float = 5.0, min_gap_days: float = 0.0) -> None:
        """Initializes the detector.

        Args:
            factor: Threshold multiple of the median cadence; must be
                positive.
            min_gap_days: Absolute threshold floor in days.

        Raises:
            PipelineError: If ``factor`` is not positive.
        """
        if factor <= 0:
            raise PipelineError(f"factor must be > 0, got {factor}.")
        self.factor = float(factor)
        self.min_gap_days = float(min_gap_days)

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Finds gaps and records them in metadata.

        Args:
            light_curve: The input curve (time must be sorted).

        Returns:
            The same data with gap metadata and provenance appended.
        """
        gaps, threshold = find_gaps(
            light_curve.time, factor=self.factor, min_gap_days=self.min_gap_days
        )
        result = light_curve.replace_flux(
            light_curve.flux, step_name=f"{self.name}(n_gaps={len(gaps)})"
        )
        result.meta["gaps"] = gaps
        result.meta["gap_threshold_days"] = threshold
        logger.info(
            "Target %s: detected %d gap(s) above %.4f d.",
            light_curve.target_id,
            len(gaps),
            threshold,
        )
        return result


@PREPROCESSORS.register("gap_interpolate")
class GapInterpolator(BasePreprocessor):
    """Fills observational gaps with interpolated synthetic cadences.

    Synthetic samples are placed on the median-cadence grid inside each
    gap shorter than ``max_gap_days`` (long downlink gaps are never
    bridged, as any interpolation across them would be pure invention).
    Gap records from a preceding :class:`GapDetector` are reused when
    present; otherwise gaps are detected with this step's parameters.

    Attributes:
        method: ``"linear"``, ``"spline"`` (cubic), or ``"none"``.
        max_gap_days: Longest gap that will be filled, in days.
        factor: Gap-detection threshold multiple (used only when no
            detector ran earlier).
    """

    _METHODS = ("linear", "spline", "none")

    def __init__(
        self,
        method: str = "linear",
        max_gap_days: float = 0.5,
        factor: float = 5.0,
    ) -> None:
        """Initializes the interpolator.

        Args:
            method: Interpolation scheme; ``"none"`` disables filling
                while keeping the stage in the pipeline for provenance.
            max_gap_days: Longest fillable gap in days; must be positive.
            factor: Gap-detection threshold multiple; must be positive.

        Raises:
            PipelineError: If the method or parameters are invalid.
        """
        if method not in self._METHODS:
            raise PipelineError(
                f"Unknown interpolation method '{method}'. Available: {self._METHODS}."
            )
        if max_gap_days <= 0:
            raise PipelineError(f"max_gap_days must be > 0, got {max_gap_days}.")
        if factor <= 0:
            raise PipelineError(f"factor must be > 0, got {factor}.")
        self.method = method
        self.max_gap_days = float(max_gap_days)
        self.factor = float(factor)

    def _interpolate(
        self,
        time: npt.NDArray[np.float64],
        flux: npt.NDArray[np.float64],
        new_time: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates the configured interpolant at new time points.

        Args:
            time: Observed times.
            flux: Observed flux values.
            new_time: Times of the synthetic cadences.

        Returns:
            Interpolated flux at ``new_time``.
        """
        if self.method == "spline":
            return CubicSpline(time, flux, extrapolate=False)(new_time)
        return np.interp(new_time, time, flux)

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Fills short gaps with synthetic, flagged cadences.

        Args:
            light_curve: The input curve (time must be sorted and
                strictly increasing for spline interpolation).

        Returns:
            The curve with synthetic cadences inserted (or unchanged
            apart from provenance when ``method="none"`` or no fillable
            gap exists).
        """
        if self.method == "none":
            return light_curve.replace_flux(
                light_curve.flux, step_name=f"{self.name}(method=none)"
            )

        gaps = light_curve.meta.get("gaps")
        if gaps is None:
            gaps, _ = find_gaps(light_curve.time, factor=self.factor)
        fillable = [gap for gap in gaps if gap["duration_days"] <= self.max_gap_days]
        step = f"{self.name}(method={self.method},n_filled_gaps={len(fillable)})"

        if not fillable:
            result = light_curve.replace_flux(light_curve.flux, step_name=step)
            result.meta.setdefault(
                "interpolated", np.zeros(len(light_curve), dtype=bool)
            )
            return result

        cadence = median_cadence(light_curve.time)
        # One small array per gap (gap counts are tens at most), then a
        # single concatenation: no per-cadence Python looping.
        inserted_chunks = [
            np.arange(
                gap["start_time"] + cadence, gap["end_time"] - 0.5 * cadence, cadence
            )
            for gap in fillable
        ]
        new_time = np.concatenate(inserted_chunks)
        new_flux = self._interpolate(light_curve.time, light_curve.flux, new_time)

        time = np.concatenate([light_curve.time, new_time])
        flux = np.concatenate([light_curve.flux, new_flux])
        order = np.argsort(time, kind="stable")

        flux_err = None
        if light_curve.flux_err is not None:
            fill_err = np.full(new_time.size, float(np.median(light_curve.flux_err)))
            flux_err = np.concatenate([light_curve.flux_err, fill_err])[order]

        n_old, n_new = len(light_curve), new_time.size
        result = light_curve.replace_flux(
            flux[order], step_name=step, time=time[order], flux_err=flux_err
        )

        existing_interp = light_curve.meta.get("interpolated")
        if not isinstance(existing_interp, np.ndarray):
            existing_interp = np.zeros(n_old, dtype=bool)
        result.meta["interpolated"] = np.concatenate(
            [existing_interp, np.ones(n_new, dtype=bool)]
        )[order]

        for key, fill in (("quality", 0), ("sector", None)):
            value = light_curve.meta.get(key)
            if isinstance(value, np.ndarray) and value.shape == (n_old,):
                if fill is None:
                    # Assign each synthetic cadence the sector of its
                    # nearest preceding observed cadence.
                    idx = np.searchsorted(light_curve.time, new_time, side="right") - 1
                    fill_values = value[np.clip(idx, 0, n_old - 1)]
                else:
                    fill_values = np.full(n_new, fill, dtype=value.dtype)
                result.meta[key] = np.concatenate([value, fill_values])[order]

        logger.info(
            "Target %s: filled %d gap(s) with %d synthetic cadences (%s).",
            light_curve.target_id,
            len(fillable),
            n_new,
            self.method,
        )
        return result

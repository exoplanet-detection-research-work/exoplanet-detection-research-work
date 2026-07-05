"""Photometric quality metrics computed as a pipeline stage.

The :class:`QualityMetrics` step leaves the data untouched and writes
a dictionary of scalar diagnostics into ``meta["quality_metrics"]``,
providing a quantitative record of the preprocessing outcome for every
target (useful for sample selection cuts and paper tables).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.preprocessing.base import PREPROCESSORS, BasePreprocessor
from exodet.preprocessing.common import median_cadence

__all__ = ["QualityMetrics", "estimate_cdpp"]

logger = logging.getLogger(__name__)


def estimate_cdpp(
    time: npt.NDArray[np.float64],
    flux: npt.NDArray[np.float64],
    *,
    duration_hours: float = 1.0,
    savgol_window: int = 101,
    savgol_polyorder: int = 2,
    sigma: float = 5.0,
) -> float:
    """Estimates the Combined Differential Photometric Precision proxy.

    Follows the standard proxy algorithm (as popularized by
    lightkurve's ``estimate_cdpp``): Savitzky-Golay flattening, sigma
    clipping, then the standard deviation of the running mean over one
    transit duration, expressed in parts per million.

    Args:
        time: Sorted observation times in days.
        flux: Flux values (any normalization).
        duration_hours: Averaging window; 1 hour is the common TESS
            convention (Kepler papers typically quote 6.5 hours).
        savgol_window: Savitzky-Golay window length in cadences
            (reduced automatically for short curves; must remain odd).
        savgol_polyorder: Savitzky-Golay polynomial order.
        sigma: Outlier rejection threshold applied before averaging.

    Returns:
        The CDPP estimate in ppm, or ``nan`` when the curve is too
        short for a meaningful estimate.
    """
    n = flux.size
    cadence_days = median_cadence(time)
    duration_cadences = max(1, round(duration_hours / 24.0 / cadence_days))
    window = min(savgol_window, n - 1 if (n - 1) % 2 else n - 2)
    if window <= savgol_polyorder or n < 2 * duration_cadences:
        return math.nan

    normalized = flux / np.median(flux)
    trend = savgol_filter(normalized, window_length=window, polyorder=savgol_polyorder)
    residual = normalized / trend - 1.0

    center = np.median(residual)
    mad_std = 1.4826 * np.median(np.abs(residual - center))
    if mad_std > 0:
        residual = residual[np.abs(residual - center) <= sigma * mad_std]
    if residual.size < 2 * duration_cadences:
        return math.nan

    # O(n) running mean over the transit duration via cumulative sums.
    cumulative = np.concatenate(([0.0], np.cumsum(residual)))
    running_mean = (
        cumulative[duration_cadences:] - cumulative[:-duration_cadences]
    ) / duration_cadences
    return float(1e6 * np.std(running_mean))


@PREPROCESSORS.register("quality_metrics")
class QualityMetrics(BasePreprocessor):
    """Computes photometric quality diagnostics into metadata.

    Metrics written to ``meta["quality_metrics"]``:
        * ``rms_ppm``: robust point-to-point scatter of the
          median-normalized flux, in ppm.
        * ``cdpp_ppm``: CDPP proxy (see :func:`estimate_cdpp`).
        * ``variance``: variance of the flux as-is.
        * ``skewness`` / ``kurtosis``: shape moments of the flux
          distribution (Fisher kurtosis: 0 for a Gaussian).
        * ``duty_cycle``: fraction of the expected cadences actually
          observed (synthetic interpolated cadences excluded).
        * ``missing_fraction``: ``1 - duty_cycle``.
        * ``n_points``, ``timespan_days``, ``median_cadence_days``.

    Attributes:
        cdpp_duration_hours: Averaging duration of the CDPP proxy.
    """

    def __init__(self, cdpp_duration_hours: float = 1.0) -> None:
        """Initializes the metric computation.

        Args:
            cdpp_duration_hours: CDPP averaging window; must be positive.

        Raises:
            PipelineError: If the duration is not positive.
        """
        if cdpp_duration_hours <= 0:
            raise PipelineError(
                f"cdpp_duration_hours must be > 0, got {cdpp_duration_hours}."
            )
        self.cdpp_duration_hours = float(cdpp_duration_hours)

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Computes metrics and stores them in metadata.

        Args:
            light_curve: The input curve with finite flux.

        Returns:
            The unchanged data with ``meta["quality_metrics"]`` set and
            provenance appended.

        Raises:
            PipelineError: If the curve has fewer than two cadences.
        """
        # Metrics are computed over finite cadences only, so the step is
        # meaningful even when placed before NaN removal.
        finite = np.isfinite(light_curve.time) & np.isfinite(light_curve.flux)
        if np.count_nonzero(finite) < 2:
            raise PipelineError(
                f"Target {light_curve.target_id}: fewer than two finite "
                "cadences; cannot compute quality metrics."
            )
        time = light_curve.time[finite]
        flux = light_curve.flux[finite]
        cadence = median_cadence(time)
        timespan = float(time[-1] - time[0])

        interpolated = light_curve.meta.get("interpolated")
        if isinstance(interpolated, np.ndarray) and interpolated.shape == finite.shape:
            n_observed = int(np.count_nonzero(~interpolated & finite))
        else:
            n_observed = flux.size
        expected = timespan / cadence + 1.0
        duty_cycle = min(1.0, n_observed / expected) if expected > 0 else math.nan

        median_flux = float(np.median(flux))
        if median_flux != 0.0:
            rms_ppm = float(1e6 * np.std(flux / median_flux))
        else:
            rms_ppm = math.nan

        metrics: dict[str, float | int] = {
            "rms_ppm": rms_ppm,
            "cdpp_ppm": estimate_cdpp(
                time, flux, duration_hours=self.cdpp_duration_hours
            )
            if median_flux != 0.0
            else math.nan,
            "variance": float(np.var(flux)),
            "skewness": float(skew(flux)),
            "kurtosis": float(kurtosis(flux)),
            "duty_cycle": duty_cycle,
            "missing_fraction": 1.0 - duty_cycle if math.isfinite(duty_cycle) else math.nan,
            "n_points": int(flux.size),
            "timespan_days": timespan,
            "median_cadence_days": cadence,
        }

        result = light_curve.replace_flux(light_curve.flux, step_name=self.name)
        result.meta["quality_metrics"] = metrics
        logger.info(
            "Target %s: quality metrics computed (rms=%.1f ppm, cdpp=%.1f ppm, "
            "duty cycle=%.3f).",
            light_curve.target_id,
            metrics["rms_ppm"],
            metrics["cdpp_ppm"],
            metrics["duty_cycle"],
        )
        return result

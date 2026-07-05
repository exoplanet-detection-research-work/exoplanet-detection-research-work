"""Detection-significance metrics: SDE, SNR, power, and FAP.

Definitions:
    * **SDE** (Signal Detection Efficiency, Kovacs et al. 2002): the
      peak power standardized against the periodogram's own
      distribution, ``(P_peak - <P>) / sd(P)``. Following standard
      practice (e.g. the trend-removed SR spectrum of Kovacs et al.
      and the running-median normalization used by TLS), a slowly
      varying continuum is removed from the power spectrum with a
      running median before standardization; the BLS continuum rises
      toward long periods and would otherwise dilute genuine peaks.
      A robust baseline (median / MAD) is used by default so strong
      peaks do not inflate their own significance baseline.
    * **SNR**: refined transit depth over its uncertainty from the
      astropy box-model fit.
    * **FAP**: an analytic approximation assuming Gaussian periodogram
      noise and ``N_eff = baseline x (f_max - f_min)`` independent
      frequencies: ``FAP = 1 - (1 - p_single)^N_eff`` with
      ``p_single = 0.5 erfc(SDE / sqrt(2))``. Correlated noise makes
      this optimistic; it is a comparative statistic, not a calibrated
      probability, and can be disabled.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter

from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.tce.candidate import Periodogram

__all__ = [
    "METRICS_COMPUTERS",
    "StandardMetricsComputer",
    "detrend_power",
    "sde_of_peak",
    "gaussian_fap",
]

logger = logging.getLogger(__name__)

METRICS_COMPUTERS: Registry["StandardMetricsComputer"] = Registry(
    "TCE detection metrics computer"
)

_MAD_TO_STD = 1.4826


def detrend_power(
    power: npt.NDArray[np.float64], window_fraction: float
) -> npt.NDArray[np.float64]:
    """Removes the slowly varying continuum from a power spectrum.

    Subtracts a running median whose window spans ``window_fraction``
    of the spectrum. The window is far wider than any physical peak,
    so genuine detections are preserved while the long-period power
    ramp of the BLS objective is flattened.

    Args:
        power: The periodogram power array (uniform in frequency).
        window_fraction: Running-median window as a fraction of the
            spectrum length; in ``(0, 1)``.

    Returns:
        The continuum-subtracted power spectrum.

    Raises:
        PipelineError: If the fraction is out of range.
    """
    if not 0.0 < window_fraction < 1.0:
        raise PipelineError(
            f"window_fraction must be in (0, 1), got {window_fraction}."
        )
    window = max(3, int(round(window_fraction * power.size)) | 1)
    if window >= power.size:
        return power - float(np.median(power))
    return power - median_filter(power, size=window, mode="nearest")


def sde_of_peak(
    power: np.ndarray, index: int, *, robust: bool = True
) -> float:
    """Computes the Signal Detection Efficiency of one peak.

    Args:
        power: Full periodogram power array.
        index: Peak index.
        robust: Use median/MAD instead of mean/std for the baseline.

    Returns:
        The SDE value, or ``nan`` if the spectrum has zero dispersion.
    """
    if robust:
        center = float(np.median(power))
        scale = _MAD_TO_STD * float(np.median(np.abs(power - center)))
    else:
        center = float(np.mean(power))
        scale = float(np.std(power))
    if scale == 0.0:
        return math.nan
    return (float(power[index]) - center) / scale


def gaussian_fap(sde: float, n_effective: float) -> float:
    """Approximates the false-alarm probability of an SDE value.

    Assumes the periodogram noise is Gaussian and that the spectrum
    contains ``n_effective`` independent frequencies. Evaluated in log
    space for numerical stability at high significance.

    Args:
        sde: Signal Detection Efficiency of the peak.
        n_effective: Number of independent trial frequencies.

    Returns:
        The false-alarm probability in ``[0, 1]`` (``nan`` for
        non-finite input).
    """
    if not math.isfinite(sde) or n_effective <= 0:
        return math.nan
    p_single = 0.5 * math.erfc(sde / math.sqrt(2.0))
    if p_single <= 0.0:
        return 0.0
    return float(-np.expm1(n_effective * math.log1p(-min(p_single, 1.0 - 1e-16))))


@METRICS_COMPUTERS.register("standard")
class StandardMetricsComputer:
    """Computes the detection metrics attached to every candidate.

    Attributes:
        robust_sde: Standardize SDE with median/MAD instead of
            mean/std.
        detrend_window_fraction: Running-median continuum window as a
            fraction of the spectrum length; ``None`` disables
            continuum removal.
        compute_fap: Whether to evaluate the analytic FAP
            approximation (stored as ``nan`` when disabled).
        odd_even_sigma: Threshold on the odd/even depth difference
            significance above which the candidate is flagged.
    """

    def __init__(
        self,
        robust_sde: bool = True,
        detrend_window_fraction: float | None = 0.02,
        compute_fap: bool = True,
        odd_even_sigma: float = 3.0,
    ) -> None:
        """Initializes the computer.

        Args:
            robust_sde: Use the robust SDE baseline.
            detrend_window_fraction: Continuum window fraction, or
                ``None`` to standardize the raw power spectrum.
            compute_fap: Enable the analytic FAP approximation.
            odd_even_sigma: Odd/even mismatch flag threshold.

        Raises:
            PipelineError: If a threshold is out of range.
        """
        if odd_even_sigma <= 0:
            raise PipelineError(f"odd_even_sigma must be > 0, got {odd_even_sigma}.")
        if detrend_window_fraction is not None and not 0 < detrend_window_fraction < 1:
            raise PipelineError(
                "detrend_window_fraction must be in (0, 1) or null, got "
                f"{detrend_window_fraction}."
            )
        self.robust_sde = robust_sde
        self.detrend_window_fraction = detrend_window_fraction
        self.compute_fap = compute_fap
        self.odd_even_sigma = float(odd_even_sigma)

    def compute(
        self,
        periodogram: Periodogram,
        index: int,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluates all metrics for one periodogram peak.

        Args:
            periodogram: The full BLS spectrum.
            index: Peak index within the periodogram.
            stats: Astropy ``compute_stats`` output for this solution.

        Returns:
            A dictionary with keys ``snr``, ``sde``, ``power``,
            ``fap``, ``depth``, ``depth_err``, ``n_transits``,
            ``n_expected_transits``, ``quality_flags``, and
            ``diagnostics`` (odd/even depths, harmonic comparison).
        """
        power = float(periodogram.power[index])
        spectrum = periodogram.power
        if self.detrend_window_fraction is not None:
            spectrum = detrend_power(spectrum, self.detrend_window_fraction)
        sde = sde_of_peak(spectrum, index, robust=self.robust_sde)

        depth, depth_err = (float(v) for v in stats["depth"])
        snr = depth / depth_err if depth_err > 0 else math.nan

        fap = math.nan
        if self.compute_fap:
            grid = periodogram.meta.get("grid", {})
            baseline = float(grid.get("baseline_days", 0.0))
            f_span = float(
                np.abs(periodogram.frequencies[-1] - periodogram.frequencies[0])
            )
            fap = gaussian_fap(sde, baseline * f_span)

        per_transit_count = np.asarray(stats["per_transit_count"])
        n_expected = int(per_transit_count.size)
        n_observed = int(np.count_nonzero(per_transit_count > 0))

        flags: list[str] = []
        depth_odd, err_odd = (float(v) for v in stats["depth_odd"])
        depth_even, err_even = (float(v) for v in stats["depth_even"])
        odd_even_err = math.hypot(err_odd, err_even)
        if odd_even_err > 0:
            odd_even_significance = abs(depth_odd - depth_even) / odd_even_err
            if odd_even_significance > self.odd_even_sigma:
                flags.append("odd_even_mismatch")
        else:
            odd_even_significance = math.nan
        if float(stats["harmonic_delta_log_likelihood"]) > 0:
            flags.append("sinusoidal_preferred")
        if n_expected > 0 and n_observed / n_expected < 0.75:
            flags.append("partial_transit_coverage")

        return {
            "snr": snr,
            "sde": sde,
            "power": power,
            "fap": fap,
            "depth": depth,
            "depth_err": depth_err,
            "n_transits": n_observed,
            "n_expected_transits": n_expected,
            "quality_flags": tuple(flags),
            "diagnostics": {
                "depth_odd": depth_odd,
                "depth_even": depth_even,
                "odd_even_significance": odd_even_significance,
                "harmonic_delta_log_likelihood": float(
                    stats["harmonic_delta_log_likelihood"]
                ),
                "per_transit_count": per_transit_count.tolist(),
                "periodogram_depth_snr": float(periodogram.depth_snr[index]),
            },
        }

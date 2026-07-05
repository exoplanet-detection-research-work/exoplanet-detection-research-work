"""Significant-peak detection in BLS periodograms.

Peaks are located with :func:`scipy.signal.find_peaks` on the
continuum-subtracted power spectrum (see
:func:`exodet.tce.metrics.detrend_power`): the BLS objective rises
slowly toward long periods, and thresholding the raw spectrum would
hide genuine short-period peaks under the long-period ramp. The
spectrum is uniformly sampled in frequency, so the sample-based
minimum separation maps directly onto a frequency separation. The
detection threshold is adaptive by default: it is expressed in robust
standard deviations (1.4826 x MAD) above the median of the detrended
power, making it insensitive to the strong peaks it is meant to find.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.tce.candidate import Periodogram
from exodet.tce.metrics import detrend_power

__all__ = ["PEAK_DETECTORS", "ProminencePeakDetector"]

logger = logging.getLogger(__name__)

PEAK_DETECTORS: Registry["ProminencePeakDetector"] = Registry("TCE peak detector")

_MAD_TO_STD = 1.4826


@PEAK_DETECTORS.register("prominence")
class ProminencePeakDetector:
    """Finds statistically significant periodogram peaks.

    Attributes:
        threshold_sigma: Adaptive height threshold in robust standard
            deviations above the median of the detrended power.
        detrend_window_fraction: Running-median continuum window as a
            fraction of the spectrum length; ``None`` disables
            continuum removal.
        min_power: Optional absolute floor on the *raw* peak power
            (applied in addition to the adaptive threshold).
        min_prominence_sigma: Required topographic prominence in robust
            standard deviations.
        min_frequency_separation: Minimum spacing between peaks in
            cycles/day; ``None`` uses ``0.25 / baseline`` (a quarter of
            the Fourier resolution — BLS peaks are much narrower than a
            Fourier mode, and nearby duplicates of the same signal are
            caught later by the ratio-based harmonic rejecter).
        max_candidates: Upper bound on returned peaks (strongest kept).
    """

    def __init__(
        self,
        threshold_sigma: float = 5.0,
        detrend_window_fraction: float | None = 0.02,
        min_power: float | None = None,
        min_prominence_sigma: float = 1.0,
        min_frequency_separation: float | None = None,
        max_candidates: int = 5,
    ) -> None:
        """Initializes the detector.

        Args:
            threshold_sigma: Adaptive height threshold; must be positive.
            detrend_window_fraction: Continuum window fraction in
                ``(0, 1)``, or ``None`` to threshold the raw spectrum.
            min_power: Optional absolute raw-power floor.
            min_prominence_sigma: Prominence requirement; non-negative.
            min_frequency_separation: Minimum peak spacing in
                cycles/day, or ``None`` for ``0.25 / baseline``.
            max_candidates: Maximum number of peaks; must be positive.

        Raises:
            PipelineError: If any parameter is out of range.
        """
        if threshold_sigma <= 0:
            raise PipelineError(f"threshold_sigma must be > 0, got {threshold_sigma}.")
        if min_prominence_sigma < 0:
            raise PipelineError(
                f"min_prominence_sigma must be >= 0, got {min_prominence_sigma}."
            )
        if min_frequency_separation is not None and min_frequency_separation <= 0:
            raise PipelineError(
                "min_frequency_separation must be > 0 when given, got "
                f"{min_frequency_separation}."
            )
        if max_candidates < 1:
            raise PipelineError(f"max_candidates must be >= 1, got {max_candidates}.")
        if detrend_window_fraction is not None and not 0 < detrend_window_fraction < 1:
            raise PipelineError(
                "detrend_window_fraction must be in (0, 1) or null, got "
                f"{detrend_window_fraction}."
            )
        self.detrend_window_fraction = detrend_window_fraction
        self.threshold_sigma = float(threshold_sigma)
        self.min_power = None if min_power is None else float(min_power)
        self.min_prominence_sigma = float(min_prominence_sigma)
        self.min_frequency_separation = min_frequency_separation
        self.max_candidates = int(max_candidates)

    def detect(self, periodogram: Periodogram) -> npt.NDArray[np.int_]:
        """Locates significant peaks in a periodogram.

        Args:
            periodogram: The BLS spectrum (uniform in frequency).

        Returns:
            Indices of significant peaks, ordered by decreasing power
            and capped at ``max_candidates``. Empty when nothing is
            significant (e.g. constant flux or pure noise below the
            threshold).
        """
        power = periodogram.power
        if self.detrend_window_fraction is not None:
            power = detrend_power(power, self.detrend_window_fraction)
        center = float(np.median(power))
        scale = _MAD_TO_STD * float(np.median(np.abs(power - center)))
        if scale == 0.0:
            # A flat spectrum has no information; adaptive thresholding
            # would degenerate to "everything is a peak".
            logger.info(
                "Target %s: periodogram has zero dispersion; no peaks.",
                periodogram.meta.get("target_id", "?"),
            )
            return np.array([], dtype=np.int_)

        height = center + self.threshold_sigma * scale

        frequencies = periodogram.frequencies
        df = float(np.abs(np.median(np.diff(frequencies))))
        if self.min_frequency_separation is None:
            baseline = periodogram.meta.get("grid", {}).get("baseline_days")
            separation = 0.25 / baseline if baseline else 10.0 * df
        else:
            separation = self.min_frequency_separation
        distance = max(1, int(round(separation / df)))

        indices, _ = find_peaks(
            power,
            height=height,
            prominence=self.min_prominence_sigma * scale,
            distance=distance,
        )
        if self.min_power is not None:
            indices = indices[periodogram.power[indices] >= self.min_power]
        indices = indices[np.argsort(power[indices])[::-1]][: self.max_candidates]
        logger.info(
            "Target %s: %d significant peak(s) above %.3f "
            "(median %.3f + %.1f sigma).",
            periodogram.meta.get("target_id", "?"),
            indices.size,
            height,
            center,
            self.threshold_sigma,
        )
        return indices.astype(np.int_)

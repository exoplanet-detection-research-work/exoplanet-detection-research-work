"""Global and local view generation (Modules 2 and 3).

Follows the Astronet / ExoMiner methodology: the *global view* bins the
entire orbit (default 2001 bins over phase −0.5 … +0.5) while the
*local view* zooms on the transit (default 401 bins over ±2 transit
durations), preserving high resolution across ingress and egress.

Binning is exact and fully vectorized: samples are lex-sorted by
(bin, flux), so each bin's values are a contiguous *sorted* run whose
median is read off directly from offset arithmetic — no Python loops,
no approximation. Weighted averaging uses inverse-variance weights via
``np.bincount``. Empty bins are filled by configurable interpolation
(linear / cubic / nearest) over the non-empty bin centers, and views
whose empty-bin fraction exceeds a threshold are rejected as malformed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from exodet.exceptions import DataError, PipelineError
from exodet.registry import Registry
from exodet.representation.containers import PhaseFoldedCurve, View

__all__ = [
    "VIEW_BUILDERS",
    "GlobalViewGenerator",
    "LocalViewGenerator",
    "bin_folded_curve",
]

logger = logging.getLogger(__name__)

VIEW_BUILDERS: Registry[object] = Registry("view builder")

_STATISTICS = ("median", "mean", "weighted_mean")
_INTERPOLATIONS = ("linear", "cubic", "nearest")
_NORMALIZATIONS = ("none", "center", "astronet")


def bin_folded_curve(
    phase: npt.NDArray[np.float64],
    flux: npt.NDArray[np.float64],
    edges: npt.NDArray[np.float64],
    statistic: str = "median",
    flux_err: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Bins a folded curve with an exact, vectorized statistic.

    Args:
        phase: Sample phases (any order).
        flux: Sample flux values.
        edges: Monotonic bin edges, length ``n_bins + 1``.
        statistic: ``"median"``, ``"mean"``, or ``"weighted_mean"``
            (inverse-variance weights from ``flux_err``).
        flux_err: Flux uncertainties; required for weighted averaging.

    Returns:
        A tuple ``(values, counts)``: per-bin statistic (NaN for empty
        bins) and per-bin sample counts.

    Raises:
        PipelineError: If the statistic is unknown or weights are
            missing/invalid for weighted averaging.
    """
    if statistic not in _STATISTICS:
        raise PipelineError(
            f"Unknown binning statistic '{statistic}'. Available: {_STATISTICS}."
        )
    n_bins = len(edges) - 1
    inside = (phase >= edges[0]) & (phase < edges[-1])
    phase = phase[inside]
    flux = flux[inside]
    if flux_err is not None:
        flux_err = flux_err[inside]

    bin_index = np.clip(
        np.searchsorted(edges, phase, side="right") - 1, 0, n_bins - 1
    )
    counts = np.bincount(bin_index, minlength=n_bins)
    values = np.full(n_bins, np.nan)
    occupied = counts > 0
    if not occupied.any():
        return values, counts

    if statistic == "median":
        order = np.lexsort((flux, bin_index))
        sorted_flux = flux[order]
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        lo = starts + (counts - 1) // 2
        hi = starts + counts // 2
        safe_lo = np.where(occupied, lo, 0)
        safe_hi = np.where(occupied, hi, 0)
        medians = 0.5 * (sorted_flux[safe_lo] + sorted_flux[safe_hi])
        values[occupied] = medians[occupied]
    elif statistic == "mean":
        sums = np.bincount(bin_index, weights=flux, minlength=n_bins)
        values[occupied] = sums[occupied] / counts[occupied]
    else:  # weighted_mean
        if flux_err is None:
            raise PipelineError(
                "weighted_mean binning requires flux uncertainties."
            )
        weights = np.zeros_like(flux)
        valid = np.isfinite(flux_err) & (flux_err > 0)
        weights[valid] = 1.0 / flux_err[valid] ** 2
        weight_sums = np.bincount(bin_index, weights=weights, minlength=n_bins)
        value_sums = np.bincount(
            bin_index, weights=weights * flux, minlength=n_bins
        )
        positive = weight_sums > 0
        values[positive] = value_sums[positive] / weight_sums[positive]
        # Bins whose every sample has invalid errors fall back to mean.
        fallback = occupied & ~positive
        if fallback.any():
            sums = np.bincount(bin_index, weights=flux, minlength=n_bins)
            values[fallback] = sums[fallback] / counts[fallback]
    return values, counts


def _fill_empty_bins(
    values: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    method: str,
) -> npt.NDArray[np.float64]:
    """Interpolates NaN bins from the occupied ones.

    Args:
        values: Per-bin values with NaN marking empty bins.
        centers: Bin center positions.
        method: ``"linear"``, ``"cubic"``, or ``"nearest"``.

    Returns:
        The filled array (a copy when filling occurs).
    """
    empty = ~np.isfinite(values)
    if not empty.any():
        return values
    filled = values.copy()
    good = ~empty
    x_good = centers[good]
    y_good = values[good]
    if method == "linear" or x_good.size < 4:
        filled[empty] = np.interp(centers[empty], x_good, y_good)
    elif method == "cubic":
        spline = CubicSpline(x_good, y_good, extrapolate=False)
        interpolated = spline(centers[empty])
        # Outside the occupied range the spline is undefined: hold the
        # nearest edge value instead of extrapolating polynomials.
        interpolated = np.where(
            np.isfinite(interpolated),
            interpolated,
            np.interp(centers[empty], x_good, y_good),
        )
        filled[empty] = interpolated
    else:  # nearest
        nearest = np.searchsorted(x_good, centers[empty], side="left")
        nearest = np.clip(nearest, 1, x_good.size - 1)
        left = x_good[nearest - 1]
        right = x_good[nearest]
        use_left = np.abs(centers[empty] - left) <= np.abs(right - centers[empty])
        filled[empty] = np.where(use_left, y_good[nearest - 1], y_good[nearest])
    return filled


def _normalize_view(
    values: npt.NDArray[np.float64], method: str
) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    """Normalizes a view for ML consumption.

    Args:
        values: Filled per-bin values.
        method: ``"none"``, ``"center"`` (median → 0), or
            ``"astronet"`` (median → 0, minimum → −1).

    Returns:
        The normalized values and the statistics used (for inversion).
    """
    if method == "none":
        return values, {}
    median = float(np.median(values))
    centered = values - median
    if method == "center":
        return centered, {"median": median}
    minimum = float(centered.min())
    scale = -minimum if minimum < 0 else 1.0
    if scale == 0.0:
        scale = 1.0
    return centered / scale, {"median": median, "scale": scale}


class _BaseViewGenerator:
    """Shared machinery of the global and local view generators."""

    kind = "base"

    def __init__(
        self,
        n_bins: int,
        statistic: str,
        interpolation: str,
        normalization: str,
        max_empty_fraction: float,
    ) -> None:
        if n_bins < 3:
            raise PipelineError(f"n_bins must be >= 3, got {n_bins}.")
        if statistic not in _STATISTICS:
            raise PipelineError(
                f"Unknown statistic '{statistic}'. Available: {_STATISTICS}."
            )
        if interpolation not in _INTERPOLATIONS:
            raise PipelineError(
                f"Unknown interpolation '{interpolation}'. "
                f"Available: {_INTERPOLATIONS}."
            )
        if normalization not in _NORMALIZATIONS:
            raise PipelineError(
                f"Unknown normalization '{normalization}'. "
                f"Available: {_NORMALIZATIONS}."
            )
        if not 0 < max_empty_fraction <= 1:
            raise PipelineError(
                f"max_empty_fraction must be in (0, 1], got {max_empty_fraction}."
            )
        self.n_bins = int(n_bins)
        self.statistic = statistic
        self.interpolation = interpolation
        self.normalization = normalization
        self.max_empty_fraction = float(max_empty_fraction)

    def _edges(self, folded: PhaseFoldedCurve) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def generate(self, folded: PhaseFoldedCurve) -> View:
        """Builds the view from a folded curve.

        Args:
            folded: The phase-folded, transit-centered curve.

        Returns:
            The binned, filled, normalized view.

        Raises:
            DataError: If the view is malformed (too many empty bins or
                no data at all).
        """
        edges = self._edges(folded)
        centers = 0.5 * (edges[:-1] + edges[1:])
        values, counts = bin_folded_curve(
            folded.phase,
            folded.flux,
            edges,
            statistic=self.statistic,
            flux_err=folded.flux_err,
        )
        n_empty = int(np.count_nonzero(counts == 0))
        empty_fraction = n_empty / self.n_bins
        if empty_fraction >= 1.0:
            raise DataError(
                f"Candidate {folded.candidate_id}: {self.kind} view has no "
                "data in any bin."
            )
        if empty_fraction > self.max_empty_fraction:
            raise DataError(
                f"Candidate {folded.candidate_id}: {self.kind} view rejected "
                f"({empty_fraction:.1%} empty bins > "
                f"{self.max_empty_fraction:.1%} allowed)."
            )

        interpolation = self.interpolation
        # Adaptive fallback: high-order interpolation over sparse
        # support oscillates, so degrade gracefully.
        if interpolation == "cubic" and empty_fraction > 0.25:
            interpolation = "linear"
        filled = _fill_empty_bins(values, centers, interpolation)
        normalized, stats = _normalize_view(filled, self.normalization)

        meta: dict[str, Any] = {
            "statistic": self.statistic,
            "normalization": self.normalization,
            "normalization_stats": stats,
            "n_samples": int(counts.sum()),
            "median_samples_per_bin": float(np.median(counts[counts > 0])),
            "requested_interpolation": self.interpolation,
        }
        return View(
            kind=self.kind,
            values=normalized,
            bin_centers=centers,
            n_empty_bins=n_empty,
            interpolation=interpolation,
            meta=meta,
        )


@VIEW_BUILDERS.register("global")
class GlobalViewGenerator(_BaseViewGenerator):
    """Bins the full orbit into a fixed-length global view.

    Attributes:
        n_bins: Number of bins (default 2001).
        phase_min: Lower phase edge (default −0.5).
        phase_max: Upper phase edge (default +0.5).
        statistic: Binning statistic.
        interpolation: Empty-bin fill method.
        normalization: View normalization for ML consumption.
        max_empty_fraction: Rejection threshold on empty bins.
    """

    kind = "global"

    def __init__(
        self,
        n_bins: int = 2001,
        phase_min: float = -0.5,
        phase_max: float = 0.5,
        statistic: str = "median",
        interpolation: str = "linear",
        normalization: str = "astronet",
        max_empty_fraction: float = 0.5,
    ) -> None:
        """Initializes the generator.

        Args:
            n_bins: Number of bins; >= 3.
            phase_min: Lower phase edge; below ``phase_max``.
            phase_max: Upper phase edge.
            statistic: ``median`` | ``mean`` | ``weighted_mean``.
            interpolation: ``linear`` | ``cubic`` | ``nearest``.
            normalization: ``none`` | ``center`` | ``astronet``.
            max_empty_fraction: Rejection threshold in ``(0, 1]``.

        Raises:
            PipelineError: If any parameter is invalid.
        """
        super().__init__(
            n_bins, statistic, interpolation, normalization, max_empty_fraction
        )
        if not -0.5 <= phase_min < phase_max <= 0.5:
            raise PipelineError(
                f"Require -0.5 <= phase_min < phase_max <= 0.5, got "
                f"{phase_min}, {phase_max}."
            )
        self.phase_min = float(phase_min)
        self.phase_max = float(phase_max)

    def _edges(self, folded: PhaseFoldedCurve) -> npt.NDArray[np.float64]:
        return np.linspace(self.phase_min, self.phase_max, self.n_bins + 1)


@VIEW_BUILDERS.register("local")
class LocalViewGenerator(_BaseViewGenerator):
    """Bins the transit neighbourhood into a fixed-length local view.

    The window spans ``±window_durations`` transit durations around
    phase 0 (clipped to the folded range), keeping ingress and egress
    resolved at roughly ``n_bins / (2 * window_durations)`` bins per
    duration.

    Attributes:
        n_bins: Number of bins (default 401).
        window_durations: Half-width in transit durations (default 2).
        statistic: Binning statistic.
        interpolation: Empty-bin fill method.
        normalization: View normalization for ML consumption.
        max_empty_fraction: Rejection threshold on empty bins.
    """

    kind = "local"

    def __init__(
        self,
        n_bins: int = 401,
        window_durations: float = 2.0,
        statistic: str = "median",
        interpolation: str = "linear",
        normalization: str = "astronet",
        max_empty_fraction: float = 0.5,
    ) -> None:
        """Initializes the generator.

        Args:
            n_bins: Number of bins; >= 3.
            window_durations: Half-width in durations; positive.
            statistic: ``median`` | ``mean`` | ``weighted_mean``.
            interpolation: ``linear`` | ``cubic`` | ``nearest``.
            normalization: ``none`` | ``center`` | ``astronet``.
            max_empty_fraction: Rejection threshold in ``(0, 1]``.

        Raises:
            PipelineError: If any parameter is invalid.
        """
        super().__init__(
            n_bins, statistic, interpolation, normalization, max_empty_fraction
        )
        if window_durations <= 0:
            raise PipelineError(
                f"window_durations must be > 0, got {window_durations}."
            )
        self.window_durations = float(window_durations)

    def _edges(self, folded: PhaseFoldedCurve) -> npt.NDArray[np.float64]:
        half = min(self.window_durations * folded.duty_cycle, 0.5)
        return np.linspace(-half, half, self.n_bins + 1)

"""Cadence-cleaning preprocessors: quality filtering, NaN handling, clipping.

All three steps are fully vectorized: masks are computed with NumPy
boolean algebra and applied once through
:func:`~exodet.preprocessing.common.mask_light_curve`, so no Python
loops run over cadences.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.preprocessing.base import PREPROCESSORS, BasePreprocessor
from exodet.preprocessing.common import mask_light_curve

__all__ = ["QualityFlagFilter", "NaNRemover", "SigmaClipper", "TESS_QUALITY_FLAGS"]

logger = logging.getLogger(__name__)

TESS_QUALITY_FLAGS: Final[dict[str, int]] = {
    "attitude_tweak": 1,
    "safe_mode": 2,
    "coarse_point": 4,
    "earth_point": 8,
    "argabrightening": 16,
    "reaction_wheel_desaturation": 32,
    "aperture_cosmic_ray": 64,
    "manual_exclude": 128,
    "discontinuity": 256,
    "impulsive_outlier": 512,
    "collateral_cosmic_ray": 1024,
    "straylight": 2048,
}
"""TESS quality flag bits (TESS Science Data Products Description Document)."""

_BITMASK_PRESETS: Final[dict[str, int]] = {
    # Conservative: spacecraft anomalies and manually excluded cadences.
    "default": (
        TESS_QUALITY_FLAGS["attitude_tweak"]
        | TESS_QUALITY_FLAGS["safe_mode"]
        | TESS_QUALITY_FLAGS["coarse_point"]
        | TESS_QUALITY_FLAGS["earth_point"]
        | TESS_QUALITY_FLAGS["argabrightening"]
        | TESS_QUALITY_FLAGS["reaction_wheel_desaturation"]
        | TESS_QUALITY_FLAGS["manual_exclude"]
    ),
    # Additionally reject cosmic rays, discontinuities, and outliers.
    "hard": (
        TESS_QUALITY_FLAGS["attitude_tweak"]
        | TESS_QUALITY_FLAGS["safe_mode"]
        | TESS_QUALITY_FLAGS["coarse_point"]
        | TESS_QUALITY_FLAGS["earth_point"]
        | TESS_QUALITY_FLAGS["argabrightening"]
        | TESS_QUALITY_FLAGS["reaction_wheel_desaturation"]
        | TESS_QUALITY_FLAGS["aperture_cosmic_ray"]
        | TESS_QUALITY_FLAGS["manual_exclude"]
        | TESS_QUALITY_FLAGS["discontinuity"]
        | TESS_QUALITY_FLAGS["impulsive_outlier"]
        | TESS_QUALITY_FLAGS["collateral_cosmic_ray"]
    ),
    # Reject every flagged cadence.
    "hardest": (1 << 12) - 1,
}


@PREPROCESSORS.register("quality_filter")
class QualityFlagFilter(BasePreprocessor):
    """Removes cadences whose TESS quality flags intersect a bitmask.

    Expects integer flags in ``meta["quality"]`` (aligned per cadence,
    as delivered by TESS SPOC light-curve files). Curves without
    quality information pass through unchanged with a warning.

    Attributes:
        bitmask: Integer bitmask; a cadence is rejected when
            ``quality & bitmask != 0``.
    """

    def __init__(self, bitmask: int | str = "default") -> None:
        """Initializes the filter.

        Args:
            bitmask: Either an explicit integer bitmask or one of the
                presets ``"default"``, ``"hard"``, ``"hardest"``.

        Raises:
            PipelineError: If a preset name is unknown or the bitmask
                is negative.
        """
        if isinstance(bitmask, str):
            try:
                self.bitmask = _BITMASK_PRESETS[bitmask.lower()]
            except KeyError:
                raise PipelineError(
                    f"Unknown quality bitmask preset '{bitmask}'. "
                    f"Available: {sorted(_BITMASK_PRESETS)}."
                ) from None
        else:
            if bitmask < 0:
                raise PipelineError(f"Quality bitmask must be >= 0, got {bitmask}.")
            self.bitmask = int(bitmask)

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Drops cadences with flagged quality bits.

        Args:
            light_curve: Input curve, ideally carrying ``meta["quality"]``.

        Returns:
            The curve restricted to good-quality cadences.
        """
        step = f"{self.name}(bitmask={self.bitmask})"
        quality = light_curve.meta.get("quality")
        if quality is None:
            logger.warning(
                "Target %s has no quality flags; %s is a no-op.",
                light_curve.target_id,
                self.name,
            )
            return light_curve.replace_flux(light_curve.flux, step_name=step)

        flags = np.asarray(quality)
        if flags.shape != (len(light_curve),):
            raise PipelineError(
                f"meta['quality'] shape {flags.shape} does not match curve "
                f"length {len(light_curve)}."
            )
        keep = (flags.astype(np.int64) & self.bitmask) == 0
        n_bad = int(np.count_nonzero(~keep))
        logger.info(
            "Target %s: removing %d/%d flagged cadences (bitmask=%d).",
            light_curve.target_id,
            n_bad,
            len(light_curve),
            self.bitmask,
        )
        return mask_light_curve(light_curve, keep, step)


@PREPROCESSORS.register("nan_removal")
class NaNRemover(BasePreprocessor):
    """Handles non-finite cadences with a configurable strategy.

    Cadences with non-finite *time* are always dropped (their location
    is unknown, so no fill is meaningful). Non-finite flux is then
    handled per strategy. Non-finite flux uncertainties are replaced by
    the median finite uncertainty.

    Attributes:
        strategy: One of ``"drop"``, ``"fill_median"``,
            ``"fill_interpolate"``.
    """

    _STRATEGIES = ("drop", "fill_median", "fill_interpolate")

    def __init__(self, strategy: str = "drop") -> None:
        """Initializes the remover.

        Args:
            strategy: ``"drop"`` removes bad cadences,
                ``"fill_median"`` replaces bad flux with the median,
                ``"fill_interpolate"`` replaces bad flux by linear
                interpolation over neighbouring good cadences.

        Raises:
            PipelineError: If the strategy is unknown.
        """
        if strategy not in self._STRATEGIES:
            raise PipelineError(
                f"Unknown NaN strategy '{strategy}'. Available: {self._STRATEGIES}."
            )
        self.strategy = strategy

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Removes or fills non-finite values.

        Args:
            light_curve: The input curve.

        Returns:
            A curve whose ``time``/``flux`` (and ``flux_err`` when
            present) are entirely finite.

        Raises:
            PipelineError: If no finite flux remains to fill from.
        """
        step = f"{self.name}(strategy={self.strategy})"
        time_ok = np.isfinite(light_curve.time)
        flux_ok = np.isfinite(light_curve.flux)

        if self.strategy == "drop":
            curve = mask_light_curve(light_curve, time_ok & flux_ok, step)
            n_dropped = len(light_curve) - len(curve)
            curve.meta["nan_removal"] = {"n_dropped": n_dropped, "n_filled": 0}
        else:
            curve = mask_light_curve(light_curve, time_ok, step)
            flux_ok = np.isfinite(curve.flux)
            n_bad = int(np.count_nonzero(~flux_ok))
            if n_bad and not flux_ok.any():
                raise PipelineError(
                    f"Target {curve.target_id}: no finite flux to fill from."
                )
            if n_bad:
                flux = curve.flux.copy()
                if self.strategy == "fill_median":
                    flux[~flux_ok] = np.median(flux[flux_ok])
                else:
                    flux[~flux_ok] = np.interp(
                        curve.time[~flux_ok], curve.time[flux_ok], flux[flux_ok]
                    )
                curve.flux = flux
            curve.meta["nan_removal"] = {
                "n_dropped": len(light_curve) - len(curve),
                "n_filled": n_bad,
            }

        if curve.flux_err is not None:
            err_ok = np.isfinite(curve.flux_err)
            if not err_ok.all():
                if not err_ok.any():
                    raise PipelineError(
                        f"Target {curve.target_id}: all flux uncertainties "
                        "are non-finite."
                    )
                err = curve.flux_err.copy()
                err[~err_ok] = np.median(err[err_ok])
                curve.flux_err = err

        logger.info(
            "Target %s: NaN handling (%s) dropped %d, filled %d cadences.",
            curve.target_id,
            self.strategy,
            curve.meta["nan_removal"]["n_dropped"],
            curve.meta["nan_removal"]["n_filled"],
        )
        return curve


@PREPROCESSORS.register("sigma_clip")
class SigmaClipper(BasePreprocessor):
    """Iterative sigma clipping of flux outliers.

    Uses a robust centre (median) and a robust scale (1.4826 x MAD,
    the Gaussian-consistent estimator) by default, so the threshold is
    not inflated by the outliers themselves. Clipping repeats until
    convergence or ``max_iterations``. Removed points are recorded in
    ``meta["clipped_time"]`` / ``meta["clipped_flux"]`` for plotting.

    Attributes:
        sigma: Rejection threshold in robust standard deviations.
        max_iterations: Upper bound on clipping iterations.
        robust: If ``True`` use MAD-based scale, else the sample std.
        clip_lower: Whether to reject downward outliers. Disable to
            protect deep transit points from being clipped.
    """

    _MAD_TO_STD: Final[float] = 1.4826

    def __init__(
        self,
        sigma: float = 5.0,
        max_iterations: int = 10,
        robust: bool = True,
        clip_lower: bool = True,
    ) -> None:
        """Initializes the clipper.

        Args:
            sigma: Rejection threshold; must be positive.
            max_iterations: Maximum clipping passes; must be positive.
            robust: Use MAD-based scale instead of the standard deviation.
            clip_lower: Also clip points below the lower threshold.

        Raises:
            PipelineError: If ``sigma`` or ``max_iterations`` is invalid.
        """
        if sigma <= 0:
            raise PipelineError(f"sigma must be > 0, got {sigma}.")
        if max_iterations <= 0:
            raise PipelineError(f"max_iterations must be > 0, got {max_iterations}.")
        self.sigma = float(sigma)
        self.max_iterations = int(max_iterations)
        self.robust = robust
        self.clip_lower = clip_lower

    def _scale(self, values: np.ndarray, center: float) -> float:
        """Estimates the dispersion of the surviving flux values.

        Args:
            values: Currently surviving flux values.
            center: Current centre estimate.

        Returns:
            The scale estimate (robust or classical).
        """
        if self.robust:
            return self._MAD_TO_STD * float(np.median(np.abs(values - center)))
        return float(np.std(values))

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Clips outliers until convergence.

        Args:
            light_curve: The input curve.

        Returns:
            The curve with outlying cadences removed and the clipped
            points stored in metadata.

        Raises:
            PipelineError: If the flux dispersion is zero (constant
                flux cannot be clipped meaningfully but passes through).
        """
        step = f"{self.name}(sigma={self.sigma})"
        flux = light_curve.flux
        keep = np.ones(flux.shape, dtype=bool)
        # Non-finite cadences are missing data, not outliers: they are
        # excluded from the statistics and never clipped here (a
        # NaNRemover stage decides their fate).
        finite = np.isfinite(flux)
        if not finite.any():
            raise PipelineError(
                f"Target {light_curve.target_id}: no finite flux to clip."
            )

        for iteration in range(self.max_iterations):
            surviving = flux[keep & finite]
            center = float(np.median(surviving))
            scale = self._scale(surviving, center)
            if scale == 0.0:
                logger.debug(
                    "Target %s: zero flux dispersion at iteration %d; stopping.",
                    light_curve.target_id,
                    iteration,
                )
                break
            deviation = flux - center
            if self.clip_lower:
                inlier = np.abs(deviation) <= self.sigma * scale
            else:
                inlier = deviation <= self.sigma * scale
            new_keep = keep & (inlier | ~finite)
            if new_keep.sum() == keep.sum():
                break
            keep = new_keep

        clipped = ~keep
        result = mask_light_curve(light_curve, keep, step)
        result.meta["clipped_time"] = light_curve.time[clipped].copy()
        result.meta["clipped_flux"] = light_curve.flux[clipped].copy()
        logger.info(
            "Target %s: sigma clip removed %d/%d cadences.",
            light_curve.target_id,
            int(clipped.sum()),
            len(light_curve),
        )
        return result

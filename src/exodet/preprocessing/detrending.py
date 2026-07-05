"""Stellar variability removal (detrending) via wotan.

Wraps `wotan <https://github.com/hippke/wotan>`_ time-windowed sliders
to remove stellar variability and residual instrumental trends while
preserving transit signals. The removed trend is stored per cadence in
``meta["trend"]`` so raw-versus-detrended figures and inverse
transforms remain possible.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.preprocessing.base import PREPROCESSORS, BasePreprocessor
from exodet.preprocessing.common import mask_light_curve

__all__ = ["WotanDetrender"]

logger = logging.getLogger(__name__)

_FlattenFn = Callable[..., tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]


def _get_flatten() -> _FlattenFn:
    """Imports :func:`wotan.flatten` lazily.

    Wotan triggers numba JIT machinery at import time, which costs
    seconds; deferring the import keeps CLI startup and non-detrending
    pipelines fast.

    Returns:
        The ``wotan.flatten`` callable.

    Raises:
        PipelineError: If wotan is not installed.
    """
    try:
        from wotan import flatten
    except ImportError as exc:
        raise PipelineError(
            "wotan is required for detrending; install it with "
            "'pip install wotan'."
        ) from exc
    return flatten


@PREPROCESSORS.register("wotan_detrend")
class WotanDetrender(BasePreprocessor):
    """Removes stellar variability with a wotan time-windowed slider.

    The output flux is the input divided by the estimated trend, i.e.
    relative flux centred on 1. Uncertainties are scaled by the same
    trend. Cadences where the trend cannot be estimated (short
    segments, extreme edges) yield non-finite flux and are dropped.

    Attributes:
        method: ``"biweight"``, ``"median"``, or ``"lowess"``.
        window_length_days: Detrending window in days. Must be at
            least ~3x the longest transit duration of interest to
            avoid distorting transit shapes.
        break_tolerance_days: Split the curve into independent segments
            at gaps longer than this, preventing interpolation of the
            trend across data gaps.
        cval: Tuning constant forwarded to wotan (biweight only).
    """

    _METHODS = ("biweight", "median", "lowess")

    def __init__(
        self,
        method: str = "biweight",
        window_length_days: float = 0.5,
        break_tolerance_days: float = 0.5,
        cval: float | None = None,
    ) -> None:
        """Initializes the detrender.

        Args:
            method: Sliding estimator of the trend.
            window_length_days: Window size in days; must be positive.
            break_tolerance_days: Segment-splitting gap threshold in
                days; must be positive.
            cval: Optional wotan tuning constant (e.g. biweight ``cval``).

        Raises:
            PipelineError: If the method or window parameters are invalid.
        """
        if method not in self._METHODS:
            raise PipelineError(
                f"Unknown detrending method '{method}'. Available: {self._METHODS}."
            )
        if window_length_days <= 0:
            raise PipelineError(
                f"window_length_days must be > 0, got {window_length_days}."
            )
        if break_tolerance_days <= 0:
            raise PipelineError(
                f"break_tolerance_days must be > 0, got {break_tolerance_days}."
            )
        self.method = method
        self.window_length_days = float(window_length_days)
        self.break_tolerance_days = float(break_tolerance_days)
        self.cval = cval

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Estimates and divides out the stellar variability trend.

        Args:
            light_curve: The input curve with finite, sorted times.

        Returns:
            The detrended curve (relative flux ~1) with the trend
            stored in ``meta["trend"]`` and untrendable cadences removed.

        Raises:
            PipelineError: If wotan fails or no cadence survives.
        """
        flatten = _get_flatten()
        kwargs: dict[str, Any] = {
            "method": self.method,
            "window_length": self.window_length_days,
            "break_tolerance": self.break_tolerance_days,
            "return_trend": True,
        }
        if self.cval is not None:
            kwargs["cval"] = self.cval

        try:
            flat_flux, trend = flatten(light_curve.time, light_curve.flux, **kwargs)
        except Exception as exc:  # wotan raises assorted exception types
            raise PipelineError(
                f"wotan flatten failed for target {light_curve.target_id}: {exc}"
            ) from exc

        step = (
            f"{self.name}(method={self.method},"
            f"window={self.window_length_days}d)"
        )
        usable = np.isfinite(flat_flux) & np.isfinite(trend) & (trend != 0)
        n_dropped = int(np.count_nonzero(~usable))

        result = mask_light_curve(light_curve, usable, step)
        result.flux = flat_flux[usable]
        if result.flux_err is not None:
            result.flux_err = result.flux_err / trend[usable]
        result.meta["trend"] = trend[usable]
        logger.info(
            "Target %s: detrended with %s (window %.3f d); dropped %d "
            "untrendable cadence(s).",
            light_curve.target_id,
            self.method,
            self.window_length_days,
            n_dropped,
        )
        return result

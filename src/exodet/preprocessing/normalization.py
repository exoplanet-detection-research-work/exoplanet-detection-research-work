"""Flux normalization schemes.

All schemes are single-pass vectorized transforms. The fitted
statistics are stored in ``meta["normalization"]`` so every transform
is exactly invertible and fully reproducible from the saved output.
"""

from __future__ import annotations

import logging

import numpy as np

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.preprocessing.base import PREPROCESSORS, BasePreprocessor

__all__ = ["Normalizer"]

logger = logging.getLogger(__name__)


@PREPROCESSORS.register("normalize")
class Normalizer(BasePreprocessor):
    """Normalizes flux with a configurable scheme.

    Supported methods:
        * ``"minmax"``: maps flux to ``[0, 1]``.
        * ``"median"``: divides by the median (relative flux ~1).
        * ``"zscore"``: subtracts the mean, divides by the std.

    Attributes:
        method: The selected normalization scheme.
    """

    _METHODS = ("minmax", "median", "zscore")

    def __init__(self, method: str = "median") -> None:
        """Initializes the normalizer.

        Args:
            method: One of ``"minmax"``, ``"median"``, ``"zscore"``.

        Raises:
            PipelineError: If the method is unknown.
        """
        if method not in self._METHODS:
            raise PipelineError(
                f"Unknown normalization method '{method}'. "
                f"Available: {self._METHODS}."
            )
        self.method = method

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Normalizes the flux (and scales uncertainties consistently).

        Args:
            light_curve: The input curve with finite flux.

        Returns:
            The normalized curve with the fitted statistics stored in
            ``meta["normalization"]``.

        Raises:
            PipelineError: If the flux has zero range/scale, making the
                transform degenerate.
        """
        flux = light_curve.flux
        stats: dict[str, float]

        if self.method == "minmax":
            low = float(np.min(flux))
            high = float(np.max(flux))
            scale = high - low
            if scale == 0.0:
                raise PipelineError(
                    f"Target {light_curve.target_id}: constant flux cannot be "
                    "min-max normalized."
                )
            new_flux = (flux - low) / scale
            stats = {"min": low, "max": high}
        elif self.method == "median":
            median = float(np.median(flux))
            if median == 0.0:
                raise PipelineError(
                    f"Target {light_curve.target_id}: zero median flux cannot "
                    "be median normalized."
                )
            scale = abs(median)
            new_flux = flux / median
            stats = {"median": median}
        else:
            mean = float(np.mean(flux))
            std = float(np.std(flux))
            if std == 0.0:
                raise PipelineError(
                    f"Target {light_curve.target_id}: constant flux cannot be "
                    "z-score normalized."
                )
            scale = std
            new_flux = (flux - mean) / std
            stats = {"mean": mean, "std": std}

        result = light_curve.replace_flux(
            new_flux,
            step_name=f"{self.name}(method={self.method})",
            flux_err=(
                None if light_curve.flux_err is None else light_curve.flux_err / scale
            ),
        )
        result.meta["normalization"] = {"method": self.method, "stats": stats}
        logger.info(
            "Target %s: normalized flux (%s, %s).",
            light_curve.target_id,
            self.method,
            stats,
        )
        return result

"""Abstract feature extraction from preprocessed light curves.

Feature extractors turn a :class:`~exodet.data.base.LightCurve` into a
fixed-size numeric vector (e.g. BLS statistics, transit-shape metrics,
global/local phase-folded views). Implementations register with
:data:`FEATURE_EXTRACTORS` to become addressable from YAML configs.
"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt

from exodet.data.base import LightCurve
from exodet.registry import Registry

__all__ = ["BaseFeatureExtractor", "FEATURE_EXTRACTORS"]

FEATURE_EXTRACTORS: Registry["BaseFeatureExtractor"] = Registry("feature extractor")


class BaseFeatureExtractor(abc.ABC):
    """Abstract mapping from a light curve to a feature vector."""

    @property
    @abc.abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """Names of the produced features, aligned with output order."""

    @abc.abstractmethod
    def extract(self, light_curve: LightCurve) -> npt.NDArray[np.float64]:
        """Computes the feature vector for one light curve.

        Args:
            light_curve: A preprocessed light curve.

        Returns:
            A 1-D array of length ``len(self.feature_names)``.

        Raises:
            PipelineError: If features cannot be computed.
        """

    def __call__(self, light_curve: LightCurve) -> npt.NDArray[np.float64]:
        """Alias for :meth:`extract`."""
        return self.extract(light_curve)

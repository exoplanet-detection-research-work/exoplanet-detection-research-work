"""Abstract preprocessing step and pipeline composition.

A preprocessing step is a pure transformation of one
:class:`~exodet.data.base.LightCurve` into another (detrending,
outlier clipping, phase folding, binning, ...). Steps are composed
into a :class:`PreprocessingPipeline` in the order given by the YAML
``preprocessing.steps`` list.

Future preprocessing implementations subclass :class:`BasePreprocessor`
and register with :data:`PREPROCESSORS`; no other code changes are
needed for them to be usable from configs.
"""

from __future__ import annotations

import abc
import logging
from typing import Sequence

from exodet.config.schema import PreprocessingConfig
from exodet.data.base import LightCurve
from exodet.registry import Registry

__all__ = ["BasePreprocessor", "PreprocessingPipeline", "PREPROCESSORS"]

logger = logging.getLogger(__name__)

PREPROCESSORS: Registry["BasePreprocessor"] = Registry("preprocessor")


class BasePreprocessor(abc.ABC):
    """Abstract transformation applied to a single light curve."""

    @property
    def name(self) -> str:
        """Human-readable step name recorded in light-curve history."""
        return type(self).__name__

    @abc.abstractmethod
    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Transforms a light curve.

        Implementations must not mutate the input; use
        :meth:`LightCurve.replace_flux` to produce the output so that
        provenance history is maintained automatically.

        Args:
            light_curve: The input light curve.

        Returns:
            The transformed light curve.

        Raises:
            PipelineError: If the transformation cannot be applied.
        """

    def __call__(self, light_curve: LightCurve) -> LightCurve:
        """Alias for :meth:`apply` enabling function-style usage."""
        return self.apply(light_curve)


class PreprocessingPipeline:
    """An ordered, immutable chain of preprocessing steps.

    Attributes:
        steps: The steps applied in order.
    """

    def __init__(self, steps: Sequence[BasePreprocessor]) -> None:
        """Initializes the pipeline.

        Args:
            steps: Steps to apply, in order.
        """
        self.steps: tuple[BasePreprocessor, ...] = tuple(steps)

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> "PreprocessingPipeline":
        """Builds a pipeline from its YAML configuration.

        Args:
            config: The ``preprocessing`` section of an experiment config.

        Returns:
            A pipeline with all configured steps instantiated from the
            :data:`PREPROCESSORS` registry.

        Raises:
            RegistryError: If a step name is not registered.
        """
        steps = [
            PREPROCESSORS.build(step.name, **step.params) for step in config.steps
        ]
        return cls(steps)

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Applies every step in order.

        Args:
            light_curve: The input light curve.

        Returns:
            The fully preprocessed light curve.
        """
        result = light_curve
        for step in self.steps:
            logger.debug(
                "Applying %s to target %s", step.name, light_curve.target_id
            )
            result = step.apply(result)
        return result

    def __call__(self, light_curve: LightCurve) -> LightCurve:
        """Alias for :meth:`apply`."""
        return self.apply(light_curve)

    def __len__(self) -> int:
        return len(self.steps)

"""Abstract classifier interface.

Framework-agnostic contract for every detection model (classical ML on
extracted features, 1-D CNNs on folded views, transformers, ...).
Implementations register with :data:`MODELS` so YAML configs can select
them by name. Nothing in this module depends on a specific ML
framework; concrete subclasses own their backend.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np
import numpy.typing as npt

from exodet.registry import Registry

__all__ = ["BaseModel", "MODELS"]

MODELS: Registry["BaseModel"] = Registry("model")


class BaseModel(abc.ABC):
    """Abstract binary classifier for exoplanet detection.

    The interface deliberately mirrors the scikit-learn estimator
    contract (``fit`` / ``predict_proba``) so classical and deep
    models are interchangeable throughout the pipeline.
    """

    @abc.abstractmethod
    def fit(
        self,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int_],
        validation: tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]] | None = None,
    ) -> "BaseModel":
        """Trains the model.

        Args:
            features: Training inputs of shape ``(n_samples, ...)``.
            labels: Integer class labels of shape ``(n_samples,)``.
            validation: Optional held-out ``(features, labels)`` pair
                used for early stopping and monitoring.

        Returns:
            ``self``, to allow call chaining.
        """

    @abc.abstractmethod
    def predict_proba(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Predicts positive-class probabilities.

        Args:
            features: Inputs of shape ``(n_samples, ...)``.

        Returns:
            Probabilities of the planet class, shape ``(n_samples,)``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """

    def predict(
        self, features: npt.NDArray[np.float64], threshold: float = 0.5
    ) -> npt.NDArray[np.int_]:
        """Predicts hard class labels by thresholding probabilities.

        Args:
            features: Inputs of shape ``(n_samples, ...)``.
            threshold: Positive-class decision threshold.

        Returns:
            Integer class labels of shape ``(n_samples,)``.
        """
        return (self.predict_proba(features) >= threshold).astype(np.int_)

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persists model weights/state to disk.

        Args:
            path: Destination file or directory.
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Restores a model previously saved with :meth:`save`.

        Args:
            path: Source file or directory.

        Returns:
            The restored model, ready for inference.
        """

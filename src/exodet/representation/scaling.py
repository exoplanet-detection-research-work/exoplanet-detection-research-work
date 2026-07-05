"""Feature normalization (Module 6).

Provides standard (mean/std), robust (median/IQR), and min-max scalers
with optional per-feature ``log10(1 + x)`` pre-transforms, exact
inverse transformation, and JSON persistence of the fitted statistics.
Non-finite values pass through untouched (imputation is a modelling
decision, not a scaling one) and never contaminate the statistics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from exodet.exceptions import DataError, NotFittedError, PipelineError
from exodet.registry import Registry
from exodet.utils.io import ensure_dir

__all__ = ["FEATURE_SCALERS", "FeatureScaler"]

logger = logging.getLogger(__name__)

FEATURE_SCALERS: Registry["FeatureScaler"] = Registry("feature scaler")

_METHODS = ("standard", "robust", "minmax")


class FeatureScaler:
    """Per-feature scaler with log transforms and exact inversion.

    Transformation order per feature: optional ``log10(1 + x)``
    (only valid for features with ``x > -1``), then the affine
    scaling ``(x - center) / scale``.

    Attributes:
        method: ``"standard"`` (mean/std), ``"robust"`` (median/IQR),
            or ``"minmax"`` (min/range → [0, 1]).
        log_features: Names of features receiving the log transform.
    """

    def __init__(
        self,
        method: str = "standard",
        log_features: Sequence[str] = (),
    ) -> None:
        """Initializes the scaler.

        Args:
            method: Scaling method.
            log_features: Feature names to log-transform before scaling.

        Raises:
            PipelineError: If the method is unknown.
        """
        if method not in _METHODS:
            raise PipelineError(
                f"Unknown scaling method '{method}'. Available: {_METHODS}."
            )
        self.method = method
        self.log_features = tuple(log_features)
        self._names: tuple[str, ...] | None = None
        self._center: npt.NDArray[np.float64] | None = None
        self._scale: npt.NDArray[np.float64] | None = None
        self._log_mask: npt.NDArray[np.bool_] | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._names is not None

    def _apply_log(
        self, matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        assert self._log_mask is not None
        if not self._log_mask.any():
            return matrix
        result = matrix.copy()
        columns = result[:, self._log_mask]
        invalid = columns <= -1.0
        if invalid.any():
            raise DataError(
                "log transform requires feature values > -1; "
                f"{int(invalid.sum())} value(s) violate this."
            )
        result[:, self._log_mask] = np.log10(1.0 + columns)
        return result

    def _invert_log(
        self, matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        assert self._log_mask is not None
        if not self._log_mask.any():
            return matrix
        result = matrix.copy()
        result[:, self._log_mask] = 10.0 ** result[:, self._log_mask] - 1.0
        return result

    def fit(
        self, matrix: npt.NDArray[np.float64], names: Sequence[str]
    ) -> "FeatureScaler":
        """Fits the per-feature statistics.

        Args:
            matrix: Feature matrix of shape ``(n_samples, n_features)``.
            names: Feature names aligned with the matrix columns.

        Returns:
            ``self`` for chaining.

        Raises:
            DataError: If shapes are inconsistent or a log feature is
                unknown.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise DataError(
                f"Feature matrix shape {matrix.shape} does not match "
                f"{len(names)} feature names."
            )
        unknown = set(self.log_features) - set(names)
        if unknown:
            raise DataError(
                f"log_features refer to unknown features: {sorted(unknown)}."
            )
        self._names = tuple(names)
        self._log_mask = np.array(
            [name in self.log_features for name in names], dtype=bool
        )
        transformed = self._apply_log(matrix)

        with np.errstate(invalid="ignore"):
            if self.method == "standard":
                center = np.nanmean(transformed, axis=0)
                scale = np.nanstd(transformed, axis=0)
            elif self.method == "robust":
                center = np.nanmedian(transformed, axis=0)
                q75 = np.nanpercentile(transformed, 75, axis=0)
                q25 = np.nanpercentile(transformed, 25, axis=0)
                scale = q75 - q25
            else:  # minmax
                center = np.nanmin(transformed, axis=0)
                scale = np.nanmax(transformed, axis=0) - center

        # Constant or all-NaN features scale by 1 (values pass through
        # centered); NaN centers become 0 so they stay untouched.
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        center = np.where(np.isfinite(center), center, 0.0)
        self._center = center
        self._scale = scale
        logger.info(
            "Fitted %s scaler on %d samples x %d features "
            "(%d log-transformed).",
            self.method,
            matrix.shape[0],
            matrix.shape[1],
            int(self._log_mask.sum()),
        )
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(
                "FeatureScaler must be fitted (or loaded) before use."
            )

    def _check_names(self, names: Sequence[str]) -> None:
        assert self._names is not None
        if tuple(names) != self._names:
            raise DataError(
                "Feature names differ from the fitted ones; refusing to "
                "scale misaligned features."
            )

    def transform(
        self, matrix: npt.NDArray[np.float64], names: Sequence[str]
    ) -> npt.NDArray[np.float64]:
        """Applies the fitted transformation.

        Args:
            matrix: Feature matrix ``(n_samples, n_features)``.
            names: Feature names (must match the fitted ones exactly).

        Returns:
            The scaled matrix (a new array).
        """
        self._require_fitted()
        self._check_names(names)
        matrix = np.asarray(matrix, dtype=np.float64)
        squeeze = matrix.ndim == 1
        if squeeze:
            matrix = matrix[None, :]
        transformed = (self._apply_log(matrix) - self._center) / self._scale
        return transformed[0] if squeeze else transformed

    def inverse_transform(
        self, matrix: npt.NDArray[np.float64], names: Sequence[str]
    ) -> npt.NDArray[np.float64]:
        """Exactly inverts :meth:`transform`.

        Args:
            matrix: Scaled feature matrix.
            names: Feature names (must match the fitted ones exactly).

        Returns:
            The matrix in original feature units.
        """
        self._require_fitted()
        self._check_names(names)
        matrix = np.asarray(matrix, dtype=np.float64)
        squeeze = matrix.ndim == 1
        if squeeze:
            matrix = matrix[None, :]
        restored = self._invert_log(matrix * self._scale + self._center)
        return restored[0] if squeeze else restored

    def to_dict(self) -> dict[str, Any]:
        """Serializes the fitted statistics.

        Returns:
            A JSON-safe dictionary of method, names, and statistics.
        """
        self._require_fitted()
        assert self._center is not None and self._scale is not None
        return {
            "method": self.method,
            "feature_names": list(self._names or ()),
            "log_features": list(self.log_features),
            "center": self._center.tolist(),
            "scale": self._scale.tolist(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeatureScaler":
        """Restores a fitted scaler from :meth:`to_dict` output.

        Args:
            raw: The serialized statistics.

        Returns:
            The fitted scaler.
        """
        scaler = cls(method=raw["method"], log_features=raw["log_features"])
        scaler._names = tuple(raw["feature_names"])
        scaler._center = np.asarray(raw["center"], dtype=np.float64)
        scaler._scale = np.asarray(raw["scale"], dtype=np.float64)
        scaler._log_mask = np.array(
            [name in scaler.log_features for name in scaler._names], dtype=bool
        )
        return scaler

    def save(self, path: Path | str) -> Path:
        """Writes the fitted statistics as JSON.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "FeatureScaler":
        """Reads statistics written by :meth:`save`.

        Args:
            path: Source JSON file.

        Returns:
            The fitted scaler.

        Raises:
            DataError: If the file is missing or malformed.
        """
        path = Path(path)
        if not path.is_file():
            raise DataError(f"Scaler statistics not found: {path}")
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError) as exc:
            raise DataError(f"Malformed scaler file {path}: {exc}") from exc


def _register_variant(name: str) -> None:
    """Registers a method-bound scaler factory under ``name``."""

    @FEATURE_SCALERS.register(name)
    class _Variant(FeatureScaler):  # noqa: D401 - thin registration shim
        """Preset scaler bound to one scaling method."""

        def __init__(self, log_features: Sequence[str] = ()) -> None:
            super().__init__(method=name, log_features=log_features)

    _Variant.__name__ = f"{name.capitalize()}FeatureScaler"
    _Variant.__qualname__ = _Variant.__name__


for _method in _METHODS:
    _register_variant(_method)

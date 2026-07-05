"""Reproducible classical baseline models registered with MODELS."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.exceptions import NotFittedError
from exodet.models.base import MODELS, BaseModel

__all__ = [
    "RandomForestModel",
    "LogisticRegressionModel",
    "MlpModel",
    "LightGBMModel",
    "BASELINE_MODEL_NAMES",
]

logger = logging.getLogger(__name__)

BASELINE_MODEL_NAMES: tuple[str, ...] = (
    "random_forest",
    "logistic_regression",
    "mlp",
    "lightgbm",
    "xgboost",
)


class SklearnBaselineModel(BaseModel):
    """Shared sklearn estimator wrapper matching :class:`~exodet.ml.models.XGBoostModel`."""

    architecture_kind = "custom"
    backend = "sklearn"
    _registry_name = "sklearn_baseline"

    def __init__(self, **params: object) -> None:
        self.params = dict(params)
        self._estimator: Any = None

    def _build_estimator(self) -> Any:
        raise NotImplementedError

    def fit(
        self,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int_],
        validation: tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]] | None = None,
    ) -> "SklearnBaselineModel":
        del validation
        self._estimator = self._build_estimator()
        self._estimator.fit(features, labels)
        return self

    def predict_proba(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._estimator is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted.")
        proba = self._estimator.predict_proba(features)
        if proba.ndim == 2:
            return proba[:, 1].astype(np.float64)
        return proba.astype(np.float64)

    def save(self, path: Path) -> None:
        import joblib

        if self._estimator is None:
            raise NotFittedError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._estimator, path.with_suffix(".joblib"))
        meta = {
            "class": type(self).__name__,
            "registry": self._registry_name,
            "params": self.params,
        }
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SklearnBaselineModel":
        import joblib

        path = Path(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        model = cls(**meta.get("params", {}))
        model._estimator = joblib.load(path.with_suffix(".joblib"))
        return model


@MODELS.register("random_forest")
class RandomForestModel(SklearnBaselineModel):
    """Balanced random forest on flattened representation features."""

    _registry_name = "random_forest"

    def _build_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(self.params.get("n_estimators", 200)),
            max_depth=self.params.get("max_depth", 8),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 2)),
            class_weight="balanced_subsample",
            random_state=int(self.params.get("random_state", 0)),
            n_jobs=-1,
        )


@MODELS.register("logistic_regression")
class LogisticRegressionModel(SklearnBaselineModel):
    """L2-regularized logistic regression baseline."""

    _registry_name = "logistic_regression"

    def _build_estimator(self) -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=float(self.params.get("C", 1.0)),
            max_iter=int(self.params.get("max_iter", 500)),
            class_weight="balanced",
            random_state=int(self.params.get("random_state", 0)),
        )


@MODELS.register("mlp")
class MlpModel(SklearnBaselineModel):
    """Multi-layer perceptron baseline."""

    _registry_name = "mlp"

    def _build_estimator(self) -> Any:
        from sklearn.neural_network import MLPClassifier

        hidden = self.params.get("hidden_layer_sizes", (128, 64))
        if isinstance(hidden, list):
            hidden = tuple(hidden)
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=float(self.params.get("alpha", 1e-4)),
            learning_rate_init=float(self.params.get("learning_rate_init", 1e-3)),
            max_iter=int(self.params.get("max_iter", 200)),
            random_state=int(self.params.get("random_state", 0)),
        )


@MODELS.register("lightgbm")
class LightGBMModel(SklearnBaselineModel):
    """LightGBM gradient boosting (optional dependency)."""

    _registry_name = "lightgbm"

    def _build_estimator(self) -> Any:
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            logger.warning("lightgbm not installed; using GradientBoostingClassifier.")
            return GradientBoostingClassifier(
                n_estimators=int(self.params.get("n_estimators", 200)),
                max_depth=int(self.params.get("max_depth", 6)),
                learning_rate=float(self.params.get("learning_rate", 0.05)),
            )
        return LGBMClassifier(
            n_estimators=int(self.params.get("n_estimators", 200)),
            max_depth=int(self.params.get("max_depth", 6)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
            objective="binary",
            random_state=int(self.params.get("random_state", 0)),
        )


# xgboost is registered in exodet.ml.models; other baselines register here.

"""Model registry and backend wrappers (Module 1).

Concrete CNN, Transformer, and fusion architectures are implemented in
the next phase. This module provides:

* :class:`BaseTorchModel` — abstract PyTorch wrapper implementing
  :class:`~exodet.models.base.BaseModel`.
* :class:`XGBoostModel` — gradient-boosted tree wrapper.
* :data:`MODEL_BACKENDS` — training-backend registry (``torch``,
  ``sklearn``).
* Architecture-kind tags for future model families (``cnn``, ``transformer``,
  ``fusion``, ``xgboost``, ``custom``).
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from exodet.exceptions import NotFittedError, PipelineError
from exodet.ml.data import MlBatch
from exodet.models.base import MODELS, BaseModel
from exodet.registry import Registry

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "MODEL_BACKENDS",
    "ARCHITECTURE_KINDS",
    "BaseTorchModel",
    "XGBoostModel",
    "build_model_inputs",
]

logger = logging.getLogger(__name__)

MODEL_BACKENDS: Registry[str] = Registry("model backend")
ARCHITECTURE_KINDS: Registry[str] = Registry("architecture kind")


@MODEL_BACKENDS.register("torch")
class _TorchBackend:
    """PyTorch training backend marker."""

    name = "torch"


@MODEL_BACKENDS.register("sklearn")
class _SklearnBackend:
    """Scikit-learn / XGBoost backend marker."""

    name = "sklearn"


@ARCHITECTURE_KINDS.register("cnn")
class _CnnKind:
    """1-D CNN architecture family (implemented next phase)."""

    family = "cnn"


@ARCHITECTURE_KINDS.register("transformer")
class _TransformerKind:
    """Transformer architecture family (implemented next phase)."""

    family = "transformer"


@ARCHITECTURE_KINDS.register("fusion")
class _FusionKind:
    """Multi-modal fusion architecture family (implemented next phase)."""

    family = "fusion"


@ARCHITECTURE_KINDS.register("xgboost")
class _XgboostKind:
    """Gradient-boosted tree family."""

    family = "xgboost"


@ARCHITECTURE_KINDS.register("custom")
class _CustomKind:
    """User-defined research architectures."""

    family = "custom"


def build_model_inputs(
    batch: MlBatch,
    use_views: str,
) -> npt.NDArray[np.float64] | "torch.Tensor":
    """Stacks batch tensors into model input features.

    For tree models and sklearn backends, returns a 2-D NumPy array.
    For torch models, returns a tensor on the same device as inputs.

    Args:
        batch: A collated mini-batch.
        use_views: Input channel selection.

    Returns:
        Feature matrix of shape ``(batch, n_features)``.
    """
    parts: list[np.ndarray] = []
    if use_views in ("global", "both") and batch.global_view is not None:
        parts.append(batch.global_view.detach().cpu().numpy())
    if use_views in ("local", "both") and batch.local_view is not None:
        parts.append(batch.local_view.detach().cpu().numpy())
    if batch.features is not None:
        parts.append(batch.features.detach().cpu().numpy())
    if not parts:
        raise PipelineError(f"No inputs available for use_views='{use_views}'.")
    return np.concatenate(parts, axis=1)


class BaseTorchModel(BaseModel):
    """Abstract PyTorch classifier wrapper.

    Subclasses implement :meth:`build_network` and :meth:`forward_batch`.
    The base class owns device placement, training mode toggling, and
    checkpoint I/O compatible with :class:`~exodet.ml.checkpoints.CheckpointManager`.
    """

    architecture_kind: ClassVar[str] = "custom"
    backend: ClassVar[str] = "torch"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self._module: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._fitted = False
        self._input_dim: int | None = None

    @abstractmethod
    def build_network(self, input_dim: int) -> "torch.nn.Module":
        """Constructs the underlying ``nn.Module``.

        Args:
            input_dim: Flattened input feature dimension.

        Returns:
            The network module (not yet on device).
        """

    @abstractmethod
    def forward_batch(self, batch: MlBatch) -> "torch.Tensor":
        """Runs a forward pass and returns logits of shape ``(batch,)``.

        Args:
            batch: Input mini-batch already on the model device.

        Returns:
            Unnormalized logits for the positive class.
        """

    def _ensure_module(self, input_dim: int, device: "torch.device") -> None:
        import torch

        if self._module is None:
            self._input_dim = input_dim
            self._module = self.build_network(input_dim)
        self._device = device
        self._module.to(device)

    @property
    def module(self) -> "torch.nn.Module":
        """The underlying PyTorch module."""
        if self._module is None:
            raise NotFittedError("Model network has not been built yet.")
        return self._module

    def state_dict(self) -> dict[str, Any]:
        """Returns the module state dict for checkpointing."""
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Loads weights from a checkpoint state dict."""
        self.module.load_state_dict(state)
        self._fitted = True

    def fit(
        self,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int_],
        validation: tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]] | None = None,
    ) -> "BaseTorchModel":
        """Sklearn-style fit is not used; training goes through the Trainer."""
        raise PipelineError(
            f"{type(self).__name__} must be trained via SupervisedTrainer, not fit()."
        )

    def predict_proba(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Predicts positive-class probabilities from a feature matrix.

        Args:
            features: Shape ``(n_samples, n_features)``.

        Returns:
            Probabilities of shape ``(n_samples,)``.
        """
        import torch

        if self._module is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted.")
        self._module.eval()
        device = self._device or torch.device("cpu")
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        with torch.no_grad():
            logits = self._forward_features(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float64)

    def _forward_features(self, features: "torch.Tensor") -> "torch.Tensor":
        """Default forward for flat feature vectors."""
        return self.module(features).squeeze(-1)

    def save(self, path: Path) -> None:
        """Persists model weights and metadata to disk.

        Args:
            path: Destination ``.pt`` file.
        """
        import torch

        payload = {
            "class": type(self).__qualname__,
            "architecture_kind": self.architecture_kind,
            "params": self.params,
            "input_dim": self._input_dim,
            "state_dict": self.state_dict() if self._module else {},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        logger.info("Saved torch model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BaseTorchModel":
        """Restores a model saved with :meth:`save`.

        Args:
            path: Source ``.pt`` file.

        Returns:
            The restored model.
        """
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**payload.get("params", {}))
        input_dim = payload.get("input_dim")
        if input_dim is not None:
            model._input_dim = int(input_dim)
            model._module = model.build_network(int(input_dim))
            if payload.get("state_dict"):
                model.load_state_dict(payload["state_dict"])
        model._fitted = True
        return model


@MODELS.register("xgboost")
class XGBoostModel(BaseModel):
    """XGBoost binary classifier on flattened representation features.

    Uses scikit-learn API when xgboost is unavailable (for CI without the
    optional dependency).
    """

    architecture_kind = "xgboost"
    backend = "sklearn"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **_: object,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self._estimator: Any = None

    def fit(
        self,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int_],
        validation: tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]] | None = None,
    ) -> "XGBoostModel":
        """Trains the booster on feature vectors.

        Args:
            features: Shape ``(n_samples, n_features)``.
            labels: Binary labels.
            validation: Optional validation pair for early stopping.

        Returns:
            ``self``.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            logger.warning("xgboost not installed; using GradientBoostingClassifier.")
            self._estimator = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
            )
            self._estimator.fit(features, labels)
            return self

        eval_set = None
        if validation is not None:
            eval_set = [(validation[0], validation[1])]
        self._estimator = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self._estimator.fit(features, labels, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Predicts positive-class probabilities.

        Args:
            features: Shape ``(n_samples, n_features)``.

        Returns:
            Probabilities of shape ``(n_samples,)``.
        """
        if self._estimator is None:
            raise NotFittedError("XGBoostModel is not fitted.")
        proba = self._estimator.predict_proba(features)
        return proba[:, 1].astype(np.float64)

    def save(self, path: Path) -> None:
        """Persists the estimator to JSON metadata + joblib sidecar.

        Args:
            path: Destination ``.json`` file (``.joblib`` written alongside).
        """
        import joblib

        if self._estimator is None:
            raise NotFittedError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_path = path.with_suffix(".joblib")
        joblib.dump(self._estimator, model_path)
        meta = {
            "class": "XGBoostModel",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "model_file": model_path.name,
        }
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Saved XGBoost model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        """Restores a model saved with :meth:`save`.

        Args:
            path: Source ``.json`` metadata file.

        Returns:
            The restored model.
        """
        import joblib

        path = Path(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        model = cls(
            n_estimators=meta["n_estimators"],
            max_depth=meta["max_depth"],
            learning_rate=meta["learning_rate"],
            subsample=meta["subsample"],
            colsample_bytree=meta["colsample_bytree"],
        )
        model._estimator = joblib.load(path.parent / meta["model_file"])
        return model

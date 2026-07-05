"""Registry integration for exoplanet neural network architectures (Module 8)."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from exodet.ml.data import MlBatch
from exodet.ml.models import BaseTorchModel
from exodet.models.base import MODELS
from exodet.models.classifier import ForwardOutput, HybridExoplanetNetwork
from exodet.models.config import ModelArchitectureConfig, parse_model_params

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "ExoplanetClassifierModel",
    "register_architecture",
]

logger = logging.getLogger(__name__)

_BRANCH_ALIASES: dict[str, str] = {
    "cnn": "cnn_only",
    "transformer": "transformer_only",
}


class ExoplanetClassifierModel(BaseTorchModel):
    """BaseTorchModel wrapper around :class:`HybridExoplanetNetwork`.

    Integrates with the existing trainer via :meth:`forward_batch` while
    exposing full multi-class inference on the underlying module.
    """

    architecture_kind: ClassVar[str] = "fusion"
    _registry_branch: ClassVar[str] = "fusion"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        branch = _BRANCH_ALIASES.get(self._registry_branch, self._registry_branch)
        self._arch_config = parse_model_params(self.params, branch_mode=branch)

    def build_network(self, input_dim: int) -> HybridExoplanetNetwork:
        config = parse_model_params(
            self.params,
            input_dim=input_dim,
            branch_mode=_BRANCH_ALIASES.get(self._registry_branch, self._registry_branch),
        )
        self._arch_config = config
        network = HybridExoplanetNetwork(config)
        logger.info(
            "Built %s network (branches: cnn=%s transformer=%s physics=%s, classes=%d).",
            config.branch_mode,
            config.use_cnn,
            config.use_transformer,
            config.use_physics,
            config.num_classes,
        )
        return network

    def _run_forward(self, batch: MlBatch) -> ForwardOutput:
        network = self.module
        network.clear_cache()
        return network(
            global_view=batch.global_view,
            local_view=batch.local_view,
            physics=batch.features,
        )

    def forward_batch(self, batch: MlBatch) -> "torch.Tensor":
        output = self._run_forward(batch)
        return self.module.trainer_logits(output)

    def predict_proba(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Returns positive-class probability for binary evaluation metrics."""
        full = self.predict_proba_multiclass(features)
        if full.shape[1] == 1:
            return full[:, 0]
        return full[:, 0]

    def predict_proba_multiclass(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Returns full softmax probabilities from a flattened feature matrix."""
        import torch

        if self._module is None:
            from exodet.exceptions import NotFittedError

            raise NotFittedError(f"{type(self).__name__} is not fitted.")
        self._module.eval()
        device = self._device or torch.device("cpu")
        global_view, local_view, physics = self._unflatten_features(
            torch.from_numpy(features.astype(np.float32)).to(device)
        )
        with torch.no_grad():
            probs = self.module.predict_proba(
                global_view=global_view,
                local_view=local_view,
                physics=physics,
            )
        return probs.cpu().numpy().astype(np.float64)

    def predict_classes(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
        """Predicts integer class labels."""
        probs = self.predict_proba_multiclass(features)
        return probs.argmax(axis=1).astype(np.int_)

    def _unflatten_features(
        self, flat: "torch.Tensor"
    ) -> tuple["torch.Tensor | None", "torch.Tensor | None", "torch.Tensor | None"]:
        cfg = self._arch_config
        offset = 0
        global_view = local_view = physics = None
        if cfg.use_transformer:
            global_view = flat[:, offset : offset + cfg.global_bins]
            offset += cfg.global_bins
        if cfg.use_cnn:
            local_view = flat[:, offset : offset + cfg.local_bins]
            offset += cfg.local_bins
        if cfg.use_physics and cfg.n_physics_features > 0:
            physics = flat[:, offset : offset + cfg.n_physics_features]
        return global_view, local_view, physics

    def save(self, path: Path) -> None:
        import torch

        payload = {
            "class": type(self).__qualname__,
            "registry_name": self._registry_branch,
            "architecture_kind": self.architecture_kind,
            "params": self.params,
            "input_dim": self._input_dim,
            "arch_config": asdict(self._arch_config),
            "state_dict": self.state_dict() if self._module else {},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        logger.info("Saved exoplanet model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ExoplanetClassifierModel":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        registry_name = payload.get("registry_name", "fusion")
        model_cls = _MODEL_CLASSES.get(registry_name, cls)
        model = model_cls(**payload.get("params", {}))
        if payload.get("arch_config"):
            model._arch_config = ModelArchitectureConfig.from_mapping(
                payload["arch_config"]
            )
        input_dim = payload.get("input_dim")
        if input_dim is not None:
            model._input_dim = int(input_dim)
            model._module = model.build_network(int(input_dim))
            if payload.get("state_dict"):
                model.load_state_dict(payload["state_dict"])
        model._fitted = True
        return model


def _make_model_class(registry_name: str, kind: str) -> type[ExoplanetClassifierModel]:
    class _Model(ExoplanetClassifierModel):
        architecture_kind = kind
        _registry_branch = registry_name

    _Model.__name__ = f"ExoplanetModel_{registry_name}"
    _Model.__qualname__ = _Model.__name__
    return _Model


_MODEL_CLASSES: dict[str, type[ExoplanetClassifierModel]] = {}


def register_architecture(name: str, kind: str | None = None) -> type[ExoplanetClassifierModel]:
    """Registers an architecture variant with :data:`~exodet.models.base.MODELS`.

    Args:
        name: YAML registry name.
        kind: Architecture kind tag (defaults to ``name``).

    Returns:
        The registered model class.
    """
    kind = kind or name
    model_cls = _make_model_class(name, kind)
    _MODEL_CLASSES[name] = model_cls
    MODELS.register(name)(model_cls)
    return model_cls


# Required registry entries
FusionModel = register_architecture("fusion", "fusion")
CNNTransformerModel = register_architecture("cnn_transformer", "fusion")
CNNOnlyModel = register_architecture("cnn_only", "cnn")
CNNModel = register_architecture("cnn", "cnn")
TransformerOnlyModel = register_architecture("transformer_only", "transformer")
TransformerModel = register_architecture("transformer", "transformer")
PhysicsOnlyModel = register_architecture("physics_only", "custom")

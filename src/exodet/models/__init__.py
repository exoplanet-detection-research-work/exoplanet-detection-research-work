"""Exoplanet detection model architectures and registry."""

from __future__ import annotations

from exodet.models.base import MODELS, BaseModel
from exodet.models.config import CLASS_LABELS, ModelArchitectureConfig, parse_model_params

__all__ = [
    "MODELS",
    "BaseModel",
    "CLASS_LABELS",
    "ModelArchitectureConfig",
    "parse_model_params",
]


def __getattr__(name: str) -> object:
    """Lazily exposes registry wrapper classes to avoid import cycles."""
    _registry_exports = {
        "ExoplanetClassifierModel",
        "FusionModel",
        "CNNTransformerModel",
        "CNNOnlyModel",
        "CNNModel",
        "TransformerOnlyModel",
        "TransformerModel",
        "PhysicsOnlyModel",
    }
    if name in _registry_exports:
        import exodet.models.registry as reg

        return getattr(reg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

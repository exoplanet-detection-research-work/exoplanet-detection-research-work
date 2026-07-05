"""Scientific inference and explainability layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "InferenceStageConfig",
    "ReportStageConfig",
    "CatalogStageConfig",
    "ScientificInferencePipeline",
    "ScientificInferenceResult",
    "ScientificInferenceBatch",
    "run_inference",
    "run_model_comparison",
]

if TYPE_CHECKING:  # pragma: no cover
    from exodet.inference.comparison import ModelComparisonReport
    from exodet.inference.config import (
        CatalogStageConfig,
        InferenceStageConfig,
        ReportStageConfig,
    )
    from exodet.inference.containers import ScientificInferenceBatch, ScientificInferenceResult
    from exodet.inference.pipeline import ScientificInferencePipeline


def __getattr__(name: str) -> object:
    if name in {
        "InferenceStageConfig",
        "ReportStageConfig",
        "CatalogStageConfig",
    }:
        from exodet.inference import config as cfg

        return getattr(cfg, name)
    if name == "ScientificInferencePipeline":
        from exodet.inference.pipeline import ScientificInferencePipeline

        return ScientificInferencePipeline
    if name in {"ScientificInferenceResult", "ScientificInferenceBatch"}:
        from exodet.inference import containers as c

        return getattr(c, name)
    if name == "run_inference":
        from exodet.inference.runner import run_inference

        return run_inference
    if name == "run_model_comparison":
        from exodet.inference.runner import run_model_comparison

        return run_model_comparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

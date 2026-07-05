"""Inference engine (Module 12)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from exodet.exceptions import PipelineError
from exodet.ml.config import load_ml_settings
from exodet.ml.trainer import SupervisedTrainer
from exodet.models.base import BaseModel
from exodet.representation.containers import DatasetSample, RepresentationDataset
from exodet.utils.io import write_json

__all__ = ["InferenceResult", "InferenceEngine"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Batch or single-candidate inference output.

    Attributes:
        sample_ids: Sample identifiers.
        target_ids: Host star identifiers.
        probabilities: Positive-class probabilities.
        uncertainties: Optional per-sample uncertainty estimates.
        calibrated_probabilities: Probabilities after calibration hook.
        meta: Additional inference metadata.
    """

    sample_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    probabilities: npt.NDArray[np.float64]
    uncertainties: npt.NDArray[np.float64] | None = None
    calibrated_probabilities: npt.NDArray[np.float64] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable summary."""
        return {
            "sample_ids": list(self.sample_ids),
            "target_ids": list(self.target_ids),
            "probabilities": self.probabilities.tolist(),
            "uncertainties": (
                None if self.uncertainties is None else self.uncertainties.tolist()
            ),
            "calibrated_probabilities": (
                None
                if self.calibrated_probabilities is None
                else self.calibrated_probabilities.tolist()
            ),
            "meta": self.meta,
        }

    def save(self, path: Path | str) -> Path:
        """Writes inference results to JSON.

        Args:
            path: Destination file.

        Returns:
            Written path.
        """
        return write_json(self.to_dict(), path)


class InferenceEngine:
    """Reusable prediction pipeline with calibration and uncertainty hooks."""

    def __init__(
        self,
        model: BaseModel,
        use_views: str = "both",
        calibration_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        | None = None,
        uncertainty_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        | None = None,
        trainer: SupervisedTrainer | None = None,
    ) -> None:
        """Initializes the inference engine.

        Args:
            model: Trained classifier.
            use_views: Input channel selection.
            calibration_fn: Optional post-hoc calibration.
            uncertainty_fn: Optional uncertainty estimator.
            trainer: Trainer used for consistent feature flattening.
        """
        self.model = model
        self.use_views = use_views
        self.calibration_fn = calibration_fn
        self.uncertainty_fn = uncertainty_fn
        self.trainer = trainer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path | str,
        model: BaseModel,
        use_views: str = "both",
        trainer: SupervisedTrainer | None = None,
    ) -> "InferenceEngine":
        """Builds an engine with weights loaded from a checkpoint directory.

        Args:
            checkpoint_dir: Directory with ``best.pt`` or ``model.json``.
            model: Model instance (architecture must match checkpoint).
            use_views: Input channel selection.
            trainer: Optional trainer for flattening.

        Returns:
            Ready inference engine.
        """
        checkpoint_dir = Path(checkpoint_dir)
        best_pt = checkpoint_dir / "best.pt"
        if best_pt.is_file():
            from exodet.ml.models import BaseTorchModel

            if isinstance(model, BaseTorchModel):
                import torch

                payload = torch.load(best_pt, map_location="cpu", weights_only=False)
                input_dim = payload.get("config_snapshot", {}).get("input_dim")
                if input_dim is not None and model._module is None:
                    model._ensure_module(int(input_dim), torch.device("cpu"))
                if payload.get("model_state"):
                    model.load_state_dict(payload["model_state"])
                model._fitted = True
        elif (checkpoint_dir / "model.json").is_file():
            model = type(model).load(checkpoint_dir / "model.json")
        return cls(model=model, use_views=use_views, trainer=trainer)

    def _flatten(self, dataset: RepresentationDataset) -> npt.NDArray[np.float64]:
        arrays = dataset.to_numpy()
        mask = np.ones(len(arrays["labels"]), dtype=bool)
        if self.trainer is not None:
            return self.trainer._flatten_numpy(arrays, mask)
        parts = []
        if self.use_views in ("global", "both"):
            parts.append(arrays["global_view"])
        if self.use_views in ("local", "both"):
            parts.append(arrays["local_view"])
        if self.use_views in ("global", "local", "both", "features_only"):
            parts.append(arrays["features"])
        return np.concatenate(parts, axis=1).astype(np.float64)

    def predict_batch(self, dataset: RepresentationDataset) -> InferenceResult:
        """Runs batch inference on a representation dataset.

        Args:
            dataset: Samples to score.

        Returns:
            Inference results with optional calibration/uncertainty.
        """
        if len(dataset) == 0:
            raise PipelineError("Cannot run inference on an empty dataset.")
        arrays = dataset.to_numpy()
        features = self._flatten(dataset)
        probabilities = self.model.predict_proba(features)
        sample_ids = tuple(str(s) for s in arrays["sample_ids"])
        target_ids = tuple(str(s) for s in arrays["target_ids"])

        calibrated = (
            self.calibration_fn(probabilities) if self.calibration_fn else None
        )
        uncertainties = (
            self.uncertainty_fn(probabilities) if self.uncertainty_fn else None
        )

        return InferenceResult(
            sample_ids=sample_ids,
            target_ids=target_ids,
            probabilities=probabilities,
            uncertainties=uncertainties,
            calibrated_probabilities=calibrated,
        )

    def predict_single(self, sample: DatasetSample) -> InferenceResult:
        """Scores one transit candidate sample.

        Args:
            sample: A single dataset sample.

        Returns:
            Single-sample inference result.
        """
        result = self.predict_batch(RepresentationDataset([sample]))
        return InferenceResult(
            sample_ids=result.sample_ids,
            target_ids=result.target_ids,
            probabilities=result.probabilities,
            uncertainties=result.uncertainties,
            calibrated_probabilities=result.calibrated_probabilities,
            meta={
                "sample_id": sample.sample_id,
                "candidate_id": sample.candidate.candidate_id,
            },
        )

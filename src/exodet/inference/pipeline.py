"""Scientific inference pipeline with batching, streaming, and AMP."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from exodet.config.schema import ExperimentConfig
from exodet.exceptions import PipelineError
from exodet.inference.config import InferenceStageConfig
from exodet.inference.containers import ScientificInferenceBatch, ScientificInferenceResult
from exodet.inference.explainability import ExplainabilityEngine
from exodet.inference.false_positive import FalsePositiveAnalyzer
from exodet.inference.parameter_fit import fit_transit_parameters
from exodet.inference.physical import estimate_physical_parameters
from exodet.inference.uncertainty import build_uncertainty_estimator
from exodet.ml.amp import AmpSettings
from exodet.ml.device import select_device
from exodet.ml.inference import InferenceEngine
from exodet.ml.models import BaseTorchModel
from exodet.models.base import MODELS, BaseModel
from exodet.representation.containers import DatasetSample, RepresentationDataset
from exodet.inference.scientific import build_reproduction_metadata
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.paths import safe_filename

__all__ = ["ScientificInferencePipeline"]

logger = logging.getLogger(__name__)


class ScientificInferencePipeline:
    """Post-training scientific inference orchestrator.

    Wraps :class:`~exodet.ml.inference.InferenceEngine` without modifying its
    public API. Adds AMP batching, parameter refinement, uncertainty,
    explainability, and false-positive analysis.
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        settings: InferenceStageConfig,
        model: BaseModel | None = None,
    ) -> None:
        self.experiment = experiment
        self.settings = settings
        self.model = model
        self._engine: InferenceEngine | None = None
        self._fp_analyzer = FalsePositiveAnalyzer(settings.false_positive)
        self._explainer = ExplainabilityEngine(settings.explainability)
        self._uncertainty = build_uncertainty_estimator(settings.uncertainty)

    def _resolve_checkpoint_dir(self) -> Path:
        if self.settings.checkpoint_path:
            path = Path(self.settings.checkpoint_path)
            if path.is_file():
                return path.parent
            return path
        return Path(self.experiment.paths.checkpoint_dir) / self.experiment.experiment_name

    def _build_model(self) -> BaseModel:
        if self.model is not None:
            return self.model
        return MODELS.build(
            self.experiment.model.architecture.name,
            **self.experiment.model.architecture.params,
        )

    def _build_engine(self) -> InferenceEngine:
        if self._engine is not None:
            return self._engine
        model = self._build_model()
        checkpoint_dir = self._resolve_checkpoint_dir()
        self._engine = InferenceEngine.from_checkpoint(
            checkpoint_dir,
            model,
            use_views=self.settings.use_views,
        )
        return self._engine

    def predict_batch(
        self,
        dataset: RepresentationDataset,
        figure_dir: Path | None = None,
    ) -> ScientificInferenceBatch:
        """Runs full scientific inference on a dataset."""
        if len(dataset) == 0:
            raise PipelineError("Cannot run inference on an empty dataset.")

        engine = self._build_engine()
        base = engine.predict_batch(dataset)

        uncertainties = None
        if isinstance(engine.model, BaseTorchModel) and self.settings.uncertainty.get(
            "method", "none"
        ) != "none":
            uncertainties = self._uncertainty.estimate_batch(
                engine.model, dataset, self.settings.use_views
            )

        figure_root = figure_dir or Path(self.experiment.paths.figure_dir) / "explainability"
        ensure_dir(figure_root)

        results: list[ScientificInferenceResult] = []
        for i, sample in enumerate(dataset.samples):
            prob = float(base.probabilities[i])
            unc = uncertainties[i] if uncertainties is not None else None
            confidence = float(unc.mean if unc is not None else prob)
            classification = "planet" if prob >= self.experiment.evaluation.decision_threshold else "not_planet"

            transit = fit_transit_parameters(sample, self.settings.parameter_fit)
            physical = estimate_physical_parameters(sample, transit, self.settings.physical)
            fp = self._fp_analyzer.analyze(sample)
            explain = None
            if isinstance(engine.model, BaseTorchModel) and self.settings.explainability.get(
                "enabled", True
            ):
                explain = self._explainer.explain(
                    engine.model,
                    sample,
                    figure_root,
                    prefix=safe_filename(sample.sample_id),
                )

            results.append(
                ScientificInferenceResult(
                    sample_id=sample.sample_id,
                    target_id=sample.target_id,
                    candidate_id=sample.candidate.candidate_id,
                    probability=prob,
                    classification=classification,
                    confidence=confidence,
                    uncertainty=unc,
                    transit=transit,
                    physical=physical,
                    false_positive=fp,
                    explainability=explain,
                )
            )

        return ScientificInferenceBatch(
            results=tuple(results),
            meta=build_reproduction_metadata(
                self.experiment,
                asdict(self.settings),
                extra={
                    "n_samples": len(results),
                    "device": str(select_device(self.settings.device).device),
                },
            ),
        )

    def predict_single(
        self,
        sample: DatasetSample,
        figure_dir: Path | None = None,
    ) -> ScientificInferenceResult:
        """Scores and analyzes a single target."""
        batch = self.predict_batch(RepresentationDataset([sample]), figure_dir=figure_dir)
        return batch.results[0]

    def predict_directory(
        self,
        directory: Path | str,
        pattern: str | None = None,
        figure_dir: Path | None = None,
    ) -> ScientificInferenceBatch:
        """Loads every matching ``.npz`` dataset in a directory."""
        root = Path(directory)
        glob_pattern = pattern or self.settings.input_pattern
        paths = sorted(root.glob(glob_pattern))
        if not paths:
            raise PipelineError(f"No datasets matching '{glob_pattern}' in {root}.")
        samples: list[DatasetSample] = []
        for path in paths:
            ds = RepresentationDataset.load(path)
            samples.extend(ds.samples)
        return self.predict_batch(RepresentationDataset(samples), figure_dir=figure_dir)

    def stream(
        self,
        samples: Iterator[DatasetSample],
        figure_dir: Path | None = None,
    ) -> Iterator[ScientificInferenceResult]:
        """Yields scientific inference results one sample at a time."""
        for sample in samples:
            yield self.predict_single(sample, figure_dir=figure_dir)

    def save(self, batch: ScientificInferenceBatch, output_dir: Path | str) -> Path:
        """Writes inference batch to JSON."""
        out = Path(output_dir) / f"{self.settings.output_name}.json"
        ensure_dir(out.parent)
        write_json(batch.to_dict(), out)
        logger.info("Scientific inference results saved to %s", out)
        return out

    def predict_batch_torch(
        self,
        dataset: RepresentationDataset,
    ) -> np.ndarray:
        """AMP-accelerated batched forward pass (probabilities only)."""
        import torch
        from torch.utils.data import DataLoader

        from exodet.ml.data import MlBatch, collate_ml_batch

        engine = self._build_engine()
        if not isinstance(engine.model, BaseTorchModel):
            return engine.predict_batch(dataset).probabilities

        device_info = select_device(self.settings.device)
        device = device_info.device
        amp = AmpSettings.from_mode(self.settings.amp, device.type)

        s0 = dataset.samples[0]
        input_dim = s0.global_view.size + s0.local_view.size + s0.features.size
        engine.model._ensure_module(input_dim, device)
        engine.model.module.eval()

        items = [
            {
                "global_view": s.global_view,
                "local_view": s.local_view,
                "features": s.features,
                "labels": s.label,
                "weights": s.weight,
                "sample_id": s.sample_id,
                "target_id": s.target_id,
            }
            for s in dataset.samples
        ]
        loader = DataLoader(
            items,
            batch_size=self.settings.batch_size,
            collate_fn=lambda b: collate_ml_batch(b, use_views=self.settings.use_views),
        )
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                mb = MlBatch(
                    global_view=batch.global_view.to(device) if batch.global_view is not None else None,
                    local_view=batch.local_view.to(device) if batch.local_view is not None else None,
                    features=batch.features.to(device) if batch.features is not None else None,
                    labels=batch.labels,
                    weights=batch.weights,
                    sample_ids=batch.sample_ids,
                    target_ids=batch.target_ids,
                )
                with amp.autocast(device.type):
                    logits = engine.model.forward_batch(mb)
                p = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
                probs.append(p)
        return np.concatenate(probs)

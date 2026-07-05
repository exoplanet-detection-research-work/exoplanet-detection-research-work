"""Inference stage orchestration."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from exodet.config.schema import ExperimentConfig
from exodet.exceptions import PipelineError
from exodet.inference.benchmark import benchmark_inference
from exodet.inference.comparison import ModelComparator
from exodet.inference.config import InferenceStageConfig, load_inference_stage_config
from exodet.inference.containers import ScientificInferenceBatch
from exodet.inference.pipeline import ScientificInferencePipeline
from exodet.inference.scientific import build_reproduction_metadata
from exodet.representation.containers import RepresentationDataset
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.timing import Timer

__all__ = ["run_inference", "run_model_comparison", "_load_dataset_split"]

logger = logging.getLogger(__name__)


def _dataset_dir(config: ExperimentConfig) -> Path:
    return Path(config.paths.processed_dir) / "dataset"


def _load_dataset_split(
    config: ExperimentConfig,
    split: str = "test",
) -> RepresentationDataset:
    path = _dataset_dir(config) / f"{split}.npz"
    if not path.is_file():
        raise PipelineError(f"Dataset split not found: {path}")
    return RepresentationDataset.load(path)


def run_inference(
    config_path: Path | str,
    overrides: list[str] | None = None,
    dataset: RepresentationDataset | None = None,
) -> ScientificInferenceBatch:
    """Runs the full scientific inference pipeline."""
    import exodet.models.registry  # noqa: F401

    experiment, settings = load_inference_stage_config(config_path, overrides)
    if not settings.enabled:
        raise PipelineError("Inference stage is disabled in config.")

    data = dataset if dataset is not None else _load_dataset_split(
        experiment, settings.input_dataset
    )
    pipeline = ScientificInferencePipeline(experiment, settings)

    with Timer("scientific inference") as timer:
        if settings.streaming:
            results = list(pipeline.stream(iter(data.samples)))
            batch = ScientificInferenceBatch(results=tuple(results))
        else:
            batch = pipeline.predict_batch(data)

    report_dir = Path(experiment.paths.report_dir)
    ensure_dir(report_dir)
    out_path = pipeline.save(batch, report_dir)

    if settings.benchmark.get("enabled", False):
        bench = benchmark_inference(pipeline, data)
        write_json(bench.to_dict(), report_dir / "inference_benchmark.json")

    write_json(
        build_reproduction_metadata(
            experiment,
            asdict(settings),
            extra={
                "runtime_seconds": timer.elapsed,
                "n_samples": len(batch),
                "output": str(out_path),
            },
        ),
        report_dir / "inference_summary.json",
    )
    logger.info("Inference complete in %.2f s (%d samples).", timer.elapsed, len(batch))
    return batch


def run_model_comparison(
    config_path: Path | str,
    model_checkpoints: dict[str, str],
    overrides: list[str] | None = None,
) -> object:
    """Compares multiple model checkpoints on the test split."""
    experiment, settings = load_inference_stage_config(config_path, overrides)
    dataset = _load_dataset_split(experiment, settings.input_dataset)
    comparator = ModelComparator(experiment, settings, model_checkpoints)
    out_dir = Path(experiment.paths.report_dir) / "model_comparison"
    return comparator.compare(
        dataset,
        out_dir,
        threshold=experiment.evaluation.decision_threshold,
    )

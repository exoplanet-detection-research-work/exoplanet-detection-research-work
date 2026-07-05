"""ML stage orchestration: train, evaluate, predict."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from exodet.config.schema import ExperimentConfig
from exodet.evaluation.base import EvaluationReport
from exodet.exceptions import PipelineError
from exodet.ml.cross_validation import CrossValidationRunner
from exodet.ml.inference import InferenceEngine, InferenceResult
from exodet.ml.metrics import build_evaluation_report
from exodet.ml.trainer import build_trainer
from exodet.models.base import MODELS, BaseModel
from exodet.representation.containers import RepresentationDataset
from exodet.training.base import TrainingResult
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.timing import Timer

__all__ = ["run_training", "run_evaluation", "run_predict"]

logger = logging.getLogger(__name__)


def _dataset_dir(config: ExperimentConfig) -> Path:
    return Path(config.paths.processed_dir) / "dataset"


def _load_splits(config: ExperimentConfig) -> dict[str, RepresentationDataset]:
    """Loads train/validation/test representation splits.

    Args:
        config: Experiment configuration.

    Returns:
        Dict with keys ``train``, ``validation``, ``test`` (missing splits
        map to empty datasets).

    Raises:
        PipelineError: If the training split is missing.
    """
    root = _dataset_dir(config)
    train_path = root / "train.npz"
    if not train_path.is_file():
        raise PipelineError(
            f"Training dataset not found at {train_path}; run 'exodet dataset' first."
        )
    splits: dict[str, RepresentationDataset] = {
        "train": RepresentationDataset.load(train_path),
    }
    for name in ("validation", "test"):
        path = root / f"{name}.npz"
        splits[name] = (
            RepresentationDataset.load(path) if path.is_file() else RepresentationDataset([])
        )
    return splits


def _build_model(config: ExperimentConfig) -> BaseModel:
    """Instantiates the configured model architecture.

    Args:
        config: Experiment configuration.

    Returns:
        A fresh model instance.
    """
    return MODELS.build(
        config.model.architecture.name,
        **config.model.architecture.params,
    )


def run_training(config: ExperimentConfig) -> TrainingResult | list[TrainingResult]:
    """Runs model training (optionally with cross-validation).

    Args:
        config: Full experiment configuration.

    Returns:
        Training result, or a list of per-fold results when CV is enabled.
    """
    import exodet.ml.metrics  # noqa: F401 — register metrics
    import exodet.training.research_trainer  # noqa: F401 — research trainer
    import exodet.models.registry  # noqa: F401 — neural architectures
    import exodet.ml.models  # noqa: F401 — register xgboost

    splits = _load_splits(config)
    trainer = build_trainer(config.training)
    checkpoint_dir = Path(config.paths.checkpoint_dir) / config.experiment_name
    ensure_dir(checkpoint_dir)

    cv_runner = CrossValidationRunner(trainer.ml_settings.cross_validation)
    if cv_runner.enabled:
        logger.info("Running cross-validation (%s, %d folds).", cv_runner.strategy, cv_runner.n_splits)
        with Timer("cross-validation") as timer:
            results = cv_runner.run(
                dataset=splits["train"],
                model_factory=lambda: _build_model(config),
                trainer=trainer,
                checkpoint_root=checkpoint_dir / "cv",
            )
        write_json(
            {
                "experiment_name": config.experiment_name,
                "strategy": cv_runner.strategy,
                "n_splits": cv_runner.n_splits,
                "runtime_seconds": timer.elapsed,
                "n_folds": len(results),
            },
            checkpoint_dir / "training_report.json",
        )
        return results

    with Timer("model training") as timer:
        model = _build_model(config)
        result = trainer.train(
            model=model,
            train_data=splits["train"],
            val_data=splits["validation"] if len(splits["validation"]) > 0 else None,
            checkpoint_dir=checkpoint_dir,
            resume_from=_resolve_resume(config, checkpoint_dir),
        )

    write_json(
        {
            "experiment_name": config.experiment_name,
            "runtime_seconds": timer.elapsed,
            "history": result.history,
            "best_checkpoint": str(result.best_checkpoint) if result.best_checkpoint else None,
            "trainer": trainer.describe(),
        },
        checkpoint_dir / "training_report.json",
    )
    logger.info(
        "Training complete in %.2f s (best checkpoint: %s).",
        timer.elapsed,
        result.best_checkpoint,
    )
    return result


def _resolve_resume(config: ExperimentConfig, checkpoint_dir: Path) -> Path | None:
    resume = config.training.trainer.params.get("resume_from")
    if resume:
        path = Path(str(resume))
        return path if path.is_file() else checkpoint_dir / str(resume)
    if config.training.trainer.params.get("auto_resume", False):
        last = checkpoint_dir / "last.pt"
        return last if last.is_file() else None
    return None


def run_evaluation(config: ExperimentConfig) -> EvaluationReport:
    """Evaluates a trained model on the test split.

    Args:
        config: Full experiment configuration.

    Returns:
        Evaluation report with all configured metrics.
    """
    import exodet.ml.metrics  # noqa: F401

    splits = _load_splits(config)
    if len(splits["test"]) == 0:
        raise PipelineError("Test split is empty or missing; cannot evaluate.")

    checkpoint_dir = Path(config.paths.checkpoint_dir) / config.experiment_name
    model = _build_model(config)
    engine = InferenceEngine.from_checkpoint(checkpoint_dir, model, use_views=_use_views(config))
    trainer = build_trainer(config.training)
    engine.trainer = trainer

    result = engine.predict_batch(splits["test"])
    arrays = splits["test"].to_numpy()
    mask = arrays["labels"] >= 0
    labels = arrays["labels"][mask].astype(np.int_)
    probs = result.probabilities[mask] if len(result.probabilities) == len(mask) else result.probabilities

    report = build_evaluation_report(
        experiment_name=config.experiment_name,
        split="test",
        metric_specs=config.evaluation.metrics,
        labels=labels,
        probabilities=probs,
        threshold=config.evaluation.decision_threshold,
    )
    report_path = Path(config.paths.report_dir) / f"{config.experiment_name}_test.json"
    ensure_dir(report_path.parent)
    report.save(report_path)
    logger.info("Evaluation report saved to %s", report_path)
    return report


def run_predict(
    config: ExperimentConfig,
    dataset: RepresentationDataset | None = None,
    output_name: str = "predictions",
) -> InferenceResult:
    """Scores samples with a trained model.

    Args:
        config: Experiment configuration.
        dataset: Optional dataset; defaults to the test split.
        output_name: Output filename stem under ``paths.report_dir``.

    Returns:
        Inference results.
    """
    splits = _load_splits(config)
    data = dataset if dataset is not None else splits["test"]
    if len(data) == 0:
        raise PipelineError("No samples available for prediction.")

    checkpoint_dir = Path(config.paths.checkpoint_dir) / config.experiment_name
    model = _build_model(config)
    engine = InferenceEngine.from_checkpoint(
        checkpoint_dir,
        model,
        use_views=_use_views(config),
    )
    engine.trainer = build_trainer(config.training)
    result = engine.predict_batch(data)

    out_path = Path(config.paths.report_dir) / f"{output_name}.json"
    ensure_dir(out_path.parent)
    result.save(out_path)
    logger.info("Predictions saved to %s (%d samples).", out_path, len(result.probabilities))
    return result


def _use_views(config: ExperimentConfig) -> str:
    return str(config.training.trainer.params.get("use_views", "both"))

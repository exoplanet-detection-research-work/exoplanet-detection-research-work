"""Benchmark suite orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from exodet.benchmarking.calibration_analysis import analyze_calibration
from exodet.benchmarking.config import (
    BenchmarkStageConfig,
    HyperparameterStageConfig,
    SensitivityStageConfig,
    load_benchmark_stage_config,
)
from exodet.benchmarking.cross_mission import evaluate_cross_mission
from exodet.benchmarking.error_analysis import analyze_errors
from exodet.benchmarking.evaluation import (
    BenchmarkModelResult,
    evaluate_probabilities,
    train_sklearn_baseline,
)
from exodet.benchmarking.figures import PublicationFigures, plot_sensitivity_curves
from exodet.benchmarking.hyperparameter import run_hyperparameter_study
from exodet.benchmarking.reports import BenchmarkReport, write_benchmark_reports
from exodet.benchmarking.sensitivity import iter_sensitivity_levels
from exodet.benchmarking.statistics import compare_model_predictions
from exodet.config.schema import ComponentConfig, ExperimentConfig
from exodet.exceptions import PipelineError
from exodet.ml.trainer import build_trainer
from exodet.representation.containers import RepresentationDataset
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.process_metrics import process_rss_bytes
from exodet.utils.seeding import seed_everything

__all__ = ["run_benchmark", "run_sensitivity", "load_benchmark_splits"]

logger = logging.getLogger(__name__)


def _stage_dict(stage: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(stage)


def load_benchmark_splits(config: ExperimentConfig) -> dict[str, RepresentationDataset]:
    """Load representation splits for benchmarking."""
    from exodet.ml.runner import _load_splits

    return _load_splits(config)


def _sklearn_trainer(experiment: ExperimentConfig) -> Any:
    training = replace(
        experiment.training,
        trainer=ComponentConfig(
            name=experiment.training.trainer.name,
            params={**experiment.training.trainer.params, "backend": "sklearn"},
        ),
    )
    return build_trainer(training)


def _dataset_summary(splits: dict[str, RepresentationDataset]) -> dict[str, Any]:
    return {
        split: {
            "n_samples": len(ds),
            "n_labeled": sum(1 for s in ds.samples if s.label >= 0),
        }
        for split, ds in splits.items()
    }


def _memory_bytes() -> int | None:
    return process_rss_bytes()


def run_benchmark(
    config_path: Path | str,
    overrides: list[str] | None = None,
) -> BenchmarkReport:
    """Run the full scientific benchmarking suite."""
    import exodet.benchmarking.baselines  # noqa: F401 — register sklearn baselines
    import exodet.ml.models  # noqa: F401 — xgboost

    experiment, stage, sensitivity_cfg, hyper_cfg, _ = load_benchmark_stage_config(
        config_path, overrides
    )
    if not stage.enabled:
        raise PipelineError("benchmark.enabled is false; nothing to run.")

    seed_everything(experiment.seed)
    splits = load_benchmark_splits(experiment)
    if len(splits["train"]) == 0:
        raise PipelineError("Training split is empty.")

    output_root = Path(
        stage.output_dir or Path(experiment.paths.report_dir) / "benchmark"
    )
    figure_dir = output_root / "figures"
    ensure_dir(output_root)

    trainer = _sklearn_trainer(experiment)
    threshold = experiment.evaluation.decision_threshold
    model_results: list[BenchmarkModelResult] = []
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    calibration_reports: dict[str, Any] = {}
    error_reports: dict[str, Any] = {}
    cross_mission_report: dict[str, Any] = {}
    figures = PublicationFigures(figure_dir, n_bins=int(stage.calibration.get("n_bins", 10)))

    for model_name in stage.models:
        if model_name not in ("xgboost", "random_forest", "logistic_regression", "mlp", "lightgbm"):
            logger.warning("Skipping unknown baseline model '%s'.", model_name)
            continue
        ckpt = output_root / "checkpoints" / model_name
        ckpt.mkdir(parents=True, exist_ok=True)
        try:
            model, _, train_runtime = train_sklearn_baseline(
                model_name,
                splits["train"],
                splits["validation"] if len(splits["validation"]) else None,
                trainer,
                ckpt,
            )
        except Exception as exc:
            logger.error("Baseline %s failed: %s", model_name, exc)
            continue

        test_split = stage.splits[0] if stage.splits else "test"
        test_data = splits.get(test_split, splits["test"])
        metrics, probs, labels, preds = evaluate_probabilities(
            model,
            test_data,
            trainer,
            threshold=threshold,
            metric_names=stage.metrics,
        )
        result = BenchmarkModelResult(
            name=model_name,
            metrics=metrics,
            probabilities=probs,
            labels=labels,
            predictions=preds,
            runtime_seconds=train_runtime,
            memory_bytes=_memory_bytes(),
            checkpoint_path=str(ckpt),
        )
        model_results.append(result)
        predictions[model_name] = preds
        probabilities[model_name] = probs
        figures.roc_pr_confusion(experiment.experiment_name, test_split, labels, probs, threshold=threshold)
        figures.confusion_matrix(labels, preds, name=model_name)
        if stage.calibration.get("enabled", True):
            cal = analyze_calibration(
                model_name,
                labels,
                probs,
                figure_dir / "calibration",
                n_bins=int(stage.calibration.get("n_bins", 10)),
            )
            calibration_reports[model_name] = cal.to_dict()
        if stage.error_analysis.get("enabled", True):
            err = analyze_errors(
                test_data, labels, preds, probs, figure_dir / "errors", model_name=model_name
            )
            error_reports[model_name] = err.to_dict()
        if stage.cross_mission.get("enabled", True):
            cm = evaluate_cross_mission(test_data, labels, probs, threshold=threshold)
            cross_mission_report[model_name] = cm.to_dict()

    labels = np.array([], dtype=np.int_)
    if model_results:
        labels = model_results[0].labels

    if len(labels) > 0 and len(predictions) >= 2:
        stats = compare_model_predictions(
            labels,
            predictions,
            probabilities,
            n_bootstrap=int(stage.statistics.get("n_bootstrap", 2000)),
            seed=experiment.seed,
        )
    else:
        stats = {}

    sensitivity_payload: dict[str, Any] = {}
    if sensitivity_cfg.enabled and model_results:
        sensitivity_payload = _run_sensitivity_inline(
            experiment,
            splits,
            trainer,
            model_results[0].name,
            sensitivity_cfg,
            output_root,
            threshold,
        )

    hyper_payload: dict[str, Any] = {}
    if hyper_cfg.enabled and hyper_cfg.parameters:
        study = run_hyperparameter_study(
            experiment,
            splits["train"],
            splits["validation"] if len(splits["validation"]) else None,
            splits["test"] if len(splits["test"]) else splits["train"],
            trainer,
            hyper_cfg.parameters,
            ranking_metric=hyper_cfg.ranking_metric,
            max_trials=hyper_cfg.max_trials,
            checkpoint_root=output_root / "hyperparameter",
            threshold=threshold,
        )
        study.save(output_root / "hyperparameter_study.json")
        hyper_payload = study.to_dict()

    from exodet.reproducibility.collector import collect_reproducibility_snapshot

    repro = collect_reproducibility_snapshot(
        experiment,
        config_path=Path(config_path),
        stage_settings={"benchmark": _stage_dict(stage)},
    )
    total_runtime = sum(r.runtime_seconds for r in model_results)
    conclusions = _build_conclusions(model_results, stats)

    report = BenchmarkReport(
        experiment_name=experiment.experiment_name,
        dataset_summary=_dataset_summary(splits),
        training_configuration={
            "epochs": experiment.training.epochs,
            "batch_size": experiment.training.batch_size,
            "learning_rate": experiment.training.learning_rate,
            "backend": "sklearn",
            "models": list(stage.models),
        },
        model_results=[r.to_dict() for r in model_results],
        statistics=stats,
        calibration=calibration_reports,
        error_analysis=error_reports,
        cross_mission=cross_mission_report,
        sensitivity=sensitivity_payload,
        hyperparameter=hyper_payload,
        runtime={"total_seconds": total_runtime, "per_model": {r.name: r.runtime_seconds for r in model_results}},
        hardware={"memory_bytes": _memory_bytes(), **repro.get("hardware", {})},
        conclusions=conclusions,
        reproduction=repro,
    )
    formats = tuple(stage.reports.get("formats", ("json", "markdown", "html", "csv", "pdf")))
    paths = write_benchmark_reports(report, output_root, formats=formats)
    write_json({**report.to_dict(), "report_paths": paths}, output_root / "benchmark_manifest.json")
    logger.info("Benchmark complete; reports at %s", output_root)
    return report


def _run_sensitivity_inline(
    experiment: ExperimentConfig,
    splits: dict[str, RepresentationDataset],
    trainer: Any,
    model_name: str,
    cfg: SensitivityStageConfig,
    output_root: Path,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one baseline across perturbation levels."""
    import exodet.benchmarking.baselines  # noqa: F401
    import exodet.ml.models  # noqa: F401
    from exodet.benchmarking.evaluation import flatten_dataset
    from exodet.ml.metrics import compute_all_metrics
    from exodet.models.base import MODELS

    test_data = splits["test"] if len(splits["test"]) else splits["train"]
    features, labels, _ = flatten_dataset(test_data, trainer)
    model = MODELS.build(model_name)
    model.fit(features, labels.astype(np.int_))

    curves: dict[str, list[dict[str, Any]]] = {}
    for pert in iter_sensitivity_levels(
        features,
        labels.astype(np.int_),
        cfg.perturbations,
        cfg.levels,
        seed=experiment.seed,
    ):
        probs = model.predict_proba(pert.features)
        metrics, _ = compute_all_metrics((), labels, probs, threshold)
        curves.setdefault(pert.perturbation, []).append(
            {"level": pert.level, "metrics": metrics, "meta": pert.meta}
        )

    figure_dir = output_root / "figures" / "sensitivity"
    plot_sensitivity_curves(curves, figure_dir)
    payload = {"curves": curves, "figure_dir": str(figure_dir)}
    write_json(payload, output_root / f"{cfg.output_name}.json")
    return payload


def run_sensitivity(
    config_path: Path | str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run sensitivity analysis as a standalone stage."""
    experiment, _, sensitivity_cfg, _, _ = load_benchmark_stage_config(config_path, overrides)
    if not sensitivity_cfg.enabled:
        raise PipelineError("sensitivity.enabled is false.")
    splits = load_benchmark_splits(experiment)
    trainer = _sklearn_trainer(experiment)
    output_root = Path(experiment.paths.report_dir) / "sensitivity"
    ensure_dir(output_root)
    return _run_sensitivity_inline(
        experiment,
        splits,
        trainer,
        "logistic_regression",
        sensitivity_cfg,
        output_root,
        experiment.evaluation.decision_threshold,
    )


def _build_conclusions(
    results: list[BenchmarkModelResult],
    stats: dict[str, Any],
) -> list[str]:
    if not results:
        return ["No models completed training."]
    best = max(results, key=lambda r: r.metrics.get("roc_auc", float("-inf")))
    lines = [
        f"Best ROC-AUC: {best.name} ({best.metrics.get('roc_auc', float('nan')):.4f}).",
    ]
    pairwise = stats.get("pairwise", {})
    for key, value in pairwise.items():
        mcnemar = value.get("mcnemar", {})
        if mcnemar.get("significant_0_05"):
            lines.append(f"McNemar test significant for {key} (p={mcnemar.get('p_value'):.4g}).")
    return lines

"""Experiment orchestration runners."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from exodet.config.schema import ExperimentConfig
from exodet.exceptions import PipelineError
from exodet.experiments.artifacts import apply_cleanup_policy, organize_artifacts
from exodet.experiments.comparison import compare_experiments
from exodet.experiments.config import (
    ArtifactsStageConfig,
    ExperimentsStageConfig,
    ReproduceStageConfig,
    SweepStageConfig,
    load_experiments_stage_config,
)
from exodet.experiments.database import ExperimentDatabase
from exodet.experiments.manager import ExperimentManager
from exodet.experiments.performance import benchmark_database_scales
from exodet.experiments.sweeps import run_sweep
from exodet.experiments.tables import export_publication_tables
from exodet.experiments.validation import validate_reproducibility
from exodet.reproducibility.collector import checksum_file, collect_reproducibility_snapshot
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.seeding import seed_everything

__all__ = [
    "execute_stage",
    "run_experiment",
    "run_experiment_sweep",
    "run_leaderboard",
    "run_reproduce_experiments",
    "run_performance_benchmark",
]

logger = logging.getLogger(__name__)


def _stage_dict(stage: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(stage)


def _default_database_path(experiment: ExperimentConfig, stage: ExperimentsStageConfig) -> Path:
    if stage.database_path:
        return Path(stage.database_path)
    return Path(experiment.paths.output_dir) / "experiments" / "index.json"


def execute_stage(
    experiment: ExperimentConfig,
    stage: str,
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    config_path: Path | str | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a named pipeline stage and return metrics/artifacts."""
    import exodet.benchmarking.baselines  # noqa: F401
    import exodet.ml.models  # noqa: F401

    result: dict[str, Any] = {"metrics": {}, "artifacts": {}, "stage": stage}
    config_path = Path(config_path) if config_path else None

    if stage == "train":
        from exodet.ml.runner import run_training

        training_result = run_training(experiment)
        if isinstance(training_result, list):
            result["metrics"] = {"n_folds": float(len(training_result))}
            ckpt = training_result[0].best_checkpoint if training_result else None
        else:
            result["metrics"] = {
                k: float(v[-1])
                for k, v in training_result.history.items()
                if v and isinstance(v[-1], (int, float))
            }
            ckpt = training_result.best_checkpoint
        if ckpt:
            result["artifacts"]["checkpoint"] = str(ckpt)
            model_path = Path(ckpt) / "model.json"
            if not model_path.is_file():
                model_path = Path(ckpt) / "last.pt"
            if model_path.is_file():
                result["model_checksum"] = checksum_file(model_path)

    elif stage == "evaluate":
        from exodet.ml.runner import run_evaluation

        report = run_evaluation(experiment)
        result["metrics"] = dict(report.scores)
        result["artifacts"]["report"] = str(
            Path(experiment.paths.report_dir) / f"{experiment.experiment_name}_test.json"
        )

    elif stage == "benchmark":
        from exodet.benchmarking.runner import run_benchmark

        if config_path is None:
            raise PipelineError("benchmark stage requires config_path.")
        report = run_benchmark(config_path, overrides=overrides)
        result["metrics"] = {
            row["name"]: row["metrics"].get("roc_auc", 0.0)
            for row in report.model_results
            if "metrics" in row
        }
        result["artifacts"]["benchmark"] = str(output_dir / "benchmark_manifest.json")

    elif stage == "ablation":
        from exodet.ablation.runner import run_ablation

        if config_path is None:
            raise PipelineError("ablation stage requires config_path.")
        payload = run_ablation(config_path, overrides=overrides)
        result["metrics"] = {
            row["label"]: row.get("metrics", {}).get("roc_auc", 0.0)
            for row in payload["variants"]
            if row.get("status") == "completed"
        }
        result["artifacts"]["ablation"] = str(output_dir / "ablation_report.json")

    elif stage == "sensitivity":
        from exodet.benchmarking.runner import run_sensitivity

        if config_path is None:
            raise PipelineError("sensitivity stage requires config_path.")
        payload = run_sensitivity(config_path, overrides=overrides)
        result["artifacts"]["sensitivity"] = str(output_dir / "sensitivity_report.json")
        result["metrics"] = {"n_perturbations": float(len(payload.get("curves", {})))}

    elif stage == "reproducibility":
        from exodet.reproducibility.runner import run_reproducibility

        if config_path is None:
            raise PipelineError("reproducibility stage requires config_path.")
        payload = run_reproducibility(config_path, overrides=overrides)
        result["artifacts"] = dict(payload.get("report_paths", {}))

    else:
        raise PipelineError(f"Unknown experiment stage: {stage}")

    dataset_root = Path(experiment.paths.processed_dir) / "dataset"
    train_npz = dataset_root / "train.npz"
    if train_npz.is_file():
        result["dataset_checksum"] = checksum_file(train_npz)

    return result


def run_experiment(
    config_path: Path | str,
    overrides: list[str] | None = None,
    *,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Register and run a single experiment."""
    experiment, stage, sweep, artifacts, reproduce, _ = load_experiments_stage_config(
        config_path, overrides
    )
    if not stage.enabled:
        raise PipelineError("experiments.enabled is false.")

    seed_everything(experiment.seed)
    db = ExperimentDatabase(_default_database_path(experiment, stage))
    manager = ExperimentManager(
        experiment,
        database=db,
        stage_config=stage,
        config_path=Path(config_path),
    )
    record = manager.register(experiment_id=experiment_id)
    paths = manager.get_output_paths(record.experiment_id)

    try:
        from exodet.experiments.profiling import ProfileContext

        with ProfileContext(output_dir=paths["root"]) as profile:
            result = execute_stage(
                experiment,
                stage.stage,
                output_dir=paths["root"],
                checkpoint_dir=paths["checkpoints"],
                config_path=config_path,
                overrides=overrides,
            )
        record = manager.mark_completed(
            record.experiment_id,
            metrics=result.get("metrics", {}),
            artifacts=result.get("artifacts", {}),
            runtime_seconds=profile.elapsed_seconds,
            dataset_checksum=result.get("dataset_checksum", ""),
            model_checksum=result.get("model_checksum", ""),
        )
    except Exception as exc:
        manager.mark_failed(record.experiment_id, str(exc))
        raise

    if artifacts.enabled:
        organize_artifacts(record, config=artifacts)
    if artifacts.cleanup.get("enabled", False):
        apply_cleanup_policy(list(db.iter_records()), artifacts.cleanup)

    repro = collect_reproducibility_snapshot(
        experiment,
        config_path=Path(config_path),
        stage_settings={"experiments": _stage_dict(stage)},
    )
    payload = {
        "experiment_id": record.experiment_id,
        "record": record.to_dict(),
        "reproduction": repro,
    }
    write_json(payload, paths["root"] / "run_summary.json")
    return payload


def run_experiment_sweep(
    config_path: Path | str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run a hyperparameter sweep campaign."""
    experiment, stage, sweep, _, _, _ = load_experiments_stage_config(config_path, overrides)
    if not sweep.enabled:
        raise PipelineError("sweep.enabled is false.")

    seed_everything(experiment.seed)
    db = ExperimentDatabase(_default_database_path(experiment, stage))
    manager = ExperimentManager(
        experiment,
        database=db,
        stage_config=stage,
        config_path=Path(config_path),
    )
    sweep_id = hashlib.sha256(f"{experiment.experiment_name}:{uuid.uuid4()}".encode()).hexdigest()[:12]
    result = run_sweep(experiment, sweep, manager=manager, sweep_id=sweep_id, seed=experiment.seed)
    comparison = compare_experiments(db, ranking_metric=sweep.ranking_metric)
    tables_dir = Path(experiment.paths.report_dir) / "experiments" / "sweeps" / sweep_id / "tables"
    table_paths = export_publication_tables(comparison, tables_dir)
    payload = {"sweep_id": sweep_id, "result": result.to_dict(), "tables": table_paths}
    write_json(payload, Path(experiment.paths.report_dir) / "experiments" / "sweeps" / sweep_id / "summary.json")
    return payload


def run_leaderboard(
    config_path: Path | str,
    overrides: list[str] | None = None,
    *,
    tags: tuple[str, ...] = (),
    metrics: tuple[str, ...] = ("roc_auc", "accuracy", "f1"),
) -> dict[str, Any]:
    """Build leaderboards from the experiment index."""
    experiment, stage, _, _, _, _ = load_experiments_stage_config(config_path, overrides)
    db = ExperimentDatabase(_default_database_path(experiment, stage))
    report = compare_experiments(db, tags=tags, metrics=metrics)
    out_dir = Path(experiment.paths.report_dir) / "experiments" / "leaderboards"
    paths = export_publication_tables(report, out_dir)
    payload = {"comparison": report.to_dict(), "table_paths": paths}
    write_json(payload, out_dir / "leaderboard.json")
    return payload


def run_reproduce_experiments(
    config_path: Path | str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Validate reproducibility for selected experiments."""
    experiment, stage, _, _, reproduce, _ = load_experiments_stage_config(config_path, overrides)
    if not reproduce.enabled:
        raise PipelineError("reproduce.enabled is false.")

    seed_everything(experiment.seed)
    db = ExperimentDatabase(_default_database_path(experiment, stage))
    manager = ExperimentManager(
        experiment,
        database=db,
        stage_config=stage,
        config_path=Path(config_path),
    )
    cert = validate_reproducibility(db, manager, reproduce)
    return cert.to_dict()


def run_performance_benchmark(
    config_path: Path | str,
    overrides: list[str] | None = None,
    *,
    scales: tuple[int, ...] = (100, 500, 1000),
) -> dict[str, Any]:
    """Benchmark experiment database at scale."""
    experiment, stage, _, _, _, _ = load_experiments_stage_config(config_path, overrides)
    out_dir = Path(experiment.paths.report_dir) / "experiments" / "performance"
    ensure_dir(out_dir)
    results = benchmark_database_scales(out_dir, scales=scales)
    payload = {"benchmarks": [r.to_dict() for r in results]}
    write_json(payload, out_dir / "database_benchmark.json")
    return payload

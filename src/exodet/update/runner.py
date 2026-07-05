"""Incremental update stage runner."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from exodet.exceptions import PipelineError
from exodet.experiments.config import ExperimentsStageConfig
from exodet.experiments.database import ExperimentDatabase
from exodet.experiments.manager import ExperimentManager
from exodet.ml.runner import run_evaluation, run_training
from exodet.update.checkpoint_manager import discover_checkpoint
from exodet.update.config import load_update_stage_config
from exodet.update.resume import apply_training_resume, checkpoint_extra_state
from exodet.update.update_pipeline import (
    UpdatePipeline,
    merge_catalog_entries,
    resolve_update_inputs,
)
from exodet.utils.timing import Timer

__all__ = ["run_update"]

logger = logging.getLogger(__name__)


def run_update(
    config_path: Path | str,
    overrides: list[str] | None = None,
    *,
    tic_ids: Sequence[str] = (),
    tic_file: str | None = None,
    fits_dir: str | None = None,
    processed_dir: str | None = None,
    resume: str | None = None,
    force_reprocess: bool = False,
    fresh_start: bool = False,
) -> dict[str, Any]:
    """Run the full incremental update workflow."""
    (
        experiment,
        update_cfg,
        tce_cfg,
        rep_cfg,
        raw_config,
    ) = load_update_stage_config(config_path, overrides)

    if not update_cfg.enabled:
        raise PipelineError("Update stage is disabled in config.")

    inputs = resolve_update_inputs(
        update_cfg,
        cli_tic_ids=tic_ids,
        cli_tic_file=tic_file,
        cli_fits_dir=fits_dir,
        cli_processed_dir=processed_dir,
    )

    pipeline = UpdatePipeline(experiment, update_cfg, tce_cfg, rep_cfg)
    with Timer("incremental dataset update") as update_timer:
        update_summary = pipeline.run(inputs, force_reprocess=force_reprocess)

    training_experiment = experiment
    checkpoint_info: dict[str, Any] | None = None
    training_result: Any = None
    train_timer_elapsed = 0.0

    if update_cfg.resume_training and not fresh_start:
        selection = resume or str(update_cfg.checkpoint.get("selection", "latest"))
        explicit = update_cfg.checkpoint.get("path")
        try:
            discovery = discover_checkpoint(
                Path(experiment.paths.checkpoint_dir),
                experiment.experiment_name,
                selection=str(selection),
                explicit_path=str(explicit) if explicit else None,
            )
            checkpoint_info = discovery.to_dict()
            checkpoint_info["extra_state"] = checkpoint_extra_state(discovery.path)
            training_experiment = apply_training_resume(experiment, discovery, fresh_start=False)
        except PipelineError as exc:
            if update_cfg.checkpoint.get("required", False):
                raise
            logger.warning("No checkpoint found; starting fresh training: %s", exc)

    experiment_record: dict[str, Any] | None = None
    experiments_cfg = ExperimentsStageConfig.from_dict(raw_config.get("experiments"))
    if experiments_cfg.enabled and experiments_cfg.auto_register:
        db_path = experiments_cfg.database_path or str(
            Path(experiment.paths.output_dir) / "experiments" / "experiments.json"
        )
        database = ExperimentDatabase(Path(db_path))
        manager = ExperimentManager(
            training_experiment,
            database=database,
            stage_config=experiments_cfg,
            config_path=Path(config_path),
            raw_config=raw_config,
        )
        if update_cfg.experiment_mode == "continuation":
            existing_id = _latest_experiment_id(database, experiment.experiment_name)
            if existing_id:
                record = database.update(existing_id, status="running")
                experiment_record = record.to_dict()
            else:
                record = manager.register(tags=("incremental_update", "continuation"))
                experiment_record = record.to_dict()
        else:
            parent_id = (
                update_cfg.parent_experiment_id
                or experiments_cfg.parent_id
                or _latest_experiment_id(database, experiment.experiment_name)
            )
            record = manager.register(
                parent_id=parent_id,
                tags=("incremental_update", "child"),
            )
            experiment_record = record.to_dict()

    if update_cfg.resume_training and update_summary.get("n_success", 0) > 0:
        with Timer("resume training") as train_timer:
            training_result = run_training(training_experiment)
        train_timer_elapsed = train_timer.elapsed
    elif update_cfg.resume_training and update_summary.get("n_success", 0) == 0:
        logger.error(
            "Skipping training: no targets were successfully processed. "
            "See %s for details.",
            Path(experiment.paths.report_dir) / "update_summary.json",
        )

    eval_payload: dict[str, Any] = {}
    if (
        update_cfg.evaluation.get("enabled", True)
        and update_cfg.resume_training
        and update_summary.get("n_success", 0) > 0
        and training_result is not None
    ):
        try:
            eval_payload["evaluation"] = asdict(run_evaluation(training_experiment))
        except PipelineError as exc:
            logger.warning("Evaluation skipped: %s", exc)
            eval_payload["evaluation_error"] = str(exc)

    inference_payload: dict[str, Any] | None = None
    if update_cfg.evaluation.get("run_inference", False):
        from exodet.inference.runner import run_inference

        batch = run_inference(config_path, overrides)
        inference_payload = {"n_results": len(batch.results)}

        if update_cfg.catalog.get("incremental", True) and raw_config.get("catalog"):
            from exodet.catalog.builder import CatalogBuilder
            from exodet.inference.config import CatalogStageConfig

            catalog_cfg = CatalogStageConfig.from_dict(raw_config.get("catalog"))
            if catalog_cfg.enabled:
                builder = CatalogBuilder(catalog_cfg)
                new_entries = builder.build(batch)
                catalog_dir = Path(experiment.paths.report_dir) / "catalog"
                merge_info = merge_catalog_entries(
                    catalog_dir,
                    catalog_cfg.output_name,
                    [entry.to_dict() for entry in new_entries],
                )
                inference_payload["catalog_merge"] = merge_info

    report_payload: dict[str, Any] | None = None
    if update_cfg.report.get("enabled", False):
        from exodet.reporting.runner import run_report

        report_payload = run_report(config_path, overrides)

    benchmark_payload: dict[str, Any] | None = None
    if update_cfg.benchmark.get("enabled", False):
        from exodet.benchmarking.runner import run_benchmark

        benchmark_payload = run_benchmark(config_path, overrides)

    leaderboard_payload: dict[str, Any] | None = None
    if update_cfg.leaderboard.get("enabled", False):
        from exodet.experiments.runner import run_leaderboard

        leaderboard_payload = run_leaderboard(config_path, overrides)

    return {
        "update": update_summary,
        "update_runtime_seconds": update_timer.elapsed,
        "checkpoint": checkpoint_info,
        "experiment": experiment_record,
        "training": _serialize_training_result(training_result),
        "training_runtime_seconds": train_timer_elapsed,
        "evaluation": eval_payload,
        "inference": inference_payload,
        "report": report_payload,
        "benchmark": benchmark_payload,
        "leaderboard": leaderboard_payload,
    }


def _latest_experiment_id(database: ExperimentDatabase, name: str) -> str | None:
    records = [
        record
        for record in database.iter_records()
        if record.metadata.get("experiment_name") == name or record.name == name
    ]
    if not records:
        return None
    records.sort(key=lambda record: record.updated_at, reverse=True)
    return records[0].experiment_id


def _serialize_training_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, list):
        return {"folds": [_serialize_training_result(item) for item in result]}
    return {
        "best_checkpoint": str(result.best_checkpoint) if result.best_checkpoint else None,
        "best_metrics": dict(result.best_metrics),
        "history": list(result.history),
    }

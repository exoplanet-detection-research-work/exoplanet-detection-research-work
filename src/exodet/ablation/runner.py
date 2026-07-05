"""Architecture ablation runner."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from exodet.ablation.config import AblationStageConfig, load_ablation_stage_config
from exodet.benchmarking.evaluation import evaluate_probabilities, train_sklearn_baseline
from exodet.benchmarking.figures import plot_ablation_summary
from exodet.benchmarking.reports import BenchmarkReport, write_benchmark_reports
from exodet.benchmarking.runner import load_benchmark_splits, _sklearn_trainer
from exodet.config.schema import ComponentConfig
from exodet.exceptions import PipelineError
from exodet.reproducibility.collector import collect_reproducibility_snapshot
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.seeding import seed_everything

__all__ = ["run_ablation", "ABLATION_SUPPORTED_ARCHITECTURES"]

logger = logging.getLogger(__name__)

ABLATION_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "cnn_only",
        "cnn",
        "transformer_only",
        "transformer",
        "physics_only",
        "cnn_transformer",
        "fusion",
        "logistic_regression",
        "random_forest",
        "mlp",
        "xgboost",
    }
)


def run_ablation(
    config_path: Path | str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Train/evaluate ablation variants and emit comparison tables."""
    import exodet.benchmarking.baselines  # noqa: F401
    import exodet.ml.models  # noqa: F401

    experiment, stage = load_ablation_stage_config(config_path, overrides)
    if not stage.enabled:
        raise PipelineError("ablation.enabled is false.")

    seed_everything(experiment.seed)
    splits = load_benchmark_splits(experiment)
    output_root = Path(stage.output_dir or Path(experiment.paths.report_dir) / "ablation")
    ensure_dir(output_root)
    trainer = _sklearn_trainer(experiment)
    threshold = experiment.evaluation.decision_threshold
    table: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for variant_id, architecture, label in stage.variants:
        if stage.backend == "sklearn":
            model_name = stage.baseline_model if architecture in {"cnn_physics", "transformer_physics"} else architecture
            if model_name not in ABLATION_SUPPORTED_ARCHITECTURES:
                rows.append(
                    {
                        "id": variant_id,
                        "label": label,
                        "architecture": architecture,
                        "status": "skipped",
                        "reason": f"Unsupported sklearn architecture '{architecture}'.",
                    }
                )
                continue
            ckpt = output_root / "checkpoints" / variant_id
            start = time.perf_counter()
            model, _, _ = train_sklearn_baseline(
                model_name,
                splits["train"],
                splits["validation"] if len(splits["validation"]) else None,
                trainer,
                ckpt,
            )
            test_data = splits["test"] if len(splits["test"]) else splits["train"]
            metrics, probs, labels, preds = evaluate_probabilities(
                model, test_data, trainer, threshold=threshold
            )
            runtime = time.perf_counter() - start
            table[label] = metrics
            rows.append(
                {
                    "id": variant_id,
                    "label": label,
                    "architecture": architecture,
                    "status": "completed",
                    "metrics": metrics,
                    "runtime_seconds": runtime,
                }
            )
            continue

        if architecture not in ABLATION_SUPPORTED_ARCHITECTURES:
            rows.append(
                {
                    "id": variant_id,
                    "label": label,
                    "architecture": architecture,
                    "status": "skipped",
                    "reason": "Architecture not registered.",
                }
            )
            continue

        try:
            import exodet.models.registry  # noqa: F401
            from exodet.ml.trainer import build_trainer
            from exodet.models.base import MODELS

            fast = stage.fast_training
            training = replace(
                experiment.training,
                epochs=int(fast.get("epochs", 2)),
                batch_size=int(fast.get("batch_size", experiment.training.batch_size)),
                learning_rate=float(fast.get("learning_rate", experiment.training.learning_rate)),
                trainer=ComponentConfig(
                    name=experiment.training.trainer.name,
                    params={
                        **experiment.training.trainer.params,
                        "backend": "torch",
                        **fast.get("trainer_params", {}),
                    },
                ),
            )
            model = MODELS.build(architecture, **fast.get("model_params", {}))
            torch_trainer = build_trainer(training)
            ckpt = output_root / "checkpoints" / variant_id
            start = time.perf_counter()
            torch_trainer.train(
                model,
                splits["train"],
                splits["validation"] if len(splits["validation"]) else None,
                checkpoint_dir=ckpt,
            )
            test_data = splits["test"] if len(splits["test"]) else splits["train"]
            probs = torch_trainer.predict(model, test_data)
            arrays = test_data.to_numpy()
            mask = arrays["labels"] >= 0
            labels = arrays["labels"][mask].astype(np.int_)
            probs = probs[mask]
            from exodet.ml.metrics import compute_all_metrics

            metrics, _ = compute_all_metrics((), labels, probs, threshold)
            runtime = time.perf_counter() - start
            table[label] = metrics
            rows.append(
                {
                    "id": variant_id,
                    "label": label,
                    "architecture": architecture,
                    "status": "completed",
                    "metrics": metrics,
                    "runtime_seconds": runtime,
                }
            )
        except Exception as exc:
            logger.error("Ablation variant %s failed: %s", variant_id, exc)
            rows.append(
                {
                    "id": variant_id,
                    "label": label,
                    "architecture": architecture,
                    "status": "failed",
                    "reason": str(exc),
                }
            )

    figure_paths = plot_ablation_summary(table, output_root / "figures", metric=stage.ranking_metric)
    repro = collect_reproducibility_snapshot(experiment, config_path=Path(config_path))
    payload = {
        "experiment_name": experiment.experiment_name,
        "variants": rows,
        "comparison_table": table,
        "figure_paths": figure_paths,
        "reproduction": repro,
    }
    write_json(payload, output_root / "ablation_report.json")
    write_json(table, output_root / "ablation_table.json")

    report = BenchmarkReport(
        experiment_name=f"{experiment.experiment_name}_ablation",
        dataset_summary={"n_train": len(splits["train"]), "n_test": len(splits["test"])},
        training_configuration={"backend": stage.backend, "variants": [v[0] for v in stage.variants]},
        model_results=[
            {
                "name": row.get("label", row.get("id")),
                "metrics": row.get("metrics", {}),
                "runtime_seconds": row.get("runtime_seconds"),
                "status": row.get("status"),
            }
            for row in rows
        ],
        reproduction=repro,
        conclusions=[f"Ablation ranking metric: {stage.ranking_metric}."],
    )
    write_benchmark_reports(report, output_root, formats=("json", "markdown", "csv"))
    return payload

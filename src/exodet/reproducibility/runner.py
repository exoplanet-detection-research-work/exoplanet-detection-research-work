"""Reproducibility report runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exodet.benchmarking.config import load_benchmark_stage_config
from exodet.exceptions import PipelineError
from exodet.reproducibility.collector import collect_reproducibility_snapshot, write_reproducibility_report
from exodet.utils.io import ensure_dir
from exodet.utils.seeding import seed_everything

__all__ = ["run_reproducibility"]


def _stage_dict(stage: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(stage)


def run_reproducibility(
    config_path: Path | str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a reproducibility report for the current experiment."""
    experiment, benchmark, _, _, ablation_raw = load_benchmark_stage_config(config_path, overrides)
    seed_everything(experiment.seed)
    output_dir = Path(experiment.paths.report_dir) / "reproducibility"
    ensure_dir(output_dir)

    dataset_paths: dict[str, Path] = {}
    try:
        from exodet.ml.runner import _load_splits

        splits = _load_splits(experiment)
        root = Path(experiment.paths.processed_dir) / "dataset"
        for name in ("train", "validation", "test"):
            path = root / f"{name}.npz"
            if path.is_file():
                dataset_paths[name] = path
        del splits
    except PipelineError:
        pass

    checkpoint_root = Path(experiment.paths.checkpoint_dir) / experiment.experiment_name
    model_paths: dict[str, Path] = {}
    for candidate in ("model.json", "last.pt", "best.pt"):
        path = checkpoint_root / candidate
        if path.is_file():
            model_paths[candidate] = path

    snapshot = collect_reproducibility_snapshot(
        experiment,
        config_path=Path(config_path),
        stage_settings={
            "benchmark": _stage_dict(benchmark),
            "ablation": ablation_raw,
        },
        dataset_paths=dataset_paths,
        model_paths=model_paths,
    )
    paths = write_reproducibility_report(snapshot, output_dir)
    return {"snapshot": snapshot, "report_paths": paths}

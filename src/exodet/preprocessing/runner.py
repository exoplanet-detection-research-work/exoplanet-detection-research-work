"""End-to-end preprocessing orchestration for the ``preprocess`` CLI stage.

Builds the configured dataset and preprocessing pipeline, processes
every light curve, persists the results as ``.npz`` files under
``paths.processed_dir``, exports diagnostic figures for the first
targets, and writes a JSON summary (per-target quality metrics and
provenance) to ``paths.report_dir``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from exodet.config.schema import ExperimentConfig
from exodet.data.base import DATASETS, LightCurve
from exodet.data.serialization import save_light_curve
from exodet.preprocessing.base import PreprocessingPipeline
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.timing import Timer
from exodet.visualization.preprocessing import generate_preprocessing_figures

__all__ = ["run_preprocessing"]

logger = logging.getLogger(__name__)


def _target_slug(curve: LightCurve) -> str:
    """Builds a filesystem-safe filename stem for a target.

    Args:
        curve: The light curve being saved.

    Returns:
        A lowercase, separator-free identifier.
    """
    return curve.target_id.replace(" ", "_").replace("/", "-").lower()


def run_preprocessing(config: ExperimentConfig, *, n_figure_targets: int = 1) -> list[Path]:
    """Runs the preprocessing stage for every target in the dataset.

    Args:
        config: The experiment configuration; ``data.dataset`` selects
            the input dataset and ``preprocessing.steps`` the pipeline.
        n_figure_targets: Number of leading targets for which
            diagnostic figures are exported.

    Returns:
        Paths of the written processed light-curve files.

    Raises:
        RegistryError: If the configured dataset or a preprocessing
            step is not registered.
        PipelineError: If a step fails on a target.
    """
    pipeline = PreprocessingPipeline.from_config(config.preprocessing)
    dataset = DATASETS.build(
        config.data.dataset.name, **config.data.dataset.params
    )
    processed_dir = ensure_dir(config.paths.processed_dir)
    logger.info(
        "Preprocessing %d target(s) with %d step(s) into %s.",
        len(dataset),
        len(pipeline),
        processed_dir,
    )

    outputs: list[Path] = []
    summary: list[dict[str, object]] = []
    with Timer(f"preprocessing of {len(dataset)} target(s)", log=logger):
        for index, raw in enumerate(dataset):
            with Timer(f"target {raw.target_id}", log=logger):
                processed = pipeline.apply(raw)
            outputs.append(
                save_light_curve(
                    processed, processed_dir / f"{_target_slug(processed)}.npz"
                )
            )
            if index < n_figure_targets:
                generate_preprocessing_figures(
                    raw, processed, config.paths.figure_dir
                )
            summary.append(
                {
                    "target_id": processed.target_id,
                    "label": processed.label,
                    "n_points_raw": len(raw),
                    "n_points_processed": len(processed),
                    "history": list(processed.history),
                    "quality_metrics": processed.meta.get("quality_metrics", {}),
                }
            )

    report_path = write_json(
        {
            "experiment_name": config.experiment_name,
            "n_targets": len(summary),
            "steps": [step.name for step in config.preprocessing.steps],
            "targets": summary,
        },
        Path(config.paths.report_dir) / "preprocessing_summary.json",
    )
    logger.info("Preprocessing summary written to %s.", report_path)
    return outputs

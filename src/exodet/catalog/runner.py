"""Catalog stage orchestration."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from exodet.catalog.builder import CatalogBuilder
from exodet.exceptions import PipelineError
from exodet.inference.scientific import build_reproduction_metadata
from exodet.inference.config import load_report_stage_config
from exodet.inference.containers import ScientificInferenceBatch
from exodet.inference.runner import run_inference
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.timing import Timer

__all__ = ["run_catalog_build"]

logger = logging.getLogger(__name__)


def run_catalog_build(
    config_path: Path | str,
    overrides: list[str] | None = None,
    inference_batch: ScientificInferenceBatch | None = None,
) -> dict[str, str]:
    """Builds and exports the exoplanet candidate catalog."""
    experiment, _report_cfg, catalog_cfg = load_report_stage_config(config_path, overrides)
    if not catalog_cfg.enabled:
        raise PipelineError("Catalog build is disabled in config.")

    batch = inference_batch if inference_batch is not None else run_inference(
        config_path, overrides
    )

    builder = CatalogBuilder(catalog_cfg)
    with Timer("catalog build") as timer:
        entries = builder.build(batch)
        out_dir = Path(experiment.paths.report_dir) / "catalog"
        paths = builder.export(entries, out_dir)

    write_json(
        build_reproduction_metadata(
            experiment,
            asdict(catalog_cfg),
            extra={
                "n_entries": len(entries),
                "runtime_seconds": timer.elapsed,
                "outputs": paths,
            },
        ),
        out_dir / "catalog_summary.json",
    )
    logger.info("Catalog built with %d entries.", len(entries))
    return paths

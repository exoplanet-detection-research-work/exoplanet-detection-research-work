"""End-to-end TCE orchestration for the ``tce`` CLI stage.

Loads every preprocessed light curve from ``paths.processed_dir``,
runs the configured TCE pipeline, and exports:

* the full candidate catalog (accepted + rejected) as JSON and CSV in
  ``paths.report_dir``;
* the run-level detection summary JSON;
* diagnostic figures for the first ``n_figure_targets`` targets in
  ``paths.figure_dir``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from exodet.data.serialization import load_light_curve
from exodet.exceptions import PipelineError
from exodet.tce.candidate import TransitCandidate, save_candidates
from exodet.tce.config import TCESearchConfig
from exodet.tce.diagnostics import export_candidates_csv, write_detection_summary
from exodet.tce.pipeline import TCEPipeline, TCEResult
from exodet.utils.timing import Timer
from exodet.visualization.tce import generate_tce_figures

__all__ = ["run_tce_search"]

logger = logging.getLogger(__name__)


def run_tce_search(config: TCESearchConfig) -> list[TCEResult]:
    """Runs the TCE search over every processed light curve.

    Args:
        config: The validated TCE configuration.

    Returns:
        Per-target TCE results, in input order.

    Raises:
        PipelineError: If no processed light curves are found.
        RegistryError: If a configured component is not registered.
    """
    processed_dir = Path(config.paths.processed_dir)
    files = sorted(processed_dir.glob(config.input_pattern))
    if not files:
        raise PipelineError(
            f"No processed light curves matching '{config.input_pattern}' in "
            f"{processed_dir}; run the 'preprocess' stage first."
        )

    pipeline = TCEPipeline(config)
    logger.info("TCE search over %d light curve(s) from %s.", len(files), processed_dir)

    results: list[TCEResult] = []
    catalog: list[TransitCandidate] = []
    with Timer(f"TCE search of {len(files)} target(s)", log=logger):
        for index, path in enumerate(files):
            curve = load_light_curve(path)
            with Timer(f"target {curve.target_id}", log=logger):
                result = pipeline.run(curve)
            results.append(result)
            catalog.extend(result.candidates)
            if index < config.n_figure_targets:
                generate_tce_figures(curve, result, config.paths.figure_dir)

    report_dir = Path(config.paths.report_dir)
    save_candidates(catalog, report_dir / "tce_candidates.json")
    export_candidates_csv(catalog, report_dir / "tce_candidates.csv")
    write_detection_summary(
        results, config.experiment_name, report_dir / "tce_detection_summary.json"
    )
    logger.info(
        "TCE search complete: %d candidate(s) across %d target(s) "
        "(catalog in %s).",
        len(catalog),
        len(results),
        report_dir,
    )
    return results

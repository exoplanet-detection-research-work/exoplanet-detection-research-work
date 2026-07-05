"""Report stage orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from exodet.exceptions import PipelineError
from exodet.inference.config import load_report_stage_config
from exodet.inference.containers import ScientificInferenceBatch
from exodet.inference.runner import run_inference, _load_dataset_split
from exodet.reporting.report import ReportGenerator
from exodet.representation.containers import RepresentationDataset
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.timing import Timer

__all__ = ["run_report_generation"]

logger = logging.getLogger(__name__)


def run_report_generation(
    config_path: Path | str,
    overrides: list[str] | None = None,
    inference_batch: ScientificInferenceBatch | None = None,
    dataset: RepresentationDataset | None = None,
) -> Path:
    """Generates candidate reports from inference results."""
    experiment, report_cfg, _catalog_cfg = load_report_stage_config(config_path, overrides)
    if not report_cfg.enabled:
        raise PipelineError("Report generation is disabled in config.")

    data = dataset if dataset is not None else _load_dataset_split(experiment, "test")
    batch = inference_batch if inference_batch is not None else run_inference(
        config_path, overrides, dataset=data
    )

    sample_map = {s.sample_id: s for s in data.samples}
    out_root = Path(report_cfg.output_dir or experiment.paths.report_dir) / "reports"
    ensure_dir(out_root)

    generator = ReportGenerator(report_cfg)
    generated = 0
    with Timer("report generation") as timer:
        for result in batch.results:
            if result.probability < report_cfg.probability_threshold:
                continue
            sample = sample_map.get(result.sample_id)
            if sample is None:
                continue
            generator.generate(result, sample, out_root / result.sample_id.replace("/", "_"))
            generated += 1
            if report_cfg.top_n > 0 and generated >= report_cfg.top_n:
                break

    summary_path = out_root / "report_generation_summary.json"
    write_json(
        {
            "experiment_name": experiment.experiment_name,
            "n_reports": generated,
            "runtime_seconds": timer.elapsed,
            "output_dir": str(out_root),
        },
        summary_path,
    )
    logger.info("Generated %d reports in %.2f s.", generated, timer.elapsed)
    return out_root

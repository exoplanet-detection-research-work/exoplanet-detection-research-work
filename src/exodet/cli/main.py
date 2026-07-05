"""``exodet`` command-line entrypoint.

Subcommands mirror the pipeline stages:

* ``exodet validate-config`` — parse and validate a YAML config.
* ``exodet info``            — show version, registries, and config summary.
* ``exodet download``        — fetch raw data via the configured source.
* ``exodet preprocess``      — run the preprocessing pipeline.
* ``exodet tce``             — run the BLS transit-candidate search.
* ``exodet dataset``         — build the ML-ready dataset from TCEs.
* ``exodet train``           — train the configured model.
* ``exodet evaluate``        — evaluate a trained model.
* ``exodet predict``         — score new targets with a trained model.
* ``exodet infer``           — run scientific inference on trained models.
* ``exodet report``          — generate candidate reports with figures.
* ``exodet catalog``         — build searchable exoplanet candidate catalog.
* ``exodet compare``         — compare multiple trained models.
* ``exodet benchmark``       — run scientific benchmarking suite.
* ``exodet ablation``        — run architecture ablation study.
* ``exodet sensitivity``     — run sensitivity / robustness analysis.
* ``exodet reproducibility`` — generate reproducibility report.
* ``exodet experiment``     — register and run a managed experiment.
* ``exodet sweep``          — run a hyperparameter sweep campaign.
* ``exodet leaderboard``    — build experiment leaderboards.
* ``exodet reproduce``      — validate experiment reproducibility.

Every subcommand takes ``--config`` and optional repeated
``--override dotted.key=value`` flags, then dispatches to a stage
runner. Stage runners for not-yet-implemented pipeline stages raise
:class:`~exodet.exceptions.PipelineError` with a clear message; they
will be filled in as the corresponding modules are implemented.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

from exodet import __version__
from exodet.config import ExperimentConfig, load_config
from exodet.data.base import DATA_SOURCES, DATASETS
from exodet.evaluation.base import METRICS
from exodet.exceptions import ExoDetError, PipelineError
from exodet.features.base import FEATURE_EXTRACTORS
from exodet.models.base import MODELS
from exodet.preprocessing.base import PREPROCESSORS
from exodet.preprocessing.runner import run_preprocessing
from exodet.representation.augmentation import AUGMENTERS
from exodet.representation.config import load_representation_config
from exodet.representation.features import PHYSICS_EXTRACTORS
from exodet.representation.folding import PHASE_FOLDERS
from exodet.representation.runner import run_dataset_build
from exodet.representation.scaling import FEATURE_SCALERS
from exodet.representation.splitting import SPLITTERS
from exodet.representation.views import VIEW_BUILDERS
from exodet.tce.config import load_tce_config
from exodet.tce.grid import GRID_GENERATORS
from exodet.tce.harmonics import HARMONIC_REJECTERS
from exodet.tce.metrics import METRICS_COMPUTERS
from exodet.tce.peaks import PEAK_DETECTORS
from exodet.tce.ranking import RANKERS
from exodet.tce.runner import run_tce_search
from exodet.tce.search import SEARCH_ENGINES
from exodet.tce.validation import VALIDATORS
try:
    import exodet.models.registry  # noqa: F401 — neural architectures
except ImportError:
    pass
from exodet.training.base import TRAINERS
from exodet.utils.logging import setup_logging
from exodet.utils.seeding import seed_everything

__all__ = ["main", "build_parser"]

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Constructs the top-level argument parser with all subcommands.

    Returns:
        The fully configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="exodet",
        description="Config-driven exoplanet detection pipeline.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("validate-config", "Parse and validate an experiment config."),
        ("info", "Show version, registered components, and config summary."),
        ("download", "Fetch raw data via the configured data source."),
        ("preprocess", "Run the preprocessing pipeline on raw data."),
        ("tce", "Run the BLS transit-candidate search on processed curves."),
        ("dataset", "Build the ML-ready dataset from TCE candidates."),
        ("train", "Train the configured model."),
        ("evaluate", "Evaluate a trained model on the test split."),
        ("predict", "Score new targets with a trained model."),
        ("infer", "Run scientific inference on trained models."),
        ("report", "Generate candidate reports with figures."),
        ("catalog", "Build searchable exoplanet candidate catalog."),
        ("compare", "Compare multiple trained models statistically."),
        ("benchmark", "Run scientific benchmarking suite."),
        ("ablation", "Run architecture ablation study."),
        ("sensitivity", "Run sensitivity and robustness analysis."),
        ("reproducibility", "Generate reproducibility report."),
        ("experiment", "Register and run a managed experiment."),
        ("sweep", "Run a hyperparameter sweep campaign."),
        ("leaderboard", "Build experiment leaderboards."),
        ("reproduce", "Validate experiment reproducibility."),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument(
            "--config",
            "-c",
            required=command != "info",
            help="Path to the experiment YAML config.",
        )
        sub.add_argument(
            "--override",
            "-o",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Dotted-key config override (repeatable), "
            "e.g. -o training.epochs=100.",
        )
    return parser


def _prepare(args: argparse.Namespace) -> ExperimentConfig:
    """Loads config, then initializes logging and seeding for a run.

    Args:
        args: Parsed CLI arguments containing ``config`` and ``override``.

    Returns:
        The validated experiment configuration.
    """
    config = load_config(args.config, overrides=args.override)
    setup_logging(
        config.logging,
        log_dir=config.paths.log_dir,
        run_name=config.experiment_name,
    )
    seed_everything(config.seed)
    logger.info("Experiment '%s' initialized.", config.experiment_name)
    return config


def _run_validate_config(args: argparse.Namespace) -> int:
    """Validates a config file and prints a short confirmation.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    config = load_config(args.config, overrides=args.override)
    print(f"OK: '{args.config}' is a valid experiment config.")
    print(f"  experiment_name: {config.experiment_name}")
    print(f"  model:           {config.model.architecture.name}")
    print(f"  trainer:         {config.training.trainer.name}")
    print(f"  preprocessing:   {len(config.preprocessing.steps)} step(s)")
    print(f"  metrics:         {[m.name for m in config.evaluation.metrics]}")
    return 0


def _run_info(args: argparse.Namespace) -> int:
    """Prints version and registry contents; validates config if given.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    print(f"exodet {__version__}")
    registries = {
        "data sources": DATA_SOURCES,
        "datasets": DATASETS,
        "preprocessors": PREPROCESSORS,
        "tce grid generators": GRID_GENERATORS,
        "tce search engines": SEARCH_ENGINES,
        "tce peak detectors": PEAK_DETECTORS,
        "tce metrics": METRICS_COMPUTERS,
        "tce validators": VALIDATORS,
        "tce harmonic rejecters": HARMONIC_REJECTERS,
        "tce rankers": RANKERS,
        "phase folders": PHASE_FOLDERS,
        "view builders": VIEW_BUILDERS,
        "physics extractors": PHYSICS_EXTRACTORS,
        "feature scalers": FEATURE_SCALERS,
        "dataset splitters": SPLITTERS,
        "augmenters": AUGMENTERS,
        "feature extractors": FEATURE_EXTRACTORS,
        "models": MODELS,
        "trainers": TRAINERS,
        "metrics": METRICS,
    }
    print("Registered components:")
    for kind, registry in registries.items():
        names = ", ".join(registry) or "<none yet>"
        print(f"  {kind:<20} {names}")
    if args.config:
        print()
        return _run_validate_config(args)
    return 0


def _run_preprocess(args: argparse.Namespace) -> int:
    """Runs the preprocessing pipeline on the configured dataset.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    config = _prepare(args)
    outputs = run_preprocessing(config)
    print(
        f"Preprocessed {len(outputs)} target(s) into "
        f"{config.paths.processed_dir} (figures: {config.paths.figure_dir}, "
        f"report: {config.paths.report_dir})."
    )
    return 0


def _run_tce(args: argparse.Namespace) -> int:
    """Runs the BLS transit-candidate search on processed light curves.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    config = load_tce_config(args.config, overrides=args.override)
    setup_logging(
        config.logging,
        log_dir=config.paths.log_dir,
        run_name=config.experiment_name,
    )
    seed_everything(config.seed)
    results = run_tce_search(config)
    n_accepted = sum(len(result.accepted) for result in results)
    print(
        f"TCE search: {n_accepted} accepted candidate(s) across "
        f"{len(results)} target(s) (catalog: {config.paths.report_dir}, "
        f"figures: {config.paths.figure_dir})."
    )
    return 0


def _run_dataset(args: argparse.Namespace) -> int:
    """Builds the ML-ready dataset from the TCE candidate catalog.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    config = load_representation_config(args.config, overrides=args.override)
    setup_logging(
        config.logging,
        log_dir=config.paths.log_dir,
        run_name=config.experiment_name,
    )
    seed_everything(config.seed)
    splits = run_dataset_build(config)
    print(
        f"Dataset build: {len(splits.train)}/{len(splits.validation)}/"
        f"{len(splits.test)} train/val/test samples "
        f"(output: {config.paths.processed_dir}/dataset)."
    )
    return 0


def _run_download(args: argparse.Namespace) -> int:
    """Fetches raw data via the configured data source."""
    config = _prepare(args)
    raise PipelineError(
        f"Stage 'download' is not implemented yet. Config "
        f"'{config.experiment_name}' was loaded and validated."
    )


def _run_train(args: argparse.Namespace) -> int:
    """Trains the configured model on representation dataset splits."""
    config = _prepare(args)
    import exodet.ml.trainer  # noqa: F401 — register supervised trainer
    from exodet.ml.runner import run_training

    result = run_training(config)
    n = len(result) if isinstance(result, list) else 1
    print(f"Training complete: {n} run(s) (checkpoints: {config.paths.checkpoint_dir}).")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    """Evaluates a trained model on the test split."""
    config = _prepare(args)
    from exodet.ml.runner import run_evaluation

    report = run_evaluation(config)
    print(f"Evaluation ({report.split}): {report.scores}")
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    """Scores the test split with a trained model."""
    config = _prepare(args)
    from exodet.ml.runner import run_predict

    result = run_predict(config)
    print(f"Predictions: {len(result.probabilities)} sample(s).")
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    """Runs scientific inference on dataset splits."""
    _prepare(args)
    from exodet.inference.runner import run_inference

    batch = run_inference(args.config, overrides=args.override)
    print(f"Inference: {len(batch)} candidate(s) analyzed.")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Generates publication-quality candidate reports."""
    _prepare(args)
    from exodet.reporting.runner import run_report_generation

    out_dir = run_report_generation(args.config, overrides=args.override)
    print(f"Reports written to {out_dir}.")
    return 0


def _run_catalog(args: argparse.Namespace) -> int:
    """Builds searchable exoplanet candidate catalog."""
    _prepare(args)
    from exodet.catalog.runner import run_catalog_build

    paths = run_catalog_build(args.config, overrides=args.override)
    print(f"Catalog outputs: {paths}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    """Compares multiple trained models."""
    _prepare(args)
    from exodet.config.loader import load_yaml, apply_overrides, _resolve_defaults
    from exodet.inference.runner import run_model_comparison

    raw = apply_overrides(
        _resolve_defaults(load_yaml(args.config), Path(args.config).parent),
        args.override,
    )
    models = raw.get("compare", {}).get("models", {})
    if not models:
        raise PipelineError("compare.models must list checkpoint paths in config.")
    report = run_model_comparison(args.config, dict(models), overrides=args.override)
    print(f"Model comparison: {report.model_names}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    from exodet.benchmarking.config import load_benchmark_stage_config
    from exodet.benchmarking.runner import run_benchmark

    experiment, _, _, _, _ = load_benchmark_stage_config(args.config, args.override)
    setup_logging(
        experiment.logging,
        log_dir=experiment.paths.log_dir,
        run_name=experiment.experiment_name,
    )
    seed_everything(experiment.seed)
    report = run_benchmark(args.config, overrides=args.override)
    print(f"Benchmark complete: {report.experiment_name} ({len(report.model_results)} models)")
    return 0


def _run_ablation(args: argparse.Namespace) -> int:
    from exodet.ablation.config import load_ablation_stage_config
    from exodet.ablation.runner import run_ablation

    experiment, _ = load_ablation_stage_config(args.config, args.override)
    setup_logging(
        experiment.logging,
        log_dir=experiment.paths.log_dir,
        run_name=experiment.experiment_name,
    )
    seed_everything(experiment.seed)
    payload = run_ablation(args.config, overrides=args.override)
    completed = sum(1 for row in payload["variants"] if row.get("status") == "completed")
    print(f"Ablation complete: {completed}/{len(payload['variants'])} variants")
    return 0


def _run_sensitivity(args: argparse.Namespace) -> int:
    from exodet.benchmarking.config import load_benchmark_stage_config
    from exodet.benchmarking.runner import run_sensitivity

    experiment, _, _, _, _ = load_benchmark_stage_config(args.config, args.override)
    setup_logging(
        experiment.logging,
        log_dir=experiment.paths.log_dir,
        run_name=experiment.experiment_name,
    )
    seed_everything(experiment.seed)
    payload = run_sensitivity(args.config, overrides=args.override)
    print(f"Sensitivity analysis: {len(payload.get('curves', {}))} perturbations")
    return 0


def _run_reproducibility(args: argparse.Namespace) -> int:
    from exodet.benchmarking.config import load_benchmark_stage_config
    from exodet.reproducibility.runner import run_reproducibility

    experiment, _, _, _, _ = load_benchmark_stage_config(args.config, args.override)
    setup_logging(
        experiment.logging,
        log_dir=experiment.paths.log_dir,
        run_name=experiment.experiment_name,
    )
    seed_everything(experiment.seed)
    payload = run_reproducibility(args.config, overrides=args.override)
    print(f"Reproducibility report: {payload['report_paths']}")
    return 0


def _run_experiment(args: argparse.Namespace) -> int:
    from exodet.experiments.config import load_experiments_stage_config
    from exodet.experiments.runner import run_experiment

    experiment, _, _, _, _, _ = load_experiments_stage_config(args.config, args.override)
    setup_logging(
        experiment.logging,
        log_dir=experiment.paths.log_dir,
        run_name=experiment.experiment_name,
    )
    seed_everything(experiment.seed)
    payload = run_experiment(args.config, overrides=args.override)
    print(f"Experiment complete: {payload['experiment_id']}")
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    from exodet.experiments.config import load_experiments_stage_config
    from exodet.experiments.runner import run_experiment_sweep

    experiment, _, _, _, _, _ = load_experiments_stage_config(args.config, args.override)
    setup_logging(
        experiment.logging,
        log_dir=experiment.paths.log_dir,
        run_name=experiment.experiment_name,
    )
    seed_everything(experiment.seed)
    payload = run_experiment_sweep(args.config, overrides=args.override)
    print(f"Sweep complete: {payload['sweep_id']}")
    return 0


def _run_leaderboard(args: argparse.Namespace) -> int:
    from exodet.experiments.config import load_experiments_stage_config
    from exodet.experiments.runner import run_leaderboard

    experiment, _, _, _, _, _ = load_experiments_stage_config(args.config, args.override)
    setup_logging(
        experiment.logging,
        log_dir=experiment.paths.log_dir,
        run_name=experiment.experiment_name,
    )
    payload = run_leaderboard(args.config, overrides=args.override)
    n = len(payload.get("comparison", {}).get("leaderboards", []))
    print(f"Leaderboard generated: {n} metrics")
    return 0


def _run_reproduce(args: argparse.Namespace) -> int:
    from exodet.experiments.config import load_experiments_stage_config
    from exodet.experiments.runner import run_reproduce_experiments

    experiment, _, _, _, _, _ = load_experiments_stage_config(args.config, args.override)
    setup_logging(
        experiment.logging,
        log_dir=experiment.paths.log_dir,
        run_name=experiment.experiment_name,
    )
    seed_everything(experiment.seed)
    payload = run_reproduce_experiments(args.config, overrides=args.override)
    print(f"Reproducibility: all_passed={payload.get('all_passed')}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 on handled failure).
    """
    # Headless backend: CLI stages only ever export figures to files,
    # and GUI backends (e.g. macosx) abort in non-interactive contexts.
    matplotlib.use("Agg")

    parser = build_parser()
    args = parser.parse_args(argv)

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "validate-config": _run_validate_config,
        "info": _run_info,
        "download": _run_download,
        "preprocess": _run_preprocess,
        "tce": _run_tce,
        "dataset": _run_dataset,
        "train": _run_train,
        "evaluate": _run_evaluate,
        "predict": _run_predict,
        "infer": _run_infer,
        "report": _run_report,
        "catalog": _run_catalog,
        "compare": _run_compare,
        "benchmark": _run_benchmark,
        "ablation": _run_ablation,
        "sensitivity": _run_sensitivity,
        "reproducibility": _run_reproducibility,
        "experiment": _run_experiment,
        "sweep": _run_sweep,
        "leaderboard": _run_leaderboard,
        "reproduce": _run_reproduce,
    }

    try:
        return handlers[args.command](args)
    except ExoDetError as exc:
        logging.getLogger("exodet").error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

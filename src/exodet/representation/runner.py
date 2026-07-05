"""End-to-end orchestration of the ``dataset`` CLI stage.

Reads processed light curves (``paths.processed_dir``) and the TCE
candidate catalog (``paths.report_dir``), builds one sample per
(accepted) candidate, splits without leakage, fits the feature scaler
on the *training split only*, applies it everywhere, optionally
augments the training split, and writes:

* ``<processed_dir>/dataset/{train,validation,test}.npz`` (+ JSON);
* the fitted scaler statistics JSON;
* CSV summaries (samples, features) and the build report JSON;
* diagnostic figures for the first ``n_figure_samples`` samples.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from exodet.data.serialization import load_light_curve
from exodet.exceptions import DataError, PipelineError
from exodet.representation.augmentation import AUGMENTERS, AugmentationPipeline
from exodet.representation.cache import RepresentationCache
from exodet.representation.config import RepresentationConfig
from exodet.representation.containers import DatasetSample, RepresentationDataset
from exodet.representation.pipeline import RepresentationPipeline
from exodet.representation.scaling import FEATURE_SCALERS
from exodet.representation.splitting import SPLITTERS, DatasetSplits
from exodet.tce.candidate import STATUS_CANDIDATE, load_candidates
from exodet.utils.io import ensure_dir, write_json
from exodet.utils.timing import Timer

__all__ = ["run_dataset_build"]

logger = logging.getLogger(__name__)


def _load_inputs(
    config: RepresentationConfig,
) -> list[tuple[Any, Any]]:
    """Pairs every candidate with its target's processed light curve.

    Args:
        config: The run configuration.

    Returns:
        List of (light curve, candidate) pairs.

    Raises:
        PipelineError: If inputs are missing.
    """
    catalog_path = Path(config.paths.report_dir) / config.candidates_file
    if not catalog_path.is_file():
        raise PipelineError(
            f"TCE catalog not found: {catalog_path}; run the 'tce' stage first."
        )
    candidates = load_candidates(catalog_path)
    if config.accepted_only:
        candidates = [c for c in candidates if c.status == STATUS_CANDIDATE]
    if not candidates:
        raise PipelineError(
            f"No {'accepted ' if config.accepted_only else ''}candidates in "
            f"{catalog_path}."
        )

    curves: dict[str, Any] = {}
    processed_dir = Path(config.paths.processed_dir)
    pairs = []
    for candidate in candidates:
        if candidate.target_id not in curves:
            slug = candidate.target_id.replace(" ", "_").lower()
            path = processed_dir / f"{slug}.npz"
            if not path.is_file():
                matches = [
                    p
                    for p in processed_dir.glob("*.npz")
                    if p.stem.replace("_", " ").lower()
                    == candidate.target_id.lower()
                ]
                if not matches:
                    logger.warning(
                        "No processed curve for target %s; skipping its "
                        "candidate(s).",
                        candidate.target_id,
                    )
                    curves[candidate.target_id] = None
                    continue
                path = matches[0]
            curves[candidate.target_id] = load_light_curve(path)
        if curves[candidate.target_id] is not None:
            pairs.append((curves[candidate.target_id], candidate))
    if not pairs:
        raise PipelineError("No (light curve, candidate) pairs could be formed.")
    return pairs


def _scale_splits(
    splits: DatasetSplits, config: RepresentationConfig, output_dir: Path
) -> DatasetSplits:
    """Fits the scaler on train only and applies it to every split.

    Args:
        splits: Unscaled splits.
        config: The run configuration.
        output_dir: Where the scaler statistics JSON is written.

    Returns:
        Splits with scaled feature vectors.
    """
    scaler = FEATURE_SCALERS.build(config.scaling.name, **config.scaling.params)
    if not splits.train:
        logger.warning("Empty training split; features left unscaled.")
        return splits
    names = splits.train[0].feature_names
    matrix = np.stack([s.features for s in splits.train])
    scaler.fit(matrix, names)
    stats_path = scaler.save(output_dir / "feature_scaler.json")
    logger.info("Feature scaler statistics saved to %s", stats_path)

    def _apply(samples: list[DatasetSample]) -> list[DatasetSample]:
        return [
            s.with_features(
                scaler.transform(s.features, s.feature_names),
                stage=f"feature_scaling({config.scaling.name})",
            )
            for s in samples
        ]

    return DatasetSplits(
        train=_apply(splits.train),
        validation=_apply(splits.validation),
        test=_apply(splits.test),
        meta={**splits.meta, "scaler": config.scaling.name},
    )


def _export_csv_summaries(
    samples: list[DatasetSample], report_dir: Path
) -> list[Path]:
    """Writes the sample table and feature matrix as CSV.

    Args:
        samples: All (unaugmented) samples.
        report_dir: Destination directory.

    Returns:
        The written file paths.
    """
    ensure_dir(report_dir)
    sample_path = report_dir / "dataset_samples.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "target_id",
                "candidate_id",
                "label",
                "weight",
                "period_days",
                "depth",
                "snr",
                "sde",
                "global_empty_fraction",
                "local_empty_fraction",
                "epoch_correction_days",
            ]
        )
        for s in samples:
            writer.writerow(
                [
                    s.sample_id,
                    s.target_id,
                    s.candidate.candidate_id,
                    s.label,
                    s.weight,
                    s.candidate.period_days,
                    s.candidate.depth,
                    s.candidate.snr,
                    s.candidate.sde,
                    s.meta.get("global_empty_fraction", ""),
                    s.meta.get("local_empty_fraction", ""),
                    s.meta.get("epoch_correction_days", ""),
                ]
            )

    feature_path = report_dir / "dataset_features.csv"
    names = samples[0].feature_names
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *names])
        for s in samples:
            writer.writerow([s.sample_id, *(float(v) for v in s.features)])
    return [sample_path, feature_path]


def run_dataset_build(config: RepresentationConfig) -> DatasetSplits:
    """Runs the full dataset construction stage.

    Args:
        config: The validated representation configuration.

    Returns:
        The final (scaled, optionally augmented) splits.

    Raises:
        PipelineError: If inputs are missing or nothing survives.
    """
    pairs = _load_inputs(config)

    cache = None
    if config.cache.enabled:
        cache_dir = config.cache.directory or (
            Path(config.paths.interim_dir) / "rep_cache"
        )
        cache = RepresentationCache(
            cache_dir, compress=config.cache.compress, mmap=config.cache.mmap
        )
    pipeline = RepresentationPipeline(config, cache=cache)

    samples: list[DatasetSample] = []
    rejected: list[dict[str, str]] = []
    with Timer(f"representation of {len(pairs)} candidate(s)", log=logger):
        for curve, candidate in pairs:
            try:
                samples.append(pipeline.build_sample(curve, candidate))
            except DataError as exc:
                logger.warning("Skipping %s: %s", candidate.candidate_id, exc)
                rejected.append(
                    {"candidate_id": candidate.candidate_id, "reason": str(exc)}
                )
    if not samples:
        raise PipelineError("Every candidate was rejected during view generation.")

    # Diagnostics need folded curves, so they are generated before
    # scaling mutates the features (views are unaffected by scaling).
    if config.n_figure_samples > 0:
        from exodet.visualization.representation import generate_representation_figures

        for curve, candidate in pairs[: config.n_figure_samples]:
            matching = [
                s for s in samples if s.candidate.candidate_id == candidate.candidate_id
            ]
            if matching:
                generate_representation_figures(
                    curve, candidate, pipeline, matching[0], config.paths.figure_dir
                )
        from exodet.visualization.representation import plot_feature_distributions

        plot_feature_distributions(samples, config.paths.figure_dir)

    splitter = SPLITTERS.build(config.splitting.name, **config.splitting.params)
    splits = splitter.split(samples)

    dataset_dir = ensure_dir(Path(config.paths.processed_dir) / "dataset")
    splits = _scale_splits(splits, config, dataset_dir)

    if config.augmentation.enabled and config.augmentation.steps:
        augmenters = [
            AUGMENTERS.build(step.name, **step.params)
            for step in config.augmentation.steps
        ]
        augmented = AugmentationPipeline(augmenters, seed=config.seed).augment(
            splits.train, copies=config.augmentation.copies
        )
        splits = DatasetSplits(
            train=splits.train + augmented,
            validation=splits.validation,
            test=splits.test,
            meta={**splits.meta, "n_augmented": len(augmented)},
        )

    for name, part in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        RepresentationDataset(
            part,
            version=config.dataset_version,
            meta={"split": name, "experiment": config.experiment_name},
        ).save(dataset_dir / f"{name}.npz")

    report_dir = Path(config.paths.report_dir)
    _export_csv_summaries(samples, report_dir)
    write_json(
        {
            "experiment_name": config.experiment_name,
            "dataset_version": config.dataset_version,
            "n_candidates": len(pairs),
            "n_samples": len(samples),
            "n_rejected": len(rejected),
            "rejected": rejected,
            "splits": {
                "train": len(splits.train),
                "validation": len(splits.validation),
                "test": len(splits.test),
            },
            "split_meta": splits.meta,
            "cache": cache.stats if cache else None,
            "feature_names": list(samples[0].feature_names),
        },
        report_dir / "dataset_build_summary.json",
    )
    logger.info(
        "Dataset build complete: %d samples -> %d/%d/%d (train/val/test) in %s.",
        len(samples),
        len(splits.train),
        len(splits.validation),
        len(splits.test),
        dataset_dir,
    )
    return splits

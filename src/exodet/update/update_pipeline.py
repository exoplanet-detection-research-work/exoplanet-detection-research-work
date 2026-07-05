"""Incremental scientific pipeline orchestration for dataset updates."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from exodet.config.schema import ExperimentConfig
from exodet.data.base import LightCurve
from exodet.data.serialization import load_light_curve, save_light_curve
from exodet.exceptions import DataError, PipelineError
from exodet.preprocessing.base import PreprocessingPipeline
from exodet.representation.cache import RepresentationCache
from exodet.representation.config import RepresentationConfig
from exodet.representation.containers import DatasetSample
from exodet.representation.pipeline import RepresentationPipeline
from exodet.representation.scaling import FEATURE_SCALERS, FeatureScaler
from exodet.tce.candidate import (
    STATUS_CANDIDATE,
    TransitCandidate,
    load_candidates,
    save_candidates,
)
from exodet.tce.config import TCESearchConfig
from exodet.tce.pipeline import TCEPipeline
from exodet.update.config import UpdateStageConfig
from exodet.update.dataset_registry import DatasetRegistry, TargetRecord
from exodet.update.versioning import append_to_splits
from exodet.utils.io import ensure_dir, sha256_of_file, write_json

__all__ = [
    "UpdatePipeline",
    "UpdateInputs",
    "TargetStageState",
    "parse_tic_ids_from_file",
    "resolve_update_inputs",
    "merge_tce_catalog",
    "merge_catalog_entries",
]

logger = logging.getLogger(__name__)

STAGE_ORDER = (
    "download",
    "preprocess",
    "tce",
    "representation",
    "registry",
)


@dataclass
class UpdateInputs:
    """Resolved inputs for an update run."""

    tic_ids: tuple[str, ...] = ()
    curves: tuple[LightCurve, ...] = ()
    source: str = "none"


@dataclass
class TargetStageState:
    """Per-target stage checkpoint for failure recovery."""

    tic_id: str
    target_id: str
    completed_stages: list[str] = field(default_factory=list)
    last_error: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TargetStageState:
        return cls(
            tic_id=str(raw["tic_id"]),
            target_id=str(raw.get("target_id", raw["tic_id"])),
            completed_stages=list(raw.get("completed_stages", [])),
            last_error=raw.get("last_error"),
            updated_at=str(raw.get("updated_at", "")),
        )

    def mark_complete(self, stage: str) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.last_error = None
        self.updated_at = datetime.now(UTC).isoformat()

    def is_complete(self, stage: str) -> bool:
        return stage in self.completed_stages


def _target_slug(target_id: str) -> str:
    return target_id.replace(" ", "_").replace("/", "-").lower()


def parse_tic_ids_from_file(path: Path | str) -> list[str]:
    """Parse TIC IDs from CSV, TXT, or JSON files."""
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"TIC input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for key in ("tic_id", "tic", "TIC", "TIC_ID", "target_id"):
                    if key in reader.fieldnames:
                        return [str(row[key]).strip() for row in reader if row.get(key)]
            handle.seek(0)
            plain = csv.reader(handle)
            return [row[0].strip() for row in plain if row and row[0].strip()]
    if suffix == ".txt":
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item).strip() for item in payload]
        if isinstance(payload, dict):
            for key in ("tic_ids", "tics", "targets", "ids"):
                if key in payload:
                    return [str(item).strip() for item in payload[key]]
        raise PipelineError(f"Unsupported JSON structure in {path}")
    raise PipelineError(f"Unsupported TIC file format: {path.suffix}")


def resolve_update_inputs(
    update: UpdateStageConfig,
    *,
    cli_tic_ids: Sequence[str] = (),
    cli_tic_file: str | None = None,
    cli_fits_dir: str | None = None,
    cli_processed_dir: str | None = None,
) -> UpdateInputs:
    """Resolve update inputs from YAML and CLI overrides."""
    tic_ids: list[str] = list(update.input_tic_ids)
    tic_ids.extend(str(t) for t in cli_tic_ids)

    input_file = cli_tic_file or update.input_file
    if input_file:
        tic_ids.extend(parse_tic_ids_from_file(input_file))

    curves: list[LightCurve] = []
    fits_dir = cli_fits_dir or update.fits_dir
    if fits_dir:
        curves.extend(_load_fits_directory(Path(fits_dir)))

    processed_dir = cli_processed_dir or update.processed_dir
    if processed_dir:
        curves.extend(_load_processed_directory(Path(processed_dir)))

    if curves:
        return UpdateInputs(tic_ids=tuple(), curves=tuple(curves), source="files")
    if tic_ids:
        deduped = list(dict.fromkeys(DatasetRegistry.normalize_tic_id(t) for t in tic_ids))
        return UpdateInputs(
            tic_ids=tuple(deduped),
            curves=tuple(),
            source="tic_ids",
        )
    raise PipelineError(
        "No update inputs supplied. Provide TIC IDs, a TIC file, FITS directory, "
        "or processed light-curve directory."
    )


def _load_processed_directory(directory: Path) -> list[LightCurve]:
    directory = Path(directory)
    if not directory.is_dir():
        raise PipelineError(f"Processed directory not found: {directory}")
    curves: list[LightCurve] = []
    for path in sorted(directory.glob("*.npz")):
        if path.name.startswith("dataset"):
            continue
        try:
            curves.append(load_light_curve(path))
        except DataError as exc:
            logger.warning("Skipping corrupted processed file %s: %s", path, exc)
    if not curves:
        raise PipelineError(f"No processed light curves found in {directory}")
    return curves


def _load_fits_directory(directory: Path) -> list[LightCurve]:
    directory = Path(directory)
    if not directory.is_dir():
        raise PipelineError(f"FITS directory not found: {directory}")
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise PipelineError(
            "astropy is required to load FITS light curves; install astropy."
        ) from exc

    curves: list[LightCurve] = []
    for path in sorted(directory.glob("*.fits")) + sorted(directory.glob("*.fit")):
        try:
            curves.append(_fits_to_light_curve(path, fits))
        except (DataError, OSError, ValueError) as exc:
            logger.warning("Skipping corrupted FITS file %s: %s", path, exc)
    if not curves:
        raise PipelineError(f"No readable FITS light curves found in {directory}")
    return curves


def _fits_to_light_curve(path: Path, fits_module: Any) -> LightCurve:
    with fits_module.open(path, memmap=False) as hdul:
        table = hdul[1].data if len(hdul) > 1 else hdul[0].data
        names = set(getattr(table, "columns", table).names if hasattr(table, "columns") else [])
        time_col = next((c for c in ("TIME", "time") if c in names), None)
        flux_col = next(
            (c for c in ("PDCSAP_FLUX", "SAP_FLUX", "FLUX", "flux") if c in names),
            None,
        )
        if time_col is None or flux_col is None:
            raise DataError(f"FITS file {path} missing TIME/FLUX columns.")
        time = np.asarray(table[time_col], dtype=np.float64)
        flux = np.asarray(table[flux_col], dtype=np.float64)
        flux_err = None
        for err_col in ("PDCSAP_FLUX_ERR", "FLUX_ERR", "flux_err"):
            if err_col in names:
                flux_err = np.asarray(table[err_col], dtype=np.float64)
                break
        quality = None
        for qual_col in ("QUALITY", "quality", "SAP_QUALITY"):
            if qual_col in names:
                quality = np.asarray(table[qual_col])
                break
        stem = path.stem.replace("_", " ")
        target_id = stem if stem.upper().startswith("TIC") else f"TIC {stem}"
        meta: dict[str, Any] = {"source_file": str(path)}
        if quality is not None:
            meta["quality"] = quality
        return LightCurve(
            target_id=target_id,
            time=time,
            flux=flux,
            flux_err=flux_err,
            label=-1,
            mission="tess",
            meta=meta,
        )


def merge_tce_catalog(
    catalog_path: Path,
    new_candidates: Iterable[TransitCandidate],
) -> list[TransitCandidate]:
    """Merge candidates into an existing catalog without duplicates."""
    existing = load_candidates(catalog_path) if catalog_path.is_file() else []
    by_id = {candidate.candidate_id: candidate for candidate in existing}
    for candidate in new_candidates:
        by_id[candidate.candidate_id] = candidate
    merged = list(by_id.values())
    save_candidates(merged, catalog_path)
    return merged


def merge_catalog_entries(
    catalog_dir: Path,
    output_name: str,
    new_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Append or update catalog entries by ``sample_id``."""
    catalog_dir = Path(catalog_dir)
    json_path = catalog_dir / f"{output_name}.json"
    existing: list[dict[str, Any]] = []
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        existing = list(payload.get("entries", []))
    by_sample = {entry["sample_id"]: entry for entry in existing}
    for entry in new_entries:
        by_sample[str(entry["sample_id"])] = entry
    merged = list(by_sample.values())
    write_json({"entries": merged, "n_entries": len(merged)}, json_path)
    return {"json": str(json_path), "n_entries": len(merged), "n_updated": len(new_entries)}


class UpdatePipeline:
    """Orchestrates incremental dataset growth for new targets."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        update: UpdateStageConfig,
        tce_config: TCESearchConfig,
        representation_config: RepresentationConfig,
    ) -> None:
        self.experiment = experiment
        self.update = update
        self.tce_config = tce_config
        self.representation_config = representation_config
        registry_path = update.registry_path or str(
            Path(experiment.paths.processed_dir) / "dataset_registry.json"
        )
        self.registry = DatasetRegistry(Path(registry_path))
        state_root = update.state_dir or str(
            Path(experiment.paths.interim_dir) / "update_state"
        )
        self.state_dir = ensure_dir(state_root)
        self.preprocess_pipeline = PreprocessingPipeline.from_config(experiment.preprocessing)
        self.tce_pipeline = TCEPipeline(tce_config)
        self.representation_pipeline = RepresentationPipeline(
            representation_config,
            cache=self._build_cache(),
        )

    def _build_cache(self) -> RepresentationCache | None:
        cache_cfg = self.representation_config.cache
        if not cache_cfg.enabled:
            return None
        cache_dir = cache_cfg.directory or (
            Path(self.experiment.paths.interim_dir) / "rep_cache"
        )
        return RepresentationCache(
            Path(cache_dir),
            compress=cache_cfg.compress,
            mmap=cache_cfg.mmap,
        )

    def run(
        self,
        inputs: UpdateInputs,
        *,
        force_reprocess: bool | None = None,
    ) -> dict[str, Any]:
        """Execute the incremental update pipeline."""
        force = self.update.force_reprocess if force_reprocess is None else force_reprocess
        targets = self._resolve_targets(inputs)
        results: list[dict[str, Any]] = []
        for target in targets:
            try:
                results.append(self._process_target(target, force=force))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.exception("Update failed for %s", target.get("tic_id"))
                state = self._load_state(target["tic_id"], target["target_id"])
                state.last_error = str(exc)
                state.updated_at = datetime.now(UTC).isoformat()
                self._save_state(state)
                results.append(
                    {
                        "tic_id": target["tic_id"],
                        "target_id": target["target_id"],
                        "status": "failed",
                        "error": str(exc),
                        "completed_stages": list(state.completed_stages),
                    }
                )
        summary = {
            "n_targets": len(results),
            "n_success": sum(1 for row in results if row.get("status") == "success"),
            "n_skipped": sum(1 for row in results if row.get("status") == "skipped"),
            "n_failed": sum(1 for row in results if row.get("status") == "failed"),
            "targets": results,
        }
        report_path = write_json(
            summary,
            Path(self.experiment.paths.report_dir) / "update_summary.json",
        )
        logger.info("Update summary written to %s", report_path)
        return summary

    def _resolve_targets(self, inputs: UpdateInputs) -> list[dict[str, str]]:
        targets: list[dict[str, str]] = []
        for curve in inputs.curves:
            tic = DatasetRegistry.normalize_tic_id(curve.target_id)
            targets.append({"tic_id": tic, "target_id": curve.target_id, "curve": curve})
        for tic in inputs.tic_ids:
            target_id = DatasetRegistry.format_target_id(tic)
            targets.append({"tic_id": tic, "target_id": target_id, "curve": None})
        return targets

    def _state_path(self, tic_id: str) -> Path:
        return self.state_dir / f"{DatasetRegistry.normalize_tic_id(tic_id)}.json"

    def _load_state(self, tic_id: str, target_id: str) -> TargetStageState:
        path = self._state_path(tic_id)
        if path.is_file():
            return TargetStageState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return TargetStageState(tic_id=tic_id, target_id=target_id)

    def _save_state(self, state: TargetStageState) -> None:
        write_json(state.to_dict(), self._state_path(state.tic_id))

    def _process_target(self, target: dict[str, Any], *, force: bool) -> dict[str, Any]:
        tic_id = target["tic_id"]
        target_id = target["target_id"]
        if not self.registry.should_process(tic_id, force=force):
            return {
                "tic_id": tic_id,
                "target_id": target_id,
                "status": "skipped",
                "reason": "already_registered",
            }

        state = self._load_state(tic_id, target_id)
        if force:
            state = TargetStageState(tic_id=tic_id, target_id=target_id)

        processed_dir = ensure_dir(self.experiment.paths.processed_dir)
        processed_path = processed_dir / f"{_target_slug(target_id)}.npz"
        catalog_path = Path(self.tce_config.paths.report_dir) / self.representation_config.candidates_file
        dataset_dir = processed_dir / "dataset"
        manifest_path = dataset_dir / "manifest.json"
        version = self.update.dataset_version or self.representation_config.dataset_version

        curve: LightCurve | None = target.get("curve")
        sample_ids: list[str] = []
        n_added = 0

        if curve is None and not state.is_complete("download"):
            logger.info("Downloading light curve for %s ...", target_id)
            curve = self._download_target(tic_id, target_id)
            state.mark_complete("download")
            self._save_state(state)
        elif curve is None and processed_path.is_file():
            curve = load_light_curve(processed_path)
        elif curve is None:
            curve = self._download_target(tic_id, target_id)

        if not state.is_complete("preprocess") or force:
            logger.info("Preprocessing %s ...", target_id)
            processed = self.preprocess_pipeline.apply(curve)
            save_light_curve(processed, processed_path)
            curve = processed
            state.mark_complete("preprocess")
            self._save_state(state)
        elif processed_path.is_file():
            curve = load_light_curve(processed_path)

        new_candidates: list[TransitCandidate] = []
        if not state.is_complete("tce") or force:
            logger.info("Running BLS/TCE search on %s ...", target_id)
            result = self.tce_pipeline.run(curve)
            new_candidates = list(result.candidates)
            merge_tce_catalog(catalog_path, new_candidates)
            state.mark_complete("tce")
            self._save_state(state)
        else:
            all_candidates = load_candidates(catalog_path)
            new_candidates = [c for c in all_candidates if c.target_id == target_id]

        accepted = [
            c for c in new_candidates if c.status == STATUS_CANDIDATE
        ] if self.representation_config.accepted_only else list(new_candidates)

        if accepted and (not state.is_complete("representation") or force):
            logger.info(
                "Building ML samples for %d candidate(s) on %s ...",
                len(accepted),
                target_id,
            )
            new_samples = self._build_samples(curve, accepted)
            new_samples = self._scale_samples(new_samples, dataset_dir)
            append_info = append_to_splits(
                dataset_dir,
                new_samples,
                split=self.update.append_split,
                version=version,
                experiment_name=self.experiment.experiment_name,
                manifest_path=manifest_path,
            )
            n_added = int(append_info.get("n_added", 0))
            sample_ids = list(append_info.get("added_sample_ids", []))
            state.mark_complete("representation")
            self._save_state(state)

        if not state.is_complete("registry") or force:
            record = TargetRecord(
                tic_id=tic_id,
                target_id=target_id,
                mission=str(curve.mission if curve else "TESS"),
                download_date=datetime.now(UTC).isoformat(),
                sectors=tuple(
                    str(int(s))
                    for s in np.unique(curve.meta.get("sector", np.array([])))
                )
                if curve and "sector" in curve.meta
                else (),
                processing_version=version,
                preprocessing_version=version,
                tce_version=version,
                phase_fold_version=version,
                dataset_split=self.update.append_split,
                sample_ids=tuple(sample_ids),
                meta={"n_candidates": len(new_candidates), "n_added_samples": n_added},
            )
            split_path = dataset_dir / f"{self.update.append_split}.npz"
            if split_path.is_file():
                record.dataset_checksum = sha256_of_file(split_path)
            self.registry.register(record)
            state.mark_complete("registry")
            self._save_state(state)

        return {
            "tic_id": tic_id,
            "target_id": target_id,
            "status": "success",
            "n_candidates": len(new_candidates),
            "n_added_samples": n_added,
            "sample_ids": sample_ids,
            "completed_stages": list(state.completed_stages),
        }

    def _download_target(self, tic_id: str, target_id: str) -> LightCurve:
        from exodet.update import sources  # noqa: F401 — register fetchers

        missions = self.update.missions or ("TESS",)
        download_cfg = dict(self.update.download)
        workers = int(download_cfg.get("parallel_workers", self.update.parallel_workers))
        raw_dir = ensure_dir(self.experiment.paths.raw_dir)

        curves = sources.fetch_tic_light_curves(
            [tic_id],
            missions=missions,
            destination=raw_dir,
            workers=workers,
            download_cfg=download_cfg,
        )
        if not curves:
            raise PipelineError(f"Failed to download light curve for TIC {tic_id}.")
        curve = curves[0]
        if curve.target_id != target_id:
            curve = LightCurve(
                target_id=target_id,
                time=curve.time,
                flux=curve.flux,
                flux_err=curve.flux_err,
                label=curve.label,
                mission=curve.mission,
                meta={**curve.meta, "original_target_id": curve.target_id},
                history=list(curve.history),
            )
        return curve

    def _build_samples(
        self,
        curve: LightCurve,
        candidates: list[TransitCandidate],
    ) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for candidate in candidates:
            try:
                samples.append(
                    self.representation_pipeline.build_sample(curve, candidate)
                )
            except DataError as exc:
                logger.warning(
                    "Skipping candidate %s during update: %s",
                    candidate.candidate_id,
                    exc,
                )
        if not samples:
            raise PipelineError(
                f"No representation samples could be built for {curve.target_id}."
            )
        return samples

    def _scale_samples(
        self,
        samples: list[DatasetSample],
        dataset_dir: Path,
    ) -> list[DatasetSample]:
        scaler_path = dataset_dir / "feature_scaler.json"
        scaler = FEATURE_SCALERS.build(
            self.representation_config.scaling.name,
            **self.representation_config.scaling.params,
        )
        if scaler_path.is_file():
            scaler = FeatureScaler.load(scaler_path)
        else:
            names = samples[0].feature_names
            matrix = np.stack([sample.features for sample in samples])
            scaler.fit(matrix, names)
            scaler.save(scaler_path)
        return [
            sample.with_features(
                scaler.transform(sample.features, sample.feature_names),
                stage=f"feature_scaling({self.representation_config.scaling.name})",
            )
            for sample in samples
        ]


def download_tic_batch(
    tic_ids: Sequence[str],
    destination: Path,
    *,
    missions: Sequence[str] = ("TESS",),
    workers: int = 4,
    download_cfg: dict[str, Any] | None = None,
) -> list[LightCurve]:
    """Public API for parallel TIC downloads."""
    from exodet.update import sources  # noqa: F401

    return sources.fetch_tic_light_curves(
        list(tic_ids),
        missions=missions,
        destination=Path(destination),
        workers=workers,
        download_cfg=download_cfg or {},
    )

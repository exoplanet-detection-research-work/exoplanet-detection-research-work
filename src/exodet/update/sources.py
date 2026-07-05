"""Data acquisition helpers for incremental updates."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from exodet.data.base import DATA_SOURCES, BaseDataSource, LightCurve
from exodet.data.serialization import save_light_curve
from exodet.exceptions import DataError, PipelineError
from exodet.update.dataset_registry import DatasetRegistry
from exodet.utils.io import ensure_dir

__all__ = [
    "fetch_tic_light_curves",
    "LightkurveMissionSource",
    "SyntheticTicSource",
]

logger = logging.getLogger(__name__)


@DATA_SOURCES.register("lightkurve_mission")
class LightkurveMissionSource(BaseDataSource):
    """Download light curves via lightkurve for TESS/Kepler/K2."""

    def __init__(
        self,
        tic_ids: Sequence[str] | None = None,
        mission: str = "TESS",
        **params: Any,
    ) -> None:
        self.tic_ids = [DatasetRegistry.normalize_tic_id(t) for t in (tic_ids or ())]
        self.mission = mission.upper()
        self.params = params

    def download(self, destination: Path) -> Path:
        curves = _download_with_lightkurve(
            self.tic_ids,
            mission=self.mission,
            destination=destination,
            **self.params,
        )
        manifest = destination / f"download_manifest_{self.mission.lower()}.json"
        manifest.write_text(
            f'{{"mission":"{self.mission}","n_targets":{len(curves)},'
            f'"downloaded_at":"{datetime.now(UTC).isoformat()}"}}',
            encoding="utf-8",
        )
        return destination

    def describe(self) -> dict[str, Any]:
        return {
            "source": "lightkurve_mission",
            "mission": self.mission,
            "n_tic_ids": len(self.tic_ids),
            "params": dict(self.params),
        }


@DATA_SOURCES.register("synthetic_tic")
class SyntheticTicSource(BaseDataSource):
    """Deterministic synthetic curves for tests and offline development."""

    def __init__(self, tic_ids: Sequence[str] | None = None, **params: Any) -> None:
        self.tic_ids = [DatasetRegistry.normalize_tic_id(t) for t in (tic_ids or ())]
        self.params = params

    def download(self, destination: Path) -> Path:
        ensure_dir(destination)
        for index, tic in enumerate(self.tic_ids):
            target_id = DatasetRegistry.format_target_id(tic)
            curve = _make_synthetic_tess_curve(
                target_id=target_id,
                seed=int(tic[-6:]) if tic[-6:].isdigit() else index,
                **self.params,
            )
            save_light_curve(curve, destination / f"{target_id.replace(' ', '_').lower()}.npz")
        return destination

    def describe(self) -> dict[str, Any]:
        return {
            "source": "synthetic_tic",
            "n_tic_ids": len(self.tic_ids),
            "params": dict(self.params),
        }


def _make_synthetic_tess_curve(
    target_id: str,
    *,
    seed: int = 0,
    n_per_sector: int = 1500,
    n_sectors: int = 2,
) -> LightCurve:
    """Deterministic synthetic TESS curve for offline update testing."""
    rng = np.random.default_rng(seed)
    cadence = 2.0 / (60.0 * 24.0)
    period = 1.3
    duration = 0.1
    depth = 0.008
    sector_span = n_per_sector * cadence
    times, fluxes, errs, quality, sectors = [], [], [], [], []
    for sector_index in range(n_sectors):
        offset = sector_index * (sector_span + 1.0)
        time = offset + np.arange(n_per_sector) * cadence
        baseline = 1000.0 * (1.0 + 0.5 * sector_index)
        flux = baseline * (
            1.0
            + 0.01 * np.sin(2.0 * np.pi * time / 2.0)
            + rng.normal(0.0, 3e-4, n_per_sector)
        )
        phase = time % period
        flux[phase < duration] -= baseline * depth
        err = np.full(n_per_sector, 3e-4 * baseline)
        flags = np.zeros(n_per_sector, dtype=np.int64)
        times.append(time)
        fluxes.append(flux)
        errs.append(err)
        quality.append(flags)
        sectors.append(np.full(n_per_sector, sector_index + 1, dtype=np.int64))
    return LightCurve(
        target_id=target_id,
        time=np.concatenate(times),
        flux=np.concatenate(fluxes),
        flux_err=np.concatenate(errs),
        label=-1,
        mission="tess",
        meta={
            "quality": np.concatenate(quality),
            "sector": np.concatenate(sectors),
        },
    )


def fetch_tic_light_curves(
    tic_ids: Sequence[str],
    *,
    missions: Sequence[str] = ("TESS",),
    destination: Path,
    workers: int = 4,
    download_cfg: dict[str, Any] | None = None,
) -> list[LightCurve]:
    """Fetch light curves for TIC IDs with parallel mission attempts."""
    cfg = download_cfg or {}
    backend = str(cfg.get("backend", "auto"))
    normalized = [DatasetRegistry.normalize_tic_id(t) for t in tic_ids]
    if not normalized:
        return []

    if backend == "synthetic":
        source = SyntheticTicSource(tic_ids=normalized, **cfg.get("synthetic_params", {}))
        source.download(destination)
        return _load_downloaded_curves(destination, normalized)

    curves: list[LightCurve] = []
    if backend in ("auto", "lightkurve"):
        try:
            import lightkurve  # noqa: F401
        except ImportError:
            if backend == "lightkurve":
                raise PipelineError(
                    "lightkurve backend requested but lightkurve is not installed."
                ) from None
        else:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {
                    pool.submit(
                        _download_single_tic,
                        tic,
                        missions=missions,
                        destination=destination,
                        download_cfg=cfg,
                    ): tic
                    for tic in normalized
                }
                for future in as_completed(futures):
                    tic = futures[future]
                    try:
                        curves.append(future.result())
                    except Exception as exc:
                        logger.warning("Download failed for TIC %s: %s", tic, exc)

    if curves:
        return curves

    if backend == "auto":
        logger.info("Falling back to synthetic TIC backend for %d target(s).", len(normalized))
        source = SyntheticTicSource(tic_ids=normalized, **cfg.get("synthetic_params", {}))
        source.download(destination)
        return _load_downloaded_curves(destination, normalized)

    missing = [tic for tic in normalized if tic not in {c.target_id.split()[-1] for c in curves}]
    raise PipelineError(f"Could not download TIC IDs: {missing}")


def _load_downloaded_curves(destination: Path, tic_ids: Sequence[str]) -> list[LightCurve]:
    from exodet.data.serialization import load_light_curve

    curves: list[LightCurve] = []
    for tic in tic_ids:
        target_id = DatasetRegistry.format_target_id(tic)
        slug = target_id.replace(" ", "_").lower()
        path = destination / f"{slug}.npz"
        if path.is_file():
            curves.append(load_light_curve(path))
    return curves


def _download_single_tic(
    tic_id: str,
    *,
    missions: Sequence[str],
    destination: Path,
    download_cfg: dict[str, Any],
) -> LightCurve:
    last_error: Exception | None = None
    for mission in missions:
        try:
            curves = _download_with_lightkurve(
                [tic_id],
                mission=mission,
                destination=destination,
                **download_cfg,
            )
            if curves:
                return curves[0]
        except Exception as exc:
            last_error = exc
            logger.debug("Mission %s failed for TIC %s: %s", mission, tic_id, exc)
    if last_error is not None:
        raise last_error
    raise PipelineError(f"No mission succeeded for TIC {tic_id}")


def _download_with_lightkurve(
    tic_ids: Sequence[str],
    *,
    mission: str,
    destination: Path,
    **params: Any,
) -> list[LightCurve]:
    import lightkurve as lk

    ensure_dir(destination)
    curves: list[LightCurve] = []
    flux_column = str(params.get("flux_column", "pdcsap_flux"))
    quality_bitmask = params.get("quality_bitmask", "default")

    for tic in tic_ids:
        target_id = DatasetRegistry.format_target_id(tic)
        search_id = int(DatasetRegistry.normalize_tic_id(tic))
        logger.info("Searching %s for TIC %s ...", mission, target_id)
        if mission == "TESS":
            search = lk.search_lightcurve(f"TIC {search_id}", mission="TESS")
        elif mission in ("KEPLER", "K2"):
            search = lk.search_lightcurve(search_id, mission=mission)
        else:
            raise PipelineError(f"Unsupported mission for lightkurve download: {mission}")

        if len(search) == 0:
            raise DataError(f"No {mission} light curves found for TIC {tic}.")

        logger.info(
            "Downloading %d %s light-curve file(s) for %s (may take several minutes) ...",
            len(search),
            mission,
            target_id,
        )
        collection = search.download_all(
            flux_column=flux_column,
            quality_bitmask=quality_bitmask,
        )
        if collection is None:
            raise DataError(f"Download returned no data for TIC {tic}.")

        logger.info("Stitching %d sector(s) for %s ...", len(search), target_id)
        if hasattr(collection, "stitch"):
            lc = collection.stitch()
        else:
            lc = collection[0] if hasattr(collection, "__getitem__") else collection

        time = np.asarray(lc.time.value, dtype=np.float64)
        flux = np.asarray(lc.flux.value, dtype=np.float64)
        flux_err = getattr(lc, "flux_err", None)
        flux_err_arr = (
            np.asarray(flux_err.value, dtype=np.float64) if flux_err is not None else None
        )
        quality = None
        if hasattr(lc, "quality") and lc.quality is not None:
            quality = np.asarray(lc.quality, dtype=np.int64)

        meta: dict[str, Any] = {
            "mission": mission.lower(),
            "flux_column": flux_column,
            "quality_bitmask": quality_bitmask,
            "download_date": datetime.now(UTC).isoformat(),
        }
        if quality is not None:
            meta["quality"] = quality
        if hasattr(lc, "sector") and lc.sector is not None:
            meta["sector"] = np.atleast_1d(np.asarray(lc.sector))

        curve = LightCurve(
            target_id=target_id,
            time=time,
            flux=flux,
            flux_err=flux_err_arr,
            label=-1,
            mission=mission.lower(),
            meta=meta,
            history=["lightkurve_download"],
        )
        save_light_curve(curve, destination / f"{target_id.replace(' ', '_').lower()}.npz")
        logger.info(
            "Downloaded %s: %d cadences across %d sector(s).",
            target_id,
            len(curve.time),
            len(search),
        )
        curves.append(curve)
    return curves

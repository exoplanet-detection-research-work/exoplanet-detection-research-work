"""Shared pytest fixtures and synthetic TESS data factories."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import pytest

matplotlib.use("Agg", force=True)

from exodet.data.base import DATASETS, BaseDataset, LightCurve  # noqa: E402

TESS_CADENCE_DAYS = 2.0 / (60.0 * 24.0)
TRANSIT_PERIOD_DAYS = 1.3
TRANSIT_DURATION_DAYS = 0.1
TRANSIT_DEPTH_REL = 0.008


def make_synthetic_tess_curve(
    target_id: str = "TIC 123456789",
    n_per_sector: int = 1500,
    n_sectors: int = 2,
    seed: int = 0,
    defects: bool = True,
) -> LightCurve:
    """Builds a realistic multi-sector TESS light curve.

    The curve contains stellar variability (2-day sinusoid), a
    periodic box transit, white noise, per-sector flux offsets, and —
    when ``defects`` is enabled — flagged cadences, NaN flux, strong
    upward outliers, and intra-sector gaps.
    """
    rng = np.random.default_rng(seed)
    sector_span = n_per_sector * TESS_CADENCE_DAYS

    times, fluxes, errs, quality, sectors = [], [], [], [], []
    for sector_index in range(n_sectors):
        offset = sector_index * (sector_span + 1.0)  # 1-day downlink gap
        time = offset + np.arange(n_per_sector) * TESS_CADENCE_DAYS
        baseline = 1000.0 * (1.0 + 0.5 * sector_index)

        flux = baseline * (
            1.0
            + 0.01 * np.sin(2.0 * np.pi * time / 2.0)
            + rng.normal(0.0, 3e-4, n_per_sector)
        )
        phase = time % TRANSIT_PERIOD_DAYS
        flux[phase < TRANSIT_DURATION_DAYS] -= baseline * TRANSIT_DEPTH_REL

        err = np.full(n_per_sector, 3e-4 * baseline)
        flags = np.zeros(n_per_sector, dtype=np.int64)

        if defects:
            bad = rng.choice(n_per_sector, size=40, replace=False)
            flags[bad[:30]] = 128  # manual exclude
            flags[bad[30:]] = 2048  # straylight
            nan_idx = rng.choice(n_per_sector, size=15, replace=False)
            flux[nan_idx] = np.nan
            outlier_idx = rng.choice(n_per_sector, size=8, replace=False)
            flux[outlier_idx] += 0.02 * baseline

        keep = np.ones(n_per_sector, dtype=bool)
        if defects:
            keep[700:720] = False  # ~29-minute intra-sector gap

        times.append(time[keep])
        fluxes.append(flux[keep])
        errs.append(err[keep])
        quality.append(flags[keep])
        sectors.append(np.full(keep.sum(), sector_index + 1, dtype=np.int64))

    return LightCurve(
        target_id=target_id,
        time=np.concatenate(times),
        flux=np.concatenate(fluxes),
        flux_err=np.concatenate(errs),
        label=1,
        mission="tess",
        meta={
            "quality": np.concatenate(quality),
            "sector": np.concatenate(sectors),
        },
    )


class SyntheticTessDataset(BaseDataset):
    """In-memory dataset of synthetic TESS curves for tests."""

    def __init__(self, n_targets: int = 2, n_per_sector: int = 800) -> None:
        self._curves = [
            make_synthetic_tess_curve(
                target_id=f"TIC {1000 + index}",
                n_per_sector=n_per_sector,
                seed=index,
            )
            for index in range(n_targets)
        ]

    def __len__(self) -> int:
        return len(self._curves)

    def __getitem__(self, index: int) -> LightCurve:
        return self._curves[index]

    @property
    def labels(self) -> npt.NDArray[np.int_]:
        return np.array([curve.label for curve in self._curves], dtype=np.int_)


if "synthetic_tess" not in DATASETS:
    DATASETS.register("synthetic_tess")(SyntheticTessDataset)


@pytest.fixture()
def light_curve() -> LightCurve:
    """A small synthetic light curve with a box-shaped transit."""
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 10.0, 500)
    flux = 1.0 + rng.normal(0.0, 1e-4, size=time.size)
    in_transit = (time % 2.5) < 0.1
    flux[in_transit] -= 0.01
    return LightCurve(
        target_id="TEST-001",
        time=time,
        flux=flux,
        label=1,
        mission="kepler",
    )


@pytest.fixture()
def tess_curve() -> LightCurve:
    """A defect-laden multi-sector synthetic TESS light curve."""
    return make_synthetic_tess_curve()


@pytest.fixture()
def clean_tess_curve() -> LightCurve:
    """A defect-free single-sector synthetic TESS light curve."""
    return make_synthetic_tess_curve(n_sectors=1, defects=False)


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """A minimal valid experiment config written to a temp file."""
    content = """\
experiment_name: unit_test
seed: 7
data:
  source:
    name: dummy_source
  dataset:
    name: dummy_dataset
model:
  architecture:
    name: dummy_model
training:
  trainer:
    name: dummy_trainer
  epochs: 3
"""
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path

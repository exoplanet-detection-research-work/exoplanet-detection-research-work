"""Dataset perturbations for sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import numpy.typing as npt

__all__ = [
    "PerturbationResult",
    "apply_perturbation",
    "iter_sensitivity_levels",
]


@dataclass(frozen=True, slots=True)
class PerturbationResult:
  """Perturbed feature matrix and metadata."""

  features: npt.NDArray[np.float64]
  labels: npt.NDArray[np.int_]
  perturbation: str
  level: float
  meta: dict[str, Any]


def _gaussian_noise(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    level: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    del labels
    sigma = level * np.std(features, axis=0, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    noisy = features + rng.normal(0.0, sigma * level, size=features.shape)
    return noisy.astype(np.float64), {"sigma_scale": level}


def _red_noise(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    level: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    del labels
    out = features.copy()
    for i in range(out.shape[0]):
        walk = np.cumsum(rng.normal(0.0, level, size=out.shape[1]))
        walk -= np.mean(walk)
        out[i] += walk
    return out.astype(np.float64), {"red_noise_amplitude": level}


def _missing_cadence(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    level: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    del labels
    out = features.copy()
    n_mask = int(round(level * out.shape[1]))
    if n_mask <= 0:
        return out, {"masked_bins": 0}
    for i in range(out.shape[0]):
        idx = rng.choice(out.shape[1], size=min(n_mask, out.shape[1]), replace=False)
        out[i, idx] = 0.0
    return out.astype(np.float64), {"masked_bins": n_mask}


def _period_offset(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    level: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    del labels, rng
    shift = int(round(level * features.shape[1]))
    return np.roll(features, shift, axis=1).astype(np.float64), {"period_bin_shift": shift}


def _epoch_offset(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    level: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    del rng
    shift = int(round(level * features.shape[1]))
    out = features.copy()
    # Epoch misalignment approximated by circular shift with amplitude damping.
    out = np.roll(out, shift, axis=1)
    out *= 1.0 - 0.5 * level
    return out.astype(np.float64), {"epoch_bin_shift": shift}


def _depth_scale(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    level: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    del labels, rng
    scale = 1.0 + level
    centered = features - np.mean(features, axis=1, keepdims=True)
    return (np.mean(features, axis=1, keepdims=True) + centered * scale).astype(np.float64), {
        "depth_scale": scale
    }


def _duration_scale(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    level: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    del labels, rng
    width = max(1, int(round((1.0 + level) * 0.05 * features.shape[1])))
    kernel = np.ones(width, dtype=np.float64) / width
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, features)
    return out.astype(np.float64), {"duration_kernel_width": width}


def _stellar_variability(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    level: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    del labels
    t = np.linspace(0.0, 2.0 * np.pi, features.shape[1], dtype=np.float64)
    out = features.copy()
    for i in range(out.shape[0]):
        amp = level * np.std(out[i])
        out[i] += amp * np.sin(t + rng.uniform(0.0, 2.0 * np.pi))
    return out.astype(np.float64), {"variability_amplitude": level}


_PERTURBATIONS: dict[str, Callable[..., tuple[npt.NDArray[np.float64], dict[str, Any]]]] = {
    "gaussian_noise": _gaussian_noise,
    "red_noise": _red_noise,
    "missing_cadence": _missing_cadence,
    "period_offset": _period_offset,
    "incorrect_period": _period_offset,
    "epoch_offset": _epoch_offset,
    "incorrect_epoch": _epoch_offset,
    "depth_scale": _depth_scale,
    "transit_depth": _depth_scale,
    "duration_scale": _duration_scale,
    "transit_duration": _duration_scale,
    "stellar_variability": _stellar_variability,
}


def apply_perturbation(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    perturbation: str,
    level: float,
    *,
    seed: int = 0,
) -> PerturbationResult:
    """Apply a named perturbation at a given severity level."""
    if perturbation not in _PERTURBATIONS:
        raise KeyError(f"Unknown perturbation: {perturbation}")
    rng = np.random.default_rng(seed)
    perturbed, meta = _PERTURBATIONS[perturbation](features, labels, level, rng)
    return PerturbationResult(
        features=perturbed,
        labels=np.asarray(labels, dtype=np.int_),
        perturbation=perturbation,
        level=level,
        meta=meta,
    )


def iter_sensitivity_levels(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    perturbations: tuple[str, ...],
    levels: tuple[float, ...],
    *,
    seed: int = 0,
) -> Iterator[PerturbationResult]:
    """Yield all perturbation/level combinations."""
    for p_idx, perturbation in enumerate(perturbations):
        for level in levels:
            yield apply_perturbation(
                features,
                labels,
                perturbation,
                level,
                seed=seed + p_idx * 1000 + int(level * 10000),
            )

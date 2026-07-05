"""Scientifically valid training-time augmentations (Module 3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.representation.augmentation import (
    AUGMENTERS,
)
from exodet.representation.containers import DatasetSample, RepresentationDataset

__all__ = [
    "TRAINING_AUGMENTERS",
    "TrainingAugmentationPipeline",
    "AugmentedTorchDataset",
    "build_training_augmentation",
]

logger = logging.getLogger(__name__)

TRAINING_AUGMENTERS: Registry[object] = Registry("training augmenter")


def _red_noise(rng: np.random.Generator, length: int, amplitude: float) -> npt.NDArray[np.float64]:
    """Generates correlated (red) noise via integrated white noise."""
    white = rng.normal(0.0, amplitude, size=length)
    red = np.cumsum(white)
    red -= np.median(red)
    scale = np.std(red)
    if scale > 0:
        red /= scale
    return red.astype(np.float64) * amplitude


@TRAINING_AUGMENTERS.register("red_noise")
class RedNoiseAugmenter:
    """Adds correlated baseline noise (stellar variability proxy)."""

    def __init__(self, amplitude_fraction: float = 0.05) -> None:
        if not 0 < amplitude_fraction <= 0.5:
            raise PipelineError("amplitude_fraction must be in (0, 0.5].")
        self.amplitude_fraction = amplitude_fraction

    def __call__(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        scale_g = np.std(sample.global_view) * self.amplitude_fraction
        scale_l = np.std(sample.local_view) * self.amplitude_fraction
        g_noise = _red_noise(rng, len(sample.global_view), scale_g)
        l_noise = _red_noise(rng, len(sample.local_view), scale_l)
        return sample.with_views(
            sample.global_view + g_noise,
            sample.local_view + l_noise,
            "red_noise_augment",
        )


@TRAINING_AUGMENTERS.register("baseline_drift")
class BaselineDriftAugmenter:
    """Slow linear baseline drift across the folded profile."""

    def __init__(self, max_slope: float = 0.02) -> None:
        self.max_slope = max_slope

    def __call__(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        slope = rng.uniform(-self.max_slope, self.max_slope)
        x = np.linspace(-0.5, 0.5, len(sample.global_view))
        drift = slope * x
        return sample.with_views(
            sample.global_view + drift,
            sample.local_view + drift * 0.1,
            "baseline_drift_augment",
        )


@TRAINING_AUGMENTERS.register("phase_shift")
class PhaseShiftAugmenter:
    """Small orbital phase shift via circular roll."""

    def __init__(self, max_shift_bins: int = 3) -> None:
        self.max_shift_bins = max_shift_bins

    def __call__(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        shift = int(rng.integers(-self.max_shift_bins, self.max_shift_bins + 1))
        return sample.with_views(
            np.roll(sample.global_view, shift),
            np.roll(sample.local_view, shift),
            "phase_shift_augment",
        )


@TRAINING_AUGMENTERS.register("depth_perturbation")
class DepthPerturbationAugmenter:
    """Small multiplicative perturbation of transit depth in local view."""

    def __init__(self, max_fraction: float = 0.15) -> None:
        self.max_fraction = max_fraction

    def __call__(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        factor = 1.0 + rng.uniform(-self.max_fraction, self.max_fraction)
        local = sample.local_view.copy()
        center = len(local) // 2
        window = max(3, len(local) // 10)
        local[center - window : center + window] *= factor
        return sample.with_views(sample.global_view, local, "depth_perturb_augment")


@TRAINING_AUGMENTERS.register("duration_perturbation")
class DurationPerturbationAugmenter:
    """Slight broadening/narrowing of the local transit via interpolation."""

    def __init__(self, max_scale: float = 0.1) -> None:
        self.max_scale = max_scale

    def __call__(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        scale = 1.0 + rng.uniform(-self.max_scale, self.max_scale)
        n = len(sample.local_view)
        src_x = np.linspace(0, 1, n)
        width = max(0.5, min(1.5, scale))
        dst_x = np.linspace(0, 1, int(n * width))
        if len(dst_x) < 3:
            return sample
        warped = np.interp(src_x, np.linspace(0, 1, len(dst_x)), sample.local_view[: len(dst_x)])
        return sample.with_views(sample.global_view, warped, "duration_perturb_augment")


@TRAINING_AUGMENTERS.register("stellar_variability")
class StellarVariabilityAugmenter:
    """Injects low-amplitude sinusoidal stellar variability."""

    def __init__(self, max_amplitude: float = 0.01, n_modes: int = 2) -> None:
        self.max_amplitude = max_amplitude
        self.n_modes = n_modes

    def __call__(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        phase = np.linspace(0, 2 * np.pi, len(sample.global_view), endpoint=False)
        signal = np.zeros_like(sample.global_view)
        for _ in range(self.n_modes):
            amp = rng.uniform(0, self.max_amplitude)
            freq = rng.uniform(0.5, 3.0)
            signal += amp * np.sin(freq * phase + rng.uniform(0, 2 * np.pi))
        return sample.with_views(
            sample.global_view + signal,
            sample.local_view + signal[: len(sample.local_view)],
            "stellar_variability_augment",
        )


@dataclass
class TrainingAugmentationPipeline:
    """Composable on-the-fly augmentation during training.

    Reuses representation-layer augmenters and adds training-specific
    transforms.  Every transform has a physical counterpart.
    """

    enabled: bool = False
    probability: float = 0.5
    steps: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 0

    def build(self) -> list[tuple[object, float]]:
        """Instantiates augmenters from YAML step specs."""
        pipeline: list[tuple[object, float]] = []
        for step in self.steps:
            name = str(step.get("name", "")).lower()
            prob = float(step.get("probability", self.probability))
            params = dict(step.get("params", {}))
            if name in TRAINING_AUGMENTERS:
                aug = TRAINING_AUGMENTERS.build(name, **params)
            elif name in AUGMENTERS:
                aug = AUGMENTERS.build(name, **params)
            else:
                raise PipelineError(f"Unknown augmenter '{name}'.")
            pipeline.append((aug, prob))
        return pipeline

    def apply(self, sample: DatasetSample, rng: np.random.Generator) -> DatasetSample:
        """Applies augmentations stochastically."""
        if not self.enabled:
            return sample
        for augmenter, prob in self.build():
            if rng.random() < prob:
                if hasattr(augmenter, "apply"):
                    sample = augmenter.apply(sample, rng)
                elif callable(augmenter):
                    sample = augmenter(sample, rng)
        return sample


def build_training_augmentation(config: dict[str, Any]) -> TrainingAugmentationPipeline:
    """Builds pipeline from YAML ``research.augmentation`` block."""
    return TrainingAugmentationPipeline(
        enabled=bool(config.get("enabled", False)),
        probability=float(config.get("probability", 0.5)),
        steps=list(config.get("steps", [])),
        seed=int(config.get("seed", 0)),
    )


class AugmentedTorchDataset:
    """PyTorch dataset wrapper applying augmentations on ``__getitem__``."""

    def __init__(
        self,
        dataset: RepresentationDataset,
        pipeline: TrainingAugmentationPipeline,
        allowed_indices: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self._dataset = dataset
        arrays = dataset.to_numpy()
        self._arrays = arrays
        self._pipeline = pipeline
        self._indices = (
            allowed_indices if allowed_indices is not None else np.arange(len(dataset))
        )
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        real_idx = int(self._indices[index])
        sample = self._dataset.samples[real_idx]
        if self._pipeline.enabled:
            sample = self._pipeline.apply(sample, self._rng)
        return {
            "global_view": sample.global_view,
            "local_view": sample.local_view,
            "features": sample.features,
            "labels": sample.label,
            "weights": sample.weight,
            "sample_id": sample.sample_id,
            "target_id": sample.target_id,
        }

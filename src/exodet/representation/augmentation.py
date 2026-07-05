"""Scientifically valid data augmentation (Module 10).

Augmenters act on the *views* of a :class:`DatasetSample` and are
restricted to transformations with a physical counterpart in real
photometry:

* ``gaussian_noise`` — extra white noise (detector/photon noise).
* ``timing_jitter`` — sub-bin ephemeris error (shifts the profile by a
  small fraction of the transit duration via linear interpolation).
* ``flux_scaling`` — small multiplicative calibration offset.
* ``dropout`` — random single-cadence losses (bins replaced by local
  linear interpolation, exactly like the empty-bin handling of real
  gaps).
* ``cadence_mask`` — one contiguous masked block (downlink gap).

Every parameter is bounded at construction: amplitudes that would
distort the transit shape into something unphysical are rejected.
Augmenters take an explicit ``numpy.random.Generator`` so results are
reproducible under the global seeding policy.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.representation.containers import DatasetSample

__all__ = [
    "AUGMENTERS",
    "GaussianNoiseAugmenter",
    "TimingJitterAugmenter",
    "FluxScalingAugmenter",
    "DropoutAugmenter",
    "CadenceMaskAugmenter",
    "AugmentationPipeline",
]

logger = logging.getLogger(__name__)

AUGMENTERS: Registry[object] = Registry("augmenter")


def _view_noise_scale(view: npt.NDArray[np.float64]) -> float:
    """Robust per-view noise scale (MAD-std of the values)."""
    median = float(np.median(view))
    return 1.4826 * float(np.median(np.abs(view - median)))


@AUGMENTERS.register("gaussian_noise")
class GaussianNoiseAugmenter:
    """Adds white noise scaled to each view's own scatter.

    Attributes:
        sigma_fraction: Added noise stddev as a fraction of the view's
            robust scatter; capped at 1 (never louder than the data).
    """

    _MAX_FRACTION = 1.0

    def __init__(self, sigma_fraction: float = 0.25) -> None:
        """Initializes the augmenter.

        Args:
            sigma_fraction: In ``(0, 1]``.

        Raises:
            PipelineError: If out of the physically sensible range.
        """
        if not 0 < sigma_fraction <= self._MAX_FRACTION:
            raise PipelineError(
                f"sigma_fraction must be in (0, {self._MAX_FRACTION}], got "
                f"{sigma_fraction}."
            )
        self.sigma_fraction = float(sigma_fraction)

    def apply(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        """Returns a noisier copy of the sample.

        Args:
            sample: The input sample (unchanged).
            rng: Seeded random generator.

        Returns:
            The augmented sample.
        """
        views = []
        for view in (sample.global_view, sample.local_view):
            sigma = self.sigma_fraction * _view_noise_scale(view)
            views.append(view + rng.normal(0.0, sigma, view.shape) if sigma > 0 else view.copy())
        return sample.with_views(
            views[0], views[1], stage=f"augment:gaussian_noise({self.sigma_fraction})"
        )


@AUGMENTERS.register("timing_jitter")
class TimingJitterAugmenter:
    """Shifts the views by a small ephemeris-error phase offset.

    Attributes:
        max_jitter_durations: Maximum shift as a fraction of the
            transit duration; capped at 0.25 so the transit remains
            effectively centered.
    """

    _MAX_JITTER = 0.25

    def __init__(self, max_jitter_durations: float = 0.1) -> None:
        """Initializes the augmenter.

        Args:
            max_jitter_durations: In ``(0, 0.25]``.

        Raises:
            PipelineError: If the jitter would decenter the transit.
        """
        if not 0 < max_jitter_durations <= self._MAX_JITTER:
            raise PipelineError(
                f"max_jitter_durations must be in (0, {self._MAX_JITTER}], got "
                f"{max_jitter_durations}."
            )
        self.max_jitter_durations = float(max_jitter_durations)

    @staticmethod
    def _shift(view: npt.NDArray[np.float64], bins: float) -> npt.NDArray[np.float64]:
        """Shifts a view by a fractional number of bins (linear interp)."""
        positions = np.arange(len(view), dtype=np.float64) - bins
        positions = np.clip(positions, 0.0, len(view) - 1.0)
        return np.interp(positions, np.arange(len(view)), view)

    def apply(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        """Returns a slightly shifted copy of the sample.

        The shift in *bins* differs between the views because they have
        different phase resolutions; both correspond to the same jitter
        in days.

        Args:
            sample: The input sample (unchanged).
            rng: Seeded random generator.

        Returns:
            The augmented sample.
        """
        candidate = sample.candidate
        duty = candidate.duration_days / candidate.period_days
        jitter_phase = rng.uniform(-1.0, 1.0) * self.max_jitter_durations * duty

        n_global = len(sample.global_view)
        global_bins = jitter_phase * n_global  # global span = 1 phase unit
        local_window = sample.meta.get("local_window_phase", 4.0 * duty)
        local_bins = (
            jitter_phase / local_window * len(sample.local_view)
            if local_window > 0
            else 0.0
        )
        return sample.with_views(
            self._shift(sample.global_view, global_bins),
            self._shift(sample.local_view, local_bins),
            stage=f"augment:timing_jitter({jitter_phase:+.2e} phase)",
        )


@AUGMENTERS.register("flux_scaling")
class FluxScalingAugmenter:
    """Applies a small multiplicative calibration factor.

    Attributes:
        max_scale_offset: Maximum relative deviation of the factor
            from 1; capped at 0.05 (5%, generous for calibration).
    """

    _MAX_OFFSET = 0.05

    def __init__(self, max_scale_offset: float = 0.01) -> None:
        """Initializes the augmenter.

        Args:
            max_scale_offset: In ``(0, 0.05]``.

        Raises:
            PipelineError: If the scaling is unrealistically large.
        """
        if not 0 < max_scale_offset <= self._MAX_OFFSET:
            raise PipelineError(
                f"max_scale_offset must be in (0, {self._MAX_OFFSET}], got "
                f"{max_scale_offset}."
            )
        self.max_scale_offset = float(max_scale_offset)

    def apply(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        """Returns a rescaled copy of the sample.

        Args:
            sample: The input sample (unchanged).
            rng: Seeded random generator.

        Returns:
            The augmented sample.
        """
        factor = 1.0 + rng.uniform(-1.0, 1.0) * self.max_scale_offset
        return sample.with_views(
            sample.global_view * factor,
            sample.local_view * factor,
            stage=f"augment:flux_scaling({factor:.4f})",
        )


@AUGMENTERS.register("dropout")
class DropoutAugmenter:
    """Simulates random single-cadence losses.

    Dropped bins are refilled by linear interpolation from their
    neighbours — the same treatment real empty bins receive — so the
    augmentation mimics genuine missing cadences rather than injecting
    artificial values.

    Attributes:
        dropout_fraction: Fraction of bins dropped per view; capped at
            0.2 to preserve the transit morphology.
    """

    _MAX_FRACTION = 0.2

    def __init__(self, dropout_fraction: float = 0.05) -> None:
        """Initializes the augmenter.

        Args:
            dropout_fraction: In ``(0, 0.2]``.

        Raises:
            PipelineError: If too much of the view would be dropped.
        """
        if not 0 < dropout_fraction <= self._MAX_FRACTION:
            raise PipelineError(
                f"dropout_fraction must be in (0, {self._MAX_FRACTION}], got "
                f"{dropout_fraction}."
            )
        self.dropout_fraction = float(dropout_fraction)

    @staticmethod
    def _drop(
        view: npt.NDArray[np.float64],
        indices: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        result = view.copy()
        keep = np.ones(len(view), dtype=bool)
        keep[indices] = False
        keep[[0, -1]] = True  # anchor the edges for interpolation
        positions = np.arange(len(view))
        result[~keep] = np.interp(positions[~keep], positions[keep], view[keep])
        return result

    def apply(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        """Returns a copy with random bins dropped and refilled.

        Args:
            sample: The input sample (unchanged).
            rng: Seeded random generator.

        Returns:
            The augmented sample.
        """
        views = []
        for view in (sample.global_view, sample.local_view):
            n_drop = max(1, int(self.dropout_fraction * len(view)))
            indices = rng.choice(len(view), size=n_drop, replace=False)
            views.append(self._drop(view, indices))
        return sample.with_views(
            views[0], views[1], stage=f"augment:dropout({self.dropout_fraction})"
        )


@AUGMENTERS.register("cadence_mask")
class CadenceMaskAugmenter:
    """Masks one contiguous block per view (simulated downlink gap).

    Attributes:
        max_mask_fraction: Maximum masked fraction of the view; capped
            at 0.15. The block never covers the central 10% of the
            local view, so the transit core is always preserved.
    """

    _MAX_FRACTION = 0.15

    def __init__(self, max_mask_fraction: float = 0.05) -> None:
        """Initializes the augmenter.

        Args:
            max_mask_fraction: In ``(0, 0.15]``.

        Raises:
            PipelineError: If the mask is unrealistically large.
        """
        if not 0 < max_mask_fraction <= self._MAX_FRACTION:
            raise PipelineError(
                f"max_mask_fraction must be in (0, {self._MAX_FRACTION}], got "
                f"{max_mask_fraction}."
            )
        self.max_mask_fraction = float(max_mask_fraction)

    def _mask(
        self,
        view: npt.NDArray[np.float64],
        rng: np.random.Generator,
        protect_center: bool,
    ) -> npt.NDArray[np.float64]:
        n = len(view)
        width = max(2, int(rng.uniform(0.5, 1.0) * self.max_mask_fraction * n))
        if protect_center:
            forbidden_lo = int(0.45 * n) - width
            forbidden_hi = int(0.55 * n)
            choices = [
                start
                for start in range(1, n - width - 1)
                if start > forbidden_hi or start < forbidden_lo
            ]
            if not choices:
                return view.copy()
            start = int(rng.choice(choices))
        else:
            start = int(rng.integers(1, max(2, n - width - 1)))
        indices = np.arange(start, min(start + width, n - 1))
        return DropoutAugmenter._drop(view, indices)

    def apply(
        self, sample: DatasetSample, rng: np.random.Generator
    ) -> DatasetSample:
        """Returns a copy with one masked block per view.

        Args:
            sample: The input sample (unchanged).
            rng: Seeded random generator.

        Returns:
            The augmented sample.
        """
        return sample.with_views(
            self._mask(sample.global_view, rng, protect_center=False),
            self._mask(sample.local_view, rng, protect_center=True),
            stage=f"augment:cadence_mask({self.max_mask_fraction})",
        )


class AugmentationPipeline:
    """Applies a configured sequence of augmenters to samples.

    Attributes:
        augmenters: The augmenters, applied in order.
        seed: Base seed; each sample gets an independent substream.
    """

    def __init__(self, augmenters: list[object], seed: int = 42) -> None:
        """Initializes the pipeline.

        Args:
            augmenters: Instantiated augmenters.
            seed: Base random seed.
        """
        self.augmenters = list(augmenters)
        self.seed = int(seed)

    def augment(
        self, samples: list[DatasetSample], copies: int = 1
    ) -> list[DatasetSample]:
        """Generates augmented copies of every sample.

        Args:
            samples: Source samples (returned untouched).
            copies: Augmented copies per source sample.

        Returns:
            The augmented copies only (originals are not included),
            with ``_augN`` suffixed sample ids.

        Raises:
            PipelineError: If ``copies`` is negative.
        """
        if copies < 0:
            raise PipelineError(f"copies must be >= 0, got {copies}.")
        augmented: list[DatasetSample] = []
        seeds = np.random.SeedSequence(self.seed).spawn(len(samples) * copies)
        index = 0
        for sample in samples:
            for copy in range(copies):
                rng = np.random.default_rng(seeds[index])
                index += 1
                result = sample
                for augmenter in self.augmenters:
                    result = augmenter.apply(result, rng)
                augmented.append(
                    replace(result, sample_id=f"{sample.sample_id}_aug{copy + 1}")
                )
        logger.info(
            "Augmented %d sample(s) x %d copies with %d augmenter(s).",
            len(samples),
            copies,
            len(self.augmenters),
        )
        return augmented

"""End-to-end TCE pipeline for a single light curve.

Chains the registered stages — grid generation, BLS search, peak
detection, metric computation, candidate construction, physical
validation, harmonic rejection, and ranking — all instantiated from a
:class:`~exodet.tce.config.TCESearchConfig`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from exodet.data.base import LightCurve
from exodet.tce.candidate import (
    STATUS_CANDIDATE,
    Periodogram,
    SearchGrid,
    TransitCandidate,
)
from exodet.tce.config import TCESearchConfig
from exodet.tce.grid import GRID_GENERATORS
from exodet.tce.harmonics import HARMONIC_REJECTERS
from exodet.tce.metrics import METRICS_COMPUTERS
from exodet.tce.peaks import PEAK_DETECTORS
from exodet.tce.ranking import RANKERS
from exodet.tce.search import SEARCH_ENGINES
from exodet.tce.validation import VALIDATORS

__all__ = ["TCEPipeline", "TCEResult"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TCEResult:
    """Outcome of the TCE search on one light curve.

    Attributes:
        target_id: The searched target.
        grid: The search grid used.
        periodogram: The full BLS spectrum.
        candidates: Every constructed candidate — accepted ones ranked
            first, followed by rejected ones (with reasons).
    """

    target_id: str
    grid: SearchGrid
    periodogram: Periodogram
    candidates: list[TransitCandidate] = field(default_factory=list)

    @property
    def accepted(self) -> list[TransitCandidate]:
        """Candidates that survived validation and harmonic rejection."""
        return [c for c in self.candidates if c.status == STATUS_CANDIDATE]

    @property
    def rejected(self) -> list[TransitCandidate]:
        """Candidates rejected at any stage (reasons preserved)."""
        return [c for c in self.candidates if c.status != STATUS_CANDIDATE]


class TCEPipeline:
    """Configured chain of TCE stages applied per light curve.

    Attributes:
        grid_generator: Builds the period/duration grid.
        engine: Computes the BLS periodogram and per-solution stats.
        peak_detector: Locates significant periodogram peaks.
        metrics_computer: Evaluates SDE/SNR/FAP and quality flags.
        validator: Applies physical plausibility criteria.
        harmonic_rejecter: Removes harmonics of stronger candidates.
        ranker: Orders the surviving candidates.
    """

    def __init__(self, config: TCESearchConfig) -> None:
        """Builds every stage from the configuration via the registries.

        Args:
            config: The validated TCE configuration.

        Raises:
            RegistryError: If a configured component is not registered.
        """
        self.grid_generator = GRID_GENERATORS.build(
            config.grid.name, **config.grid.params
        )
        self.engine = SEARCH_ENGINES.build(config.search.name, **config.search.params)
        self.peak_detector = PEAK_DETECTORS.build(
            config.peaks.name, **config.peaks.params
        )
        self.metrics_computer = METRICS_COMPUTERS.build(
            config.detection_metrics.name, **config.detection_metrics.params
        )
        self.validator = VALIDATORS.build(
            config.validation.name, **config.validation.params
        )
        self.harmonic_rejecter = HARMONIC_REJECTERS.build(
            config.harmonics.name, **config.harmonics.params
        )
        self.ranker = RANKERS.build(config.ranking.name, **config.ranking.params)

    def _build_candidate(
        self,
        light_curve: LightCurve,
        periodogram: Periodogram,
        index: int,
        ordinal: int,
    ) -> TransitCandidate:
        """Constructs one candidate from a periodogram peak.

        Args:
            light_curve: The searched light curve.
            periodogram: The full BLS spectrum.
            index: Peak index within the periodogram.
            ordinal: 1-based candidate number for the target.

        Returns:
            The fully populated (unvalidated) candidate.
        """
        period = float(periodogram.periods[index])
        duration = float(periodogram.duration[index])
        epoch = float(periodogram.transit_time[index])
        stats = self.engine.compute_stats(light_curve, period, duration, epoch)
        metrics = self.metrics_computer.compute(periodogram, index, stats)

        sector_meta = light_curve.meta.get("sector")
        if isinstance(sector_meta, np.ndarray):
            sectors = tuple(int(s) for s in np.unique(sector_meta))
        else:
            sectors = ()

        slug = light_curve.target_id.replace(" ", "_")
        return TransitCandidate(
            candidate_id=f"{slug}-{ordinal:02d}",
            target_id=light_curve.target_id,
            sectors=sectors,
            period_days=period,
            epoch_days=epoch,
            duration_days=duration,
            depth=metrics["depth"],
            depth_err=metrics["depth_err"],
            n_transits=metrics["n_transits"],
            n_expected_transits=metrics["n_expected_transits"],
            snr=metrics["snr"],
            sde=metrics["sde"],
            power=metrics["power"],
            fap=metrics["fap"],
            quality_flags=metrics["quality_flags"],
            meta={"peak_index": int(index), **metrics["diagnostics"]},
            history=(
                *light_curve.history,
                f"bls_search(objective={periodogram.objective})",
                f"peak_detection(index={int(index)})",
                "detection_metrics",
            ),
        )

    def run(self, light_curve: LightCurve) -> TCEResult:
        """Executes the full TCE search on one light curve.

        Args:
            light_curve: A preprocessed (detrended, normalized) curve.

        Returns:
            The search result with the periodogram and all candidates.

        Raises:
            PipelineError: If any stage fails irrecoverably.
        """
        grid = self.grid_generator.generate(light_curve)
        periodogram = self.engine.search(light_curve, grid)
        peak_indices = self.peak_detector.detect(periodogram)

        candidates = [
            self._build_candidate(light_curve, periodogram, int(index), ordinal)
            for ordinal, index in enumerate(peak_indices, start=1)
        ]
        candidates = self.validator.validate(candidates, grid)
        candidates = self.harmonic_rejecter.reject(candidates)
        candidates = self.ranker.rank(candidates)

        n_accepted = sum(1 for c in candidates if c.status == STATUS_CANDIDATE)
        logger.info(
            "Target %s: %d peak(s) -> %d accepted candidate(s), %d rejected.",
            light_curve.target_id,
            len(peak_indices),
            n_accepted,
            len(candidates) - n_accepted,
        )
        return TCEResult(
            target_id=light_curve.target_id,
            grid=grid,
            periodogram=periodogram,
            candidates=candidates,
        )

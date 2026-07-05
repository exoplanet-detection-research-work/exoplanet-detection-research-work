"""Physical plausibility validation of transit candidates.

Every candidate is checked against configurable physical and
statistical criteria; failing candidates are *retained* with
``status = "rejected_validation"`` and a rejection reason listing every
violated criterion, so downstream analyses (and the future ML stage)
can inspect the full population.
"""

from __future__ import annotations

import logging
import math

from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.tce.candidate import (
    STATUS_REJECTED_VALIDATION,
    SearchGrid,
    TransitCandidate,
)

__all__ = ["VALIDATORS", "PhysicalValidator"]

logger = logging.getLogger(__name__)

VALIDATORS: Registry["PhysicalValidator"] = Registry("TCE validator")


@VALIDATORS.register("physical")
class PhysicalValidator:
    """Rejects physically or statistically implausible candidates.

    Criteria (each individually configurable):
        * at least ``min_n_transits`` transit windows contain data;
        * observed/expected transit coverage >= ``min_coverage``;
        * duration/period <= ``max_duration_ratio`` (a central transit
          around a sun-like star has d/P well below ~0.1; larger values
          indicate blends, EBs, or detrending artefacts);
        * ``min_depth`` < depth <= ``max_depth`` (deeper events are
          eclipsing binaries or systematics, not planets);
        * period finite, positive, and inside the searched grid;
        * SDE >= ``min_sde`` and SNR >= ``min_snr``;
        * no NaN in any decision-relevant field.

    Attributes:
        min_n_transits: Minimum observed transit count.
        min_coverage: Minimum observed/expected transit fraction.
        max_duration_ratio: Maximum duration/period ratio.
        min_depth: Minimum fractional depth (> 0 rejects inverted fits).
        max_depth: Maximum fractional depth.
        min_sde: Minimum Signal Detection Efficiency.
        min_snr: Minimum signal-to-noise ratio.
    """

    def __init__(
        self,
        min_n_transits: int = 2,
        min_coverage: float = 0.5,
        max_duration_ratio: float = 0.2,
        min_depth: float = 0.0,
        max_depth: float = 0.5,
        min_sde: float = 7.0,
        min_snr: float = 5.0,
    ) -> None:
        """Initializes the validator.

        Args:
            min_n_transits: Minimum observed transits; >= 1.
            min_coverage: Coverage threshold in ``(0, 1]``.
            max_duration_ratio: Duration/period cap in ``(0, 1)``.
            min_depth: Lower depth bound (inclusive reject).
            max_depth: Upper depth bound.
            min_sde: SDE threshold.
            min_snr: SNR threshold.

        Raises:
            PipelineError: If thresholds are out of range.
        """
        if min_n_transits < 1:
            raise PipelineError(f"min_n_transits must be >= 1, got {min_n_transits}.")
        if not 0 < min_coverage <= 1:
            raise PipelineError(f"min_coverage must be in (0, 1], got {min_coverage}.")
        if not 0 < max_duration_ratio < 1:
            raise PipelineError(
                f"max_duration_ratio must be in (0, 1), got {max_duration_ratio}."
            )
        if not 0 <= min_depth < max_depth:
            raise PipelineError(
                f"Require 0 <= min_depth < max_depth, got {min_depth}, {max_depth}."
            )
        self.min_n_transits = int(min_n_transits)
        self.min_coverage = float(min_coverage)
        self.max_duration_ratio = float(max_duration_ratio)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_sde = float(min_sde)
        self.min_snr = float(min_snr)

    def _failures(
        self, candidate: TransitCandidate, grid: SearchGrid
    ) -> list[str]:
        """Collects every violated criterion for a candidate.

        Args:
            candidate: The candidate under test.
            grid: The search grid it was found in.

        Returns:
            Human-readable failure descriptions (empty if valid).
        """
        failures: list[str] = []

        numeric_fields = {
            "period_days": candidate.period_days,
            "epoch_days": candidate.epoch_days,
            "duration_days": candidate.duration_days,
            "depth": candidate.depth,
            "sde": candidate.sde,
            "snr": candidate.snr,
        }
        nan_fields = [name for name, value in numeric_fields.items() if not math.isfinite(value)]
        if nan_fields:
            failures.append(f"non-finite fields: {', '.join(nan_fields)}")
            return failures  # further comparisons are meaningless

        if not grid.min_period <= candidate.period_days <= grid.max_period:
            failures.append(
                f"period {candidate.period_days:.4f} d outside searched range "
                f"[{grid.min_period:.4f}, {grid.max_period:.4f}] d"
            )
        if candidate.n_transits < self.min_n_transits:
            failures.append(
                f"only {candidate.n_transits} observed transit(s) "
                f"(need >= {self.min_n_transits})"
            )
        if candidate.n_expected_transits > 0:
            coverage = candidate.n_transits / candidate.n_expected_transits
            if coverage < self.min_coverage:
                failures.append(
                    f"transit coverage {coverage:.2f} below {self.min_coverage:.2f}"
                )
        ratio = candidate.duration_days / candidate.period_days
        if ratio > self.max_duration_ratio:
            failures.append(
                f"duration/period {ratio:.3f} exceeds {self.max_duration_ratio:.3f}"
            )
        if not self.min_depth < candidate.depth <= self.max_depth:
            failures.append(
                f"depth {candidate.depth:.5f} outside "
                f"({self.min_depth}, {self.max_depth}]"
            )
        if candidate.sde < self.min_sde:
            failures.append(f"SDE {candidate.sde:.2f} below {self.min_sde:.2f}")
        if candidate.snr < self.min_snr:
            failures.append(f"SNR {candidate.snr:.2f} below {self.min_snr:.2f}")
        return failures

    def validate(
        self, candidates: list[TransitCandidate], grid: SearchGrid
    ) -> list[TransitCandidate]:
        """Applies all criteria to every candidate.

        Args:
            candidates: Candidates to validate.
            grid: The search grid used to find them.

        Returns:
            The same candidates (order preserved); failing ones carry
            ``status = "rejected_validation"`` and the full list of
            violated criteria, passing ones gain a provenance entry.
        """
        validated: list[TransitCandidate] = []
        n_rejected = 0
        for candidate in candidates:
            failures = self._failures(candidate, grid)
            if failures:
                n_rejected += 1
                validated.append(
                    candidate.with_status(
                        STATUS_REJECTED_VALIDATION,
                        "; ".join(failures),
                        stage=f"{type(self).__name__}:rejected",
                    )
                )
            else:
                validated.append(
                    candidate.with_meta(stage=f"{type(self).__name__}:accepted")
                )
        logger.info(
            "Validation: %d/%d candidate(s) rejected.", n_rejected, len(candidates)
        )
        return validated

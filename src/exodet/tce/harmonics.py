"""Harmonic and alias rejection among transit candidates.

A strong periodic signal produces periodogram peaks at integer
multiples and fractions of its true period (P/2, 2P, 3P, ...) and at
rational aliases (e.g. 3P/2). This stage keeps, for each family of
harmonically related candidates, only the most significant one; the
others are retained with ``status = "rejected_harmonic"`` and a reason
naming the surviving reference candidate and the matched ratio.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.tce.candidate import (
    STATUS_CANDIDATE,
    STATUS_REJECTED_HARMONIC,
    TransitCandidate,
)

__all__ = ["HARMONIC_REJECTERS", "PeriodRatioHarmonicRejecter"]

logger = logging.getLogger(__name__)

HARMONIC_REJECTERS: Registry["PeriodRatioHarmonicRejecter"] = Registry(
    "TCE harmonic rejecter"
)


@HARMONIC_REJECTERS.register("period_ratio")
class PeriodRatioHarmonicRejecter:
    """Rejects candidates whose period is a harmonic of a stronger one.

    Candidates are processed in decreasing order of a significance
    metric (default: power). For each candidate, the period ratio to
    every already-accepted candidate is compared against all rational
    ratios ``m/n`` with ``1 <= m, n <= max_multiple``; a match within
    ``tolerance`` (relative) marks the weaker candidate as a harmonic.
    ``m/n = 1/1`` also catches duplicate/alias peaks of the same signal.

    Attributes:
        tolerance: Relative tolerance on the ratio match.
        max_multiple: Largest integer numerator/denominator checked
            (5 covers P/2 ... P/5, 2P ... 5P, and rational aliases such
            as 3P/2 or 4P/3).
        metric: Candidate attribute used to decide which of two related
            candidates survives (``"power"``, ``"snr"``, or ``"sde"``).
    """

    _METRICS = ("power", "snr", "sde")

    def __init__(
        self,
        tolerance: float = 0.01,
        max_multiple: int = 5,
        metric: str = "power",
    ) -> None:
        """Initializes the rejecter.

        Args:
            tolerance: Relative ratio tolerance; in ``(0, 0.5)``.
            max_multiple: Highest harmonic order checked; >= 1.
            metric: Significance attribute for ordering.

        Raises:
            PipelineError: If parameters are out of range.
        """
        if not 0 < tolerance < 0.5:
            raise PipelineError(f"tolerance must be in (0, 0.5), got {tolerance}.")
        if max_multiple < 1:
            raise PipelineError(f"max_multiple must be >= 1, got {max_multiple}.")
        if metric not in self._METRICS:
            raise PipelineError(
                f"Unknown harmonic metric '{metric}'. Available: {self._METRICS}."
            )
        self.tolerance = float(tolerance)
        self.max_multiple = int(max_multiple)
        self.metric = metric
        self._ratios: tuple[Fraction, ...] = tuple(
            sorted(
                {
                    Fraction(m, n)
                    for m in range(1, self.max_multiple + 1)
                    for n in range(1, self.max_multiple + 1)
                }
            )
        )

    def _harmonic_match(self, period: float, reference: float) -> Fraction | None:
        """Tests whether two periods are harmonically related.

        Args:
            period: Candidate period in days.
            reference: Accepted reference period in days.

        Returns:
            The matched ratio ``period/reference`` as a fraction, or
            ``None`` if no ratio matches within tolerance.
        """
        ratio = period / reference
        for target in self._ratios:
            value = float(target)
            if abs(ratio - value) <= self.tolerance * value:
                return target
        return None

    def reject(self, candidates: list[TransitCandidate]) -> list[TransitCandidate]:
        """Marks harmonically related duplicates of stronger candidates.

        Only candidates with ``status == "candidate"`` participate;
        previously rejected candidates pass through untouched.

        Args:
            candidates: Candidates after physical validation.

        Returns:
            All candidates (order preserved) with harmonics of stronger
            signals marked as ``rejected_harmonic``.
        """
        active = [c for c in candidates if c.status == STATUS_CANDIDATE]
        ordered = sorted(
            active, key=lambda c: getattr(c, self.metric), reverse=True
        )

        accepted: list[TransitCandidate] = []
        decisions: dict[str, TransitCandidate] = {}
        for candidate in ordered:
            match: tuple[TransitCandidate, Fraction] | None = None
            for reference in accepted:
                ratio = self._harmonic_match(
                    candidate.period_days, reference.period_days
                )
                if ratio is not None:
                    match = (reference, ratio)
                    break
            if match is None:
                accepted.append(candidate)
                decisions[candidate.candidate_id] = candidate.with_meta(
                    stage=f"{type(self).__name__}:accepted"
                )
            else:
                reference, ratio = match
                decisions[candidate.candidate_id] = candidate.with_status(
                    STATUS_REJECTED_HARMONIC,
                    (
                        f"period is {ratio.numerator}/{ratio.denominator} x "
                        f"{reference.period_days:.5f} d of stronger candidate "
                        f"{reference.candidate_id} ({self.metric}="
                        f"{getattr(reference, self.metric):.2f})"
                    ),
                    stage=f"{type(self).__name__}:rejected",
                )

        n_rejected = sum(
            1
            for c in decisions.values()
            if c.status == STATUS_REJECTED_HARMONIC
        )
        logger.info(
            "Harmonic rejection: %d/%d active candidate(s) rejected.",
            n_rejected,
            len(active),
        )
        return [decisions.get(c.candidate_id, c) for c in candidates]

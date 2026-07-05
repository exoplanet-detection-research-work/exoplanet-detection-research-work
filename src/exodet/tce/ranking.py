"""Configurable ranking of accepted transit candidates.

Two strategies are provided: ranking by a single detection metric, and
a composite weighted score over min-max-normalized metrics. Ranks and
scores are written into candidate metadata; rejected candidates keep
their position in the returned list but receive no rank.
"""

from __future__ import annotations

import logging

import numpy as np

from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.tce.candidate import STATUS_CANDIDATE, TransitCandidate

__all__ = ["RANKERS", "MetricRanker", "CompositeRanker"]

logger = logging.getLogger(__name__)

RANKERS: Registry[object] = Registry("TCE ranker")

_RANKABLE_METRICS = ("snr", "sde", "power")


def _apply_ranks(
    candidates: list[TransitCandidate],
    scored: list[tuple[TransitCandidate, float]],
    stage: str,
) -> list[TransitCandidate]:
    """Writes rank/score metadata and reorders accepted candidates first.

    Args:
        candidates: The full candidate list (any status).
        scored: Accepted candidates paired with their ranking score.
        stage: Provenance entry naming the ranking strategy.

    Returns:
        Accepted candidates in rank order followed by rejected ones in
        their original order.
    """
    scored.sort(key=lambda pair: pair[1], reverse=True)
    ranked = [
        candidate.with_meta(stage=stage, rank=rank, ranking_score=float(score))
        for rank, (candidate, score) in enumerate(scored, start=1)
    ]
    rejected = [c for c in candidates if c.status != STATUS_CANDIDATE]
    return ranked + rejected


@RANKERS.register("metric")
class MetricRanker:
    """Ranks accepted candidates by a single detection metric.

    Attributes:
        metric: One of ``"snr"``, ``"sde"``, ``"power"``.
    """

    def __init__(self, metric: str = "snr") -> None:
        """Initializes the ranker.

        Args:
            metric: The candidate attribute to rank by.

        Raises:
            PipelineError: If the metric is not rankable.
        """
        if metric not in _RANKABLE_METRICS:
            raise PipelineError(
                f"Unknown ranking metric '{metric}'. Available: {_RANKABLE_METRICS}."
            )
        self.metric = metric

    def rank(self, candidates: list[TransitCandidate]) -> list[TransitCandidate]:
        """Ranks accepted candidates by the configured metric.

        Args:
            candidates: Candidates of any status.

        Returns:
            Accepted candidates first (best rank first, with
            ``meta["rank"]`` and ``meta["ranking_score"]`` set),
            followed by rejected candidates unchanged.
        """
        accepted = [c for c in candidates if c.status == STATUS_CANDIDATE]
        scored = [(c, float(getattr(c, self.metric))) for c in accepted]
        logger.info(
            "Ranked %d candidate(s) by %s.", len(accepted), self.metric
        )
        return _apply_ranks(
            candidates, scored, stage=f"{type(self).__name__}({self.metric})"
        )


@RANKERS.register("composite")
class CompositeRanker:
    """Ranks by a weighted sum of min-max-normalized metrics.

    Each metric is scaled to ``[0, 1]`` across the accepted candidates
    of the target (a constant metric contributes 1 for everyone), then
    combined with the configured weights. This makes weights comparable
    across metrics with wildly different scales (e.g. SNR vs power).

    Attributes:
        weights: Mapping from metric name to non-negative weight.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initializes the ranker.

        Args:
            weights: Metric weights; defaults to equal weights over
                SNR, SDE, and power.

        Raises:
            PipelineError: If a weight refers to an unknown metric, is
                negative, or all weights are zero.
        """
        weights = weights or {metric: 1.0 for metric in _RANKABLE_METRICS}
        unknown = set(weights) - set(_RANKABLE_METRICS)
        if unknown:
            raise PipelineError(
                f"Unknown metrics in ranking weights: {sorted(unknown)}. "
                f"Available: {_RANKABLE_METRICS}."
            )
        if any(weight < 0 for weight in weights.values()):
            raise PipelineError("Ranking weights must be non-negative.")
        total = sum(weights.values())
        if total == 0:
            raise PipelineError("At least one ranking weight must be positive.")
        self.weights = {name: weight / total for name, weight in weights.items()}

    def rank(self, candidates: list[TransitCandidate]) -> list[TransitCandidate]:
        """Ranks accepted candidates by the composite score.

        Args:
            candidates: Candidates of any status.

        Returns:
            Accepted candidates in composite-score order followed by
            rejected candidates unchanged.
        """
        accepted = [c for c in candidates if c.status == STATUS_CANDIDATE]
        if not accepted:
            return list(candidates)

        scores = np.zeros(len(accepted))
        for metric, weight in self.weights.items():
            values = np.array([getattr(c, metric) for c in accepted], dtype=float)
            finite = np.isfinite(values)
            values[~finite] = np.nanmin(values[finite]) if finite.any() else 0.0
            span = values.max() - values.min()
            normalized = (values - values.min()) / span if span > 0 else np.ones_like(values)
            scores += weight * normalized

        scored = list(zip(accepted, scores.tolist()))
        logger.info(
            "Ranked %d candidate(s) by composite score (weights: %s).",
            len(accepted),
            self.weights,
        )
        return _apply_ranks(
            candidates, scored, stage=f"{type(self).__name__}({self.weights})"
        )

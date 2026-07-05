"""Shared helpers for representation-layer tests."""

from __future__ import annotations

from dataclasses import replace

from exodet.tce import inject_box_transit, make_noise_light_curve
from exodet.tce.candidate import TransitCandidate
from tests.test_tce import make_candidate


def make_representation_pair(
    seed: int = 0,
    target_id: str | None = None,
    period_days: float = 2.7,
    duration_days: float = 0.12,
    depth: float = 0.004,
    epoch_days: float = 0.9,
    n_points: int = 15_000,
    noise_level: float = 5e-4,
) -> tuple:
    """Builds a (light curve, candidate) pair without running the TCE stage.

    Uses a long noise curve with an injected transit and a matching
    :class:`TransitCandidate` record. This keeps representation tests
    independent of BLS recovery on short TESS baselines.
    """
    target_id = target_id or f"TIC {9000 + seed}"
    curve = make_noise_light_curve(
        target_id=target_id,
        n_points=n_points,
        noise_level=noise_level,
        seed=seed,
    )
    injected = inject_box_transit(
        curve, period_days, duration_days, depth, epoch_days
    )
    candidate = make_candidate(
        candidate_id=f"{target_id.replace(' ', '_')}-01",
        target_id=target_id,
        period_days=period_days,
        epoch_days=epoch_days,
        duration_days=duration_days,
        depth=depth,
        meta={"depth_odd": depth * 0.99, "depth_even": depth * 1.01},
    )
    return injected, candidate


def wrong_period_candidate(candidate: TransitCandidate, factor: float = 1.7) -> TransitCandidate:
    """Returns a copy with an incorrect period (for mis-fold edge cases)."""
    return replace(
        candidate,
        period_days=candidate.period_days * factor,
        candidate_id=candidate.candidate_id + "-wrong",
    )

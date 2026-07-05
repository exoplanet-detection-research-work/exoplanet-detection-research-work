"""Phase folding and robust transit alignment (Modules 1 and 4).

Folding computes ``phi = ((t - epoch) / P) mod 1``, remapped to
``[-0.5, 0.5)`` so the transit sits at phase 0. It is robust to
multiple sectors (already merged in time), observational gaps (phase
coverage simply becomes uneven), duplicated cadences (deduplicated by
phase with flux averaging), and uneven sampling (all downstream binning
is density-aware).

Alignment corrects small ephemeris errors: catalog epochs from the BLS
grid can be off by a fraction of the transit duration, and a missing
central cadence biases naive minimum-finding. The corrector uses the
flux-weighted centroid of the below-baseline dip within one duration of
phase 0 — a statistic that is insensitive to individual missing
cadences — and caps the correction at half a duration.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.representation.containers import PhaseFoldedCurve
from exodet.tce.candidate import TransitCandidate

__all__ = ["PHASE_FOLDERS", "PhaseFolder", "fold_phase"]

logger = logging.getLogger(__name__)

PHASE_FOLDERS: Registry["PhaseFolder"] = Registry("phase folder")


def fold_phase(
    time: npt.NDArray[np.float64], period_days: float, epoch_days: float
) -> npt.NDArray[np.float64]:
    """Computes transit-centered orbital phase.

    Args:
        time: Observation times in days.
        period_days: Orbital period in days.
        epoch_days: Mid-transit epoch in days.

    Returns:
        Phases in ``[-0.5, 0.5)`` with the transit at phase 0.
    """
    return ((time - epoch_days) / period_days + 0.5) % 1.0 - 0.5


@PHASE_FOLDERS.register("standard")
class PhaseFolder:
    """Folds light curves on candidate ephemerides with alignment.

    Attributes:
        align: Whether to apply the epoch-offset correction.
        max_correction_durations: Cap on the alignment correction, in
            units of the transit duration.
        alignment_window_durations: Half-width of the search window
            around phase 0, in transit durations.
        deduplicate: Whether to merge cadences with (numerically)
            identical phases by averaging their flux.
        phase_min: Lower edge of the retained phase range.
        phase_max: Upper edge of the retained phase range.
    """

    def __init__(
        self,
        align: bool = True,
        max_correction_durations: float = 0.5,
        alignment_window_durations: float = 1.0,
        deduplicate: bool = True,
        phase_min: float = -0.5,
        phase_max: float = 0.5,
    ) -> None:
        """Initializes the folder.

        Args:
            align: Enable transit alignment.
            max_correction_durations: Correction cap; positive.
            alignment_window_durations: Alignment window; positive.
            deduplicate: Merge duplicated cadences.
            phase_min: Retained range lower edge; in ``[-0.5, 0)``.
            phase_max: Retained range upper edge; in ``(0, 0.5]``.

        Raises:
            PipelineError: If any parameter is out of range.
        """
        if max_correction_durations <= 0:
            raise PipelineError(
                f"max_correction_durations must be > 0, got "
                f"{max_correction_durations}."
            )
        if alignment_window_durations <= 0:
            raise PipelineError(
                f"alignment_window_durations must be > 0, got "
                f"{alignment_window_durations}."
            )
        if not -0.5 <= phase_min < 0 < phase_max <= 0.5:
            raise PipelineError(
                f"Require -0.5 <= phase_min < 0 < phase_max <= 0.5, got "
                f"{phase_min}, {phase_max}."
            )
        self.align = align
        self.max_correction_durations = float(max_correction_durations)
        self.alignment_window_durations = float(alignment_window_durations)
        self.deduplicate = deduplicate
        self.phase_min = float(phase_min)
        self.phase_max = float(phase_max)

    def _epoch_correction(
        self,
        phase: npt.NDArray[np.float64],
        flux: npt.NDArray[np.float64],
        candidate: TransitCandidate,
    ) -> float:
        """Estimates the epoch offset from the folded dip centroid.

        Args:
            phase: Transit-centered phases.
            flux: Corresponding flux values.
            candidate: The candidate providing period and duration.

        Returns:
            The epoch correction in days (0 when it cannot be
            estimated reliably).
        """
        duty = candidate.duration_days / candidate.period_days
        half_window = self.alignment_window_durations * duty
        window = np.abs(phase) < half_window
        if np.count_nonzero(window) < 5:
            return 0.0

        window_phase = phase[window]
        window_flux = flux[window]
        # Depth below the local baseline; only genuine dips contribute.
        baseline = float(np.median(window_flux))
        depth = baseline - window_flux
        positive = depth > 0
        total = float(depth[positive].sum())
        if total <= 0:
            return 0.0
        centroid = float((window_phase[positive] * depth[positive]).sum() / total)

        cap = self.max_correction_durations * duty
        centroid = float(np.clip(centroid, -cap, cap))
        return centroid * candidate.period_days

    def fold(
        self, light_curve: LightCurve, candidate: TransitCandidate
    ) -> PhaseFoldedCurve:
        """Folds a light curve on a candidate ephemeris.

        Args:
            light_curve: The (preprocessed) light curve; may contain
                multiple sectors, gaps, and NaNs.
            candidate: The candidate providing period, epoch, and
                duration.

        Returns:
            The folded curve, phase-sorted, aligned, and restricted to
            the configured phase range.

        Raises:
            PipelineError: If the ephemeris is invalid or fewer than 5
                finite cadences remain.
        """
        if not np.isfinite(candidate.period_days) or candidate.period_days <= 0:
            raise PipelineError(
                f"Candidate {candidate.candidate_id}: invalid period "
                f"{candidate.period_days}."
            )
        if not np.isfinite(candidate.epoch_days):
            raise PipelineError(
                f"Candidate {candidate.candidate_id}: invalid epoch "
                f"{candidate.epoch_days}."
            )

        finite = np.isfinite(light_curve.time) & np.isfinite(light_curve.flux)
        time = light_curve.time[finite]
        flux = light_curve.flux[finite]
        flux_err = (
            light_curve.flux_err[finite] if light_curve.flux_err is not None else None
        )
        if time.size < 5:
            raise PipelineError(
                f"Candidate {candidate.candidate_id}: only {time.size} finite "
                "cadences; folding requires at least 5."
            )

        epoch = candidate.epoch_days
        phase = fold_phase(time, candidate.period_days, epoch)
        correction = 0.0
        if self.align:
            correction = self._epoch_correction(phase, flux, candidate)
            if correction != 0.0:
                epoch = epoch + correction
                phase = fold_phase(time, candidate.period_days, epoch)

        order = np.argsort(phase, kind="stable")
        phase = phase[order]
        flux = flux[order]
        if flux_err is not None:
            flux_err = flux_err[order]

        n_duplicates = 0
        if self.deduplicate and phase.size > 1:
            unique_mask = np.empty(phase.size, dtype=bool)
            unique_mask[0] = True
            np.not_equal(phase[1:], phase[:-1], out=unique_mask[1:])
            n_duplicates = int(np.count_nonzero(~unique_mask))
            if n_duplicates:
                # Average flux over runs of identical phase.
                group = np.cumsum(unique_mask) - 1
                counts = np.bincount(group)
                flux = np.bincount(group, weights=flux) / counts
                if flux_err is not None:
                    # Error of the mean of the duplicated cadences.
                    flux_err = np.sqrt(
                        np.bincount(group, weights=flux_err**2)
                    ) / counts
                phase = phase[unique_mask]

        in_range = (phase >= self.phase_min) & (phase <= self.phase_max)
        phase = phase[in_range]
        flux = flux[in_range]
        if flux_err is not None:
            flux_err = flux_err[in_range]

        meta: dict[str, Any] = {
            "n_input_cadences": int(light_curve.time.size),
            "n_folded_cadences": int(phase.size),
            "n_duplicates_merged": n_duplicates,
            "phase_range": [self.phase_min, self.phase_max],
            "aligned": self.align,
        }
        sectors = light_curve.meta.get("sector")
        if isinstance(sectors, np.ndarray):
            meta["sectors"] = [int(s) for s in np.unique(sectors)]

        logger.debug(
            "Candidate %s: folded %d cadences (epoch correction %+.5f d).",
            candidate.candidate_id,
            phase.size,
            correction,
        )
        return PhaseFoldedCurve(
            candidate_id=candidate.candidate_id,
            target_id=candidate.target_id,
            phase=phase,
            flux=flux,
            flux_err=flux_err,
            period_days=candidate.period_days,
            epoch_days=epoch,
            duration_days=candidate.duration_days,
            epoch_correction_days=correction,
            meta=meta,
            history=(
                *light_curve.history,
                f"phase_fold(P={candidate.period_days:.6f},"
                f"t0={epoch:.6f},correction={correction:+.6f})",
            ),
        )

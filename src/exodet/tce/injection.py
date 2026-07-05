"""Synthetic transit injection and recovery framework.

Provides:
    * :func:`inject_box_transit` — multiplicative box-transit injection
      into any light curve (real or synthetic), fully vectorized.
    * :func:`make_noise_light_curve` — white-noise synthetic curves.
    * :class:`InjectionRecoveryExperiment` — runs the complete TCE
      pipeline over a grid of injected signals and measures recovery
      rate, precision, recall, parameter-recovery errors, and detection
      efficiency as a function of depth, duration, and expected SNR.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.tce.pipeline import TCEPipeline
from exodet.utils.io import ensure_dir, write_json

__all__ = [
    "inject_box_transit",
    "make_noise_light_curve",
    "InjectionTrial",
    "InjectionSummary",
    "InjectionRecoveryExperiment",
]

logger = logging.getLogger(__name__)

_MAD_TO_STD = 1.4826


def inject_box_transit(
    light_curve: LightCurve,
    period_days: float,
    duration_days: float,
    depth: float,
    epoch_days: float,
) -> LightCurve:
    """Injects a periodic box-shaped transit into a light curve.

    The injection is multiplicative — ``flux * (1 - depth)`` inside
    transit — which is correct for relative (normalized) flux and a
    good approximation for any smoothly varying baseline. The input is
    never modified; injection parameters are recorded in
    ``meta["injection"]`` and provenance.

    Args:
        light_curve: The base curve (real or synthetic).
        period_days: Orbital period in days; must be positive.
        duration_days: Transit duration in days; in ``(0, period)``.
        depth: Fractional depth; in ``(0, 1)``.
        epoch_days: Mid-transit time of one transit in days.

    Returns:
        A new light curve with the transit signal injected.

    Raises:
        PipelineError: If any parameter is out of range.
    """
    if period_days <= 0:
        raise PipelineError(f"period_days must be > 0, got {period_days}.")
    if not 0 < duration_days < period_days:
        raise PipelineError(
            f"duration_days must be in (0, period), got {duration_days}."
        )
    if not 0 < depth < 1:
        raise PipelineError(f"depth must be in (0, 1), got {depth}.")

    phase = (light_curve.time - epoch_days + 0.5 * period_days) % period_days
    in_transit = np.abs(phase - 0.5 * period_days) < 0.5 * duration_days
    flux = light_curve.flux * np.where(in_transit, 1.0 - depth, 1.0)

    result = light_curve.replace_flux(
        flux,
        step_name=(
            f"inject_box_transit(P={period_days:.5f},d={depth:.5f},"
            f"dur={duration_days:.5f},t0={epoch_days:.5f})"
        ),
    )
    result.meta["injection"] = {
        "period_days": float(period_days),
        "duration_days": float(duration_days),
        "depth": float(depth),
        "epoch_days": float(epoch_days),
        "n_in_transit": int(np.count_nonzero(in_transit)),
    }
    return result


def make_noise_light_curve(
    target_id: str = "SYNTH-NOISE",
    n_points: int = 20_000,
    cadence_days: float = 2.0 / (60.0 * 24.0),
    noise_level: float = 1e-3,
    seed: int = 0,
) -> LightCurve:
    """Builds a pure white-noise synthetic light curve around flux 1.

    Args:
        target_id: Identifier of the synthetic target.
        n_points: Number of cadences.
        cadence_days: Sampling interval in days.
        noise_level: Gaussian noise standard deviation (relative flux).
        seed: RNG seed for exact reproducibility.

    Returns:
        The synthetic light curve (with ``flux_err`` set to the noise
        level).

    Raises:
        PipelineError: If sizes or the noise level are invalid.
    """
    if n_points < 10:
        raise PipelineError(f"n_points must be >= 10, got {n_points}.")
    if noise_level <= 0 or cadence_days <= 0:
        raise PipelineError("noise_level and cadence_days must be > 0.")
    rng = np.random.default_rng(seed)
    time = np.arange(n_points) * cadence_days
    return LightCurve(
        target_id=target_id,
        time=time,
        flux=1.0 + rng.normal(0.0, noise_level, n_points),
        flux_err=np.full(n_points, noise_level),
        mission="tess",
        meta={"synthetic": True, "noise_level": noise_level, "seed": seed},
    )


@dataclass(frozen=True, slots=True)
class InjectionTrial:
    """Outcome of one injection-and-recovery trial.

    Attributes:
        trial_id: Sequential trial number.
        base_target_id: Identifier of the base light curve.
        period_days: Injected period.
        duration_days: Injected duration.
        depth: Injected depth.
        epoch_days: Injected epoch.
        expected_snr: Analytic expected detection SNR of the injection.
        recovered: Whether the signal was recovered by an accepted
            candidate.
        matched_ratio: Period ratio of the matching candidate (1.0 for
            an exact match, 2.0 for the 2P harmonic, ...); NaN if not
            recovered.
        recovered_period_days: Period of the matching candidate.
        recovered_depth: Depth of the matching candidate.
        recovered_duration_days: Duration of the matching candidate.
        period_error_rel: Relative period recovery error (exact matches).
        depth_error_rel: Relative depth recovery error.
        duration_error_rel: Relative duration recovery error.
        n_false_alarms: Accepted candidates not matching the injection.
    """

    trial_id: int
    base_target_id: str
    period_days: float
    duration_days: float
    depth: float
    epoch_days: float
    expected_snr: float
    recovered: bool
    matched_ratio: float = math.nan
    recovered_period_days: float = math.nan
    recovered_depth: float = math.nan
    recovered_duration_days: float = math.nan
    period_error_rel: float = math.nan
    depth_error_rel: float = math.nan
    duration_error_rel: float = math.nan
    n_false_alarms: int = 0


@dataclass(frozen=True, slots=True)
class InjectionSummary:
    """Aggregated statistics of an injection-recovery experiment.

    Attributes:
        n_trials: Total number of trials.
        recovery_rate: Fraction of injections recovered (== recall).
        precision: True detections over all accepted candidates.
        recall: Recovered injections over all injections.
        median_period_error_rel: Median relative period error among
            exact-period recoveries.
        median_depth_error_rel: Median relative depth error.
        median_duration_error_rel: Median relative duration error.
        efficiency_by_depth: Recovery fraction per injected depth.
        efficiency_by_duration: Recovery fraction per injected duration.
        efficiency_by_snr: Recovery fraction per expected-SNR bin.
        trials: Every individual trial record.
    """

    n_trials: int
    recovery_rate: float
    precision: float
    recall: float
    median_period_error_rel: float
    median_depth_error_rel: float
    median_duration_error_rel: float
    efficiency_by_depth: dict[str, float]
    efficiency_by_duration: dict[str, float]
    efficiency_by_snr: dict[str, float]
    trials: list[InjectionTrial] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Converts the summary (including trials) to JSON-native types.

        Returns:
            A dictionary safe for ``json.dump``.
        """
        raw = asdict(self)
        raw["trials"] = [asdict(trial) for trial in self.trials]
        return raw

    def save(self, json_path: Path | str, csv_path: Path | str | None = None) -> Path:
        """Writes the summary as JSON and optionally the trials as CSV.

        Args:
            json_path: Destination for the full summary JSON.
            csv_path: Optional destination for a per-trial CSV table.

        Returns:
            The JSON file path.
        """
        path = write_json(self.to_dict(), Path(json_path))
        if csv_path is not None:
            csv_path = Path(csv_path)
            ensure_dir(csv_path.parent)
            fields = [f.name for f in InjectionTrial.__dataclass_fields__.values()]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for trial in self.trials:
                    writer.writerow(asdict(trial))
        return path


class InjectionRecoveryExperiment:
    """Measures the detection efficiency of a configured TCE pipeline.

    Attributes:
        pipeline: The TCE pipeline under test.
        period_tolerance: Relative period tolerance for a match.
        epoch_tolerance_durations: Epoch tolerance for exact-period
            matches, in units of the injected duration.
        accept_harmonics: Whether recoveries at P/2, 2P, ... count as
            recovered (standard practice; the ratio is recorded).
        snr_bin_edges: Bin edges for the efficiency-vs-SNR curve.
    """

    _HARMONIC_RATIOS = (
        1.0, 0.5, 2.0, 1.0 / 3.0, 3.0, 0.25, 4.0, 0.2, 5.0,
    )

    def __init__(
        self,
        pipeline: TCEPipeline,
        period_tolerance: float = 0.01,
        epoch_tolerance_durations: float = 1.0,
        accept_harmonics: bool = True,
        snr_bin_edges: Sequence[float] = (0.0, 5.0, 10.0, 20.0, 50.0, math.inf),
    ) -> None:
        """Initializes the experiment.

        Args:
            pipeline: Configured TCE pipeline.
            period_tolerance: Relative period-match tolerance; positive.
            epoch_tolerance_durations: Epoch tolerance in durations.
            accept_harmonics: Count harmonic recoveries as recovered.
            snr_bin_edges: Increasing SNR bin edges.

        Raises:
            PipelineError: If tolerances are invalid.
        """
        if period_tolerance <= 0:
            raise PipelineError(
                f"period_tolerance must be > 0, got {period_tolerance}."
            )
        if epoch_tolerance_durations <= 0:
            raise PipelineError(
                "epoch_tolerance_durations must be > 0, got "
                f"{epoch_tolerance_durations}."
            )
        self.pipeline = pipeline
        self.period_tolerance = float(period_tolerance)
        self.epoch_tolerance_durations = float(epoch_tolerance_durations)
        self.accept_harmonics = accept_harmonics
        self.snr_bin_edges = tuple(snr_bin_edges)

    def _expected_snr(self, base: LightCurve, injected: LightCurve) -> float:
        """Analytic expected SNR of an injection.

        ``depth / sigma * sqrt(n_in_transit)`` with a robust (MAD)
        noise estimate from the base curve.

        Args:
            base: The curve before injection.
            injected: The curve after injection (carries the params).

        Returns:
            The expected SNR (NaN when the noise estimate is zero).
        """
        params = injected.meta["injection"]
        flux = base.flux[np.isfinite(base.flux)]
        median = np.median(flux)
        sigma = _MAD_TO_STD * float(np.median(np.abs(flux - median)))
        if sigma == 0.0 or median == 0.0:
            return math.nan
        relative_sigma = sigma / abs(float(median))
        return params["depth"] / relative_sigma * math.sqrt(params["n_in_transit"])

    def _match(
        self, injected_params: dict[str, float], candidate_period: float,
        candidate_epoch: float,
    ) -> float | None:
        """Tests whether a candidate matches the injected signal.

        Args:
            injected_params: The injection parameter record.
            candidate_period: Recovered period in days.
            candidate_epoch: Recovered epoch in days.

        Returns:
            The matched period ratio (candidate/injected), or ``None``.
        """
        p_inj = injected_params["period_days"]
        ratios = self._HARMONIC_RATIOS if self.accept_harmonics else (1.0,)
        for ratio in ratios:
            if abs(candidate_period / p_inj - ratio) <= self.period_tolerance * ratio:
                if ratio == 1.0:
                    # For exact matches, additionally demand epoch
                    # agreement modulo the period.
                    delta = (
                        candidate_epoch
                        - injected_params["epoch_days"]
                        + 0.5 * p_inj
                    ) % p_inj - 0.5 * p_inj
                    tolerance = (
                        self.epoch_tolerance_durations
                        * injected_params["duration_days"]
                    )
                    if abs(delta) > tolerance:
                        continue
                return ratio
        return None

    def run_trial(
        self,
        base: LightCurve,
        trial_id: int,
        period_days: float,
        duration_days: float,
        depth: float,
        epoch_days: float,
    ) -> InjectionTrial:
        """Injects one signal, runs the pipeline, and scores recovery.

        Args:
            base: The base light curve.
            trial_id: Sequential trial number.
            period_days: Injected period.
            duration_days: Injected duration.
            depth: Injected depth.
            epoch_days: Injected epoch.

        Returns:
            The trial record.
        """
        injected = inject_box_transit(
            base, period_days, duration_days, depth, epoch_days
        )
        expected_snr = self._expected_snr(base, injected)
        result = self.pipeline.run(injected)

        match: tuple[Any, float] | None = None
        false_alarms = 0
        for candidate in result.accepted:
            ratio = self._match(
                injected.meta["injection"],
                candidate.period_days,
                candidate.epoch_days,
            )
            if ratio is not None and match is None:
                match = (candidate, ratio)
            else:
                false_alarms += 1

        if match is None:
            return InjectionTrial(
                trial_id=trial_id,
                base_target_id=base.target_id,
                period_days=period_days,
                duration_days=duration_days,
                depth=depth,
                epoch_days=epoch_days,
                expected_snr=expected_snr,
                recovered=False,
                n_false_alarms=false_alarms,
            )

        candidate, ratio = match
        exact = ratio == 1.0
        return InjectionTrial(
            trial_id=trial_id,
            base_target_id=base.target_id,
            period_days=period_days,
            duration_days=duration_days,
            depth=depth,
            epoch_days=epoch_days,
            expected_snr=expected_snr,
            recovered=True,
            matched_ratio=ratio,
            recovered_period_days=candidate.period_days,
            recovered_depth=candidate.depth,
            recovered_duration_days=candidate.duration_days,
            period_error_rel=(
                abs(candidate.period_days - period_days) / period_days
                if exact
                else math.nan
            ),
            depth_error_rel=abs(candidate.depth - depth) / depth,
            duration_error_rel=(
                abs(candidate.duration_days - duration_days) / duration_days
            ),
            n_false_alarms=false_alarms,
        )

    def run(
        self,
        base_curves: Sequence[LightCurve],
        periods: Sequence[float],
        durations: Sequence[float],
        depths: Sequence[float],
        seed: int = 0,
    ) -> InjectionSummary:
        """Runs the full injection grid over the base curves.

        One trial is executed per (base curve, period, duration, depth)
        combination; epochs are drawn uniformly within one period from
        a seeded RNG for reproducibility.

        Args:
            base_curves: Base light curves (real or synthetic).
            periods: Injected periods in days.
            durations: Injected durations in days.
            depths: Injected fractional depths.
            seed: RNG seed for the random epochs.

        Returns:
            The aggregated summary with all trial records.

        Raises:
            PipelineError: If any input sequence is empty.
        """
        if not (len(base_curves) and len(periods) and len(durations) and len(depths)):
            raise PipelineError("All injection grids must be non-empty.")

        rng = np.random.default_rng(seed)
        trials: list[InjectionTrial] = []
        trial_id = 0
        for base in base_curves:
            for period in periods:
                for duration in durations:
                    for depth in depths:
                        epoch = float(base.time[0] + rng.uniform(0.0, period))
                        trials.append(
                            self.run_trial(
                                base, trial_id, period, duration, depth, epoch
                            )
                        )
                        trial_id += 1

        return self._summarize(trials)

    def _summarize(self, trials: list[InjectionTrial]) -> InjectionSummary:
        """Aggregates trial records into the final statistics.

        Args:
            trials: All trial records.

        Returns:
            The aggregated summary.
        """
        n = len(trials)
        recovered = np.array([t.recovered for t in trials])
        n_recovered = int(recovered.sum())
        n_false = int(sum(t.n_false_alarms for t in trials))
        precision = (
            n_recovered / (n_recovered + n_false)
            if (n_recovered + n_false) > 0
            else math.nan
        )
        recall = n_recovered / n if n else math.nan

        def _median_error(attribute: str) -> float:
            values = np.array([getattr(t, attribute) for t in trials])
            values = values[np.isfinite(values)]
            return float(np.median(values)) if values.size else math.nan

        def _efficiency(key: Any) -> dict[str, float]:
            groups: dict[str, list[bool]] = {}
            for trial in trials:
                groups.setdefault(key(trial), []).append(trial.recovered)
            return {
                label: float(np.mean(flags)) for label, flags in sorted(groups.items())
            }

        def _snr_bin(trial: InjectionTrial) -> str:
            edges = self.snr_bin_edges
            index = int(np.searchsorted(edges, trial.expected_snr, side="right")) - 1
            index = min(max(index, 0), len(edges) - 2)
            return f"[{edges[index]:g}, {edges[index + 1]:g})"

        summary = InjectionSummary(
            n_trials=n,
            recovery_rate=recall,
            precision=precision,
            recall=recall,
            median_period_error_rel=_median_error("period_error_rel"),
            median_depth_error_rel=_median_error("depth_error_rel"),
            median_duration_error_rel=_median_error("duration_error_rel"),
            efficiency_by_depth=_efficiency(lambda t: f"{t.depth:g}"),
            efficiency_by_duration=_efficiency(lambda t: f"{t.duration_days:g}"),
            efficiency_by_snr=_efficiency(_snr_bin),
            trials=trials,
        )
        logger.info(
            "Injection recovery: %d/%d recovered (recall %.2f, precision %.2f).",
            n_recovered,
            n,
            recall,
            precision,
        )
        return summary

"""Physics feature extraction (Module 5).

Assembles the named feature vector consumed by the ML classifier
alongside the views. Features come from three sources:

* the :class:`~exodet.tce.candidate.TransitCandidate` record (ephemeris
  and detection statistics computed by the TCE stage);
* the phase-folded curve (photometric statistics, in/out-of-transit
  RMS, residuals against the binned model);
* the local view (transit-shape descriptors: symmetry, ingress and
  egress durations estimated from the depth crossings).

Feature groups are individually selectable; names are always stored
alongside values so downstream scalers and models stay aligned.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

from exodet.exceptions import PipelineError
from exodet.registry import Registry
from exodet.representation.containers import FeatureVector, PhaseFoldedCurve, View
from exodet.tce.candidate import TransitCandidate

__all__ = ["PHYSICS_EXTRACTORS", "PhysicsFeatureExtractor"]

logger = logging.getLogger(__name__)

PHYSICS_EXTRACTORS: Registry["PhysicsFeatureExtractor"] = Registry(
    "physics feature extractor"
)

_MAD_TO_STD = 1.4826

_GROUPS = ("ephemeris", "detection", "shape", "photometry", "quality")


@PHYSICS_EXTRACTORS.register("standard")
class PhysicsFeatureExtractor:
    """Computes the named physics feature vector for one candidate.

    Attributes:
        groups: Enabled feature groups (subset of ``ephemeris``,
            ``detection``, ``shape``, ``photometry``, ``quality``).
        percentiles: Flux percentiles included in the photometry group.
        log_snr_floor: Floor applied inside log10 transforms of
            strictly positive detection statistics.
    """

    def __init__(
        self,
        groups: tuple[str, ...] | list[str] = _GROUPS,
        percentiles: tuple[float, ...] | list[float] = (5.0, 25.0, 50.0, 75.0, 95.0),
        log_snr_floor: float = 1e-3,
    ) -> None:
        """Initializes the extractor.

        Args:
            groups: Feature groups to compute.
            percentiles: Percentile levels for the photometry group.
            log_snr_floor: Positive floor for log-transformed stats.

        Raises:
            PipelineError: If a group is unknown or parameters invalid.
        """
        unknown = set(groups) - set(_GROUPS)
        if unknown:
            raise PipelineError(
                f"Unknown feature groups {sorted(unknown)}. Available: {_GROUPS}."
            )
        if not groups:
            raise PipelineError("At least one feature group must be enabled.")
        if any(not 0 <= p <= 100 for p in percentiles):
            raise PipelineError(f"Percentiles must lie in [0, 100], got {percentiles}.")
        if log_snr_floor <= 0:
            raise PipelineError(f"log_snr_floor must be > 0, got {log_snr_floor}.")
        self.groups = tuple(groups)
        self.percentiles = tuple(float(p) for p in percentiles)
        self.log_snr_floor = float(log_snr_floor)

    def _ephemeris(
        self, candidate: TransitCandidate
    ) -> dict[str, float]:
        return {
            "period_days": candidate.period_days,
            "epoch_days": candidate.epoch_days,
            "duration_days": candidate.duration_days,
            "depth": candidate.depth,
            "n_transits": float(candidate.n_transits),
            "duty_cycle": candidate.duration_days / candidate.period_days,
        }

    def _detection(self, candidate: TransitCandidate) -> dict[str, float]:
        floor = self.log_snr_floor
        fap = candidate.fap
        # -log10(FAP) is bounded and monotone in significance; FAP == 0
        # (numerically) maps to a large constant instead of infinity.
        if math.isfinite(fap):
            neg_log_fap = -math.log10(max(fap, 1e-300))
        else:
            neg_log_fap = math.nan
        return {
            "sde": candidate.sde,
            "snr": candidate.snr,
            "power": candidate.power,
            "neg_log_fap": neg_log_fap,
            "log_snr": math.log10(max(candidate.snr, floor))
            if math.isfinite(candidate.snr)
            else math.nan,
        }

    def _shape(
        self, candidate: TransitCandidate, local_view: View
    ) -> dict[str, float]:
        values = local_view.values
        centers = local_view.bin_centers
        n = len(values)
        center_index = n // 2

        # Odd/even depth ratio from the TCE stage diagnostics.
        depth_odd = candidate.meta.get("depth_odd", math.nan)
        depth_even = candidate.meta.get("depth_even", math.nan)
        if (
            isinstance(depth_even, (int, float))
            and isinstance(depth_odd, (int, float))
            and math.isfinite(depth_even)
            and depth_even != 0
        ):
            odd_even_ratio = float(depth_odd) / float(depth_even)
        else:
            odd_even_ratio = math.nan

        # Symmetry: RMS difference between the mirrored transit halves,
        # normalized by the view depth. 0 = perfectly symmetric.
        left = values[:center_index]
        right = values[center_index + 1 :][::-1]
        m = min(len(left), len(right))
        depth_scale = float(np.median(values) - values.min())
        if m > 0 and depth_scale > 0:
            symmetry = float(
                np.sqrt(np.mean((left[-m:] - right[-m:]) ** 2)) / depth_scale
            )
        else:
            symmetry = math.nan

        # Ingress/egress duration from 10%/90% depth crossings of the
        # binned profile, converted from phase to days.
        baseline = float(np.median(values))
        minimum = float(values.min())
        depth_view = baseline - minimum
        ingress = egress = math.nan
        if depth_view > 0:
            level_hi = baseline - 0.1 * depth_view
            level_lo = baseline - 0.9 * depth_view
            below_hi = values <= level_hi
            below_lo = values <= level_lo
            if below_lo.any() and below_hi.any():
                first_hi = int(np.argmax(below_hi))
                first_lo = int(np.argmax(below_lo))
                last_hi = n - 1 - int(np.argmax(below_hi[::-1]))
                last_lo = n - 1 - int(np.argmax(below_lo[::-1]))
                bin_width_phase = float(centers[1] - centers[0])
                to_days = bin_width_phase * candidate.period_days
                ingress = max(first_lo - first_hi, 0) * to_days
                egress = max(last_hi - last_lo, 0) * to_days

        return {
            "odd_even_depth_ratio": odd_even_ratio,
            "transit_symmetry": symmetry,
            "ingress_duration_days": ingress,
            "egress_duration_days": egress,
        }

    def _photometry(
        self,
        folded: PhaseFoldedCurve,
        global_view: View,
        local_view: View,
    ) -> dict[str, float]:
        flux = folded.flux
        phase = folded.phase
        duty = folded.duty_cycle

        in_transit = np.abs(phase) < 0.5 * duty
        out_transit = ~in_transit
        global_rms = float(np.sqrt(np.mean((flux - np.median(flux)) ** 2)))
        if out_transit.any():
            oot = flux[out_transit]
            local_window = (np.abs(phase) < 2.0 * duty) & out_transit
            local_flux = flux[local_window] if local_window.any() else oot
            local_rms = float(
                np.sqrt(np.mean((local_flux - np.median(local_flux)) ** 2))
            )
        else:
            local_rms = math.nan

        # Residual RMS: folded flux minus the binned global-view model
        # evaluated at each cadence (both in normalized-view units when
        # normalization is active, so we residualize in flux space by
        # re-binning without normalization statistics applied).
        model = np.interp(phase, global_view.bin_centers, global_view.values)
        norm_stats = global_view.meta.get("normalization_stats", {})
        scale = float(norm_stats.get("scale", 1.0))
        median = float(norm_stats.get("median", 0.0))
        model_flux = model * scale + median
        residual_rms = float(np.sqrt(np.mean((flux - model_flux) ** 2)))

        mad = _MAD_TO_STD * float(np.median(np.abs(flux - np.median(flux))))
        features: dict[str, float] = {
            "global_rms": global_rms,
            "local_rms": local_rms,
            "residual_rms": residual_rms,
            "flux_skewness": float(scipy_stats.skew(flux)),
            "flux_kurtosis": float(scipy_stats.kurtosis(flux)),
            "flux_mad": mad,
        }
        levels = np.percentile(flux, self.percentiles)
        for level, value in zip(self.percentiles, levels):
            features[f"flux_p{level:g}"] = float(value)
        return features

    def _quality(
        self,
        candidate: TransitCandidate,
        global_view: View,
        local_view: View,
    ) -> dict[str, float]:
        return {
            "missing_fraction_global": global_view.empty_fraction,
            "missing_fraction_local": local_view.empty_fraction,
            "n_interpolated_bins": float(
                global_view.n_empty_bins + local_view.n_empty_bins
            ),
            "n_quality_flags": float(len(candidate.quality_flags)),
            "flag_odd_even_mismatch": float(
                "odd_even_mismatch" in candidate.quality_flags
            ),
            "flag_sinusoidal": float(
                "sinusoidal_preferred" in candidate.quality_flags
            ),
            "flag_partial_coverage": float(
                "partial_transit_coverage" in candidate.quality_flags
            ),
        }

    def extract(
        self,
        candidate: TransitCandidate,
        folded: PhaseFoldedCurve,
        global_view: View,
        local_view: View,
    ) -> FeatureVector:
        """Computes the full named feature vector.

        Args:
            candidate: The transit candidate record.
            folded: The phase-folded curve.
            global_view: The global orbital view.
            local_view: The local transit view.

        Returns:
            The feature vector with aligned names and values.
        """
        features: dict[str, float] = {}
        if "ephemeris" in self.groups:
            features.update(self._ephemeris(candidate))
        if "detection" in self.groups:
            features.update(self._detection(candidate))
        if "shape" in self.groups:
            features.update(self._shape(candidate, local_view))
        if "photometry" in self.groups:
            features.update(self._photometry(folded, global_view, local_view))
        if "quality" in self.groups:
            features.update(self._quality(candidate, global_view, local_view))

        names = tuple(features)
        values: npt.NDArray[np.float64] = np.array(
            [features[name] for name in names], dtype=np.float64
        )
        logger.debug(
            "Candidate %s: extracted %d physics features.",
            candidate.candidate_id,
            len(names),
        )
        return FeatureVector(names=names, values=values)

"""False positive diagnostic analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from exodet.representation.containers import DatasetSample

__all__ = ["FalsePositiveAssessment", "FalsePositiveAnalyzer"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FalsePositiveAssessment:
    """False-positive flags with confidence scores in ``[0, 1]``."""

    eclipsing_binary_score: float
    secondary_eclipse_score: float
    odd_even_mismatch_score: float
    v_shaped_score: float
    sinusoidal_variability_score: float
    stellar_activity_score: float
    blend_signature_score: float
    overall_fp_risk: float
    triggered_flags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eclipsing_binary_score": self.eclipsing_binary_score,
            "secondary_eclipse_score": self.secondary_eclipse_score,
            "odd_even_mismatch_score": self.odd_even_mismatch_score,
            "v_shaped_score": self.v_shaped_score,
            "sinusoidal_variability_score": self.sinusoidal_variability_score,
            "stellar_activity_score": self.stellar_activity_score,
            "blend_signature_score": self.blend_signature_score,
            "overall_fp_risk": self.overall_fp_risk,
            "triggered_flags": list(self.triggered_flags),
            "meta": dict(self.meta),
        }


class FalsePositiveAnalyzer:
    """Rule-based false positive diagnostics on folded views."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.secondary_threshold = float(cfg.get("secondary_threshold", 0.002))
        self.v_shape_slope_threshold = float(cfg.get("v_shape_slope_threshold", 0.15))
        self.variability_threshold = float(cfg.get("variability_threshold", 0.01))

    def analyze(self, sample: DatasetSample) -> FalsePositiveAssessment:
        """Scores common astrophysical false-positive morphologies."""
        global_view = sample.global_view.astype(np.float64)
        local_view = sample.local_view.astype(np.float64)
        candidate = sample.candidate
        flags: list[str] = []

        secondary = self._secondary_eclipse_score(global_view)
        if secondary > 0.5:
            flags.append("secondary_eclipse")

        odd_even = self._odd_even_score(candidate)
        if odd_even > 0.5:
            flags.append("odd_even_mismatch")

        v_shape = self._v_shaped_score(local_view)
        if v_shape > 0.5:
            flags.append("v_shaped_transit")

        sinusoidal = self._sinusoidal_score(global_view)
        if sinusoidal > 0.5:
            flags.append("sinusoidal_variability")

        activity = self._stellar_activity_score(sample)
        if activity > 0.5:
            flags.append("stellar_activity")

        blend = self._blend_score(candidate)
        if blend > 0.5:
            flags.append("blend_signature")

        eb_score = max(secondary, odd_even, v_shape)
        if eb_score > 0.6:
            flags.append("eclipsing_binary")

        scores = [secondary, odd_even, v_shape, sinusoidal, activity, blend, eb_score]
        overall = float(np.clip(np.mean(scores), 0.0, 1.0))

        return FalsePositiveAssessment(
            eclipsing_binary_score=float(eb_score),
            secondary_eclipse_score=float(secondary),
            odd_even_mismatch_score=float(odd_even),
            v_shaped_score=float(v_shape),
            sinusoidal_variability_score=float(sinusoidal),
            stellar_activity_score=float(activity),
            blend_signature_score=float(blend),
            overall_fp_risk=overall,
            triggered_flags=tuple(flags),
            meta={"candidate_flags": list(candidate.quality_flags)},
        )

    def _secondary_eclipse_score(self, global_view: np.ndarray) -> float:
        n = len(global_view)
        mid = n // 2
        anti = np.concatenate([global_view[mid:], global_view[:mid]])
        baseline = np.median(global_view)
        peak = float(np.max(anti) - baseline)
        return float(np.clip(peak / max(self.secondary_threshold, 1e-6), 0.0, 1.0))

    def _odd_even_score(self, candidate: object) -> float:
        meta = getattr(candidate, "meta", {})
        odd = meta.get("odd_depth")
        even = meta.get("even_depth")
        if odd is None or even is None:
            return 0.0
        ratio = abs(float(odd) - float(even)) / max(float(odd) + float(even), 1e-6)
        return float(np.clip(ratio * 2.0, 0.0, 1.0))

    def _v_shaped_score(self, local_view: np.ndarray) -> float:
        center = len(local_view) // 2
        wing = max(3, len(local_view) // 8)
        left = local_view[center - wing : center]
        right = local_view[center : center + wing]
        if len(left) < 2 or len(right) < 2:
            return 0.0
        slope_l = float(np.polyfit(np.arange(len(left)), left, 1)[0])
        slope_r = float(np.polyfit(np.arange(len(right)), right, 1)[0])
        asym = abs(slope_l + slope_r) / max(abs(slope_l) + abs(slope_r), 1e-6)
        return float(np.clip(asym / self.v_shape_slope_threshold, 0.0, 1.0))

    def _sinusoidal_score(self, global_view: np.ndarray) -> float:
        centered = global_view - np.median(global_view)
        fft = np.fft.rfft(centered)
        power = np.abs(fft[1:]) ** 2
        if len(power) == 0:
            return 0.0
        peak = float(power.max())
        total = float(power.sum()) + 1e-12
        return float(np.clip(peak / total / self.variability_threshold, 0.0, 1.0))

    def _stellar_activity_score(self, sample: DatasetSample) -> float:
        names = sample.feature_names
        if "global_rms" in names:
            rms = float(sample.features[names.index("global_rms")])
            return float(np.clip(rms / self.variability_threshold, 0.0, 1.0))
        return float(np.clip(np.std(sample.global_view) / self.variability_threshold, 0.0, 1.0))

    def _blend_score(self, candidate: object) -> float:
        flags = getattr(candidate, "quality_flags", ())
        score = 0.0
        for flag in flags:
            if "blend" in flag.lower() or "contamination" in flag.lower():
                score = max(score, 0.8)
        return score

"""Scientific inference result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from exodet.inference.explainability import ExplainabilityResult
from exodet.inference.false_positive import FalsePositiveAssessment
from exodet.inference.parameter_fit import RefinedTransitParameters
from exodet.inference.physical import PhysicalParameters
from exodet.inference.uncertainty import UncertaintyEstimate

__all__ = ["ScientificInferenceResult", "ScientificInferenceBatch"]


@dataclass(frozen=True, slots=True)
class ScientificInferenceResult:
    """Full scientific inference output for one candidate."""

    sample_id: str
    target_id: str
    candidate_id: str
    probability: float
    classification: str
    confidence: float
    uncertainty: UncertaintyEstimate | None = None
    transit: RefinedTransitParameters | None = None
    physical: PhysicalParameters | None = None
    false_positive: FalsePositiveAssessment | None = None
    explainability: ExplainabilityResult | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "target_id": self.target_id,
            "candidate_id": self.candidate_id,
            "probability": self.probability,
            "classification": self.classification,
            "confidence": self.confidence,
            "uncertainty": None if self.uncertainty is None else self.uncertainty.to_dict(),
            "transit": None if self.transit is None else self.transit.to_dict(),
            "physical": None if self.physical is None else self.physical.to_dict(),
            "false_positive": (
                None if self.false_positive is None else self.false_positive.to_dict()
            ),
            "explainability": (
                None if self.explainability is None else self.explainability.to_dict()
            ),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class ScientificInferenceBatch:
    """Collection of scientific inference results."""

    results: tuple[ScientificInferenceResult, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_results": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "meta": dict(self.meta),
        }

    def to_records(self) -> list[dict[str, Any]]:
        """Flattens results for catalog / CSV export."""
        records: list[dict[str, Any]] = []
        for r in self.results:
            row: dict[str, Any] = {
                "sample_id": r.sample_id,
                "target_id": r.target_id,
                "candidate_id": r.candidate_id,
                "tic_id": r.target_id.replace("TIC ", "").strip(),
                "classification": r.classification,
                "confidence": r.confidence,
                "probability": r.probability,
            }
            if r.uncertainty is not None:
                row["probability_std"] = r.uncertainty.std
                row["probability_lower"] = r.uncertainty.lower
                row["probability_upper"] = r.uncertainty.upper
            if r.transit is not None:
                row.update(
                    {
                        "depth": r.transit.depth,
                        "duration_days": r.transit.duration_days,
                        "period_days": r.transit.period_days,
                        "epoch_days": r.transit.epoch_days,
                        "rp_rs": r.transit.rp_rs,
                        "impact_parameter": r.transit.impact_parameter,
                        "n_observed_transits": r.transit.n_observed_transits,
                    }
                )
            if r.physical is not None:
                row["planet_radius_rearth"] = r.physical.planet_radius_rearth
                row["semi_major_axis_au"] = r.physical.semi_major_axis_au
                row["equilibrium_temperature_k"] = r.physical.equilibrium_temperature_k
            if r.false_positive is not None:
                row["fp_risk"] = r.false_positive.overall_fp_risk
            if r.explainability is not None:
                for key, path in r.explainability.to_dict().items():
                    if path and key.endswith("_path"):
                        row[key] = path
            records.append(row)
        return records

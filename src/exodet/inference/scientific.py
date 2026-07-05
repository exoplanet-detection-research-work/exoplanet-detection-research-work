"""Scientific metadata, units, and reproducibility helpers for inference."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

from exodet import __version__
from exodet.config.schema import ExperimentConfig
from exodet.constants import DEFAULT_RANDOM_SEED

__all__ = [
    "UNITS",
    "PHYSICAL_ASSUMPTIONS",
    "PARAMETER_RATIONALE",
    "build_reproduction_metadata",
]

# Canonical units for exported scientific quantities.
UNITS: dict[str, str] = {
    "depth": "dimensionless relative flux decrement",
    "duration_days": "days",
    "epoch_days": "days (mission time system of light curve)",
    "period_days": "days",
    "rp_rs": "dimensionless radius ratio R_p/R_*",
    "impact_parameter": "dimensionless projected separation in stellar radii",
    "ingress_duration_days": "days",
    "egress_duration_days": "days",
    "planet_radius_rearth": "Earth radii (R_Earth)",
    "semi_major_axis_au": "astronomical units (au)",
    "equilibrium_temperature_k": "kelvin (K)",
    "incident_flux_searth": "Earth insolation units (S_Earth)",
    "inclination_deg": "degrees",
    "probability": "dimensionless [0, 1]",
    "confidence": "dimensionless [0, 1]",
    "phase_bin": "orbital phase bin index (phase span [-0.5, 0.5))",
    "flux": "normalized relative flux (detrended, unitless)",
}

PHYSICAL_ASSUMPTIONS: tuple[str, ...] = (
    "Transit depth maps to R_p/R_* via delta ~ (R_p/R_*)^2 for opaque disks.",
    "Kepler's third law uses Newtonian gravity with nominal solar mass units.",
    "Equilibrium temperature uses gray-atmosphere re-radiation with bond albedo A=0.3.",
    "Incident flux normalized to Earth's bolometric insolation at 1 au.",
    "Impact parameter and inclination relation assumes spherical star and circular orbit.",
    "Trapezoidal transit model approximates ingress/egress with linear ramps.",
    "BLS period is held fixed during local transit shape refinement.",
)

# Scientific rationale and valid ranges for YAML-configurable inference parameters.
PARAMETER_RATIONALE: dict[str, dict[str, Any]] = {
    "parameter_fit.method": {
        "rationale": "Least squares for Gaussian residuals; robust for outliers.",
        "allowed": ("least_squares", "robust"),
        "default": "least_squares",
    },
    "parameter_fit.loss": {
        "rationale": "Robust loss down-weights shallow outliers in phase bins.",
        "allowed": ("linear", "soft_l1", "huber", "cauchy"),
        "default": "soft_l1",
    },
    "parameter_fit.bootstrap_samples": {
        "rationale": "Non-parametric depth uncertainty from residual resampling.",
        "range": [0, 500],
        "default": 0,
    },
    "parameter_fit.seed": {
        "rationale": "Controls bootstrap resampling for reproducibility.",
        "default": DEFAULT_RANDOM_SEED,
    },
    "uncertainty.method": {
        "rationale": "MC dropout estimates epistemic spread; bootstrap for sampling noise.",
        "allowed": ("none", "mc_dropout", "bootstrap"),
        "default": "none",
    },
    "uncertainty.n_samples": {
        "rationale": "Monte Carlo / bootstrap draws; larger is stabler but slower.",
        "range": [1, 500],
        "default": 30,
    },
    "uncertainty.credible_alpha": {
        "rationale": "Central credible interval level (e.g. 0.68 ~ 1 sigma).",
        "range": [0.5, 0.99],
        "default": 0.68,
    },
    "physical.default_albedo": {
        "rationale": "Bond albedo for equilibrium temperature (gas giants ~0.3).",
        "range": [0.0, 1.0],
        "default": 0.3,
    },
}


def build_reproduction_metadata(
    experiment: ExperimentConfig,
    stage_settings: Mapping[str, Any] | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds metadata block required to reproduce a scientific export."""
    meta: dict[str, Any] = {
        "package": "exodet",
        "package_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment.experiment_name,
        "random_seed": experiment.seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "units": dict(UNITS),
        "physical_assumptions": list(PHYSICAL_ASSUMPTIONS),
        "parameter_rationale": PARAMETER_RATIONALE,
    }
    if stage_settings is not None:
        meta["stage_settings"] = dict(stage_settings)
    if extra:
        meta.update(dict(extra))
    return meta

"""Physical parameter estimation from transit and stellar metadata.

Units
-----
All outputs use explicit field suffixes (``_days``, ``_au``, ``_k``, ``_rearth``).
Internal calculations use SI constants from :mod:`exodet.constants`.

Assumptions
-----------
* Depth maps to :math:`R_p/R_* \\approx \\sqrt{\\delta}` for small opaque transits.
* Kepler's third law with nominal solar mass.
* Gray equilibrium temperature with configurable Bond albedo (default 0.3).
* Incident flux normalized to Earth's bolometric flux at 1 au.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from exodet.constants import (
    ASTRONOMICAL_UNIT_M,
    EARTH_RADIUS_M,
    GRAVITATIONAL_CONSTANT,
    SECONDS_PER_DAY,
    SOLAR_MASS_KG,
    SOLAR_RADIUS_M,
    STEFAN_BOLTZMANN,
)
from exodet.inference.parameter_fit import RefinedTransitParameters
from exodet.inference.scientific import PHYSICAL_ASSUMPTIONS, UNITS
from exodet.representation.containers import DatasetSample

__all__ = ["PhysicalParameters", "estimate_physical_parameters"]

logger = logging.getLogger(__name__)

# Solar reference values for flux normalization (SI).
_SOLAR_TEFF_K = 5772.0
_DEFAULT_ALBEDO = 0.3
_COV_REGULARIZATION = 1e-10


@dataclass(frozen=True, slots=True)
class PhysicalParameters:
    """Derived planetary and orbital quantities with documented units."""

    planet_radius_rearth: float | None
    semi_major_axis_au: float | None
    equilibrium_temperature_k: float | None
    incident_flux_searth: float | None
    inclination_deg: float | None
    rp_rs: float | None
    missing_fields: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = PHYSICAL_ASSUMPTIONS
    units: dict[str, str] = field(default_factory=lambda: dict(UNITS))
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planet_radius_rearth": self.planet_radius_rearth,
            "semi_major_axis_au": self.semi_major_axis_au,
            "equilibrium_temperature_k": self.equilibrium_temperature_k,
            "incident_flux_searth": self.incident_flux_searth,
            "inclination_deg": self.inclination_deg,
            "rp_rs": self.rp_rs,
            "missing_fields": list(self.missing_fields),
            "assumptions": list(self.assumptions),
            "units": dict(self.units),
            "meta": dict(self.meta),
        }


def _stellar_value(sample: DatasetSample, keys: tuple[str, ...]) -> float | None:
    for container in (sample.meta, sample.candidate.meta):
        for key in keys:
            if key in container:
                try:
                    value = float(container[key])
                    if math.isfinite(value):
                        return value
                except (TypeError, ValueError):
                    continue
    return None


def estimate_physical_parameters(
    sample: DatasetSample,
    transit: RefinedTransitParameters,
    config: dict[str, Any] | None = None,
) -> PhysicalParameters:
    """Estimates physical parameters, tolerating missing stellar metadata."""
    cfg = config or {}
    albedo = float(cfg.get("default_albedo", _DEFAULT_ALBEDO))
    if not 0.0 <= albedo < 1.0:
        albedo = _DEFAULT_ALBEDO

    missing: list[str] = []
    rp_rs = transit.rp_rs

    radius_rsun = _stellar_value(sample, ("radius_rsun", "stellar_radius", "R_star"))
    mass_msun = _stellar_value(sample, ("mass_msun", "stellar_mass", "M_star"))
    teff = _stellar_value(sample, ("teff", "stellar_teff", "T_eff"))

    planet_radius_rearth: float | None = None
    if rp_rs is not None and radius_rsun is not None:
        r_planet_m = rp_rs * radius_rsun * SOLAR_RADIUS_M
        planet_radius_rearth = r_planet_m / EARTH_RADIUS_M
    else:
        missing.append("planet_radius_rearth")

    semi_major_axis_au: float | None = None
    if mass_msun is not None and transit.period_days > 0:
        period_s = transit.period_days * SECONDS_PER_DAY
        m_star_kg = mass_msun * SOLAR_MASS_KG
        # Kepler's third law: a = (G M P^2 / 4 pi^2)^(1/3)
        a_m = (GRAVITATIONAL_CONSTANT * m_star_kg * period_s**2 / (4.0 * math.pi**2)) ** (
            1.0 / 3.0
        )
        semi_major_axis_au = a_m / ASTRONOMICAL_UNIT_M
    else:
        missing.append("semi_major_axis_au")

    incident_flux_searth: float | None = None
    equilibrium_temperature_k: float | None = None
    if teff is not None and radius_rsun is not None and semi_major_axis_au is not None:
        a_m = semi_major_axis_au * ASTRONOMICAL_UNIT_M
        r_star_m = radius_rsun * SOLAR_RADIUS_M
        # Bolometric flux at orbital distance: F = sigma T^4 (R/a)^2
        flux_star = STEFAN_BOLTZMANN * (teff**4) * (r_star_m**2) / max(a_m**2, _COV_REGULARIZATION)
        flux_earth = STEFAN_BOLTZMANN * (_SOLAR_TEFF_K**4) * (SOLAR_RADIUS_M**2) / (
            ASTRONOMICAL_UNIT_M**2
        )
        incident_flux_searth = flux_star / max(flux_earth, _COV_REGULARIZATION)
        equilibrium_temperature_k = teff * math.sqrt(
            r_star_m / max(2.0 * a_m, _COV_REGULARIZATION)
        ) * (1.0 - albedo) ** 0.25
    else:
        if teff is None or radius_rsun is None:
            missing.append("incident_flux_searth")
            missing.append("equilibrium_temperature_k")

    inclination_deg: float | None = None
    if radius_rsun is not None and semi_major_axis_au is not None:
        a_m = semi_major_axis_au * ASTRONOMICAL_UNIT_M
        r_star_m = radius_rsun * SOLAR_RADIUS_M
        # b = (a cos i) / R_*  =>  cos i = sqrt(1 - (b R_* / a)^2) for b in stellar radii
        b_projected = transit.impact_parameter * r_star_m / max(a_m, _COV_REGULARIZATION)
        cos_i_sq = max(1.0 - b_projected**2, 0.0)
        inclination_deg = math.degrees(math.acos(min(1.0, math.sqrt(cos_i_sq))))
    else:
        missing.append("inclination_deg")

    return PhysicalParameters(
        planet_radius_rearth=planet_radius_rearth,
        semi_major_axis_au=semi_major_axis_au,
        equilibrium_temperature_k=equilibrium_temperature_k,
        incident_flux_searth=incident_flux_searth,
        inclination_deg=inclination_deg,
        rp_rs=rp_rs,
        missing_fields=tuple(sorted(set(missing))),
        meta={"bond_albedo": albedo},
    )

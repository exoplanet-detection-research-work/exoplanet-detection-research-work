"""Physical, astronomical, and project-wide constants.

All constants are module-level, immutable values. Mission-specific
parameters are grouped into frozen dataclasses so that future missions
can be added without touching consuming code.

Values follow CODATA 2018 / IAU nominal definitions where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Physical constants (SI units)
# ---------------------------------------------------------------------------

GRAVITATIONAL_CONSTANT: Final[float] = 6.67430e-11
"""Newtonian constant of gravitation ``G`` in m^3 kg^-1 s^-2."""

SPEED_OF_LIGHT: Final[float] = 2.99792458e8
"""Speed of light in vacuum ``c`` in m s^-1."""

STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8
"""Stefan-Boltzmann constant in W m^-2 K^-4."""

# ---------------------------------------------------------------------------
# Astronomical constants (IAU nominal values, SI units)
# ---------------------------------------------------------------------------

SOLAR_MASS_KG: Final[float] = 1.98892e30
"""Nominal solar mass in kg."""

SOLAR_RADIUS_M: Final[float] = 6.957e8
"""Nominal solar radius in m."""

EARTH_RADIUS_M: Final[float] = 6.3781e6
"""Nominal Earth equatorial radius in m."""

JUPITER_RADIUS_M: Final[float] = 7.1492e7
"""Nominal Jupiter equatorial radius in m."""

ASTRONOMICAL_UNIT_M: Final[float] = 1.495978707e11
"""Astronomical unit in m."""

SECONDS_PER_DAY: Final[float] = 86400.0
"""Number of SI seconds in one day."""

# ---------------------------------------------------------------------------
# Mission parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MissionParameters:
    """Observing parameters for a photometric survey mission.

    Attributes:
        name: Canonical mission name.
        cadence_seconds: Nominal long-cadence sampling interval in seconds.
        time_reference: Description of the mission time system offset
            (e.g. ``BJD - 2454833.0`` for Kepler's BKJD).
        time_offset_days: Offset in days to add to mission time to
            recover BJD.
    """

    name: str
    cadence_seconds: float
    time_reference: str
    time_offset_days: float


KEPLER: Final[MissionParameters] = MissionParameters(
    name="Kepler",
    cadence_seconds=1765.5,
    time_reference="BKJD (BJD - 2454833.0)",
    time_offset_days=2454833.0,
)

K2: Final[MissionParameters] = MissionParameters(
    name="K2",
    cadence_seconds=1765.5,
    time_reference="BKJD (BJD - 2454833.0)",
    time_offset_days=2454833.0,
)

TESS: Final[MissionParameters] = MissionParameters(
    name="TESS",
    cadence_seconds=120.0,
    time_reference="BTJD (BJD - 2457000.0)",
    time_offset_days=2457000.0,
)

MISSIONS: Final[dict[str, MissionParameters]] = {
    m.name.lower(): m for m in (KEPLER, K2, TESS)
}

# ---------------------------------------------------------------------------
# Project-wide labels and defaults
# ---------------------------------------------------------------------------

LABEL_PLANET: Final[int] = 1
"""Integer class label for confirmed/candidate planet signals."""

LABEL_NON_PLANET: Final[int] = 0
"""Integer class label for false positives / non-planet signals."""

CLASS_NAMES: Final[dict[int, str]] = {
    LABEL_NON_PLANET: "non-planet",
    LABEL_PLANET: "planet",
}

DEFAULT_RANDOM_SEED: Final[int] = 42
"""Default seed used when the configuration does not specify one."""

PACKAGE_NAME: Final[str] = "exodet"
"""Canonical package name, used for logger namespacing."""

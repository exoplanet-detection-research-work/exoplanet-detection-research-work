"""Explicit floating-point tolerances for scientific tests."""

from __future__ import annotations

# Relative flux / depth comparisons on normalized light curves.
DEPTH_RTOL: float = 1e-3
DEPTH_ATOL: float = 1e-5

# Ephemeris quantities (days).
PERIOD_RTOL: float = 1e-9
PERIOD_ATOL: float = 0.0
DURATION_RTOL: float = 1e-2

# Probabilities and scores in [0, 1].
PROBABILITY_RTOL: float = 1e-4
PROBABILITY_ATOL: float = 1e-6

# Physical parameters (order-of-magnitude aware).
RADIUS_RTOL: float = 1e-2
SEMI_MAJOR_AXIS_RTOL: float = 1e-2

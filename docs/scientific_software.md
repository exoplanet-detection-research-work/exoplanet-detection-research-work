# Scientific software requirements

This document records how **exodet** satisfies publication-grade scientific software standards.

## Numerical stability

- Optimization uses `scipy.optimize.least_squares` with explicit `ftol`/`xtol` and bounded parameters.
- Covariance estimation uses **regularized solves** (`JᵀJ + εI`) rather than raw matrix inversion.
- Physical formulas use SI constants from `exodet.constants` with division guards (`max(denominator, ε)`).
- MC dropout and bootstrap algorithms honor configurable seeds for reproducibility.

## Units

Canonical units are defined in `exodet.inference.scientific.UNITS` and embedded in exported JSON (`units` field on transit and physical results).

| Quantity | Unit |
|----------|------|
| `depth` | dimensionless relative flux decrement |
| `*_days` | days (mission time system) |
| `planet_radius_rearth` | Earth radii |
| `semi_major_axis_au` | astronomical units |
| `equilibrium_temperature_k` | kelvin |
| `incident_flux_searth` | Earth insolation units |
| `inclination_deg` | degrees |
| `flux` | normalized relative flux |

Project-wide SI/IAU constants live in `exodet.constants`.

## Physical assumptions

Documented in code and exports via `PHYSICAL_ASSUMPTIONS` (`inference/scientific.py`):

1. Small-opaque-disk depth–radius relation: \(\delta \approx (R_p/R_\*)^2\)
2. Kepler's third law with nominal solar mass
3. Gray equilibrium temperature with Bond albedo (default \(A=0.3\))
4. Flux normalized to Earth's bolometric insolation at 1 au
5. Spherical star, circular orbit for inclination–impact parameter mapping
6. Trapezoidal transit model for ingress/egress
7. BLS period held fixed during local shape refinement

## Configurable parameters

Scientific rationale and valid ranges: `PARAMETER_RATIONALE` in `inference/scientific.py`. YAML examples with inline comments: `configs/inference.yaml`.

## Reproducibility

- Global seed: `experiment.seed` (default 42 in `exodet.constants.DEFAULT_RANDOM_SEED`)
- Stage seeds: `parameter_fit.seed`, `uncertainty.seed`
- `build_reproduction_metadata()` attaches version, platform, timestamp, settings, units, and assumptions to every inference export.

## Test tolerances

Explicit tolerances in `tests/scientific_tolerances.py`:

| Constant | Value | Use |
|----------|-------|-----|
| `DEPTH_RTOL` | 1e-3 | transit depth |
| `PERIOD_RTOL` | 1e-9 | orbital period |
| `PROBABILITY_RTOL` | 1e-4 | classifier scores |

## Figures

All inference and reporting figures include:

- Descriptive titles
- Axis labels with units where applicable (e.g. "Relative flux (normalized)", "Orbital phase bin")
- Probability axes labeled `[0, 1]`

## Export metadata

Every scientific JSON export includes a `meta` block with:

- `package_version`, `timestamp_utc`, `random_seed`
- `stage_settings`, `units`, `physical_assumptions`, `parameter_rationale`
- Sufficient context to reproduce the computation

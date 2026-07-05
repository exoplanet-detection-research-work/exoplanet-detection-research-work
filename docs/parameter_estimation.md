# Parameter estimation

## Transit refinement

BLS provides initial ephemerides. `TransitParameterRefiner` fits a trapezoidal model to the local phase-folded view:

\[
F(\phi) = 1 - \delta \cdot \Pi\bigl(|\phi - \phi_0|;\, w, t_\mathrm{ingress}\bigr)
\]

where \(\delta\) is depth, \(w\) is half the transit width in phase, and \(\Pi\) is a trapezoidal gate function.

Parameters are optimized with `scipy.optimize.least_squares` (or robust `soft_l1` loss). Uncertainties come from the Jacobian covariance and optional bootstrap resampling.

### Estimated quantities

| Quantity | Symbol |
|----------|--------|
| Transit depth | \(\delta\) |
| Duration | \(T_\mathrm{dur}\) |
| Mid-transit time | \(T_0\) |
| Orbital period | \(P\) (held from BLS, refined offset) |
| Planet radius ratio | \(R_p/R_\*\approx\sqrt{\delta}\) |
| Impact parameter | \(b\) |
| Ingress / egress | \(T_\mathrm{ingress}, T_\mathrm{egress}\) |
| Observed transits | \(N_\mathrm{transit}\) |

## Physical parameters

When stellar metadata is present in `sample.meta` or `candidate.meta`:

**Kepler's third law** (semi-major axis):

\[
a = \left(\frac{G M_\* P^2}{4\pi^2}\right)^{1/3}
\]

**Planet radius** from depth:

\[
R_p = \frac{R_\*}{R_\oplus}\sqrt{\delta}
\]

**Equilibrium temperature** (gray atmosphere, albedo \(A\)):

\[
T_\mathrm{eq} = T_\* \sqrt{\frac{R_\*}{2a}} (1 - A)^{1/4}
\]

**Incident flux** (relative to Earth):

\[
F = \frac{\sigma T_\*^4 R_\*^2 / a^2}{\sigma T_\odot^4 R_\odot^2 / \mathrm{AU}^2}
\]

Missing stellar parameters are recorded in `PhysicalParameters.missing_fields`; derived quantities are left `None`.

## YAML

```yaml
inference:
  parameter_fit:
    method: least_squares   # or robust
    loss: soft_l1
    bootstrap_samples: 20
  physical: {}
```

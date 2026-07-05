"""Transit parameter refinement via numerical optimization.

Units
-----
* ``depth``: dimensionless relative flux decrement
* ``duration_days``, ``epoch_days``, ``period_days``: days
* Phase coordinates: orbital phase in [-0.5, 0.5)

Assumptions
-----------
* Local phase-folded view is aligned with transit centered at phase 0.
* Period fixed to BLS estimate; epoch offset refined in phase units.
* Trapezoidal ingress/egress ramps approximate finite limb darkening.
* :math:`R_p/R_* \\approx \\sqrt{\\delta}` for small opaque disks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from exodet.inference.scientific import UNITS
from exodet.representation.containers import DatasetSample

__all__ = [
    "RefinedTransitParameters",
    "TransitParameterRefiner",
    "fit_transit_parameters",
]

logger = logging.getLogger(__name__)

# Parameter bounds (scientifically motivated ranges).
_DEPTH_MIN = 1e-6
_DEPTH_MAX = 0.5
_HALF_WIDTH_PHASE_MIN = 1e-5
_HALF_WIDTH_PHASE_MAX = 0.5
_CENTER_PHASE_MIN = -0.1
_CENTER_PHASE_MAX = 0.1
_INGRESS_PHASE_MIN = 1e-4
_INGRESS_PHASE_MAX = 0.25
_COV_REGULARIZATION = 1e-8
_NOISE_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class RefinedTransitParameters:
    """Refined transit ephemeris and shape parameters."""

    depth: float
    duration_days: float
    epoch_days: float
    period_days: float
    rp_rs: float
    impact_parameter: float
    ingress_duration_days: float
    egress_duration_days: float
    n_observed_transits: int
    chi2: float
    dof: int
    fit_method: str
    uncertainties: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=lambda: dict(UNITS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "duration_days": self.duration_days,
            "epoch_days": self.epoch_days,
            "period_days": self.period_days,
            "rp_rs": self.rp_rs,
            "impact_parameter": self.impact_parameter,
            "ingress_duration_days": self.ingress_duration_days,
            "egress_duration_days": self.egress_duration_days,
            "n_observed_transits": self.n_observed_transits,
            "chi2": self.chi2,
            "dof": self.dof,
            "fit_method": self.fit_method,
            "uncertainties": dict(self.uncertainties),
            "units": dict(self.units),
        }


def _trapezoid_model(
    phase: npt.NDArray[np.float64],
    depth: float,
    half_width: float,
    ingress: float,
) -> npt.NDArray[np.float64]:
    """Trapezoidal transit model on orbital phase (numerically clipped)."""
    depth = float(np.clip(depth, _DEPTH_MIN, _DEPTH_MAX))
    ingress = max(float(ingress), _INGRESS_PHASE_MIN)
    hw = max(float(half_width), ingress + _INGRESS_PHASE_MIN)
    abs_phase = np.abs(phase)
    flux = np.ones_like(phase, dtype=np.float64)
    flat = abs_phase <= (hw - ingress)
    ramp = (abs_phase > (hw - ingress)) & (abs_phase < hw)
    flux[flat] = 1.0 - depth
    if np.any(ramp):
        slope = depth / ingress
        flux[ramp] = 1.0 - slope * (hw - abs_phase[ramp])
    return flux


def _phase_axis(n_bins: int) -> npt.NDArray[np.float64]:
    return np.linspace(-0.5, 0.5, n_bins, endpoint=False, dtype=np.float64)


def _stable_covariance(jacobian: npt.NDArray[np.float64], cost: float, dof: int) -> npt.NDArray[np.float64]:
    """Regularized covariance from Jacobian (stability over raw matrix inverse)."""
    jtj = jacobian.T @ jacobian
    n = jtj.shape[0]
    regularized = jtj + _COV_REGULARIZATION * np.eye(n)
    try:
        return np.linalg.solve(
            regularized,
            np.eye(n),
        ) * (cost / max(dof, 1))
    except np.linalg.LinAlgError:
        return np.full((n, n), np.nan)


class TransitParameterRefiner:
    """Refines BLS estimates using least-squares or robust fitting."""

    def __init__(
        self,
        method: str = "least_squares",
        loss: str = "linear",
        bootstrap_samples: int = 0,
        seed: int = 0,
    ) -> None:
        self.method = method
        self.loss = loss
        self.bootstrap_samples = max(0, int(bootstrap_samples))
        self._rng = np.random.default_rng(seed)

    def refine(self, sample: DatasetSample) -> RefinedTransitParameters:
        """Fits a trapezoidal model to the local phase-folded view."""
        candidate = sample.candidate
        flux = sample.local_view.astype(np.float64)
        phase = _phase_axis(len(flux))
        scatter = max(float(np.std(flux)), _NOISE_FLOOR)
        weights = 1.0 / scatter

        p0 = np.array(
            [
                max(candidate.depth, _DEPTH_MIN),
                max(candidate.duration_days / (2.0 * candidate.period_days), _HALF_WIDTH_PHASE_MIN),
                0.0,
                max(0.05 * candidate.duration_days / candidate.period_days, _INGRESS_PHASE_MIN),
            ],
            dtype=np.float64,
        )
        bounds = (
            [_DEPTH_MIN, _HALF_WIDTH_PHASE_MIN, _CENTER_PHASE_MIN, _INGRESS_PHASE_MIN],
            [_DEPTH_MAX, _HALF_WIDTH_PHASE_MAX, _CENTER_PHASE_MAX, _INGRESS_PHASE_MAX],
        )

        def residuals(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            depth, half_width, center, ingress = params
            shifted = (phase - center + 0.5) % 1.0 - 0.5
            model = _trapezoid_model(shifted, depth, half_width, ingress)
            return (model - flux) * weights

        result = least_squares(
            residuals,
            p0,
            bounds=bounds,
            loss=self.loss if self.method == "robust" else "linear",
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
        )
        depth, half_width, center, ingress = result.x
        duration_days = 2.0 * half_width * candidate.period_days
        ingress_days = ingress * candidate.period_days
        egress_days = ingress_days
        rp_rs = float(np.sqrt(max(depth, 0.0)))
        impact = float(np.clip(1.0 - half_width * candidate.period_days / 0.05, 0.0, 1.5))

        uncertainties: dict[str, float] = {}
        if result.jac is not None and result.jac.size > 0:
            cov = _stable_covariance(result.jac, result.cost, max(len(flux) - 4, 1))
            if np.all(np.isfinite(cov)):
                sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                uncertainties = {
                    "depth": float(sig[0]),
                    "half_width_phase": float(sig[1]),
                    "center_phase": float(sig[2]),
                    "ingress_phase": float(sig[3]),
                }

        if self.bootstrap_samples > 0:
            boot = self._bootstrap_uncertainty(flux, phase, p0, bounds, scatter)
            uncertainties = {**uncertainties, **boot}

        chi2 = float(2.0 * result.cost)
        return RefinedTransitParameters(
            depth=float(depth),
            duration_days=float(duration_days),
            epoch_days=float(candidate.epoch_days + center * candidate.period_days),
            period_days=float(candidate.period_days),
            rp_rs=rp_rs,
            impact_parameter=impact,
            ingress_duration_days=float(ingress_days),
            egress_duration_days=float(egress_days),
            n_observed_transits=int(candidate.n_transits),
            chi2=chi2,
            dof=max(len(flux) - 4, 1),
            fit_method=self.method,
            uncertainties=uncertainties,
        )

    def _bootstrap_uncertainty(
        self,
        flux: npt.NDArray[np.float64],
        phase: npt.NDArray[np.float64],
        p0: npt.NDArray[np.float64],
        bounds: tuple[list[float], list[float]],
        scatter: float,
    ) -> dict[str, float]:
        depths: list[float] = []
        for _ in range(self.bootstrap_samples):
            noise = self._rng.normal(0.0, scatter * 0.01, size=len(flux))
            sample_flux = flux + noise

            def residuals(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                depth, half_width, center, ingress = params
                shifted = (phase - center + 0.5) % 1.0 - 0.5
                model = _trapezoid_model(shifted, depth, half_width, ingress)
                return model - sample_flux

            try:
                fit = least_squares(residuals, p0, bounds=bounds, method="trf", ftol=1e-8, xtol=1e-8)
                depths.append(float(fit.x[0]))
            except ValueError:
                continue
        if len(depths) < 2:
            return {}
        return {"depth_bootstrap_std": float(np.std(depths, ddof=1))}


def fit_transit_parameters(
    sample: DatasetSample,
    config: dict[str, Any] | None = None,
) -> RefinedTransitParameters:
    """Convenience wrapper from YAML ``parameter_fit`` block."""
    cfg = config or {}
    refiner = TransitParameterRefiner(
        method=str(cfg.get("method", "least_squares")),
        loss=str(cfg.get("loss", "soft_l1")),
        bootstrap_samples=int(cfg.get("bootstrap_samples", 0)),
        seed=int(cfg.get("seed", 0)),
    )
    return refiner.refine(sample)

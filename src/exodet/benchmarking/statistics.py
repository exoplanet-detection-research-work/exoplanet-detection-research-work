"""Statistical significance tests for model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

__all__ = [
    "BootstrapCI",
    "McNemarResult",
    "PairedTestResult",
    "bootstrap_confidence_interval",
    "mcnemar_test",
    "paired_t_test",
    "wilcoxon_signed_rank_test",
    "compare_model_predictions",
]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Bootstrap confidence interval for a scalar metric."""

    metric: str
    point_estimate: float
    lower: float
    upper: float
    n_bootstrap: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "point_estimate": self.point_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "n_bootstrap": self.n_bootstrap,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """McNemar test for paired binary classifiers."""

    statistic: float
    p_value: float
    n01: int
    n10: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n01": self.n01,
            "n10": self.n10,
            "significant_0_05": bool(self.p_value < 0.05),
        }


@dataclass(frozen=True, slots=True)
class PairedTestResult:
    """Result of a paired comparison on per-sample scores."""

    test: str
    statistic: float
    p_value: float
    mean_difference: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "mean_difference": self.mean_difference,
            "significant_0_05": bool(self.p_value < 0.05),
        }


def bootstrap_confidence_interval(
    values: npt.NDArray[np.float64],
    *,
    metric: str = "mean",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Non-parametric bootstrap CI for the mean of ``values``.

    For sample mean :math:`\\bar{x}` with :math:`B` bootstrap replicates
    :math:`\\bar{x}^{(b)}`, the percentile interval uses quantiles
    :math:`\\alpha/2` and :math:`1-\\alpha/2` of :math:`\\{\\bar{x}^{(b)}\\}`.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must be non-empty.")
    point = float(np.mean(arr))
    boot = np.empty(n_bootstrap, dtype=np.float64)
    n = arr.size
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        boot[i] = float(np.mean(sample))
    alpha = 1.0 - confidence
    lower, upper = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        metric=metric,
        point_estimate=point,
        lower=float(lower),
        upper=float(upper),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


def mcnemar_test(
    y_true: npt.NDArray[np.int_],
    pred_a: npt.NDArray[np.int_],
    pred_b: npt.NDArray[np.int_],
    *,
    continuity: bool = True,
) -> McNemarResult:
    """McNemar test with continuity correction.

    Contingency table counts:
    - n01: A wrong, B correct
    - n10: A correct, B wrong

    With continuity correction (when n01+n10 > 0):

    .. math::

        \\chi^2 = \\frac{(|n_{01}-n_{10}|-1)^2}{n_{01}+n_{10}}
    """
    y_true = np.asarray(y_true, dtype=np.int_)
    pred_a = np.asarray(pred_a, dtype=np.int_)
    pred_b = np.asarray(pred_b, dtype=np.int_)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n01 = int(np.sum(correct_b & ~correct_a))
    n10 = int(np.sum(correct_a & ~correct_b))
    denom = n01 + n10
    if denom == 0:
        return McNemarResult(statistic=0.0, p_value=1.0, n01=n01, n10=n10)
    if continuity:
        statistic = (abs(n01 - n10) - 1) ** 2 / denom
    else:
        statistic = (n01 - n10) ** 2 / denom
    p_value = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(statistic=float(statistic), p_value=p_value, n01=n01, n10=n10)


def paired_t_test(
    scores_a: npt.NDArray[np.float64],
    scores_b: npt.NDArray[np.float64],
) -> PairedTestResult:
    """Two-sided paired Student t-test on per-sample scores.

    Tests :math:`H_0: \\mu_d = 0` for differences :math:`d_i = a_i - b_i`.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have the same shape.")
    diff = a - b
    if diff.size < 2:
        return PairedTestResult("paired_t_test", 0.0, 1.0, float(np.mean(diff)))
    statistic, p_value = stats.ttest_rel(a, b, nan_policy="omit")
    return PairedTestResult(
        test="paired_t_test",
        statistic=float(statistic),
        p_value=float(p_value),
        mean_difference=float(np.mean(diff)),
    )


def wilcoxon_signed_rank_test(
    scores_a: npt.NDArray[np.float64],
    scores_b: npt.NDArray[np.float64],
) -> PairedTestResult:
    """Wilcoxon signed-rank test for paired non-Gaussian score differences."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have the same shape.")
    diff = a - b
    if diff.size < 2 or np.allclose(diff, 0.0):
        return PairedTestResult("wilcoxon", 0.0, 1.0, float(np.mean(diff)))
    try:
        statistic, p_value = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return PairedTestResult("wilcoxon", 0.0, 1.0, float(np.mean(diff)))
    return PairedTestResult(
        test="wilcoxon",
        statistic=float(statistic),
        p_value=float(p_value),
        mean_difference=float(np.mean(diff)),
    )


def compare_model_predictions(
    y_true: npt.NDArray[np.int_],
    predictions: dict[str, npt.NDArray[np.int_]],
    probabilities: dict[str, npt.NDArray[np.float64]] | None = None,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Pairwise statistical comparison across named models."""
    names = list(predictions.keys())
    results: dict[str, Any] = {"pairwise": {}, "bootstrap_accuracy": {}}
    y_true = np.asarray(y_true, dtype=np.int_)
    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=np.int_)
        correct = (pred == y_true).astype(np.float64)
        results["bootstrap_accuracy"][name] = bootstrap_confidence_interval(
            correct, metric=f"accuracy_{name}", n_bootstrap=n_bootstrap, seed=seed
        ).to_dict()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = f"{a}_vs_{b}"
            pair: dict[str, Any] = {
                "mcnemar": mcnemar_test(
                    y_true, predictions[a], predictions[b]
                ).to_dict(),
            }
            if probabilities is not None and a in probabilities and b in probabilities:
                pa = np.asarray(probabilities[a], dtype=np.float64)
                pb = np.asarray(probabilities[b], dtype=np.float64)
                pair["paired_t_test"] = paired_t_test(pa, pb).to_dict()
                pair["wilcoxon"] = wilcoxon_signed_rank_test(pa, pb).to_dict()
            results["pairwise"][key] = pair
    return results

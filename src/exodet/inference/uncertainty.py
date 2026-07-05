"""Uncertainty estimation for classification and transit parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from exodet.ml.data import MlBatch, collate_ml_batch
from exodet.ml.models import BaseTorchModel
from exodet.representation.containers import DatasetSample, RepresentationDataset

__all__ = [
    "UncertaintyEstimate",
    "build_uncertainty_estimator",
    "monte_carlo_dropout",
    "bootstrap_uncertainty",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    """Mean, spread, and credible interval for a scalar quantity."""

    mean: float
    std: float
    lower: float
    upper: float
    method: str
    n_samples: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "mean": self.mean,
            "std": self.std,
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
            "n_samples": self.n_samples,
        }


def _credible_interval(samples: npt.NDArray[np.float64], alpha: float = 0.68) -> tuple[float, float]:
    lo = (1.0 - alpha) / 2.0
    return float(np.quantile(samples, lo)), float(np.quantile(samples, 1.0 - lo))


def monte_carlo_dropout(
    model: BaseTorchModel,
    batch: MlBatch,
    n_samples: int = 30,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Draws MC-dropout probability samples per row (deterministic given seed)."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    network = model.module
    network.train()
    for module in network.modules():
        if module.__class__.__name__ == "Dropout":
            module.train()

    probs: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model.forward_batch(batch)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
            probs.append(p)
    network.eval()
    return np.stack(probs, axis=0)


def bootstrap_uncertainty(
    predict_fn: Callable[[RepresentationDataset], npt.NDArray[np.float64]],
    dataset: RepresentationDataset,
    n_samples: int = 50,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Bootstrap resampling uncertainty on dataset-level predictions."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    samples: list[npt.NDArray[np.float64]] = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        subset = RepresentationDataset([dataset.samples[int(i)] for i in idx])
        samples.append(predict_fn(subset))
    return np.stack(samples, axis=0)


def _ensemble_predict(
    models: list[BaseTorchModel],
    dataset: RepresentationDataset,
    use_views: str,
) -> npt.NDArray[np.float64]:
    from exodet.ml.inference import InferenceEngine

    probs = []
    for m in models:
        engine = InferenceEngine(m, use_views=use_views)
        probs.append(engine.predict_batch(dataset).probabilities)
    return np.stack(probs, axis=0)


class UncertaintyEstimator:
    """Configurable uncertainty engine."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.method = str(config.get("method", "none"))
        self.n_samples = int(config.get("n_samples", 30))
        self.credible_alpha = float(config.get("credible_alpha", 0.68))
        self.seed = int(config.get("seed", 0))

    def estimate_batch(
        self,
        model: BaseTorchModel,
        dataset: RepresentationDataset,
        use_views: str = "both",
    ) -> list[UncertaintyEstimate]:
        """Returns per-sample uncertainty estimates."""
        if self.method == "none":
            from exodet.ml.inference import InferenceEngine

            base = InferenceEngine(model, use_views=use_views).predict_batch(dataset).probabilities
            return [
                UncertaintyEstimate(
                    mean=float(p),
                    std=0.0,
                    lower=float(p),
                    upper=float(p),
                    method="none",
                    n_samples=1,
                )
                for p in base
            ]

        if self.method == "mc_dropout":
            return self._mc_dropout_batch(model, dataset)

        if self.method == "bootstrap":
            from exodet.ml.inference import InferenceEngine

            engine = InferenceEngine(model, use_views=use_views)

            def predict_fn(ds: RepresentationDataset) -> npt.NDArray[np.float64]:
                return engine.predict_batch(ds).probabilities

            stacked = bootstrap_uncertainty(predict_fn, dataset, self.n_samples, self.seed)
            return self._summarize_stack(stacked)

        raise ValueError(f"Unknown uncertainty method: {self.method}")

    def _mc_dropout_batch(
        self,
        model: BaseTorchModel,
        dataset: RepresentationDataset,
    ) -> list[UncertaintyEstimate]:
        from torch.utils.data import DataLoader

        items = []
        for sample in dataset.samples:
            items.append(
                {
                    "global_view": sample.global_view,
                    "local_view": sample.local_view,
                    "features": sample.features,
                    "labels": sample.label,
                    "weights": sample.weight,
                    "sample_id": sample.sample_id,
                    "target_id": sample.target_id,
                }
            )
        loader = DataLoader(
            items,
            batch_size=min(32, len(items)),
            collate_fn=lambda b: collate_ml_batch(b, use_views="both"),
        )
        all_samples: list[list[float]] = [[] for _ in range(len(dataset))]
        offset = 0
        for batch in loader:
            stacked = monte_carlo_dropout(model, batch, self.n_samples, self.seed)
            for row in range(stacked.shape[1]):
                all_samples[offset + row] = stacked[:, row].tolist()
            offset += stacked.shape[1]

        estimates: list[UncertaintyEstimate] = []
        for row_samples in all_samples:
            arr = np.asarray(row_samples, dtype=np.float64)
            lo, hi = _credible_interval(arr, self.credible_alpha)
            estimates.append(
                UncertaintyEstimate(
                    mean=float(arr.mean()),
                    std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    lower=lo,
                    upper=hi,
                    method="mc_dropout",
                    n_samples=len(arr),
                )
            )
        return estimates

    def _summarize_stack(self, stacked: npt.NDArray[np.float64]) -> list[UncertaintyEstimate]:
        estimates: list[UncertaintyEstimate] = []
        for col in range(stacked.shape[1]):
            arr = stacked[:, col]
            lo, hi = _credible_interval(arr, self.credible_alpha)
            estimates.append(
                UncertaintyEstimate(
                    mean=float(arr.mean()),
                    std=float(arr.std(ddof=1)),
                    lower=lo,
                    upper=hi,
                    method=self.method,
                    n_samples=stacked.shape[0],
                )
            )
        return estimates


def build_uncertainty_estimator(config: dict[str, Any]) -> UncertaintyEstimator:
    """Factory from YAML ``uncertainty`` block."""
    return UncertaintyEstimator(config)

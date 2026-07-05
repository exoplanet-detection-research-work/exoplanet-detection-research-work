"""Inference latency and throughput benchmarking."""

from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import numpy as np

from exodet.inference.pipeline import ScientificInferencePipeline
from exodet.representation.containers import DatasetSample, RepresentationDataset

__all__ = ["InferenceBenchmarkResult", "benchmark_inference"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkResult:
    """Inference benchmark measurement."""

    device: str
    amp: str
    batch_size: int
    n_samples: int
    single_latency_ms: float
    batch_throughput_per_s: float
    peak_memory_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "amp": self.amp,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
            "single_latency_ms": self.single_latency_ms,
            "batch_throughput_per_s": self.batch_throughput_per_s,
            "peak_memory_mb": self.peak_memory_mb,
        }


def benchmark_inference(
    pipeline: ScientificInferencePipeline,
    dataset: RepresentationDataset,
    n_single: int = 5,
) -> InferenceBenchmarkResult:
    """Benchmarks single-target and batch inference."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    tracemalloc.start()
    single_times: list[float] = []
    for sample in dataset.samples[: min(n_single, len(dataset))]:
        t0 = time.perf_counter()
        pipeline.predict_single(sample)
        single_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    pipeline.predict_batch(dataset)
    batch_elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    from exodet.ml.device import select_device

    device = str(select_device(pipeline.settings.device).device)
    return InferenceBenchmarkResult(
        device=device,
        amp=pipeline.settings.amp,
        batch_size=pipeline.settings.batch_size,
        n_samples=len(dataset),
        single_latency_ms=float(np.mean(single_times) * 1000.0) if single_times else 0.0,
        batch_throughput_per_s=len(dataset) / max(batch_elapsed, 1e-9),
        peak_memory_mb=peak / (1024 * 1024),
    )

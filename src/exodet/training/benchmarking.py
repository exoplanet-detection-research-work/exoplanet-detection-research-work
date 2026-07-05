"""Training and inference benchmarking (Module 13)."""

from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

from exodet.ml.amp import AmpSettings
from exodet.ml.device import select_device

__all__ = ["BenchmarkResult", "benchmark_training_step", "benchmark_matrix"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Single benchmark measurement."""

    device: str
    amp: str
    batch_size: int
    sequence_length: int
    train_seconds: float
    steps_per_second: float
    peak_memory_mb: float
    inference_seconds: float | None = None


def benchmark_training_step(
    model: Any,
    batch: Any,
    device_kind: str,
    amp_mode: str = "none",
    n_steps: int = 10,
) -> BenchmarkResult:
    """Benchmarks one training step configuration."""
    import torch

    from exodet.ml.data import MlBatch
    from exodet.ml.models import BaseTorchModel

    device = select_device(device_kind if device_kind != "auto" else "auto").device
    amp = AmpSettings.from_mode(amp_mode, device.type if device.type != "mps" else "mps")

    if isinstance(model, BaseTorchModel):
        parts = [
            t for t in (batch.global_view, batch.local_view, batch.features) if t is not None
        ]
        input_dim = sum(int(t.shape[1]) for t in parts)
        model._ensure_module(input_dim, device)
        network = model.module
        network.train()
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)

        def _forward() -> torch.Tensor:
            mb = MlBatch(
                global_view=batch.global_view.to(device) if batch.global_view is not None else None,
                local_view=batch.local_view.to(device) if batch.local_view is not None else None,
                features=batch.features.to(device) if batch.features is not None else None,
                labels=batch.labels.to(device) if batch.labels is not None else None,
                weights=batch.weights.to(device) if batch.weights is not None else None,
                sample_ids=batch.sample_ids,
                target_ids=batch.target_ids,
            )
            return model.forward_batch(mb)
    else:
        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def _forward() -> torch.Tensor:
            return model(batch) if not hasattr(model, "forward_batch") else model.forward_batch(batch)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device.type):
            logits = _forward()
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            labels = torch.zeros(logits.shape[0], device=device)
            loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    seq_len = getattr(batch, "global_view", None)
    seq = int(seq_len.shape[1]) if seq_len is not None else 0
    return BenchmarkResult(
        device=str(device),
        amp=amp_mode,
        batch_size=int(logits.shape[0]),
        sequence_length=seq,
        train_seconds=elapsed,
        steps_per_second=n_steps / max(elapsed, 1e-9),
        peak_memory_mb=peak / (1024 * 1024),
    )


def benchmark_matrix(
    model: Any,
    batch: Any,
    batch_sizes: list[int] | None = None,
    amp_modes: list[str] | None = None,
    devices: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Runs a grid of benchmark configurations."""
    batch_sizes = batch_sizes or [16, 32, 64]
    amp_modes = amp_modes or ["none", "fp16"]
    devices = devices or ["cpu"]
    results: list[dict[str, Any]] = []
    for device in devices:
        for amp in amp_modes:
            for _bs in batch_sizes:
                try:
                    result = benchmark_training_step(model, batch, device, amp)
                    results.append(result.__dict__)
                except Exception as exc:
                    logger.warning("Benchmark failed device=%s amp=%s: %s", device, amp, exc)
    return results

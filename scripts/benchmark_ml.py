#!/usr/bin/env python3
"""Benchmark ML training throughput, AMP speedup, and memory usage."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        print("torch not installed; run: pip install 'exodet[ml]'")
        sys.exit(1)
    import torch

    return torch


def _benchmark_training(
    n_samples: int,
    epochs: int,
    batch_size: int,
    amp: str,
) -> dict[str, float]:
    from exodet.config.schema import TrainingConfig
    from exodet.ml.amp import AmpSettings
    from exodet.ml.device import select_device
    from exodet.ml.trainer import SupervisedTrainer
    from exodet.models.base import MODELS
    from tests.ml_fixtures import fast_training_config, make_representation_dataset

    import tests.ml_fixtures  # noqa: F401 — register linear_probe

    torch = _require_torch()
    dataset = make_representation_dataset(n_samples=n_samples, n_stars=max(4, n_samples // 8))
    split = int(0.8 * len(dataset))
    train = type(dataset)(dataset.samples[:split])
    val = type(dataset)(dataset.samples[split:])
    raw = fast_training_config(
        epochs=epochs,
        batch_size=batch_size,
        trainer_params={"amp": amp},
    )
    trainer = SupervisedTrainer(TrainingConfig.from_dict(raw))
    model = MODELS.build("linear_probe")
    device_info = select_device("auto")

    tracemalloc.start()
    t0 = time.perf_counter()
    trainer.train(model, train, val, checkpoint_dir=None)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    samples_per_sec = (epochs * len(train)) / max(elapsed, 1e-9)
    return {
        "n_samples": float(n_samples),
        "epochs": float(epochs),
        "batch_size": float(batch_size),
        "amp": amp,
        "device": device_info.kind,
        "elapsed_s": elapsed,
        "samples_per_sec": samples_per_sec,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ML training.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("ML training benchmark")
    print("=" * 60)
    results = []
    for n in args.sizes:
        for amp in ("none", "fp16"):
            if amp == "fp16":
                torch = _require_torch()
                if not torch.cuda.is_available():
                    print(f"n={n:5d} amp=fp16 skipped (CUDA not available)")
                    continue
            row = _benchmark_training(n, args.epochs, args.batch_size, amp)
            results.append(row)
            print(
                f"n={int(row['n_samples']):5d} amp={amp:4s} device={row['device']:4s} "
                f"time={row['elapsed_s']:.3f}s throughput={row['samples_per_sec']:.1f} samp/s "
                f"mem={row['peak_memory_mb']:.1f} MB"
            )

    if len(results) >= 2:
        base = next(r for r in results if r["amp"] == "none")
        fp16 = next((r for r in results if r["amp"] == "fp16" and r["n_samples"] == base["n_samples"]), None)
        if fp16 and base["elapsed_s"] > 0:
            speedup = base["elapsed_s"] / fp16["elapsed_s"]
            print(f"\nAMP speedup (n={int(base['n_samples'])}): {speedup:.2f}x")


if __name__ == "__main__":
    main()

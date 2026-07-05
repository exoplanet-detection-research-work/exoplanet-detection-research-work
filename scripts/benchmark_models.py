#!/usr/bin/env python3
"""Benchmark exoplanet neural network architectures."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        print("torch not installed; run: pip install 'exodet[ml]'")
        sys.exit(1)
    import torch

    return torch


def _bench_model(name: str, batch_size: int, device: str) -> dict[str, float]:
    import exodet.models.registry  # noqa: F401
    from exodet.models.base import MODELS
    from exodet.models.config import parse_model_params

    torch = _require_torch()
    params = {
        "global_bins": 128,
        "local_bins": 64,
        "embed_dim": 64,
        "hidden_dim": 128,
        "cnn_channels": [16, 32],
        "cnn_kernel_sizes": [5, 3],
        "transformer_depth": 2,
        "transformer_heads": 2,
        "physics_hidden_dims": [32],
        "num_classes": 5,
        "compile_model": False,
    }
    if name == "physics_only":
        input_dim = 33
        global_view = local_view = None
        physics = torch.randn(batch_size, input_dim)
    elif name in ("cnn_only", "cnn"):
        input_dim = params["local_bins"]
        global_view = local_view = torch.randn(batch_size, input_dim)
        physics = None
    elif name in ("transformer_only", "transformer"):
        input_dim = params["global_bins"]
        global_view = torch.randn(batch_size, input_dim)
        local_view = physics = None
    elif name == "cnn_transformer":
        input_dim = params["global_bins"] + params["local_bins"]
        global_view = torch.randn(batch_size, params["global_bins"])
        local_view = torch.randn(batch_size, params["local_bins"])
        physics = None
    else:
        input_dim = params["global_bins"] + params["local_bins"] + 33
        global_view = torch.randn(batch_size, params["global_bins"])
        local_view = torch.randn(batch_size, params["local_bins"])
        physics = torch.randn(batch_size, 33)

    model = MODELS.build(name, **params)
    model._ensure_module(input_dim, torch.device(device))
    network = model.module
    network.train()

    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        model._module.to(dev)
        if global_view is not None:
            global_view = global_view.to(dev)
        if local_view is not None:
            local_view = local_view.to(dev)
        if physics is not None:
            physics = physics.to(dev)
    else:
        dev = torch.device("cpu")

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(10):
        out = network(
            global_view=global_view,
            local_view=local_view,
            physics=physics,
        )
        loss = out.class_logits.sum()
        loss.backward()
        network.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "model": name,
        "batch_size": float(batch_size),
        "device": str(dev),
        "elapsed_s": elapsed,
        "steps_per_sec": 10.0 / max(elapsed, 1e-9),
        "peak_memory_mb": peak / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exoplanet models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "fusion",
            "cnn_transformer",
            "cnn_only",
            "transformer_only",
            "physics_only",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print("Model architecture benchmark")
    print("=" * 72)
    for name in args.models:
        row = _bench_model(name, args.batch_size, args.device)
        print(
            f"{row['model']:<18} device={row['device']:<4} "
            f"batch={int(row['batch_size']):3d} "
            f"time={row['elapsed_s']:.3f}s "
            f"{row['steps_per_sec']:.1f} step/s "
            f"mem={row['peak_memory_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()

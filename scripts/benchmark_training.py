#!/usr/bin/env python3
"""Benchmark research training throughput and memory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.ml_fixtures import make_labeled_sample

from exodet.ml.data import collate_ml_batch
from exodet.models.base import MODELS


def _make_batch(n_global: int = 64, n_local: int = 32, n_physics: int = 12):
    items = []
    rng = np.random.default_rng(0)
    for i in range(4):
        s = make_labeled_sample(seed=i, n_global=n_global, n_local=n_local, n_features=n_physics)
        items.append(
            {
                "global_view": s.global_view,
                "local_view": s.local_view,
                "features": s.features,
                "labels": s.label,
                "weights": s.weight,
                "sample_id": s.sample_id,
                "target_id": s.target_id,
            }
        )
    return collate_ml_batch(items, use_views="both")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark training strategies.")
    parser.add_argument("--model", default="fusion")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    params = {
        "global_bins": 64,
        "local_bins": 32,
        "embed_dim": 32,
        "hidden_dim": 64,
        "cnn_channels": [16, 32],
        "cnn_kernel_sizes": [5, 3],
        "transformer_depth": 2,
        "transformer_heads": 2,
        "physics_hidden_dims": [32],
        "num_classes": 5,
    }
    model = MODELS.build(args.model, **params)
    batch = _make_batch(n_global=64, n_local=32, n_physics=12)
    input_dim = 64 + 32 + 12
    model._ensure_module(input_dim, __import__("torch").device("cpu"))

    results = benchmark_matrix(
        model,
        batch,
        amp_modes=["none"],
        devices=[args.device],
    )
    print("Training benchmark")
    for row in results:
        print(
            f"device={row['device']} batch={row['batch_size']} "
            f"steps/s={row['steps_per_second']:.1f} mem={row['peak_memory_mb']:.1f}MB"
        )


if __name__ == "__main__":
    main()

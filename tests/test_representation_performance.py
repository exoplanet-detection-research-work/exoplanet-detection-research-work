"""Performance regression tests for the representation stage."""

from __future__ import annotations

import time

import numpy as np

from exodet.representation import RepresentationPipeline
from exodet.representation.cache import RepresentationCache
from tests.representation_helpers import make_representation_pair
from tests.test_representation_integration import fast_representation_config


class TestRepresentationThroughput:
    def test_single_sample_runtime(self) -> None:
        pipeline = RepresentationPipeline(fast_representation_config())
        curve, candidate = make_representation_pair()
        pipeline.build_sample(curve, candidate)  # warm-up
        start = time.perf_counter()
        sample = pipeline.build_sample(curve, candidate)
        elapsed = time.perf_counter() - start
        print(f"single sample: {elapsed:.3f} s")
        assert sample.global_view.size == 201
        assert elapsed < 5.0

    def test_ten_samples_scale(self) -> None:
        pipeline = RepresentationPipeline(fast_representation_config())
        pairs = [make_representation_pair(seed=i) for i in range(10)]
        pipeline.build_sample(*pairs[0])  # warm-up
        start = time.perf_counter()
        for curve, candidate in pairs:
            pipeline.build_sample(curve, candidate)
        total = time.perf_counter() - start
        print(f"10 samples: {total:.3f} s ({total / 10:.3f} s/sample)")
        assert total < 30.0

    def test_cache_speedup(self, tmp_path) -> None:
        config = fast_representation_config(
            cache={"enabled": True, "directory": str(tmp_path / "cache"), "compress": True}
        )
        pipeline = RepresentationPipeline(
            config, cache=RepresentationCache(tmp_path / "cache", compress=True)
        )
        curve, candidate = make_representation_pair()
        pipeline.build_sample(curve, candidate)  # populate cache
        start = time.perf_counter()
        for _ in range(20):
            pipeline.build_sample(curve, candidate)
        cached = time.perf_counter() - start
        print(f"20 cached reads: {cached:.3f} s")
        assert cached < 1.0

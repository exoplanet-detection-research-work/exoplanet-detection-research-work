"""Tests for cross-platform process metrics."""

from __future__ import annotations

from exodet.utils.process_metrics import (
    process_cpu_seconds,
    process_num_threads,
    process_rss_bytes,
    process_stats,
    system_memory_bytes,
)


class TestProcessMetrics:
    def test_rss_bytes_positive(self) -> None:
        rss = process_rss_bytes()
        assert rss is None or rss > 0

    def test_cpu_seconds_non_negative(self) -> None:
        cpu = process_cpu_seconds()
        assert cpu is None or cpu >= 0.0

    def test_thread_count_positive(self) -> None:
        threads = process_num_threads()
        assert threads is None or threads >= 1

    def test_system_memory_positive(self) -> None:
        memory = system_memory_bytes()
        assert memory is None or memory > 0

    def test_process_stats_keys(self) -> None:
        stats = process_stats()
        assert "rss_bytes" in stats
        assert "platform" in stats

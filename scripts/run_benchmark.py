#!/usr/bin/env python3
"""Run the scientific benchmarking suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exodet.benchmarking.runner import run_benchmark
from exodet.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run exodet scientific benchmark suite.")
    parser.add_argument("--config", "-c", required=True, help="Benchmark YAML config path.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key override.",
    )
    args = parser.parse_args(argv)
    setup_logging()
    run_benchmark(Path(args.config), overrides=args.override)
    return 0


if __name__ == "__main__":
    sys.exit(main())

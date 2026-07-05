#!/usr/bin/env python3
"""Reproduce and validate experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exodet.experiments.runner import run_reproduce_experiments
from exodet.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce and validate exodet experiments.")
    parser.add_argument("--config", "-c", required=True, help="Experiment YAML config.")
    parser.add_argument(
        "--override", "-o", action="append", default=[], metavar="KEY=VALUE",
    )
    args = parser.parse_args(argv)
    setup_logging()
    payload = run_reproduce_experiments(Path(args.config), overrides=args.override)
    print(f"Reproducibility validation: all_passed={payload.get('all_passed')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

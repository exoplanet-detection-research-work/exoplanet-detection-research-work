#!/usr/bin/env python3
"""Run a hyperparameter sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exodet.experiments.runner import run_experiment_sweep
from exodet.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an exodet hyperparameter sweep.")
    parser.add_argument("--config", "-c", required=True, help="Experiment YAML config.")
    parser.add_argument(
        "--override", "-o", action="append", default=[], metavar="KEY=VALUE",
    )
    args = parser.parse_args(argv)
    setup_logging()
    payload = run_experiment_sweep(Path(args.config), overrides=args.override)
    print(f"Sweep complete: {payload['sweep_id']} ({len(payload['result']['trials'])} trials)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

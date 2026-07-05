#!/usr/bin/env python3
"""Run a registered experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exodet.experiments.runner import run_experiment
from exodet.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an exodet experiment.")
    parser.add_argument("--config", "-c", required=True, help="Experiment YAML config.")
    parser.add_argument(
        "--override", "-o", action="append", default=[], metavar="KEY=VALUE",
    )
    parser.add_argument("--id", default=None, help="Optional experiment ID.")
    args = parser.parse_args(argv)
    setup_logging()
    payload = run_experiment(Path(args.config), overrides=args.override, experiment_id=args.id)
    print(f"Experiment complete: {payload['experiment_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate reproducibility report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exodet.reproducibility.runner import run_reproducibility
from exodet.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate exodet reproducibility report.")
    parser.add_argument("--config", "-c", required=True, help="Experiment YAML config path.")
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
    run_reproducibility(Path(args.config), overrides=args.override)
    return 0


if __name__ == "__main__":
    sys.exit(main())

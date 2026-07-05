#!/usr/bin/env python3
"""Run architecture ablation study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exodet.ablation.runner import run_ablation
from exodet.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run exodet ablation study.")
    parser.add_argument("--config", "-c", required=True, help="Ablation YAML config path.")
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
    run_ablation(Path(args.config), overrides=args.override)
    return 0


if __name__ == "__main__":
    sys.exit(main())

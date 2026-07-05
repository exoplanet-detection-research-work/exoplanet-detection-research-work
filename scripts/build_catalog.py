#!/usr/bin/env python3
"""Build exoplanet candidate catalog from inference results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exodet.catalog.runner import run_catalog_build


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exoplanet candidate catalog.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    paths = run_catalog_build(args.config, overrides=args.override)
    for fmt, path in paths.items():
        print(f"{fmt}: {path}")


if __name__ == "__main__":
    main()

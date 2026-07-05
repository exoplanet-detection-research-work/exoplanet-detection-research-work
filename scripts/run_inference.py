#!/usr/bin/env python3
"""Run scientific inference from YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exodet.inference.runner import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scientific inference pipeline.")
    parser.add_argument("-c", "--config", required=True, help="Path to inference YAML.")
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    batch = run_inference(args.config, overrides=args.override)
    print(f"Inference complete: {len(batch)} samples.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare multiple trained models on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exodet.config.loader import _resolve_defaults, apply_overrides, load_yaml
from exodet.exceptions import PipelineError
from exodet.inference.runner import run_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained models.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    raw = apply_overrides(_resolve_defaults(load_yaml(args.config)), args.override)
    models = raw.get("compare", {}).get("models", {})
    if not models:
        raise PipelineError("Set compare.models in the config YAML.")
    report = run_model_comparison(args.config, dict(models), overrides=args.override)
    print(f"Compared models: {report.model_names}")


if __name__ == "__main__":
    main()

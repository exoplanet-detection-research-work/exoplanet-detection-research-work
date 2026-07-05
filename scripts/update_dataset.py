#!/usr/bin/env python3
"""CLI entrypoint for incremental dataset updates."""

from __future__ import annotations

import sys

from exodet.cli.main import main


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if not argv or argv[0] != "update":
        argv = ["update", *argv]
    sys.exit(main(argv))

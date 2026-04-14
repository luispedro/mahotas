#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "mahotas" / "mahotas_version.py"


def read_version() -> str:
    version_source = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", version_source)
    if match is None:
        raise SystemExit(f"Could not parse version from {VERSION_FILE}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["numpy-include", "version"])
    args = parser.parse_args()

    if args.action == "version":
        print(read_version())
        return

    import numpy

    print(numpy.get_include())


if __name__ == "__main__":
    main()

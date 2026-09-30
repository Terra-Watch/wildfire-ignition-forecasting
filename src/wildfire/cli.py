"""Command-line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from wildfire import __version__, config
from wildfire.data import manifest
from wildfire.features import REGISTRY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wildfire")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features", help="list registered feature sets")
    sub.add_parser("data-root", help="print the resolved data root")
    verify = sub.add_parser("verify-data", help="compare data folders with data/MANIFEST.json")
    verify.add_argument("--manifest", default=str(config.REPO_ROOT / "data" / "MANIFEST.json"))
    args = parser.parse_args(argv)

    if args.command == "features":
        for name, cls in sorted(REGISTRY.items()):
            print(f"{cls.tier}\t{name}")
    elif args.command == "data-root":
        print(config.data_root())
    elif args.command == "verify-data":
        problems = manifest.verify(Path(args.manifest))
        for p in problems:
            print(p)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

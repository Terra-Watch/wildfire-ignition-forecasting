"""Command-line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from wildfire import __version__, config
from wildfire.data import manifest
from wildfire.features import REGISTRY

PIPELINES = {
    "build-labels": "labels",
    "build-features": "features",
    "build-model-table": "model_table",
    "score-holdout": "holdout",
    "leave-year-out": "leave_year_out",
    "out-of-fold": "out_of_fold",
    "sanity-checks": "sanity",
}


def _run_pipeline(name: str) -> None:
    import importlib
    if name == "build-features":
        from wildfire.pipelines import features
        features.build_feature_table()
    elif name == "build-model-table":
        from wildfire.pipelines import features
        features.build_model_table()
    else:
        importlib.import_module(f"wildfire.pipelines.{PIPELINES[name]}").run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wildfire")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features", help="list registered feature sets")
    sub.add_parser("data-root", help="print the resolved data root")
    verify = sub.add_parser("verify-data", help="compare data folders with data/MANIFEST.json")
    verify.add_argument("--manifest", default=str(config.REPO_ROOT / "data" / "MANIFEST.json"))
    for name in PIPELINES:
        sub.add_parser(name, help=f"run the {name} pipeline (settings come from configs/ only)")
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
    elif args.command in PIPELINES:
        _run_pipeline(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

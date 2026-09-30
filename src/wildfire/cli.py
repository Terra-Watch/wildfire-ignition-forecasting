"""Command-line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from wildfire import __version__, config
from wildfire.data import manifest
from wildfire.features import REGISTRY

PIPELINES = {                     # command -> (module, function)
    "build-labels": ("labels", "run"),
    "build-features": ("features", "build_feature_table"),
    "build-model-table": ("features", "build_model_table"),
    "build-weather-series": ("enriched", "build_series"),
    "build-enriched": ("enriched", "build_arrays"),
    "build-fresh-rows": ("fresh_states", "build_rows"),
    "build-fresh-dataset": ("fresh_states", "build_dataset"),
    "score-holdout": ("holdout", "run"),
    "leave-year-out": ("leave_year_out", "run"),
    "out-of-fold": ("out_of_fold", "run"),
    "sanity-checks": ("sanity", "run"),
    "tune": ("tuning", "run"),
    "rolling-origin": ("rolling_origin", "run"),
    "baseline-tables": ("baseline_tables", "run"),
    "score-fresh-states": ("fresh_states", "score_once"),
}


def _run_pipeline(name: str) -> None:
    import importlib
    module, function = PIPELINES[name]
    getattr(importlib.import_module(f"wildfire.pipelines.{module}"), function)()


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

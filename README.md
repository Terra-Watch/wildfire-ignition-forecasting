# wildfire

Next-day wildfire ignition risk over Indian forest landscapes.

## Layout
- `configs/` every constant (domain, data, features, models, evaluation)
- `src/wildfire/` library: `data`, `labels`, `features`, `models`, `evaluation`, `guards`, `pipelines`
- `scripts/` download and build entry points
- `tests/` mirrors `src/`
- `docs/` data, methodology and decisions, named by topic and dated
- `data/`, `artifacts/` not tracked; `data/MANIFEST.json` describes the data

Exploratory studies and diagnostics live in the sibling repository `wildfire-research`, which depends on this one, never the reverse.

## Features
Two tiers scored by one harness (`wildfire.evaluation.harness`):
1. summary: `tabular`, `weather_windows`, `neighbourhood`
2. imagery: `imagery_summary` (placeholder)

## Setup
    cp .env.example .env     # set DATA_ROOT
    make setup
    make test
    wildfire verify-data     # compare DATA_ROOT with data/MANIFEST.json
    pre-commit install       # forbidden-label check on commit

Data is never copied into the repository; `DATA_ROOT` points at it.

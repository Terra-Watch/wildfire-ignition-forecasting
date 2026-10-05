# wildfire

Next-day wildfire ignition risk over Indian forest landscapes.

## Layout
- `configs/` every constant (domain, data, features, models, evaluation)
- `src/wildfire/` library: `data`, `labels`, `features`, `models`, `evaluation`, `guards`, `pipelines`
- `scripts/` download and build entry points
- `tests/` mirrors `src/`
- `results/` recorded result tables of the extended boosting baseline (small, tracked)
- `deep_multimodal/` the proposed deep multimodal model, self-contained
- `docs/` data, methodology and decisions, named by topic and dated
- `data/`, `artifacts/` not tracked; `data/MANIFEST.json` describes the data

## Baseline and proposed model
- The **extended boosting baseline** (gradient boosting on 251 inputs: 42 plain inputs plus weather windows, anomalies and neighbourhood summaries) is the model every later model is compared with. See `docs/extended_boosting_baseline.md`; recorded results (AP 0.2315 on 2025, 0.1977 pooled over five folds, in-season rows) are in `results/extended_boosting_baseline/`.
- `deep_multimodal/` is a self-contained folder for the proposed deep multimodal model (code, configs, tests, data and model descriptions, results, notebooks). It does not import this library.

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

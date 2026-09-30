# How to reproduce

## Purpose
Every table in this repository from a fresh checkout.

## Inputs
The data folders (`DATA_ROOT`), about 28 GB of space under `ARTIFACTS_ROOT` for derived files.

## Steps
    cp .env.example .env            # set DATA_ROOT and ARTIFACTS_ROOT
    make setup
    make test                       # tests and the name check
    wildfire verify-data
    wildfire build-labels
    wildfire build-features
    wildfire build-model-table
    wildfire build-weather-series
    wildfire build-enriched
    wildfire tune
    wildfire rolling-origin
    wildfire baseline-tables
    wildfire build-fresh-rows
    wildfire build-fresh-dataset
    wildfire score-fresh-states
    make notebooks

Commit code before scoring: scoring refuses to run on a dirty tree and on changed specification files.

## Outputs
Tables (`baseline_*.csv`), predictions, guard results and the ledger under `ARTIFACTS_ROOT/results`, and the notebooks with outputs.

## Known limits
Runs take hours (tuning and the network are the long parts). The earlier results were reproduced exactly from the same data; see the research repository.

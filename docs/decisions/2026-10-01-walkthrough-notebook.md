# Walk-through notebook 00: what is official and what is not
Decision: notebook `00_baseline_pipeline` follows the official fold step for step and labels everything that is not an official output.
- Official: dataset with the enriched arrays, the fit set `dataset.fit_rows(ds, year)` with the configured cap (the same call the rolling-origin run makes), design weights, prior-shift correction, model classes, ladder metric functions. Test-year numbers are read from the saved rolling-origin predictions.
- Not official, labelled in the notebook: accuracy, precision, recall, F1 and the decision threshold (weighted F1 maximised on the previous year's official fold); training-set metrics; the per-epoch network history (`models/network_traced.py`); confusion matrices; block permutation importance (`evaluation/classification.py`, `evaluation/importance.py`).
- The network rows of the train / validation / test comparison are seed 0; the official tables average 10 seeds.
- All logic lives in `src/` (`reporting/baseline_walkthrough.py`); notebook cells only call it. 2025, the fresh states and the 2026 season are never loaded; nothing is written to the results folder.
Why: an earlier draft re-implemented the fit with tabular inputs and unweighted rows; numbers from it would not have been comparable with the official tables.

# Plan: document the extended boosting baseline and add the deep multimodal model folder

Sources of truth: the research working copy (`~/wildfire-research/proposed`: saved predictions, recorded tables, headline model code) and its metric table (`results/METRIC_TABLE.*`). Local commits only; nothing is pushed, merged or deleted.

## What was found
- The main repository already holds the baseline code, configs, tests, eight notebooks and decision records, but its recorded results (local, untracked `artifacts/results`) are on the original all-year row set (2025 AP 0.2461, pooled 0.2261). The baseline of record, restricted to in-season rows, gives 0.2315 on 2025 and 0.1977 pooled; its build and re-run scripts and result tables live only in the research copy.
- The headline deep multimodal model lives in the research copy as one large training module with switches for many other arms, and imports helper code from the same copy.

## Part 1: extended boosting baseline (main repository)
**Copy** (small tables, rewritten with plain column names where needed): the in-season baseline tables (all arms, per year, five folds), and the 2024 and 2023 metric table.
**Rewrite**: the in-season table build and baseline re-run scripts as `scripts/` entry points without machine-specific paths; one document, `docs/extended_boosting_baseline.md` (what it is, which inputs it adds over plain boosting, both versions' recorded average precision on the same fold, how to re-run); README, `docs/models.md` and `STATUS.md` get the one name "extended boosting baseline".
**Add**: tests that read the committed result tables and check the quoted numbers; a short results notebook that runs from the committed tables.
**Leave out**: every proposed-model arm, the bagged tree comparator, imagery work, screening and cloud job scripts, and rewriting the historical decision records and executed notebooks (their outputs stay as they were).

## Part 2: `deep_multimodal/` (self-contained)
**Port** (headline model only): weather sequence stream (recurrent layers), neighbourhood-by-days fire-history transformer, day-ahead forecast tokens, static-feature tokens, cross-attention fusion with conditioning from static and calendar values, two-stage training with the prior-shift-corrected stopping score, 8-seed ensemble, hard labels only; the metric code of the 2024 and 2023 table; hazard-map drawing.
**Rewrite**: plain module, class and function names; docstrings in full words; one final-settings config; a synthetic package generator (CPU) with tests; a script that regenerates the table and sample hazard maps from the saved predictions; README, `docs/DATA.md`, `docs/MODEL.md`, `docs/RESULTS.md`; three notebooks (data, model, results and maps).
**Check**: an equivalence test of the ported network against the original on identical weights (run once against the research copy; the outcome is recorded in `docs/MODEL.md`).
**Leave out entirely**: the bagged tree comparator, the distillation arm, the blend, every other arm and switch (no-forecast, concatenation, MLP statics, no auxiliary, latent pooling, transformer weather, anchored, piecewise codes, listwise loss, weight averaging, candidate tokens, larger neighbourhood), the imagery work, cloud job and queue scripts, fold 2025 (no 2025 number appears), trained weights (not committed; location class and checksums recorded in `docs/MODEL.md`).

## Rules applied
Plain words only (the repository's name check runs on every commit and is never bypassed); no credentials, no private account or bucket names (bucket paths are written with a placeholder); tests run in both folders and the full suite runs at the end.

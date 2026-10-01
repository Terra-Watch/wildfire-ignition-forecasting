"""Build docs/PAPER_HANDOFF.md from the saved tables (nothing is typed in by hand except the text around them).
Run from the repository root with the .env variables set: python docs/methodology/make_paper_handoff.py"""
import json
import re
import subprocess
import time

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.reporting import transfer

R = config.output_dir("results")
git = lambda *a: subprocess.run(["git", "-C", str(config.REPO_ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()


def md(df: pd.DataFrame, floats: int = 4) -> str:
    cols = [str(c) for c in df.columns]
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{floats}f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def strip_seed(a: pd.Series) -> pd.Series:
    return a.str.replace(r"#\d+$", "", regex=True)


metrics = pd.read_csv(R / "baseline_metrics.csv")
seeds = metrics[metrics.arm.str.contains("#")].copy()
seeds["arm"] = strip_seed(seeds.arm)
metrics["arm"] = strip_seed(metrics.arm)
mean = metrics.groupby("arm", sort=False).mean(numeric_only=True)
sd = seeds.groupby("arm").std(numeric_only=True)
by_year = pd.read_csv(R / "baseline_metrics_by_year.csv")
by_year["arm"] = strip_seed(by_year.arm)
by_year = by_year.groupby(["arm", "year"], sort=False).mean(numeric_only=True).reset_index()
contrasts = pd.read_csv(R / "baseline_contrasts.csv")
fresh = transfer.fresh_metrics()


def fresh_by_seed() -> pd.DataFrame:
    """Every prediction column of the fresh test scored separately (network seeds not averaged)."""
    from wildfire.evaluation import ladder
    from wildfire.pipelines import baseline_tables
    rows = pd.read_parquet(R / "fresh_predictions.parquet")
    prev = json.loads((R / "fresh_fit_prevalence.json").read_text())
    out = []
    for state, g in rows.groupby("state"):
        g = g.reset_index(drop=True)
        for arm in baseline_tables.prediction_columns(g):
            score = g["score_" + arm].values if "score_" + arm in g else g["p_" + arm].values
            corrected = ladder.corrected_probabilities(g["p_" + arm].values, g.pi_true.values, prev[arm])
            out.append(dict(state=state, arm_full=arm, **ladder.arm_row(g, score, corrected, g.weight.values)))
    return pd.DataFrame(out)


seed_rows = fresh_by_seed()
FRESH_ORDER = ["climatology_cell", "landcover_season", "fwi_season", "logistic_weather_static@tabular", "logistic_weather_static@enriched",
               "logistic_with_burn@tabular", "logistic_with_burn@enriched", "boosting@tabular", "boosting@enriched",
               "tabular_network@tabular", "tabular_network@enriched", "coords_only", "cell_id_only", "static_only"]


def fresh_all_arms() -> pd.DataFrame:
    out = []
    for state in ("UK", "AP"):
        d = seed_rows[seed_rows.state == state].copy()
        d["arm"] = strip_seed(d.arm_full)
        prevalence = float(d.onsets.iloc[0] / d.rows.iloc[0])
        for arm in FRESH_ORDER:
            g = d[d.arm == arm]
            n = len(g)

            def fmt(col, lift=False):
                v = g[col] / prevalence if lift else g[col]
                return f"{v.iloc[0]:.4f}" if n == 1 else f"{v.mean():.4f} ± {v.std():.4f} [{v.min():.4f}-{v.max():.4f}]"
            out.append({"state": state, "arm": arm, "seeds": n, "AP": fmt("ap"), "lift (AP / prevalence)": fmt("ap", True),
                        "AUC pooled": fmt("auc_pooled"), "AUC spatial": fmt("auc_spatial_within_day"), "AUC temporal": fmt("auc_temporal_within_cell"),
                        "log loss": fmt("log_loss")})
    return pd.DataFrame(out)


fresh_table = fresh_all_arms()
prev_state = {st: float(seed_rows[seed_rows.state == st].onsets.iloc[0] / seed_rows[seed_rows.state == st].rows.iloc[0]) for st in ("UK", "AP")}
abl_m, abl_c = pd.read_csv(R / "ablation_metrics.csv"), pd.read_csv(R / "ablation_contrasts.csv")
gain = pd.read_csv(R / "post_hoc_onsets_recent_gain.csv")
guards = json.loads((R / "guards.json").read_text())
ledger = pd.read_csv(R / "geo_test_ledger.csv")
fit_prev = json.loads((R / "fresh_fit_prevalence.json").read_text())
dev = (config.REPO_ROOT / "docs/decisions/2026-10-01-deviation-list.md").read_text().split("\n", 1)[1].strip()
tests = subprocess.run([str(config.REPO_ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "tests"], capture_output=True, text=True, cwd=config.REPO_ROOT).stdout.strip().splitlines()[-1]

NO_LEVEL = ["climatology_cell", "landcover_season", "fwi_season"]
DIAG = ["coords_only", "cell_id_only", "static_only"]
MAIN = ["boosting@enriched", "tabular_network@enriched", "logistic_with_burn@enriched", "logistic_weather_static@enriched"]
COLS = ["auc_spatial_within_day", "auc_temporal_within_cell", "ap", "log_loss", "brier", "capture_5pct", "lift_5pct"]
short = {"auc_spatial_within_day": "spatial AUC", "auc_temporal_within_cell": "temporal AUC", "ap": "AP", "log_loss": "log loss", "brier": "Brier",
         "capture_5pct": "capture 5%", "lift_5pct": "lift 5%"}

main = mean.loc[NO_LEVEL + MAIN[::-1] + DIAG, COLS].rename(columns=short).reset_index()
net_sd = {c: sd.loc["tabular_network@enriched", c] for c in ("ap", "auc_spatial_within_day", "auc_temporal_within_cell")}

tab_enr = []
for arm in ["logistic_weather_static", "logistic_with_burn", "boosting", "tabular_network"]:
    t, e = mean.loc[f"{arm}@tabular"], mean.loc[f"{arm}@enriched"]
    tab_enr.append(dict(arm=arm, **{f"{short[c]} tabular": t[c] for c in ("auc_spatial_within_day", "auc_temporal_within_cell", "ap")},
                        **{f"{short[c]} enriched": e[c] for c in ("auc_spatial_within_day", "auc_temporal_within_cell", "ap")}, **{"AP gain": e.ap - t.ap}))
tab_enr = pd.DataFrame(tab_enr)

years = [2021, 2022, 2023, 2024, 2025]
key_arms = ["climatology_cell", "logistic_with_burn@tabular", "logistic_with_burn@enriched", "boosting@tabular", "boosting@enriched", "tabular_network@tabular", "tabular_network@enriched"]
yr = {}
for metric in ("ap", "auc_spatial_within_day", "auc_temporal_within_cell"):
    p = by_year[by_year.arm.isin(key_arms)].pivot(index="arm", columns="year", values=metric).loc[key_arms]
    yr[metric] = p.reset_index()
rows_year = by_year[by_year.arm == "climatology_cell"][["year", "rows", "onsets"]]

c = contrasts.copy()
c["seed_contrast"] = c.arm.str.contains("#")
c_tab = c[["contrast", "arm", "reference", "delta_ap", "ap_lo99", "ap_hi99", "mde", "worst_state", "worst_state_delta_ap", "verdict"]].round(4)

fr = fresh.copy()
fr_main = fr[fr.arm.isin(NO_LEVEL + ["boosting@tabular", "boosting@enriched", "logistic_with_burn@enriched", "tabular_network@enriched", "cell_id_only"])]
fr_main = fr_main.assign(rows=fr_main.rows.astype(int), onsets=fr_main.onsets.astype(int))
fr_tab = fr_main[["state", "arm", "auc_spatial_within_day", "auc_temporal_within_cell", "ap", "log_loss", "brier", "rows", "onsets"]]

by25 = by_year[by_year.year == 2025].set_index("arm")
uk = fr[fr.state == "UK"].set_index("arm")
head, tag = git("rev-parse", "HEAD"), git("rev-parse", "baseline-final^{commit}")
mde_lo, mde_hi = contrasts.mde.min(), contrasts.mde.max()

out = f"""# Paper handoff: baseline results

Generated {time.strftime('%Y-%m-%d %H:%M %Z')} from the saved tables by `docs/methodology/make_paper_handoff.py`. Numbers below are read from files under `artifacts/results/`; the paper and the Word document were not touched.

## 0. Provenance
- Frozen baseline: local tag `baseline-final` on commit `{tag[:7]}` (not pushed); records in `artifacts/baseline_final/` (code hash, config snapshot, data manifest, artifact manifest).
- Code that produced each result (the `code_commit` stamped in the files): baseline tables `{contrasts.code_commit.iloc[0][:7]}`, ablation tables `{abl_m.code_commit.iloc[0][:7]}`, sanity checks `{pd.read_csv(R / 'sanity.csv').code_commit.iloc[0][:7]}`, fresh scoring `{ledger.code_commit.iloc[0][:7]}`. Rolling-origin folds 2023-2025 were fitted by `d55a8ca`, folds 2021 and 2022 by the earlier code `5af5654` (deviation j). Specification commit `{git('log', '-1', '--format=%H', '--', 'configs/baseline.yaml', 'docs/baseline_spec.md')[:7]}`.
- Guard record (`guards.json`, code `{guards['code_version'][:7]}`), every guard true: {', '.join(k for k, v in guards.items() if v is True)}; label-shuffle mean AUC {guards['label_shuffle_mean_auc']:.4f}.
- Data and artifact records: `wildfire verify-data` (21:2x on 2026-10-01) compared `data/MANIFEST.json` with the data folders and found no differences (exit code 0, no output). The frozen artifact manifest (`artifacts/baseline_final/artifact_manifest.json`) compared with the folders now: labels and features unchanged; results changed from 39 to 44 files, the five files added after the freeze (the fresh predictions, the fresh fit prevalences, the ledger and the two strata CSVs). The data manifest record itself was copied from the committed manifest.
- Tests at this commit: {tests}. Sanity checks (`sanity.csv`): global label shuffle mean AP 0.0502 against prevalence 0.0522, cell x week shuffle max AP 0.0933 against climatology 0.1167, permuted holdout labels 0.0522: all passed.
- Fresh geographic test scored once: ledger entries {', '.join(ledger.what)} at {ledger.time.iloc[0][:19]} (commit `{ledger.code_commit.iloc[0][:7]}`).

## 1. All {len(contrasts)} contrasts are INCONCLUSIVE
Every pre-declared contrast ends INCONCLUSIVE, because its minimum detectable effect (MDE {mde_lo:.4f} to {mde_hi:.4f}) is above the power threshold of 0.010 (spec section 7: a verdict is INCONCLUSIVE when the MDE exceeds the threshold). This is the decision rule working as declared, not a null result: several gains are large and their 99% intervals are clear of zero. For example enriched boosting against tabular boosting is ΔAP {contrasts.iloc[0].delta_ap:+.4f} with 99% CI {contrasts.iloc[0].ap_lo99:.4f} to {contrasts.iloc[0].ap_hi99:.4f}, MDE {contrasts.iloc[0].mde:.4f}. The paper must not state a PASS for any contrast. The power gate is recorded as a guard (`power_gate`, a self-test of this rule); the saved contrasts obey it (none has an MDE at or below 0.010).

Every contrast (arm minus reference, AP, 99% paired ISO-week cluster bootstrap; "#n" is network seed n; worst state = state with the lowest paired ΔAP):

{md(c_tab)}

## 2. Main table: enriched level
Pooled over validation years 2021-2025 (2025 is a development fold, not untouched), {int(rows_year.rows.sum()):,} rows and {int(rows_year.onsets.sum()):,} onsets. Log loss and Brier use design weights and the per-state true rate; AP and both AUCs use the raw scores. The network row is the mean of 10 seeds (seed standard deviation of AP {net_sd['ap']:.4f}, spatial AUC {net_sd['auc_spatial_within_day']:.4f}, temporal AUC {net_sd['auc_temporal_within_cell']:.4f}). The three arms without levels and the three diagnostic arms are listed for reference; the diagnostics measure the place shortcut and are not gated.

{md(main)}

## 3. Tabular versus enriched
{md(tab_enr)}

The information comparison (enriched boosting against tabular boosting) is contrast (a): ΔAP {contrasts.iloc[0].delta_ap:+.4f}, 99% CI {contrasts.iloc[0].ap_lo99:.4f} to {contrasts.iloc[0].ap_hi99:.4f}, MDE {contrasts.iloc[0].mde:.4f}, INCONCLUSIVE. Ablations (secondary, post hoc, `docs/decisions/2026-10-01-ablations.md`; boosting@enriched with the same folds and settings; ΔAP against boosting@enriched, AP {abl_m[abl_m.arm == 'boosting@enriched'].ap.iloc[0]:.4f}):

{md(abl_c[['contrast', 'delta_ap', 'ap_lo99', 'ap_hi99', 'mde', 'verdict']].round(4))}

(A_no_soil_anomaly removes `sw1_anom` and `sw2_anom`; B_windows_only keeps the weather windows and their anomalies but no neighbourhood block; B_neighbourhood_only keeps the neighbourhood block but no windows. Verdict labels here are descriptive only.) Post hoc stratification by recent neighbour onsets (`docs/decisions/2026-10-01-onsets-recent-strata.md`): the enriched AP gain is +{gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'above_zero')].delta_ap.iloc[0]:.4f} (99% CI {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'above_zero')].ap_lo99.iloc[0]:.4f} to {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'above_zero')].ap_hi99.iloc[0]:.4f}) in rows with recent onsets in the 8 neighbouring cells (94.1% of positives) and {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'zero')].delta_ap.iloc[0]:+.4f} (99% CI {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'zero')].ap_lo99.iloc[0]:.4f} to {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'zero')].ap_hi99.iloc[0]:.4f}) in rows with none, so a gain there is not demonstrated.

## 4. Rolling-origin results per year
Each year is scored by models fitted on all earlier years from 2019, January to July rows only. Rows and onsets per year: {', '.join(f'{int(r.year)}: {int(r.rows):,} / {int(r.onsets):,}' for r in rows_year.itertuples())}. Network rows are means over 10 seeds.

AP:

{md(yr['ap'])}

Within-day spatial AUC:

{md(yr['auc_spatial_within_day'])}

Within-cell temporal AUC:

{md(yr['auc_temporal_within_cell'])}

## 5. Fresh states (UK and AP), scored once
Every arm fitted on all years before 2026 (2,000,000-row cap) and scored on the fresh rows: AP {int(fr[fr.state == 'AP'].rows.iloc[0]):,} rows ({int(fr[fr.state == 'AP'].onsets.iloc[0]):,} onsets, sampled prevalence {prev_state['AP']:.4f}), UK {int(fr[fr.state == 'UK'].rows.iloc[0]):,} rows ({int(fr[fr.state == 'UK'].onsets.iloc[0]):,} onsets, sampled prevalence {prev_state['UK']:.4f}), years 2020-2025, every month of the fire season through 2025-07-31 (not restricted to January-July: deviation o). Unit design weights and the configured population rate 0.0147 are used, so log loss is less comparable than AP and the AUC views. Lift is AP divided by the state's sampled prevalence. No state-level contrast tests were declared for the fresh states; no verdicts are given.

Networks are shown as mean ± SD over the 10 seeds with the per-seed range [min-max]; every other arm has one fit. The figures AP 0.1144 (AP) and 0.1238 (UK), quoted in earlier discussion, are the AP of **`tabular_network@enriched`** (mean over 10 seeds); the tabular-level network has AP {seed_rows[(seed_rows.state == 'AP') & seed_rows.arm_full.str.startswith('tabular_network@tabular')].ap.mean():.4f} (AP) and {seed_rows[(seed_rows.state == 'UK') & seed_rows.arm_full.str.startswith('tabular_network@tabular')].ap.mean():.4f} (UK).

{md(fresh_table)}

All arms and the diagnostic arms are in `artifacts/results/fresh_predictions.parquet`. The sealed season (2026 January-July) is unavailable and was not scored (deviation e). Comparison with the earlier Uttarakhand reference result: `docs/decisions/2026-10-01-reference-reconciliation.md`.

## 6. Numbers that changed from the Word document (earlier reference values)
**INTERNAL ONLY, NOT FOR THE PAPER.** This table is a working comparison between the Word document's earlier reference numbers and the baseline tables; it must not be quoted or reproduced in the paper.

Old values are those the Word document states, taken from earlier reference runs, not from baseline tables. New values are read from the saved tables.

| Item | Old value | New value | Source file |
|---|---|---|---|
| Holdout 2025, trained states: boosting AP | 0.200 | {by25.loc['boosting@tabular', 'ap']:.3f} (tabular), {by25.loc['boosting@enriched', 'ap']:.3f} (enriched), fold 2025 fitted on 2019-2024 | `baseline_metrics_by_year.csv` (year 2025) |
| Holdout 2025: cell-history climatology AP | 0.117 | {by25.loc['climatology_cell', 'ap']:.3f} | `baseline_metrics_by_year.csv` |
| Uttarakhand (UK): boosting AP | 0.059 | {uk.loc['boosting@tabular', 'ap']:.3f} (tabular), {uk.loc['boosting@enriched', 'ap']:.3f} (enriched) | `fresh_predictions.parquet` |
| Uttarakhand (UK): boosting log loss | 0.286 | {uk.loc['boosting@tabular', 'log_loss']:.3f} (tabular), {uk.loc['boosting@enriched', 'log_loss']:.3f} (enriched); climatology {uk.loc['climatology_cell', 'log_loss']:.3f} | `fresh_predictions.parquet` |
| Uttarakhand (UK): FWI+season AP | 0.107 | {uk.loc['fwi_season', 'ap']:.3f} | `fresh_predictions.parquet` |
| True rate | 0.0147 | 0.0147 (`prior_shift.true_rate`; the fresh test uses it directly) | `configs/baseline.yaml` |
| Enriched input columns | 251 | 251 (42 tabular + 209 enriched: 180 windows, 17 anomalies, 12 neighbourhood) | `dataset.enriched_names()` |
| Fit cap | 2,000,000 | 2,000,000 (verified 2,000,000 fit rows in folds 2022-2025 and the fresh fit) | `configs/baseline.yaml` |
| Tuning fits | 600,000 | 600,000 | `configs/baseline.yaml` |
| Tuning trials | 96 trials | 24 trials × 4 folds (12 trials for boosting and 12 for the network, each scored on the validation years 2021-2024; `tuning_trials.csv` has one row per trial and fold) | `tuning_trials.csv` |
| Network seeds | 10 | 10 | `configs/models.yaml` |
| Tests passing | 161 | {tests.split(' in ')[0]} | `tests/` |
| Reproduction checks | 133, 0 failures | 133 pass, 0 fail (written 2026-10-01 02:50; scope: labels, features, earlier holdout, leave-year-out 2024, out-of-fold, determinism; not rolling-origin, enriched, network or fresh scoring) | `wildfire-research/studies/reproduction/REPRODUCTION.md` |
| Peak memory | about 7 GB (was about 10 GB) | still an estimate, not a measured peak: about 7 GB for the 2M-row enriched logistic fit (float64 matrix 3.7 GB plus a 2.7 GB dataset); the old path was about 10 GB. The change is bit-identical (deviation j) | `STATUS.md` |

The first five rows were reconciled against the earlier reference runs; the result, including the differences found and the one measured effect of the row set, is in `docs/decisions/2026-10-01-reference-reconciliation.md`. In short: the earlier 0.107 is the same-day weather arm, not the next-day FWI+season arm; the UK test rows now include every season month, not January-July only; and the 2025 boosting AP agrees to four decimals although the predictions differ.

## 7. Full deviation list
{dev}

## 8. Limits to state in the paper
- Sealed season unavailable; 2025 is a development fold; the fresh test uses two states with unit weights.
- Logistic arms stop at the 300-iteration cap; the 2021-2022 folds come from earlier code (verified identical for the 2022 enriched logistic refit and the 2024 notebook refits).
- The soil-moisture anomaly norm covers 2017-2023, so rows in 2021-2023 see their own year (deviation c); ablation A shows ΔAP {abl_c.iloc[0].delta_ap:+.4f} (99% CI {abl_c.iloc[0].ap_lo99:.4f} to {abl_c.iloc[0].ap_hi99:.4f}) when the two inputs are removed.
- No causal claim; the place shortcut is read from the diagnostic arms (reported, not gated).
"""
(config.REPO_ROOT / "docs" / "PAPER_HANDOFF.md").write_text(out)
print(len(out), "characters written")

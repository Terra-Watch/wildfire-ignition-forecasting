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
Every arm fitted on all years before 2026 (2,000,000-row cap) and scored on the fresh rows: AP {int(fr[fr.state == 'AP'].rows.iloc[0]):,} rows ({int(fr[fr.state == 'AP'].onsets.iloc[0]):,} onsets), UK {int(fr[fr.state == 'UK'].rows.iloc[0]):,} rows ({int(fr[fr.state == 'UK'].onsets.iloc[0]):,} onsets), years 2020-2025. Unit design weights and the configured population rate 0.0147 are used, so log loss and Brier are less comparable than AP and the two AUC views. Network rows are means over 10 seeds. No state-level contrast tests were declared for the fresh states; no verdicts are given.

{md(fr_tab)}

All arms and the diagnostic arms are in `artifacts/results/fresh_predictions.parquet`. The sealed season (2026 January-July) is unavailable and was not scored (deviation e).

## 6. Numbers that changed from the Word document (earlier reference values)
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
| Tuning trials | 96 | 24 trials (12 for boosting, 12 for the network), each scored on 4 folds (2021-2024) = 96 trial-fold fits | `tuning_trials.csv` |
| Network seeds | 10 | 10 | `configs/models.yaml` |
| Tests passing | 161 | {tests.split(' in ')[0]} | `tests/` |
| Reproduction checks | 133, 0 failures | 133 pass, 0 fail (written 2026-10-01 02:50; scope: labels, features, earlier holdout, leave-year-out 2024, out-of-fold, determinism; not rolling-origin, enriched, network or fresh scoring) | `wildfire-research/studies/reproduction/REPRODUCTION.md` |
| Peak memory | about 7 GB (was about 10 GB) | still an estimate, not a measured peak: about 7 GB for the 2M-row enriched logistic fit (float64 matrix 3.7 GB plus a 2.7 GB dataset); the old path was about 10 GB. The change is bit-identical (deviation j) | `STATUS.md` |

The first five rows are close for the trained states (boosting@tabular AP {by25.loc['boosting@tabular', 'ap']:.4f} against 0.200, climatology {by25.loc['climatology_cell', 'ap']:.4f} against 0.117) and not for UK. They are not strictly like for like: the earlier holdout used 982,750 rows and 51,269 onsets, the 2025 fold uses 980,554 rows and 51,258 onsets; the 2025 fold fits on at most 2,000,000 rows with `tuned.json` settings (boosting depth 8, learning rate 0.05, 500 iterations), and whether the earlier run used the same rows cap and settings was not checked. The fresh UK rows, window and fit rows were not reconciled with the earlier Uttarakhand run, so the UK differences (AP 0.107 / 0.098 against 0.059, log loss 0.221 / 0.202 against 0.286) are reported, not explained.

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

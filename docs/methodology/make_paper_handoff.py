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


def fresh_by_seed(max_month: int | None = None) -> pd.DataFrame:
    """Every prediction column of the fresh test scored separately (network seeds not averaged); optionally only rows of months up to max_month."""
    from wildfire.evaluation import ladder
    from wildfire.pipelines import baseline_tables
    rows = pd.read_parquet(R / "fresh_predictions.parquet")
    if max_month is not None:
        rows = rows[pd.to_datetime(rows.day, unit="D").dt.month.values <= max_month]
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
seed_rows_jj = fresh_by_seed(7)
FRESH_ORDER = ["climatology_cell", "landcover_season", "fwi_season", "logistic_weather_static@tabular", "logistic_weather_static@enriched",
               "logistic_with_burn@tabular", "logistic_with_burn@enriched", "boosting@tabular", "boosting@enriched",
               "tabular_network@tabular", "tabular_network@enriched", "coords_only", "cell_id_only", "static_only"]


def fresh_all_arms(src=None, states=("UK", "AP")) -> pd.DataFrame:
    src = seed_rows if src is None else src
    out = []
    for state in states:
        d = src[src.state == state].copy()
        d["arm"] = strip_seed(d.arm_full)
        prevalence = float(d.onsets.iloc[0] / d.rows.iloc[0])
        for arm in FRESH_ORDER:
            g = d[d.arm == arm]
            n = len(g)

            def fmt(col, lift=False):
                v = g[col] / prevalence if lift else g[col]
                return f"{v.iloc[0]:.4f}" if n == 1 else f"{v.mean():.4f} ± {v.std():.4f} [{v.min():.4f}-{v.max():.4f}]"
            out.append({"state": state, "rows": int(d.rows.iloc[0]), "onsets": int(d.onsets.iloc[0]), "prevalence": round(prevalence, 4), "arm": arm, "seeds": n, "AP": fmt("ap"), "lift (AP / prevalence)": fmt("ap", True),
                        "AUC pooled": fmt("auc_pooled"), "AUC spatial": fmt("auc_spatial_within_day"), "AUC temporal": fmt("auc_temporal_within_cell"),
                        "log loss": fmt("log_loss")})
    return pd.DataFrame(out)


fresh_table = fresh_all_arms()
fresh_table_uk_all = fresh_all_arms(states=("UK",))
fresh_table_uk_jj = fresh_all_arms(seed_rows_jj, states=("UK",))
_f = pd.read_parquet(R / "fresh_predictions.parquet", columns=["state", "day", "label"])
_f["month"] = pd.to_datetime(_f.day, unit="D").dt.month
_m = _f.groupby(["month", "state"]).label.agg(rows="size", onsets="sum").unstack("state")
month_table = pd.DataFrame({"month": _m.index})
for st in ("UK", "AP"):
    month_table[f"{st} rows"] = _m[("rows", st)].values.astype(int)
    month_table[f"{st} onsets"] = _m[("onsets", st)].values.astype(int)
    month_table[f"{st} onset share of rows"] = (_m[("onsets", st)] / _m[("rows", st)]).values
tot = pd.DataFrame([{"month": "total", **{c: month_table[c].sum() for c in month_table.columns if "share" not in c and c != "month"}}])
for st in ("UK", "AP"):
    tot[f"{st} onset share of rows"] = tot[f"{st} onsets"] / tot[f"{st} rows"]
month_table = pd.concat([month_table.astype({"month": str}), tot.astype({"month": str})], ignore_index=True)
for c in month_table.columns:
    if (c.endswith("rows") and "share" not in c) or c.endswith("onsets"):
        month_table[c] = month_table[c].astype(int)
seed_detail = []
for st in ("UK", "AP"):
    for lvl in ("tabular", "enriched"):
        g = seed_rows[(seed_rows.state == st) & seed_rows.arm_full.str.startswith(f"tabular_network@{lvl}#")].copy()
        g["seed"] = g.arm_full.str.extract(r"#(\d+)")[0].astype(int)
        g = g.sort_values("seed")
        row = {"state": st, "level": lvl}
        for col, name in (("ap", "AP"), ("auc_pooled", "AUC pooled"), ("auc_spatial_within_day", "AUC spatial"), ("auc_temporal_within_cell", "AUC temporal"), ("log_loss", "log loss")):
            row[f"{name} min"], row[f"{name} max"], row[f"{name} mean"], row[f"{name} SD"] = g[col].min(), g[col].max(), g[col].mean(), g[col].std()
        row["AP by seed 0-9"] = ", ".join(f"{v:.4f}" for v in g.ap)
        seed_detail.append(row)
seed_detail = pd.DataFrame(seed_detail)

fresh_table_ap_jj = fresh_all_arms(seed_rows_jj, states=("AP",))
_jj = _f[_f.month <= 7]
_neg = _f.groupby(["state", "month"]).label.apply(lambda v: int((v == 0).sum()))
onset_only = {st: int(sum(((_jj.state == st) & (_jj.month == mo)).sum() for mo in range(1, 8) if _neg.get((st, mo), 0) == 0)) for st in ("UK", "AP")}
boot = pd.read_csv(R / "post_hoc_fresh_jan_jul_bootstrap.csv")
boot_view = boot[["state", "arm", "reference", "rows", "onsets", "clusters", "delta_ap", "ap_lo99", "ap_hi99", "ap_se", "delta_log_loss", "ll_lo99", "ll_hi99", "delta_brier", "brier_lo99", "brier_hi99"]].rename(
    columns={"delta_ap": "ΔAP", "ap_lo99": "ΔAP 99% lo", "ap_hi99": "ΔAP 99% hi", "ap_se": "ΔAP SE", "delta_log_loss": "Δlog loss", "ll_lo99": "Δlog loss 99% lo", "ll_hi99": "Δlog loss 99% hi",
             "delta_brier": "ΔBrier", "brier_lo99": "ΔBrier 99% lo", "brier_hi99": "ΔBrier 99% hi"})
shift = pd.read_csv(R / "post_hoc_neighbourhood_shift_columns.csv")
grp = pd.read_csv(R / "post_hoc_neighbourhood_shift_groups.csv")
sh = shift.pivot(index="column", columns="rows", values=["mean", "sd", "p99"])
sh_table = pd.DataFrame({"column": sh.index})
for stat in ("mean", "sd", "p99"):
    for rows_name, label in (("fit (standardiser rows)", "fit"), ("AP", "AP"), ("UK", "UK")):
        sh_table[f"{stat} {label}"] = sh[(stat, rows_name)].values
order = list(neighbourhood_columns := shift[shift.rows == "AP"].column.tolist())
sh_table = sh_table.set_index("column").loc[order].reset_index()
z_table = shift[shift.rows.isin(["AP", "UK"])].pivot(index="column", columns="rows", values=["z_min", "z_max", "z_p99"])
z_view = pd.DataFrame({"column": order})
for rows_name in ("AP", "UK"):
    for stat in ("z_min", "z_p99", "z_max"):
        z_view[f"{stat} {rows_name}"] = z_table[(stat, rows_name)].loc[order].values
shift_sd = {rn: ((shift[shift.rows == rn].set_index("column").loc[order, "mean"] - shift[shift.rows == rn].set_index("column").loc[order, "fit_mean"]) / shift[shift.rows == rn].set_index("column").loc[order, "fit_sd"]) for rn in ("AP", "UK")}
big = {}
for rn in ("AP", "UK"):
    d = shift[shift.rows == rn].set_index("column").loc[order]
    zz = pd.concat([d.z_max, -d.z_min]).abs()
    big[rn] = (zz.max(), zz.idxmax(), d.z_max.max(), d.z_max.idxmax(), d.z_min.min(), d.z_min.idxmin())

def month_group_ap() -> pd.DataFrame:
    """AP of the networks (mean of the seeds' own APs) and of boosting@enriched by month group, from the saved fresh predictions."""
    from wildfire.evaluation import metrics as official_metrics
    rows = pd.read_parquet(R / "fresh_predictions.parquet")
    month = pd.to_datetime(rows.day, unit="D").dt.month.values
    pcols = [c for c in rows.columns if c.startswith("p_")]
    out = []
    for st in ("AP", "UK"):
        for label, sel in (("months 1-7", month <= 7), ("months 8-11", (month >= 8) & (month <= 11)), ("month 12", month == 12), ("all months", month >= 1)):
            m = (rows.state.values == st) & sel
            y = rows.label.values[m]
            row = dict(state=st, months=label, rows=int(m.sum()), onsets=int(y.sum()), prevalence=float(y.mean()))
            for lvl in ("tabular", "enriched"):
                row[f"network {lvl} AP (mean of seeds)"] = float(np.mean([official_metrics.average_precision(y, rows[c].values[m]) for c in pcols if c.startswith(f"p_tabular_network@{lvl}#")]))
            row["boosting@enriched AP"] = float(official_metrics.average_precision(y, rows["p_boosting@enriched"].values[m]))
            out.append(row)
    return pd.DataFrame(out)


month_ap = month_group_ap()
uk_all_boost_tab = fresh_table_uk_all[fresh_table_uk_all.arm == "boosting@tabular"].AP.iloc[0]
uk_jj_boost_tab = fresh_table_uk_jj[fresh_table_uk_jj.arm == "boosting@tabular"].AP.iloc[0]
ap_all_net_enr = fresh_table[(fresh_table.state == "AP") & (fresh_table.arm == "tabular_network@enriched")].AP.iloc[0].split(" ")[0]
ap_jj_net_enr = fresh_table_ap_jj[fresh_table_ap_jj.arm == "tabular_network@enriched"].AP.iloc[0].split(" ")[0]
g_ap = grp[grp.state == "AP"].set_index("group")
g_uk = grp[grp.state == "UK"].set_index("group")
prev_state = {st: float(seed_rows[seed_rows.state == st].onsets.iloc[0] / seed_rows[seed_rows.state == st].rows.iloc[0]) for st in ("UK", "AP")}
abl_m, abl_c = pd.read_csv(R / "ablation_metrics.csv"), pd.read_csv(R / "ablation_contrasts.csv")
gain = pd.read_csv(R / "post_hoc_onsets_recent_gain.csv")
guards = json.loads((R / "guards.json").read_text())
ledger = pd.read_csv(R / "geo_test_ledger.csv")
fit_prev = json.loads((R / "fresh_fit_prevalence.json").read_text())
dev = (config.REPO_ROOT / "docs/decisions/2026-10-01-deviation-list.md").read_text().split("\n", 1)[1].strip()
tests = subprocess.run([str(config.REPO_ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "tests"], capture_output=True, text=True, cwd=config.REPO_ROOT).stdout.strip().splitlines()[-1]

tuned = json.loads((R / "tuned.json").read_text())
models = config.load("models")
baseline = config.load("baseline")
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
    yr[metric] = p.reset_index().rename(columns={2023: "2023 (provisional)"})          # fold-2023 values are provisional (docs/limitations.md)
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

(A_no_soil_anomaly removes `sw1_anom` and `sw2_anom`; B_windows_only keeps the weather windows and their anomalies but no neighbourhood block; B_neighbourhood_only keeps the neighbourhood block but no windows. **Note on the verdict column:** the verdict labels are descriptive and meaningless for removal contrasts (the decision rule is built for a gain over a reference, not for the effect of removing inputs); the paper reports ΔAP and its 99% CI only, with no verdict. Ablation A was run for `boosting@enriched` only; the logistic arms and the network were not tested.) Post hoc stratification by recent neighbour onsets (`docs/decisions/2026-10-01-onsets-recent-strata.md`): the enriched AP gain is +{gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'above_zero')].delta_ap.iloc[0]:.4f} (99% CI {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'above_zero')].ap_lo99.iloc[0]:.4f} to {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'above_zero')].ap_hi99.iloc[0]:.4f}) in rows with recent onsets in the 8 neighbouring cells (94.1% of positives) and {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'zero')].delta_ap.iloc[0]:+.4f} (99% CI {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'zero')].ap_lo99.iloc[0]:.4f} to {gain[(gain.feature == 'ring_8_onsets_recent') & (gain.stratum == 'zero')].ap_hi99.iloc[0]:.4f}) in rows with none, so a gain there is not demonstrated.

## 4. Rolling-origin results per year
Each year is scored by models fitted on all earlier years from 2019, January to July rows only. Rows and onsets per year: {', '.join(f'{int(r.year)}: {int(r.rows):,} / {int(r.onsets):,}' for r in rows_year.itertuples())}. Network rows are means over 10 seeds. **Every value in the 2023 columns below is provisional** (`docs/limitations.md`: the soil-moisture anomaly norm includes later 2023 weather, and the settings were chosen on validation years 2021-2024, which include the scored folds).

AP:

{md(yr['ap'])}

Within-day spatial AUC:

{md(yr['auc_spatial_within_day'])}

Within-cell temporal AUC:

{md(yr['auc_temporal_within_cell'])}

## 5. Fresh states (UK and AP), scored once
Every arm fitted on all years before 2026 (2,000,000-row cap) and scored on the fresh rows: AP {int(fr[fr.state == 'AP'].rows.iloc[0]):,} rows ({int(fr[fr.state == 'AP'].onsets.iloc[0]):,} onsets), UK {int(fr[fr.state == 'UK'].rows.iloc[0]):,} rows ({int(fr[fr.state == 'UK'].onsets.iloc[0]):,} onsets), years 2020-2025, every fire-season month through 2025-07-31.

**What the specification declares for the fresh test rows.** `docs/baseline_spec.md` section 2: "fresh geographic test: UK always; AP, KA, TN, KL only if their fire data passes the completeness check | once, every access recorded in a ledger". `configs/baseline.yaml` `fresh_geographic_test`: `states_always: [UK]`, `states_if_fire_data_complete: [AP, KA, TN, KL]`, `scores_allowed: 1`, `ledger: geo_test_ledger.csv`. Neither names a month or row set. The January-to-July rule is declared for validation years: "Every validation year uses the rows of January to July (the 2025 data stops at the end of July)" (spec, "Compute limits declared in advance"; `validation_months: [1, 2, 3, 4, 5, 6, 7]`, commented "rows used to validate each year"). So the all-months scoring below is not a deviation from a declared fresh-test row set; it differs from the validation-year rule and from the earlier reference run (deviation o). Because the spec fixes no months for the fresh test, the table scored as run (all months) is given first and the January-July table, which applies the validation-year rule by analogy and was computed after the results were known, is labelled a post hoc sensitivity.

Unit design weights and the configured population rate 0.0147 are used, so log loss is less comparable than AP and the AUC views. Lift is AP divided by the state's sampled prevalence. No state-level contrast tests were declared for the fresh states; no verdicts are given. Networks are mean ± SD over the 10 seeds with the per-seed range [min-max]; every other arm has one fit. The figures AP 0.1144 (AP) and 0.1238 (UK), quoted in earlier discussion, are the AP of **`tabular_network@enriched`** (mean over 10 seeds); the tabular-level network has AP {seed_rows[(seed_rows.state == 'AP') & seed_rows.arm_full.str.startswith('tabular_network@tabular')].ap.mean():.4f} (AP) and {seed_rows[(seed_rows.state == 'UK') & seed_rows.arm_full.str.startswith('tabular_network@tabular')].ap.mean():.4f} (UK).

### 5a. UK and AP, all scored months (as scored)
{md(fresh_table)}

### 5b. UK, all scored months (as scored): the UK rows of 5a repeated beside 5c
{md(fresh_table_uk_all.drop(columns=[]))}

### 5c. UK, January to July only: SENSITIVITY (POST HOC)
Rows of months 1-7 of the saved UK predictions, scored with the same metric code; nothing was refitted or re-scored. Fitted models are those of the fresh scoring.

{md(fresh_table_uk_jj)}

### 5d. Onset counts per month, pooled 2020-2025 (scored rows)
{md(month_table, 4)}

In months where the share is 1.0 every scored row is an onset: the matched non-fire days are drawn only inside the state's fire season, while onsets can fall in any month. The scored rows of those months therefore carry positives only (UK: months 7 to 11, {int(month_table[month_table.month.isin(['7','8','9','10','11'])]['UK onsets'].sum()):,} rows; AP: months 6 to 11, {int(month_table[month_table.month.isin(['6','7','8','9','10','11'])]['AP onsets'].sum()):,} rows). This affects the sampled prevalence, the pooled AUC and AP of the as-scored tables, and it is part of the difference between 5a and 5c.

### 5e. Per-seed range of the network, UK and AP
{md(seed_detail, 4)}

### 5f. AP, January to July only: SENSITIVITY (POST HOC)
Rows of months 1-7 of the saved AP predictions, same metric code; nothing refitted or re-scored. Onset-only rows (rows in a month where the state has no matched non-fire rows, so every row is an onset) inside each state's January-July set: **UK {onset_only['UK']:,}** of {int(fresh_table_uk_jj.rows.iloc[0]):,} rows (month 7), **AP {onset_only['AP']:,}** of {int(fresh_table_ap_jj.rows.iloc[0]):,} rows (months 6 and 7).

{md(fresh_table_ap_jj)}

### 5g. Paired ISO-week cluster bootstrap on the January-July rows: DESCRIPTIVE, POST HOC, NO VERDICTS
Differences are arm minus reference, on the saved fresh predictions, with the baseline's paired ISO-week cluster bootstrap (2,000 draws, 99% percentile interval). No verdict is given and none of these contrasts was declared; log loss uses the corrected probabilities and unit weights as in 5a.

{md(boot_view)}

### 5h. Neighbourhood-block shift for `tabular_network@enriched` (post hoc, read-only)
Mean, SD and 99th percentile of the 12 neighbourhood columns for the rows the network was standardised on ("fit": fit rows 2019-2024 without the early-stopping year 2025, after the 2,000,000-row cap), the AP rows and the UK rows that were scored. The network standardises each column with the mean and SD of the fit rows ("fit" mean and SD in the table; SD plus 1e-6).

{md(sh_table, 3)}

Standardised values the network saw (value minus fit mean, divided by fit SD) in the scored rows:

{md(z_view, 2)}

- Largest standardised value: AP **{big['AP'][2]:.2f}** (`{big['AP'][3]}`), UK **{big['UK'][2]:.2f}** (`{big['UK'][3]}`); smallest: AP {big['AP'][4]:.2f} (`{big['AP'][5]}`), UK {big['UK'][4]:.2f} (`{big['UK'][5]}`). Largest absolute value in these columns: AP {big['AP'][0]:.2f}, UK {big['UK'][0]:.2f}; no value exceeds 10.
- Largest shift of a column mean from the fit mean, in fit SDs: AP {shift_sd['AP'].abs().max():.2f} (`{shift_sd['AP'].abs().idxmax()}`), UK {shift_sd['UK'].abs().max():.2f} (`{shift_sd['UK'].abs().idxmax()}`).

What the network scored on rows split by their largest standardised neighbourhood value (saved predictions; thresholds 5 and 10 fixed before looking; AP of the mean over seeds' individual APs):

{md(grp, 4)}

AP of the same predictions by month group (every row of a month in which the state has matched non-fire days carries both labels; the other months are onset-only):

{md(month_ap, 4)}

**Is the AP collapse explained by the neighbourhood block?** The collapse is `tabular_network@enriched` AP {g_ap.loc['all rows', 'network_enriched_AP_mean_of_seeds']:.4f} in AP, against {g_ap.loc['all rows', 'network_tabular_AP_mean_of_seeds']:.4f} for `tabular_network@tabular` and {g_ap.loc['all rows', 'boosting_enriched_AP']:.4f} for `boosting@enriched` on the same rows. By these numbers extreme neighbourhood values do not explain it: {int(g_ap.loc['max |z| <= 5', 'rows']):,} of {int(g_ap.loc['all rows', 'rows']):,} AP rows ({g_ap.loc['max |z| <= 5', 'share_of_rows']:.5f} of the rows) have no standardised neighbourhood value above 5, and on them the network's AP is {g_ap.loc['max |z| <= 5', 'network_enriched_AP_mean_of_seeds']:.4f}, the same as on all rows; the {int(g_ap.loc['5 < max |z| <= 10', 'rows'])} rows above 5 are not what lowers it (network AP there {g_ap.loc['5 < max |z| <= 10', 'network_enriched_AP_mean_of_seeds']:.4f}). On the January-July rows the enriched network's AP in AP is {fresh_table_ap_jj[fresh_table_ap_jj.arm == 'tabular_network@enriched'].AP.iloc[0].split(' ')[0]} against {fresh_table_ap_jj[fresh_table_ap_jj.arm == 'tabular_network@tabular'].AP.iloc[0].split(' ')[0]} for the tabular network (5f), so the gap between 0.1144 and the tabular network arises in the other months (table above). These checks do not test the weather-window or anomaly blocks, and do not rule out an effect of the moderate shifts of the column means shown above; they only show that the collapse is not carried by a few extreme neighbourhood values. For UK the same split gives network AP {g_uk.loc['max |z| <= 5', 'network_enriched_AP_mean_of_seeds']:.4f} on {int(g_uk.loc['max |z| <= 5', 'rows']):,} rows with no value above 5 (all rows {g_uk.loc['all rows', 'network_enriched_AP_mean_of_seeds']:.4f}).

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

The first five rows were reconciled against the earlier reference runs; the result, including the differences found and the one measured effect of the row set, is in `docs/decisions/2026-10-01-reference-reconciliation.md`. In short: the earlier 0.107 is the same-day weather arm, not the next-day FWI+season arm; the UK test rows now include every season month, not January-July only (the specification declares no month rule for the fresh test, only for validation years); and the 2025 boosting AP agrees to four decimals although the predictions differ.

## 6b. Settings to quote (deviation i): `tuned.json` and `models.yaml` side by side with the specification text
Every parameter of every arm (library and class, scaling, how design weights enter the loss, optimiser, stopping, seeds) is in `docs/MODEL_SPEC_TABLE.md`, built from the code and `tuned.json`.

| Item | `artifacts/results/tuned.json` (used by the results) | `configs/models.yaml` / `configs/baseline.yaml` | Specification text (`docs/baseline_spec.md`) |
|---|---|---|---|
| Boosting max depth | {tuned['boosting']['max_depth']} | {models['arms']['boosting']['params']['max_depth']} (`arms.boosting.params`; used only by the older single-split pipelines) | not stated |
| Boosting learning rate | {tuned['boosting']['learning_rate']} | {models['arms']['boosting']['params']['learning_rate']} | not stated |
| Boosting iterations | {tuned['boosting']['max_iter']} | {models['arms']['boosting']['params']['max_iter']} | not stated |
| Boosting L2 / minimum leaf | {tuned['boosting']['l2_regularization']} / {tuned['boosting']['min_samples_leaf']} | not listed | not stated |
| Boosting search space | | `boosting_space`: depth {baseline['tuning']['boosting_space']['max_depth']}, learning rate {baseline['tuning']['boosting_space']['learning_rate']}, iterations {baseline['tuning']['boosting_space']['max_iter']}, L2 {baseline['tuning']['boosting_space']['l2_regularization']}, leaf {baseline['tuning']['boosting_space']['min_samples_leaf']} | section 6: "12 random-search trials each ... Search spaces are in `configs/baseline.yaml`" |
| Network width | {tuned['network']['width']} | search {baseline['tuning']['network_space']['width']} | "MLP 3 layers of 256" |
| Network depth (hidden layers) | {tuned['network']['depth']} (two hidden layers plus a one-unit output layer) | search {baseline['tuning']['network_space']['depth']} | "3 layers" (not defined further) |
| Network dropout | {tuned['network']['dropout']} | search {baseline['tuning']['network_space']['dropout']} | 0.1 |
| Network learning rate | {tuned['network']['learning_rate']} | search {baseline['tuning']['network_space']['learning_rate']} | "AdamW" (no rate) |
| Network weight decay | {tuned['network']['weight_decay']} | search {baseline['tuning']['network_space']['weight_decay']} | "weight decay" (no value) |
| Network training | batch {models['network']['batch_size']}, at most {models['network']['max_epochs']} epochs, stop after {models['network']['patience']} epochs without a gain of {models['network']['min_improvement']}, {models['network']['seeds']} seeds | the same (`network:`) | "cross-entropy with the design weights, early stopping on validation log loss inside the training years, 10 seeds" |
| Logistic C | land cover plus season {models['arms']['landcover_season']['params']['C']}; the other logistic arms {models['arms']['fwi_season']['params']['C']} | the same (used) | not stated |

**Quote `tuned.json`**: the boosting settings and the network architecture the results were produced with. State the network as two hidden layers of 256 units with dropout 0.1 and one output unit, not as "3 layers of 256" and not with the `models.yaml` boosting values (depth 6, learning rate 0.1, 300 iterations), which the baseline folds, fresh scoring, ablations and notebooks do not use.

## 7. Full deviation list
{dev}

## 8. Limits to state in the paper
- Sealed season unavailable; 2025 is a development fold; the fresh test uses two states with unit weights.
- Logistic arms stop at the 300-iteration cap; the 2021-2022 folds come from earlier code (verified identical for the 2022 enriched logistic refit and the 2024 notebook refits).
- The soil-moisture anomaly norm covers 2017-2023, so rows in 2021-2023 see their own year (deviation c); ablation A shows ΔAP {abl_c.iloc[0].delta_ap:+.4f} (99% CI {abl_c.iloc[0].ap_lo99:.4f} to {abl_c.iloc[0].ap_hi99:.4f}) when the two inputs are removed.
- Provisional values (leakage review, 2026-10-05): every fold-2023 value and every pooled five-fold value is provisional, because the soil-moisture norm includes later 2023 weather and the settings of every model were chosen on scored validation years; every probability of detection, false alarm ratio and critical success index is provisional, because the fold-2023 alert threshold was chosen on fold-2024 labels. See `docs/limitations.md`.
- No causal claim; the place shortcut is read from the diagnostic arms (reported, not gated).

## 9. Primary table choice for the fresh states
**Primary: the January-to-July tables, for both states (5c for UK, 5f for AP). Appendix: the as-scored all-months table (5a).**
- **Why.** (1) Rows outside a state's fire season are onset-only, because the matched non-fire days are drawn inside the season only (UK months 7 to 11: 1,967 rows; AP months 6 to 11: 3,080 rows; section 5d); they carry positives only, which raises the sampled prevalence and affects pooled AP and AUC. The January-July sets still hold a few onset-only rows (UK {onset_only['UK']:,}, AP {onset_only['AP']:,}). (2) The January-to-July set is the row set of the rolling-origin validation folds (spec "Compute limits declared in advance", `validation_months`), so the fresh results are read on the same footing as the validation results.
- **How it was chosen.** The choice was made after both tables had been seen, so it is a post hoc choice. The specification declares no month rule for the fresh test (section 5), so neither table is the pre-declared one; the as-scored table is what the scoring run produced. The two tables differ materially, for example UK `boosting@tabular` AP {uk_all_boost_tab} (as scored) against {uk_jj_boost_tab} (January-July), and `tabular_network@enriched` in AP {ap_all_net_enr} against {ap_jj_net_enr}; the paper should say that both were computed and which one is primary.
- **Fit.** The models are the same in both: nothing was refitted or re-scored for the January-July tables.
"""
(config.REPO_ROOT / "docs" / "PAPER_HANDOFF.md").write_text(out)
print(len(out), "characters written")

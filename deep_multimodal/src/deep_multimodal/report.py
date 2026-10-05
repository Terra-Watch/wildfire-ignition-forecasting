"""Writing the results table (markdown and CSV) from the output of `evaluation.evaluate`. All labels are plain words."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [("average_precision", "Average precision (pooled)", 4), ("average_precision_within_day", "Average precision, ranking cells within each day", 4),
          ("average_precision_within_cell", "Average precision, ranking days within each cell", 4), ("auroc", "AUROC (pooled)", 4), ("auroc_within_day", "AUROC, ranking cells within each day", 4),
          ("auroc_within_cell", "AUROC, ranking days within each cell", 4), ("precision_top1pct", "Precision, top 1% of cells per day", 4), ("recall_top1pct", "Recall, top 1% of cells per day", 4),
          ("lift_top1pct", "Lift, top 1% of cells per day", 2), ("precision_top5pct", "Precision, top 5% of cells per day", 4), ("recall_top5pct", "Recall, top 5% of cells per day", 4),
          ("lift_top5pct", "Lift, top 5% of cells per day", 2), ("precision_top10pct", "Precision, top 10% of cells per day", 4), ("brier", "Brier score (lower is better)", 5),
          ("brier_cell_climatology", "Brier score of the cell climatology (reference)", 5), ("brier_skill_score", "Brier skill score against the cell climatology", 4),
          ("log_loss", "Log loss (lower is better)", 4), ("calibration_error_ece", "Calibration error, ECE (lower is better)", 5),
          ("alert_threshold", "Alert threshold (probability, fixed on the other fold)", 4), ("alert_rate", "Share of cell-days alerted", 4),
          ("probability_of_detection", "Probability of detection", 4), ("false_alarm_ratio", "False alarm ratio (lower is better)", 4), ("critical_success_index", "Critical success index", 4),
          ("rows", "Rows", 0), ("onsets", "Fire onsets", 0), ("fire_rate_weighted", "Fire rate (population-weighted)", 4)]
STATE_NAMES = {"CG": "Chhattisgarh", "JH": "Jharkhand", "MH": "Maharashtra", "MP": "Madhya Pradesh", "OD": "Odisha", "TS": "Telangana"}
STATE_METRICS = [("average_precision", "Average precision", 4), ("auroc", "AUROC", 4), ("precision_top5pct", "Precision, top 5% per day", 4), ("recall_top5pct", "Recall, top 5% per day", 4),
                 ("lift_top5pct", "Lift, top 5% per day", 2), ("brier", "Brier score", 5), ("brier_skill_score", "Brier skill score", 4)]
MODEL_COLUMNS = {"baseline": "Frozen baseline", "deep": "Deep multimodal model"}
# Provisional values (docs/MODEL.md, Limitations): every fold-2023 value, and every probability of detection, false alarm ratio and critical success index.
# Only the markdown is labelled; the CSV files keep their column names and values.
PROVISIONAL_FOLDS = (2023,)
PROVISIONAL_METRICS = ("probability_of_detection", "false_alarm_ratio", "critical_success_index")
PROVISIONAL_NOTE = "Every value in this section is **provisional** (fold 2023; see Limitations in `docs/MODEL.md`)."


def fold_name(fold) -> str:
    return f"{fold} (provisional)" if int(fold) in PROVISIONAL_FOLDS else f"{fold}"


def fold_note(fold) -> str:
    return f"{PROVISIONAL_NOTE}\n\n" if int(fold) in PROVISIONAL_FOLDS else ""


def value(table, fold, rows_scored, model, metric):
    q = table[(table.fold == fold) & (table.rows_scored == rows_scored) & (table.model == model) & (table.metric == metric)]
    return float(q.value.iloc[0]) if len(q) else float("nan")


def interval_sentence(table, fold, rows_scored) -> str:
    def get(model, metric):
        q = table[(table.fold == fold) & (table.rows_scored == rows_scored) & (table.model == model) & (table.metric == metric)].iloc[0]
        return q.value, q.lower_95, q.upper_95
    b, d, diff = get("baseline", "average_precision_interval"), get("deep", "average_precision_interval"), get("deep minus baseline", "average_precision_difference")
    return (f"Average precision with 95% interval: baseline {b[0]:.4f} ({b[1]:.4f} to {b[2]:.4f}); deep model {d[0]:.4f} ({d[1]:.4f} to {d[2]:.4f}); "
            f"difference {diff[0]:+.4f} ({diff[1]:+.4f} to {diff[2]:+.4f}).")


def wide_block(table, fold, rows_scored) -> str:
    text = "| Metric | Frozen baseline | Deep multimodal model | Deep minus baseline |\n|---|---|---|---|\n"
    for key, label, places in LABELS:
        b, d = value(table, fold, rows_scored, "baseline", key), value(table, fold, rows_scored, "deep", key)
        shown = lambda x: f"{x:,.0f}" if places == 0 else f"{x:.{places}f}"
        label = f"{label} (provisional)" if key in PROVISIONAL_METRICS else label
        text += f"| {label} | {shown(b)} | {shown(d)} | " + ("" if places == 0 else f"{d - b:+.{places}f}") + " |\n"
    return text


def write_report(results: dict, out_dir: Path) -> None:
    """Write RESULTS_TABLE.md, results_table.csv, per_state.csv, quiet_stratum.csv and calibration_curve.csv under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, calibration = results["table"], results["calibration"]
    folds = sorted(table.fold.unique(), reverse=True)
    text = ["# Results table: frozen baseline against the deep multimodal model\n\n",
            "Folds 2024 and 2023: validation years the models were not trained on. Both models are scored on the same rows with the same labels and weights. Probabilities are corrected from the matched-negative sample "
            "back to the true fire rate and are not calibrated further. The top-1% and top-5% rows are chosen among the cells of each day. \"Deep minus baseline\" is positive where the deep model is higher, which is better "
            "for every metric except Brier score, log loss, calibration error and false alarm ratio. **Only average precision has a confidence interval** (exact paired bootstrap over week blocks); every other number is a point estimate. "
            "**Provisional values:** every fold-2023 value, and every probability of detection, false alarm ratio and critical success index on either fold, are provisional "
            "(see Limitations in `docs/MODEL.md`); they will be replaced when corrected numbers exist.\n"]
    for fold in folds:
        text.append(f"\n## Fold {fold_name(fold)}\n\n{fold_note(fold)}{interval_sentence(table, fold, 'all rows')}\n\n{wide_block(table, fold, 'all rows')}")
    for fold in folds:
        text.append(f"\n## Quiet stratum, fold {fold_name(fold)} (no neighbouring fire detection in the previous 7 days)\n\n{fold_note(fold)}{interval_sentence(table, fold, 'quiet stratum')} "
                    f"The alert threshold is the one fixed for the whole fold.\n\n{wide_block(table, fold, 'quiet stratum')}")
    state_rows = []
    for fold in folds:
        text.append(f"\n## Per state, fold {fold_name(fold)}\n\n{fold_note(fold)}Each state is ranked on its own rows.\n\n| State | Metric | Frozen baseline | Deep multimodal model | Deep minus baseline |\n|---|---|---|---|---|\n")
        for rows_scored in sorted(s for s in table.rows_scored.unique() if s.startswith("state ")):
            name = STATE_NAMES.get(rows_scored.split()[1], rows_scored.split()[1])
            for key, label, places in STATE_METRICS:
                b, d = value(table, fold, rows_scored, "baseline", key), value(table, fold, rows_scored, "deep", key)
                text[-1] += f"| {name} | {label} | {b:.{places}f} | {d:.{places}f} | {d - b:+.{places}f} |\n"
                state_rows.append(dict(fold=fold, state=name, metric=label, frozen_baseline=b, deep_model=d, deep_minus_baseline=d - b))
    text.append("\n## Calibration curves (all rows)\n\nTen equal-count groups of predicted probability; mean predicted probability against the observed fire rate.\n")
    for fold in folds:
        c = calibration[(calibration.fold == fold) & (calibration.rows_scored == "all rows")]
        b, d = c[c.model == "baseline"].reset_index(), c[c.model == "deep"].reset_index()
        text.append(f"\nFold {fold_name(fold)}\n\n{fold_note(fold)}| Group | Baseline: predicted | observed | Deep model: predicted | observed |\n|---|---|---|---|---|\n")
        for i in range(len(b)):
            text[-1] += f"| {i + 1} | {b.mean_prediction[i]:.5f} | {b.observed_rate[i]:.5f} | {d.mean_prediction[i]:.5f} | {d.observed_rate[i]:.5f} |\n"
    (out_dir / "RESULTS_TABLE.md").write_text("".join(text))
    blocks = []
    for fold in folds:
        for rows_scored, shown in (("all rows", "All rows"), ("quiet stratum", "Quiet stratum")):
            for model, name in MODEL_COLUMNS.items():
                row = {"fold": fold, "rows_scored": shown, "model": name}
                row.update({label: value(table, fold, rows_scored, model, key) for key, label, _ in LABELS})
                q = table[(table.fold == fold) & (table.rows_scored == rows_scored) & (table.model == model) & (table.metric == "average_precision_interval")].iloc[0]
                row["Average precision, 95% interval low"], row["Average precision, 95% interval high"] = q.lower_95, q.upper_95
                blocks.append(row)
    frame = pd.DataFrame(blocks)
    frame.to_csv(out_dir / "results_table.csv", index=False)
    frame[frame.rows_scored == "Quiet stratum"].to_csv(out_dir / "quiet_stratum.csv", index=False)
    pd.DataFrame(state_rows).to_csv(out_dir / "per_state.csv", index=False)
    calibration.assign(model=calibration.model.map(MODEL_COLUMNS)).to_csv(out_dir / "calibration_curve.csv", index=False)

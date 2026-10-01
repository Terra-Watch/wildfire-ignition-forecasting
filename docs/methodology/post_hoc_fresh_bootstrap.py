"""Post hoc, descriptive: paired ISO-week cluster bootstrap (99% interval, no verdict) of three differences on the January-July rows of
the saved fresh predictions, for UK and for AP. Same bootstrap code as the baseline contrasts (2,000 draws). Read-only: nothing is
refitted or re-scored. Run from the repository root with the .env variables set."""
import json

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import bootstrap, ladder

R = config.output_dir("results")
OUT = R / "post_hoc_fresh_jan_jul_bootstrap.csv"
CONTRASTS = [("logistic_with_burn@tabular", "boosting@tabular"), ("boosting@enriched", "fwi_season"), ("boosting@enriched", "logistic_with_burn@enriched")]


def main() -> None:
    rows = pd.read_parquet(R / "fresh_predictions.parquet")
    prev = json.loads((R / "fresh_fit_prevalence.json").read_text())
    month = pd.to_datetime(rows.day, unit="D").dt.month.values
    out = []
    for state in ("UK", "AP"):
        g = rows[(rows.state == state) & (month <= 7)].reset_index(drop=True)
        clusters = ladder.cluster_labels(g)
        for arm, ref in CONTRASTS:
            arms = {a: dict(score=g["p_" + a].values, prob=ladder.corrected_probabilities(g["p_" + a].values, g.pi_true.values, prev[a])) for a in (arm, ref)}
            o = bootstrap.paired_week_blocked_bootstrap(g.label.values, clusters, arms, ref, weights=g.weight.values)[arm]
            d, ll, br = o["delta_ap"], o["delta_logloss"], o["delta_brier"]
            out.append(dict(state=state, arm=arm, reference=ref, rows=len(g), onsets=int(g.label.sum()), clusters=int(len(np.unique(np.asarray(clusters).astype(str)))),
                            delta_ap=d["point"], ap_lo99=d["lo99"], ap_hi99=d["hi99"], ap_se=d["se"],
                            delta_log_loss=ll["point"], ll_lo99=ll["lo99"], ll_hi99=ll["hi99"], delta_brier=br["point"], brier_lo99=br["lo99"], brier_hi99=br["hi99"]))
    table = pd.DataFrame(out)
    table.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()

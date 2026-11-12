"""Which replayed days of 2025 to show as the demo archive, and how a typical day did.
Agreement of a day = F1 of "rated High or Extreme" against "a fire was detected in the area" (precision: share of High/Extreme areas that had a fire; recall: share of areas with a fire that were rated High/Extreme).
Needs every day replayed first:  .venv/bin/python scripts/15_replay_area_forecast.py --out OUT --dates 2025-01-02:2025-12-30
Run:  .venv/bin/python scripts/16_archive_pick.py OUT/area_forecast [min_fire_areas=40]
Prints the best days and the medians over the active days (the `typical` block written into the archive's index.json). The shown days are chosen for how well they agree, so the page says what a typical day looks like."""
import glob, json, statistics as st, sys

root = sys.argv[1]; floor = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rows = []
for f in sorted(glob.glob(f"{root}/2*.json")):
    d = json.load(open(f)); a = d["areas"]
    if "fire_detected" not in a[0]: continue
    hi = [x for x in a if x["level"] in ("high", "extreme")]; fire = [x for x in a if x["fire_detected"]]; tp = sum(1 for x in hi if x["fire_detected"])
    p = tp / len(hi) if hi else 0.0; r = tp / len(fire) if fire else 0.0
    rows.append(dict(date=d["date"], fire=len(fire), rated=len(hi), p=p, r=r, f1=2 * p * r / (p + r) if p + r else 0.0))
act = [x for x in rows if x["fire"] >= floor]
print(f"{len(rows)} days; {len(act)} active days (at least {floor} areas with a fire)")
print("typical:", json.dumps({"active_days": len(act), "min_fire_areas": floor, "high_or_extreme_had_fire": round(st.median(x["p"] for x in act), 2), "fire_rated_high_or_extreme": round(st.median(x["r"] for x in act), 2)}))
for x in sorted(act, key=lambda x: -x["f1"])[:12]: print(f"{x['date']}  F1 {x['f1']:.2f}  fire areas {x['fire']}  rated {x['rated']}  precision {x['p']:.2f}  recall {x['r']:.2f}")

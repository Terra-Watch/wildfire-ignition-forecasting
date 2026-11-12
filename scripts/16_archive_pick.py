"""Which replayed days of 2025 to show as the demo archive, and how a typical day did.
Agreement of a day = F1 of "rated High or Extreme" against "a fire was detected in the area" (precision: share of High/Extreme areas that had a fire; recall: share of areas with a fire that were rated High/Extreme).
Needs every day replayed first:  .venv/bin/python scripts/15_replay_area_forecast.py --out OUT --dates 2025-01-02:2025-12-30
Run:  .venv/bin/python scripts/16_archive_pick.py OUT/area_forecast [min_fire_areas=40]
Scenes (one state on one day, small enough to read on a map):  .venv/bin/python scripts/16_archive_pick.py OUT/area_forecast scenes
  keeps the scenes with 8 to 45 areas rated High or Extreme and 8 to 45 areas with a fire, at least 0.7 precision and recall, and at least one miss and one false alarm, best first.
Prints the best days and the medians over the active days (the `typical` block written into the archive's index.json). The shown days are chosen for how well they agree, so the page says what a typical day looks like."""
import glob, json, statistics as st, sys

root = sys.argv[1]; scenes = len(sys.argv) > 2 and sys.argv[2] == 'scenes'; floor = int(sys.argv[2]) if len(sys.argv) > 2 and not scenes else 40
if scenes:
    import collections
    out = []
    for f in sorted(glob.glob(f"{root}/2*.json")):
        d = json.load(open(f)); a = d["areas"]
        if not any("fire_detected" in x for x in a): continue
        by = collections.defaultdict(list)
        for x in a: by[x["state"]].append(x)
        for state, L in by.items():
            hi = [x for x in L if x["level"] in ("high", "extreme")]; fire = [x for x in L if x["fire_detected"]]; tp = [x for x in hi if x["fire_detected"]]
            if len(L) < 12 or not (8 <= len(fire) <= 45 and 8 <= len(hi) <= 45): continue
            p, r = len(tp) / len(hi), len(tp) / len(fire)
            if p < 0.7 or r < 0.7 or len(fire) == len(tp) or len(hi) == len(tp): continue
            out.append((2 * p * r / (p + r), d["date"], state, len(L), len(fire), len(hi), len(tp)))
    for x in sorted(out, reverse=True)[:40]: print(f"{x[1]} {x[2]}  areas {x[3]}  fire {x[4]}  rated {x[5]}  hit {x[6]}  missed {x[4]-x[6]}  false alarms {x[5]-x[6]}  F1 {x[0]:.2f}")
    sys.exit(0)
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

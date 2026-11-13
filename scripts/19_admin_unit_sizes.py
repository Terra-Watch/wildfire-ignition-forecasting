"""How big are the units a forest department actually works in? Areas of beats, ranges and divisions from the Bharatmaps reserved-forest polygons (FSI/SOI), in km2.

Each polygon (a compartment or block) carries its division, range and beat names where the state filled them in (27 states have named divisions). Polygon areas are computed in an equal-area
projection (EPSG:6933); the area of a unit is the sum of its reserved-forest polygons. This is the reserved-forest area only: the jurisdiction of a real division or range also contains
other forest types and non-forest land, so every figure here is a lower bound on the size of the unit.
Usage: python scripts/19_admin_unit_sizes.py --rfa ../terrawatch-datasets/dataset/fsi_boundaries/Bharatmaps_RFA.parquet --out results/admin_unit_sizes.csv
"""
import argparse
import numpy as np, pandas as pd, pyarrow.parquet as pq, shapely
from pyproj import Transformer

ap = argparse.ArgumentParser(); ap.add_argument("--rfa", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
tr = Transformer.from_crs(4326, 6933, always_xy=True)
t = pq.ParquetFile(a.rfa).read(columns=["division", "range", "beat_name", "st_name", "geometry"]).to_pandas()
g = shapely.from_wkb(t.geometry.values); t = t.drop(columns="geometry")
fn = lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1]))
t["km2"] = np.concatenate([shapely.area(shapely.transform(g[i:i + 20000], fn)) for i in range(0, len(g), 20000)]) / 1e6
for c in ("division", "range", "beat_name"): t[c] = t[c].fillna("").astype(str).str.strip().str.upper()
rows = []
for name, key, mask in (("division", ["st_name", "division"], t.division != ""), ("range", ["st_name", "division", "range"], (t.division != "") & (t.range != "")),
                        ("beat", ["st_name", "division", "range", "beat_name"], (t.division != "") & (t.range != "") & (t.beat_name != ""))):
    s = t[mask].groupby(key).km2.sum()
    rows.append({"unit": name, "n": len(s), "median_km2": round(s.median(), 1), "p25_km2": round(s.quantile(.25), 1), "p75_km2": round(s.quantile(.75), 1), "p90_km2": round(s.quantile(.9), 1), "max_km2": round(s.max(), 1)})
res = pd.DataFrame(rows); res.to_csv(a.out, index=False); print(res.to_string(index=False)); print("reserved-forest polygons:", len(t), "| states with named divisions:", t[t.division != ""].st_name.nunique())

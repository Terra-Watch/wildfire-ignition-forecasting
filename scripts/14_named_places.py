"""Named villages, hamlets, towns and localities from the OpenStreetMap extract (January 2021) that the spread work already cuts from India, as a small table for the area forecast:
"where to look" lists the nearest named place of each high-chance cell. Coverage is thin in many forest areas, so a cell may have no place within reach.
Run: .venv/bin/python scripts/14_named_places.py [--src ../fire-spread/data/places.geojsonseq]"""
import argparse, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--src", default=os.path.join(os.path.dirname(ROOT), "fire-spread", "data", "places.geojsonseq")); a = ap.parse_args()
KEEP = {"village", "hamlet", "town", "locality", "city"}; rows = []
with open(a.src, encoding="utf8") as f:
    for line in f:
        line = line.lstrip("\x1e").strip()
        if not line: continue
        ft = json.loads(line); p = ft["properties"]; g = ft["geometry"]
        if g["type"] != "Point" or p.get("place") not in KEEP or not p.get("name"): continue
        lon, lat = g["coordinates"][:2]
        if 6 <= lat <= 36 and 67 <= lon <= 98: rows.append((p["name"], p["place"], round(lat, 4), round(lon, 4)))
df = pd.DataFrame(rows, columns=["name", "kind", "lat", "lon"]).drop_duplicates(["name", "lat", "lon"])
os.makedirs(f"{ROOT}/results", exist_ok=True); df.to_parquet(f"{ROOT}/results/named_places.parquet", index=False, compression="zstd")
print(len(df), "places;", df.kind.value_counts().to_dict(), f"{os.path.getsize(f'{ROOT}/results/named_places.parquet') / 1e6:.1f} MB")

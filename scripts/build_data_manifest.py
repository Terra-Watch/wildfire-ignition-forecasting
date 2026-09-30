"""Write data/MANIFEST.json describing everything under DATA_ROOT."""
from pathlib import Path

from wildfire.data import manifest

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "data" / "MANIFEST.json"
    manifest.write(target, manifest.build())
    print(f"wrote {target}")

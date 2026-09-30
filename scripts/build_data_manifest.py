"""Write data/MANIFEST.json (files under DATA_ROOT) and artifacts/MANIFEST.json (files under ARTIFACTS_ROOT)."""
from pathlib import Path

from wildfire import config
from wildfire.data import manifest

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    manifest.write(repo / "data" / "MANIFEST.json", manifest.build())
    manifest.write(config.artifacts_root() / "MANIFEST.json", manifest.build(group="artifact_groups"))
    print("wrote data/MANIFEST.json and", config.artifacts_root() / "MANIFEST.json")

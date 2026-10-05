"""Build the in-season model table and its aligned arrays in a second artifacts folder.
Usage: python scripts/build_in_season_table.py <target folder>   (the source is the current artifacts folder; needs the .env variables)."""
import json
import sys
from pathlib import Path

from wildfire import config
from wildfire.pipelines import in_season

if __name__ == "__main__":
    record = in_season.build_in_season_folder(config.artifacts_root(), Path(sys.argv[1]))
    print(json.dumps(record, indent=1))

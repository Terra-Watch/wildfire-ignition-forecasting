"""Guard: the pre-declared baseline specification must not change once scoring has started."""
from __future__ import annotations

import json
from pathlib import Path

from wildfire import config
from wildfire.guards import provenance

STAMP_NAME = "spec_stamp.json"


def spec_files() -> list[str]:
    return config.load("baseline")["spec_files"]


def spec_commit() -> str:
    """Last commit that touched the specification files."""
    return provenance._git("log", "-1", "--format=%H", "--", *spec_files())


def spec_is_dirty() -> bool:
    """True when a specification file has uncommitted changes (or is not tracked)."""
    return bool(provenance._git("status", "--porcelain", "--", *spec_files()))


def check_stamp(stamp_path: Path, commit: str) -> None:
    """Write the stamp on first use; refuse when it names a different commit."""
    if stamp_path.exists():
        stamped = json.loads(stamp_path.read_text())["spec_commit"]
        if stamped != commit:
            raise SystemExit(f"the specification changed after scoring started (stamped {stamped[:10]}, now {commit[:10]})")
    else:
        stamp_path.write_text(json.dumps({"spec_commit": commit}))


def require_spec() -> str:
    """Return the specification commit; refuse if the files are dirty or differ from the stamped commit."""
    if spec_is_dirty():
        raise SystemExit("specification files have uncommitted changes")
    commit = spec_commit()
    check_stamp(config.output_dir("results") / STAMP_NAME, commit)
    return commit

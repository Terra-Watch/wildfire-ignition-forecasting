"""Code provenance: results carry the commit hash of the code that produced them."""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

from wildfire import config

CODE_PATHS = ["src", "configs", "scripts", "tests", "pyproject.toml"]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(config.REPO_ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def code_state() -> dict:
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "-uall", "--", *CODE_PATHS).splitlines()
    return dict(commit=head, dirty=bool(dirty), dirty_files=[line[3:] for line in dirty])


def require_clean() -> str:
    """Return the commit hash; raise when code has uncommitted changes."""
    s = code_state()
    if s["dirty"]:
        raise SystemExit("uncommitted code changes: " + ", ".join(s["dirty_files"][:10]) + " -- commit first so results carry a real hash")
    return s["commit"]


def tag() -> str:
    s = code_state()
    return s["commit"] + ("+DIRTY" if s["dirty"] else "")


def stamp_df(df, path, **to_csv_kw):
    d = df.copy()
    d.insert(0, "code_commit", tag())
    d.to_csv(path, index=to_csv_kw.pop("index", False), **to_csv_kw)
    return d


def stamp_text(path, text: str) -> None:
    Path(path).write_text(f"code_commit: {tag()}\nwritten: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n" + text)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

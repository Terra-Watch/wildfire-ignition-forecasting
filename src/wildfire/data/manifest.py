"""Describe and verify the contents of DATA_ROOT with content hashes."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wildfire import config


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _files(root: Path, patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(found)


def describe_dataset(root: Path, spec: dict, workers: int = 8) -> dict:
    files = _files(root, spec["paths"])
    with ThreadPoolExecutor(workers) as ex:
        hashes = list(ex.map(_sha256, files))
    digest = hashlib.sha256()
    for f, h in zip(files, hashes):
        digest.update(f"{f.relative_to(root)}\0{h}\n".encode())
    return {
        "description": spec.get("description", ""),
        "paths": spec["paths"],
        "coverage": spec.get("coverage"),
        "complete": spec.get("complete"),
        "file_count": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "sha256": digest.hexdigest(),
    }


def build(root: Path | None = None, group: str = "datasets") -> dict:
    """Describe the datasets of a group in configs/data.yaml: `datasets` (under DATA_ROOT) or `artifact_groups`
    (under ARTIFACTS_ROOT)."""
    root = root or (config.data_root() if group == "datasets" else config.artifacts_root())
    specs = config.load("data")[group]
    return {"root_env": "DATA_ROOT" if group == "datasets" else "ARTIFACTS_ROOT",
            "datasets": {n: describe_dataset(root, s) for n, s in specs.items()}}


def write(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify(path: Path, root: Path | None = None, group: str = "datasets") -> list[str]:
    """Return a list of differences between the manifest and the files on disk."""
    recorded = json.loads(path.read_text())["datasets"]
    current = build(root, group)["datasets"]
    problems = []
    for name, rec in recorded.items():
        cur = current.get(name)
        if cur is None:
            problems.append(f"{name}: no longer defined in configs/data.yaml")
        elif cur["sha256"] != rec["sha256"]:
            problems.append(f"{name}: contents changed "
                            f"({rec['file_count']} -> {cur['file_count']} files)")
    return problems

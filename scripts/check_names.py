"""Fail when a file name or file content carries a forbidden label.

Labels are matched as whole tokens (split on punctuation, digits boundaries and
camelCase), so ordinary words such as "background" do not match.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Built from fragments so this file does not contain the labels it forbids.
_WORDS = ["pi" + "lot", "pha" + "se", "rou" + "nd", "au" + "dit", "experi" + "ment"]
_LABEL = re.compile(
    r"^(?:" + "|".join(_WORDS) + r"|v\d+|[pr]\d)s?$", re.IGNORECASE
)
_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|[A-Za-z]\d+")
SELF = Path(__file__).resolve()
SKIP_SUFFIX = {".json", ".lock", ".png", ".pdf"}


def tokens(text: str) -> list[str]:
    out = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        if chunk:
            out.append(chunk)
            out.extend(_TOKEN.findall(chunk))
    return out


def bad_tokens(text: str) -> list[str]:
    return sorted({t for t in tokens(text) if _LABEL.match(t)})


def check_file(path: Path) -> list[str]:
    problems = [f"{path}: file name uses '{t}'" for t in bad_tokens(str(path))]
    if path.resolve() == SELF or path.suffix in SKIP_SUFFIX:
        return problems
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return problems
    for n, line in enumerate(lines, 1):
        for t in bad_tokens(line):
            problems.append(f"{path}:{n}: uses '{t}'")
    return problems


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return [Path(p) for p in out]


def main(argv: list[str]) -> int:
    files = tracked_files() if argv == ["--all"] else [Path(a) for a in argv]
    problems = [p for f in files if f.is_file() for p in check_file(f)]
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

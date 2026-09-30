# Repository layout

- Fresh history; no code ported at creation.
- Constants live only in `configs/`.
- Data is referenced through `DATA_ROOT`, never copied.
- Exploratory work goes to the sibling `wildfire-research` repository, which imports `wildfire`; the reverse is not allowed.
- Files, folders, code, comments, docs and commit messages carry no stage or attempt labels; `scripts/check_names.py` enforces this on commit and in `make test`.

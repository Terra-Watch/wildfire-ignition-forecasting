PY ?= .venv/bin/python
export PYTHONPATH := src

test:  # tests that need the dataset (../terrawatch-datasets/dataset, or WILDFIRE_DATA) skip without it
	$(PY) -m pytest -q tests
check:  # no data needed: every file compiles
	$(PY) -m compileall -q src scripts tests

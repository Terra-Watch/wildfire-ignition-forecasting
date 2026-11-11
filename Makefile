PY ?= .venv/bin/python
export PYTHONPATH := src

test:  # needs the dataset at ../wildfire-dataset/dataset (or WILDFIRE_DATA)
	$(PY) -m unittest discover -s tests
check:  # no data needed: every file compiles
	$(PY) -m compileall -q src scripts tests

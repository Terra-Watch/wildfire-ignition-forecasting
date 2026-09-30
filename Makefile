PYTHON ?= .venv/bin/python

.PHONY: setup test check-names manifest

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

test: check-names
	@set -a; [ -f .env ] && . ./.env; set +a; $(PYTHON) -m pytest -q

check-names:
	$(PYTHON) scripts/check_names.py --all

manifest:
	$(PYTHON) scripts/build_data_manifest.py

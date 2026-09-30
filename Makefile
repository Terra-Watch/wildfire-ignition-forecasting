PYTHON ?= .venv/bin/python

.PHONY: setup test check-names manifest notebooks

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

test: check-names
	@set -a; [ -f .env ] && . ./.env; set +a; $(PYTHON) -m pytest -q

check-names:
	$(PYTHON) scripts/check_names.py --all

manifest:
	$(PYTHON) scripts/build_data_manifest.py

notebooks:
	@set -a; . ./.env; set +a; for n in notebooks/*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace $$n || exit 1; done

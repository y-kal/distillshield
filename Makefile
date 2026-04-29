PYTHON=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYTHON_BIN=$(VENV)/bin/python
UVICORN=$(VENV)/bin/uvicorn
PYTEST=$(VENV)/bin/pytest
NPM=npm

.PHONY: venv install api web generate evaluate test

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	cd apps/web && $(NPM) install

api:
	$(UVICORN) distillshield_api.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd apps/web && $(NPM) run dev -- --host 0.0.0.0 --port 5173

generate:
	$(PYTHON_BIN) scripts/generate_scenarios.py

evaluate:
	$(PYTHON_BIN) scripts/evaluate.py

test:
	$(PYTEST) -q

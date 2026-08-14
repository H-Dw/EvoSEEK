PYTHON ?= python3
VENV ?= .venv

.PHONY: setup data models demo baselines ablation science test test-unit test-integration test-e2e test-leakage clean-artifacts

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements/dev.txt

data:
	bash scripts/data/download_flip_gb1.sh
	$(VENV)/bin/python scripts/data/prepare_gb1.py --source data/raw/flip/gb1/four_mutations_full_data.csv

models:
	$(VENV)/bin/python scripts/models/download_models.py --profile baseline

demo:
	$(VENV)/bin/python scripts/run_demo.py

baselines:
	$(VENV)/bin/python scripts/run_baselines.py --seeds 11,22,33,44,55

ablation:
	$(VENV)/bin/python scripts/run_ablation.py --seed 17

science:
	$(VENV)/bin/python scripts/run_scientific_thinking.py --seed 23

test: test-unit test-integration test-leakage test-e2e

test-unit:
	bash scripts/tests/run_unit.sh

test-integration:
	bash scripts/tests/run_integration.sh

test-leakage:
	bash scripts/tests/run_leakage.sh

test-e2e:
	bash scripts/tests/run_e2e.sh

clean-artifacts:
	$(VENV)/bin/python scripts/clean_artifacts.py --path artifacts/runs


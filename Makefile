PY ?= python
export PYTHONPATH := src

.PHONY: install test data quick full report

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q tests

data:
	$(PY) scripts/prepare_data.py --raw data/raw --out data/processed

quick:
	scripts/run_experiments.sh quick

full:
	scripts/run_experiments.sh full

report:
	$(PY) scripts/aggregate.py --results results/full

PYTHON ?= python

.PHONY: install test audit-notebook run elastic-net

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

audit-notebook:
	$(PYTHON) tools/audit_complete_notebook.py notebooks/01_Complete_Executed_Analysis.ipynb

run:
	$(PYTHON) src/mortality_prediction_pipeline.py --data ami_patient_data.csv --output outputs/current

elastic-net:
	$(PYTHON) src/elastic_net_screen.py --data ami_patient_data.csv --output outputs/elastic_net

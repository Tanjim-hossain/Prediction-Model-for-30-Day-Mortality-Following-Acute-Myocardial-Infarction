PYTHON ?= python

.PHONY: install test run elastic-net

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

run:
	$(PYTHON) src/isds_option_a_pipeline.py --data ami_patient_data.csv --output outputs/current

elastic-net:
	$(PYTHON) src/elastic_net_screen.py --data ami_patient_data.csv --output outputs/elastic_net

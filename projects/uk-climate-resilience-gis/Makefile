PYTHON := ./scripts/run_python.sh
.PHONY: setup download process analyse qgis build-qgis export validate test all
setup download process analyse export validate:
	$(PYTHON) -m resilience $@
qgis build-qgis:
	$(PYTHON) -m resilience build-qgis
test:
	$(PYTHON) -m unittest discover -s tests -v
all:
	$(PYTHON) -m resilience all

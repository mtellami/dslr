PYTHON ?= .venv/bin/python
SYSTEM_PYTHON ?= python3

.PHONY: help install cli-check test check

help:
	@printf '%s\n' 'Available targets:' \
	  '  install   Create a virtual environment and install the project' \
	  '  cli-check Check the six command-line interfaces' \
	  '  test      Run regression tests' \
	  '  check     Run cli-check and test'

install:
	$(SYSTEM_PYTHON) -m venv .venv
	$(PYTHON) -m pip install -e .

cli-check:
	$(PYTHON) -m dslr.commands.describe --help
	$(PYTHON) -m dslr.commands.histogram --help
	$(PYTHON) -m dslr.commands.scatter_plot --help
	$(PYTHON) -m dslr.commands.pair_plot --help
	$(PYTHON) -m dslr.commands.logreg_train --help
	$(PYTHON) -m dslr.commands.logreg_predict --help

test:
	$(PYTHON) -m unittest discover -s tests -v

check: cli-check test

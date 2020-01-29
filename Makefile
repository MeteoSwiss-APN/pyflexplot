SHELL := /bin/bash

.PHONY: clean clean-test clean-pyc clean-build venv venv-install venv-install-pinned venv-install-dev docs help
.DEFAULT_GOAL := help

#==============================================================================
# Helper Functions
#==============================================================================

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

#==============================================================================

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

#==============================================================================
# Options
#==============================================================================

VENV_DIR ?= venv
VENV_NAME ?= pyflexplot

#==============================================================================
# Parameters
#==============================================================================

BROWSER := python -c "$$BROWSER_PYSCRIPT"

#==============================================================================
# Help
#==============================================================================

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

#==============================================================================
# Cleanup
#==============================================================================

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	@\rm -rfv build/
	@\rm -rfv dist/
	@\rm -rfv .eggs/
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg-info' -exec \rm -rfv {} +
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec \rm -f {} +

clean-pyc: ## remove Python file artifacts
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyc' -exec \rm -f {} +
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyo' -exec \rm -f {} +
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*~' -exec \rm -f {} +
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '__pycache__' -exec \rm -rfv {} +

clean-test: ## remove test and coverage artifacts
	@\rm -rfv .tox/
	@\rm -fv .coverage
	@\rm -rfv htmlcov/
	@\rm -rfv .pytest_cache

#==============================================================================
# Installation
#==============================================================================

install: ## install the package (non-editable)
	# python -m pip install -r requirements/setup.txt
	python -m pip install .

install-pinned: ## install the package (non-editable)
	# python -m pip install -r requirements/setup.txt
	python -m pip install -r requirements/run-pinned.txt
	python -m pip install .

install-dev: install ## install the package (editable) and the development dependencies
	python -m pip install -r requirements/dev-unpinned.txt

#==============================================================================
# Virtual Environments
#==============================================================================

venv: ## create a virtual environment; options: VENV_DIR=venv, VENV_NAME=pyflexplot
	python -m venv ${VENV_DIR} --prompt='${VENV_NAME}'
	${VENV_DIR}/bin/python -m pip install -U pip

venv-install: venv ## install the package (non-editable) into a virtual environment; options: VENV_DIR=venv, VENV_NAME=pyflexplot
	# ${VENV_DIR}/bin/python -m pip install -r requirements/setup.txt
	${VENV_DIR}/bin/python -m pip install .

venv-install-pinned: venv ## install the package (non-editable) into a virtual environment with pinned dependencies; options: VENV_DIR=venv, VENV_NAME=pyflexplot
	# ${VENV_DIR}/bin/python -m pip install -r requirements/setup.txt
	${VENV_DIR}/bin/python -m pip install -r requirements/run-pinned.txt
	${VENV_DIR}/bin/python -m pip install .

venv-install-dev: venv-install ## install the package (editable) and the development dependencies into a virtual environment; options: VENV_DIR=venv, VENV_NAME=pyflexplot
	${VENV_DIR}/bin/python -m pip install -r requirements/dev-unpinned.txt

#==============================================================================
# Git
#==============================================================================

git: clean ## initialize a git repository and make initial commit
ifeq ($(shell git tag >/dev/null 2>&1 && echo 0 || echo 1),0)
	@echo "git already initialized"
else
	git init
	git add .
	git commit -m 'initial commit'
	git --no-pager log -n1 --stat
endif

#==============================================================================
# Formatting & Linting
#==============================================================================

format: ## reformat the code to conform to standard
	black src tests

lint: ## check the code style
	flake8 src tests

#==============================================================================
# Testing
#==============================================================================

test: ## run tests quickly with the default Python version
	pytest -r s

coverage: ## check code coverage of tests
	coverage run --source src -m pytest
	coverage report -m

coverage-html: coverage ## check code coverage of tests and show results in browser
	coverage html
	$(BROWSER) htmlcov/index.html

test-all: ## run tests on all specified Python versions with tox
	tox

#==============================================================================
# Documentation
#==============================================================================

# docs: ## generate Sphinx HTML documentation, including API docs
# 	\rm -f docs/pyflexplot.rst
# 	\rm -f docs/modules.rst
# 	sphinx-apidoc -o docs/ src/pyflexplot
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	$(BROWSER) docs/_build/html/index.html

# servedocs: docs ## compile the docs watching for changes
# 	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

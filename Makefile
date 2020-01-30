SHELL := /bin/bash

.PHONY: clean clean-test clean-pyc clean-build venv venv-install venv-install-pinned venv-install-dev docs help
.DEFAULT_GOAL := help

#==============================================================================
# Options
#==============================================================================

VENV_DIR ?= venv #OPT Path to virtual environment to be created and/or used.
VENV_NAME ?= pyflexplot #OPT Name of virtual environment if one is created.

#==============================================================================
# Function: Print Help
#==============================================================================

define print_help_py
import re
import sys

def parse_makefile(lines):
	options = {}
	commands = {}
	rx_opt = re.compile(
		r"^(?P<name>[A-Z_]+) *\?= *(?P<value>[^ ]*) *(#OPT (?P<help>.*))? *$$"
	)
	rx_cmd = re.compile(
		r"^(?P<name>[a-zA-Z_-]+):.*?#CMD (?P<help>.*) *$$"
	)
	for line in lines:
		match_opt = rx_opt.match(line)
		match_cmd = rx_cmd.match(line)
		if match_opt:
			m = match_opt
			help = m.group("help").split(r"\n")
			options[m.group("name")] = (m.group("value"), help)
		elif match_cmd:
			m = match_cmd
			commands[m.group("name")] = m.group("help").split(r"\n")
	return options, commands

def format_options(items):
	s = ""
	for name, (value, help) in items.items():
		name_value = f"{name}={value}"
		for idx, line in enumerate(help):
			s += f"  {name_value if idx == 0 else '':25s} {line}\n"
	return s.rstrip()

def format_commands(items):
	s = ""
	for name, help in items.items():
		for idx, line in enumerate(help):
			s += f"  {name if idx == 0 else '':25s} {line}\n"
	return s.rstrip()

options, commands = parse_makefile(sys.stdin)

print(f"""\
Usage: make COMMAND [OPTIONS]

Options:
{format_options(options)}

Commands:
{format_commands(commands)}
""")
endef
export print_help_py

#==============================================================================
# Function: Open Browser
#==============================================================================

define browser_py
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open(f"file://{pathname2url(os.path.abspath(sys.argv[1]))}")
endef
export browser_py
browser := python -c "$$browser_py"

#==============================================================================
# Help
#==============================================================================

help:
	@python -c "$$print_help_py" < $(MAKEFILE_LIST)

#==============================================================================
# Cleanup
#==============================================================================

clean: clean-build clean-pyc clean-test #CMD Remove all build, test, coverage and Python artifacts.

clean-build: #CMD Remove build artifacts.
	@\rm -rfv build/
	@\rm -rfv dist/
	@\rm -rfv .eggs/
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg-info' -exec \rm -rfv {} +
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec \rm -f {} +

clean-pyc: #CMD Remove Python file artifacts.
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyc' -exec \rm -f {} +
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyo' -exec \rm -f {} +
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '*~' -exec \rm -f {} +
	@\find . -not -path './venv*' -and -not -path './ENV*' -name '__pycache__' -exec \rm -rfv {} +

clean-test: #CMD Remove test and coverage artifacts.
	@\rm -rfv .tox/
	@\rm -fv .coverage
	@\rm -rfv htmlcov/
	@\rm -rfv .pytest_cache

#==============================================================================
# Installation
#==============================================================================

install: #CMD Install the package with unpinned runtime dependencies.
	python -m pip install .

install-pinned: #CMD Install the package with pinned runtime dependencies.
	python -m pip install -r requirements/run-pinned.txt
	python -m pip install .

install-dev: install #CMD Install the package as editable with unpinned runtime\nand development dependencies.
	python -m pip install -r requirements/dev-unpinned.txt

#==============================================================================
# Virtual Environments
#==============================================================================

venv: #CMD Create a virtual environment.\nOptions: VENV_DIR=venv, VENV_NAME=pyflexplot
	python -m venv ${VENV_DIR} --prompt='${VENV_NAME}'
	${VENV_DIR}/bin/python -m pip install -U pip

venv-install: venv #CMD Install the package with unpinned runtime dependencies.\nOptions: VENV_DIR=venv, VENV_NAME=pyflexplot
	${VENV_DIR}/bin/python -m pip install .

venv-install-pinned: venv #CMD Install the package with pinned runtime dependencies.\nOptions: VENV_DIR=venv, VENV_NAME=pyflexplot
	${VENV_DIR}/bin/python -m pip install -r requirements/run-pinned.txt
	${VENV_DIR}/bin/python -m pip install .

venv-install-dev: venv-install #CMD Install the package as editable with unpinned runtime\nand development dependencies.\nOptions: VENV_DIR=venv, VENV_NAME=pyflexplot
	${VENV_DIR}/bin/python -m pip install -r requirements/dev-unpinned.txt

#==============================================================================
# Git
#==============================================================================

git: clean #CMD Initialize a git repository and make initial commit.
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

format: #CMD Reformat the code to conform with standards like PEP 8.
	black src tests

lint: #CMD Check the code style.
	flake8 src tests

#==============================================================================
# Testing
#==============================================================================

test: #CMD Run all tests with the default Python version.
	python -m pytest

coverage: #CMD Check code coverage of tests.
	coverage run --source src -m pytest
	coverage report -m

coverage-html: coverage #CMD Check code coverage of tests and show results in browser.
	coverage html
	$(browser) htmlcov/index.html

test-all: #CMD Run tests on all specified Python versions with tox.
	tox

#==============================================================================
# Documentation
#==============================================================================

# docs: #CMD Generate Sphinx HTML documentation, including API docs.
# 	\rm -f docs/pyflexplot.rst
# 	\rm -f docs/modules.rst
# 	sphinx-apidoc -o docs/ src/pyflexplot
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	$(browser) docs/_build/html/index.html

# servedocs: docs #CMD Compile the docs watching for changes.
# 	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

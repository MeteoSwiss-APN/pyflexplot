SHELL := /bin/bash

.PHONY: clean-all clean-test clean-pyc clean-build clean-venv
.PHONY: venv install install-edit install-pinned install-test install-test-pinned install-dev
.PHONY: test test-cov test-cov-html test-all
.PHONY: docs help
.DEFAULT_GOAL := help

#==============================================================================
# Options
#==============================================================================

IGNORE_VENV ?= 0#OPT Don't create and/or use a virtual environment.
INSTALL ?= 1#OPT Always ensure that dependencies etc. are installed
VENV_DIR ?= venv#OPT Path to virtual environment to be created and/or used.
VENV_NAME ?= pyflexplot#OPT Name of virtual environment if one is created.

#------------------------------------------------------------------------------

PREFIX_VENV = ${VENV_DIR}/bin/#
export PREVIX_VENV

ifneq (${IGNORE_VENV}, 0)
PREFIX ?=#
else
ifeq (${VIRTUAL_ENV},)
PREFIX =#
else
PREFIX = ${PREFIX_VENV}
endif
endif
export PREFIX

#==============================================================================
# Python script: Print help
#==============================================================================

define PRINT_HELP_PY
import re
import sys

def parse_makefile(lines):
    options = {}
    commands = {}
    rx_opt = re.compile(
        r"^(?P<name>[A-Z_]+) *\?= *(?P<value>[^ ]*) *#OPT (?P<help>.*)? *$$"
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
export PRINT_HELP_PY

#==============================================================================
# Python script: Open browser
#==============================================================================

define BROWSER_PY
import os
import sys
import webbrowser

try:
    from urllib import pathname2url
except:
    from urllib.request import pathname2url

webbrowser.open(f"file://{pathname2url(os.path.abspath(sys.argv[1]))}")
endef
export BROWSER_PY
browser = ${PREFIX}python -c "$$BROWSER_PY"

#==============================================================================
# Help
#==============================================================================

help:
	@python -c "$$PRINT_HELP_PY" < $(MAKEFILE_LIST)

#==============================================================================
# Cleanup
#==============================================================================

clean-all: clean-venv clean-test clean-build clean-pyc #CMD Remove all build, test, coverage and Python artifacts.

clean-build: #CMD Remove build artifacts.
	\rm -rf "build/"
	\rm -rf "dist/"
	\rm -rf ".eggs/"
	@\rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)
	@\rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)

clean-pyc: #CMD Remove Python file artifacts.
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyc'       -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyo'       -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*~'          -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '__pycache__' -exec echo "rm -rf '{}'" \;)

clean-test: #CMD Remove test and coverage artifacts.
	\rm -rf ".tox/"
	\rm -f ".coverage"
	\rm -rf "htmlcov/"
	\rm -rf ".pytest_cache"

clean-venv: #CMD Remove virtual environment.
	\rm -rf "${VENV_DIR}"

#==============================================================================
# Virtual Environments
#==============================================================================

venv: #CMD Create a virtual environment.
ifeq (${IGNORE_VENV}, 0)
	$(eval PREFIX = ${PREFIX_VENV})
	@export PREFIX
ifeq (${VIRTUAL_ENV},)
ifneq (${INSTALL}, 0)
	python -m venv ${VENV_DIR} --prompt='${VENV_NAME}'
	${PREFIX}python -m pip install -U pip
endif
endif
endif

#==============================================================================
# Installation
#==============================================================================

install: venv #CMD Install the package with unpinned runtime dependencies.
ifneq (${INSTALL}, 0)
	${PREFIX}python -m pip install .
endif

install-edit: venv #CMD Install the package as editable with unpinned runtime dependencies.
ifneq (${INSTALL}, 0)
	${PREFIX}python -m pip install -e .
endif

install-pinned: venv #CMD Install the package with pinned runtime dependencies.
ifneq (${INSTALL}, 0)
	${PREFIX}python -m pip install -r requirements/run-pinned.txt
	${PREFIX}python -m pip install .
endif

install-test-pinned: venv #CMD Install the package with pinned runtime and testing dependencies.
ifneq (${INSTALL}, 0)
	${PREFIX}python -m pip install -r requirements/test-pinned.txt
	${PREFIX}python -m pip install -e .
endif

install-test: install-edit #CMD Install the package with unpinned runtime and testing dependencies.
ifneq (${INSTALL}, 0)
	${PREFIX}python -m pip install -r requirements/test-unpinned.txt
endif

install-dev: install-test #CMD Install the package as editable with unpinned runtime,\ntesting, and development dependencies.
ifneq (${INSTALL}, 0)
	${PREFIX}python -m pip install -r requirements/dev-unpinned.txt
endif

#==============================================================================
# Version control
#==============================================================================

git: clean-all #CMD Initialize a git repository and make initial commit.
ifeq ($(shell git tag >/dev/null 2>&1 && echo 0 || echo 1), 0)
	@echo "git already initialized"
else
	git init
	git add .
	git commit -m 'initial commit'
	git --no-pager log -n1 --stat
endif

#==============================================================================
# Versioning
#==============================================================================

bump-patch: install-dev #CMD Increment patch component of version number (x.y.Z), incl. git commit and tag
	${PREFIX}bumpversion patch

bump-minor: install-dev #CMD Increment minor component of version number (x.Y.z), incl. git commit and tag
	${PREFIX}bumpversion minor

bump-major: install-dev #CMD Increment minor component of version number (X.y.z), incl. git commit and tag
	${PREFIX}bumpversion major

bump-patch-dry: install-dev #CMD Increment patch component of version number (x.y.Z), without git commit and tag
	${PREFIX}bumpversion patch --no-commit --no-tag

bump-minor-dry: install-dev #CMD Increment minor component of version number (x.Y.z), without git commit and tag
	${PREFIX}bumpversion minor --no-commit --no-tag

bump-major-dry: install-dev #CMD Increment minor component of version number (X.y.z), without git commit and tag
	${PREFIX}bumpversion major --no-commit --no-tag

#==============================================================================
# Formatting and linting
#==============================================================================

format: #CMD Reformat the code to conform with standards like PEP 8.
	black src tests

lint: #CMD Check the code style.
	flake8 src tests

#==============================================================================
# Testing
#==============================================================================

test: install-test #CMD Run all tests with the default Python version.
	${PREFIX}pytest

test-cov: install-test #CMD Check code coverage of tests.
	${PREFIX}pytest --cov=src

test-cov-html: install-test #CMD Check code coverage of tests and show results in browser.
	${PREFIX}pytest --cov=src --cov-report=html
	${browser} htmlcov/index.html

test-all: install-test #CMD Run tests on all specified Python versions with tox.
	${PREFIX}tox

#==============================================================================
# Documentation
#==============================================================================

# docs: #CMD Generate Sphinx HTML documentation, including API docs.
# 	\rm -f "docs/pyflexplot.rst"
# 	\rm -f "docs/modules.rst"
# 	sphinx-apidoc -o docs/ src/pyflexplot
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	${browser} docs/_build/html/index.html

# servedocs: docs #CMD Compile the docs watching for changes.
# 	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

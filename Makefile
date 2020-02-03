SHELL := /bin/bash

.PHONY: clean-all clean-test clean-pyc clean-build clean-venv
.PHONY: venv install install-edit install-pinned install-test install-test-pinned install-dev
.PHONY: bump-major bump-minor bump-patch bump-major-dry bump-minor-dry bump-patch-dry
.PHONY: test test-cov test-cov-html test-all
.PHONY: docs help
.DEFAULT_GOAL := help

#==============================================================================
# Options
#==============================================================================

ECHO_PREFIX ?= \nMAKE: #OPT Prefix of command echos
IGNORE_VENV ?= 0#OPT Don't create and/or use a virtual environment.
INSTALL ?= 1#OPT Always ensure that dependencies etc. are installed
VENV_DIR ?= venv#OPT Path to virtual environment to be created and/or used.
VENV_NAME ?= pyflexplot#OPT Name of virtual environment if one is created.

#------------------------------------------------------------------------------

PREFIX_VENV = ${VENV_DIR}/bin/#
export PREVIX_VENV

ifneq (${IGNORE_VENV}, 0)
# Ignore virtual env
PREFIX ?=#
else
ifneq (${VIRTUAL_ENV},)
# Virtual env is active
PREFIX =#
else
# Virtual env is not active
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
	@echo -e "${ECHO_PREFIX}cleaning up"

clean-build: #CMD Remove build artifacts.
	@echo -e "${ECHO_PREFIX}removing build artifacts"
	\rm -rf "build/"
	\rm -rf "dist/"
	\rm -rf ".eggs/"
	@\rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)
	@\rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)

clean-pyc: #CMD Remove Python file artifacts.
	@echo -e "${ECHO_PREFIX}removing Python file artifacts"
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyc'       -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyo'       -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*~'          -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '__pycache__' -exec echo "rm -rf '{}'" \;)

clean-test: #CMD Remove testing artifacts.
	@echo -e "${ECHO_PREFIX}removing testing artifacts"
	\rm -rf ".tox/"
	\rm -f ".coverage"
	\rm -rf "htmlcov/"
	\rm -rf ".pytest_cache"

clean-venv: #CMD Remove virtual environment.
	@echo -e "${ECHO_PREFIX}removing virtual environment at '${VENV_DIR}'"
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
	@echo -e "${ECHO_PREFIX}creating virtual environment '${VENV_NAME}' at '${VENV_DIR}'"
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
	@echo -e "${ECHO_PREFIX}installing the package"
	${PREFIX}python -m pip install .
endif

install-edit: venv #CMD Install the package as editable with unpinned runtime dependencies.
ifneq (${INSTALL}, 0)
	@echo -e "${ECHO_PREFIX}installing the package as editable"
	${PREFIX}python -m pip install -e .
endif

install-pinned: venv #CMD Install the package with pinned runtime dependencies.
ifneq (${INSTALL}, 0)
	@echo -e "${ECHO_PREFIX}installing the package with pinned dependencies"
	${PREFIX}python -m pip install -r requirements/run-pinned.txt
	${PREFIX}python -m pip install .
endif

install-test-pinned: venv #CMD Install the package with pinned runtime and testing dependencies.
ifneq (${INSTALL}, 0)
	@echo -e "${ECHO_PREFIX}installing the package as editable with pinned testing dependencies"
	${PREFIX}python -m pip install -r requirements/test-pinned.txt
	${PREFIX}python -m pip install -e .
endif

install-test: install-edit #CMD Install the package with unpinned runtime and testing dependencies.
ifneq (${INSTALL}, 0)
	@echo -e "${ECHO_PREFIX}installing the package with testing dependencies"
	${PREFIX}python -m pip install -r requirements/test-unpinned.txt
endif

install-dev: install-test #CMD Install the package as editable with unpinned runtime,\ntesting, and development dependencies.
ifneq (${INSTALL}, 0)
	@echo -e "${ECHO_PREFIX}installing the package as editable with testing and development dependencies"
	${PREFIX}python -m pip install -r requirements/dev-unpinned.txt
endif

#==============================================================================
# Version control
#==============================================================================

git: clean-all #CMD Initialize a git repository and make initial commit.
ifeq ($(shell git tag >/dev/null 2>&1 && echo 0 || echo 1), 0)
	@echo -e "${ECHO_PREFIX}git already initialized"
else
	@echo -e "${ECHO_PREFIX}initializing Git repository"
	\git init
	\git add .
	\git commit -m 'initial commit'
	\git --no-pager log -n1 --stat
endif

#==============================================================================
# Versioning
#==============================================================================

bump-patch: install-dev #CMD Bump patch component of version number (x.y.Z), incl. git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment patch component"
	${PREFIX}bumpversion patch

bump-minor: install-dev #CMD Bump minor component of version number (x.Y.z), incl. git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment minor component"
	${PREFIX}bumpversion minor

bump-major: install-dev #CMD Bump minor component of version number (X.y.z), incl. git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment major component"
	${PREFIX}bumpversion major

bump-patch-dry: install-dev #CMD Bump patch component of version number (x.y.Z), without git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment patch component (dry run)"
	${PREFIX}bumpversion patch --no-commit --no-tag

bump-minor-dry: install-dev #CMD Bump minor component of version number (x.Y.z), without git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment minor component (dry run)"
	${PREFIX}bumpversion minor --no-commit --no-tag

bump-major-dry: install-dev #CMD Bump minor component of version number (X.y.z), without git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment major component (dry run)"
	${PREFIX}bumpversion major --no-commit --no-tag

#==============================================================================
# Formatting and linting
#==============================================================================

format: #CMD Reformat the code to conform with standards like PEP 8.
	@echo -e "${ECHO_PREFIX}reformatting the code"
	${PREFIX}black src tests

lint: #CMD Check the code style.
	@echo -e "${ECHO_PREFIX}checking the code style (linting)"
	${PREFIX}flake8 src tests

#==============================================================================
# Testing
#==============================================================================

test: install-test #CMD Run all tests with the default Python version.
	@echo -e "${ECHO_PREFIX}running tests"
	${PREFIX}pytest

test-cov: install-test #CMD Check code coverage of tests.
	@echo -e "${ECHO_PREFIX}running tests with coverage check"
	${PREFIX}pytest --cov=src

test-cov-html: install-test #CMD Check code coverage of tests and show results in browser.
	@echo -e "${ECHO_PREFIX}running tests with coverage check and browser report"
	${PREFIX}pytest --cov=src --cov-report=html
	${browser} htmlcov/index.html

test-all: install-test #CMD Run tests on all specified Python versions with tox.
	@echo -e "${ECHO_PREFIX}running tests in isolated environments"
	${PREFIX}tox

#==============================================================================
# Documentation
#==============================================================================

# docs: install-dev #CMD Generate HTML documentation, including API docs.
#	@echo -e "${ECHO_PREFIX}generating HTML documentation"
# 	\rm -f docs/{{ cookiecutter.project_slug }}.rst
# 	\rm -f docs/modules.rst
# 	${PREFIX}sphinx-apidoc -o docs/ src/{{ cookiecutter.project_slug }}
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	${browser} docs/_build/html/index.html

# servedocs: docs #CMD Compile the docs watching for changes.
#	@echo -e "${ECHO_PREFIX}continuously regenerating HTML documentation"
# 	${PREFIX}watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

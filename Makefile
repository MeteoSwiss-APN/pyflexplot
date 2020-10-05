SHELL := /bin/bash

.DEFAULT_GOAL := help

#==============================================================================
# Options
#==============================================================================

ECHO_PREFIX ?= \nMAKE: #OPT Prefix of command echos
IGNORE_VENV ?= 0#OPT Don't create and/or use a virtual environment.
CHAIN ?= 0#OPT Whether to chain targets, e.g., let test depend on install-test.
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

#
# Targets conditional on ${CHAIN}.
#
# Use these to specify the respective targets for post-installation targets
# like ``test``, for which installation is optionally activated with CHAIN=1.
#
# Note that for interdependencies between venv/installation targets -- like
# of ``install`` on ``venv``, or if ``install-dev`` on ``install-test`` -- the
# targets are hard-coded w/o these variables, such that they are always executed
# regardless of ${CHAIN}.
#
ifeq (${CHAIN}, 0)
	_VENV :=
	_INSTALL :=
	_INSTALL_EDIT :=
	_INSTALL_DEV :=
	_INSTALL_PINNED :=
else
	_VENV := venv
	_INSTALL := install
	_INSTALL_EDIT := install-edit
	_INSTALL_DEV := install-dev
	_INSTALL_PINNED := install-pinned
endif
export _VENV
export _INSTALL
export _INSTALL_EDIT
export _INSTALL_DEV
export _INSTALL_PINNED

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
    rx_cmd1 = re.compile(
        r"^(?P<name>[a-zA-Z_-]+):.*?#CMD (?P<help>.*) *$$"
    )
    rx_cmd2 = re.compile(
        r"^.PHONY: (?P<name>[a-zA-Z_-]+) *#CMD (?P<help>.*) *$$"
    )
    for line in lines:
        match_opt = rx_opt.match(line)
        match_cmd1 = rx_cmd1.match(line)
        match_cmd2 = rx_cmd2.match(line)
        if match_opt:
            m = match_opt
            help = m.group("help").split(r"\n")
            options[m.group("name")] = (m.group("value"), help)
        elif match_cmd1 or match_cmd2:
            m = match_cmd1 if match_cmd1 else match_cmd2
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

.PHONY: help #CMD Print this help page.
help:
	@python -c "$$PRINT_HELP_PY" < $(MAKEFILE_LIST)

#==============================================================================
# Cleanup
#==============================================================================

.PHONY: clean-all #CMD Remove all build, test, coverage and Python artifacts.
clean-all: clean-venv clean-test clean-build clean-pyc
	@echo -e "${ECHO_PREFIX}cleaning up"

.PHONY: clean-build #CMD Remove build artifacts.
clean-build:
	@echo -e "${ECHO_PREFIX}removing build artifacts"
	\rm -rf "build/"
	\rm -rf "dist/"
	\rm -rf ".eggs/"
	@\rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)
	@\rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)

.PHONY: clean-pyc #CMD Remove Python file artifacts.
clean-pyc:
	@echo -e "${ECHO_PREFIX}removing Python file artifacts"
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyc'       -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyo'       -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*~'          -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '__pycache__' -exec echo "rm -rf '{}'" \;)

.PHYNO: clean-test #CMD Remove testing artifacts.
clean-test:
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

.PHONY: venv #CMD Create a virtual environment.
venv:
ifeq (${IGNORE_VENV}, 0)
	$(eval PREFIX = ${PREFIX_VENV})
	@export PREFIX
ifeq (${VIRTUAL_ENV},)
	@echo -e "${ECHO_PREFIX}creating virtual environment '${VENV_NAME}' at '${VENV_DIR}'"
	python -m venv ${VENV_DIR} --prompt='${VENV_NAME}'
	${PREFIX}python -m pip install -U pip
endif
endif

#==============================================================================
# Installation
#==============================================================================

.PHONY: install #CMD Install the package with unpinned runtime dependencies.
install: venv
	@echo -e "${ECHO_PREFIX}installing the package"
	${PREFIX}python -m pip install . --use-feature=2020-resolver

.PHONY: install-edit #CMD Install the package as editable with unpinned runtime dependencies.
install-edit: venv
	@echo -e "${ECHO_PREFIX}installing the package as editable"
	${PREFIX}python -m pip install -e . --use-feature=2020-resolver

.PHONY: install-pinned #CMD Install the package with pinned runtime dependencies.
install-pinned: venv
	@echo -e "${ECHO_PREFIX}installing the package with pinned dependencies"
	${PREFIX}python -m pip install -r requirements/run-pinned.txt --use-feature=2020-resolver
	${PREFIX}python -m pip install . --use-feature=2020-resolver

.PHONY: install-dev #CMD Install the package as editable with unpinned runtime and development dependencies.
install-dev: install-edit
	@echo -e "${ECHO_PREFIX}installing the package as editable with runtime and development dependencies"
	${PREFIX}python -m pip install -r requirements/dev-unpinned.txt --use-feature=2020-resolver
	${PREFIX}pre-commit install

#==============================================================================
# Dependencies
#==============================================================================

_TMP_VENV != echo venv-tmp-$$(date +%s)
export _TMP_VENV

.PHONY: update-run-deps #CMD Update pinned runtime dependencies based on setup.py
update-run-deps:
	@echo -e "${ECHO_PREFIX}updating pinned runtime dependencies"
	@echo -e "temporary virtual environment: ${_TMP_VENV}"
	python -m venv ${_TMP_VENV}
	${_TMP_VENV}/bin/python -m pip install -U pip
	\rm -fv requirements.txt
	${_TMP_VENV}/bin/python -m pip install . --use-feature=2020-resolver; \
		ln -sfv requirements/run-pinned.txt requirements.txt
	${_TMP_VENV}/bin/python -m pip freeze | \grep -v '\<file:' > requirements/run-pinned.txt
	\rm -rf ${_TMP_VENV}

.PHONY: update-dev-deps #CMD Update pinned development dependencies based on\nrequirements/run-pinned.txt and requitements/dev-unpinned.txt
update-dev-deps:
	@echo -e "${ECHO_PREFIX}updating pinned development dependencies"
	@echo -e "temporary virtual environment: ${_TMP_VENV}"
	python -m venv ${_TMP_VENV}
	${_TMP_VENV}/bin/python -m pip install -U pip
	${_TMP_VENV}/bin/python -m pip install -r requirements/dev-unpinned.txt --use-feature=2020-resolver
	${_TMP_VENV}/bin/python -m pip install -r requirements/run-pinned.txt --no-deps --use-feature=2020-resolver
	${_TMP_VENV}/bin/python -m pip freeze > requirements/dev-pinned.txt
	\rm -rf ${_TMP_VENV}

.PHONY: update-tox-deps #CMD Update pinned tox testing dependencies based on\nrequirements/tox-unpinned.txt
update-tox-deps:
	@echo -e "${ECHO_PREFIX}updating pinned tox testing dependencies"
	@echo -e "temporary virtual environment: ${_TMP_VENV}"
	python -m venv ${_TMP_VENV}
	${_TMP_VENV}/bin/python -m pip install -U pip
	${_TMP_VENV}/bin/python -m pip install -r requirements/tox-unpinned.txt --use-feature=2020-resolver
	${_TMP_VENV}/bin/python -m pip freeze > requirements/tox-pinned.txt
	\rm -rf ${_TMP_VENV}

#==============================================================================
# Version control
#==============================================================================

.PHONY: git #CMD Initialize a git repository and make initial commit.
git: clean-all
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

.PHONY: bump-patch #CMD Bump patch component of version number (x.y.Z), incl. git commit and tag
bump-patch: ${_INSTALL_DEV}
	@echo -e "${ECHO_PREFIX}bumping version number: increment patch component"
	${PREFIX}bumpversion patch

.PHONY: bump-minor #CMD Bump minor component of version number (x.Y.z), incl. git commit and tag
bump-minor: ${_INSTALL_DEV}
	@echo -e "${ECHO_PREFIX}bumping version number: increment minor component"
	${PREFIX}bumpversion minor

.PHONY: bump-major #CMD Bump minor component of version number (X.y.z), incl. git commit and tag
bump-major: ${_INSTALL_DEV}
	@echo -e "${ECHO_PREFIX}bumping version number: increment major component"
	${PREFIX}bumpversion major

.PHONY: bump-patch-dry #CMD Bump patch component of version number (x.y.Z), without git commit and tag
bump-patch-dry: ${_INSTALL_DEV}
	@echo -e "${ECHO_PREFIX}bumping version number: increment patch component (dry run)"
	${PREFIX}bumpversion patch --no-commit --no-tag

.PHONY: bump-minor-dry #CMD Bump minor component of version number (x.Y.z), without git commit and tag
bump-minor-dry: ${_INSTALL_DEV}
	@echo -e "${ECHO_PREFIX}bumping version number: increment minor component (dry run)"
	${PREFIX}bumpversion minor --no-commit --no-tag

.PHONY: bump-major-dry #CMD Bump minor component of version number (X.y.z), without git commit and tag
bump-major-dry: ${_INSTALL_DEV}
	@echo -e "${ECHO_PREFIX}bumping version number: increment major component (dry run)"
	${PREFIX}bumpversion major --no-commit --no-tag

#==============================================================================
# Format and check the code
#==============================================================================

.PHONY: format #CMD Check and fix the code formatting
format: ${_INSTALL_DEV}
	@echo -e "${ECHO_PREFIX}checking and fixing code formatting"
	${PREFIX}pre-commit run --all-files

.PHONY: check # CMD
check: ${_INSTALL_DEV}
	@echo -e "${ECHO_PREFIX}checking code correctness and quality"
	${PREFIX}tox --parallel -e mypy -e flake8 -e pylint

#==============================================================================
# Run the tests
#==============================================================================

.PHONY: test-fast #CMD Run only fast tests
test-fast: ${_INSTALL_TEST}
	@echo -e "${ECHO_PREFIX}running fast tests"
	${PREFIX}tox -e py37 --skip-pkg-install -- tests/fast

.PHONY: test-medium #CMD Run only medium-fast tests
test-medium: ${_INSTALL_TEST}
	@echo -e "${ECHO_PREFIX}running medium-fast tests"
	${PREFIX}tox -e py37 --skip-pkg-install -- tests/medium

.PHONY: test-slow #CMD Run only slow tests
test-slow: ${_INSTALL_TEST}
	@echo -e "${ECHO_PREFIX}running tests"
	${PREFIX}tox -e py37 --skip-pkg-install -- tests/slow

.PHONY: test #CMD Run all tests
test: ${_INSTALL_TEST}
	@echo -e "${ECHO_PREFIX}running all tests"
	${PREFIX}tox -e py37 --skip-pkg-install

.PHONY: test-iso #CMD Run all tests in isolation
test-iso: ${_INSTALL_TEST}
	@echo -e "${ECHO_PREFIX}running all tests in isolation"
	${PREFIX}tox -e py37

.PHONY: test-all #CMD Run tests on all specified Python versions with tox.
test-all: ${_INSTALL_TEST}
	@echo -e "${ECHO_PREFIX}running tests in isolated environments"
	${PREFIX}tox

#==============================================================================
# Documentation
#==============================================================================

# .PHONY: docs #CMD Generate HTML documentation, including API docs.
# docs: ${_INSTALL_DEV}
#	@echo -e "${ECHO_PREFIX}generating HTML documentation"
# 	\rm -f docs/{{ cookiecutter.project_slug }}.rst
# 	\rm -f docs/modules.rst
# 	${PREFIX}sphinx-apidoc -o docs/ src/{{ cookiecutter.project_slug }}
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	${browser} docs/_build/html/index.html

# .PHONY: servedocs #CMD Compile the docs watching for changes.
# servedocs: docs
#	@echo -e "${ECHO_PREFIX}continuously regenerating HTML documentation"
# 	${PREFIX}watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

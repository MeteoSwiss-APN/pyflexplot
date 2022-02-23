SHELL := /bin/bash

.DEFAULT_GOAL := help

#==============================================================================
# Options
#==============================================================================

CHAIN ?= 0#OPT Whether to chain targets, e.g., let test depend on install-test
IGNORE_VENV ?= 0#OPT Don't create and/or use a virtual environment
MSG ?= #OPT Message used as, e.g., tag annotation in version bump commands
PYTHON ?= 3.9.7#OPT Python version used to create conda virtual environment
VENV_DIR ?= #OPT Path to existing or new conda virtual environment (overrides VENV_NAME)
VENV_NAME ?= pyflexplot#OPT Name of conda virtual environment (overridden by VENV_DIR)

# Default values used below (caution: keep values in sync with above)
DEFAULT_VENV_NAME = pyflexplot#
export DEFAULT_VENV_NAME

#------------------------------------------------------------------------------

# Options for all calls to up-do-date pip (i.e., AFTER `pip install -U pip`)
# Example: `--use-feature=2020-resolver` before the new resolver became the default
PIP_OPTS = # --use-feature=in-tree-build is now default in pip
export PIP_OPTS

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
	_INSTALL := venv
	_INSTALL_DEV := venv
else
	_INSTALL := install
	_INSTALL_DEV := install-dev
endif
export _INSTALL
export _INSTALL_DEV

_TMP_VENV := $(shell date +venv-tmp-%s)
export _TMP_VENV

# If conda base environment is active, ignore it
ifeq (${CONDA_DEFAULT_ENV},base)
	CONDA_DEFAULT_ENV := #
	export CONDA_DEFAULT_ENV
endif

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
	@echo -e "\n[make clean-all] cleaning up"

.PHONY: clean-build #CMD Remove build artifacts.
clean-build:
	@echo -e "\n[make clean-build] removing build artifacts"
	\rm -rf "build/"
	\rm -rf "dist/"
	\rm -rf ".eggs/"
	@\rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)
	@\rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)

.PHONY: clean-pyc #CMD Remove Python file artifacts.
clean-pyc:
	@echo -e "\n[make clean-pyc] removing Python file artifacts"
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyc'       -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyo'       -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*~'          -exec echo "rm -rf '{}'" \;)
	@\rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '__pycache__' -exec echo "rm -rf '{}'" \;)

.PHONY: clean-test #CMD Remove testing artifacts.
clean-test:
	@echo -e "\n[make clean-test] removing testing artifacts"
	\rm -rf ".tox/"
	\rm -f ".coverage"
	\rm -rf "htmlcov/"
	\rm -rf ".pytest_cache"
	\rm -rf ".mypy_cache"

.PHONY: clean-venv #CMD Remove virtual environment.
clean-venv:
ifeq (${IGNORE_VENV}, 0)
	@# Do not ignore existing venv
ifeq (${VENV_DIR},)
	@# Path to conda venv has not been passed
ifneq ($(shell conda list --name $(VENV_NAME) 2>/dev/null 1>&2; echo $$?),0)
	@echo -e "\n[make clean-venv] no conda virtual environment '${VENV_NAME}' to remove"
else
	@echo -e "\n[make clean-venv] removing conda virtual environment '${VENV_NAME}'"
	conda env remove --yes --name ${VENV_NAME}
endif
else
	@# Path to conda venv has been passed
ifneq ($(shell conda list --prefix $(VENV_DIR) 2>/dev/null 1>&2; echo $$?),0)
	@echo -e "\n[make clean-venv] no conda virtual environment at '${VENV_DIR}' to remove"
else
	@echo -e "\n[make clean-venv] removing conda virtual environment at '${VENV_DIR}'"
	conda env remove --yes --prefix ${VENV_DIR}
endif
endif
endif

#==============================================================================
# Version control
#==============================================================================

# Note: Check if git is initialized by checking error status of git rev-parse
.PHONY: git #CMD Initialize a git repository and make initial commit.
git:
	@git rev-parse 2>&1 && $(MAKE) _git_ok || $(MAKE) _git_init

# Hidden target called when git is already initialized
.PHONY: _git_ok
_git_ok:
	@echo -e "\n[make git] git already initialized"

# Hidden target called when git is not yet initialized
.PHONY: _git_init
_git_init:
	$(MAKE) clean-all
	@echo -e "\n[make git] initializing git repository"
	\git init
	\git add .
	\git commit -m 'initial commit'
	\git --no-pager log -n1 --stat

#==============================================================================
# Conda Virtual Environment
#==============================================================================

.PHONY: venv #CMD Create/locate conda virtual environment.
venv: _create_conda_venv
	@# Use an active or existing conda env if possible, otherwise create new one
ifneq (${VIRTUAL_ENV},)
	@# An active Python virtual environment has been found, so abort this mission!
	@echo -e "[make venv] error: active non-conda virtual environment found: ${VIRTUAL_ENV}"
	exit 1
endif  # VIRTUAL_ENV
ifeq (${IGNORE_VENV}, 0)
	@# Don't ignore an existing conda env, and if there is none, create one
ifeq (${VENV_DIR},)
	@# Path VENV_DIR to conda env has NOT been passed
	$(eval VENV_DIR = $(shell conda run --name $(VENV_NAME) python -c 'import pathlib, sys; print(pathlib.Path(sys.executable).parent.parent)'))
	@export VENV_DIR
else  # VENV_DIR
	@# Path VENV_DIR to conda venv has been passed
	$(eval VENV_DIR = $(shell conda run --prefix $(VENV_DIR) python -c 'import pathlib, sys; print(pathlib.Path(sys.executable).parent.parent)'))
	@export VENV_DIR
ifneq (${VENV_NAME},${DEFAULT_VENV_NAME})
	@# Name VENV_NAME of conda env has been passed alongside path VENV_DIR
	@echo -e "[make venv] warning: VENV_DIR=${VENV_DIR} overrides VENV_NAME=${VENV_NAME}"
endif  # VENV_NAME
endif  # VENV_DIR
	$(eval PREFIX = ${VENV_DIR}/bin/)
	@export PREFIX
endif  # IGNORE_VENV
	${PREFIX}python -V

.PHONY: _create_conda_venv
_create_conda_venv: git
	@# If there is an active conda environment, use it, regardless of VENV_NAME and VENV_DIR;
	@# if there is an existing env matching VENV_NAME and (if set) VENV_DIR, use that;
	@# otherwise create a new one based on VENV_NAME and (if set) VENV_DIR.
ifeq (${IGNORE_VENV}, 0)
	@# Do not ignore existing venv
ifneq (${CONDA_DEFAULT_ENV},)
	@# There already is an active conda environment, so use it (regardless of its name and path)
	@echo -e "\n[make venv] found active conda environment '${CONDA_DEFAULT_ENV}' at '${CONDA_PREFIX}'"
ifneq (${CONDA_DEFAULT_ENV}, ${VENV_NAME})
	@# The name of the active env does not match VENV_NAME, but we assume that's OK over override it
	@echo -e "[make venv] warning: name of active venv '${CONDA_DEFAULT_ENV}' overrides VENV_NAME='${VENV_NAME}'"
	$(eval VENV_NAME = ${CONDA_DEFAULT_ENV})
	@export VENV_NAME
endif  # CONDA_DEFAULT_ENV
ifneq (${CONDA_PREFIX}, ${VENV_DIR})
	@# The path to the active env does not match VENV_DIR (if set), but we assume that's OK and override it
ifneq (${VENV_DIR},)
	@echo -e "[make venv] warning: path to active venv '${CONDA_PREFIX}' overrides VENV_DIR='${VENV_DIR}'"
endif  # VENV_DIR
	$(eval VENV_DIR = ${CONDA_PREFIX})
	@export VENV_DIR
endif  # CONDA_PREFIX
else  # CONDA_DEFAULT_ENV
	@# The is no active conda environment
ifeq (${VENV_DIR},)
	@# Path VENV_DIR to conda env has NOT been passed
ifeq ($(shell conda list --name $(VENV_NAME) 2>/dev/null 1>&2; echo $$?),0)
	@# Conda venv with name VENV_NAME already exists, so use it
	@echo -e "\n[make venv] conda virtual environment '${VENV_NAME}' already exists"
else  # shell conda ...
	@# Conda env with name VENV_NAME doesn't exist yet, so create it
	@echo -e "\n[make venv] creating conda virtual environment '${VENV_NAME}'"
	conda create -y --name "${VENV_NAME}" python==${PYTHON}
endif  # shell conda ...
else  # VENV_DIR
	@# Path to conda env VENV_DIR has been passed
ifeq ($(shell conda list --prefix $(VENV_DIR) 2>/dev/null 1>&2; echo $$?),0)
	@# Conda env at path VENV_DIR already exists, so use it
	@echo -e "\n[make venv] conda virtual environment at '${VENV_DIR}'"
else  # shell conda ...
	@# Conda env at path VENV_DIR does NOT yet exist, so create it
	@echo -e "\n[make venv] creating conda virtual environment at '${VENV_DIR}'"
	conda create -y --prefix "${VENV_DIR}" python==${PYTHON}
endif  # shell conda ...
endif  # VENV_DIR
endif  # CONDA_DEFAULT_ENV
endif  # IGNORE_VENV

#==============================================================================
# Installation
#==============================================================================

.PHONY: install #CMD Install the package with pinned runtime dependencies.
install: venv
	@echo -e "\n[make install] installing the package"
	conda env update --prefix "${VENV_DIR}" --file=environment.yml
	# conda install --yes --prefix "${VENV_DIR}" --file requirements/requirements.txt  # pinned
	# conda install --yes --prefix "${VENV_DIR}" --file requirements/requirements.in  # unpinned
	# ${PREFIX}python -m pip install -U pip
	${PREFIX}python -m pip install . ${PIP_OPTS}
	${PREFIX}pyflexplot -V

.PHONY: install-dev #CMD Install the package as editable with pinned runtime and\ndevelopment dependencies.
install-dev: venv
	@echo -e "\n[make install-dev] installing the package as editable with development dependencies"
	conda env update --prefix "${VENV_DIR}" --file=dev-environment.yml
	# conda install --yes --prefix "${VENV_DIR}" --file requirements/dev-requirements.txt  # pinned
	# conda install --yes --prefix "${VENV_DIR}" --file requirements/requirements.in  # unpinned
	# conda install --yes --prefix "${VENV_DIR}" --file requirements/dev-requirements.in  # unpinned
	# ${PREFIX}python -m pip install -U pip
	${PREFIX}python -m pip install --editable . ${PIP_OPTS}
	${PREFIX}pre-commit install
	${PREFIX}pyflexplot -V

#==============================================================================
# Dependencies
#==============================================================================

.PHONY: update-run-deps #CMD Update pinned runtime dependencies based on setup.py;\nshould be followed by update-dev-deps (consider update-run-dev-deps)
update-run-deps: git
	@echo -e "\n[make update-run-deps] not yet implemented for conda"
	exit 1
	# @echo -e "\n[make update-run-deps] updating pinned runtime dependencies in requirements/requirements.txt"
	# \rm -f requirements/requirements.txt
	# @echo -e "temporary virtual environment: ${_TMP_VENV}-run"
	# python -m venv ${_TMP_VENV}-run
	# ${_TMP_VENV}-run/bin/python -m pip install -U pip
	# ${_TMP_VENV}-run/bin/python -m pip install . ${PIP_OPTS}
	# ${_TMP_VENV}-run/bin/python -m pip freeze | \grep -v '\<file:' > requirements/requirements.txt
	# \rm -rf ${_TMP_VENV}-run

.PHONY: update-dev-deps #CMD Update pinned development dependencies based on\nrequirements/dev-requirements.in; includes runtime dependencies in\nrequirements/requirements.txt
update-dev-deps: git
	@echo -e "\n[make update-dev-deps] not yet implemented for conda"
	exit 1
	# @echo -e "\n[make update-dev-deps] updating pinned development dependencies in requirements/requirements.txt"
	# \rm -f requirements/dev-requirements.txt
	# @echo -e "temporary virtual environment: ${_TMP_VENV}-dev"
	# python -m venv ${_TMP_VENV}-dev
	# ${_TMP_VENV}-dev/bin/python -m pip install -U pip
	# ${_TMP_VENV}-dev/bin/python -m pip install -r requirements/dev-requirements.in ${PIP_OPTS}
	# ${_TMP_VENV}-dev/bin/python -m pip install -r requirements/requirements.txt --no-deps ${PIP_OPTS}
	# ${_TMP_VENV}-dev/bin/python -m pip freeze > requirements/dev-requirements.txt
	# \rm -rf ${_TMP_VENV}-dev

# Note: Updating run and dev deps MUST be done in sequence
.PHONY: update-run-dev-deps #CMD Update pinned runtime and development dependencies
update-run-dev-deps:
	$(MAKE) update-run-deps
	$(MAKE) update-dev-deps

.PHONY: update-tox-deps #CMD Update pinned tox testing dependencies based on\nrequirements/tox-requirements.in
update-tox-deps: git
	@echo -e "\n[make update-tox-deps] not yet implemented for conda"
	exit 1
	# \rm -f requirements/tox-requirements.txt
	# @echo -e "\n[make update-tox-deps] updating pinned tox testing dependencies in requirements/tox-requirements.txt"
	# @echo -e "temporary virtual environment: ${_TMP_VENV}-tox"
	# python -m venv ${_TMP_VENV}-tox
	# ${_TMP_VENV}-tox/bin/python -m pip install -U pip
	# ${_TMP_VENV}-tox/bin/python -m pip install -r requirements/tox-requirements.in ${PIP_OPTS}
	# ${_TMP_VENV}-tox/bin/python -m pip freeze > requirements/tox-requirements.txt
	# \rm -rf ${_TMP_VENV}-tox

.PHONY: update-precommit-deps #CMD Update pinned pre-commit dependencies specified in\n.pre-commit-config.yaml
update-precommit-deps: git
	@echo -e "\n[make update-precommit-deps] not yet implemented for conda"
	exit 1
	# @echo -e "\n[make update-precommit-deps] updating pinned tox testing dependencies in .pre-commit-config.yaml"
	# python -m venv ${_TMP_VENV}-precommit
	# ${_TMP_VENV}-precommit/bin/python -m pip install -U pip
	# ${_TMP_VENV}-precommit/bin/python -m pip install pre-commit
	# ${_TMP_VENV}-precommit/bin/pre-commit autoupdate
	# \rm -rf ${_TMP_VENV}-precommit

.PHONY: update-deps #CMD Update all pinned dependencies (run, dev, tox, precommit)
update-deps: update-run-dev-deps update-tox-deps update-precommit-deps
	@echo -e "\n[make update-deps] updating all pinned dependencies (run, dev, tox, precommit)"

#==============================================================================
# Versioning
#==============================================================================

# Note:
# Bump2version v1.0.0 is incompatible with pre-commit hook fix-trailing-whitespace
# (https://ithub.com/c4urself/bump2version/issues/124), therefore we pre-commit,
# commit, and tag manually. Once the whitespace problem is fixed, this can again
# be done in one command:
#  @read -p "Please annotate new tag: " msg \
#  && ${PREFIX}bumpversion patch --verbose --tag-message="$${msg}"

.PHONY: bump-patch #CMD Increment patch component Z of version number X.Y.Z,\nincl. git commit and tag
bump-patch: ${_INSTALL_DEV}
ifeq ($(MSG),)
	@echo -e "\n[make bump-patch] Error: Please provide a description with MSG='...' (use '"'\\n'"' for multiple lines)"
else
	@echo -e "\n[make bump-patch] bumping version number: increment patch component\n"
	@echo -e '\nTag annotation:\n\n$(subst ',",$(MSG))\n'
	@${PREFIX}bumpversion patch --verbose --no-commit --no-tag && echo
	@${PREFIX}pre-commit run --files $$(git diff --name-only) && git add -u
	@git commit -m "new version v$$(cat VERSION) (patch bump)"$$'\n\n$(subst ',",$(MSG))' --no-verify && echo
	@git tag -a v$$(cat VERSION) -m $$'$(subst ',",$(MSG))'
	@echo -e "\ngit tag -n -l v$$(cat VERSION)" && git tag -n -l v$$(cat VERSION)
	@echo -e "\ngit log -n1" && git log -n1
endif
# ' (close quote that vim thinks is still open to get the syntax highlighting back in order)

.PHONY: bump-minor #CMD Increment minor component Y of version number X.Y.Z,\nincl. git commit and tag
bump-minor: ${_INSTALL_DEV}
ifeq ($(MSG),)
	@echo -e "\n[make bump-minor] Error: Please provide a description with MSG='...' (use '"'\\n'"' for multiple lines)"
else
	@echo -e '\nTag annotation:\n\n$(subst ',",$(MSG))\n'
	@${PREFIX}bumpversion minor --verbose --no-commit --no-tag && echo
	@${PREFIX}pre-commit run --files $$(git diff --name-only) && git add -u
	@git commit -m "new version v$$(cat VERSION) (minor bump)"$$'\n\n$(subst ',",$(MSG))' --no-verify && echo
	@git tag -a v$$(cat VERSION) -m $$'$(subst ',",$(MSG))'
	@echo -e "\ngit tag -n -l v$$(cat VERSION)" && git tag -n -l v$$(cat VERSION)
	@echo -e "\ngit log -n1" && git log -n1
endif
# ' (close quote that vim thinks is still open to get the syntax highlighting back in order)

.PHONY: bump-major #CMD Increment major component X of version number X.Y.Z,\nincl. git commit and tag
bump-major: ${_INSTALL_DEV}
ifeq ($(MSG),)
	@echo -e "\n[make bump-major] Error: Please provide a description with MSG='...' (use '"'\\n'"' for multiple lines)"
else
	@echo -e '\nTag annotation:\n\n$(subst ',",$(MSG))\n'
	@${PREFIX}bumpversion major --verbose --no-commit --no-tag && echo
	@${PREFIX}pre-commit run --files $$(git diff --name-only) && git add -u
	@git commit -m "new version v$$(cat VERSION) (major bump)"$$'\n\n$(subst ',",$(MSG))' --no-verify && echo
	@git tag -a v$$(cat VERSION) -m $$'$(subst ',",$(MSG))'
	@echo -e "\ngit tag -n -l v$$(cat VERSION)" && git tag -n -l v$$(cat VERSION)
	@echo -e "\ngit log -n1" && git log -n1
endif
# ' (close quote that vim thinks is still open to get the syntax highlighting back in order)

#==============================================================================
# Format and check the code
#==============================================================================

.PHONY: format #CMD Check and fix the code formatting
format: ${_INSTALL_DEV}
	@echo -e "\n[make format] checking and fixing code formatting"
	${PREFIX}pre-commit run --all-files

.PHONY: check #CMD Check the code for correctness and best practices
check: ${_INSTALL_DEV}
	@echo -e "\n[make check] checking code correctness and best practices"
	${PREFIX}tox --parallel -e mypy -e flake8 -e pylint

.PHONY: spellcheck #CMD Check for spelling errors
spellcheck: ${_INSTALL_DEV}
	@echo -e "\n[make spellcheck] checking for spelling errors"
	${PREFIX}codespell *.rst
	find src tests docs -name '*.rst' -exec ${PREFIX}codespell {} \+
	${PREFIX}codespell *.md
	find src tests docs -name '*.md' -exec ${PREFIX}codespell {} \+
	${PREFIX}codespell *.py
	find src tests docs -name '*.py' -exec ${PREFIX}codespell {} \+

#==============================================================================
# Run the tests
#==============================================================================

.PHONY: test-fast #CMD Run only fast tests in the development environment
test-fast: ${_INSTALL_DEV}
	@echo -e "\n[make test-fast] running fast tests locally"
	# ${PREFIX}tox -e py37 -- tests/fast
	${PREFIX}pytest tests/fast

.PHONY: test-medium #CMD Run only medium-fast tests in the development environment
test-medium: venv ${_INSTALL_TEST}
	@echo -e "\n[make test-medium] running medium-fast tests locally"
	# ${PREFIX}tox -e py37 -- tests/medium
	${PREFIX}pytest tests/medium

.PHONY: test-slow #CMD Run only slow tests in the development environment
test-slow: ${_INSTALL_DEV}
	@echo -e "\n[make test-slow] running slow tests locally"
	# ${PREFIX}tox -e py37 -- tests/slow
	${PREFIX}pytest tests/slow

.PHONY: test #CMD Run all tests in the development environment
test: ${_INSTALL_DEV}
	@echo -e "\n[make test] running all tests locally"
	# ${PREFIX}tox -e py37
	${PREFIX}pytest tests

.PHONY: test-iso #CMD Run all tests in an isolated environment
test-iso: ${_INSTALL_DEV}
	@echo -e "\n[make test-iso] running all tests in isolation"
	${PREFIX}tox -e py37

.PHONY: test-check #CMD Run tests and checks in an isolated environment
test-check: ${_INSTALL_DEV}
	@echo -e "\n[make test-check] running tests and checks in isolated environments"
	${PREFIX}tox --parallel

#==============================================================================
# Documentation
#==============================================================================

#.PHONY: docs #CMD Generate HTML documentation, including API docs.
#docs: ${_INSTALL_DEV}
#	@echo -e "\n[make docs] generating HTML documentation"
#	\rm -f docs/pyflexplot.rst
#	\rm -f docs/modules.rst
#	${PREFIX}sphinx-apidoc -o docs/ src/pyflexplot
#	$(MAKE) -C docs clean
#	$(MAKE) -C docs html
#	${browser} docs/_build/html/index.html

#.PHONY: servedocs #CMD Compile the docs watching for changes.
#servedocs: docs
#	@echo -e "\n[make servedocs] continuously regenerating HTML documentation"
#	${PREFIX}watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

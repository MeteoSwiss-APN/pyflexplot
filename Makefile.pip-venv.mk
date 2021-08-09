SHELL := /bin/bash

.DEFAULT_GOAL := help

#==============================================================================
# Options
#==============================================================================

CHAIN ?= 0#OPT Whether to chain targets, e.g., let test depend on install-test
IGNORE_VENV ?= 0#OPT Don't create and/or use a virtual environment
MSG ?= ""#OPT Message used as, e.g., tag annotation in version bump commands
VENV_DIR ?= venv#OPT Path to virtual environment to be created and/or used
VENV_NAME ?= pyflexplot#OPT Name of virtual environment if one is created

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

# Options for all calls to up-do-date pip (i.e., AFTER `pip install -U pip`)
# Example: `--use-feature=2020-resolver` before the new resolver became the default
PIP_OPTS = --use-feature=in-tree-build
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
	_VENV :=
	_INSTALL :=
	_INSTALL_EDIT :=
	_INSTALL_DEV :=
else
	_VENV := venv
	_INSTALL := install
	_INSTALL_EDIT := install-edit
	_INSTALL_DEV := install-dev
endif
export _VENV
export _INSTALL
export _INSTALL_EDIT
export _INSTALL_DEV

_TMP_VENV := $(shell date +venv-tmp-%s)
export _TMP_VENV

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
	@echo -e "\n[make clean-venv] removing virtual environment at '${VENV_DIR}'"
	\rm -rf "${VENV_DIR}"

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
# Virtual Environments
#==============================================================================

.PHONY: venv #CMD Create a virtual environment.
venv: git
ifeq (${IGNORE_VENV}, 0)
	$(eval PREFIX = ${PREFIX_VENV})
	@export PREFIX
ifeq (${VIRTUAL_ENV},)
	@echo -e "\n[make venv] creating virtual environment '${VENV_NAME}' at '${VENV_DIR}'"
	python -m venv ${VENV_DIR} --prompt='${VENV_NAME}'
	${PREFIX}python -m pip install -U pip
endif
endif

#==============================================================================
# Installation
#==============================================================================

.PHONY: install #CMD Install the package with pinned runtime dependencies.
install: ${_VENV}
	@echo -e "\n[make install] installing the package"
	# SR_NOTE Pinned deps fail on tsa, probably due to cartopy/shapely vs. geos/proj
	# ${PREFIX}python -m pip install -r requirements/requirements.txt ${PIP_OPTS}
	${PREFIX}python -m pip install . ${PIP_OPTS}

.PHONY: install-dev #CMD Install the package as editable with pinned runtime and\ndevelopment dependencies.
install-dev: ${_VENV}
	@echo -e "\n[make install-dev] installing the package as editable with development dependencies"
	# SR_NOTE Pinned deps fail on tsa, probably due to cartopy/shapely vs. geos/proj
	# ${PREFIX}python -m pip install -r requirements/dev-requirements.txt ${PIP_OPTS}
	${PREFIX}python -m pip install -r requirements/dev-requirements.in ${PIP_OPTS}
	${PREFIX}python -m pip install -e . ${PIP_OPTS}
	${PREFIX}pre-commit install

#==============================================================================
# Dependencies
#==============================================================================

.PHONY: update-run-deps #CMD Update pinned runtime dependencies based on setup.py;\nshould be followed by update-dev-deps (consider update-run-dev-deps)
update-run-deps: git
	@echo -e "\n[make update-run-deps] updating pinned runtime dependencies in requirements/requirements.txt"
	\rm -f requirements/requirements.txt
	@echo -e "temporary virtual environment: ${_TMP_VENV}-run"
	python -m venv ${_TMP_VENV}-run
	${_TMP_VENV}-run/bin/python -m pip install -U pip
	${_TMP_VENV}-run/bin/python -m pip install . ${PIP_OPTS}
	${_TMP_VENV}-run/bin/python -m pip freeze | \grep -v '\<file:' > requirements/requirements.txt
	\rm -rf ${_TMP_VENV}-run

.PHONY: update-dev-deps #CMD Update pinned development dependencies based on\nrequirements/dev-requirements.in; includes runtime dependencies in\nrequirements/requirements.txt
update-dev-deps: git
	@echo -e "\n[make update-dev-deps] updating pinned development dependencies in requirements/requirements.txt"
	\rm -f requirements/dev-requirements.txt
	@echo -e "temporary virtual environment: ${_TMP_VENV}-dev"
	python -m venv ${_TMP_VENV}-dev
	${_TMP_VENV}-dev/bin/python -m pip install -U pip
	${_TMP_VENV}-dev/bin/python -m pip install -r requirements/dev-requirements.in ${PIP_OPTS}
	${_TMP_VENV}-dev/bin/python -m pip install -r requirements/requirements.txt --no-deps ${PIP_OPTS}
	${_TMP_VENV}-dev/bin/python -m pip freeze > requirements/dev-requirements.txt
	\rm -rf ${_TMP_VENV}-dev

# Note: Updating run and dev deps MUST be done in sequence
.PHONY: update-run-dev-deps #CMD Update pinned runtime and development dependencies
update-run-dev-deps:
	$(MAKE) update-run-deps
	$(MAKE) update-dev-deps

.PHONY: update-tox-deps #CMD Update pinned tox testing dependencies based on\nrequirements/tox-requirements.in
update-tox-deps: git
	\rm -f requirements/tox-requirements.txt
	@echo -e "\n[make update-tox-deps] updating pinned tox testing dependencies in requirements/tox-requirements.txt"
	@echo -e "temporary virtual environment: ${_TMP_VENV}-tox"
	python -m venv ${_TMP_VENV}-tox
	${_TMP_VENV}-tox/bin/python -m pip install -U pip
	${_TMP_VENV}-tox/bin/python -m pip install -r requirements/tox-requirements.in ${PIP_OPTS}
	${_TMP_VENV}-tox/bin/python -m pip freeze > requirements/tox-requirements.txt
	\rm -rf ${_TMP_VENV}-tox

.PHONY: update-precommit-deps #CMD Update pinned pre-commit dependencies specified in\n.pre-commit-config.yaml
update-precommit-deps: git
	@echo -e "\n[make update-precommit-deps] updating pinned tox testing dependencies in .pre-commit-config.yaml"
	python -m venv ${_TMP_VENV}-precommit
	${_TMP_VENV}-precommit/bin/python -m pip install -U pip
	${_TMP_VENV}-precommit/bin/python -m pip install pre-commit
	${_TMP_VENV}-precommit/bin/pre-commit autoupdate
	\rm -rf ${_TMP_VENV}-precommit

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
ifeq ($(MSG), "")
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
ifeq ($(MSG), "")
	@echo -e "\n[make bump-minor] Error: Please provide a description with MSG='...' (use '"'\\n'"' for multiple lines)"
	@echo -e '\nTag annotation:\n\n$(subst ',",$(MSG))\n'
	@${PREFIX}bumpversion minor --verbose --no-commit --no-tag && echo
	@${PREFIX}pre-commit run --files $$(git diff --name-only) && git add -u
	@git commit -m "new version v$$(cat VERSION) (minor bump)"$$'\n\n$(subst ',",$(MSG))' --no-verify && echo
	@git tag -a v$$(cat VERSION) -m $$'$(subst ',",$(MSG))'
	@echo -e "\ngit tag -n -l v$$(cat VERSION)" && git tag -n -l v$$(cat VERSION)
	@echo -e "\ngit log -n1" && git log -n1
else
endif
# ' (close quote that vim thinks is still open to get the syntax highlighting back in order)

.PHONY: bump-major #CMD Increment major component X of version number X.Y.Z,\nincl. git commit and tag
bump-major: ${_INSTALL_DEV}
ifeq ($(MSG), "")
	@echo -e "\n[make bump-major] Error: Please provide a description with MSG='...' (use '"'\\n'"' for multiple lines)"
	@echo -e '\nTag annotation:\n\n$(subst ',",$(MSG))\n'
	@${PREFIX}bumpversion major --verbose --no-commit --no-tag && echo
	@${PREFIX}pre-commit run --files $$(git diff --name-only) && git add -u
	@git commit -m "new version v$$(cat VERSION) (major bump)"$$'\n\n$(subst ',",$(MSG))' --no-verify && echo
	@git tag -a v$$(cat VERSION) -m $$'$(subst ',",$(MSG))'
	@echo -e "\ngit tag -n -l v$$(cat VERSION)" && git tag -n -l v$$(cat VERSION)
	@echo -e "\ngit log -n1" && git log -n1
else
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
test-medium: ${_INSTALL_TEST}
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

# .PHONY: docs #CMD Generate HTML documentation, including API docs.
# docs: ${_INSTALL_DEV}
# 	@echo -e "\n[make docs] generating HTML documentation"
# 	\rm -f docs/pyflexplot.rst
# 	\rm -f docs/modules.rst
# 	${PREFIX}sphinx-apidoc -o docs/ src/pyflexplot
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	${browser} docs/_build/html/index.html

# .PHONY: servedocs #CMD Compile the docs watching for changes.
# servedocs: docs
# 	@echo -e "\n[make servedocs] continuously regenerating HTML documentation"
# 	${PREFIX}watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

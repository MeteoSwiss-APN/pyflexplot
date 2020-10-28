SHELL := /bin/bash

.DEFAULT_GOAL := help

#==============================================================================
# Options
#==============================================================================

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
	_GIT :=
	_VENV :=
	_INSTALL :=
	_INSTALL_EDIT :=
	_INSTALL_DEV :=
else
	_GIT := git
	_VENV := venv
	_INSTALL := install
	_INSTALL_EDIT := install-edit
	_INSTALL_DEV := install-dev
endif
export _GIT
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

.PHONY: clean-venv #CMD Remove virtual environment.
clean-venv:
	@echo -e "\n[make clean-venv] removing virtual environment at '${VENV_DIR}'"
	\rm -rf "${VENV_DIR}"

#==============================================================================
# Version control
#==============================================================================

.PHONY: git #CMD Initialize a git repository and make initial commit.
git:
ifeq ($(shell git rev-parse >/dev/null 2>&1 && echo 0 || echo 1), 0)
	@echo -e "\n[make git] git already initialized"
else
	make clean-all
	@echo -e "\n[make git] initializing git repository"
	\git init
	\git add .
	\git commit -m 'initial commit'
	\git --no-pager log -n1 --stat
endif

#==============================================================================
# Virtual Environments
#==============================================================================

.PHONY: venv #CMD Create a virtual environment.
venv: ${_GIT}
	@git rev-parse 2>&1 && make _venv_run || make _venv_err

.PHONY: _venv_err
_venv_err:
	@echo -e "\n[make venv] error: git not initialized (run 'make git')"

.PHONY: _venv_run
_venv_run:
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
	${PREFIX}python -m pip install . --use-feature=2020-resolver

.PHONY: install-dev #CMD Install the package as editable with pinned runtime and development dependencies.
install-dev: ${_VENV}
	@echo -e "\n[make install-dev] installing the package as editable with development dependencies"
	${PREFIX}python -m pip install -r requirements/dev-pinned.txt --use-feature=2020-resolver
	${PREFIX}python -m pip install -e . --use-feature=2020-resolver
	${PREFIX}pre-commit install

#==============================================================================
# Dependencies
#==============================================================================

.PHONY: update-run-deps #CMD Update pinned runtime dependencies based on setup.py
update-run-deps:
	@echo -e "\n[make update-run-deps] updating pinned runtime dependencies in requirements/run-pinned.txt"
	@echo -e "temporary virtual environment: ${_TMP_VENV}-run"
	python -m venv ${_TMP_VENV}-run
	${_TMP_VENV}-run/bin/python -m pip install -U pip
	\rm -fv requirements.txt
	${_TMP_VENV}-run/bin/python -m pip install . --use-feature=2020-resolver; \
		ln -sfv requirements/run-pinned.txt requirements.txt
	${_TMP_VENV}-run/bin/python -m pip freeze | \grep -v '\<file:' > requirements/run-pinned.txt
	\rm -rf ${_TMP_VENV}-run

.PHONY: update-dev-deps #CMD Update pinned development dependencies based on\nrequirements/run-pinned.txt and requitements/dev-unpinned.txt
update-dev-deps:
	@echo -e "\n[make update-dev-deps] updating pinned development dependencies in requirements/dev-pinned.txt"
	@echo -e "temporary virtual environment: ${_TMP_VENV}-dev"
	python -m venv ${_TMP_VENV}-dev
	${_TMP_VENV}-dev/bin/python -m pip install -U pip
	${_TMP_VENV}-dev/bin/python -m pip install -r requirements/dev-unpinned.txt --use-feature=2020-resolver
	${_TMP_VENV}-dev/bin/python -m pip install -r requirements/run-pinned.txt --no-deps --use-feature=2020-resolver
	${_TMP_VENV}-dev/bin/python -m pip freeze > requirements/dev-pinned.txt
	\rm -rf ${_TMP_VENV}-dev

.PHONY: update-tox-deps #CMD Update pinned tox testing dependencies based on\nrequirements/tox-unpinned.txt
update-tox-deps:
	@echo -e "\n[make update-tox-deps] updating pinned tox testing dependencies in requirements/tox-pinned.txt"
	@echo -e "temporary virtual environment: ${_TMP_VENV}-tox"
	python -m venv ${_TMP_VENV}-tox
	${_TMP_VENV}-tox/bin/python -m pip install -U pip
	${_TMP_VENV}-tox/bin/python -m pip install -r requirements/tox-unpinned.txt --use-feature=2020-resolver
	${_TMP_VENV}-tox/bin/python -m pip freeze > requirements/tox-pinned.txt
	\rm -rf ${_TMP_VENV}-tox

.PHONY: update-precommit-deps #CMD Update pinned pre-commit dependencies\nspecified in .pre-commit-config.yaml
update-precommit-deps: ${_VENV}
	@echo -e "\n[make update-precommit-deps] updating pinned tox testing dependencies in .pre-commit-config.yaml"
	${PREFIX}python -m pip install pre-commit  # ensure pre-commit is installed
	${PREFIX}pre-commit autoupdate

.PHONY: update-deps #CMD Update all pinned dependencies (run, dev, tox, precommit)
update-deps: update-run-deps update-dev-deps update-tox-deps update-precommit-deps
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
	@echo -e "\n[make bump-patch] bumping version number: increment patch component\n"
	@${PREFIX}bumpversion patch --verbose --no-commit --no-tag && echo
	@${PREFIX}pre-commit run --files $$(git diff --name-only) && git add -u
	@git commit -m "new version v$$(cat VERSION) (patch bump)" --no-verify && echo
	@read -p "Please annotate new tag: " msg && git tag v$$(cat VERSION) -a  -m "$${msg}"
	@echo -e "\ngit tag -n v$$(cat VERSION)" && git tag -n v$$(cat VERSION)
	@echo -e "\ngit log -n1" && git log -n1

.PHONY: bump-minor #CMD Increment minor component Y of version number X.Y.Z,\nincl. git commit and tag
bump-minor: ${_INSTALL_DEV}
	@echo -e "\n[make bump-minor] bumping version number: increment minor component\n"
	@${PREFIX}bumpversion minor --verbose --no-commit --no-tag && echo
	@${PREFIX}pre-commit run --files $$(git diff --name-only) && git add -u
	@git commit -m "new version v$$(cat VERSION) (minor bump)" --no-verify && echo
	@read -p "Please annotate new tag: " msg && git tag v$$(cat VERSION) -a  -m "$${msg}"
	@echo -e "\ngit tag -n v$$(cat VERSION)" && git tag -n v$$(cat VERSION)
	@echo -e "\ngit log -n1" && git log -n1

.PHONY: bump-major #CMD Increment major component X of version number X.Y.Z,\nincl. git commit and tag
bump-major: ${_INSTALL_DEV}
	@echo -e "\n[make bump-major] bumping version number: increment major component\n"
	@${PREFIX}bumpversion major --verbose --no-commit --no-tag && echo
	@${PREFIX}pre-commit run --files $$(git diff --name-only) && git add -u
	@git commit -m "new version v$$(cat VERSION) (major bump)" --no-verify && echo
	@read -p "Please annotate new tag: " msg && git tag v$$(cat VERSION) -a  -m "$${msg}"
	@echo -e "\ngit tag -n v$$(cat VERSION)" && git tag -n v$$(cat VERSION)
	@echo -e "\ngit log -n1" && git log -n1

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
# 	\rm -f docs/{{ cookiecutter.project_slug }}.rst
# 	\rm -f docs/modules.rst
# 	${PREFIX}sphinx-apidoc -o docs/ src/{{ cookiecutter.project_slug }}
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	${browser} docs/_build/html/index.html

# .PHONY: servedocs #CMD Compile the docs watching for changes.
# servedocs: docs
# 	@echo -e "\n[make servedocs] continuously regenerating HTML documentation"
# 	${PREFIX}watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

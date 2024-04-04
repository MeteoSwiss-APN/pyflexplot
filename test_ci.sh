#!/bin/sh

run_tests_with_coverage() {
  python -m coverage run --source pyflexplot --data-file test_reports/.coverage --module pytest --junitxml=test_reports/junit.xml test/
  python -m coverage xml --data-file test_reports/.coverage -o test_reports/coverage.xml
}

run_pylint() {
  python -m pylint --output-format=parseable --exit-zero pyflexplot | tee test_reports/pylint.log
}


run_mypy() {
  mypy -p pyflexplot | grep error | tee test_reports/mypy.log
}

run_ci_tools() {
  run_tests_with_coverage && run_pylint && run_mypy
}

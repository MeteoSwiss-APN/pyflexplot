# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Types of Contributions

You can contribute in many ways.

### Report Bugs

Report bugs as [GitHub issues](https://github.com/MeteoSwiss-APN/pyflexplot/issues).

If you are reporting a bug, please include

- your operating system name and version,
- any details about your local setup that might be helpful in troubleshooting, and
- detailed steps to reproduce the bug.

### Fix Bugs

Look through the [GitHub issues](https://github.com/MeteoSwiss-APN/pyflexplot/issues) for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the  [GitHub issues](https://github.com/MeteoSwiss-APN/pyflexplot/issues) for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

PyFlexPlot could always use more documentation, whether as part of the official PyFlexPlot docs, in docstrings --- or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file a [GitHub issue]( https://github.com/MeteoSwiss-APN/pyflexplot/issues).

If you are proposing a feature,

- explain in detail how it would work;
- keep the scope as narrow as possible, to make it easier to implement; and
- remember that this is a volunteer-driven project, and that contributions are welcome! :)

## Get Started!

Ready to contribute? Here's how to set up `pyflexplot` for local development.

1. Fork the [`pyflexplot` repo](https://github.com/) on GitHub.
2. Clone your fork locally:

    ```bash
    git clone git@github.com:your_name_here/pyflexplot.git
    ```

3. Create a virtual environment and install the dependencies:

    ```bash
    cd pyflexplot/
    poetry install
    poetry run pre-commit install
    ```

4. Create a branch for local development:

    ```bash
    git switch -c name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5. When you're done with a change, format and check the code using various installed tools like `black`, `isort`, `mypy`, `flake8` or `pylint`. Those that are set up as pre-commit hooks can be run together with:

    ```bash
    pre-commit run -a
    ```

    Next, ensure that the code does what it is supposed to do by running the tests with pytest:

    ```bash
    pytest
    ```

6. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "fixed this and did that"
    git push origin name-of-your-bugfix-or-feature
    ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.
3. The pull request should work for Python 3.10, and for PyPy. Make sure that the tests pass for all supported Python versions.

## Tips

For a subset of tests or a specific test, run:

```bash
pytest tests.test_pyflexplot
pytest tests.test_pyflexplot/test_feature::test_edge_case
```

## Versioning

In order to release a new version of your project, follow these steps:

- Make sure everything is committed, cleaned up and validating (duh!). Don't forget to keep track of the changes in `HISTORY.md`.
- Increase the version number that is hardcoded in `pyproject.toml` (and only there) and commit.
- Either create a (preferentially annotated) tag with `git tag`, or directly create a release on GitHub.

## Project Structure

Following is a description of the most important files and folders in the project in alphabetic order.

- `docs/`: Documentation.
- `jenkins/`: Jenkins setup.
- `src/pyflexplot/`: Source code of the project package.
- `tests/`: Unit tests of the project package; run with `pytest`.
- `.gitignore`: Files and folders ignored by `git`.
- `.pre-commit-config.yaml`: Configuration of pre-commit hooks, which are formatters and checkers run before a successful commit.
- `AUTHORS.md`: Project authors.
- `CONTRIBUTING.md`: Instructions on how to contribute to the project.
- `HISTORY.md`: List of changes for each version of the project.
- `LICENSE`: License of the project.
- `MANIFEST.in`: Files installed alongside the source code.
- `pyproject.toml`: Main package specification file, including build dependencies, metadata and the configurations of development tools like `black`, `pytest`, `mypy` etc.
- `poetry.lock`: File containing all the dependency versions of pyflexplot.
- `README.md`: Description of the project.
- `USAGE.md`: Information on how to use the package.

## Managing dependencies

PyFlexPlot uses [poetry](https://python-poetry.org/) to manage dependencies. Dependencies are specified in the `pyproject.toml` file.

> poetry keeps all the dependency versions pinned in the poetry.lock file, which needs to be included in the repository.

## How to provide executable scripts

By default, a single executable script called pyflexplot is provided. It is created when the package is installed. When you call it, the main function (`cli`) in `src/pyflexplot/cli.py` is called.

When the package is installed, a executable script named `pyflexplot` is created in the bin folder of the active poetry environment. Upon calling this script in the shell, the `main` function in `src/pyflexplot/cli.py` is executed.

The scripts, their names and entry points are specified in `pyproject.toml` in the `[project.scripts]` section. Just add additional entries to provide more scripts to the users of your package.


### Testing and Coding Standards

Testing your code and compliance with the most important Python standards is a requirement for Python software written in SEN. To make the life of package
administrators easier, the most important checks are run automatically on GitHub actions. If your code goes into production, it must additionally be tested on CSCS
machines, which is only possible with a Jenkins pipeline (GitHub actions is running on a GitHub server).

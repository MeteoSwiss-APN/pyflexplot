.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/MeteoSwiss-APN/pyflexplot/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs.
Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features.
Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

PyFlexPlot could always use more documentation, whether as part of the official PyFlexPlot docs, in docstrings, or even on the web in blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/MeteoSwiss-APN/pyflexplot/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome! :)

Get Started!
------------

Ready to contribute? Here's how to set up `pyflexplot` for local development.

1. Fork the `pyflexplot` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pyflexplot.git

3. Install your local copy into a virtualenv. This is how you set up your fork for local development::

    $ cd pyflexplot/
    $ mkvirtualenv flexplot-dev
    $ pip install -r requirements/dev-requirements.txt

    $ python setup.py develop # or
    $ pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, format them with black, check that your changes pass the static code analyses with flake8 and the tests with pytest, including testing other Python versions with tox::

    $ black src
    $ tox  # runs pytest, pylint, flake8

   To get black, flake8, tox etc., just install them into your virtualenv (``pip install -r requirements/dev-requirements.txt``).

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add the feature to the list in ``README.rst``.
3. The pull request should work for Python 3.6 and 3.7, and for PyPy.
   Make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

    $ pytest tests.test_pyflexplot

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in ``HISTORY.rst``).
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

Jenkins will then deploy to PyPI if tests pass.

Project Structure
-----------------

.. list-table:: Structure
   :widths: 25 75
   :header-rows: 1

   * - File or directory
     - Description
   * - AUTHORS.rst
     - Contains information about the lead developer and contributors.
   * - .bumpversion.cfg
     - Configuration file of bumpversion.
   * - CONTRIBUTION.rst
     - Contains all the information you need when you contribute to this project.
   * - .gitignore
     - Files and directories ignored by git.
   * - HISTORY.rst
     - Lists the releases and their new features.
   * - LICENSE
     - License of this project.
   * - MANIFEST.in
     - Specifies the files and directories which will be added to the Pip package.
   * - Makefile
     - Build file for cleaning, installing the tool and its dependencies, for testing, formatting and linting code, and much more.
       Type ``make help`` to see all available commands.
   * - mypy.ini
     - Configuration file of mypy.
   * - .pre-commit-config.yaml
     - Configuration file of pre-commit, which, among other things, runs the formatters black and isort.
   * - pyproject.toml
     - Project specification file as defined by PEP 518.
   * - README.rst
     - Short documentation about the package.
       It lists features and contains a quick start.
   * - setup.py
     - Script used to build the package.
       It reads the unpinned top-level requirements from ``requirements/requirements.in`` into the variable ``requirements``.
   * - tox.ini
     - A configuration file for tox carring out the test for different Python verions.
       The listed versions should be the same as in the file ``setup.py``.
   * - USAGE.txt
     - Instructions on using pyflexplot.
   * - VERSION
     - Text file containing the current package version.
       Handled by bumpversion.
   * - docs/
     - Directory containing the documentation.
   * - requirements/
     - Directory containing requirements files with various types of dependencies.
   * - requirements/dev-requirements.in
     - A text file containing top-level unpinned development dependencies (critical version restrictions only).
       It is managed manually.
   * - requirements/dev-requirements.txt
     - A text file containing recursive pinned development and runtime dependencies (all versions specified), a superset of those in ``requirements/requirements.txt``.
       It is created automatically with ``pip freeze`` or the pip-tools command ``pip-compile``.
   * - requirements/requirements.in
     - A text file containing top-level unpinned runtime dependencies (critical version restrictions only).
       It is managed manually and read in ``setup.py``.
   * - requirements/requirements.txt
     - A text file containing recursive pinned runtime dependencies (all versions specified).
       It is created automatically with ``pip freeze`` or the pip-tools command ``pip-compile``.
   * - requirements/tox-requirements.in
     - A text file containing top-level unpinned testing dependencies (critical version restrictions only) used by tox as specified in ``tox.ini``.
       It is managed manually.
   * - requirements/tox-requirements.txt
     - A text file containing recursive pinned testing dependencies (critical version restrictions only) used by tox.
       It is created automatically with ``pip freeze`` or the pip-tools command ``pip-compile``.
   * - scripts/
     - Directory containing some small helper scripts.
   * - src/
     - Source directory.
   * - tests/
     - Directory containing the tests.
       The directory structure in this folder follows that in the source folder (src).
       For each file in the source folder, there is a file with the same name, but with the prefix ``text_``.
   * - various/
     - Directory containing various files.

Managing dependencies
---------------------

Generally, projects make use of other libraries, be it as (production) dependencies (e.g., ``import numpy`` in source code)
Which libraries -- and any critical restrictions of their versions -- have to be listed in different places in the project:

* Unpinned top-level runtime dependencies, which are required to run the application/library, belong in ``requirements/requirements.in`` (from which they are read in ``setup.py``).
  The versions of unpinned dependencies are only restricted as necessary, e.g., if a minimum version is required for a certain feature or bugfix.
* Unpinned top-level development dependencies, which are additional packages required during development, belong in ``requirements/dev-requirements.in``.
* Unpinned top-level testing dependencies, which are packages required by the testing framework ``tox`` to run unit tests, linters etc. as specified in ``tox.ini``, belong in ``requirements/tox-requirements.in``.
* Pinned runtime, development and testing dependencies belong in ``requirements/requirements.txt``, ``requirements/dev-requirements.txt`` and ``requirements/tox-requirements.txt``, respectively.
  Pinned dependencies are recursive, i.e., include all dependencies of dependencies, and restricted to a specific version.
  This ensures a reproducible environment that is guaranteed to work.

How to provide executable scripts
---------------------------------

By default, a single executable script called pyflexplot is provided.
It is created when the package is installed.
When you call it, the main function (``cli``) in ``src/pyflexplot/cli/cli.py`` is called.

How many scripts that are created, their names and which functions are called can be configured in the
``setup.py`` file.
The function ``setup`` has a named argument called ``entry_point`` which is a
dictionary with an element ``console_scripts``.
The element is an array of string.
For Example::

    entry_points={
        'console_scripts': [
            'pyflexplot=pyflexplot.cli.cli:cli',
    ],

When the package is installed, a executable script is created in the Python's bin folder with the name ``pyflexplot``.
If a user calls this script, the function ``cli`` in the file ``src/pyflexplot/cli/cli.py`` is called.
If more scripts should be created, add further entries to array ``console_scripts``.

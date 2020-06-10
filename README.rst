==========
PyFlexPlot
==========

PyFlexPlot is a Python-based tool to visualize FLEXPART dispersion simulation results stored in NetCDF format.

Installation
============

PyFlexPlot is hosted on `Github`_.
For the available releases, see `Releases`_.

.. _`Github`: https://github.com/MeteoSwiss-APN/pyflexplot
.. _`Releases`: https://github.com/MeteoSwiss-APN/pyflexplot/releases


At CSCS
-------

To install PyFlexPlot at CSCS, first prepare the Python environment::

    source /oprusers/osm/.opr_setup_dir
    source ${OPR_SETUP_DIR}/.python_base

Then follow the instructions for installation either with Pipx (for deployment) or by hand (for testing and development).

The locations where Pipx installs the code (takes some space) and links the commands (must be in `$PATH`) can be set with environment variables (e.g., in `.bashrc`).
On a system with persistent `$SCRATCH`, a suitable setup may look like this::

    # Location to install tools, libraries, etc.
    export LOCAL_SCRATCH="${SCRATCH}/local"
    export PATH="${LOCAL_SCRATCH}/bin:${PATH}"

    # Location where Pipx places virtualenvs
    export PIPX_HOME="${LOCAL_SCRATCH}/pipx"

    # Location where Pipx links commands (must be in $PATH)
    # Can be shared with other tools and manual installations
    export PIPX_BIN_DIR="${LOCAL_SCRATCH}/bin"

    # If necessary:  [we already did this above]
    # export PATH="${PIPX_BIN_DIR}:${PATH}"


With Pipx
---------

If you only want to use PyFlexPlot, `Pipx`_ provides a fast way to install it along with its dependencies into a designated virtual environment.
The virtual environment is entirely handled by `Pipx`_ -- there is no need for users to activate or otherwise case about it.

Install the latest version::

    pipx install git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot

Overwrite or upgrade an existing installation::

    pipx install git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot
    # or:
    pipx upgrade pyflexplot

Install a specific `release`_, e.g., 0.8.2 (the latest at the time of writing)::

    pipx install git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot@v0.8.2
    # or:
    pipx install --force git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot@v0.8.2

Specifying a release is recommended unless you specifically want the latest version (i.e., commit), as it prevents upgrading by accident.

.. _`release`: https://github.com/MeteoSwiss-APN/pyflexplot/releases

To kick PyFlexPlot off your system::

    pipx uninstall pyflexplot

Note that `Pipx`_ does nothing magic,
In fact, it merely saves you a few manual steps during installation:

    * cloning the repository,
    * creating a designated virtual environment,
    * installing the tool and its dependencies into it, and
    * symlinking all commands to a directory in PATH.

If Pipx is not already available on your system, you can easily install it manually::

    cd <local/installs>
    git clone git@github.com:pipxproject/pipx.git
    cd pipx
    python -m venv venv
    ./venv/bin/python -m pip install .
    cd <dir/in/PATH>
    ln -s <local/installs>/pipx/venv/bin/pipx

You can also install it with your system package manager, but this may tie it to your system Python installation (using system Python is bad!) and does not work easily with `Pyenv`_ (which you should check out!).

.. _`Pipx`: https://github.com/pipxproject/pipx
.. _`Pyenv`: https://github.com/pyenv/pyenv


By Hand
-------

Install PyFlexPlot manually from Github (i) to run the tests to verify that it works on your system, and/or (ii) to work on the code.
A range of useful commands for installation, testing, etc. are provided in a Makefile -- type `make` in the project root for a list of available commands.

In short::

    git@github.com:MeteoSwiss-APN/pyflexplot.git
    cd pyflexplot
    make install-dev
    make test-all

    ./venv/bin/pyflexplot -h
    # or:
    source ./venv/bin/activate
    pyflexplot -h

In detail::

    # Clone the repository
    git@github.com:MeteoSwiss-APN/pyflexplot.git
    cd pyflexplot
    make  # list available commands

    # Create a local virtual environment
    # If omitted, called by `make install*` commands
    make venv

    # Install tool and dependencies in virtual environment
    make install  # runtime dependencies only
    # or
    make install-test  # editable, run + test deps
    # or
    make install-dev  # editable, run + test + dev deps

    # Verify the installation (show help)
    ./venv/bin/pyflexplot -h
    # or
    source ./venv/bin/activate
    pyflexplot -h

    # Run tests
    make test  # all tests
    # or
    make test-fast  # fast tests only
    # or
    make test-medium  # fast and medium-fast tests only
    # or
    make test-all  # all tests and some checkers, in isolated envirnoment

Express::

    git@github.com:MeteoSwiss-APN/pyflexplot.git
    cd pyflexplot
    make test-all CHAIN=1
With `CHAIN=1`, the `make test*` commands run `make install-test` if necessary.


Usage
=====

To get a list of all available commands, just type::

    pyflexplot --help  # or -h

Plots -- including in- and output files -- are defined in setup files written in the `TOML`_ format.
(`TOML`_ files look similar to INI-files common in system configuration, but with a more well-defined syntax.)
Most command line flags are primarily useful during development and testing.

_`TOML`: https://github.com/toml-lang/toml

PyFlexPlot ships with a few sets of predefined plots for both operations and testing.
The standard operational deterministic dispersion plots based on COSMO-1 can be produced as follows::

    pyflexplot --preset=opr/cosmo1

This produces the plots defined in `pyflexplot/src/pyflexplot/data/presets/opr/cosmo1.toml` (check that file for input data paths etc.).
Specifically, it looks for a file matching `opr/cosmo1` (suffix omitted) in any preset path, which by default contains `pyflexplot/src/pyflexplot/data/presets`.

You can open all produced plots in an image viewer like `eog`::

    pyflexplot --open-all=eog --preset=test/cosmo1/concentration
    # or
    pyflexplot --open-first=feh --preset=test/cosmo2d/*

It's always good to double-check before-hand::

    pyflexplot --dry-run -vv --preset=opr/*

The presets interface is fairly powerful and useful during testing and development.
Some useful functionality includes::

    # List available presets (add `-v` or `-vv` for additional details)
    pyflexplot --preset=?
    # or:
    pyflexplot --preset-list

    # Use wild cards and multiple preset patterns
    pyflexplot --preset=test/cosmo1/* --preset=test/cosmo2e/basic_stats

    # Exclude some presets
    pyflexplot --preset=test/* --preset-skip=test/cosmo2d/multipanel*

    # Add your own search paths
    pyflexplot --preset-add my/presets --preset=foo/bar/*

While the plots are best specified in the setup files, sometimes you may want to change some parameters::

    pyflexplot --preset=test/cosmo1/deposition --setup lang en --setup domain ch

This will first read the setup files, and then substitute parameters you specified with `--setup` (removing duplicate specifications in the process).


Credits
-------

This package was created with `Cookiecutter`_ and the `MeteoSwiss-APN/mch-python-blueprint`_ project template.

.. _`Cookiecutter`: https://github.com/audreyr/cookiecutter
.. _`MeteoSwiss-APN/mch-python-blueprint`: https://github.com/MeteoSwiss-APN/mch-python-blueprint

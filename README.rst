##########
PyFlexPlot
##########

PyFlexPlot is a Python-based tool to visualize FLEXPART dispersion simulation
results stored in NetCDF format.

Installation
============

PyFlexPlot is hosted on `Github`_.
For the available releases, see `Releases`_.

.. _`Github`: https://github.com/MeteoSwiss-APN/pyflexplot
.. _`Releases`: https://github.com/MeteoSwiss-APN/pyflexplot/releases

With Conda
----------

Due to the dependency of cartopy on contemporary system libraries, it is the recommended
to install pyflexplot with conda. Conda is, as opposed to pipx and pip, not only able to
manage Python code, but also other items like system libraries.

To install conda, look for the latest Miniconda version on
https://docs.conda.io/en/latest/miniconda.html.

If you are working on a HPC system of the CSCS, consult the next section for more details.

The installation of pyflexplot needs some packages only available in the `conda-forge` channel.
Check your configured conda channels with::

    conda config --get channels

The list should contain 'conda-forge' with highest and 'defaults' with
a lower priority. If 'conda-forge' is missing, add it at the top of the
channel list with::

    conda config --add channels conda-forge

The following, although recommended by conda,
is not strictly necessary, and may even prevent conda from solving the environment
if you don't have conda-forge as priority channel::

    conda config --set channel_priority strict

When you have installed conda, copy the pyflexplot repository (approx. 260 MB)::

    git clone git@github.com:MeteoSwiss-APN/pyflexplot.git

Change into the base directory of the repository and check the links `Makefile` and `setup.py`::

    cd pyflexplot
    ls -l Makefile setup.py

They should point to `Makefile.conda.mk` and `setup.conda.py`.
If they do not, overwrite the links with::

    ln -sf Makefile.conda.mk Makefile
    ln -sf setup.conda.py setup.py

Check the Python version::

    python --version

If the python version provided by the system is < 3.7,
use the conda base environment for the further steps::

    conda activate

Display help for make options and targets::

    make help

Important: Before starting the installation, make sure none or the conda base
environment is active, otherwise pyflexplot will be installed in the active
conda environment instead of creating its own!

For the installation of pyflexplot with conda,
continue with the section `Installation from repository` further below.


At CSCS
-------

As contemporary system libraries are not available on the HPC systems
of the CSCS, it is highly recommended to use conda for the installation.
Install the latest Miniconda for 64-bit Linux. Choose a directory within
your `$SCRATCH` file system as installation location to
avoid cluttering the limited space on the user's `$HOME`.

For the same reason, it is better to clone the git repository to a
location within your `$SCRATCH` file system, although conda uses the
conda installation location to store the environment, unlike pipx and
pip, which by default use a `venv` subdirectory of the repository.

`More information for MeteoSwiss installation at CSCS <README.meteoswiss.rst>`__


Deployment with repository
++++++++++++++++++++++++++

The installation, starting with a git clone of the repository, can be achieved as follows::

    cd <local/installs>
    git clone git+ssh://github.com/MeteoSwiss-APN/pyflexplot --branch=v0.9.5 --depth=1 manual/git/pyflexplot/v0.9.5
    cd manual/git/pyflexplot/v0.9.5
    python -m venv manual/venvs/pyflexplot/v0.9.5
    make install VENV_DIR=<local/installs>/manual/venvs/pyflexplot/v0.9.5
    cd <dir/in/PATH>
    ln -s <local/installs>/manual/venvs/pyflexplot/v0.9.5/bin/pyflexplot pyflexplot_v0.9.5
    pyflexplot_v0.9.5 --version

Note that without `--depth=1`, the whole git history is downloaded, not just the tagged commit.
Also note that without `VENV_DIR=...`, the conda environment is created with
the default name ``pyflexplot``. If an environment other than base is activated,
pyflexplot is installed in the currently activated environment.


Installation from repository
----------------------------

The most convenient way to install, test and/or develop PyFlexPlot is by using the
Makefile, which provides commands for the most common operations related to
installation, testing etc. (and may also serve as a reference for the respective
Python commands).

Type `make help` (or just `make`)) in the root of the project to see all available
commands and options.
(Note that the options must come after `make <command>`, even though they look like
environment variables.)

Express::

    git clone git@github.com:MeteoSwiss-APN/pyflexplot.git
    cd pyflexplot
    make test CHAIN=1

(With `CHAIN=1`, the `make test*` commands first run `make install`.)

Short::

    git clone git@github.com:MeteoSwiss-APN/pyflexplot.git
    cd pyflexplot
    make install
    make test

    conda activate pyflexplot
    pyflexplot --help

    # or (if installed with pip):
    ./venv/bin/pyflexplot --help

    # or (if installed with pip):
    source ./venv/bin/activate
    pyflexplot --help

Details::

    # Clone the repository
    git clone git@github.com:MeteoSwiss-APN/pyflexplot.git
    cd pyflexplot
    make  # list available commands

    # Create a local virtual environment
    # If omitted, called by `make install*` commands
    make venv

    # Install tool and dependencies in virtual environment
    make install      # runtime dependencies only
    # or
    make install-dev  # editable, run + test + dev deps

    # Check if conda environment `pyflexplot` now exists
    conda info --env

    # Verify the installation (show help)
    conda activate pyflexplot
    pyflexplot --help
    # or (if installed with pip):
    ./venv/bin/pyflexplot --help
    # or (if installed with pip):
    source ./venv/bin/activate
    pyflexplot --help

    # Check if correct version is installed
    pyflexplot --version

    # Run tests
    make test  # all tests
    # or
    make test-fast  # fast tests only
    # or
    make test-medium  # fast and medium-fast tests only

Usage
=====

Activate the conda environment::

    conda activate pyflexplot

To get a list of all available commands, just type::

    pyflexplot --help  # or -h

Plots -- including in- and output files -- are defined in setup files written in the `TOML`_ format.
(`TOML`_ files look similar to INI-files common in system configuration,
but with a more well-defined syntax.)
Most command line flags are primarily useful during development and testing.

_`TOML`: https://github.com/toml-lang/toml

PyFlexPlot ships with a few sets of predefined plots for both operations and testing.
To get a list of available presets, use::

    pyflexplot --preset=?

To see the contents of one of the presets, use::

    pyflexplot --preset-cat <preset>

The standard operational deterministic dispersion plots based on the COSMO-1E control run
can be produced as follows::

    pyflexplot --preset=opr/cosmo-1e-ctrl/all_pdf

This produces the plots defined in `pyflexplot/src/pyflexplot/data/presets/opr/cosmo-1e-ctrl.toml`
(check that file for input data paths etc.).
Specifically, it looks for a file matching `opr/cosmo-1e-ctrl` (suffix omitted) in any preset path,
which by default contains `pyflexplot/src/pyflexplot/data/presets`.

You can open all produced plots in an image viewer like `eog`::

    pyflexplot --open-all=eog --preset=test/cosmo-1e-ctrl/concentration
    # or
    pyflexplot --open-first=eog --preset=test/cosmo-2e/*

It's always good to double-check what pyflexplot would do before-hand::

    pyflexplot --dry-run -vv --preset=opr/*

The presets interface is fairly powerful and useful during testing and development.
Some useful functionality includes::

    # List available presets (add `-v` or `-vv` for additional details)
    pyflexplot --preset=?

    # Use wild cards and multiple preset patterns
    pyflexplot --preset=test/cosmo-1e-ctrl/* --preset=test/cosmo-2e/stats

    # Exclude some presets
    pyflexplot --preset=test/* --preset-skip=test/cosmo2d/multipanel*

While the plots are best specified in the setup files,
sometimes you may want to change some parameters::

    pyflexplot --preset=test/cosmo-1e-ctrl/deposition --setup lang en --setup domain ch

This will first read the setup files, and then substitute parameters you specified with `--setup`
(removing duplicate specifications in the process).

Credits
-------

This package was created with `Cookiecutter`_ and the `MeteoSwiss-APN/mch-python-blueprint`_
project template.

.. _`Cookiecutter`: https://github.com/audreyr/cookiecutter
.. _`MeteoSwiss-APN/mch-python-blueprint`: https://github.com/MeteoSwiss-APN/mch-python-blueprint

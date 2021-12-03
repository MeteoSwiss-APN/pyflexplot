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

If you use conda for the installation, skip the
following sections that describe the installation with pipx and pip.
Continue with the section  `Installation from repository` further below.

Preparation for installation with pipx
++++++++++++++++++++++++++++++++++++++

If you nevertheless wish to use pipx or pip, you first need to prepare
a suitable Python environment::

    source /oprusers/osm/.opr_setup_dir
    source ${OPR_SETUP_DIR}/.python_base

This gives you Python 3.7 instead of the system default 2.7.
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

Specifying a release is recommended unless you specifically want the latest
version (i.e., commit), as it prevents upgrading by accident.

As of early August 2020, Pipx supports a way to install multiple releases of the same command in parallel by adding a suffix, but only in its master branch, with a release containing this feature still pending.
It works as follows::

    pipx install git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot@v0.8.2 --suffix=_v0.8.2
    pipx install git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot@v0.9.3 --suffix=0.9.3
    pipx install git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot
    pyflexplot_v0.8.2 --version  # -> 0.8.2
    pyflexplot0.9.3 --version  # -> 0.9.3
    pyflexplot --version  # -> 0.9.5

.. _`release`: https://github.com/MeteoSwiss-APN/pyflexplot/releases

To remove PyFlexPlot from your system::

    pipx uninstall pyflexplot

Note that `Pipx`_ does nothing magic.
In fact, it merely saves you a few manual steps during installation:

    * creating a designated virtual environment,
    * installing the tool and its dependencies into it, and
    * symlinking all commands to a directory in PATH.

If Pipx is not already available on your system, you can easily install it manually::

    cd <local/installs>
    python -m venv venvs/pipx
    ./venvs/pipx/bin/python -m pip install git+ssh://github.com/pipxproject/pipx.git
    cd <dir/in/PATH>
    ln -s <local/installs>/venvs/pipx/bin/pipx

You can also install it with your system package manager, but this may tie it to your system Python installation (using system Python is bad!) and does not work easily with `Pyenv`_ (which you should check out).

.. _`Pipx`: https://github.com/pipxproject/pipx
.. _`Pyenv`: https://github.com/pyenv/pyenv

With Pip
--------

You may install PyFlexPlot manually from Github if you cannot or do not want to use Pipx, if you want to run tests to verify your installation, and/or if you want to develop it further.
If you only want to deploy the tool -- without easy access to the code, without running tests, etc. -- you can directly install it into a virtual environment with pip, without cloning the git repository yourself.
However, if you also want to test and/or develop, you first have to clone the git repository and can then use the Makefile (or manual commands) to install the tool for your purposes.

Deployment only
+++++++++++++++

Install the latest version of PyFlexPlot and its dependencies directly into a virtual environment::

    cd <local/installs>
    python -m venv manual/venvs/pyflexplot/master
    ./manual/venvs/pyflexplot/master/bin/python -m pip install -U pip git+ssh://github.com/MeteoSwiss-APN/pyflexplot
    cd <dir/in/PATH>
    ln -s <local/installs>/manual/venvs/pyflexplot/master/bin/pyflexplot
    pyflexplot --version

Same for a specific version::

    cd <local/installs>
    python -m venv manual/venvs/pyflexplot/v0.9.5
    ./manual/venvs/pyflexplot/v0.9.5/bin/python -m pip install -U pip git+ssh://github.com/MeteoSwiss-APN/pyflexplot@v0.9.5
    cd <dir/in/PATH>
    ln -s <local/installs>/manual/venvs/pyflexplot/v0.9.5/bin/pyflexplot pyflexplot_v0.9.5
    pyflexplot_v0.9.5 --version
    ln -s pyflexplot_v0.9.5 pyflexplot  # use as default version
    pyflexplot --version

Note that `manual/venvs/pyflexplot/v0.9.5` is merely a suggestion and can be adapted as desired.

Deployment with repository
++++++++++++++++++++++++++

The same installations as just described, but starting with a git clone of the repository, can be achieved as follows::

    cd <local/installs>
    git clone git+ssh://github.com/MeteoSwiss-APN/pyflexplot --branch=v0.9.5 --depth=1 manual/git/pyflexplot/v0.9.5
    cd manual/git/pyflexplot/v0.9.5
    python -m venv manual/venvs/pyflexplot/v0.9.5
    make install VENV_DIR=<local/installs>/manual/venvs/pyflexplot/v0.9.5
    cd <dir/in/PATH>
    ln -s <local/installs>/manual/venvs/pyflexplot/v0.9.5/bin/pyflexplot pyflexplot_v0.9.5
    pyflexplot_v0.9.5 --version

Note that without `--depth=1`, the whole git history is downloaded, not just the tagged commit.
Also note that without `VENV_DIR=...`, the virtual environment is created in `./venv` instead of in `<local/installs>/manual/venvs/pyflexplot/v0.9.5`.


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

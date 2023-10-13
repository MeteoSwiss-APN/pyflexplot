# PyFlexPlot
PyFlexPlot is a Python-based tool to visualize FLEXPART dispersion simulation results stored in NetCDF format.
## Table of Contents
- [Prerequisites and Cloning the Repository](#prerequisites-and-cloning-the-repository)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Usage Example](#usage-example)
- [The Developer's Guide through the Project](#the-developers-guide-through-the-project)
  - [Getting Started](#getting-started)
  - [Implemented Debugging Features](#implemented-debugging-features)
  - [Roadmap to your first Contribution](#roadmap-to-your-first-contribution)
  - [Testing and Coding Standards](#testing-and-coding-standards)
    - [Pre-commit on GitHub Actions](#pre-commit-on-github-actions)
    - [Jenkins](#jenkins)
- [Features](#features)
- [Credits](#credits)
- [External Links](#external-links)
- [License](#license)
## Prerequisites and Cloning the Repository
Before you get started with this repository, ensure you have the following software/tools installed on your system: Git, Python and conda/mamba.
To get a local copy of this repository, run following commands and naviate into the repository:
```bash
git clone https://github.com/MeteoSwiss-APN/pyflexplot <custom_directory_name>
cd <custom_directory_name>
```
## Quick Start
For a quick setup to use pyflexplot, run the following commands within the root folder:
1. Install pinned environment: ```tools/setup_env.sh```
2. Activate the environment and build the package:
```bash
conda activate pyflexplot
pip install --no-deps .
```
3. To check if the tool runs properly you can run the tests by running ```pytest tests```

## Usage
To utilize pyflexplot, first ensure you are in the root directory of the project and have activated the necessary conda environment:
```bash
cd <custom_directory_name>
conda activate <custom_environment_name>
```
The primary command for pyflexplot follows this structure:
```bash
pyflexplot [OPTIONS] CONFIG_FILE_DIRECTORY
```
To see the available options, run:
 ```bash
 pyflexplot -h
 ```
To utilize all available CPUs for the command, use the option:
```bash
--num-procs=$SLURM_CPUS_PER_TASK
```

### Usage Example
After you've set up pyflexplot ([Prerequisites and cloning the repository](#prerequisites-and-cloning-the-repository) and [Getting started](#getting-started)),
you'll need to specify a configuration file and an output directory.
Define an output directory:
```bash
dest=test_output/
```
Note: The directory will be created on run time if it doesn't already exist.

Furthermore, there are already several default config files available in the directory ```src/pyflexplot/data/presets/opr```.
To run the program for all presets in the PDF graphics format with the default input data, use:
 ```bash
preset='opr/*/all_pdf'
```
Alternatively, select a specific preset from the table below:
| Model            | Type                 | Preset                           |
|------------------|----------------------|----------------------------------|
| FLEXPART-IFS     | Global output:       | preset=opr/ifs-hres/all_pdf      |
| FLEXPART-IFS     | Europe output:       | preset=opr/ifs-hres-eu/all_pdf   |
| FLEXPART-COSMO   | deterministic output:| preset=opr/cosmo-1e-ctrl/all_pdf |
| FLEXPART-COSMO   | deterministic output:| preset=opr/cosmo-2e-ctrl/all_pdf |
| FLEXPART-COSMO-1E| ensemble output:     | preset=opr/cosmo-1e/all_pdf      |
| FLEXPART-COSMO-2E| ensemble output:     | preset=opr/cosmo-2e/all_pdf      |

After selecting a preset, run pyflexplot interactively:
 ```bash
pyflexplot --preset "$preset" --merge-pdfs --dest=$dest
```

## The Developer's Guide through the Project

### Getting Started

Once you created or cloned this repository, make sure the installation is running properly. Install the package dependencies with the provided script `setup_env.sh`.
Check available options with
```bash
tools/setup_env.sh -h
```
We distinguish pinned installations based on exported (reproducible) environments and free installations where the installation
is based on top-level dependencies listed in `requirements/requirements.yml`. If you start developing, you might want to do an unpinned installation and export the environment:

```bash
tools/setup_env.sh -u -e -n <custom_environment_name>
```
*Hint*: If you are the package administrator, it is a good idea to understand what this script does, you can do everything manually with `conda` instructions.

*Hint*: Use the flag `-m` to speed up the installation using mamba. Of course you will have to install mamba first (we recommend to install mamba into your base
environment `conda install -c conda-forge mamba`. If you install mamba in another (maybe dedicated) environment, environments installed with mamba will be located
in `<miniconda_root_dir>/envs/mamba/envs`, which is not very practical.

The package itself is installed with `pip`. For development, install in editable mode:

```bash
conda activate <custom_environment_name>
pip install --editable .
```

*Warning:* Make sure you use the right pip, i.e. the one from the installed conda environment (`which pip` should point to something like `path/to/miniconda/envs/<custom_environment_name>/bin/pip`).

Once your package is installed, navigate to the root directory and run the tests by typing:

```bash
cd <custom_directory_name>
conda activate <custom_environment_name>
pytest
```

If the tests pass, you are good to go. Make sure to update your requirement files and export your environments after installation
every time you add new imports while developing. Check the next section to find some guidance on the development process if you are new to Python and/or SEN.

As this package was created with the SEN Python blueprint, it comes with a stack of development tools, which are described in more detail on the [Website](https://meteoswiss-apn.github.io/mch-python-blueprint/). Here, we give a brief overview on what is implemented.

### Implemented Debugging Features
pyflexplot offers several debugging options to assist in troubleshooting and
refining your workflow (see ```pyflexplot -h```).
Here are some of the key debugging features:
 ```
--pdb / --no-pdb                Drop into debugger when an exception is raised.  [default: no-pdb]
--preset-cat PATTERN            Show the contents of preset setup files
--only N                        Only create the first N plots based on the given setup.
--raise / --no-raise            Raise exception in place of user-friendly but uninformative error message.
--dry-run                       Perform a trial run with no changes made.
--merge-pdfs-dry                Merge PDF plots even in a dry run.
```

### Roadmap to your first Contribution

Generally, the source code of your library is located in `src/<library_name>`. `cli.py` thereby serves as an entry
point for functionalities you want to execute from the command line and it is based on the Click library. If you do not need interactions with the command line, you should remove `cli.py`. Moreover, of course there exist other options for command line interfaces,
a good overview may be found [here](https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/), we recommend however to use click. The provided example
code should provide some guidance on how the individual source code files interact within the library. In addition to the example code in `src/<library_name>`, there are examples for
unit tests in `tests/<library_name>/`, which can be triggered with `pytest` from the command line. Once you implemented a feature (and of course you also
implemented a meaningful test ;-)), you are likely willing to commit it. First, go to the root directory of your package and run pytest.

```bash
conda activate <custom_environment_name>
cd <custom_directory_name>
pytest
```

If you use the tools provided by the blueprint as is, pre-commit will not be triggered locally but only if you push to the main branch
(or push to a PR to the main branch). If you consider it useful, you can set up pre-commit to run locally before every commit by initializing it once. In the root directory of
your package, type:

```bash
pre-commit install
```

If you run `pre-commit` without installing it before (line above), it will fail and the only way to recover it, is to do a forced reinstallation (`conda install --force-reinstall pre-commit`).
You can also just run pre-commit selectively, whenever you want by typing (`pre-commit run --all-files`). Note that mypy and pylint take a bit of time, so it is really
up to you, if you want to use pre-commit locally or not. In any case, after running pytest, you can commit and the linters will run at the latest on the GitHub actions server,
when you push your changes to the main branch. Note that pytest is currently not invoked by pre-commit, so it will not run automatically. Automated testing can be set up with
GitHub Actions or be implemented in a Jenkins pipeline (template for a plan available in `jenkins/`. See the next section for more details.


### Testing and Coding Standards

Testing your code and compliance with the most important Python standards is a requirement for Python software written in SEN. To make the life of package
administrators easier, the most important checks are run automatically on GitHub actions. If your code goes into production, it must additionally be tested on CSCS
machines, which is only possible with a Jenkins pipeline (GitHub actions is running on a GitHub server).

### Pre-commit on GitHub Actions

`.github/workflows/pre-commit.yml` contains a hook that will trigger the creation of your environment (unpinned) on the GitHub actions server and
then run various formatters and linters through pre-commit. This hook is only triggered upon pushes to the main branch (in general: don't do that)
and in pull requests to the main branch.

### Jenkins

A jenkinsfile is available in the `jenkins/` folder. It can be used for a multibranch jenkins project, which builds
both commits on branches and PRs. Your jenkins pipeline will not be set up
automatically. If you need to run your tests on CSCS machines, contact DevOps to help you with the setup of the pipelines. Otherwise, you can ignore the jenkinsfiles
and exclusively run your tests and checks on GitHub actions.

## Features

- TODO

## Credits

This package was created with [`copier`](https://github.com/copier-org/copier) and the [`MeteoSwiss-APN/mch-python-blueprint`](https://github.com/MeteoSwiss-APN/mch-python-blueprint) project template.

## External Links

* [copier](https://github.com/copier-org/copier) - Based library and CLI app for rendering project templates.
* [mch-python-blueprint](https://github.com/MeteoSwiss-APN/mch-python-blueprint) - Project template embedded in this project based on copier.


## License
This project is licensed under the terms of the MeteoSwiss. The full license text can be found in the [LICENSE](LICENSE) file.
In essence, you are free to use, modify, and distribute the software, provided the associated copyright notice and disclaimers are included.

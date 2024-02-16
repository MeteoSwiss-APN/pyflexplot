# PyFlexPlot

PyFlexPlot is a Python-based tool to visualize/post-process FLEXPART dispersion simulation results stored in NetCDF format.

## Table of Contents

- [Prerequisites and Cloning the Repository](#prerequisites-and-cloning-the-repository)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Usage Example](#usage-example)
- [Developer Notes](#developer-notes)
  - [Implemented Debugging Features](#implemented-debugging-features)
  - [Roadmap to your first Contribution](#roadmap-to-your-first-contribution)
  - [Testing and Coding Standards](#testing-and-coding-standards)
    - [Pre-commit on GitHub Actions](#pre-commit-on-github-actions)
    - [Jenkins](#jenkins)
    - [Updating the Test References](#updating-the-test-references)
- [Features](#key-features)
- [Credits](#credits)
- [External Links](#external-links)
- [License](#license)

## Prerequisites and Cloning the Repository

Before you get started with this repository, ensure you have the following software tools installed on your system: Git, Python and Conda.

To get a local copy of this repository, run the following commands and naviate into the repository:

```bash
git clone https://github.com/MeteoSwiss-APN/pyflexplot <custom_directory_name>
cd <custom_directory_name>
```

## Getting Started

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

The package itself is installed with `pip`. As all dependencies have already been installed by Conda, use the `--no-deps` option for pip. For development, install the package in "editable" mode:

```bash
conda activate <custom_environment_name>
pip install --no-deps --editable .
```

*Warning:* Make sure you use the right pip, i.e. the one from the installed conda environment (`which pip` should point to something like `path/to/miniconda/envs/<custom_environment_name>/bin/pip`).

Once your package is installed, navigate to the root directory and run the tests by typing:

```bash
cd <custom_directory_name>
conda activate <custom_environment_name>
pytest
```

If the tests pass, you are good to go. If not, contact the package administrator Stefan Ruedisuehli. Make sure to update your requirement files and export your environments after installation
every time you add new imports while developing. Check the next section to find some guidance on the development process if you are new to Python and/or SEN.

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
 pyflexplot --help
 ```

To utilize all available CPUs for the command, add the option:

```bash
 --num-procs=$SLURM_CPUS_PER_TASK
```

### Usage Example

After you've set up pyflexplot ([Prerequisites and cloning the repository](#prerequisites-and-cloning-the-repository) and [Getting started](#getting-started)),
you'll need to specify a configuration file and an output directory.
Define the variable `dest` to contain the name of the output directory, e.g.:

```bash
dest=plot-test/
```

Note: The directory will be automatically created if it doesn't already exist.

There are several default config files available under [`src/pyflexplot/data/presets/opr`](src/pyflexplot/data/presets/opr/).
To run the program for all presets that produce the graphics in PDF format, define the `preset` variable as:

 ```bash
preset='opr/*/all_pdf'
```

This preset however only works for the default (test) input files.

To produce graphics for a specific FLEXPART output, select the fitting specific preset from the table below and define the `preset` variable accordingly:

| Model                 | Type         | Define Preset Variable             |
|-----------------------|--------------|------------------------------------|
| FLEXPART-IFS          | global       | `preset=opr/ifs-hres/all_pdf`      |
| FLEXPART-IFS          | Europe       | `preset=opr/ifs-hres-eu/all_pdf`   |
| FLEXPART-COSMO-1E-CTRL| deterministic| `preset=opr/cosmo-1e-ctrl/all_pdf` |
| FLEXPART-COSMO-2E-CTRL| deterministic| `preset=opr/cosmo-2e-ctrl/all_pdf` |
| FLEXPART-COSMO-1E     | ensemble     | `preset=opr/cosmo-1e/all_pdf`      |
| FLEXPART-COSMO-2E     | ensemble     | `preset=opr/cosmo-2e/all_pdf`      |
| FLEXPART-ICON-CH1-CTRL| deterministic| `preset=opr/icon-ch1-ctrl/all_pdf` |
| FLEXPART-ICON-CH2-EPS | ensemble     | `preset=opr/icon-ch2-eps/all_pdf`  |

After selecting a preset, run pyflexplot interactively for the default test data:

```bash
pyflexplot --preset "$preset" --merge-pdfs --dest=$dest
```

Alternatively, specify the FLEXPART output file in NetCDF format as input data:

```bash
pyflexplot --preset "$preset" --merge-pdfs --dest=$dest --setup infile <netcdf-file>
```

For ensemble input, the placeholder `{ens_member:03}` may be used within the path of `<netcdf-file>`. Instead of `03` for `%03d`, other formats for the ensemble member field can be used.

## Developer Notes

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

Generally, the source code of your library is located in `src/<library_name>`. The blueprint will generate some example code in `mutable_number.py`, `utils.py` and `cli.py`. `cli.py` thereby serves as an entry
point for functionalities you want to execute from the command line, it is based on the Click library. If you do not need interactions with the command line, you should remove `cli.py`. Moreover, of course there exist other options for command line interfaces,
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
your package, type `pre-commit install`. If you run `pre-commit` without having it installed before, it will fail, and the only way to recover it is to do a forced reinstallation (`conda install --force-reinstall pre-commit`).

Alternatively, you can instead run `pre-commit` selectively, without installing and whenever you want, by typing

```
pre-commit run --all-files
```

Note that mypy and pylint take a bit of time, so it is really
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

### Updating the Test References

Pyflexplot includes a set of functionality tests that compare generated output against predefined reference data.
These reference files, which are in the .py format, are derived from and stored alongside the original data in the tests/data directory.
To update these references, uncomment the lines of code in the test file you wish to update and run the test.

## Key Features

### PDF and PNG Files

Pyflexplot allows to visualize data on a map plot and save the output in either PDF or PNG format. To utilize this feature, simply adjust the outfile variable with the appropriate file extension.
![Example Image Output](img/integrated_concentration_site-Goesgen_species-1_domain-full_lang-de_ts-20200217T0900.png)

### Shape File Generation

Furthermore, Pyflexplot provides the functionality to export data into shape files (.shp) to utilize them in GIS programs such as QGIS 3. The output is a ZIP archive containing the essential components of a shapefile: .shp, .dbf, .shx, .prj, and .shp.xml.
Key aspects of this feature include:
- __Filtering Zero Values__: The tool initially removes zero values from fields (e.g., concentration) before processing.
- __Logarithmic Transformation__: Field values undergo a log_10 transformation to optimize the visualization of data ranges.
- __Precision Handling__: The transformed field values are recorded with 15 decimal places, accommodating the precision limitations of some GIS software.
- __Metadata Storage__: Information, such as details about released materials, are stored within a .shp.xml file as metadata.

### Scaling the field values

Another feature is to manipulate the field values by scaling with an arbitrary factor. This factor can be set in the preset with the variable `multiplier`.

## Credits

This package was created with [`copier`](https://github.com/copier-org/copier) and the [`MeteoSwiss-APN/mch-python-blueprint`](https://github.com/MeteoSwiss-APN/mch-python-blueprint) project template.

## External Links

- [copier](https://github.com/copier-org/copier) - Based library and CLI app for rendering project templates.
- [mch-python-blueprint](https://github.com/MeteoSwiss-APN/mch-python-blueprint) - Project template embedded in this project based on copier.
- [pyshp](https://github.com/GeospatialPython/pyshp) - Python module to generate Shapefiles

## License

This project is licensed under the terms of the MeteoSwiss. The full license text can be found in the [LICENSE](LICENSE) file.
In essence, you are free to use, modify, and distribute the software, provided the associated copyright notice and disclaimers are included.

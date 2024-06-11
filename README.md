# PyFlexPlot

PyFlexPlot is a Python-based tool to visualize FLEXPART dispersion simulation results stored in NetCDF format.

## Table of Contents

- [Features](#key-features)
- [Installation](#installation)
- [Run pyflexplot](#run-pyflexplot)
  - [Examples](#examples-how-to-run-pyflexplot)
- [Development (CSCS)](#development)
- [External Links](#external-links)
- [License](#license)

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

## Installation

You can install pyflexplot from MCH pypi repository using pip:

```bash
pip install pyflexplot -i https://service.meteoswiss.ch/nexus/repository/python-all/simple
```

## Run pyflexplot

The primary command for pyflexplot follows this structure:

```bash
pyflexplot [OPTIONS] CONFIG_FILE_DIRECTORY
```

To see the available options, run:

```bash
pyflexplot --help
```

If you want to run the following examples interactively,
you may want to allocate parallel resources
with the help of SLURM (if available), e.g. 10 cores:

```bash
salloc -c 10
```

To use all allocated cpus, add the `--num-procs` option to the pyflexplot command
(note that for a complete pyflexplot command, the definition of a preset or input and output need to be added, see below):

```bash
pyflexplot --num-procs=$SLURM_CPUS_PER_TASK
```

Important: Free resources when done!

```bash
exit
```

### Examples how to run pyflexplot

Example using default input file.
This example assumes you are in the pyflexplot directory.

Default input files are searched for in  `./data`.
If you want to use the files as defined in the presets for your tests,
link thm into the root of the repository. At CSCS on Alps, use:

```bash
ln -s /store_new/mch/msopr/pyflexplot_testdata data
```

There are several default config files available under [`src/pyflexplot/data/presets/opr`](src/pyflexplot/data/presets/opr/).

To produce graphics for a specific FLEXPART output, select the
corresponding preset from the table below and define the `preset` variable accordingly:

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

You may use the `*` wildcard to operate `pyflexplot` with several presets at once. For example, to run pyflexplot with all presets 
that produce the graphics in PDF format for a specific
NWP model, define the `preset` variable as one of:

```bash
preset='opr/cosmo*/all_pdf'
preset='opr/icon*/all_pdf'
preset='opr/ifs*/all_pdf'
```

Note however that the preset in this form requires the respective default input files to be accessible through the `./data` directory link.

Define an output directory and create it, if it does not exist, e.g.

```bash
nwp=$(echo ${preset} | cut -d/ -f2 | sed 's/*//g') # Extract NWP model name
dest=plot_$nwp
mkdir $dest
```

After selecting a preset, you may run pyflexplot interactively for the default test data:

```bash
pyflexplot --preset "$preset" --merge-pdfs --dest=$dest
```

On the production server at the CSCS, it is however highly recommended to create and run a batch job using the `batchPP` utility:

```bash
batchPP -t 2 -T 10 -n pfp_$nwp -- \
  $CONDA_PREFIX/bin/pyflexplot --preset "$preset" \
  --merge-pdfs --dest=$dest --num-procs=\$SLURM_CPUS_PER_TASK
```

To use your own or an operational FLEXPART output file in NetCDF format as input for pyflexplot,
modify the settings of the preset with the help of the `--setup` option as follows:

```bash
pyflexplot --preset "$preset" --merge-pdfs --dest=$dest --setup infile <netcdf-file>
```

To use a FLEXPART ensemble as input, the placeholder `{ens_member:03}` may be used within the path of *\<netcdf-file\>*.
Instead of `03` (for `%03d`), another C-style field width for the ensemble member field can be used.

Example using operational Flexpart ensemble output based on ICON-CH2-EPS:

```bash
preset=opr/icon-ch2-eps/all_pdf  # Preset for ICON-CH2-EPS
nwp=$(echo ${preset} | cut -d/ -f2 | sed 's/*//g')  # Extract NWP model name
basetime=$(date --utc --date="today 00" +%Y%m%d%H)  # Recent base time
# Get name of first input file
infile000=$(echo /store_new/mch/msopr/osm/ICON-CH2-EPS/FCST${basetime:2:2}/${basetime:2:8}_???/flexpart_c/000/grid_conc_*_BEZ.nc)
infile=${infile000/\/000\//\/\{ens_member:03\}\/}   # Input file definition
dest=plot_${basetime:2:8}    # Output directory with base time of NWP model
mkdir $dest                  # Create output directory
# Submit job with the help of the batchPP utility
batchPP -t 1 -T 10 -n pfp-$nwp -- \
  $CONDA_PREFIX/bin/pyflexplot --preset $preset \
    --merge-pdfs --setup infile $infile --setup base_time $basetime --dest=$dest \
    --num-procs=\$SLURM_CPUS_PER_TASK
```

The following expamles use FLEXPART output generated with the `test-fp` script
in the `test_meteoswiss` subdirectory of the [flexpart](https://github.com/MeteoSwiss/flexpart) repository of MeteoSwiss. Define `FP_JOBS`
as path to the FLEXPART output files that are to be used as input for pyflexplot, e.g.

```bash
FP_JOBS=$SCRATCH/flexpart/job
```

Write output to a location where you have write access, e.g.

```bash
FP_OUT=$SCRATCH/flexpart/job
```

After additionally defining the `preset` as above and `nwp` as the
job name (directory name below FP_JOBS), create the output directory with

```bash
infile=$(echo $FP_JOBS/$nwp/output/*.nc)
basetime=$(cat $FP_JOBS/$nwp/output/plot_info)
dest=$FP_OUT/$nwp/plots
mkdir -p $dest
```

and submit the job with the batchPP command as above.

For ensembles, the `infile` needs to be a pattern rather than a single file:

```bash
infile000=$(echo $FP_JOBS/$nwp/output/000/*.nc)
infile=${infile000/\/000\//\/\{ens_member:03\}\/}
```

After job completion, list and visualize results e.g. with `evince`:

```bash
ls $dest/*pdf
evince $dest/*pdf
```

### Running Pyflexplot with S3 input (and output)

In order to download input NETCDF data from S3, and S3 URI can be specified as the setup parameter `infile` as below:

```bash
pyflexplot --preset "$preset" --merge-pdfs --dest=$dest --setup infile s3://<s3-bucket-name>/flexpart_cosmo-2e_2021030503_{ens_member:03d}_MUE.nc
```

In order to output the resulting plots to an S3 bucket, specify the S3 bucket name as the `--dest`. The plots will still be created locally at the dest dir path defined in the config/settings.yaml

```bash
pyflexplot --preset "$preset" --merge-pdfs --dest=s3://<s3-bucket-name>
```

## Development

__Prerequisites__: Git, [Miniconda](https://docs.anaconda.com/free/miniconda/) 
(for installation of Poetry) or 
[Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

### Install dependencies & start the service locally (CSCS)

Clone the repo and enter the project folder:

```bash
git clone git@github.com:MeteoSwiss-APN/pyflexplot.git && cd pyflexplot
```

Create an Conda (or mamba/micromamba) environment with only the desired Python version and activate:

```bash
conda create --yes --prefix ./.conda-env python=3.10
conda activate ./.conda-env
```

Install Poetry into this environment and
configure Poetry to not create a new virtual environment. If it detects an already enabled virtual (eg Conda) environment it will install dependencies into it:

```bash
conda install --yes poetry
poetry config --local virtualenvs.create false
```

Install packages:

```bash
poetry install
```

### Run the tests and quality tools

Run tests:

```bash
poetry run pytest
```

Run pylint to check code style of Python files (if any):

```bash
poetry run pylint src
```

Run mypy to check typing:

```bash
poetry run mypy
```

### Updating the Test References

Pyflexplot includes a set of functionality tests that compare generated output against predefined reference data.
These reference files, which contain summary dicts, begin with `ref_` and have
the nomal Python file ending `.py`, and are stored in the directory
`tests/slow/pyflexplot/test_plots`.
To update these reference files, uncomment the following line near the end
of the file
[`shared.py`](tests/slow/test_pyflexplot/test_plots/shared.py)
in the same directory:

```bash
_TestBase = _TestCreateReference
```
Then re-run the (slow) tests to generate the new reference files. After generating the new reference files, comment out the above line again
or simply revert the file with git.

## External Links

- [pyshp](https://github.com/GeospatialPython/pyshp) - Python module to generate Shapefiles

## License

This project is licensed under the terms of the MIT License. The full license text can be found in the [LICENSE](LICENSE) file.
In essence, you are free to use, modify, and distribute the software, provided the associated copyright notice and disclaimers are included.

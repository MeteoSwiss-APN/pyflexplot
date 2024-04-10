# PyFlexPlot

PyFlexPlot is a Python-based tool to visualize FLEXPART dispersion simulation results stored in NetCDF format.

## Table of Contents

- [Installation](#installation)
- [Run pyflexplot](#run-pyflexplot)
  - [Examples](#examples-how-to-run-pyflexplot)
- [Features](#key-features)
- [External Links](#external-links)
- [License](#license)

## Installation

You can install pyflexplot from MCH pypi repository using pip:

    pip install pyflexplot -i https://service.meteoswiss.ch/nexus/repository/python-all/simple

## Run pyflexplot

To use all allocated cpus, add the following option to the pyflexplot command

    --num-procs=$SLURM_CPUS_PER_TASK

If you want to run the following examples interatcively,
you may want do allocate parallel resources, e.g. 10 cores

    salloc -c 10

Run `pyflexplot`
Important: Free resources when done!

    exit

### Examples how to run pyflexplot

Example using default input file.
This example assumes you are in the pyflexplot directory.

Default input files are searched for in  `./data`.
Link the default input files if you want to use these for tests.

    ln -s /store/mch/msopr/pyflexplot_testdata data

Create an output directory

    exp=test
    dest=plot_$exp
    mkdir $dest

Run all presets for pdf graphics format with the default input data

    preset='opr/*/all_pdf'

or choose an appropriate preset (define the variable preset)

| Model            | Type                 | Preset                           |
|------------------|----------------------|----------------------------------|
| FLEXPART-IFS     | Global output:       | preset=opr/ifs-hres/all_pdf      |
| FLEXPART-IFS     | Europe output:       | preset=opr/ifs-hres-eu/all_pdf   |
| FLEXPART-COSMO   | deterministic output:| preset=opr/cosmo-1e-ctrl/all_pdf |
| FLEXPART-COSMO   | deterministic output:| preset=opr/cosmo-2e-ctrl/all_pdf |
| FLEXPART-COSMO-1E| ensemble output:     | preset=opr/cosmo-1e/all_pdf      |
| FLEXPART-COSMO-2E| ensemble output:     | preset=opr/cosmo-2e/all_pdf      |

and run pyflexplot with the chosen preset interactively

    pyflexplot --preset "$preset" --merge-pdfs --dest=$dest

or as a batch job (recommended)

    batchPP -t 2 -T 10 -n pfp_$exp -- \
      $CONDA_PREFIX/bin/pyflexplot --preset $preset \
        --merge-pdfs --dest=$dest --num-procs=\$SLURM_CPUS_PER_TASK

Example using operational Flexpart ensemble output
```
exp=test-2e
preset=opr/cosmo-2e/all_pdf
basetime=$(date --utc --date="today 00" +%Y%m%d%H)
infile000=$(echo /store/mch/msopr/osm/COSMO-2E/FCST${basetime:2:2}/${basetime:2:8}_5??/flexpart_c/000/grid_conc_*_BEZ.nc)
infile=${infile000/\/000\//\/\{ens_member:03\}\/}
dest=plot_${basetime:2:8}
mkdir $dest
batchPP -t 1 -T 10 -n pfp-$exp -- \
  $CONDA_PREFIX/bin/pyflexplot --preset $preset \
    --merge-pdfs --setup infile $infile --setup base_time $basetime --dest=$dest \
    --num-procs=\$SLURM_CPUS_PER_TASK
```

The following expamles use FLEXPART output generated with the test-fp script
in the test subdirectory of the flexpart repository of MeteoSwiss. Define FP_JOBS
as path to the FLEXPART output files, e.g.

    FP_JOBS=/scratch/kaufmann/flexpart/job

Write output to a location where you have write access, e.g.

    FP_OUT=$SCRATCH/flexpart/job

After additionally defining preset and exp, create the output directory
and submit a job with

    infile=$(echo $FP_JOBS/$exp/output/*.nc)
    basetime=$(cat $FP_JOBS/$exp/output/plot_info)
    dest=$FP_OUT/$exp/plots
    mkdir -p $dest

and submit the job with the batchPP command as above

Example: FLEXPART with COSMO-2E Control Run

    preset=opr/cosmo-2e-ctrl/all_pdf
    exp=1074

Example: FLEXPART with COSMO-1E Control Run

    preset=opr/cosmo-1e-ctrl/all_pdf
    exp=1076

Example: FLEXPART with COSMO-2E Ensemble Run.
For ensembles, the infile needs to be a pattern rather than a single file

    preset=opr/cosmo-2e/all_pdf
    exp=short-bug
    infile000=$(echo $FP_JOBS/$exp/output/000/*.nc)
    infile=${infile000/\/000\//\/\{ens_member:03\}\/}

Example: New test case with cloud crossing dateline from both sides

    exp=test
    preset=opr/ifs-hres/all_pdf
    infile=data/ifs-hres/grid_conc_20220406180000_Chmelnyzkyj.nc
    basetime=2022040612
    dest=plot_$exp
    mkdir $dest

submit job with batchPP command above

After job completion, list and visualize results with

    ls $dest/*pdf
    evince $dest/*pdf

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

## External Links

- [pyshp](https://github.com/GeospatialPython/pyshp) - Python module to generate Shapefiles

## License

This project is licensed under the terms of the MeteoSwiss. The full license text can be found in the [LICENSE](LICENSE) file.
In essence, you are free to use, modify, and distribute the software, provided the associated copyright notice and disclaimers are included.

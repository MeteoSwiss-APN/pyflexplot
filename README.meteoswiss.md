PyFlexPlot
==========

PyFlexPlot is a Python-based tool to visualize FLEXPART dispersion
simulation results stored in NetCDF format.

Installation
------------

See [README.md](README.md) for installation instructions.

Testing
-------

### Test using pyflexplot-test: this section needs to be updated!

Compare two versions (old and new) of pyflexplot with pyflexplot-test
pyflexplot-test takes two git tags and compares the resulting plots.

Install pyflexplot-test from git clone

    git clone git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot-test
    cd pyflexplot-test
    make install CHAIN=1

Make "editable" installation, creating soft links in the venv directory

    make install-dev CHAIN=1

Check installation of pyflexplot-test and prepare for tests

    pyflexplot-test --version  # show version
    pyflexplot-test --help     # show help


#### Automated test using pyflexplot-test

Example scripts to run test cases are in examples

    ls examples

The script run_pyflexplot_test.sh runs the default test cases
Important: Before running any of these scripts, adapt the version
in the variables "old" and "new" as required.

    $EDITOR examples/run_pyflexplot_test.sh

Run the script in a parallel environment
Submit it as parallel job to SLURM

    batchPP -t 3 -T 10 examples/run_pyflexplot_test.sh

or run in interactively

    salloc -c 10
    examples/run_pyflexplot_test.sh
    exit


#### Manual test using pyflexplot-test

Generic call of pyflexplot-test with placeholders
> Note: the result will be stored in `./pyflexplot-test/work`
```
pyflexplot-test --old-rev=<old-rev> --new-rev=<new-rev> \
    --preset=opr/cosmo-1e-ctrl/all_png --infile=<path/to/cosmo-1e/case/file.nc> \
    --preset=opr/ifs-hres-eu/all_png --infile=<path/to/ifs-hres-eu/case/file.nc> \
    --preset=opr/ifs-hres/all_png --infile=<path/to/ifs-hres/case/file.nc>
```

View result (generic call with placeholders)

    eog pyflexplot-test/work/<old-rev>_vs_<new-rev>/*.png

#### Examples for manual tests

Allocate parallel resources, e.g. 10 cores

    salloc -c 10

Compare 2 git tags based on a flexpart output file

Settings for call to pyflexplot-test
```
pyflexplot_test_home=$SCRATCH/pyflexplot-test
data=$SCRATCH/flexpart/job
old_rev=v0.15.3-post
new_rev=v0.15.4-pre
model=cosmo-1e-ctrl
testname=6releases
job=1033
preset=opr/$model/all_png
infile=$(cd $data ; echo $job/output/grid_conc_*.nc)
```
Run `pyflexplot-test` with above variables
```
pyflexplot-test --num-procs=$SLURM_CPUS_PER_TASK \
    --old-rev=$old_rev --new-rev=$new_rev \
    --install-dir=$pyflexplot_test_home/install \
    --data=$data \
    --preset=$preset \
    --work-dir=$pyflexplot_test_home/work/$model/$testname \
    --infile=$infile
```
For consecutive calls, you save time when adding the options
(if appropriate)
```
    --reuse-installs
    --reuse-plots
```
Run default test cases manually,
see also `pyflexplot-test/examples/run_pyflexplot_test.sh`
Check if still in parallel environment and if revisions defined
```
printenv SLURM_CPUS_PER_TASK
echo old-rev=$old_rev new-rev=$new_rev
```
Run default test cases
```
pyflexplot-test --num-procs=$SLURM_CPUS_PER_TASK --old-rev=$old_rev --new-rev=$new_rev \
    --install-dir=/scratch/kaufmann/pyflexplot-test/install \
    --data=/scratch/ruestefa/shared/test/pyflexplot/data \
    --preset=opr/cosmo-1e-ctrl/all_png \
    --preset=opr/ifs-hres-eu/all_png \
    --preset=opr/ifs-hres/all_png \
    --preset=opr/cosmo-1e/all_png \
    --preset=opr/cosmo-2e/all_png \
    --work-dir=/scratch/kaufmann/pyflexplot-test/work/cosmo-1e-ctrl/default \
    --work-dir=/scratch/kaufmann/pyflexplot-test/work/ifs-hres-eu/default \
    --work-dir=/scratch/kaufmann/pyflexplot-test/work/ifs-hres/default \
    --work-dir=/scratch/kaufmann/pyflexplot-test/work/cosmo-1e/default \
    --work-dir=/scratch/kaufmann/pyflexplot-test/work/cosmo-2e/default
```
Release allocated resources

    exit


Run pyflexplot
--------------

Activate the conda environment

    conda activate pyflexplot

To use all allocated cpus, add the following option to the pyflexplot command

    --num-procs=$SLURM_CPUS_PER_TASK

If you want to run the following examples interatcively,
you may want do allocate parallel resources, e.g. 10 cores

    salloc -c 10

Run `pyflexplot`
Important: Free resources when done!


    exit

Examples how to run pyflexplot
------------------------------

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


Commit a new version
--------------------

Bring all changes to the dev branch if they are not already there.

    git checkout dev

Copy or merge changes into dev

Create a new version number, indicating the prerelease status (--new-version
with release and build tags), regardless of uncommitted files (--allow-dirty)
and without committing it as new tag (--no-commit --no-tag)

    bumpversion --verbose --allow-dirty --no-commit --no-tag --new-version=1.0.6-pre-1 dummy


Save environment specificatons to allow for an exact replication of
environment with make install

    make install-dev
    conda env export --no-builds --file=environment.yml

Remove the line specifying the pyflexplot version and the preset

If plots change, create new reference plots
Situation: make test-slow fails (tests/slow/pyflexplot/test_plots/test_*.py)
Reason: Plot references (summary dicts; tests/slow/pyflexplot/test_plots/ref_*.py)
        have changed

Steps: In [`tests/slow/test_pyflexplot/test_plots/shared.py`](tests/slow/test_pyflexplot/test_plots/shared.py), uncomment the line


       _TestBase = _TestCreateReference

at the end of the file and re-install if not installed with install-dev.

Rerun test to generate new reference

    make test-slow

Revert tests/slow/pyflexplot/test_plots/shared.py

    git checkout tests/slow/test_pyflexplot/test_plots/shared.py

Inspect the changes to the reference

    git diff tests/slow/test_pyflexplot/test_plots

Stage and commit all changes in dev

    git add ...
    git commit ...

Merge changes into master
(unless you have merged them from master)

    git checkout master
    git merge dev

Increase version, commit, push to github including tag
Choose target according to change level:
bump-patch, bump-minor, bump-major

    make bump-patch MSG=<message>

Push commit and associated tag to GitHub

    git push
    git push --tag

Continue development, preferably in the dev branch

    git checkout dev
    git merge master

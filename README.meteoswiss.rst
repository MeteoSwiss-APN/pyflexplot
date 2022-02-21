##########
PyFlexPlot
##########

PyFlexPlot is a Python-based tool to visualize FLEXPART dispersion
simulation results stored in NetCDF format.

Installation
============

See README.rst for installation instructions.

Testing
=======

Test using pyflexplot-test, currently outdated: needs update!
-------------------------------------------------------------

Compare two versions (old and new) of pyflexplot with pyflexplot-test
pyflexplot-test takes two git tags and compares the resulting plots.

# Install pyflexplot-test from git clone
git clone git+ssh://git@github.com/MeteoSwiss-APN/pyflexplot-test
cd pyflexplot-test
make install CHAIN=1

# Make "editable" installation, creating soft links in the venv directory
make install-dev CHAIN=1

# Check installation of pyflexplot-test and prepare for tests
pyflexplot-test --version  # show version
pyflexplot-test --help     # show help


# Automated test using pyflexplot-test

# Example scripts to run test cases are in examples
ls examples

# The script run_pyflexplot_test.sh runs the default test cases
# Important: Before running any of these scripts, adapt the version
# in the variables "old" and "new" as required.
$EDITOR examples/run_pyflexplot_test.sh

# Run the script in a parallel environment
# Submit it as parallel job to SLURM
batchPP -t 3 -T 10 examples/run_pyflexplot_test.sh

# or run in interatively
salloc -c 10
examples/run_pyflexplot_test.sh
exit


# Manual test using pyflexplot-test

# Generic call of pyflexplot-test with placeholders
# Note: the result will be stored in ./pyflexplot-test/work
pyflexplot-test --old-rev=<old-rev> --new-rev=<new-rev> \
    --preset=opr/cosmo-1e-ctrl/all_png --infile=<path/to/cosmo-1e/case/file.nc> \
    --preset=opr/ifs-hres-eu/all_png --infile=<path/to/ifs-hres-eu/case/file.nc> \
    --preset=opr/ifs-hres/all_png --infile=<path/to/ifs-hres/case/file.nc>
# View result (generic call with placeholders)
eog pyflexplot-test/work/<old-rev>_vs_<new-rev>/*.png

# Examples for manual tests

# Allocate parallel resources, e.g. 10 cores
salloc -c 10

# Compare 2 git tags based on a flexpart output file

# Settings for call to pyflexplot-test
pyflexplot_test_home=$SCRATCH/pyflexplot-test
data=$SCRATCH/flexpart/job
old_rev=v0.15.3-post
new_rev=v0.15.4-pre
model=cosmo-1e-ctrl
testname=6releases
job=1033
preset=opr/$model/all_png
infile=$(cd $data ; echo $job/output/grid_conc_*.nc)
# Run pyflexplot-test with above variables
pyflexplot-test --num-procs=$SLURM_CPUS_PER_TASK \
    --old-rev=$old_rev --new-rev=$new_rev \
    --install-dir=$pyflexplot_test_home/install \
    --data=$data \
    --preset=$preset \
    --work-dir=$pyflexplot_test_home/work/$model/$testname \
    --infile=$infile
# For consecutive calls, you save time when adding the options
# (if appropriate)
    --reuse-installs
    --reuse-plots

# Run default test cases manually
# see also pyflexplot-test/examples/run_pyflexplot_test.sh
# Check if still in parallel environment and if revisions defined
printenv SLURM_CPUS_PER_TASK
echo old-rev=$old_rev new-rev=$new_rev
# Run default test cases
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

# Release allocated resources
exit


Usage
=====

Run pyflexplot
--------------

Activate the conda environment::
  conda activate pyflexplot

To use all allocated cpus, add the following option to the pyflexplot command::
  --num-procs=$SLURM_CPUS_PER_TASK

If you want to run the following examples interatcively,
you may want do allocate parallel resources, e.g. 10 cores::
  salloc -c 10

Important: Free resources when done!::
  exit

Examples how to run pyflexplot
------------------------------

Example using default input file
This example assumes you are in the pyflexplot directory.

Default input files are searched for in ./data
Link the default input files if you want to use these for tests.::
  ln -s /store/mch/msopr/pyflexplot_testdata data

Create an output directory::
  exp=106c
  dest=plot_$exp
  mkdir $dest

Run all presets for pdf graphics format, define preset as::
  preset='opr/*/all_pdf'

or run all presets for the COSMO-2E ensemble for pdf and png graphics format::
  preset='opr/cosmo-2e/all_*'

or choose another preset as appropriate::
  preset=opr/ifs-hres/all_pdf       # for FLEXPART-IFS      Global output:
  preset=opr/ifs-hres-eu/all_pdf    # for FLEXPART-IFS      Europe output:
  preset=opr/cosmo-1e-ctrl/all_pdf  # for FLEXPART-COSMO    deterministic output:
  preset=opr/cosmo-2e-ctrl/all_pdf  # for FLEXPART-COSMO    deterministic output:
  preset=opr/cosmo-1e/all_pdf       # for FLEXPART-COSMO-1E ensemble output:
  preset=opr/cosmo-2e/all_pdf       # for FLEXPART-COSMO-2E ensemble output:

and run pyflexplot with the chosen preset, either interactively::
  pyflexplot --preset "$preset" --merge-pdfs --dest=$dest

or as a batch job (recommended)::
  batchPP -t 2 -T 10 -n $exp "$CONDA_PREFIX/bin/pyflexplot --preset $preset --merge-pdfs --dest=$dest --num-procs=\$SLURM_CPUS_PER_TASK"

Example using operational Flexpart ensemble output::
  preset=opr/cosmo-2e/all_pdf
  basetime=2021112500
  site=BEZ
  infile000=$(echo /store/mch/msopr/osm/COSMO-2E/FCST${basetime:2:2}/${basetime:2:8}_5??/flexpart_c/000/grid_conc_*_${site}.nc)
  infile=${infile000/\/000\//\/\{ens_member:03\}\/}
  dest=plot_${basetime:2:8}
  mkdir $dest
  batchPP -t 1 -T 10 -n pfp-2e "$CONDA_PREFIX/bin/pyflexplot --preset $preset --merge-pdfs --setup infile $infile --setup base_time $basetime --dest=$dest --num-procs=\$SLURM_CPUS_PER_TASK"


Examples using arbitrary FLEXPART output files.

If the FLEXPART output was produced by the test-fp script, define
the corresponding location for the numbered job directories.::
  FP_JOBS=$SCRATCH/flexpart/job

Find the flexpart output file by the job number::
  job=....
  infile=FP_JOBS/$job/output/*.nc

The following examples use Flexpart output from::
  FP_JOBS=/scratch/kaufmann/flexpart/job

Write output to a location where you have write access, e.g.::
  FP_OUT=$SCRATCH/flexpart/job

After defining preset and job, create the output directory
and submit job with::
  infile=$FP_JOBS/$job/output/*.nc
  basetime=$(cat $FP_JOBS/$job/output/plot_info)
  dest=$FP_OUT/$job/plots
  mkdir -p $dest
  batchPP -t 1 -T 10 -n plot$job "$CONDA_PREFIX/bin/pyflexplot --preset $preset --merge-pdfs --setup infile $infile --setup base_time $basetime --dest=$dest --num-procs=\$SLURM_CPUS_PER_TASK"


Example FLEXPART with COSMO-2E Control Run::
  preset=opr/cosmo-2e-ctrl/all_pdf
  job=1074

Submit pyflexplot job

Example FLEXPART with COSMO-1E Control Run::
  preset=opr/cosmo-1e-ctrl/all_pdf
  job=1076

Submit job with same commands as above

Example: FLEXPART with COSMO-2E Ensemble Run
For ensembles, the infile needs to be a pattern rather than a single file::
  preset=opr/cosmo-2e/all_pdf
  FP_JOBS=/scratch/kaufmann/flexpart/job
  FP_OUT=$SCRATCH/flexpart/job
  job=short-bug
  infile000=$(echo $FP_JOBS/$job/output/000/*.nc)
  infile=${infile000/\/000\//\/\{ens_member:03\}\/}
  basetime=2021090612
  dest=$FP_OUT/$job/plots
  mkdir -p $dest

submit job with batchPP command above
after job completion, list and vizualize results with::
  ls $dest/*pdf
  evince $dest/*pdf


Commit a new version
--------------------

Bring all changes to the dev branch if they are not already there.::
  git checkout dev

Copy or merge changes into dev

Create a new version number, indicating the prerelease status (--new-version
with release and buld tags), regardless of uncommited files (--allow-dirty)
and without committing it as new tag (--no-commit --no-tag)::
  bumpversion --verbose --allow-dirty --no-commit --no-tag --new-version=1.0.6.dev1 dummy


Save environment specificatons to allow for an exact replication of the
environment with make install::
  make update-run-deps

Remove the line specifying the pyflexplot version and the preset in the file
environment.yml

If plots change, create new reference plots
Situation: make test-slow fails (tests/slow/pyflexplot/test_plots/test_*.py)
Reason: Plot references (summary dicts; tests/slow/pyflexplot/test_plots/ref_*.py)
        have changed
Steps: In ``tests/slow/test_pyflexplot/test_plots/shared.py``, uncomment the line::

       _TestBase = _TestCreateReference

at the end of the file and re-install if not installed with install-dev.

Rerun test to generate new reference::
  make test-slow

Revert tests/slow/pyflexplot/test_plots/shared.py::
  git checkout tests/slow/test_pyflexplot/test_plots/shared.py

Inspect the changes to the reference::
  git diff tests/slow/test_pyflexplot/test_plots

Stage and commit all changes in dev::
  git add ...
  git commit ...

Merge changes into master
(unless you have merged them from master)::
  git checkout master
  git merge dev

Increase version, commit, push to github including tag
Choose target according to change level:
bump-patch, bump-minor, bump-major::
  make bump-patch MSG=<message>

Push commit and associated tag to GitHub::
  git push
  git push --tag

Continue development, preferably in the dev branch::
  git checkout dev
  git merge master

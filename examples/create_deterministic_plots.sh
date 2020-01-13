#!/bin/bash

data="/scratch/ruestefa/flexpart_visualization/test/data"
infile="${data}/cosmo-1_2019052800.nc"

dest="./png"
mkdir -pv "${dest}"

outfile_fmt="${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png"

for lang in en de; do
for domain in auto ch; do

flags=(
    -i "${infile}"
    -o "${outfile_fmt}"
    --lang="${lang}"
    --simulation-type="deterministic"
    --domain="${domain}"
)

pyflexplot "${flags[@]}" --field=concentration --plot-var=auto               --species-id=2   --time-ind=3  --no-integrate --level-ind=0
pyflexplot "${flags[@]}" --field=concentration --plot-var=auto               --species-id=1   --time-ind=10 --integrate    --level-ind=0
pyflexplot "${flags[@]}" --field=deposition    --plot-var=auto               --species-id=2   --time-ind=3  --integrate    --deposition-type=tot
pyflexplot "${flags[@]}" --field=deposition    --plot-var=affected_area_mono --species-id=1+2 --time-ind=10 --integrate    --deposition-type=tot

done
done

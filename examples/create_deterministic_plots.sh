#!/bin/bash

data="/scratch/ruestefa/flexpart_visualization/test/data"
infile="${data}/cosmo-1_2019052800.nc"

dest="./png"
mkdir -pv "${dest}"

outfile_fmt="${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png"

for lang in en de; do
for domain in auto ch; do

pyflexplot --lang=${lang} deterministic concentration --domain=${domain} -i "${infile}" -o "${outfile_fmt}" --species-id=2 --level-ind=0 --time-ind=3 --no-integrate

pyflexplot --lang=${lang} deterministic concentration --domain=${domain} -i "${infile}" -o "${outfile_fmt}" --species-id=1 --level-ind=0 --time-ind=10 --integrate

pyflexplot --lang=${lang} deterministic deposition --domain=${domain} -i "${infile}" -o "${outfile_fmt}" --species-id=2 --deposition-type=tot --time-ind=3 --integrate

pyflexplot --lang=${lang} deterministic deposition --domain=${domain} -i "${infile}" -o "${outfile_fmt}" --plot-var=affected_area_mono --species-id=1+2 --deposition-type=tot --time-ind=10 --integrate

done
done

#!/bin/bash

data="/scratch/ruestefa/flexpart_visualization/test/data"

dest="./png"
mkdir -pv "${dest}"

pyflexplot --lang=en deterministic concentration --domain=auto -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=2 --level-ind=0 --time-ind=3 --no-integrate
pyflexplot --lang=de deterministic concentration --domain=auto -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=2 --level-ind=0 --time-ind=3 --no-integrate
pyflexplot --lang=en deterministic concentration --domain=ch -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=2 --level-ind=0 --time-ind=3 --no-integrate
pyflexplot --lang=de deterministic concentration --domain=ch -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=2 --level-ind=0 --time-ind=3 --no-integrate

pyflexplot --lang=en deterministic concentration --domain=auto -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=1 --level-ind=0 --time-ind=10 --integrate
pyflexplot --lang=de deterministic concentration --domain=auto -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=1 --level-ind=0 --time-ind=10 --integrate
pyflexplot --lang=en deterministic concentration --domain=ch -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=1 --level-ind=0 --time-ind=10 --integrate
pyflexplot --lang=de deterministic concentration --domain=ch -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=1 --level-ind=0 --time-ind=10 --integrate

pyflexplot --lang=en deterministic deposition --domain=auto -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=2 --deposition-type=tot --time-ind=3 --integrate
pyflexplot --lang=de deterministic deposition --domain=auto -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=2 --deposition-type=tot --time-ind=3 --integrate
pyflexplot --lang=en deterministic deposition --domain=ch -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=2 --deposition-type=tot --time-ind=3 --integrate
pyflexplot --lang=de deterministic deposition --domain=ch -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --species-id=2 --deposition-type=tot --time-ind=3 --integrate

pyflexplot --lang=en deterministic deposition --domain=auto -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --plot-var=affected_area_mono --species-id=1+2 --deposition-type=tot --time-ind=10 --integrate
pyflexplot --lang=de deterministic deposition --domain=auto -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --plot-var=affected_area_mono --species-id=1+2 --deposition-type=tot --time-ind=10 --integrate
pyflexplot --lang=en deterministic deposition --domain=ch -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --plot-var=affected_area_mono --species-id=1+2 --deposition-type=tot --time-ind=10 --integrate
pyflexplot --lang=de deterministic deposition --domain=ch -i "${data}/test_case_2.nc" -o "${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png" --plot-var=affected_area_mono --species-id=1+2 --deposition-type=tot --time-ind=10 --integrate

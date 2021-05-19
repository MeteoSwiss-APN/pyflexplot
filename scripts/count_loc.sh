#!/bin/bash
#
# Count lines of code.
#
# Dependencies:
#  - pygount (https://github.com/roskakori/pygount)
#  - datamash (https://www.gnu.org/software/datamash)
#

dirs=(${@})
[ ${#dirs[@]} -eq 0 ] && dirs=($(echo src/*/ tests/*/*/))

ntot=0
for dir in ${dirs[@]}
do
    [ "${dir: -1}" == "/" ] && dir="${dir:0: -1}"
    [ "${dir: -9}" == ".egg-info" ] && continue
    n=$(pygount -s=py ${dir} | cut -f1 | datamash sum 1)
    [ "${n}" == "" ] && n=0
    ntot=$((ntot + n))
    echo -e "${n}\t${ntot}\t${dir}"
done
echo "${ntot}"

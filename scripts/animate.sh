#!/bin/bash

if [ ${#} -lt 3 ]; then
    echo "usage: $(basename "${0}") INFILE1 INFILE2 ... OUTFILE" >&2
    exit 1
fi

infiles=("${@}")
outfile=${infiles[ -1]}
unset infiles[-1]

echo "${#infiles[@]} infiles:"
for infile in ${infiles[@]}; do
    echo "${infile}"
done
echo "outfile: ${outfile}"

delay=33  # in 0.01 s
loop=0  # 0 for infinite
convert -delay ${delay} -loop ${loop} ${infiles[@]} ${outfile%.gif}.gif

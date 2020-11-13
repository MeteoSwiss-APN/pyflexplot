#!/bin/bash

orig_dir="orig"
work_dir="work"
mkdir -pv "${work_dir}"

orig_names=(ctrl.png p10.png p50.png p90.png)
labels=(ctrl 10th 50th 90th)
result="composite.png"

origs=()
crops=()
for i in $(seq 0 3); do
    orig_name="${orig_names[$i]}"
    origs[$i]="${orig_dir}/${orig_name}"
    crops[$i]="${work_dir}/${orig_name%.png}-crop.png"
done

top_left="${work_dir}/top_left.png"
top_right="${work_dir}/top_right.png"
mid_right="${work_dir}/mid_right.png"
bot_right="${work_dir}/bot_right.png"
bot_left="${work_dir}/bot_left.png"

# Cut out boxes
orig="${origs[0]}"
echo "${orig} -> ${top_left}"
convert -gravity north-west -crop 886x58+14+14 +repage "${orig}" "${top_left}" || exit 1
echo "${orig} -> ${top_right}"
convert -gravity north-east -crop 226x58+12+14 +repage "${orig}" "${top_right}" || exit 1
echo "${orig} -> ${mid_right}"
convert -gravity south-east -crop 226x613+12+48 +repage "${orig}" "${mid_right}" || exit 1

# Cut out bottom labels
orig="${origs[1]}"
echo "${orig} -> ${bot_right}"
convert -gravity south-east -crop 226x12+12+35 +repage "${orig}" "${bot_right}" || exit 1
echo "${orig} -> ${bot_left}"
convert -gravity south-west -crop 886x12+14+35 +repage "${orig}" "${bot_left}" || exit 1

# Cut out and label maps
crop_map_arg=886x613+14+48
for i in $(seq 0 3); do
    orig="${origs[$i]}"
    label="${labels[$i]}"
    crop="${crops[$i]}"
    echo "${orig} -> ${crop}"
    convert -gravity south-west -crop ${crop_map_arg} +repage "${orig}" "${crop}" || exit 1
    mogrify -gravity north-west -pointsize 56 -annotate +25+25 "${label}" "${crop}" || exit 1
done

# Assemble a plot
function assemble()
{
    local map="${1}"
    local result="${2}"
    echo "... -> ${result}"
    local nb=12
    convert \
        -bordercolor white \
        \( \
            \( "${top_left}" -border ${nb}x${nb} \) \
            \( "${map}" -border ${nb}x0 \) \
            \( "${bot_left}" -border ${nb}x2 \) \
            -append \
        \) \
        \( \
            \( "${top_right}" -border 0x${nb} \) \
            \( "${mid_right}" -border 0x0 \) \
            \( "${bot_right}" -border 0x2 \) \
            -append \
        \) \
        +append \
        -trim -border 10x10 +repage \
        "${result}" \
        || return 1
}

# Reassemble original plot to double-check cropping
test="${work_dir}/test.png"
assemble "${crops[0]}" "${test}" || exit 1

# Assemble maps
grid="${work_dir}/grid.png"
echo "... -> ${grid}"
args=(
    -chop 2x2
    -bordercolor black -border 2x2
    -bordercolor white -border 15x10
    +append
)
convert \
    \( "${crops[0]}" "${crops[1]}" ${args[@]} \) \
    \( "${crops[2]}" "${crops[3]}" ${args[@]} \) \
    -append \
    -trim -resize 886x613 +repage \
    "${grid}" \
    || exit 1

# Assemble result plot
assemble "${grid}" "${result}" || exit 1

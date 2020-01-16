# -*- coding: utf-8 -*-
"""
Command examples.
"""
import click

from textwrap import dedent


def show_example(ctx, param, value):
    if value is None:
        return
    elif value == "naz-det-sh":
        naz_det_sh()
    else:
        raise NotImplementedError(f"example={value}")
    ctx.exit(0)


def naz_det_sh():
    """Bash scripts with deterministic NAZ plots."""
    s = """\
        #!/bin/bash

        data="/scratch/ruestefa/shared/flexpart_visualization/test/data"
        infile="${data}/cosmo-1_2019052800.nc"

        dest="./png"
        mkdir -pv "${dest}"

        outfile_con_fmt="${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png"
        outfile_dep_fmt="${dest}/test_case1_{variable}_species-{species_id}_time-{time_ind:02d}_domain-{domain}_{lang}.png"

        langs=(
            de
            en
        )

        domains=(
            auto
            ch
        )

        for lang in ${langs[@]}; do
        for domain in ${domains[@]}; do

        args=(
            "${infile}"
            --lang="${lang}"
            --simulation-type="deterministic"
            --domain="${domain}"
        )

        pyflexplot "${args[@]}" "${outfile_con_fmt}" --field=concentration --plot-type=auto               --species-id=2   --time-ind=3  --no-integrate --level-ind=0
        pyflexplot "${args[@]}" "${outfile_con_fmt}" --field=concentration --plot-type=auto               --species-id=1   --time-ind=10 --integrate    --level-ind=0
        pyflexplot "${args[@]}" "${outfile_dep_fmt}" --field=deposition    --plot-type=auto               --species-id=2   --time-ind=3  --integrate    --deposition-type=tot
        pyflexplot "${args[@]}" "${outfile_dep_fmt}" --field=deposition    --plot-type=affected_area_mono --species-id=1+2 --time-ind=10 --integrate    --deposition-type=tot

        done
        done"""
    click.echo(dedent(s))

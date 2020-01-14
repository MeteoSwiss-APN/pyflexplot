# -*- coding: utf-8 -*-
"""
Command examples.
"""
import click

from textwrap import dedent


def show_example(ctx, param, value):
    if value == "naz-det-sh":
        naz_det_sh()
    else:
        raise NotImplementedError(f"example '{value}'")
    ctx.exit(0)


def naz_det_sh():
    """Bash scripts with deterministic NAZ plots."""
    s = """\
        #!/bin/bash

        data="/scratch/ruestefa/flexpart_visualization/test/data"
        infile="${data}/cosmo-1_2019052800.nc"

        dest="./png"
        mkdir -pv "${dest}"

        outfile_fmt="${dest}/test_case1_{variable}_species-{species_id}_level-{level_ind}_time-{time_ind:02d}_domain-{domain}_{lang}.png"

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
            "${outfile_fmt}"
            --lang="${lang}"
            --simulation-type="deterministic"
            --domain="${domain}"
        )

        pyflexplot "${args[@]}" --field=concentration --plot-var=auto               --species-id=2   --time-ind=3  --no-integrate --level-ind=0
        pyflexplot "${args[@]}" --field=concentration --plot-var=auto               --species-id=1   --time-ind=10 --integrate    --level-ind=0
        pyflexplot "${args[@]}" --field=deposition    --plot-var=auto               --species-id=2   --time-ind=3  --integrate    --deposition-type=tot
        pyflexplot "${args[@]}" --field=deposition    --plot-var=affected_area_mono --species-id=1+2 --time-ind=10 --integrate    --deposition-type=tot

        done
        done"""
    click.echo(dedent(s))

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

        outfile_con_fmt="${dest}/test_case1_{variable}_species-{species_id}_level-{level_idx}_time-{time_idx:02d}_domain-{domain}_{lang}.png"
        outfile_dep_fmt="${dest}/test_case1_{variable}_species-{species_id}_time-{time_idx:02d}_domain-{domain}_{lang}.png"

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

        config_toml='
        [plot]
        infiles = ["'${infile}'"]
        outfile = "'${outfile_con_fmt}'"
        lang = "'${lang}'"
        domain = "'${domain}'"
        simulation_type = "'deterministic'"
        variable = "concentration"
        plot_type = "auto"
        level_idx = 0
        time_idx = 3
        species_id = 2
        integrate = false
        '
        pyflexplot --config=<(echo -e "${config_toml}")

        config_toml='
        [plot]
        infiles = ["'${infile}'"]
        outfile = "'${outfile_con_fmt}'"
        lang = "'${lang}'"
        domain = "'${domain}'"
        simulation_type = "'deterministic'"
        variable = "concentration"
        plot_type = "auto"
        level_idx = 0
        time_idx = 10
        species_id = 1
        integrate = true
        '
        pyflexplot --config=<(echo -e "${config_toml}")

        config_toml='
        [plot]
        infiles = ["'${infile}'"]
        outfile = "'${outfile_dep_fmt}'"
        lang = "'${lang}'"
        domain = "'${domain}'"
        simulation_type = "'deterministic'"
        variable = "deposition"
        deposition_type = "tot"
        plot_type = "auto"
        level_idx = 0
        time_idx = 3
        species_id = 2
        integrate = true
        '
        pyflexplot --config=<(echo -e "${config_toml}")

        config_toml='
        [plot]
        infiles = ["'${infile}'"]
        outfile = "'${outfile_dep_fmt}'"
        lang = "'${lang}'"
        domain = "'${domain}'"
        simulation_type = "'deterministic'"
        variable = "deposition"
        plot_type = "affected_area_mono"
        level_idx = 0
        time_idx = 10
        species_id = [1, 2]
        integrate = true
        '
        pyflexplot --config=<(echo -e "${config_toml}")

        done
        done"""
    click.echo(dedent(s))

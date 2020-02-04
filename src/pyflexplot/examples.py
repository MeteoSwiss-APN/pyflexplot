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
        field = "concentration"
        plot_type = "auto"
        level_idxs = [0]
        time_idxs = [3]
        species_ids = [2]
        integrates = [false]
        '
        pyflexplot --config=<(echo -e "${config_toml}")

        config_toml='
        [plot]
        infiles = ["'${infile}'"]
        outfile = "'${outfile_con_fmt}'"
        lang = "'${lang}'"
        domain = "'${domain}'"
        simulation_type = "'deterministic'"
        field = "concentration"
        plot_type = "auto"
        level_idxs = [0]
        time_idxs = [10]
        species_ids = [1]
        integrates = [true]
        '
        pyflexplot --config=<(echo -e "${config_toml}")

        config_toml='
        [plot]
        infiles = ["'${infile}'"]
        outfile = "'${outfile_dep_fmt}'"
        lang = "'${lang}'"
        domain = "'${domain}'"
        simulation_type = "'deterministic'"
        field = "deposition"
        deposition_types = ["tot"]
        plot_type = "auto"
        level_idxs = [0]
        time_idxs = [3]
        species_ids = [2]
        integrates = [true]
        '
        pyflexplot --config=<(echo -e "${config_toml}")

        config_toml='
        [plot]
        infiles = ["'${infile}'"]
        outfile = "'${outfile_dep_fmt}'"
        lang = "'${lang}'"
        domain = "'${domain}'"
        simulation_type = "'deterministic'"
        field = "deposition"
        plot_type = "affected_area_mono"
        level_idxs = [0]
        time_idxs = [10]
        species_ids = [[1, 2]]
        integrates = [true]
        '
        pyflexplot --config=<(echo -e "${config_toml}")

        done
        done"""
    click.echo(dedent(s))

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

        config_toml='

        [_base]
        infiles = ["'${infile}'"]
        lang = "en"
        domain = "auto"
        simulation_type = "deterministic"
        plot_type = "auto"
        level_idx = 0
        species_id = 2
        time_idx = 3

        [_base.conc]
        outfile = "'${outfile_con_fmt}'"
        variable = "concentration"
        integrate = false

        [_base.conc.de]
        lang = "de"

        [_base.conc.ch]
        domain = "ch"

        [_base.conc.ch.de]
        lang = "de"

        [_base.conc.int]
        time_idx = 10
        species_id = 1
        integrate = true

        [_base.conc.int.de]
        lang = "de"

        [_base.conc.int.ch]
        domain = "ch"

        [_base.conc.int.ch.de]
        lang = "de"

        [_base.depos]
        outfile = "'${outfile_dep_fmt}'"
        variable = "deposition"
        deposition_type = "tot"
        integrate = true

        [_base.depos.de]
        lang = "de"

        [_base.depos.ch]
        domain = "ch"

        [_base.depos.ch.de]
        lang = "de"

        [_base.depos.affect]
        plot_type = "affected_area_mono"
        time_idx = 10
        species_id = [1, 2]

        [_base.depos.affect.de]
        lang = "de"

        [_base.depos.affect.ch]
        domain = "ch"

        [_base.depos.affect.ch.de]
        lang = "de"
        '

        pyflexplot --config=<(echo -e "${config_toml}")

        """
    click.echo(dedent(s))

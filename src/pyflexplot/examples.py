# -*- coding: utf-8 -*-
"""
Command examples.
"""
# Standard library
from textwrap import dedent

# Third-party
import click


def print_example(ctx, param, value):
    if value is None:
        return
    choices = {
        "naz_det_toml": naz_det_toml,
        "naz_det_sh": naz_det_sh,
    }
    try:
        choice = choices[value]
    except AttributeError:
        raise NotImplementedError(f"example={value}")
    else:
        choice()
    ctx.exit(0)


def naz_det_toml():
    """
    TOML config file to create deterministic NAZ plots.
    """
    s = """\
        # PyFlexPlot config file to create deterministic NAZ plots

        [_base]
        infiles = [
            "/scratch/ruestefa/shared/flexpart_visualization/test/data/cosmo-1_2019052800.nc",
        ]
        lang = "en"
        domain = "auto"
        simulation_type = "deterministic"
        plot_type = "auto"
        level_idx = 0
        species_id = 2
        time_idx = 3

        [_base.concentration]
        outfile = "test_case1_{variable}_species-{species_id}_level-{level_idx}_time-{time_idx:02d}_domain-{domain}_{lang}.png"
        variable = "concentration"
        integrate = false

        [_base.concentration.lang_de]
        lang = "de"

        [_base.concentration.domain_ch]
        domain = "ch"

        [_base.concentration.domain_ch.lang_de]
        lang = "de"

        [_base.concentration.integrated]
        time_idx = 10
        species_id = 1
        integrate = true

        [_base.concentration.integrated.lang_de]
        lang = "de"

        [_base.concentration.integrated.domain_ch]
        domain = "ch"

        [_base.concentration.integrated.domain_ch.lang_de]
        lang = "de"

        [_base.deposition]
        outfile = "test_case1_{variable}_species-{species_id}_time-{time_idx:02d}_domain-{domain}_{lang}.png"
        variable = "deposition"
        deposition_type = "tot"
        integrate = true

        [_base.deposition.lang_de]
        lang = "de"

        [_base.deposition.domain_ch]
        domain = "ch"

        [_base.deposition.domain_ch.lang_de]
        lang = "de"

        [_base.deposition.affected_area]
        plot_type = "affected_area_mono"
        time_idx = 10
        species_id = [1, 2]

        [_base.deposition.affected_area.lang_de]
        lang = "de"

        [_base.deposition.affected_area.domain_ch]
        domain = "ch"

        [_base.deposition.affected_area.domain_ch.lang_de]
        lang = "de"
        """
    click.echo(dedent(s))


def naz_det_sh():
    """
    Bash script to create deterministic NAZ plots.
    """
    s = """\
        #!/bin/bash

        # Create deterministic NAZ plots
        pyflexplot --config=<(pyflexplot --example=naz_det_toml)
        """
    click.echo(dedent(s))

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
        outfile = "test_{variable}_{plot_type}_{domain}_{lang}_{time_idx:02d}.png"
        simulation_type = "deterministic"
        plot_type = "auto"
        level_idx = 0
        species_id = 2
        time_idx = 3

        [_base._concentration]
        variable = "concentration"
        integrate = false

        [_base._concentration._integr]
        time_idx = 10
        species_id = 1
        integrate = true

        [_base._deposition]
        variable = "deposition"
        deposition_type = "tot"
        integrate = true

        [_base._deposition._affected_area]
        plot_type = "affected_area_mono"
        time_idx = 10
        species_id = [1, 2]

        [_base._concentration._auto.en]
        domain = "auto"
        lang = "en"

        [_base._concentration._auto.de]
        domain = "auto"
        lang = "de"

        [_base._concentration._ch.en]
        domain = "ch"
        lang = "en"

        [_base._concentration._ch.de]
        domain = "ch"
        lang = "de"

        [_base._concentration._integr._auto.en]
        domain = "auto"
        lang = "en"

        [_base._concentration._integr._auto.de]
        domain = "auto"
        lang = "de"

        [_base._concentration._integr._ch.en]
        domain = "ch"
        lang = "en"

        [_base._concentration._integr._ch.de]
        domain = "ch"
        lang = "de"

        [_base._deposition._auto.en]
        domain = "auto"
        lang = "en"

        [_base._deposition._auto.de]
        domain = "auto"
        lang = "de"

        [_base._deposition._ch.en]
        domain = "ch"
        lang = "en"

        [_base._deposition._ch.de]
        domain = "ch"
        lang = "de"

        [_base._deposition._affected_area._auto.en]
        domain = "auto"
        lang = "en"

        [_base._deposition._affected_area._auto.de]
        domain = "auto"
        lang = "de"

        [_base._deposition._affected_area._ch.en]
        domain = "ch"
        lang = "en"

        [_base._deposition._affected_area._ch.de]
        domain = "ch"
        lang = "de"

        """  # noqa:E501
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

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
    try:
        choice = choices[value]
    except AttributeError:
        raise NotImplementedError(f"example={value}")
    else:
        choice()
    ctx.exit(0)


choices = {}


def naz_det_toml():
    """
    TOML config file to create deterministic NAZ plots.
    """
    s = """\
        # PyFlexPlot config file to create deterministic NAZ plots

        [_base]
        # Sampe data directory: /scratch/ruestefa/shared/flexpart_visualization/test/
        infiles = ["data/cosmo-1_2019052800.nc"]
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


choices["naz_det_toml"] = naz_det_toml


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


choices["naz_det_sh"] = naz_det_sh


def ens_thr_agrmt():
    """Plot ensemble threshold agreement."""
    s = """\
        # PyFlexPlot config file to create ensemble threshold agreement plots.

        [_base]
        # Sampe data directory: /scratch/ruestefa/shared/flexpart_visualization/test/
        infiles = ["data/cosmo-2e_2019073100_{member_id:03d}.nc"]
        simulation_type = "ensemble"
        member_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ]
        outfile = "test_{variable}_{domain}_{lang}_ts{time_idx:02d}_{member_ids}.png"
        plot_type = "ens_thr_agrmt"
        time_idx = 10

        [_base.en]
        lang = "en"

        [_base.de]
        lang = "de"

        """  # noqa:E501
    click.echo(dedent(s))


choices["ens_thr_agrmt"] = ens_thr_agrmt


def ens_basic_stats():
    """Plot basic ensemble statistical measures."""
    s = """\
        # PyFlexPlot config file to plot basic statistical measures.

        [_base]
        # Sampe data directory: /scratch/ruestefa/shared/flexpart_visualization/test/
        infiles = ["data/cosmo-2e_2019073100_{member_id:03d}.nc"]
        simulation_type = "ensemble"
        member_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ]
        outfile = "test_{variable}_{domain}_{lang}_ts{time_idx:02d}_{member_ids}.png"
        time_idx = 10

        [_base._concentration]
        variable = "concentration"

        [_base._deposition]
        variable = "deposition"
        deposition_type = "wet"

        [_base._concentration.mean]
        plot_type = "ens_mean"

        [_base._deposition.mean]
        plot_type = "ens_mean"

        [_base._concentration.median]
        plot_type = "ens_median"

        [_base._deposition.median]
        plot_type = "ens_median"

        [_base._concentration.min]
        plot_type = "ens_min"

        [_base._deposition.min]
        plot_type = "ens_min"

        [_base._concentration.max]
        plot_type = "ens_max"

        [_base._deposition.max]
        plot_type = "ens_max"

        """  # noqa:E501
    click.echo(dedent(s))


choices["ens_basic_stats"] = ens_basic_stats

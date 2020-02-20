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


def register_choice(fct):
    """Register a function by its name as a choice for --example."""
    choices[fct.__name__] = fct
    return fct


@register_choice
def naz_det():
    """
    TOML setup file to create deterministic NAZ plots.
    """
    s = """\
        # PyFlexPlot setup file to create deterministic NAZ plots

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

        # TODO Derive from `_concentration` once wildcards/hooks more sophisticated
        [_base._concentration_integrated]
        variable = "concentration"
        integrate = true
        time_idx = 10
        species_id = 1

        [_base._deposition]
        variable = "deposition"
        deposition_type = "tot"
        integrate = true

        # TODO Derive from `_deposition` once wildcards/hooks more sophisticated
        [_base._affected_area]
        variable = "deposition"
        deposition_type = "tot"
        integrate = true
        plot_type = "affected_area_mono"
        time_idx = 10
        species_id = [1, 2]

        [_base."*"._auto.en]
        domain = "auto"
        lang = "en"

        [_base."*"._auto.de]
        domain = "auto"
        lang = "de"

        [_base."*"._ch.en]
        domain = "ch"
        lang = "en"

        [_base."*"._ch.de]
        domain = "ch"
        lang = "de"

        """  # noqa:E501
    click.echo(dedent(s))


@register_choice
def ens_thr_agrmt():
    """Plot ensemble threshold agreement."""
    s = """\
        # PyFlexPlot setup file to create ensemble threshold agreement plots.

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


@register_choice
def ens_basic_stats():
    """Plot basic ensemble statistical measures."""
    s = """\
        # PyFlexPlot setup file to plot basic statistical measures.

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
        integrate = true

        [_base."*".mean]
        plot_type = "ens_mean"

        [_base."*".median]
        plot_type = "ens_median"

        [_base."*".min]
        plot_type = "ens_min"

        [_base."*".max]
        plot_type = "ens_max"

        """  # noqa:E501
    click.echo(dedent(s))


@register_choice
def ens_cloud_arrival_time():
    """Plot cloud arrival time."""
    s = """\
        # PyFlexPlot setup file to plot cloud arrival time.

        [_base]
        # Sampe data directory: /scratch/ruestefa/shared/flexpart_visualization/test/
        infiles = ["data/cosmo-2e_2019073100_{member_id:03d}.nc"]
        simulation_type = "ensemble"
        member_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ]
        outfile = "test_{variable}_{domain}_{lang}_ts{time_idx:02d}_{member_ids}.png"
        plot_type = "ens_cloud_arrival_time"
        variable = "concentration"
        time_idx = 0

        ["**".en]
        lang = "en"
        """
    click.echo(dedent(s))

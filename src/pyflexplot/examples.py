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
    """TOML setup file to create deterministic NAZ plots."""
    s = """\
        # PyFlexPlot setup file to create deterministic NAZ plots

        [_base]
        # Sampe data directory: /scratch/ruestefa/shared/flexpart_visualization/test/
        infile = "data/cosmo-1_2019052800.nc"
        simulation_type = "deterministic"
        plot_type = "auto"
        species_id = 2
        time = 3

        [_base._concentration]
        outfile = "concentration_{domain}_{lang}_ts-{time:02d}.png"
        variable = "concentration"
        level = 0
        integrate = false

        # TODO Derive from `_concentration` once wildcards/hooks more sophisticated
        [_base._concentration_integrated]
        outfile = "integrated_concentration_{domain}_{lang}_ts-{time:02d}.png"
        variable = "concentration"
        integrate = true
        time = 10
        species_id = 1

        [_base._deposition]
        outfile = "deposition_{domain}_{lang}_ts-{time:02d}.png"
        variable = "deposition"
        deposition_type = "tot"
        integrate = true

        # TODO Derive from `_deposition` once wildcards/hooks more sophisticated
        [_base._affected_area]
        outfile = "affected_area_{domain}_{lang}_ts-{time:02d}.png"
        variable = "deposition"
        deposition_type = "tot"
        integrate = true
        plot_type = "affected_area_mono"
        time = 10
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
        infile = "data/cosmo-2e_2019073100_{ens_member:03d}.nc"
        simulation_type = "ensemble"
        ens_member_id = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ]
        outfile = "ensemble_threshold_agreement_{domain}_{lang}_ts-{time:02d}.png"
        plot_type = "ens_thr_agrmt"
        variable = "deposition"
        deposition_type = "tot"
        time = 10

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
        infile = "data/cosmo-2e_2019073100_{ens_member:03d}.nc"
        simulation_type = "ensemble"
        ens_member_id = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ]
        time = 10

        [_base._concentration]
        variable = "concentration"

        [_base._deposition]
        variable = "deposition"
        deposition_type = "wet"
        integrate = true

        [_base."*"._mean]
        outfile = "ens_mean_{variable}_{lang}_ts-{time:02d}.png"
        plot_type = "ens_mean"

        [_base."*"._median]
        outfile = "ens_median_{variable}_{lang}_ts-{time:02d}.png"
        plot_type = "ens_median"

        [_base."*"._min]
        outfile = "ens_min_{variable}_{lang}_ts-{time:02d}.png"
        plot_type = "ens_min"

        [_base."*"._max]
        outfile = "ens_max_{variable}_{lang}_ts-{time:02d}.png"
        plot_type = "ens_max"

        ["**".de]
        lang = "de"

        ["**".en]
        lang = "en"

        """  # noqa:E501
    click.echo(dedent(s))


@register_choice
def ens_cloud_arrival_time():
    """Plot cloud arrival time."""
    s = """\
        # PyFlexPlot setup file to plot cloud arrival time.

        [_base]
        # Sampe data directory: /scratch/ruestefa/shared/flexpart_visualization/test/
        infile = "data/cosmo-2e_2019073100_{ens_member:03d}.nc"
        simulation_type = "ensemble"
        ens_member_id = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ]
        outfile = "test_{domain}_{lang}_ts-{time:02d}.png"
        plot_type = "ens_cloud_arrival_time"
        variable = "concentration"
        time = 0

        ["**".en]
        lang = "en"

        ["**".de]
        lang = "de"
        """  # noqa:E501
    click.echo(dedent(s))

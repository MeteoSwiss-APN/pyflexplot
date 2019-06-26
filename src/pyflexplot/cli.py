# -*- coding: utf-8 -*-
"""
Command line interface.
"""
import click
import functools
import logging as log
import os
import sys

from pprint import pformat

from .io import FlexFieldSpecs
from .io import FlexFileRotPole
from .utils import count_to_log_level
from .plot import FlexPlotter

from .utils_dev import ipython  #SR_DEV

__version__ = '0.1.0'

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'],)


# yapf: disable
@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--dry-run', '-n',
    help="Perform a trial run with no changes made.",
    flag_value='dry_run', default=False)
@click.option(
    '--verbose', '-v',
    help="Increase verbosity (specify multiple times for more).",
    count=True)
@click.option(
    '--version', '-V',
    help="Print version.",
    is_flag=True)
@click.option(
    '--noplot',
    help="Skip plotting (for debugging etc.).",
    is_flag=True)
@click.option(
    '--open', 'open_cmd',
    help=(
        "Shell command to open the first plot as soon as it is available."
        " The file path follows the command, unless explicitly set by"
        " including the format key '{file_path}'."))
@click.pass_context
# yapf: enable
def main(ctx, **kwargs):
    """Console script for test_cli_project."""

    click.echo("Hi fellow PyFlexPlotter!")
    #click.echo(f"{len(kwargs)} kwargs:\n{pformat(kwargs)}\n")

    log.basicConfig(level=count_to_log_level(kwargs['verbose']))

    if kwargs['version']:
        click.echo(__version__)
        return 0

    if kwargs.pop('dry_run'):
        raise NotImplementedError("dry run")
        return 0

    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Store shared keyword arguments in ctx.obj
    ctx.obj.update(kwargs)

    return 0


def common_options(f):
    """Options common to all commands in the group.

    Defining the common options this way instead of on the group
    enables putting them after the command instead of before it.

    src: https://stackoverflow.com/a/52147284
    """
    # yapf: disable
    options = [
        click.option(
            '--infile', '-i', 'in_file_path',
            help="Input file path.",
            type=click.Path(exists=True, readable=True), required=True),
        click.option(
            '--outfile', '-o', 'out_file_path_fmt',
            help=(
                "Output file path. If multiple plots are to be created, e.g.,"
                " for multiple fields or levels, ``outfile`` must contain"
                " format keys for inserting all changing parameters"
                " (example: ``plot_lvl-{level}.png`` for multiple levels)."),
            type=click.Path(writable=True), required=True),
    ]
    # yapf: enable
    return functools.reduce(lambda x, opt: opt(x), options, f)


# yapf: disable
@main.command()
@common_options
@click.option(
    '--time-ind', 'time_inds',
    help="Index of time (zero-based).",
    type=int, default=[0], multiple=True)
@click.option(
    '--age-class-ind', 'age_inds',
    help="Index of age class (zero-based).",
    type=int, default=[0], multiple=True)
@click.option(
    '--release-point-ind', 'release_point_inds',
    help="Index of release point (zero-based).",
    type=int, default=[0], multiple=True)
@click.option(
    '--level-ind', 'level_inds',
    help="Index of vertical level (zero-based, bottom-up).",
    type=int, default=[0], multiple=True)
@click.option(
    '--species-id', 'species_ids',
    help="Species id (default: all).",
    type=int, default=[0], multiple=True)
@click.option(
    '--source-ind', 'source_inds',
    help="Point source index (zero-based).",
    type=int, default=[0], multiple=True)
@click.pass_context
# yapf: enable
def concentration(
        ctx, in_file_path, out_file_path_fmt, time_inds, age_inds,
        release_point_inds, level_inds, species_ids, source_inds):

    # Determine fields specifications (one for each eventual plot)
    fields_specs = [
        FlexFieldSpecs(
            field_type='3d',
            time_ind=time_ind,
            age_ind=age_ind,
            release_point_ind=release_point_ind,
            level_ind=level_ind,
            species_id=species_id,
            source_ind=source_ind,
        )
        for time_ind in time_inds
        for age_ind in age_inds
        for release_point_ind in release_point_inds
        for level_ind in level_inds
        for species_id in species_ids
        for source_ind in source_inds
    ]

    # Read fields
    flex_data_lst = FlexFileRotPole(in_file_path).read(fields_specs)

    # Create plot
    if not ctx.obj['noplot']:
        # Note: FlexPlotter.run yields the output file paths on-the-go
        f = functools.partial(
            FlexPlotter('concentration').run, flex_data_lst, out_file_path_fmt)
        for i, out_file_path in enumerate(f()):
            if ctx.obj['open_cmd'] and i == 0:
                # Open the first file as soon as it's available
                open_plot(ctx.obj['open_cmd'], out_file_path)


def open_plot(cmd, file_path):
    """Open a plot file using a shell command."""

    # If not yet included, append the output file path
    if not '{file_path}' in cmd:
        cmd += ' {file_path}'

    # Ensure that the command is run in the background
    if not cmd.rstrip().endswith('&'):
        cmd += ' &'

    # Run the command
    cmd = cmd.format(file_path=file_path)
    os.system(cmd)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

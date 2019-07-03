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
from .flexplotter import FlexPlotter

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


def common_options_global(f):
    """Common options of all commands in the group.

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
                "Output file path. If multiple plots are to be created, "
                "e.g., for multiple fields or levels, ``outfile`` must "
                "contain format keys for inserting all changing parameters "
                "(example: ``plot_lvl-{level}.png`` for multiple levels). "
                "The format key for the plotted variable is '{variable}'. "
                "See individual options for the respective format keys."),
            type=click.Path(writable=True), required=True),
    ]
    # yapf: enable
    return functools.reduce(lambda x, opt: opt(x), options, f)


def common_options_dispersion_field(f):
    """Common options of dispersion plots (field selection)."""
    # yapf: disable
    options = [
        click.option(
            '--time-ind', 'time_lst',
            help="Index of time (zero-based). Format key: '{time_ind}'.",
            type=int, default=[0], multiple=True),
        click.option(
            '--age-class-ind', 'nageclass_lst',
            help=("Index of age class (zero-based). Format key: "
                "'{age_class_ind}'."),
            type=int, default=[0], multiple=True),
        click.option(
            '--release-point-ind', 'numpoint_lst',
            help=("Index of release point (zero-based). Format key: "
                "'{rls_pt_ind}'."),
            type=int, default=[0], multiple=True),
        click.option(
            '--level-ind', 'level_lst',
            help=(
                "Index of vertical level (zero-based, bottom-up). "
                "Format key: '{level_ind}'."),
            type=int, default=[0], multiple=True),
        click.option(
            '--species-id', 'species_id_lst',
            help="Species id (default: all). Format key: '{species_id}'.",
            type=int, default=[0], multiple=True),
    ]
    # yapf: enable
    return functools.reduce(lambda x, opt: opt(x), options, f)


def common_options_dispersion_various(f):
    """Common options of dispersion plots (various)."""
    # yapf: disable
    options = [
        click.option(
            '--integrate',
            help=("Integrate field over time. If set, '-int' is appended to "
                "variable name (format key: '{variable}')."),
            is_flag=True, default=False),
    ]
    # yapf: enable
    return functools.reduce(lambda x, opt: opt(x), options, f)


@main.command()
@common_options_global
@common_options_dispersion_field
@common_options_dispersion_various
@click.pass_context
def concentration(
        ctx, in_file_path, out_file_path_fmt, integrate, **kwargs_specs):

    # Determine fields specifications (one for each eventual plot)
    kwargs_specs['prefix'] = None
    kwargs_specs['integrate'] = integrate
    fields_specs = FlexFieldSpecs.many(**kwargs_specs)

    # Read fields
    flex_data_lst = FlexFileRotPole(in_file_path).read(fields_specs)

    # Create plots
    fct = functools.partial(
        FlexPlotter.concentration, flex_data_lst, out_file_path_fmt)
    create_plots(fct, ctx)


@main.command()
@common_options_global
@common_options_dispersion_field
@common_options_dispersion_various
# yapf: disable
@click.option(
    '--deposition-type', 'deposit_type_lst',
    help=("Type of deposition. Part of plot variable (format key: "
        "'{variable}')."),
    type=click.Choice(['both', 'wet', 'dry']),
    default='both', multiple=True)
# yapf: enable
@click.pass_context
def deposition(
        ctx, in_file_path, out_file_path_fmt, integrate, deposit_type_lst,
        **kwargs_specs):

    prefixe_lst = []
    for deposit_type in deposit_type_lst:
        #SR_TMP<
        if deposit_type == 'both':
            raise NotImplementedError("deposit_type='both'")
        #SR_TMP>
        prefix = {
            'wet': 'WD',
            'dry': 'DD',
            'both': ('WD', 'DD')  #SR_TODO figure out how to specify this!
        }[deposit_type]
        prefixe_lst.append(prefix)
    kwargs_specs['prefix_lst'] = prefixe_lst

    # Determine fields specifications (one for each eventual plot)
    kwargs_specs['integrate'] = integrate
    fields_specs = FlexFieldSpecs.many(**kwargs_specs)

    # Read fields
    flex_data_lst = FlexFileRotPole(in_file_path).read(fields_specs)

    # Create plots
    fct = functools.partial(
        FlexPlotter.deposition, flex_data_lst, out_file_path_fmt)
    create_plots(fct, ctx)


def create_plots(fct, ctx):
    """Create FLEXPART plots.

    Args:
        fct (callable): Callable that creates plots while yielding
            the output file paths on-the-fly.

        ctx (Context): Click context object.

    """
    if ctx.obj['noplot']:
        return

    # Note: FlexPlotter.run yields the output file paths on-the-go
    for i, out_file_path in enumerate(fct()):

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

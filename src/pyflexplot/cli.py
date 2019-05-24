# -*- coding: utf-8 -*-
"""
Command line interface of pyflexplot.
"""
import sys
import click
import logging as log
from pprint import pformat

from .utils import count_to_log_level
from .pyflexplot import FlexReader

#SRU_DEV<
try:
    from .utils_dev import ipython
except ImportError:
    pass
#SRU_DEV>

__version__ = '0.1.0'

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"],)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--dry-run',
    '-n',
    flag_value='dry_run',
    default=False,
    help="Perform a trial run with no changes made",
)
@click.option(
    '--verbose',
    '-v',
    count=True,
    help="Increase verbosity (specify multiple times for more)",
)
@click.option(
    '--version',
    '-V',
    is_flag=True,
    help="Print version",
)
@click.option(
    "--infile",
    "-i",
    help="Input file.",
    type=click.Path(exists=True, readable=True),
    required=True,
)
@click.option(
    "--outfile",
    "-o",
    help="Output file.",
    type=click.Path(writable=True),
    required=True,
)
@click.pass_context
def cli(ctx, **kwargs):
    """Console script for test_cli_project."""

    click.echo("Hi fellow PyFlexPlotter!")
    #click.echo("{} kwargs:\n{}\n".format(len(kwargs), pformat(kwargs)))

    log.basicConfig(level=count_to_log_level(kwargs['verbose']))

    #SRU_TMP< TODO Remove at some point!
    log.warning("This is a warning.")
    log.info("This is an info message.")
    log.debug("This is a debug message.")
    #SRU_TMP>

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


@cli.command()
@click.option(
    "--species-id",
    "species_ids",
    help="Species id to plot; plot all if none given.",
    type=int,
    multiple=True,
)
@click.option(
    "--timestep-ind",
    "timestep_inds",
    help="Index of timestep to plot; plot all if none given.",
    type=int,
    multiple=True,
)
@click.option(
    "--field-type",
    "field_types",
    help="Type of field to plot; plot all if none given.",
    type=click.Choice(["3D", "WD", "DD"]),
    multiple=True,
)
@click.option(
    "--level-ind",
    "level_inds",
    help="Vertical level index (bottom-up) to plot; plot all if none given.",
    type=int,
    multiple=True,
)
@click.pass_context
def foo(ctx, **kwargs):

    # Read input
    infile = ctx.obj["infile"]
    reader = FlexReader(**kwargs)
    data = reader.read(infile)

    ipython(globals(), locals(), "foo")

    #click.echo("ctx:\n{}\n".format(pformat(ctx)))
    #click.echo("{} args:\n{}\n".format(len(args), pformat(args)))
    #click.echo("{} kwargs:\n{}\n".format(len(kwargs), pformat(kwargs)))

    return 0


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover

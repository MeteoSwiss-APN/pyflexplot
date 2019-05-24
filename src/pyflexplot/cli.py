# -*- coding: utf-8 -*-
"""
Command line interface of pyflexplot.
"""
import sys
import click
import logging as log
from pprint import pformat

from .utils import count_to_log_level
from .pyflexplot import FlexFileReader

#SRU_DEV<
try:
    from .utils_dev import ipython
except ImportError:
    pass
#SRU_DEV>

__version__ = '0.1.0'

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"],)


# yapf: disable
@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--dry-run', '-n',
    help="Perform a trial run with no changes made",
    flag_value='dry_run', default=False)
@click.option(
    '--verbose', '-v',
    help="Increase verbosity (specify multiple times for more)",
    count=True)
@click.option(
    '--version', '-V',
    help="Print version",
    is_flag=True)
@click.option(
    "--infile", "-i",
    help="Input file.",
    type=click.Path(exists=True, readable=True), required=True)
@click.option(
    "--outfile", "-o",
    help="Output file.",
    type=click.Path(writable=True), required=True)
@click.pass_context
# yapf: enable
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


# yapf: disable
@cli.command()
@click.option(
    "--age-class-ind", "age_class_inds",
    help="Index of age class (zero-based) (default: all).",
    type=int, multiple=True)
@click.option(
    "--field-type", "field_types",
    help="Type of field (default: all).",
    type=click.Choice(["3D", "WD", "DD"]), multiple=True)
@click.option(
    "--level-ind", "level_inds",
    help="Index of vertical level (zero-based, bottom-up) (default: all).",
    type=int, multiple=True)
@click.option(
    "--source-ind", "source_inds",
    help="Point source indices (zero-based) (default: all).",
    type=int, multiple=True)
@click.option(
    "--species-id", "species_ids",
    help="Species id (default: all).",
    type=int, multiple=True)
@click.option(
    "--time-ind", "time_inds",
    help="Index of time (zero-based) (default: all).",
    type=int, multiple=True)
@click.pass_context
# yapf: enable
def foo(ctx, **kwargs):

    # Read input
    infile = ctx.obj["infile"]
    reader = FlexFileReader(**kwargs)
    data = reader.read(infile)

    ipython(globals(), locals(), "foo")

    #click.echo("ctx:\n{}\n".format(pformat(ctx)))
    #click.echo("{} args:\n{}\n".format(len(args), pformat(args)))
    #click.echo("{} kwargs:\n{}\n".format(len(kwargs), pformat(kwargs)))

    return 0


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover

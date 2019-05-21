# -*- coding: utf-8 -*-
"""Console script for pyflexplot."""
import sys
import click
import logging
from pprint import pformat

from pyflexplot.utils import count_to_log_level

__version__ = '0.1.0'

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

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
@click.pass_context
def cli(ctx, *args, **kwargs):
    """Console script for test_cli_project."""

    logging.basicConfig(level=count_to_log_level(kwargs['verbose']))

    logging.warning("This is a warning.")
    logging.info("This is an info message.")
    logging.debug("This is a debug message.")

    if kwargs['version']:
        click.echo(__version__)
        return 0

    if kwargs['dry_run']:
        click.echo("TODO: Implement dry run!")
        return 0

    click.echo(
        "Replace this message by putting your code into test_cli_project.cli.main"
    )
    click.echo("See click documentation at http://click.pocoo.org/")

    return 0


@cli.command()
@click.pass_context
def foo(ctx):

    ipython(globals(), locals())
    click.echo("ctx:\n{}\n".format(pformat(ctx)))
    #click.echo("{} args:\n{}\n".format(len(args), pformat(args)))
    #click.echo("{} kwargs:\n{}\n".format(len(kwargs), pformat(kwargs)))

    return 0

if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover

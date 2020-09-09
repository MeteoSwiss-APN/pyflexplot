"""
CLI utils.
"""
# Standard library
import sys
from typing import Type

# Third-party
import click
from click import Context


def click_exit(ctx: Context, msg: str, stat: int = 0) -> None:
    """Exit with a message."""
    click.echo(msg, file=(sys.stdout if stat == 0 else sys.stderr))
    ctx.exit(stat)


def click_error(
    ctx: Context,
    msg: str,
    exception: Type[Exception] = Exception,
    echo_prefix: str = "Error: ",
) -> None:
    """Print an error message and exit, or raise an exception with traceback."""
    if ctx.obj["raise"]:
        raise exception(msg)
    else:
        click_exit(ctx, f"{echo_prefix}{msg}", stat=1)

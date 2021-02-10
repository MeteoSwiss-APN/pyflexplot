"""CLI utils."""
# Standard library
import sys
import traceback
from typing import Type

# Third-party
import click
from click import Context

# First-party
from srutils.exceptions import InvalidParameterNameError

# Local
from ..setups.plot_setup import PlotSetup
from ..utils.logging import set_log_level


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


def click_exit(ctx: Context, msg: str, stat: int = 0) -> None:
    """Exit with a message."""
    click.echo(msg, file=(sys.stdout if stat == 0 else sys.stderr))
    ctx.exit(stat)


# pylint: disable=W0613  # unused-argument (param)
def click_prepare_setup_params(ctx, param, value):
    # pylint: disable=W0613  # unused-argument
    if not value:
        return None
    try:
        return PlotSetup.prepare_params(value)
    except InvalidParameterNameError as e:
        click_error(ctx, f"Invalid setup parameter name: {e}")


# pylint: disable=W0613  # unused-argument (param)
def click_set_pdb(ctx, param, value):
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["pdb"] = value
    if value:
        ctx.obj["raise"] = True


# pylint: disable=W0613  # unused-argument (param)
def click_set_raise(ctx, param, value) -> None:
    if ctx.obj is None:
        ctx.obj = {}
    if value is None:
        if "raise" not in ctx.obj:
            ctx.obj["raise"] = False
    else:
        ctx.obj["raise"] = value


# pylint: disable=W0613  # unused-argument (ctx, param)
def click_set_verbosity(ctx, param, value) -> None:
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["verbosity"] = value
    set_log_level(value)


def wrap_callback(fct):
    """Wrapp click callback functions to conditionally drop into ipdb."""

    def wrapper(ctx, param, value):
        fct_loc = wrap_pdb(fct) if (ctx.obj or {}).get("pdb") else fct
        return fct_loc(ctx, param, value)

    return wrapper


def wrap_pdb(fct):
    """Decorate a function to drop into ipdb if an exception is raised."""

    def wrapper(*args, **kwargs):
        try:
            return fct(*args, **kwargs)
        except Exception as e:  # pylint: disable=W0703  # broad-except
            if isinstance(e, click.exceptions.Exit):
                if e.exit_code == 0:  # pylint: disable=E1101  # no-member
                    sys.exit(0)
            pdb = __import__("ipdb")  # trick pre-commit hook "debug-statements"
            traceback.print_exc()
            click.echo()
            pdb.post_mortem()
            sys.exit(1)

    return wrapper

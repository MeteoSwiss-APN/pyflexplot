"""Command line interface of PyTrajPlot."""
# Standard library
from typing import Any

# Third-party
import click
from click import Context

# First-party
from pyflexplot.cli.click import click_set_pdb
from pyflexplot.cli.click import click_set_raise
from pyflexplot.cli.click import click_set_verbosity
from pyflexplot.cli.click import wrap_callback
from pyflexplot.cli.click import wrap_pdb
from pyflexplot.cli.preset import add_to_preset_paths
from pyflexplot.cli.preset_click import click_use_preset

# Local
from . import __version__
from . import presets_data_path
from .main import main

add_to_preset_paths(presets_data_path)


# Show default values of options by default
_click_option = click.option
click.option = lambda *args, **kwargs: _click_option(
    *args, **{**kwargs, "show_default": True}
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.option(
    "--pdb/--no-pdb",
    help="Drop into debugger when an exception is raised.",
    callback=click_set_pdb,
    is_eager=True,
    expose_value=False,
)
@click.option(
    "--preset",
    help=(
        "Run with preset setup files matching PATTERN (wildcards: '*', '?')."
        " A single '?' lists all available setups (like --setup-list)."
    ),
    metavar="PATTERN",
    multiple=True,
    callback=wrap_callback(click_use_preset),
    expose_value=False,
)
@click.option(
    "--raise/--no-raise",
    help="Raise exception in place of user-friendly but uninformative error message.",
    callback=wrap_callback(click_set_raise),
    is_eager=True,
    default=None,
    expose_value=False,
)
@click.option(
    "--verbose",
    "-v",
    "verbose",
    help="Increase verbosity; specify multiple times for more.",
    count=True,
    callback=wrap_callback(click_set_verbosity),
    is_eager=True,
    expose_value=False,
)
@click.pass_context
def cli(ctx: Context, **kwargs: Any) -> None:
    wrapped_main = wrap_pdb(main) if ctx.obj["raise"] else main
    wrapped_main(ctx, **kwargs)

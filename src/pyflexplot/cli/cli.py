# pylint: disable=R0914  # too-many-locals
"""Command line interface of PyFlexPlot."""
# Standard library
import warnings
from typing import Any

# Third-party
import click
from click import Context
from shapely.errors import ShapelyDeprecationWarning

# Local
from .. import __version__
from .. import presets_data_path
from .click import click_prepare_setup_params
from .click import click_set_pdb
from .click import click_set_raise
from .click import click_set_verbosity
from .click import wrap_callback
from .click import wrap_pdb
from .main import main
from .preset import add_to_preset_paths
from .preset_click import click_cat_preset_and_exit
from .preset_click import click_use_preset

# # To debug segmentation fault, uncomment and run with PYTHONFAULTHANDLER=1
# import faulthandler
# faulthandler.enable()


add_to_preset_paths(presets_data_path)


# Show default values of options by default
_click_option = click.option
click.option = lambda *args, **kwargs: _click_option(
    *args, **{**kwargs, "show_default": True}
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.argument(
    "setup_file_paths",
    metavar="CONFIG_FILE...",
    type=click.Path(exists=True, readable=True, allow_dash=True),
    nargs=-1,
)
@click.option(
    "--auto-tmp/--no-auto-tmp",
    help=(
        "Use a temporary directory with an automatically generated name. Overridden by"
        " --tmp=TMP_DIR."
    ),
    default=False,
)
@click.option(
    "--cache/--no-cache",
    help="Cache input fields to avoid reading the same data multiple times.",
    is_flag=True,
    # SR_TMP < TODO fix the input file cache (currently broken)
    # +default=True,
    default=False,
    # SR_TMP >
)
@click.option(
    "--dry-run",
    help="Perform a trial run with no changes made.",
    is_flag=True,
    default=False,
)
@click.option(
    "--dest",
    "dest_dir",
    help=(
        "Directory where the plots are saved to. Defaults to the current directory."
        " Note that this option is incompatible with absolute paths specified in the"
        " setup parameter 'outfile'."
    ),
    metavar="DEST_DIR",
    type=click.Path(exists=False),
    default="output/",
)
@click.option(
    "--merge-pdfs/--no-merge-pdfs",
    help="Merge PDF plots with the same output file name.",
)
@click.option(
    "--merge-pdfs-dry",
    help="Merge PDF plots even in a dry run.",
    is_flag=True,
)
@click.option(
    "--num-procs",
    "-P",
    help=(
        "Number of parallel processes. Note that only the creation of plots for"
        " a given input file (ensemble) is parallelized, while the input files"
        " (or ensembles) themselves are processed sequentially."
    ),
    type=int,
    default=1,
)
@click.option(
    "--only",
    help=(
        "Only create the first N plots based on the given setup. Useful during "
        "development; not supposed to be used in production."
    ),
    type=int,
    metavar="N",
)
@click.option(
    "--open",
    "open_cmd",
    help=(
        "Shell command to all plots. The file paths are appended to the command, unless"
        " explicitly embedded with the format key '{file}', which allows one to use"
        " more complex commands than simple application names (example: 'eog {file}"
        " >/dev/null 2>&1' instead of 'eog' to silence the application 'eog')."
    ),
)
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
    "--preset-cat",
    help=(
        "Show the contents of preset setup files matching PATTERN (wildcards:"
        " '*', '?')."
    ),
    metavar="PATTERN",
    callback=wrap_callback(click_cat_preset_and_exit),
    expose_value=False,
)
@click.option(
    "--preset-skip",
    help=(
        "Among preset setup files specified with --preset, skip those matching "
        " PATTERN (wildcards: '*', '?')."
    ),
    metavar="PATTERN",
    multiple=True,
    is_eager=True,
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
    "--keep-merged-pdfs/--remove-merged-pdfs",
    help="Keep individual PDF files after merging them with --merge-pdfs.",
)
@click.option(
    "--setup",
    "input_setup_params",
    help="Setup parameter overriding those in the setup file(s).",
    metavar="PARAM VALUE",
    nargs=2,
    multiple=True,
    callback=wrap_callback(click_prepare_setup_params),
)
@click.option(
    "--show-version/--no-show-version",
    help="Show version number on plot.",
    default=True,
)
@click.option(
    "--suffix",
    "suffixes",
    help=(
        "Override suffix of output files. May be passed multiple times to"
        " create plots in multiple formats."
    ),
    multiple=True,
)
@click.option(
    "--tmp",
    "tmp_dir",
    help=(
        "Temporary directory in which the plots are created before being moved to"
        " DEST_DIR in the end."
    ),
    metavar="TMP_DIR",
    type=click.Path(exists=False),
    default=None,
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
    ignore_shapely_warnings()
    wrapped_main(ctx, **kwargs)


def ignore_shapely_warnings() -> None:
    """Ignore selected shapely deprecation warnings."""
    messages = []
    messages.append(
        "__len__ for multi-part geometries is deprecated and will be removed in Shapely"
        " 2.0. Check the length of the `geoms` property instead to get the  number of"
        " parts of a multi-part geometry."
    )
    messages.append(
        "Iteration over multi-part geometries is deprecated and will be removed in"
        " Shapely 2.0. Use the `geoms` property to access the constituent parts of a"
        " multi-part geometry."
    )
    for message in messages:
        warnings.filterwarnings(
            "ignore",
            category=ShapelyDeprecationWarning,
            message=f"[\n.]*{message}[\n.]*",
        )

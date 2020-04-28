# -*- coding: utf-8 -*-
# pylint: disable=R0914  # too-many-locals
"""
Command line interface.
"""
# Standard library
import functools
import os
import sys

# Third-party
import click

# First-party
from srutils.click import CharSepList

# Local
from . import __version__
from .io import read_files
from .plot import plot_fields
from .preset import click_add_preset_path
from .preset import click_cat_preset_and_exit
from .preset import click_find_presets_and_exit
from .preset import click_list_presets_and_exit
from .preset import click_use_preset
from .setup import InputSetup
from .setup import InputSetupFile

# # To debug segmentation fault, uncomment and run with PYTHONFAULTHANDLER=1
# import faulthandler
# faulthandler.enable()


# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)  # type: ignore


# comma_sep_list_of_unique_ints = CharSepList(int, ",", unique=True)
plus_sep_list_of_unique_ints = CharSepList(int, "+", unique=True)


def not_implemented(msg):
    def f(ctx, param, value):  # pylint: disable=W0613  # unused-argument
        if value:
            click.echo(f"not implemented: {msg}")

    return f


def set_verbosity(ctx, param, value):  # pylint: disable=W0613  # unused-argument
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["verbosity"] = value


def prepare_input_setup_params(ctx, param, value):
    # pylint: disable=W0613  # unused-argument
    if not value:
        return None
    try:
        return InputSetup.cast_many(value, list_separator=",")
    except ValueError as e:
        click.echo(f"Error: Invalid setup parameter: {e}", file=sys.stderr)
        ctx.exit(1)


@click.command(context_settings={"help_option_names": ["-h", "--help"]},)
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.argument(
    "setup_file_paths",
    metavar="CONFIG_FILE...",
    type=click.Path(exists=True, readable=True, allow_dash=True),
    nargs=-1,
)
@click.option(
    "--dry-run",
    "dry_run",
    help="Perform a trial run with no changes made.",
    is_flag=True,
    default=False,
)
@click.option(
    "--verbose",
    "-v",
    "verbose",
    help="Increase verbosity; specify multiple times for more.",
    count=True,
    callback=set_verbosity,
    is_eager=True,
)
@click.option(
    "--setup",
    "input_setup_params",
    help="Setup parameter overriding those in the setup file(s).",
    metavar="PARAM VALUE",
    nargs=2,
    multiple=True,
    callback=prepare_input_setup_params,
)
@click.option(
    "--preset",
    help="Run with preset setup file(s) matching name (may contain wildcards).",
    metavar="NAME",
    multiple=True,
    callback=click_use_preset,
    expose_value=False,
)
@click.option(
    "--cat-preset",
    help="Show the content of a preset setup file.",
    metavar="NAME",
    callback=click_cat_preset_and_exit,
    expose_value=False,
)
@click.option(
    "--list-presets",
    help="List the names of all preset setup files.",
    callback=click_list_presets_and_exit,
    is_flag=True,
)
@click.option(
    "--find-presets",
    help="List preset setup file(s) by name (may contain wildcards).",
    metavar="NAME",
    callback=click_find_presets_and_exit,
    multiple=True,
)
@click.option(
    "--add-presets",
    help="Add a directory containing preset setup files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    callback=click_add_preset_path,
    is_eager=True,
    expose_value=False,
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
    "--each-only",
    help=(
        "Only create the first N plots (at most) for each input file. Useful "
        "during development; not supposed to be used in production."
    ),
    type=int,
    metavar="N",
)
@click.option(
    "--open-first",
    "open_first_cmd",
    help=(
        "Shell command to open the first plot as soon as it is available. The file "
        "path is appended to the command, unless explicitly embedded with the format "
        "key '{file}', which allows one to use more complex commands than simple "
        "application names (example: 'eog {file} >/dev/null 2>&1' instead of 'eog' to "
        "silence the application 'eog')."
    ),
)
@click.option(
    "--open-all", "open_all_cmd", help="Like --open-first, but for all plots.",
)
# ---
@click.pass_context
# pylint: disable=R0913  # too-many-arguments
# pylint: disable=R0912  # too-many-branches
def cli(
    ctx, setup_file_paths, input_setup_params, dry_run, only, each_only, **cli_args
):
    """Create dispersion plot as specified in CONFIG_FILE(S)."""

    ctx.obj.update(cli_args)

    # Add preset setup file paths
    setup_file_paths = list(setup_file_paths)
    for path in ctx.obj.get("preset_setup_file_paths", []):
        if path not in setup_file_paths:
            setup_file_paths.append(path)
    if not setup_file_paths:
        click.echo(
            "Error: Must pass explicit and/or preset setup file(s)", file=sys.stderr,
        )
        ctx.exit(1)

    # Read setup files
    setups = InputSetupFile.read_many(setup_file_paths, override=input_setup_params)

    # Group setups by input file(s)
    setups_by_infile = setups.group("infile")

    open_first_cmd = ctx.obj["open_first_cmd"]
    open_all_cmd = ctx.obj["open_all_cmd"]

    class BreakInner(Exception):
        """Break out of inner loop, but continue outer loop."""

    class BreakOuter(Exception):
        """Break out of inner and outer loop."""

    # Create plots input file(s) by input file(s)
    out_file_paths = []
    n_in = len(setups_by_infile)
    i_tot = -1
    for i_in, (in_file_path, sub_setups) in enumerate(setups_by_infile.items()):

        # Read input fields
        var_setups_lst = sub_setups.decompress_grouped_by_time()

        fields, mdata_lst = read_files(in_file_path, var_setups_lst, dry_run)

        try:
            # Note: plot_fields(...) yields the output file paths on-the-go
            # pylint: disable=W0612  # unused-variable (plot_handle)
            n_fld = len(fields)
            for i_fld, (out_file_path, plot_handle) in enumerate(
                plot_fields(fields, mdata_lst, dry_run)
            ):
                i_tot += 1
                click.echo(f"[{i_in + 1}/{n_in}][{i_fld + 1}/{n_fld}] {out_file_path}")

                if out_file_path in out_file_paths:
                    raise Exception("duplicate output file", out_file_path)
                out_file_paths.append(out_file_path)

                if open_first_cmd and i_in + i_fld == 0:
                    open_plots(open_first_cmd, [out_file_path], dry_run)

                remaining_plots = n_fld - i_fld - 1
                if remaining_plots and each_only and (i_fld + 1) >= each_only:
                    click.echo(f"skip remaining {remaining_plots} plots")
                    raise BreakInner()
                if only and (i_tot + 1) >= only:
                    if remaining_plots:
                        click.echo(f"skip remaining {remaining_plots} plots")
                    raise BreakOuter()
        except BreakInner:
            continue
        except BreakOuter:
            remaining_files = n_in - i_in - 1
            if remaining_files:
                click.echo(f"skip remaining {remaining_files} input files")
            break

    if open_all_cmd:
        open_plots(open_all_cmd, out_file_paths, dry_run)

    return 0


def open_plots(cmd, file_paths, dry_run):
    """Open a plot file using a shell command."""

    file_paths = sorted(file_paths)

    # If not yet included, append the output file path
    if "{file}" not in cmd:
        cmd += " {file}"

    # Ensure that the command is run in the background
    if not cmd.rstrip().endswith("&"):
        cmd += " &"

    # Run the command
    cmd = cmd.format(file=" ".join(file_paths))
    click.echo(cmd)
    if not dry_run:
        os.system(cmd)


if __name__ == "__main__":
    sys.exit(1)  # pragma: no cover

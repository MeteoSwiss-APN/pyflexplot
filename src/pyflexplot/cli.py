# -*- coding: utf-8 -*-
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
from .plot import plot
from .preset import click_add_preset_path
from .preset import click_cat_preset
from .preset import click_find_presets
from .preset import click_list_presets
from .preset import click_use_preset
from .setup import InputSetupFile
from .specs import FldSpecs

# # To debug segmentation fault, uncomment and run with PYTHONFAULTHANDLER=1
# import faulthandler
# faulthandler.enable()


# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)  # type: ignore


# comma_sep_list_of_unique_ints = CharSepList(int, ",", unique=True)
plus_sep_list_of_unique_ints = CharSepList(int, "+", unique=True)


def not_implemented(msg):
    def f(ctx, param, value):
        if value:
            click.echo(f"not implemented: {msg}")
            ctx.exit(1)

    return f


def set_verbosity(ctx, param, value):
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["verbosity"] = value


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
    # SR_TMP <
    is_eager=True,
    callback=not_implemented("--dry-run"),
    # SR_TMP >
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
    callback=click_cat_preset,
    expose_value=False,
)
@click.option(
    "--list-presets",
    help="List the names of all preset setup files.",
    callback=click_list_presets,
    is_flag=True,
)
@click.option(
    "--find-presets",
    help="List preset setup file(s) by name (may contain wildcards).",
    metavar="NAME",
    callback=click_find_presets,
)
@click.option(
    "--add-presets",
    help="Add a directory containing preset setup files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    callback=click_add_preset_path,
    is_eager=True,
    expose_value=False,
)
# ---
@click.pass_context
def cli(ctx, setup_file_paths, **cli_args):
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
    setups = InputSetupFile.read_multiple(setup_file_paths)

    # Group setups by input file(s)
    setups_by_infile = setups.group("infile")

    # Create plots input file(s) by input file(s)
    out_file_paths = []
    for in_file_path, sub_setups in setups_by_infile.items():

        # Read input fields
        fld_specs_lst = FldSpecs.create(sub_setups)
        fields, mdata_lst = read_files(in_file_path, fld_specs_lst)
        assert len(fields) == len(mdata_lst)

        for (field, mdata) in zip(fields, mdata_lst):
            # Note: plot(...) yields the output file paths on-the-go
            for out_file_path in plot([field], [mdata], field.fld_specs.fld_setup):
                if ctx.obj["open_first_cmd"] and not out_file_paths:
                    open_plots(ctx.obj["open_first_cmd"], [out_file_path])
                out_file_paths.append(out_file_path)

    if ctx.obj["open_all_cmd"]:
        open_plots(ctx.obj["open_all_cmd"], out_file_paths)

    return 0


def open_plots(cmd, file_paths):
    """Open a plot file using a shell command."""

    # If not yet included, append the output file path
    if "{file}" not in cmd:
        cmd += " {file}"

    # Ensure that the command is run in the background
    if not cmd.rstrip().endswith("&"):
        cmd += " &"

    # Run the command
    cmd = cmd.format(file=" ".join(file_paths))
    click.echo(cmd)
    os.system(cmd)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover

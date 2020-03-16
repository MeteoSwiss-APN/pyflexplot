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
from .examples import choices as example_choices
from .examples import list_examples
from .examples import print_example
from .io import read_files
from .plot import plot
from .setup import SetupFile
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]},)
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.argument(
    "setup_file_paths",
    metavar="CONFIG_FILE...",
    type=click.Path(exists=True, readable=True, allow_dash=True),
    required=True,
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
    "--example",
    help="Print example setup file.",
    type=click.Choice(list(example_choices)),
    callback=print_example,
    expose_value=False,
)
@click.option(
    "--list-examples",
    help="List the names of all example setup files.",
    callback=list_examples,
    is_flag=True,
)
# ---
@click.pass_context
def cli(ctx, setup_file_paths, **cli_args):
    """Create dispersion plot as specified in CONFIG_FILE(S)."""

    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Read setup file
    setups = [
        setup
        for setup_file_path in setup_file_paths
        for setup in SetupFile(setup_file_path).read()
    ]

    # Create plots
    create_plots(setups, cli_args)

    return 0


def create_plots(setups, cli_args):
    """Read and plot FLEXPART data."""

    out_file_paths = []

    # SR_TMP <<< TODO find better solution
    for idx_setup, setup in enumerate(setups):

        # Read input fields
        fields, attrs_lst = read_fields(setup)

        # Note: Plotter.run yields the output file paths on-the-go
        for idx_plot, out_file_path in enumerate(plot(fields, attrs_lst, setup)):
            out_file_paths.append(out_file_path)

            if cli_args["open_first_cmd"] and idx_setup + idx_plot == 0:
                # Open the first file as soon as it's available
                open_plots(cli_args["open_first_cmd"], [out_file_path])

    if cli_args["open_all_cmd"]:
        # Open all plots
        open_plots(cli_args["open_all_cmd"], out_file_paths)


def read_fields(setup):

    # Determine fields specifications (one for each eventual plot)
    fld_specs_lst = FldSpecs.create(setup)

    # Read fields
    fields = []
    attrs_lst = []
    for raw_path in setup.infile:
        fields_i, attrs_lst_i = read_files(raw_path, setup, fld_specs_lst)
        fields.extend(fields_i)
        attrs_lst.extend(attrs_lst_i)

    return fields, attrs_lst


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
    os.system(cmd)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover

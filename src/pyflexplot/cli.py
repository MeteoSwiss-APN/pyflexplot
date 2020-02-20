# -*- coding: utf-8 -*-
"""
Command line interface.
"""
# Standard library
import functools
import logging as log
import os
import sys

# Third-party
import click

# First-party
from srutils.click import CharSepList

# Local
from . import __version__
from .examples import choices as example_choices
from .examples import print_example
from .field_specs import FieldSpecs
from .io import FileReader
from .plotter import Plotter
from .setup import SetupFile
from .utils import count_to_log_level
from .var_specs import MultiVarSpecs

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
    help="Example commands.",
    type=click.Choice(list(example_choices)),
    callback=print_example,
    expose_value=False,
)
# ---
@click.pass_context
def cli(ctx, setup_file_paths, **cli_args):
    """
    Create dispersion plot as specified in CONFIG_FILE(S).
    """

    log.basicConfig(level=count_to_log_level(cli_args["verbose"]))

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
    """
    Read and plot FLEXPART data.
    """

    out_file_paths = []

    # SR_TMP <<< TODO find better solution
    for idx_setup, setup in enumerate(setups):

        # Read input fields
        field_lst = read_fields(setup)

        def fct_plot():
            return Plotter().run(field_lst, setup)

        # Note: Plotter.run yields the output file paths on-the-go
        for idx_plot, out_file_path in enumerate(fct_plot()):
            out_file_paths.append(out_file_path)

            if cli_args["open_first_cmd"] and idx_setup + idx_plot == 0:
                # Open the first file as soon as it's available
                open_plots(cli_args["open_first_cmd"], [out_file_path])

    if cli_args["open_all_cmd"]:
        # Open all plots
        open_plots(cli_args["open_all_cmd"], out_file_paths)


def prep_var_specs_dct(setup):
    """
    Prepare the variable specifications dict from the raw CLI input.
    """

    var_specs_raw = {
        "time_lst": (setup.time_idx,),
        "nageclass_lst": (setup.age_class_idx,),
        "noutrel_lst": (setup.nout_rel_idx,),
        "numpoint_lst": (setup.release_point_idx,),
        "species_id_lst": (setup.species_id,),
        "integrate_lst": (setup.integrate,),
    }

    if setup.variable == "concentration":
        var_specs_raw["level_lst"] = (setup.level_idx,)
    elif setup.variable == "deposition":
        var_specs_raw["deposition_lst"] = (setup.deposition_type,)
    else:
        raise NotImplementedError(f"variable='{setup.variable}'")

    var_specs_dct = {}
    for key, val in var_specs_raw.items():
        if key.endswith("_lst"):
            key = key[: -len("_lst")]
        if isinstance(val, (tuple, list)):
            val = [tuple(e) if isinstance(e, list) else e for e in val]
        else:
            raise Exception(f"invalid value type {type(val).__name__}", key)
        if len(val) == 1:
            val = next(iter(val))
        var_specs_dct[key] = val

    return var_specs_dct


def read_fields(setup):

    # SR_TMP < TODO find cleaner solution
    if setup.simulation_type == "ensemble":
        if setup.plot_type == "ens_thr_agrmt":
            ens_var_setup = {"thr": 1e-9}
        elif setup.plot_type == "ens_cloud_arrival_time":
            ens_var_setup = {"thr": 1e-9, "n_mem_min": 11}
        else:
            ens_var_setup = {}
    else:
        ens_var_setup = None
    # SR_TMP >

    # SR_TMP <<< TODO clean this up
    attrs = {"lang": setup.lang}
    if setup.simulation_type == "ensemble":
        if setup.member_ids is not None:
            attrs["member_ids"] = setup.member_ids
        assert setup.plot_type.startswith("ens_"), setup.plot_type
        attrs["ens_var"] = setup.plot_type[4:]
        attrs["ens_var_setup"] = ens_var_setup

    # Create variable specification objects
    var_specs_dct = prep_var_specs_dct(setup)
    multi_var_specs_lst = MultiVarSpecs.create(
        setup, var_specs_dct, lang=setup.lang, words=None,
    )

    # Determine fields specifications (one for each eventual plot)
    fld_specs_lst = [
        FieldSpecs(setup.tmp_cls_name(), multi_var_specs, attrs)
        for multi_var_specs in multi_var_specs_lst
    ]

    # Read fields
    field_lst = []
    for raw_path in setup.infiles:
        field_lst.extend(FileReader(raw_path).run(fld_specs_lst, lang=setup.lang))

    return field_lst


def open_plots(cmd, file_paths):
    """
    Open a plot file using a shell command.
    """

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

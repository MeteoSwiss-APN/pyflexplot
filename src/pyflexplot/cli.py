# -*- coding: utf-8 -*-
"""
Command line interface.
"""
import click
import functools
import logging as log
import os
import re
import sys

from srutils.click import CharSepList
from srutils.various import group_kwargs
from srutils.various import isiterable

from .config import Config
from .examples import show_example
from .field_specs import FieldSpecs
from . import __version__
from .io import FileReader
from .plotter import Plotter
from .utils import count_to_log_level
from .var_specs import MultiVarSpecs

# # To debug segmentation fault, uncomment and run with PYTHONFAULTHANDLER=1
# import faulthandler
# faulthandler.enable()


# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)


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
    type=click.Choice(["naz-det-sh"]),
    callback=show_example,
    expose_value=False,
)
# --- Input
@click.option(
    "--config",
    "config_file",
    help="Configuration file (TOML).",
    type=click.File("r"),
    required=True,
)
@click.option(
    "--time",
    "config_cli__time_idx",
    help="Index of time (zero-based). Format key: '{time_idx}'.",
    type=int,
)
@click.option(
    "--member-id",
    "-m",
    "config_cli__ens_member_id_lst",
    help=(
        "Ensemble member id. Repeat for multiple members. Omit for deterministic "
        "simulation data. Use the format key '{member_id}' to embed the member id(s) "
        "in the plot file path."
    ),
    type=int,
    multiple=True,
)
@click.option(
    "--lang",
    "config_cli__lang",
    help="Language. Use the format key '{lang}' to embed it into the plot file path.",
    type=click.Choice(["en", "de"]),
)
@click.option(
    "--scale",
    "scale_fact",
    help="Scale field before plotting. Useful for debugging.",
    type=float,
    default=None,
)
@click.option(
    "--reverse-legend/--no-reverse-legend",
    "reverse_legend",
    help=(
        "Reverse the order of the level ranges and corresponding colors in the plot "
        "legend such that the levels increase from top to bottom instead of decreasing."
    ),
    is_flag=True,
    default=False,
)
# ---
@group_kwargs("config_cli")
@click.pass_context
def cli(ctx, config_file, config_cli, **cli_args):
    """
    Read NetCDF output of a deterministic or ensemble ``FLEXPART`` dispersion
    simulation from INFILE(S) to create the plot OUTFILE.

    Both INFILE(S) and OUTFILE may contain format keys that are replaced by
    values derived from options and/or the input data, which allows one to
    specify multipe input and/or output files with a single string. The syntax
    is that of Python format strings, with the variable name wrapped in curly
    braces.

    Example:

    \b
    $ pyflexplot "ens_mem{member_id:03}.nc" "ens_spc-{species_id:02}.png" \\
    >   --member-id={1..10} --species-id={0,1} ...
    # input  : ens_mem001.nc, ens_mem002.nc, ...,  ens_mem010.nc
    # output : ens_spc00.png, ens_spc01.png

    """

    click.echo("Welcome fellow PyFlexPlotter!")

    log.basicConfig(level=count_to_log_level(cli_args["verbose"]))

    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Read config file
    config = Config.from_file(config_file)
    config.update(config_cli, skip_none=True)

    # SR_TMP <
    required_args = ["variable", "simulation_type"]
    for arg in required_args:
        if getattr(config, arg) is None:
            raise Exception(f"argument missing: {arg}")
    # SR_TMP >

    # Create plots
    create_plots(config, cli_args)

    return 0


def create_plots(config, cli_args):
    """
    Read and plot FLEXPART data.
    """

    # SR_TMP <
    if config.plot_type == "auto":
        config.plot_type = config.variable
    # SR_TMP >

    if config.simulation_type == "deterministic":
        cls_name = f"{config.variable}"
    elif config.simulation_type == "ensemble":
        cls_name = f"{config.plot_type}_{config.variable}"

    # Read input fields
    field_lst = read_fields(cls_name, config)

    # Prepare plotter
    plotter = Plotter()

    def fct_plot():
        return plotter.run(
            cls_name,
            field_lst,
            config.outfile,
            lang=config.lang,
            scale_fact=cli_args["scale_fact"],
        )

    # Note: Plotter.run yields the output file paths on-the-go
    out_file_paths = []
    for i, out_file_path in enumerate(fct_plot()):
        out_file_paths.append(out_file_path)

        if cli_args["open_first_cmd"] and i == 0:
            # Open the first file as soon as it's available
            open_plots(cli_args["open_first_cmd"], [out_file_path])

    if cli_args["open_all_cmd"]:
        # Open all plots
        open_plots(cli_args["open_all_cmd"], out_file_paths)


def prep_var_specs_dct(config):
    """
    Prepare the variable specifications dict from the raw CLI input.
    """

    var_specs_raw = {
        "time_lst": (config.time_idx,),
        "nageclass_lst": (config.age_class_idx,),
        "noutrel_lst": (config.nout_rel_idx,),
        "numpoint_lst": (config.release_point_idx,),
        "species_id_lst": (config.species_id,),
        "integrate_lst": (config.integrate,),
    }

    if config.variable == "concentration":
        var_specs_raw["level_lst"] = (config.level_idx,)
    elif config.variable == "deposition":
        var_specs_raw["deposition_lst"] = (config.deposition_type,)
    else:
        raise NotImplementedError(f"variable='{config.variable}'")

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


def read_fields(cls_name, config):
    """
    TODO
    """

    # SR_TMP < TODO find cleaner solution
    if config.simulation_type == "ensemble":
        if config.plot_type == "ens_thr_agrmt":
            ens_var_setup = {"thr": 1e-9}
        else:
            ens_var_setup = {}
    else:
        ens_var_setup = None
    # SR_TMP >

    # SR_TMP <<< TODO clean this up
    attrs = {"lang": config.lang}
    if config.simulation_type == "ensemble":
        if config.member_ids is not None:
            attrs["member_ids"] = config.member_ids
        assert config.plot_type.startswith("ens_"), config.plot_type
        attrs["ens_var"] = config.plot_type[4:]
        attrs["ens_var_setup"] = ens_var_setup

    # Create variable specification objects
    var_specs_dct = prep_var_specs_dct(config)
    multi_var_specs_lst = MultiVarSpecs.create(
        cls_name, var_specs_dct, lang=config.lang, words=None,
    )

    # Determine fields specifications (one for each eventual plot)
    fld_specs_lst = [
        FieldSpecs(cls_name, multi_var_specs, attrs)
        for multi_var_specs in multi_var_specs_lst
    ]

    # Read fields
    field_lst = []
    for raw_path in config.infiles:
        field_lst.extend(FileReader(raw_path).run(fld_specs_lst, lang=config.lang))

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

# -*- coding: utf-8 -*-
"""
Command line interface.
"""
import click
import functools
import logging as log
import os
import sys

from click_config_file import configuration_option

from srutils.click import click_options
from srutils.click import CharSepList
from srutils.click import DerivChoice
from srutils.various import group_kwargs

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


class ClickOptionsGroup:
    pass


# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)


# comma_sep_list_of_unique_ints = CharSepList(int, ",", unique=True)
plus_sep_list_of_unique_ints = CharSepList(int, "+", unique=True)


#
# TODO
#
# - Collect all passed options in a dict and pass it on as **options
#
# - In the CLI functions of the plots, catch all necessary options explicitly
#   -> Catch all unnecessary options as **kwargs and show warnings for those
#
# - For ALL options, default to "everything available", e.g., all species etc.
#
# - Turn the input and output files from options into arguments (no flags)
#
# - Check for duplicate outfiles and number those, e.g., plot-1.png, plot-2.png
#
# - Add flag --unique-plots or so to abort if there are duplicate outfiles
#   -> To prevent accidental *-i.png (potential pitfall)
#


def create_plots(
    *,
    ctx,
    in_file_path_raw_lst,
    out_file_path_raw,
    var_specs_raw,
    field,
    plot_type,
    simulation_type,
    ens_member_id_lst=None,
    lang,
    **kwargs,
):
    """
    Read and plot FLEXPART data.

    Args:
        TODO

    """

    # SR_TMP <
    if plot_type == "auto":
        plot_type = field
    # SR_TMP >

    # SR_TMP <
    if field == "concentration":
        var_specs_raw.pop("deposition_lst", None)
    elif field == "deposition":
        var_specs_raw.pop("level_lst", None)
    else:
        raise NotImplementedError(f"field='{field}'")
    # SR_TMP >

    # SR_TMP < TODO find cleaner solution
    if simulation_type == "deterministic":
        cls_name = f"{field}"
        ens_var_setup = None
    elif simulation_type == "ensemble":
        cls_name = f"{plot_type}_{field}"
        ens_var_setup = {"thr": 1e-9} if plot_type == "ens_thr_agrmt" else {}
    else:
        raise NotImplementedError(f"simulation_type='{simulation_type}'")
    # SR_TMP >

    # Create variable specification objects
    var_specs_dct = prep_var_specs_dct(var_specs_raw)
    multi_var_specs_lst = MultiVarSpecs.create(
        cls_name, var_specs_dct, lang=lang, words=None,
    )

    field_lst = read_fields(
        simulation_type=simulation_type,
        plot_type=plot_type,
        cls_name=cls_name,
        multi_var_specs_lst=multi_var_specs_lst,
        ens_member_id_lst=ens_member_id_lst,
        ens_var_setup=ens_var_setup,
        in_file_path_raw_lst=in_file_path_raw_lst,
        lang=lang,
        **kwargs,
    )

    if ctx.obj["no_plot"]:
        return

    # Prepare plotter
    plotter = Plotter()

    def fct_plot():
        return plotter.run(cls_name, field_lst, out_file_path_raw, **kwargs)

    # Note: Plotter.run yields the output file paths on-the-go
    out_file_paths = []
    for i, out_file_path in enumerate(fct_plot()):
        out_file_paths.append(out_file_path)

        if ctx.obj["open_first_cmd"] and i == 0:
            # Open the first file as soon as it's available
            open_plots(ctx.obj["open_first_cmd"], [out_file_path])

    if ctx.obj["open_all_cmd"]:
        # Open all plots
        open_plots(ctx.obj["open_all_cmd"], out_file_paths)


def prep_var_specs_dct(var_specs_raw):
    """
    Prepare the variable specifications dict from the raw CLI input.

    Example:
        >>> prep_var_specs_dct({"a_lst": ([0], [1, 2])})
        {"a": [0, (1, 2)]}

    """
    var_specs_dct = {}
    for key, val in var_specs_raw.items():
        if key.endswith("_lst"):
            key = key[: -len("_lst")]
        if isinstance(val, tuple):
            val = [tuple(e) if isinstance(e, list) else e for e in val]
        else:
            raise Exception(f"invalid value type {type(val).__name__}")
        if len(val) == 1:
            val = next(iter(val))
        var_specs_dct[key] = val
    return var_specs_dct


def read_fields(
    *,
    simulation_type,
    plot_type,
    in_file_path_raw_lst,
    cls_name,
    multi_var_specs_lst,
    lang,
    ens_member_id_lst,
    ens_var_setup,
    **kwargs,
):
    """
    TODO
    """

    attrs = {"lang": lang}

    # SR_TMP <<< TODO clean this up
    if simulation_type == "ensemble":
        if ens_member_id_lst is not None:
            attrs["member_ids"] = ens_member_id_lst
        assert plot_type.startswith("ens")
        attrs["ens_var"] = plot_type
        attrs["ens_var_setup"] = ens_var_setup

    # Determine fields specifications (one for each eventual plot)
    fld_specs_lst = [
        FieldSpecs(cls_name, multi_var_specs, attrs)
        for multi_var_specs in multi_var_specs_lst
    ]

    # Read fields
    field_lst = []
    for in_file_path_raw in in_file_path_raw_lst:
        field_lst.extend(FileReader(in_file_path_raw).run(fld_specs_lst, lang=lang))

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


def not_implemented(msg):
    def f(ctx, param, value):
        if value:
            click.echo(f"not implemented: {msg}")
            ctx.exit(1)

    return f


class GlobalOptions(ClickOptionsGroup):
    """
    Options shared by all types of plots.
    """

    @click_options
    def execution():
        """Options to be passed before any command."""
        return [
            click.option(
                "--dry-run",
                "exe__dry_run",
                help="Perform a trial run with no changes made.",
                is_flag=True,
                default=False,
                # SR_TMP <
                is_eager=True,
                callback=not_implemented("--dry-run"),
                # SR_TMP >
            ),
            click.option(
                "--verbose",
                "-v",
                "exe__verbose",
                help="Increase verbosity; specify multiple times for more.",
                count=True,
            ),
            click.option(
                "--no-plot",
                "exe__no_plot",
                help="Skip plotting (for debugging etc.).",
                is_flag=True,
            ),
            click.option(
                "--open-first",
                "exe__open_first_cmd",
                help=(
                    "Shell command to open the first plot as soon as it is available. "
                    "The file path is appended to the command, unless explicitly "
                    "embedded with the format key '{file}', which allows one to use "
                    "more complex commands than simple application names (example: "
                    "'eog {file} >/dev/null 2>&1' instead of 'eog' to silence the "
                    "application 'eog')."
                ),
            ),
            click.option(
                "--open-all",
                "exe__open_all_cmd",
                help="Like --open-first, but for all plots.",
            ),
            click.option(
                "--example",
                help="Example commands.",
                type=click.Choice(["naz-det-sh"]),
                callback=show_example,
                expose_value=False,
            ),
        ]

    @click_options
    def input():
        return [
            click.option(
                "--time-ind",
                "var_specs_raw__time_lst",
                help="Index of time (zero-based). Format key: '{time_ind}'.",
                type=int,
                default=[0],
                multiple=True,
            ),
            click.option(
                "--age-class-ind",
                "var_specs_raw__nageclass_lst",
                help="Index of age class (zero-based). Format key: '{age_class_ind}'.",
                type=int,
                default=[0],
                multiple=True,
            ),
            click.option(
                "--nout-rel-ind",
                "var_specs_raw__noutrel_lst",
                help="Index of noutrel (zero-based). Format key: '{noutrel_ind}'.",
                type=int,
                default=[0],
                multiple=True,
            ),
            click.option(
                "--release-point-ind",
                "var_specs_raw__numpoint_lst",
                help="Index of release point (zero-based). Format key: '{rls_pt_ind}'.",
                type=int,
                default=[0],
                multiple=True,
            ),
            click.option(
                "--species-id",
                "var_specs_raw__species_id_lst",
                help=(
                    "Species id(s) (default: 0). To sum up multiple species, combine "
                    "their ids with '+'. Format key: '{species_id}'."
                ),
                type=plus_sep_list_of_unique_ints,
                default=["1"],
                multiple=True,
            ),
            click.option(
                "--level-ind",
                "var_specs_raw__level_lst",
                help=(
                    "Index/indices of vertical level (zero-based, bottom-up). To sum "
                    "up multiple levels, combine their indices with '+'. Format key: "
                    "'{level_ind}'."
                ),
                type=plus_sep_list_of_unique_ints,
                default=["0"],
                multiple=True,
            ),
            click.option(
                "--member-id",
                "-m",
                "ens_member_id_lst",
                help=(
                    "Ensemble member id. Repeat for multiple members. Omit for "
                    "deterministic simulation data. Use the format key '{member_id}' "
                    "to embed the member id(s) in the plot file path."
                ),
                type=int,
                multiple=True,
            ),
            click.option(
                "--deposition-type",
                "var_specs_raw__deposition_lst",
                help=(
                    "Type of deposition. Part of the plot variable name that may be "
                    "embedded in the plot file path with the format key '{variable}'."
                ),
                type=DerivChoice(["wet", "dry"], {"tot": ("wet", "dry")}),
                default=["tot"],
                multiple=True,
            ),
        ]

    @click_options
    def preproc():
        return [
            click.option(
                "--integrate/--no-integrate",
                "var_specs_raw__integrate_lst",
                help=(
                    "Integrate field over time. Use the format key '{integrate}' to "
                    "embed '[no-]int' in the plot file path."
                ),
                is_flag=True,
                default=[False],
                multiple=True,
            ),
            click.option(
                "--scale",
                "scale_fact",
                help="Scale field before plotting. Useful for debugging.",
                type=float,
                default=None,
            ),
        ]

    @click_options
    def plot():
        return [
            # SR_TMP <
            click.option(
                "--field",
                help="Input field to be plotted.",
                type=click.Choice(["concentration", "deposition"]),
                required=True,
            ),
            click.option(
                "--simulation-type",
                help="Type of simulation.",
                type=click.Choice(["deterministic", "ensemble"]),
                required=True,
            ),
            click.option(
                "--plot-type",
                help="Type of plot.",
                type=click.Choice(
                    [
                        "auto",
                        "affected_area",
                        "affected_area_mono",
                        "ens_mean",
                        "ens_max",
                        "ens_thr_agrmt",
                    ]
                ),
                default="auto",
                required=True,
            ),
            # SR_TMP >
            click.option(
                "--lang",
                "lang",
                help="Language. Format key: '{lang}'.",
                type=click.Choice(["en", "de"]),
                default="en",
            ),
            click.option(
                "--domain",
                help=(
                    "Plot domain. Defaults to 'data', which derives the domain size "
                    "from the input data. Use the format key '{domain}' to embed the "
                    "domain name in the plot file path."
                ),
                type=click.Choice(["auto", "ch"]),
                default="auto",
            ),
            click.option(
                "--reverse-legend/--no-reverse-legend",
                help=(
                    "Reverse the order of the level ranges and corresponding colors in "
                    "the plot legend such that the levels increase from top to bottom "
                    "instead of decreasing."
                ),
                is_flag=True,
                default=False,
            ),
        ]


@click.command(context_settings={"help_option_names": ["-h", "--help"]},)
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.argument(
    "in_file_path_raw_lst", metavar="INFILE(S)", type=str, nargs=-1, required=True,
)
@click.argument(
    "out_file_path_raw", metavar="OUTFILE", type=str, required=True,
)
@GlobalOptions.execution
@GlobalOptions.input
@GlobalOptions.preproc
@GlobalOptions.plot
@click.pass_context
@group_kwargs("exe")
@group_kwargs("var_specs_raw")
def cli(ctx, exe, **kwargs):
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

    log.basicConfig(level=count_to_log_level(exe["verbose"]))

    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Store shared keyword arguments in ctx.obj
    ctx.obj.update(exe)

    # Create plots
    create_plots(ctx=ctx, **kwargs)

    return 0


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover

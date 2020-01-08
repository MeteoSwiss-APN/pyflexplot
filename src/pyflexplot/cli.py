# -*- coding: utf-8 -*-
"""
Command line interface.
"""
import click
import functools
import logging as log
import os
import sys

from pprint import pformat

from srutils.various import group_kwargs

from . import __version__
from .field_specs import FieldSpecs
from .io import FileReader
from .plotter import Plotter
from .utils import count_to_log_level
from .var_specs import MultiVarSpecs

# To debug segmentation fault, uncomment and run with PYTHONFAULTHANDLER=1
# + import faulthandler
# + faulthandler.enable()


def click_options(f_options):
    """Define a list of click options shared by multiple commands.

    Args:
        f_options (function): Function returning a list of ``click.option``
            objects.

    Example:
        > @click_options          # <= define options
        > def common_options():
        >     return [click.option(...), click.option(...), ...]

        > @click.group
        > def main(...):
        >     ...

        > @CLI.command
        > @common_options         # <= use options
        > def foo(...):
        >     ...

        > @CLI.command
        > @common_options         # <= use options
        > def bar(...):
        >     ...


    Applications:
        Define options that are shared by multiple commands but are passed
        after the respective command, instead of before as group options (the
        native way to define shared options) are.

        Define options used only by a single group or command in a function
        instead of as decorators, which allows them to be folded by the editor
        more easily.

    Source:
        https://stackoverflow.com/a/52147284

    """
    return lambda f: functools.reduce(lambda x, opt: opt(x), f_options(), f)


class ClickOptionsGroup:
    pass


# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)


class CharacterSeparatedList(click.ParamType):
    """List of elements of a given type separated by a given character."""

    def __init__(self, type_, separator, *, name=None, unique=False):
        """Create an instance of ``CharacterSeparatedList``.

        Args:
            type_ (type): Type of list elements.

            separator (str): Separator of list elements.

            name (str, optional): Name of the type. If omitted, the default
                name is derived from ``type_`` and ``separator``. Defaults to
                None.

            unique (bool, optional): Whether the list elements must be unique.
                Defaults to False.

        Example:
            Create type for comma-separated list of (unique) integers:

            > comma_separated_list_of_unique_ints = CharacterSeparatedList(int, ',')

        """
        if isinstance(type_, float) and separator == ".":
            raise ValueError(
                f"invalid separator '{separator}' for type " f"'{type_.__name__}'"
            )

        self.type_ = type_
        self.separator = separator
        self.unique = unique
        if name is not None:
            self.name = name
        else:
            self.name = f"'{separator}'-separated {type_.__name__} list"

    def convert(self, value, param, ctx):
        """Convert a string to a list of ``type_`` elements."""
        values_str = value.split(self.separator)
        values = []
        for i, value_str in enumerate(values_str):
            try:
                value = self.type_(value_str)
            except (ValueError, TypeError) as e:
                self.fail(
                    f"Invalid '{self.separator}'-separated list '{value}': "
                    f"Value '{value_str}' ({i + 1}/{len(values_str)}) "
                    f"incompatible with type '{self.type_.__name__}' "
                    f"({type(e).__name__}: {e})"
                )
            else:
                if self.unique and value in values:
                    n = len(values_str)
                    self.fail(
                        f"Invalid '{self.separator}'-separated list "
                        f"'{value}': Value '{value_str}' ({i + 1}/{n}) "
                        f"not unique"
                    )
                values.append(value)
        return values


comma_separated_list_of_unique_ints = CharacterSeparatedList(int, ",", unique=True)
plus_separated_list_of_unique_ints = CharacterSeparatedList(int, "+", unique=True)


class CombinationChoices(click.ParamType):
    """Choices that can also be combined."""

    def __init__(self, base_choices, combination_choices):
        """Create instance of ``CombinationChoices``.

        Args:
            base_choices (list[str]): Base choices.

            combination_choices (dict[str, list[str]]): Derived choices
                constituting combinations of multiple base choices.

        """
        self.base_choices = base_choices
        self.combination_choices = combination_choices
        self._check_combination_choices()

    def convert(self, value, param, ctx):
        """Check that a string is among the given choices or combinations."""
        if value in self.base_choices:
            return value
        try:
            return self.combination_choices[value]
        except KeyError:
            choices = self.base_choices + list(self.combination_choices)
            s_choices = ", ".join([f"'{s}'" for s in choices])
            self.fail(f"wrong choice '{value}': must be one of {s_choices}")

    def _check_combination_choices(self):
        for choice, combination in self.combination_choices.items():
            if choice in self.base_choices:
                raise ValueError(
                    "combination choice is already a base choice",
                    choice=choice,
                    base_choices=self.base_choices,
                )
            try:
                it = iter(combination)
            except TypeError:
                raise ValueError(
                    "combination choice is defined as non-iterable "
                    f"{type(combination).__name__} object",
                    choice=choice,
                    combination=combination,
                )
            else:
                for element in it:
                    if element not in self.base_choices:
                        raise ValueError(
                            "combination choice element is not a base choice",
                            choice=choice,
                            element=element,
                            base_choices=self.base_choices,
                        )


#
# TODO
#
# - Eliminate subcommands (e.g., "deterministic concentration")
#   -> Instead, use flags, e.g., --[no-]ens or --sim-type or whatnot
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
    var_in,
    in_file_path_raw,
    out_file_path_raw,
    var_specs_raw,
    ens_var=None,
    ens_member_id_lst=None,
    **kwargs,
):
    """Read and plot FLEXPART data.

    Args:
        TODO

    """

    lang = ctx.obj["lang"]

    # SR_TMP < TODO find cleaner solution
    if ens_var is None:
        cls_name = f"{var_in}"
        ens_var_setup = None
    else:
        cls_name = f"ens_{ens_var}_{var_in}"
        ens_var_setup = {"thr_agrmt": {"thr": 1e-9},}.get(ens_var)  # SR_TMP SR_HC
    # SR_TMP >

    # Create variable specification objects
    var_specs_dct = prep_var_specs_dct(var_specs_raw)
    multi_var_specs_lst = MultiVarSpecs.create(
        cls_name, var_specs_dct, lang=lang, words=None,
    )

    field_lst = read_fields(
        cls_name=cls_name,
        multi_var_specs_lst=multi_var_specs_lst,
        ens_member_id_lst=ens_member_id_lst,
        ens_var=ens_var,
        ens_var_setup=ens_var_setup,
        in_file_path_raw=in_file_path_raw,
        lang=lang,
    )

    if ctx.obj["noplot"]:
        return

    # Transfer some global options
    kwargs = {key: ctx.obj[key] for key in ["lang", "scale_fact"]}

    # Prepare plotter
    plotter = Plotter()
    fct_plot = lambda: plotter.run(cls_name, field_lst, out_file_path_raw, **kwargs)

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
    """Prepare the variable specifications dict from the raw CLI input.

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
    in_file_path_raw,
    cls_name,
    multi_var_specs_lst,
    lang,
    ens_member_id_lst,
    ens_var,
    ens_var_setup,
):
    """TODO"""

    attrs = {"lang": lang}
    if ens_member_id_lst is not None:
        attrs["member_ids"] = ens_member_id_lst
    if ens_var is not None:
        attrs["ens_var"] = ens_var
    if ens_var_setup is not None:
        attrs["ens_var_setup"] = ens_var_setup

    # Determine fields specifications (one for each eventual plot)
    fld_specs_lst = [
        FieldSpecs(cls_name, multi_var_specs, attrs)
        for multi_var_specs in multi_var_specs_lst
    ]

    # Read fields
    field_lst = FileReader(in_file_path_raw).run(fld_specs_lst, lang=lang)

    return field_lst


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


def not_implemented(msg):
    def f(ctx, param, value):
        if value:
            click.echo(f"not implemented: {msg}")
            ctx.exit(1)

    return f


class GlobalOptions(ClickOptionsGroup):
    """Options shared by all types of plots."""

    @click_options
    def execution():
        """Options passed before any command."""
        return [
            click.option(
                "--dry-run",
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
                help="Increase verbosity; specify multiple times for more.",
                count=True,
            ),
            click.option(
                "--noplot", help="Skip plotting (for debugging etc.).", is_flag=True,
            ),
            click.option(
                "--lang",
                help="Language. Format key: '{lang}'.",
                type=click.Choice(["en", "de"]),
                default="en",
            ),
            click.option(
                "--open-first",
                "open_first_cmd",
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
                "open_all_cmd",
                help="Like --open-first, but for all plots.",
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
    def input():
        return [
            click.option(
                "--infile",
                "-i",
                "in_file_path_raw",
                help=(
                    "Input file path. May contain format keys to embed parameters such "
                    "as the member id of ensemble simulation output (see respective "
                    "options for the key names)."
                ),
                type=str,
                required=True,
            ),
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
                type=plus_separated_list_of_unique_ints,
                default=["1"],
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
        ]

    @click_options
    def plot():
        return [
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

    @click_options
    def output():
        return [
            click.option(
                "--outfile",
                "-o",
                "out_file_path_raw",
                help=(
                    "Output file path. If multiple plots are to be created, e.g., for "
                    "multiple fields or levels, ``outfile`` must contain format keys "
                    "for inserting all changing parameters (example: "
                    "``plot_lvl-{level}.png`` for multiple levels). The format key for "
                    "the plotted variable is '{variable}'. See individual options for "
                    "the respective format keys."
                ),
                type=click.Path(writable=True),
                required=True,
            ),
        ]


class EnsembleOptions(ClickOptionsGroup):
    @click_options
    def input():
        return [
            click.option(
                "--ens-member-id",
                "-m",
                "ens_member_id_lst",
                help=(
                    "Ensemble member id. Repeat for multiple members. Omit for "
                    "deterministic simulation data. Use the format key '{member_id}' "
                    "to embed the member id(s) in the plot file path."
                ),
                type=int,
                multiple=True,
                required=True,
            ),
        ]

    @click_options
    def plot():
        return [
            click.option(
                "--ens-var",
                help="Ensemble variable to plot. Requires ensemble simulation data.",
                type=click.Choice(["mean", "max", "thr_agrmt"]),
                required=True,
            )
        ]


class ConcentrationOptions(ClickOptionsGroup):
    @click_options
    def input():
        return [
            click.option(
                "--level-ind",
                "var_specs_raw__level_lst",
                help=(
                    "Index/indices of vertical level (zero-based, bottom-up). To sum "
                    "up multiple levels, combine their indices with '+'. Format key: "
                    "'{level_ind}'."
                ),
                type=plus_separated_list_of_unique_ints,
                default=["0"],
                multiple=True,
            ),
        ]

    @click_options
    def plot():
        return [
            click.option(
                "--plot-var",
                help="What variable to plot/how to plot the input variable.",
                type=click.Choice(["auto"]),
                default="auto",
            ),
        ]


class DepositionOptions(ClickOptionsGroup):
    @click_options
    def input():
        """Common options of dispersion plots (deposition)."""
        return [
            click.option(
                "--deposition-type",
                "var_specs_raw__deposition_lst",
                help=(
                    "Type of deposition. Part of the plot variable name that may be "
                    "embedded in the plot file path with the format key '{variable}'."
                ),
                type=CombinationChoices(["wet", "dry"], {"tot": ("wet", "dry")}),
                default=["tot"],
                multiple=True,
            )
        ]

    @click_options
    def plot_deterministic():
        return [
            click.option(
                "--plot-var",
                help="What variable to plot/how to plot the input variable.",
                type=click.Choice(["auto", "affected_area", "affected_area_mono",]),
                default="auto",
            )
        ]

    @click_options
    def plot_ensemble():
        return [
            click.option(
                "--plot-var",
                help="What variable to plot/how to plot the input variable.",
                type=click.Choice(["auto",]),
                default="auto",
            )
        ]


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]}, no_args_is_help=True,
)
@GlobalOptions.execution
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.pass_context
def cli(ctx, **kwargs):
    """Create FLEXPART dispersion plots."""

    click.echo("Welcome fellow PyFlexPlotter!")

    log.basicConfig(level=count_to_log_level(kwargs["verbose"]))

    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Store shared keyword arguments in ctx.obj
    ctx.obj.update(kwargs)

    return 0


@click.group()
def deterministic():
    return 0


cli.add_command(deterministic)


@click.group()
def ensemble():
    return 0


cli.add_command(ensemble)


@deterministic.command(
    name="concentration", help="Activity concentration; deterministic simulation data.",
)
@GlobalOptions.input
@GlobalOptions.output
@GlobalOptions.preproc
@GlobalOptions.plot
@ConcentrationOptions.input
@ConcentrationOptions.plot
@click.pass_context
@group_kwargs("var_specs_raw")
def deterministic_concentration(ctx, plot_var, **kwargs):
    var_in = "concentration"
    create_plots(ctx=ctx, var_in=var_in, **kwargs)


@deterministic.command(
    name="deposition", help="Surface deposition; deterministic simulation data.",
)
@GlobalOptions.input
@GlobalOptions.output
@GlobalOptions.preproc
@GlobalOptions.plot
@DepositionOptions.input
@DepositionOptions.plot_deterministic
@click.pass_context
@group_kwargs("var_specs_raw")
def deterministic_deposition(ctx, plot_var, **kwargs):
    var_in = "deposition" if plot_var == "auto" else plot_var
    create_plots(ctx=ctx, var_in=var_in, **kwargs)


@ensemble.command(
    name="concentration", help="Activity concentration; ensemble simulation data.",
)
@GlobalOptions.input
@GlobalOptions.output
@GlobalOptions.preproc
@GlobalOptions.plot
@EnsembleOptions.input
@EnsembleOptions.plot
@ConcentrationOptions.input
@ConcentrationOptions.plot
@click.pass_context
@group_kwargs("var_specs_raw")
def ensemble_concentration(ctx, plot_var, **kwargs):
    var_in = "concentration"
    create_plots(ctx=ctx, var_in=var_in, **kwargs)


@ensemble.command(
    name="deposition", help="Surface deposition; ensemble simulation data.",
)
@GlobalOptions.input
@GlobalOptions.output
@GlobalOptions.preproc
@GlobalOptions.plot
@EnsembleOptions.input
@EnsembleOptions.plot
@DepositionOptions.input
@DepositionOptions.plot_ensemble
@click.pass_context
@group_kwargs("var_specs_raw")
def ensemble_deposition(ctx, plot_var, **kwargs):
    var_in = {"auto": "deposition"}.get(plot_var, plot_var)
    create_plots(ctx=ctx, var_in=var_in, **kwargs)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover

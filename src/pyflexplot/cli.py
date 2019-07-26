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

from .io import FlexFieldSpecsConcentration
from .io import FlexFieldSpecsDeposition
from .io import FlexFieldSpecsAffectedArea
from .io import FlexFileRotPole
from .utils import count_to_log_level
from .flexplotter import FlexPlotter

from .utils_dev import ipython  #SR_DEV

__version__ = '0.1.0'

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)


class CharSepListParamType(click.ParamType):

    def __init__(self, type_, separator, *, name=None, dupl_ok=False):
        """Create an instance of ``CharSepListParamType``.

        Args:
            type_ (type): Type of list elements.

            separator (str): Separator of list elements.

            name (str, optional): Name of the type. If omitted, the
                default name is derived from ``type_`` and ``separator``.
                Defaults to None.

            dupl_ok (bool, optional): Whether duplicate values are
                allowed. Defaults to False.

        Example:
            Create type for comma-separated list of (unique) integers:

            > INT_LIST_COMMA_SEP_UNIQ = CharSepListParamType(int, ',')

        """
        if isinstance(type_, float) and separator == '.':
            raise ValueError(
                f"invalid separator '{separator}' for type '{type_.__name__}'")

        self.type_ = type_
        self.separator = separator
        self.dupl_ok = dupl_ok
        if name is not None:
            self.name = name
        else:
            self.name = f"'{separator}'-separated {type_.__name__} list"

    def convert(self, value, param, ctx):
        """Convert string to list of given type based on separator."""
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
                    f"({type(e).__name__}: {e})")
            else:
                if not self.dupl_ok and value in values:
                    self.fail(
                        f"Invalid '{self.separator}'-separated list '{value}': "
                        f"Value '{value_str}' ({i + 1}/{len(values_str)}) "
                        f"not unique")
                values.append(value)
        return values


INT_LIST_COMMA_SEP_UNIQ = CharSepListParamType(
    int, ',', dupl_ok=False, name='integers (comma-separated)')
INT_LIST_PLUS_SEP_UNIQ = CharSepListParamType(
    int, '+', dupl_ok=False, name='integers (plus-separated)')


# yapf: disable
@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--dry-run', '-n',
    help="Perform a trial run with no changes made.",
    flag_value='dry_run', default=False)
@click.option(
    '--verbose', '-v',
    help="Increase verbosity (specify multiple times for more).",
    count=True)
@click.option(
    '--version', '-V',
    help="Print version.",
    is_flag=True)
@click.option(
    '--noplot',
    help="Skip plotting (for debugging etc.).",
    is_flag=True)
@click.option(
    '--open-first', 'open_first_cmd',
    help=(
        "Shell command to open the first plot as soon as it is available."
        " The file path follows the command, unless explicitly set by"
        " including the format key '{file_path}'."))
@click.option(
    '--open-all', 'open_all_cmd',
    help="Like --open-first, but for all plots.")
@click.pass_context
# yapf: enable
def main(ctx, **kwargs):
    """Console script for test_cli_project."""

    click.echo("Hi fellow PyFlexPlotter!")
    #click.echo(f"{len(kwargs)} kwargs:\n{pformat(kwargs)}\n")

    log.basicConfig(level=count_to_log_level(kwargs['verbose']))

    if kwargs['version']:
        click.echo(__version__)
        return 0

    if kwargs.pop('dry_run'):
        raise NotImplementedError("dry run")
        return 0

    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Store shared keyword arguments in ctx.obj
    ctx.obj.update(kwargs)

    return 0


def common_options_global(f):
    """Common options of all commands in the group.

    Defining the common options this way instead of on the group
    enables putting them after the command instead of before it.

    src: https://stackoverflow.com/a/52147284
    """
    # yapf: disable
    options = [
        click.option(
            '--infile', '-i', 'in_file_path',
            help="Input file path.",
            type=click.Path(exists=True, readable=True), required=True),
        click.option(
            '--outfile', '-o', 'out_file_path_fmt',
            help=(
                "Output file path. If multiple plots are to be created, "
                "e.g., for multiple fields or levels, ``outfile`` must "
                "contain format keys for inserting all changing parameters "
                "(example: ``plot_lvl-{level}.png`` for multiple levels). "
                "The format key for the plotted variable is '{variable}'. "
                "See individual options for the respective format keys."),
            type=click.Path(writable=True), required=True),
    ]
    # yapf: enable
    return functools.reduce(lambda x, opt: opt(x), options, f)


def common_options_dispersion_input(f):
    """Common options of dispersion plots (field selection)."""
    # yapf: disable
    options = [
        click.option(
            '--time-ind', 'time_lst',
            help="Index of time (zero-based). Format key: '{time_ind}'.",
            type=int, default=[0], multiple=True),
        click.option(
            '--age-class-ind', 'nageclass_lst',
            help=("Index of age class (zero-based). Format key: "
                "'{age_class_ind}'."),
            type=int, default=[0], multiple=True),
        click.option(
            '--release-point-ind', 'numpoint_lst',
            help=("Index of release point (zero-based). Format key: "
                "'{rls_pt_ind}'."),
            type=int, default=[0], multiple=True),
        click.option(
            '--species-id', 'species_id_lst',
            help=(
                "Species id(s) (default: 0). To sum up multiple species, "
                "combine their ids with '+'. Format key: '{species_id}'."),
            type=INT_LIST_PLUS_SEP_UNIQ, default=[0], multiple=True),
    ]
    # yapf: enable
    return functools.reduce(lambda x, opt: opt(x), options, f)


def common_options_dispersion_preproc(f):
    """Common options of dispersion plots (pre-processing)."""
    # yapf: disable
    options = [
        click.option(
            '--integrate/--no-integrate', 'integrate_lst',
            help=("Integrate field over time. If set, '-int' is "
                "appended to variable name (format key: '{variable}')."),
            is_flag=True, default=[False], multiple=True),
    ]
    # yapf: enable
    return functools.reduce(lambda x, opt: opt(x), options, f)


def common_options_dispersion_deposition(f):
    """Common options of dispersion plots (deposition)."""
    # yapf: disable
    options = [
        click.option(
            '--deposition-type', 'deposition_lst',
            help=("Type of deposition. Part of plot variable (format "
                "key: '{variable}')."),
            type=click.Choice(['tot', 'wet', 'dry']),
            default='tot', multiple=True)
    ]
    # yapf: enable
    return functools.reduce(lambda x, opt: opt(x), options, f)


@main.command(help="Activity concentration in the air.")
@common_options_global
@common_options_dispersion_input
# yapf: disable
@click.option(
    '--level-ind', 'level_lst',
    help=(
        "Index/indices of vertical level (zero-based, bottom-up). To sum up "
        "multiple levels, combine their indices with '+'. Format key: "
        "'{level_ind}'."),
    type=INT_LIST_PLUS_SEP_UNIQ, default=[0], multiple=True)
# yapf: enable
@common_options_dispersion_preproc
@click.pass_context
def concentration(ctx, in_file_path, out_file_path_fmt, **vars_specs):

    # Determine fields specifications (one for each eventual plot)
    fld_specs_lst = FlexFieldSpecsConcentration.multiple(vars_specs)

    # Read fields
    flex_data_lst = FlexFileRotPole(in_file_path).read(fld_specs_lst)

    # Create plots
    create_plots(
        ctx, FlexPlotter.concentration, [flex_data_lst, out_file_path_fmt])


@main.command(help="Surface deposition.")
@common_options_global
@common_options_dispersion_input
@common_options_dispersion_preproc
@common_options_dispersion_deposition
@click.pass_context
def deposition(ctx, in_file_path, out_file_path_fmt, **vars_specs):

    # Determine fields specifications (one for each eventual plot)
    field_specs_lst = FlexFieldSpecsDeposition.multiple(vars_specs)

    # Read fields
    flex_data_lst = FlexFileRotPole(in_file_path).read(field_specs_lst)

    # Create plots
    create_plots(
        ctx, FlexPlotter.deposition, [flex_data_lst, out_file_path_fmt])


@main.command(
    name='affected-area',
    help="Area affected by surface deposition.",
)
@common_options_global
@common_options_dispersion_input
@common_options_dispersion_preproc
@common_options_dispersion_deposition
# yapf: disable
@click.option(
    '--mono/--no-mono',
    help="Only use one threshold (monochromatic plot).",
    is_flag=True, default=[False], multiple=True)
# yapf: enable
@click.pass_context
def affected_area(ctx, in_file_path, out_file_path_fmt, mono, **vars_specs):

    # Determine fields specifications (one for each eventual plot)
    field_specs_lst = FlexFieldSpecsAffectedArea.multiple(vars_specs)

    # Read fields
    flex_data_lst = FlexFileRotPole(in_file_path).read(field_specs_lst)

    # Create plots
    fct = FlexPlotter.affected_area_mono if mono else FlexPlotter.affected_area
    create_plots(ctx, fct, [flex_data_lst, out_file_path_fmt])


def create_plots(ctx, fct, args=None, kwargs=None):
    """Create FLEXPART plots.

    Args:
        ctx (Context): Click context object.

        fct (callable): Callable that creates plots while yielding
            the output file paths on-the-fly.

        args (list, optional): Positional arguments for ``fct``.
            Defaults to [].

        kwargs (dict, optional): Keyword arguments for ``fct``.
            Defaults to {}.

    """
    if ctx.obj['noplot']:
        return

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    # Note: FlexPlotter.run yields the output file paths on-the-go
    out_file_paths = []
    for i, out_file_path in enumerate(fct(*args, **kwargs)):
        out_file_paths.append(out_file_path)

        if ctx.obj['open_first_cmd'] and i == 0:
            # Open the first file as soon as it's available
            open_plots(ctx.obj['open_first_cmd'], [out_file_path])

    if ctx.obj['open_all_cmd']:
        # Open all plots
        open_plots(ctx.obj['open_all_cmd'], out_file_paths)


def open_plots(cmd, file_paths):
    """Open a plot file using a shell command."""

    # If not yet included, append the output file path
    if '{file_paths}' not in cmd:
        cmd += ' {file_paths}'

    # Ensure that the command is run in the background
    if not cmd.rstrip().endswith('&'):
        cmd += ' &'

    # Run the command
    cmd = cmd.format(file_paths=' '.join(file_paths))
    os.system(cmd)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

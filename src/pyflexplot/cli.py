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

from .io import FlexFieldSpecs
from .io import FlexFileReader
from .utils import count_to_log_level
from .flexplotter import FlexPlotter

from .utils_dev import ipython  #SR_DEV

__version__ = '0.1.0'

#======================================================================


def click_options(f_options):
    """Define a list of click options shared by multiple commands.

    Args:
        f_options (function): Function returning a list of ``click.option``
            objects.

    Example:
        > @click_options          # <== define options
        > def common_options():
        >     return [click.option(...), click.option(...), ...]

        > @click.group
        > def main(...):
        >     ...

        > @CLI.command
        > @common_options         # <== use options
        > def foo(...):
        >     ...

        > @CLI.command
        > @common_options         # <== use options
        > def bar(...):
        >     ...


    Applications:
        * Define options that are shared by multiple commands but are
          passed after the respective command, instead of before as
          group options (the native way to define shared options) are.

        * Define options used only by a single group or command in a
          function instead of as decorators, which allows them to be
          folded by the editor more easily.

    Source:
        https://stackoverflow.com/a/52147284

    """
    return lambda f: functools.reduce(lambda x, opt: opt(x), f_options(), f)


class ClickGroup:

    def command(*args, **kwargs):
        return CLI.cli.command(*args, **kwargs)


class ClickOptionsGroup:

    pass


class ClickCommand:

    pass


#======================================================================


class FloatOrStrParamType(click.ParamType):
    """Float or certain string."""

    def __init__(self, str_lst):
        self.str_lst = str_lst

    def convert(self, value, param, ctx):
        """Convert string to float, unless it is among ``str_lst``."""
        if value in self.str_lst:
            return value
        return float(value)


FLOAT_OR_AUTO = FloatOrStrParamType('auto')


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
                f"invalid separator '{separator}' for type "
                f"'{type_.__name__}'")

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
                    n = len(values_str)
                    self.fail(
                        f"Invalid '{self.separator}'-separated list "
                        f"'{value}': Value '{value_str}' ({i + 1}/{n}) "
                        f"not unique")
                values.append(value)
        return values


INT_LIST_COMMA_SEP_UNIQ = CharSepListParamType(
    int, ',', dupl_ok=False, name='integers (comma-separated)')
INT_LIST_PLUS_SEP_UNIQ = CharSepListParamType(
    int, '+', dupl_ok=False, name='integers (plus-separated)')

#======================================================================

# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)


class CLI(ClickGroup):

    @click_options
    def options():
        return [
            click.option(
                '--dry-run',
                '-n',
                help="Perform a trial run with no changes made.",
                flag_value='dry_run',
                default=False,
            ),
            click.option(
                '--verbose',
                '-v',
                help="Increase verbosity; specify multiple times for more.",
                count=True,
            ),
            click.option(
                '--version',
                '-V',
                help="Print version.",
                is_flag=True,
            ),
            click.option(
                '--noplot',
                help="Skip plotting (for debugging etc.).",
                is_flag=True,
            ),
            click.option(
                '--lang',
                help="Language. Format key: '{lang}'.",
                type=click.Choice(['en', 'de']),
                default='en',
            ),
            click.option(
                '--open-first',
                'open_first_cmd',
                help=(
                    "Shell command to open the first plot as soon as it is "
                    "available. The file path follows the command, unless "
                    "explicitly set by including the format key '{file_path}'."
                ),
            ),
            click.option(
                '--open-all',
                'open_all_cmd',
                help="Like --open-first, but for all plots.",
            ),
        ]

    @click.group(
        context_settings={
            'help_option_names': ['-h', '--help'],  # Add short flag '-h'
        },)
    @options
    @click.pass_context
    def cli(ctx, **kwargs):
        """Point of entry."""

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


class GlobalOptions(ClickOptionsGroup):

    @click_options
    def input_deterministic():
        return [
            click.option(
                '--infile',
                '-i',
                'in_file_path',
                help="Input file path.",
                type=click.Path(exists=True, readable=True),
                required=True,
            ),
        ]

    @click_options
    def input_ensemble():
        return [
            click.option(
                '--infile-fmt',
                '-i',
                'in_file_path_fmt',
                help=(
                    "Input file path format string, containing format key "
                    "'{member_id}' for the ensemble member. If the ids are "
                    "zero-padded, use, e.g., '{member_id:03d}'."),
                type=click.Path(exists=False, readable=True),
                required=True,
            ),
            click.option(
                '--member-id',
                '-m',
                'member_id_lst',
                help=(
                    "ID of ensemble member. Repeat for multiple members. "
                    "Input file format key: {member_id}."),
                type=int,
                multiple=True,
                required=True,
            ),
        ]

    @click_options
    def output():
        return [
            click.option(
                '--outfile',
                '-o',
                'out_file_path_fmt',
                help=(
                    "Output file path. If multiple plots are to be created, "
                    "e.g., for multiple fields or levels, ``outfile`` must "
                    "contain format keys for inserting all changing parameters "
                    "(example: ``plot_lvl-{level}.png`` for multiple levels). "
                    "The format key for the plotted variable is '{variable}'. "
                    "See individual options for the respective format keys."),
                type=click.Path(writable=True),
                required=True,
            ),
        ]


def create_plots(ctx, cls_plotter, args=None, kwargs=None):
    """Create FLEXPART plots.

    Args:
        ctx (Context): Click context object.

        cls_plotter (type): Plotter class, derived from FlexPlotter.

        args (list, optional): Positional arguments for ``cls_plotter.run``.
            Defaults to [].

        kwargs (dict, optional): Keyword arguments for ``cls_plotter.run``.
            Defaults to {}.

    """

    if ctx.obj['noplot']:
        return

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    plotter = cls_plotter()

    # Note: FlexPlotter.run yields the output file paths on-the-go
    out_file_paths = []
    for i, out_file_path in enumerate(plotter.run(*args, **kwargs)):
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


#======================================================================


class DispersionOptions(ClickOptionsGroup):

    @click_options
    def input():
        """Common options of dispersion plots (field selection)."""
        return [
            click.option(
                '--time-ind',
                'time_lst',
                help="Index of time (zero-based). Format key: '{time_ind}'.",
                type=int,
                default=[0],
                multiple=True,
            ),
            click.option(
                '--age-class-ind',
                'nageclass_lst',
                help=(
                    "Index of age class (zero-based). Format key: "
                    "'{age_class_ind}'."),
                type=int,
                default=[0],
                multiple=True,
            ),
            click.option(
                '--release-point-ind',
                'numpoint_lst',
                help=(
                    "Index of release point (zero-based). Format key: "
                    "'{rls_pt_ind}'."),
                type=int,
                default=[0],
                multiple=True,
            ),
            click.option(
                '--species-id',
                'species_id_lst',
                help=(
                    "Species id(s) (default: 0). To sum up multiple species, "
                    "combine their ids with '+'. Format key: '{species_id}'."),
                type=INT_LIST_PLUS_SEP_UNIQ,
                default=[0],
                multiple=True,
            ),
        ]

    @click_options
    def preproc():
        """Common options of dispersion plots (pre-processing)."""
        return [
            click.option(
                '--integrate/--no-integrate',
                'integrate_lst',
                help=(
                    "Integrate field over time. If set, '-int' is "
                    "appended to variable name (format key: '{variable}')."),
                is_flag=True,
                default=[False],
                multiple=True,
            ),
        ]


class Concentration(ClickCommand):

    @click_options
    def options():
        return [
            click.option(
                '--level-ind',
                'level_lst',
                help=(
                    "Index/indices of vertical level (zero-based, bottom-up). "
                    "To sum up multiple levels, combine their indices with "
                    "'+'. Format key: '{level_ind}'."),
                type=INT_LIST_PLUS_SEP_UNIQ,
                default=[0],
                multiple=True,
            ),
        ]

    @CLI.command(
        name='concentration',
        help="Activity concentration in the air.",
    )
    @GlobalOptions.input_deterministic
    @GlobalOptions.output
    @DispersionOptions.input
    @DispersionOptions.preproc
    @options
    @click.pass_context
    def concentration(ctx, in_file_path, out_file_path_fmt, **vars_specs):

        lang = ctx.obj['lang']

        # Determine fields specifications (one for each eventual plot)
        fld_specs_lst = FlexFieldSpecs.Concentration.multiple(
            vars_specs, lang=lang)

        # Read fields
        flex_field_lst = FlexFileReader(in_file_path).run(
            fld_specs_lst, lang=lang)

        # Create plots
        create_plots(
            ctx,
            FlexPlotter.Concentration,
            [flex_field_lst, out_file_path_fmt],
            {'lang': lang},
        )


class Deposition(ClickCommand):

    @click_options
    def options():
        """Common options of dispersion plots (deposition)."""
        return [
            click.option(
                '--deposition-type',
                'deposition_lst',
                help=(
                    "Type of deposition. Part of plot variable (format "
                    "key: '{variable}')."),
                type=click.Choice(['tot', 'wet', 'dry']),
                default='tot',
                multiple=True,
            )
        ]

    @CLI.command(
        name='deposition',
        help="Surface deposition.",
    )
    @GlobalOptions.input_deterministic
    @GlobalOptions.output
    @DispersionOptions.input
    @DispersionOptions.preproc
    @options
    @click.pass_context
    def deposition(ctx, in_file_path, out_file_path_fmt, **vars_specs):

        lang = ctx.obj['lang']

        # Determine fields specifications (one for each eventual plot)
        field_specs_lst = FlexFieldSpecs.Deposition.multiple(
            vars_specs, lang=lang)

        # Read fields
        flex_field_lst = FlexFileReader(in_file_path).run(
            field_specs_lst, lang=lang)

        # Create plots
        create_plots(
            ctx,
            FlexPlotter.Deposition,
            [flex_field_lst, out_file_path_fmt],
            {'lang': lang},
        )


class AffectedArea(ClickCommand):

    @click_options
    def options():
        return [
            click.option(
                '--mono/--no-mono',
                'mono_lst',
                help="Only use one threshold (monochromatic plot).",
                is_flag=True,
                default=[False],
                multiple=True,
            ),
        ]

    @CLI.command(
        name='affected-area',
        help="Area affected by surface deposition.",
    )
    @GlobalOptions.input_deterministic
    @GlobalOptions.output
    @DispersionOptions.input
    @DispersionOptions.preproc
    @Deposition.options
    @options
    @click.pass_context
    def affected_area(
            ctx, in_file_path, out_file_path_fmt, mono_lst, **vars_specs):

        lang = ctx.obj['lang']

        # Determine fields specifications (one for each eventual plot)
        field_specs_lst = FlexFieldSpecs.AffectedArea.multiple(
            vars_specs, lang=lang)

        # Read fields
        flex_field_lst = FlexFileReader(in_file_path).run(
            field_specs_lst, lang=lang)

        # Create plots
        for mono in mono_lst:
            if mono:
                fct = FlexPlotter.AffectedAreaMono
            else:
                fct = FlexPlotter.AffectedArea
            create_plots(
                ctx,
                fct,
                [flex_field_lst, out_file_path_fmt],
                {'lang': lang},
            )


#----------------------------------------------------------------------


class EnsMeanConcentration(ClickCommand):

    @click_options
    def options():
        return []

    @CLI.command(
        name='ens-mean-concentration',
        help="Ensemble-mean of activity concentration in the air.",
    )
    @GlobalOptions.input_ensemble
    @GlobalOptions.output
    @DispersionOptions.input
    @DispersionOptions.preproc
    @Concentration.options
    @options
    @click.pass_context
    def end_mean_concentration(
            ctx, in_file_path_fmt, out_file_path_fmt, member_id_lst,
            **vars_specs):

        lang = ctx.obj['lang']

        # Determine fields specifications (one for each eventual plot)
        fld_specs_lst = FlexFieldSpecs.EnsMeanConcentration.multiple(
            vars_specs, member_ids=member_id_lst, ens_var='mean', lang=lang)

        # Read fields
        flex_field_lst = FlexFileReader(in_file_path_fmt).run(
            fld_specs_lst, lang=lang)

        # Create plots
        create_plots(
            ctx,
            FlexPlotter.EnsMeanConcentration,
            [flex_field_lst, out_file_path_fmt],
            {'lang': lang},
        )


class EnsMeanDeposition(ClickCommand):

    @click_options
    def options():
        return []

    @CLI.command(
        name='ens-mean-deposition',
        help="Ensemble-mean of surface deposition.",
    )
    @GlobalOptions.input_ensemble
    @GlobalOptions.output
    @DispersionOptions.input
    @DispersionOptions.preproc
    @Deposition.options
    @options
    @click.pass_context
    def end_mean_deposition(
            ctx, in_file_path_fmt, out_file_path_fmt, member_id_lst,
            **vars_specs):

        lang = ctx.obj['lang']

        # Determine fields specifications (one for each eventual plot)
        fld_specs_lst = FlexFieldSpecs.EnsMeanDeposition.multiple(
            vars_specs, member_ids=member_id_lst, ens_var='mean', lang=lang)

        # Read fields
        flex_field_lst = FlexFileReader(in_file_path_fmt).run(
            fld_specs_lst, lang=lang)

        # Create plots
        create_plots(
            ctx,
            FlexPlotter.EnsMeanDeposition,
            [flex_field_lst, out_file_path_fmt],
            {'lang': lang},
        )


class EnsMeanAffectedArea(ClickCommand):

    @click_options
    def options():
        return []

    @CLI.command(
        name='ens-mean-affected-area',
        help="Ensemble-mean of area affected by surface deposition.",
    )
    @GlobalOptions.input_ensemble
    @GlobalOptions.output
    @DispersionOptions.input
    @DispersionOptions.preproc
    @Deposition.options
    @AffectedArea.options
    @options
    @click.pass_context
    def end_mean_affected_area(
            ctx, in_file_path_fmt, out_file_path_fmt, mono_lst, member_id_lst,
            **vars_specs):

        lang = ctx.obj['lang']

        # Determine fields specifications (one for each eventual plot)
        fld_specs_lst = FlexFieldSpecs.EnsMeanAffectedArea.multiple(
            vars_specs, member_ids=member_id_lst, ens_var='mean', lang=lang)

        # Read fields
        flex_field_lst = FlexFileReader(in_file_path_fmt).run(
            fld_specs_lst, lang=lang)

        # Create plots
        for mono in mono_lst:
            if mono:
                fct = FlexPlotter.EnsMeanAffectedAreaMono
            else:
                fct = FlexPlotter.EnsMeanAffectedArea
            create_plots(
                ctx,
                fct,
                [flex_field_lst, out_file_path_fmt],
                {'lang': lang},
            )


class EnsThrAgrmt(ClickCommand):

    @click_options
    def options():
        return [
            click.option(
                '--threshold',
                #'threshold_lst',
                'threshold',
                help=(
                    "Threshold to be exceeded. Pass 'auto' to derive "
                    "it automatically from the input field."),
                type=FLOAT_OR_AUTO,
                #default=['auto'],
                #multiple=True,
                default='auto',
            ),
        ]

    @CLI.command(
        name='ens-threshold-agreement-concentration',
        help=(
            "Ensemble threshold agreement of activity concentration "
            "in the air."),
    )
    @GlobalOptions.input_ensemble
    @GlobalOptions.output
    @DispersionOptions.input
    @DispersionOptions.preproc
    @Concentration.options
    @options
    @click.pass_context
    def end_threshold_agreement_concentration(
            ctx, in_file_path_fmt, out_file_path_fmt, member_id_lst, threshold,
            **vars_specs):

        lang = ctx.obj['lang']

        #SR_TMP<
        if threshold == 'auto':
            threshold = 0.0
        #SR_TMP>

        # Determine fields specifications (one for each eventual plot)
        _cls = FlexFieldSpecs.EnsThrAgrmtConcentration
        fld_specs_lst = _cls.multiple(
            vars_specs,
            member_ids=member_id_lst,
            ens_var='threshold-agreement',
            ens_var_setup={'thr': threshold},
            lang=lang)

        # Read fields
        flex_field_lst = FlexFileReader(in_file_path_fmt).run(
            fld_specs_lst, lang=lang)

        # Create plots
        create_plots(
            ctx,
            FlexPlotter.EnsThrAgrmtConcentration,
            [flex_field_lst, out_file_path_fmt],
            {'lang': lang},
        )


#======================================================================

if __name__ == "__main__":
    sys.exit(CLI.main())  # pragma: no cover

# -*- coding: utf-8 -*-
# pylint: disable=R0914  # too-many-locals
"""
Command line interface.
"""
# Standard library
import functools
import os
import re
import sys
import traceback
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

# Third-party
import click

# Local
from .. import __version__
from .. import data_path
from ..input import read_fields
from ..plots import create_plot
from ..plots import prepare_plot
from ..setup import Setup
from ..setup import SetupFile
from ..utils.formatting import format_range
from ..utils.logging import log
from ..utils.logging import set_log_level
from .click import click_error
from .preset import add_to_preset_paths
from .preset_click import click_add_to_preset_paths
from .preset_click import click_cat_preset_and_exit
from .preset_click import click_find_presets_and_exit
from .preset_click import click_list_presets_and_exit
from .preset_click import click_use_preset

# # To debug segmentation fault, uncomment and run with PYTHONFAULTHANDLER=1
# import faulthandler
# faulthandler.enable()


add_to_preset_paths(data_path / "presets")


# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)  # type: ignore


# pylint: disable=W0613  # unused-argument (ctx, param)
def click_set_verbosity(ctx, param, value):
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["verbosity"] = value
    set_log_level(value)


# pylint: disable=W0613  # unused-argument (param)
def click_set_raise(ctx, param, value):
    if ctx.obj is None:
        ctx.obj = {}
    if value is None:
        if "raise" not in ctx.obj:
            ctx.obj["raise"] = False
    else:
        ctx.obj["raise"] = value


def wrap_pdb(fct):
    """Function decorator that drops into ipdb if an exception is raised."""

    def wrapper(*args, **kwargs):
        try:
            return fct(*args, **kwargs)
        except Exception:  # pylint: disable=
            pdb = __import__("ipdb")  # trick pre-commit hook "debug-statements"
            traceback.print_exc()
            click.echo()
            pdb.post_mortem()
            exit(1)

    return wrapper


def wrap_callback(fct):
    """Wrapper for click callback functions to conditionally drop into ipdb."""

    def wrapper(ctx, param, value):
        fct_loc = wrap_pdb(fct) if (ctx.obj or {}).get("pdb") else fct
        return fct_loc(ctx, param, value)

    return wrapper


# pylint: disable=W0613  # unused-argument (param)
def click_set_pdb(ctx, param, value):
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["pdb"] = value
    if value:
        ctx.obj["raise"] = True


# pylint: disable=W0613  # unused-argument (param)
def click_prep_setup_params(ctx, param, value):
    # pylint: disable=W0613  # unused-argument
    if not value:
        return None

    def prepare_params(raw_params: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        value: Any
        for param, value in raw_params:
            if value == "*":
                value = None
            elif "," in value:
                value = value.split(",")
            elif re.match(r"[0-9]+-[0-9]+", value):
                start, end = value.split("-")
                value = range(int(start), int(end) + 1)
            params[param] = value
        return params

    params = prepare_params(value)
    try:
        return Setup.cast_many(params)
    except ValueError as e:
        click_error(ctx, f"Invalid setup parameter ({e})")


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
    "--each-only",
    help=(
        "Only create the first N plots (at most) for each input file. Useful "
        "during development; not supposed to be used in production."
    ),
    type=int,
    metavar="N",
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
    "--open-all", "open_all_cmd", help="Like --open-first, but for all plots.",
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
    "--preset-add",
    help="Add a directory containing preset setup files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    callback=wrap_callback(click_add_to_preset_paths),
    is_eager=True,
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
    "--preset-find",
    help="List preset setup file(s) by name (may contain wildcards).",
    metavar="NAME",
    callback=wrap_callback(click_find_presets_and_exit),
    multiple=True,
    expose_value=False,
)
@click.option(
    "--preset-list",
    help="List the names of all preset setup files.",
    callback=wrap_callback(click_list_presets_and_exit),
    is_flag=True,
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
    "--setup",
    "input_setup_params",
    help="Setup parameter overriding those in the setup file(s).",
    metavar="PARAM VALUE",
    nargs=2,
    multiple=True,
    callback=wrap_callback(click_prep_setup_params),
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
# ---
@click.pass_context
def cli(ctx, **kwargs):
    main_loc = wrap_pdb(main) if ctx.obj["raise"] else main
    main_loc(ctx, **kwargs)


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0913  # too-many-arguments
# pylint: disable=R0915  # too-many-statements
def main(
    ctx,
    setup_file_paths,
    input_setup_params,
    dry_run,
    only,
    each_only,
    open_first_cmd,
    open_all_cmd,
):
    """Create dispersion plot as specified in CONFIG_FILE(S)."""

    # Add preset setup file paths
    setup_file_paths = list(setup_file_paths)
    for path in ctx.obj.get("preset_setup_file_paths", []):
        if path not in setup_file_paths:
            setup_file_paths.append(path)
    if not setup_file_paths:
        click_error(ctx, "Must pass explicit and/or preset setup file(s)")

    # Read setup files
    # Note: Already added argument `each_only` to `read_many` in order to plot
    #       N plots per input setup file instead of per input data file, as it
    #       is done now, but this does not yet work because it would select
    #       n unexpanded setups per setup file, which may well correspond to a
    #       large number of plots, e.g., in case of many time steps!
    #       The setups would either have to be pre-expanded during reading, or
    #       this had to be solved some other way, but for now, let's just stick
    #       with the current implementation of N plots per input data file...
    setups = SetupFile.read_many(setup_file_paths, override=input_setup_params)

    # Group setups by input file(s)
    setups_by_infile = setups.group("infile")

    class BreakInner(Exception):
        """Break out of inner loop, but continue outer loop."""

    class BreakOuter(Exception):
        """Break out of inner and outer loop."""

    # Create plots input file(s) by input file(s)
    out_file_paths = []
    n_in = len(setups_by_infile)
    i_tot = -1
    for ip_in, (in_file_path, sub_setups) in enumerate(setups_by_infile.items(), 1):
        log(vbs=f"[{ip_in}/{n_in}] read {in_file_path}",)
        field_lst_lst = read_fields(
            in_file_path, sub_setups, add_ts0=True, dry_run=dry_run
        )
        n_fld = len(field_lst_lst)
        try:
            for ip_fld, field_lst in enumerate(field_lst_lst, 1):
                i_tot += 1
                log(dbg=f"[{ip_in}/{n_in}][{ip_fld}/{n_fld}] prepare plot")
                in_file_path_fmtd = format_in_file_path(in_file_path, field_lst)
                plot = prepare_plot(field_lst, out_file_paths, dry_run=dry_run)
                log(
                    inf=f"{in_file_path_fmtd} -> {plot.file_path}",
                    vbs=f"[{ip_in}/{n_in}][{ip_fld}/{n_fld}] plot {plot.file_path}",
                )
                if not dry_run:
                    create_plot(plot)

                if open_first_cmd and (ip_in - 1) + (ip_fld - 1) == 0:
                    open_plots(open_first_cmd, [plot.file_path], dry_run)

                n_plt_todo = n_fld - ip_fld
                if n_plt_todo and each_only and ip_fld >= each_only:
                    log(vbs=f"skip remaining {n_plt_todo} plots")
                    raise BreakInner()
                if only and (i_tot + 1) >= only:
                    if n_plt_todo:
                        log(vbs=f"skip remaining {n_plt_todo} plots")
                    raise BreakOuter()
                log(dbg=f"done plotting {plot.file_path}")
            log(dbg=f"done processing {in_file_path}")
        except BreakInner:
            continue
        except BreakOuter:
            remaining_files = n_in - ip_in
            if remaining_files:
                log(vbs=f"skip remaining {remaining_files} input files")
            break

    if open_all_cmd:
        open_plots(open_all_cmd, out_file_paths, dry_run)

    return 0


def format_in_file_path(in_file_path, field_lst):
    def collect_ens_member_ids(field_lst):
        ens_member_ids = None
        for i, field in enumerate(field_lst):
            ens_member_ids_i = field.var_setups.collect_equal("ens_member_id")
            if i == 0:
                ens_member_ids = ens_member_ids_i
            elif ens_member_ids_i != ens_member_ids:
                raise Exception(
                    "ens_member_id differs between fields",
                    ens_member_ids,
                    ens_member_ids_i,
                )
        return ens_member_ids

    ens_member_ids = collect_ens_member_ids(field_lst)
    if ens_member_ids is None:
        return in_file_path
    match = re.match(
        r"(?P<start>.*)(?P<pattern>{ens_member(:(?P<fmt>[0-9]*d))?})(?P<end>.*)",
        in_file_path,
    )
    s_ids = format_range(
        sorted(ens_member_ids), fmt=match.group("fmt"), join_range="..", join_others=","
    )
    return f"{match.group('start')}{{{s_ids}}}{match.group('end')}"


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

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
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

# Third-party
import click
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter

# First-party
from pyflexplot.utils.pydantic import InvalidParameterNameError

# Local
from .. import __version__
from .. import data_path
from ..input import read_fields
from ..plots import create_plot
from ..plots import prepare_plot
from ..setup import Setup
from ..setup import SetupCollection
from ..setup import SetupFile
from ..utils.formatting import format_range
from ..utils.logging import log
from ..utils.logging import set_log_level
from .click import click_error
from .preset import add_to_preset_paths
from .preset_click import click_cat_preset_and_exit
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
        except Exception as e:  # pylint: disable=W0703  # broad-except
            if isinstance(e, click.exceptions.Exit):
                if e.exit_code == 0:  # pylint: disable=E1101  # no-member
                    sys.exit(0)
            pdb = __import__("ipdb")  # trick pre-commit hook "debug-statements"
            traceback.print_exc()
            click.echo()
            pdb.post_mortem()
            sys.exit(1)

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
            if value in ["None", "*"]:
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
    except InvalidParameterNameError as e:
        click_error(ctx, f"Invalid setup parameter name: {e}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]},)
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.argument(
    "setup_file_paths",
    metavar="CONFIG_FILE...",
    type=click.Path(exists=True, readable=True, allow_dash=True),
    nargs=-1,
)
@click.option(
    "--cache/--no-cache",
    help="Cache input fields to avoid reading the same data multiple times.",
    is_flag=True,
    default=True,
)
@click.option(
    "--dry-run",
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
    "--merge-pdfs/--no-merge-pdfs",
    help="Merge PDF plots with the same output file name.",
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
    "--suffix",
    "suffixes",
    help=(
        "Override suffix of output files. May be passed multiple times to"
        " create plots in multiple formats."
    ),
    multiple=True,
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
@click.pass_context
def cli(ctx, **kwargs):
    main_loc = wrap_pdb(main) if ctx.obj["raise"] else main
    main_loc(ctx, **kwargs)


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0913  # too-many-arguments
# pylint: disable=R0915  # too-many-statements
def main(
    ctx,
    *,
    cache,
    dry_run,
    each_only,
    input_setup_params,
    merge_pdfs,
    only,
    open_all_cmd,
    open_first_cmd,
    setup_file_paths,
    suffixes,
):
    """Create dispersion plot as specified in CONFIG_FILE(S)."""

    # Add preset setup file paths
    setup_file_paths = list(setup_file_paths)
    for path in ctx.obj.get("preset_setup_file_paths", []):
        if path not in setup_file_paths:
            setup_file_paths.append(path)
    if not setup_file_paths:
        setups = SetupCollection.from_raw_params([input_setup_params])
    else:
        # Read setup files
        # Note: Already added argument `each_only` to `read_many` in order to
        # plot N plots per input setup file instead of per input data file, as
        # it is done now, but this does not yet work because it would select N
        # unexpanded setups per setup file, which may well correspond to a large
        # number of plots, e.g., in case of many time steps!
        # The setups would either have to be pre-expanded during reading, or
        # this had to be solved some other way, but for now, let's just stick
        # with the current implementation of N plots per input data file...
        setups = SetupFile.read_many(setup_file_paths, override=input_setup_params)
    setups = setups.compress_partially("outfile")

    if suffixes:
        setups.override_output_suffixes(suffixes)

    # Group setups by input file(s)
    setups_by_infile = setups.group("infile")

    class BreakInner(Exception):
        """Break out of inner loop, but continue outer loop."""

    class BreakOuter(Exception):
        """Break out of inner and outer loop."""

    # Create plots input file(s) by input file(s)
    all_out_file_paths = []
    n_in = len(setups_by_infile)
    i_tot = -1
    for ip_in, (in_file_path, sub_setups) in enumerate(setups_by_infile.items(), 1):
        log(vbs=f"[{ip_in}/{n_in}] read {in_file_path}")
        field_lst_lst = read_fields(
            in_file_path, sub_setups, add_ts0=True, dry_run=dry_run, cache_on=cache,
        )
        n_fld = len(field_lst_lst)
        try:
            for ip_fld, field_lst in enumerate(field_lst_lst, 1):
                i_tot += 1
                log(dbg=f"[{ip_in}/{n_in}][{ip_fld}/{n_fld}] prepare plot")
                in_file_path_fmtd = format_in_file_path(
                    in_file_path, [field.var_setups for field in field_lst]
                )
                out_file_paths_i, plot = prepare_plot(
                    field_lst, all_out_file_paths, dry_run=dry_run
                )
                for out_file_path in out_file_paths_i:
                    log(
                        inf=f"{in_file_path_fmtd} -> {out_file_path}",
                        vbs=f"[{ip_in}/{n_in}][{ip_fld}/{n_fld}] plot {out_file_path}",
                    )
                if not dry_run:
                    create_plot(plot, out_file_paths_i)

                if open_first_cmd and (ip_in - 1) + (ip_fld - 1) == 0:
                    open_plots(open_first_cmd, out_file_paths_i[0], dry_run)

                n_plt_todo = n_fld - ip_fld
                if n_plt_todo and each_only and ip_fld >= each_only:
                    log(vbs=f"skip remaining {n_plt_todo} plots")
                    raise BreakInner()
                if only and (i_tot + 1) >= only:
                    if n_plt_todo:
                        log(vbs=f"skip remaining {n_plt_todo} plots")
                    raise BreakOuter()
                for out_file_path in out_file_paths_i:
                    log(dbg=f"done plotting {out_file_path}")
            log(dbg=f"done processing {in_file_path}")
        except BreakInner:
            continue
        except BreakOuter:
            remaining_files = n_in - ip_in
            if remaining_files:
                log(vbs=f"skip remaining {remaining_files} input files")
            break

    if merge_pdfs:
        log(vbs="merge PDF plots")
        all_out_file_paths = merge_pdf_plots(all_out_file_paths, dry_run)

    if open_all_cmd:
        log(vbs=f"open {len(all_out_file_paths)} plots:")
        open_plots(open_all_cmd, all_out_file_paths, dry_run)

    return 0


def format_in_file_path(in_file_path, setups_lst):
    def collect_ens_member_ids(setups_lst):
        ens_member_ids = None
        for i, setups in enumerate(setups_lst):
            ens_member_ids_i = setups.collect_equal("ens_member_id")
            if i == 0:
                ens_member_ids = ens_member_ids_i
            elif ens_member_ids_i != ens_member_ids:
                raise Exception(
                    "ens_member_id differs between setups",
                    ens_member_ids,
                    ens_member_ids_i,
                )
        return ens_member_ids

    ens_member_ids = collect_ens_member_ids(setups_lst)
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


def merge_pdf_plots(paths: Sequence[str], dry_run: bool = False) -> List[str]:
    paths = list(paths)

    # Collect PDFs
    pdf_paths: List[str] = []
    for path in paths:
        if path.endswith(".pdf"):
            pdf_paths.append(path)

    # Group PDFs by shared base name
    grouped_pdf_paths: List[List[str]] = []
    rx_numbered = re.compile(r"\.[0-9]+.pdf$")
    for path in list(pdf_paths):
        if path not in pdf_paths:
            # Already handled
            continue
        if rx_numbered.search(path):
            # Numbered, so not the first
            continue
        path_base = re.sub(r"\.pdf$", "", path)
        pdf_paths.remove(path)
        grouped_pdf_paths.append([path])
        rx_related = re.compile(path_base + r"\.[0-9]+\.pdf")
        for other_path in list(pdf_paths):
            if rx_related.search(other_path):
                pdf_paths.remove(other_path)
                grouped_pdf_paths[-1].append(other_path)
    grouped_pdf_paths = [group for group in grouped_pdf_paths if len(group) > 1]

    def merge_paths(paths: List[str]) -> str:
        unnumbered = False
        idcs: List[int] = []
        rx = re.compile(r"(?P<base>^.*?)(\.(?P<idx>[0-9]+))?.pdf$")
        base: Optional[str] = None
        for path in paths:
            match = rx.match(path)
            if not match:
                raise Exception(f"invalid path: {path}")
            base_i = match.group("base")
            if base is None:
                base = base_i
            elif base_i != base:
                raise Exception(f"different path bases: {base_i} != {base}")
            try:
                idx = int(match.group("idx"))
            except TypeError:
                unnumbered = True
            else:
                idcs.append(idx)
        if len(idcs) == 1:
            s_idcs = str(next(iter(idcs)))
        else:
            s_idcs = (
                f"{{{format_range(sorted(idcs), join_range='..', join_others=',')}}}"
            )
        if unnumbered:
            s_idcs = f"{{,.{s_idcs}}}"
        return f"{base}{s_idcs}.pdf"

    # Merge PDFs with shared base name
    for group in grouped_pdf_paths:
        merged = group[0]
        log(inf=f"{merge_paths(group)} -> {merged}")
        if not dry_run:
            writer = PdfFileWriter()
            for path in group:
                writer.addPage(PdfFileReader(path).getPage(0))
            with open(merged, "wb") as fo:
                writer.write(fo)
        for path in group:
            if path != merged:
                log(dbg=f"remove {path}")
                paths.remove(path)
                if not dry_run:
                    Path(path).unlink()

    return paths


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
    click.echo(cmd.format(file=" \\\n  ".join(file_paths)))
    if not dry_run:
        os.system(cmd.format(file=" ".join(file_paths)))


if __name__ == "__main__":
    sys.exit(1)  # pragma: no cover

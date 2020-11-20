# pylint: disable=R0914  # too-many-locals
"""Command line interface."""
# Standard library
import functools
import multiprocessing
import os
import re
import sys
import time
import traceback
from functools import partial
from os.path import abspath
from os.path import relpath
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from warnings import warn

# Third-party
import click
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter
from PyPDF2.utils import PdfReadError

# First-party
from srutils.str import sorted_paths

# Local
from .. import __version__
from .. import data_path
from ..data import Field
from ..input import read_fields
from ..plots import create_plot
from ..plots import format_out_file_paths
from ..plots import prepare_plot
from ..plotting.boxed_plot import BoxedPlot
from ..setup import Setup
from ..setup import SetupCollection
from ..setup import SetupFile
from ..utils.formatting import format_range
from ..utils.logging import log
from ..utils.logging import set_log_level
from ..utils.pydantic import InvalidParameterNameError
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
    """Decorate a function to drop into ipdb if an exception is raised."""

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
    """Wrapp click callback functions to conditionally drop into ipdb."""

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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.argument(
    "setup_file_paths",
    metavar="CONFIG_FILE...",
    type=click.Path(exists=True, readable=True, allow_dash=True),
    nargs=-1,
)
@click.option(
    "--auto-tmp/--no-auto-tmp",
    help=(
        "Use a temporary directory with an automatically generated name. Overridden by"
        " --tmp=TMP_DIR."
    ),
    default=False,
)
@click.option(
    "--cache/--no-cache",
    help="Cache input fields to avoid reading the same data multiple times.",
    is_flag=True,
    # SR_TMP < TODO fix the input file cache (currently broken)
    # +default=True,
    default=False,
    # SR_TMP >
)
@click.option(
    "--dry-run",
    help="Perform a trial run with no changes made.",
    is_flag=True,
    default=False,
)
@click.option(
    "--dest",
    "dest_dir",
    help=(
        "Directory where the plots are saved to. Defaults to the current directory."
        " Note that this option is incompatible with absolute paths specified in the"
        " setup parameter 'outfile'."
    ),
    metavar="DEST_DIR",
    default=None,
)
@click.option(
    "--merge-pdfs/--no-merge-pdfs",
    help="Merge PDF plots with the same output file name.",
)
@click.option(
    "--merge-pdfs-dry",
    help="Merge PDF plots even in a dry run.",
    is_flag=True,
)
@click.option(
    "--num-procs",
    "-P",
    help=(
        "Number of parallel processes. Note that only the creation of plots for"
        " a given input file (ensemble) is parallelized, while the input files"
        " (or ensembles) themselves are processed sequentially."
    ),
    type=int,
    default=1,
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
    "--open",
    "open_cmd",
    help=(
        "Shell command to all plots. The file paths are appended to the command, unless"
        " explicitly embedded with the format key '{file}', which allows one to use"
        " more complex commands than simple application names (example: 'eog {file}"
        " >/dev/null 2>&1' instead of 'eog' to silence the application 'eog')."
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
    "--keep-merged-pdfs/--remove-merged-pdfs",
    help="Keep individual PDF files after merging them with --merge-pdfs.",
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
    "--show-version/--no-show-version",
    help="Show version number on plot.",
    default=True,
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
    "--tmp",
    "tmp_dir",
    help=(
        "Temporary directory in which the plots are created before being moved to"
        " DEST_DIR in the end."
    ),
    metavar="TMP_DIR",
    default=None,
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


class SharedIterationState:
    """Shared iteration state."""

    def __init__(
        self,
        ip_tot: int = 0,
        n_in: int = -1,
        ip_in: int = 0,
        n_fld: int = -1,
    ) -> None:
        """Create an instance of ``SharedIterationState``."""
        self._dict: Dict[str, Any] = multiprocessing.Manager().dict()
        self.ip_tot = ip_tot
        self.n_in = n_in
        self.ip_in = ip_in
        self.n_fld = n_fld

    @property
    def ip_tot(self) -> int:
        return self._dict["ip_tot"]

    @ip_tot.setter
    def ip_tot(self, value: int) -> None:
        self._dict["ip_tot"] = value

    @property
    def n_in(self) -> int:
        return self._dict["n_in"]

    @n_in.setter
    def n_in(self, value: int) -> None:
        self._dict["n_in"] = value

    @property
    def ip_in(self) -> int:
        return self._dict["ip_in"]

    @ip_in.setter
    def ip_in(self, value: int) -> None:
        self._dict["ip_in"] = value

    @property
    def n_fld(self) -> int:
        return self._dict["n_fld"]

    @n_fld.setter
    def n_fld(self, value: int) -> None:
        self._dict["n_fld"] = value


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0913  # too-many-arguments
# pylint: disable=R0915  # too-many-statements
def main(
    ctx,
    *,
    auto_tmp,
    cache,
    dest_dir,
    dry_run,
    input_setup_params,
    merge_pdfs,
    merge_pdfs_dry,
    num_procs,
    only,
    open_cmd,
    keep_merged_pdfs,
    setup_file_paths,
    show_version,
    suffixes,
    tmp_dir,
):
    """Create dispersion plot as specified in CONFIG_FILE(S)."""
    if dest_dir is None:
        dest_dir = "."
    if tmp_dir is None:
        if auto_tmp:
            tmp_dir = f"tmp-pyflexplot-{int(time.time())}"
        else:
            tmp_dir = dest_dir

    # Check if temporary directory (if given) already exists
    if tmp_dir != "." and os.path.exists(tmp_dir):
        log(dbg="using existing temporary directory '{tmp_dir}'")

    # Add preset setup file paths
    setup_file_paths = list(setup_file_paths)
    for path in ctx.obj.get("preset_setup_file_paths", []):
        if path not in setup_file_paths:
            setup_file_paths.append(path)
    if not setup_file_paths:
        setups = SetupCollection.from_raw_params([input_setup_params])
    else:
        setups = SetupFile.read_many(setup_file_paths, override=input_setup_params)
    setups = setups.compress_partially("outfile")

    if suffixes:
        setups.override_output_suffixes(suffixes)

    # Group setups by input file(s)
    setups_by_infile = setups.group("infile")
    if only and len(setups_by_infile) > only:
        n_old = len(setups_by_infile)
        setups_by_infile, skip = (
            dict(list(setups_by_infile.items())[:only]),
            list(setups_by_infile)[only:],
        )
        s_skip = "\n   ".join([""] + skip)
        log(vbs=f"[only:{only}] skip {len(skip)}/{n_old} infiles:{s_skip}")

    pool = multiprocessing.Pool(processes=num_procs)

    # Create plots input file(s) by input file(s)
    all_out_file_paths: List[str] = []
    istat = SharedIterationState(ip_tot=0, n_in=len(setups_by_infile))
    for istat.ip_in, (in_file_path, sub_setups) in enumerate(
        setups_by_infile.items(), start=1
    ):
        if only and istat.ip_tot >= only:
            continue
        if only and len(sub_setups) > only:
            n_old = len(sub_setups)
            sub_setups = SetupCollection(list(sub_setups)[:only])
            n_skip = n_old - len(sub_setups)
            log(vbs=f"[only:{only}] skip {n_skip}/{n_old} sub-setups")
        log(vbs=f"[{istat.ip_in}/{istat.n_in}] read {in_file_path}")
        field_lst_lst = read_fields(
            in_file_path,
            sub_setups,
            add_ts0=True,
            missing_ok=True,
            dry_run=dry_run,
            cache_on=cache,
        )
        # SR_TMP <  SR_MULTIPANEL
        for field_lst in field_lst_lst:
            if len(field_lst) > 1:
                raise NotImplementedError("multipanel plot")
        field_lst = [next(iter(field_lst)) for field_lst in field_lst_lst]
        # SR_TMP >  SR_MULTIPANEL
        in_file_path_lst = [
            format_in_file_path(in_file_path, field.var_setups) for field in field_lst
        ]
        out_file_paths_lst = [
            format_out_file_paths(
                field,
                prev_paths=all_out_file_paths,
                dest_dir=tmp_dir,
            )
            for field in field_lst
        ]
        istat.n_fld = len(field_lst)
        fct = partial(create_plots, only, dry_run, show_version, istat)
        iter_args = zip(
            range(1, len(field_lst) + 1),
            in_file_path_lst,
            out_file_paths_lst,
            field_lst,
        )
        if num_procs > 1:
            pool.starmap(fct, iter_args)
        else:
            for args in iter_args:
                fct(*args)
        log(dbg=f"done processing {in_file_path}")
        if only and istat.ip_tot >= only:
            remaining_files = istat.n_in - istat.ip_in
            if remaining_files:
                log(vbs=f"[only:{only}] skip remaining {remaining_files} input files")
            break

    # Sort output file paths with numbered duplicates are in the correct order
    # The latter is necessary because parallel execution randomizes their order
    all_out_file_paths = sorted_paths(all_out_file_paths, dup_sep=".")

    if merge_pdfs:
        log(vbs="merge PDF plots")
        pdf_dry_run = dry_run and not merge_pdfs_dry
        iter_max = 1 if abspath(dest_dir) == abspath(tmp_dir) else 10
        for iter_i in range(iter_max):
            all_out_file_paths_tmp = list(all_out_file_paths)
            try:
                pdf_page_paths = merge_pdf_plots(
                    all_out_file_paths_tmp,
                    tmp_dir=tmp_dir,
                    dest_dir=dest_dir,
                    keep_merged=keep_merged_pdfs,
                    dry_run=pdf_dry_run,
                )
            except PdfReadError as e:
                log(err=f"Error merging PDFs ({e}); retry {iter_i + 1}/{iter_max}")
                continue
            else:
                all_out_file_paths = all_out_file_paths_tmp
                for path in pdf_page_paths:
                    if not pdf_dry_run and not keep_merged_pdfs:
                        log(dbg=f"remove {path}")
                        Path(path).unlink()
                break
        else:
            log(err="Could not merge PDFs in {iter_max} attempts")

    # Move output files (remaining after PDF merging) to destination
    if abspath(dest_dir) != abspath(tmp_dir):
        log(inf=f"{tmp_dir} -> {dest_dir}")
        for file_path in all_out_file_paths:
            if not abspath(file_path).startswith(abspath(tmp_dir)):
                continue
            file_path_dest = f"{dest_dir}/{relpath(file_path, start=tmp_dir)}"
            log(vbs=f"{file_path} -> {file_path_dest}")
            if not dry_run:
                os.makedirs(os.path.dirname(file_path_dest), exist_ok=True)
                os.replace(file_path, file_path_dest)

    # Remove temporary directory (if given) unless it already existed before
    remove_tmpdir = tmp_dir and not dry_run and not os.listdir(tmp_dir)
    if remove_tmpdir:
        log(
            dbg=f"recursively removing temporary directory '{tmp_dir}'",
        )
        for dir_path, _, _ in os.walk(tmp_dir, topdown=False):
            log(vbs=f"rmdir '{dir_path}'")
            os.rmdir(dir_path)

    if open_cmd:
        log(vbs=f"open {len(all_out_file_paths)} plots:")
        open_plots(open_cmd, all_out_file_paths, dry_run)

    return 0


def get_pid() -> int:
    name = multiprocessing.current_process().name
    if name == "MainProcess":
        return 0
    elif name.startswith("ForkPoolWorker-"):
        return int(name.split("-")[1])
    else:
        raise NotImplementedError(f"cannot derive pid from process name: {name}")


def create_plots(
    only: Optional[int],
    dry_run: bool,
    show_version: bool,
    istat: SharedIterationState,
    ip_fld: int,
    in_file_path: str,
    out_file_paths: List[str],
    field: Field,
):
    pid = get_pid()
    if only and istat.ip_tot >= only:
        return

    log(
        dbg=(
            f"[P{pid}][{istat.ip_in}/{istat.n_in}][{ip_fld}/{istat.n_fld}]"
            " prepare plot"
        )
    )

    plot = prepare_plot(field, dry_run=dry_run)
    if only:
        only_i = only - istat.ip_tot
        if len(out_file_paths) > only_i:
            n_old = len(out_file_paths)
            out_file_paths, skip = (
                out_file_paths[:only_i],
                out_file_paths[only_i:],
            )
            n_skip = len(skip)
            s_skip = "\n   ".join([""] + skip)
            log(vbs=f"[only:{only}] skip {n_skip}/{n_old} plot files:{s_skip}")
    n_out = len(out_file_paths)
    istat.ip_tot += n_out
    for ip_out, out_file_path in enumerate(out_file_paths, start=1):
        log(
            inf=f"{in_file_path} -> {out_file_path}",
            vbs=(
                f"[P{pid}][{istat.ip_in}/{istat.n_in}][{ip_fld}/{istat.n_fld}]"
                f"[{ip_out}/{n_out}] plot {out_file_path}"
            ),
        )
    if not dry_run:
        assert isinstance(plot, BoxedPlot)  # mypy
        create_plot(plot, out_file_paths, show_version=show_version)
    n_plt_todo = istat.n_fld - ip_fld
    if only and istat.ip_tot >= only:
        if n_plt_todo:
            log(vbs=f"[only:{only}] skip remaining {n_plt_todo} plot fields")
    for out_file_path in out_file_paths:
        log(dbg=f"done plotting {out_file_path}")


def format_in_file_path(in_file_path, setups: SetupCollection) -> str:
    ens_member_ids = setups.collect_equal("ens_member_id")
    if ens_member_ids is None:
        return in_file_path
    pattern = r"(?P<start>.*)(?P<pattern>{ens_member(:(?P<fmt>[0-9]*d))?})(?P<end>.*)"
    match = re.match(pattern, in_file_path)
    if not match:
        raise Exception("file path did not match '{pattern}': {in_file_path}")
    s_ids = format_range(
        sorted(ens_member_ids), fmt=match.group("fmt"), join_range="..", join_others=","
    )
    return f"{match.group('start')}{{{s_ids}}}{match.group('end')}"


def merge_pdf_plots(
    paths: List[str],
    *,
    tmp_dir: str = None,
    dest_dir: str = None,
    keep_merged: bool = True,
    dry_run: bool = False,
) -> List[str]:

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
    grouped_pdf_paths = [
        sorted_paths(group, dup_sep=".")
        for group in grouped_pdf_paths
        if len(group) > 1
    ]

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
    merged_pages: List[str] = []
    for group in grouped_pdf_paths:
        merged = f"{dest_dir}/{relpath(group[0], start=tmp_dir)}"
        if keep_merged and abspath(merged) == abspath(group[0]):
            raise Exception(
                "input and output files are the same file, which is not allowed for"
                f" remove_merged=T: '{merged}'"
                + ("" if merged == group[0] else f" == '{merged[0]}'")
            )
        log(inf=f"{merge_paths(group)} -> {merged}")
        if not dry_run:
            writer = PdfFileWriter()
            for path in group:
                try:
                    file = PdfFileReader(path)
                except (ValueError, TypeError, PdfReadError) as e:
                    # Occur sporadically; likely a file system issue
                    raise PdfReadError(path) from e
                page = file.getPage(0)
                writer.addPage(page)
            dir_path = os.path.dirname(merged)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(merged, "wb") as fo:
                writer.write(fo)
        for path in group:
            paths.remove(path)
            if abspath(path) != abspath(merged):
                merged_pages.append(path)
        paths.append(merged)
    return merged_pages


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

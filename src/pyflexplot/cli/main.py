"""Main module of PyFlexPlot."""
# Standard library
import multiprocessing
import os
import time
from copy import copy
from functools import partial
from os.path import abspath
from os.path import relpath
from pathlib import Path
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import click
from click import Context

# First-party
from srutils.format import ordinal
from srutils.format import sfmt
from srutils.paths import PathsOrganizer
from srutils.pdf import MultiPagePDF
from srutils.pdf import PdfReadError
from srutils.str import sorted_paths

# Local
from ..input.field import FieldGroup
from ..input.read_fields import read_fields
from ..plots import create_plot
from ..plots import format_out_file_paths
from ..setups.plot_setup import PlotSetupGroup
from ..setups.setup_file import prepare_raw_params
from ..setups.setup_file import SetupFile
from ..utils.logging import log


# pylint: disable=R0912  # too-many-branches (>12)
# pylint: disable=R0913  # too-many-arguments (>5)
# pylint: disable=R0914  # too-many-locals (>15)
# pylint: disable=R0915  # too-many-statements (>50)
def main(
    ctx: Context,
    *,
    auto_tmp: bool,
    cache: bool,
    dest_dir: Optional[str],
    dry_run: bool,
    input_setup_params: Tuple[Tuple[str, Any], ...],
    merge_pdfs: bool,
    merge_pdfs_dry: bool,
    num_procs: int,
    only: Optional[int],
    open_cmd: Optional[str],
    keep_merged_pdfs: bool,
    setup_file_paths: Sequence[str],
    show_version: bool,
    suffixes: Collection[str],
    tmp_dir: Optional[str],
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
    if tmp_dir != "." and Path(tmp_dir).exists():
        log(dbg="using existing temporary directory '{tmp_dir}'")

    preset_setup_file_paths = ctx.obj.get("preset_setup_file_paths", [])
    setup_groups = prepare_setups(
        setup_file_paths,
        preset_setup_file_paths,
        dict(input_setup_params),
        suffixes,
        only,
    )

    pool = multiprocessing.Pool(processes=num_procs)

    # Create plots input file(s) by input file(s)
    log(vbs="read fields and create plots")
    all_out_file_paths: List[str] = []
    iter_state = SharedIterationState(n_input_files=len(setup_groups))
    for iter_state.i_input_file, setup_group in enumerate(setup_groups, start=1):
        if only and iter_state.n_field_groups_curr >= only:
            log(
                dbg=(
                    f"[only:{only}] skip {ordinal(iter_state.n_field_groups_curr)}"
                    " field group"
                )
            )
            break
        log(
            vbs=(
                f"[{iter_state.i_input_file + 1}/{iter_state.n_input_files}]"
                f" read {setup_group.infile}"
            )
        )
        # Group fields into one group per plot (with possibly multiple outfiles)
        field_groups = read_fields(
            setup_group,
            config={
                "add_ts0": True,
                "missing_ok": True,
                "dry_run": dry_run,
                "cache_on": cache,
            },
            only=(None if not only else only - iter_state.n_field_groups_curr),
        )
        iter_state.n_field_groups_curr += len(field_groups)
        iter_state.n_field_groups_i = len(field_groups)
        # Format all outfile paths ahead of parallelized plotting loop to ensure
        # correct order of derived paths (e.g., "a.pdf", "a.1.pdf", "a.2.pdf")
        out_file_paths_lst: List[List[str]] = [
            format_out_file_paths(
                field_group,
                prev_paths=all_out_file_paths,
                dest_dir=tmp_dir,
            )
            for field_group in field_groups
        ]
        fct = partial(create_plots, dry_run, show_version, iter_state)
        iter_args = zip(
            range(1, iter_state.n_field_groups_i + 1),
            out_file_paths_lst,
            field_groups,
        )
        if num_procs > 1:
            pool.starmap(fct, iter_args)
        else:
            for args in iter_args:
                fct(*args)
        log(dbg=f"done processing {setup_group.infile}")
        n_input_files_todo = iter_state.n_input_files - iter_state.i_input_file
        if only and iter_state.n_field_groups_curr >= only and n_input_files_todo:
            log(vbs=f"[only:{only}] skip remaining {n_input_files_todo} input files")
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


def prepare_setups(
    setup_file_paths: Sequence[Union[Path, str]],
    preset_setup_file_paths: Sequence[Union[Path, str]],
    input_setup_params: Mapping[str, Any],
    suffixes: Optional[Union[str, Collection[str]]],
    only: Optional[int],
) -> List[PlotSetupGroup]:

    # Extend preset setup file paths
    setup_file_paths = list(setup_file_paths)

    # Read setups from infile(s) and/or preset(s)
    setup_groups = read_setup_groups(
        setup_file_paths, preset_setup_file_paths, input_setup_params
    )

    if suffixes:
        # Replace outfile suffixes by one or more; may increase oufile number
        for setup_group in setup_groups:
            setup_group.override_output_suffixes(suffixes)

    # Combine setups that only differ in outfile
    setup_groups = [
        setup_group.compress_partially("files.output") for setup_group in setup_groups
    ]

    if only:
        # Restrict the total number of setup objects in the setup groups
        # Note that the number of plots is likely still higher than only because
        # each setup can define many plots, but it's a first elimination step
        setup_groups = restrict_grouped_setups(
            setup_groups, only, grouped_by="files.input"
        )

    return setup_groups


def read_setup_groups(
    setup_file_paths: Sequence[Union[Path, str]],
    preset_setup_file_paths: Sequence[Union[Path, str]],
    input_setup_params: Mapping[str, Any],
) -> List[PlotSetupGroup]:
    log(
        vbs=(
            f"reading setups from {len(preset_setup_file_paths)} preset and"
            f" {len(setup_file_paths)} input files"
        )
    )
    setup_file_paths = list(setup_file_paths)
    for path in preset_setup_file_paths:
        if path not in setup_file_paths:
            setup_file_paths.append(path)
    if not setup_file_paths:
        return [PlotSetupGroup.create(prepare_raw_params(input_setup_params))]
    return SetupFile.read_many(setup_file_paths, override=input_setup_params)


def restrict_grouped_setups(
    setup_groups: Sequence[PlotSetupGroup], only: int, *, grouped_by: str = "group"
) -> List[PlotSetupGroup]:
    """Restrict total number of ``Setup``s in ``SetupGroup``s."""
    old_setup_groups = copy(setup_groups)
    new_setup_groups: List[PlotSetupGroup] = []
    n_setups_old = sum(map(len, old_setup_groups))
    n_setups_new = 0
    for setup_group in old_setup_groups:
        if n_setups_new + len(setup_group) > only:
            setup_group = PlotSetupGroup(list(setup_group)[: only - n_setups_new])
            new_setup_groups.append(setup_group)
            n_setups_new += len(setup_group)
            assert n_setups_new == only
            break
        n_setups_new += len(setup_group)
        new_setup_groups.append(setup_group)
    n_groups_old = len(old_setup_groups)
    n_groups_new = len(new_setup_groups)
    if n_setups_new < n_setups_old:
        log(
            dbg=(
                f"[only:{only}] skip {n_setups_old - n_setups_new}/{n_setups_old}"
                f" setups and {n_groups_old - n_groups_new}"
                f"/{n_groups_old} {grouped_by}s"
            )
        )
    return new_setup_groups


def get_pid() -> int:
    name = multiprocessing.current_process().name
    if name == "MainProcess":
        return 0
    elif name.startswith("ForkPoolWorker-"):
        return int(name.split("-")[1])
    else:
        raise NotImplementedError(f"cannot derive pid from process name: {name}")


class SharedIterationState:
    """Shared iteration state in parallel loop."""

    def __init__(
        self,
        *,
        n_plot_files_curr: int = 0,
        n_input_files: int = -1,
        i_input_file: int = 0,
        n_field_groups_curr: int = 0,
        n_field_groups_i: int = -1,
    ) -> None:
        """Create an instance of ``SharedIterationState``."""
        self._dict: Dict[str, Any] = multiprocessing.Manager().dict()
        self.n_plot_files_curr = n_plot_files_curr
        self.n_input_files = n_input_files
        self.i_input_file = i_input_file
        self.n_field_groups_curr = n_field_groups_curr
        self.n_field_groups_i = n_field_groups_i

    @property
    def n_plot_files_curr(self) -> int:
        return self._dict["n_plot_files_curr"]

    @n_plot_files_curr.setter
    def n_plot_files_curr(self, value: int) -> None:
        self._dict["n_plot_files_curr"] = value

    @property
    def n_input_files(self) -> int:
        return self._dict["n_input_files"]

    @n_input_files.setter
    def n_input_files(self, value: int) -> None:
        self._dict["n_input_files"] = value

    @property
    def i_input_file(self) -> int:
        return self._dict["i_input_file"]

    @i_input_file.setter
    def i_input_file(self, value: int) -> None:
        self._dict["i_input_file"] = value

    @property
    def n_field_groups_curr(self) -> int:
        return self._dict["n_field_groups_curr"]

    @n_field_groups_curr.setter
    def n_field_groups_curr(self, value: int) -> None:
        self._dict["n_field_groups_curr"] = value

    @property
    def n_field_groups_i(self) -> int:
        return self._dict["n_field_groups_i"]

    @n_field_groups_i.setter
    def n_field_groups_i(self, value: int) -> None:
        self._dict["n_field_groups_i"] = value

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            + ", ".join([f"{key}={sfmt(value)}" for key, value in self._dict.items()])
            + ")"
        )


# pylint: disable=R0914  # too-many-locals (>15)
def create_plots(
    dry_run: bool,
    show_version: bool,
    iter_state: SharedIterationState,
    ip_fld: int,
    out_file_paths: List[str],
    field_group: FieldGroup,
):
    pid = get_pid()
    log(
        dbg=(
            f"[P{pid}][{iter_state.i_input_file}/{iter_state.n_input_files}][{ip_fld}"
            f"/{iter_state.n_field_groups_i}] prepare plot"
        )
    )
    n_out = len(out_file_paths)
    iter_state.n_plot_files_curr += n_out
    for ip_out, out_file_path in enumerate(out_file_paths, start=1):
        log(
            inf=f"{field_group.attrs.format_path()} -> {out_file_path}",
            vbs=(
                f"[P{pid}][{iter_state.i_input_file}/{iter_state.n_input_files}]"
                f"[{ip_fld}{iter_state.n_field_groups_i}][{ip_out}/{n_out}]"
                f" plot {out_file_path}"
            ),
        )
    if not dry_run:
        create_plot(field_group, out_file_paths, show_version=show_version)
    for out_file_path in out_file_paths:
        log(dbg=f"done plotting {out_file_path}")


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

    paths_organizer = PathsOrganizer(suffix="pdf", dup_sep=".")
    grouped_pdf_paths = paths_organizer.group_related(pdf_paths)
    merged_pages: List[str] = []
    for group in grouped_pdf_paths:
        merged = f"{dest_dir}/{relpath(paths_organizer.merge(group), start=tmp_dir)}"
        if keep_merged and abspath(merged) == abspath(group[0]):
            raise Exception(
                "input and output files are the same file, which is not allowed for"
                f" remove_merged=T: '{merged}'"
                + ("" if merged == group[0] else f" == '{merged[0]}'")
            )
        log(inf=f"{paths_organizer.format_compact(group)} -> {merged}")
        if not dry_run:
            MultiPagePDF.from_files(group).write(merged)
        for path in group:
            paths.remove(path)
            if abspath(path) != abspath(merged):
                merged_pages.append(path)
        paths.append(merged)
    return merged_pages


def open_plots(cmd: str, file_paths: Collection[str], dry_run: bool) -> None:
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

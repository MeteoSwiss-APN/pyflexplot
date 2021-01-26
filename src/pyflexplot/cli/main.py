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
from typing import Union

# Third-party
import click

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
from ..plots import prepare_plot
from ..plotting.boxed_plot import BoxedPlot
from ..setup import SetupFile
from ..setup import SetupGroup
from ..utils.logging import log


# pylint: disable=R0912  # too-many-branches (>12)
# pylint: disable=R0913  # too-many-arguments (>5)
# pylint: disable=R0914  # too-many-locals (>15)
# pylint: disable=R0915  # too-many-statements (>50)
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

    preset_setup_file_paths = ctx.obj.get("preset_setup_file_paths", [])
    setups_by_infile = prepare_setups(
        setup_file_paths, preset_setup_file_paths, input_setup_params, suffixes, only
    )

    pool = multiprocessing.Pool(processes=num_procs)

    # Create plots input file(s) by input file(s)
    log(vbs="read fields and create plots")
    all_out_file_paths: List[str] = []
    iter_state = SharedIterationState(n_input_files=len(setups_by_infile))
    for iter_state.i_input_file, (in_file_path, sub_setups) in enumerate(
        setups_by_infile.items(), start=1
    ):
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
                f" read {in_file_path}"
            )
        )
        field_groups = read_fields(
            in_file_path,
            sub_setups,
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
        out_file_paths_lst: List[str] = [
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
        log(dbg=f"done processing {in_file_path}")
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
) -> Dict[str, SetupGroup]:

    # Extend preset setup file paths
    setup_file_paths = list(setup_file_paths)

    # Read setups from infile(s) and/or preset(s)
    setups = read_setups(setup_file_paths, preset_setup_file_paths, input_setup_params)

    if suffixes:
        setups.override_output_suffixes(suffixes)

    # Separate setups with different outfiles
    setups = setups.compress_partially("outfile")

    # Group setups by input file(s)
    log(dbg=f"grouping {len(setups)} setups by infile")
    setups_by_infile = setups.group("infile")
    if only:
        # Note that the number of plots is likely still higher than only because
        # each setup can define many plots, but it's a first elimination step
        setups_by_infile = restrict_grouped_setups(
            setups_by_infile, only, grouped_by="infile"
        )

    return setups_by_infile


def read_setups(
    setup_file_paths: Sequence[Union[Path, str]],
    preset_setup_file_paths: Sequence[Union[Path, str]],
    input_setup_params: Mapping[str, Any],
) -> SetupGroup:
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
        return SetupGroup.from_raw_params(input_setup_params)
    return SetupFile.read_many(setup_file_paths, override=input_setup_params)


def restrict_grouped_setups(
    grouped_setups: Mapping[str, SetupGroup], only: int, *, grouped_by: str = "group"
) -> Dict[str, SetupGroup]:
    """Restrict total number of ``Setup``s among grouped ``SetupGroup``s."""
    old_grouped_setups = copy(grouped_setups)
    new_grouped_setups = {}
    n_setups_old = sum([len(setups) for setups in old_grouped_setups.values()])
    n_setups_new = 0
    for group_name, setups in old_grouped_setups.items():
        if n_setups_new + len(setups) > only:
            setups = SetupGroup(list(setups)[: only - n_setups_new])
            new_grouped_setups[group_name] = setups
            n_setups_new += len(setups)
            assert n_setups_new == only
            break
        n_setups_new += len(setups)
        new_grouped_setups[group_name] = setups
    n_groups_old = len(old_grouped_setups)
    n_groups_new = len(new_grouped_setups)
    if n_setups_new < n_setups_old:
        log(
            dbg=(
                f"[only:{only}] skip {n_setups_old - n_setups_new}/{n_setups_old}"
                f" setups and {n_groups_old - n_groups_new}"
                f"/{n_groups_old} {grouped_by}s"
            )
        )
    return new_grouped_setups


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
    plot = prepare_plot(field_group, dry_run=dry_run)
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
        assert isinstance(plot, BoxedPlot)  # mypy
        create_plot(plot, out_file_paths, show_version=show_version)
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

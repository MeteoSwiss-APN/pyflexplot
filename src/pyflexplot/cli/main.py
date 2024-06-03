"""Main module of PyFlexPlot."""
# Standard library
import multiprocessing as mp
import os
import shutil
import time
import zipfile
from copy import copy
from functools import partial
from multiprocessing.pool import Pool
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

from pyflexplot.input.field import FieldGroup
from pyflexplot.input.read_fields import read_fields
from pyflexplot.plots import create_plot
from pyflexplot.plots import format_out_file_paths
from pyflexplot.setups.plot_setup import PlotSetupGroup
from pyflexplot.setups.setup_file import SetupFile
from pyflexplot.utils.logging import log
from pyflexplot.s3 import (
    download_key_from_bucket, 
    split_s3_uri,
    expand_key, 
    upload_outpaths_to_s3)
from pyflexplot.config.service_settings import Bucket
from pyflexplot import CONFIG


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
    input_setup_params: Optional[Tuple[Tuple[str, Any], ...]],
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
    if dest_dir.startswith('s3://'):
        s3_dest = dest_dir
        dest_dir = CONFIG.main.local.paths.output
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
        dict(input_setup_params or {}),
        suffixes,
        only,
    )

    fct = partial(
        create_all_plots,
        setup_groups,
        cache,
        dry_run,
        only,
        show_version,
        tmp_dir,
    )

    if num_procs == 1:
        all_out_file_paths = fct()
    else:
        with Pool(processes=num_procs) as pool:
            all_out_file_paths = fct(pool)

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
                log(
                    err=f"""Error merging PDFs ({e});
                retry {iter_i + 1}/{iter_max}"""
                )
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

    # Merge Shape Files (already packed in .zip files each)
    log(vbs="merging shape files")
    all_out_file_paths_tmp = list(all_out_file_paths)
    try:
        redundant_shape_files = merge_shape_files(
            all_out_file_paths_tmp,
            tmp_dir=tmp_dir,
            dest_dir=dest_dir,
            dry_run=pdf_dry_run,
        )
    except FileNotFoundError:
        log(err="Error merging shape files.")
    else:
        for path in redundant_shape_files:
            if not dry_run:
                log(dbg=f"remove {path}")
                Path(path).unlink()

    if s3_dest:
        bucket_name, _, _ = split_s3_uri(s3_dest)
        # Take the bucket region and retries from CONFIG
        # but override the name from CLI input --dest.
        bucket = CONFIG.main.aws.s3.output
        bucket.name = bucket_name
        upload_outpaths_to_s3(
            all_out_file_paths,
            setup_groups[0]._setups[0].model,
            bucket=bucket)

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


def create_all_plots(
    setup_groups: Sequence[PlotSetupGroup],
    cache: bool,
    dry_run: bool,
    only: Optional[int],
    show_version: bool,
    tmp_dir: Optional[str],
    pool: Optional[Pool] = None,
) -> List[str]:
    """Create plots input file(s) by input file(s)."""
    log(vbs="read fields and create plots")
    all_out_file_paths: List[str] = []
    iter_state = SharedIterationState(n_input_files=len(setup_groups))
    for iter_state.i_input_file, setup_group in enumerate(
        setup_groups, start=1
    ):  # noqa: E501
        if only and iter_state.n_field_groups_curr >= only:
            log(
                dbg=(
                    f"""[only:{only}] skip
                    {ordinal(iter_state.n_field_groups_curr)}"""
                    " field group"
                )
            )
            break
        log(
            vbs=(
                f"[{iter_state.i_input_file}/{iter_state.n_input_files}]"
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
        fct = partial(create_plots_i, dry_run, show_version, iter_state)
        iter_args = zip(
            range(1, iter_state.n_field_groups_i + 1),
            out_file_paths_lst,
            field_groups,
        )
        if pool is not None:
            pool.starmap(fct, iter_args)
        else:
            # Don't use pool in sequential run to facilitate debugging
            for args in iter_args:
                fct(*args)
        log(dbg=f"done processing {setup_group.infile}")
        n_input_files_todo = iter_state.n_input_files - iter_state.i_input_file
        maximum_reached = (
            only and iter_state.n_field_groups_curr >= only and n_input_files_todo
        )
        if maximum_reached:
            log(
                vbs=f"""[only:{only}] skip remaining
            {n_input_files_todo} input files"""
            )
            break

    # Sort output file paths with numbered duplicates are in the correct order
    # The latter is necessary because parallel execution randomizes their order
    return sorted_paths(all_out_file_paths, dup_sep=".")


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

    s3_input = False
    bucket: Bucket = CONFIG.main.aws.s3.input
    s3_keys_to_get = set()

    for setup_group in setup_groups:
        if setup_group.infile.startswith("s3://"):
            ens_member_ids = setup_group.ens_member_ids
            s3_input = True
            bucket.name, key, filename = split_s3_uri(setup_group.infile)
            s3_keys_to_get.add(key)
            setup_group.override_s3_infile_location(
                Path(CONFIG.main.local.paths.input), 
                filename)

    if s3_input:
        for key in s3_keys_to_get:
            if '{ens_member:' in key:
                expanded_keys = expand_key(key, ens_member_ids)
                for expanded_key in expanded_keys:
                    download_key_from_bucket(
                        expanded_key,
                        Path(CONFIG.main.local.paths.input),
                        bucket)
            else:
                download_key_from_bucket(
                    key,
                    Path(CONFIG.main.local.paths.input),
                    bucket)

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
        return [
            PlotSetupGroup.create(
                SetupFile.prepare_raw_params(input_setup_params)
            )  # noqa: E501
        ]
    return SetupFile.read_many(setup_file_paths, override=input_setup_params)


# noqa: E501
def restrict_grouped_setups(
    setup_groups: Sequence[PlotSetupGroup],
    only: int,
    *,
    grouped_by: str = "group",  # noqa: E501
) -> List[PlotSetupGroup]:
    """Restrict total number of ``Setup``s in ``SetupGroup``s."""
    old_setup_groups = copy(setup_groups)
    new_setup_groups: List[PlotSetupGroup] = []
    n_setups_old = sum(map(len, old_setup_groups))
    n_setups_new = 0
    for setup_group in old_setup_groups:
        if n_setups_new + len(setup_group) > only:
            setup_group = PlotSetupGroup(
                list(setup_group)[: only - n_setups_new]
            )  # noqa: E501
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
                f"[only:{only}] skip {n_setups_old - n_setups_new}/{n_setups_old}"  # noqa: E501
                f" setups and {n_groups_old - n_groups_new}"
                f"/{n_groups_old} {grouped_by}s"
            )
        )
    return new_setup_groups


def get_pid() -> int:
    name = mp.current_process().name
    if name == "MainProcess":
        return 0
    elif name.startswith("ForkPoolWorker-"):
        return int(name.split("-")[1])
    else:
        raise NotImplementedError(
            f"cannot derive pid from process name: {name}"
        )  # noqa: E501


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
        self._dict: Dict[str, Any] = mp.Manager().dict()  # type: ignore
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
            + ", ".join(
                [f"{key}={sfmt(value)}" for key, value in self._dict.items()]
            )  # noqa: E501
            + ")"
        )


# pylint: disable=R0914  # too-many-locals (>15)
def create_plots_i(
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
            f"""[P{pid}]
            [{iter_state.i_input_file}/{iter_state.n_input_files}]
            [{ip_fld}"""
            f"/{iter_state.n_field_groups_i}] prepare plot"
        )
    )
    n_out = len(out_file_paths)
    iter_state.n_plot_files_curr += n_out
    for ip_out, out_file_path in enumerate(out_file_paths, start=1):
        log(
            inf=f"{field_group.attrs.format_path()} -> {out_file_path}",
            vbs=(
                f"""[P{pid}]
                [{iter_state.i_input_file}/{iter_state.n_input_files}]"""
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
    tmp_dir: Optional[str] = None,
    dest_dir: Optional[str] = None,
    keep_merged: bool = True,
    dry_run: bool = False,
) -> List[str]:
    # Collect PDFs
    pdf_paths: List[str] = [path for path in paths if path.endswith(".pdf")]

    paths_organizer = PathsOrganizer(suffix="pdf", dup_sep=".")
    grouped_pdf_paths = paths_organizer.group_related(pdf_paths)
    merged_pages: List[str] = []
    for group in grouped_pdf_paths:
        merged = f"{dest_dir}/{relpath(paths_organizer.merge(group), start=tmp_dir)}"  # noqa: E501
        if keep_merged and abspath(merged) == abspath(group[0]):
            raise Exception(
                """input and output files are the same file,
                which is not allowed for"""
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


def merge_shape_files(
    paths: List[str],
    *,
    tmp_dir: Optional[str] = None,
    dest_dir: Optional[str] = None,
    dry_run: bool = False,
) -> List[str]:
    # Collect PDFs
    shape_paths: List[str] = [
        f"{path}.zip" for path in paths if path.endswith(".shp")
    ]  # noqa: E501
    paths_organizer = PathsOrganizer(suffix="shp.zip", dup_sep=".")
    grouped_file_paths = paths_organizer.group_related(shape_paths)
    merged_files: List[str] = []
    n = 0
    for i, group in enumerate(grouped_file_paths):
        merged = (
            f"""{dest_dir}/{relpath(paths_organizer.merge(group), start=tmp_dir)}"""
        )
        tmp_zip_name = f"{dest_dir}/temp_shape{i}.zip"
        if not dry_run:
            with zipfile.ZipFile(tmp_zip_name, "w") as main_zip:
                for file in group:
                    with zipfile.ZipFile(file, "r") as zip_to_merge:
                        n += len(zip_to_merge.namelist())
                        for sub_file in zip_to_merge.namelist():
                            main_zip.writestr(
                                sub_file, zip_to_merge.open(sub_file).read()
                            )
            shutil.copy(tmp_zip_name, merged)
            os.remove(tmp_zip_name)
        for path in group:
            base_name = path.rsplit(".", 1)[0]
            paths.remove(base_name)
            if abspath(path) != abspath(merged):
                merged_files.append(path)
        paths.append(merged)
    return merged_files


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

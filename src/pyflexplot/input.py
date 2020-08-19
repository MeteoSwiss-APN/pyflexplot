# -*- coding: utf-8 -*-
"""
Data input.
"""
# Standard library
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import Collection
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type

# Third-party
import netCDF4 as nc4  # types: ignore  # pylance
import numpy as np

# First-party
from srutils.various import check_array_indices

# Local
from .data import ensemble_probability
from .data import EnsembleCloud
from .data import Field
from .data import FieldTimeProperties
from .data import merge_fields
from .fix_nc_input import FlexPartDataFixer
from .meta_data import collect_meta_data
from .meta_data import MetaData
from .meta_data import nc_var_name
from .nc_meta_data import read_meta_data
from .setup import Setup
from .setup import SetupCollection
from .utils.exceptions import MissingCacheEntryError
from .utils.logging import log


# pylint: disable=R0914  # too-many-locals
def read_fields(
    in_file_path: str,
    setups: SetupCollection,
    *,
    add_ts0: bool = False,
    dry_run: bool = False,
    cache_on: bool = False,
) -> List[List[Field]]:
    """Read fields from an input file, or multiple files derived from one path.

    Args:
        in_file_path: Input file path. In case of ensemble data, it must contain
            the format key '{ens_member[:0?d]}', in which case a separate path
            is derived for each member.

        setups: Collection variable setups, containing among other things the
            ensemble member IDs in case of an ensemble simulation.

        add_ts0: Whether to insert an additional time step 0 in the beginning
            with empty fields, given that the first data time step may not
            correspond to the beginning of the simulation, but constitute the
            sum over the first few hours of the simulation.

        dry_run (optional): Whether to skip reading the data from disk.

        cache_on (optional): Whether to activate the cache.

    """
    log(dbg=f"reading fields from {in_file_path}")
    reader = FieldInputOrganizer(
        in_file_path,
        add_ts0=add_ts0,
        dry_run=dry_run,
        cls_fixer=FlexPartDataFixer,
        cache_on=cache_on,
    )
    field_lst_lst = reader.run(setups)
    n_plt = len(field_lst_lst)
    n_tot = sum([len(field_lst) for field_lst in field_lst_lst])
    log(dbg=f"don reading {in_file_path}: read {n_tot} fields for {n_plt} plots")
    return field_lst_lst


# pylint: disable=R0902  # too-many-instance-attributes
class FieldInputOrganizer:
    """Organize input of fields from FLEXPART NetCDF files.

    It represents a single input file for deterministic FLEXPART runs, or an
    ensemble of input files for ensemble FLEXPART runs (one file per ensemble
    member).

    """

    choices_ens_var = ["mean", "median", "min", "max"]

    def __init__(
        self,
        in_file_path: str,
        *,
        add_ts0: bool = False,
        dry_run: bool = False,
        cache_on: bool = False,
        cls_fixer: Optional[Type["FlexPartDataFixer"]] = None,
    ):
        """Create an instance of ``FileReader``.

        Args:
            in_file_path: File path. In case of ensemble data, it must
                contain the format key '{ens_member[:0?d]}'.

            ens_member_ids (optional): Ensemble member ids.

            add_ts0: Whether to insert an additional time step 0 in the
                beginning with empty fields, given that the first data time step
                may not correspond to the beginning of the simulation, but
                constitute the sum over the first few hours of the simulation.

            dry_run (optional): Whether to skip reading the data from disk.

            cls_fixer (optional): Class providing methods to fix issues with the
                input data. Is instatiated with this instance of ``FileReader``.

            cache_on (optional): Activate cache.

        """
        self.in_file_path_fmt = in_file_path
        self.add_ts0 = add_ts0
        self.dry_run = dry_run
        self.cache_on = cache_on
        self.cls_fixer = cls_fixer

    def run(self, setups: SetupCollection) -> List[List[Field]]:
        def prep_in_file_path_lst(
            path_fmt: str, ens_member_ids: Optional[Collection[int]]
        ) -> List[str]:
            path_lst: List[str]
            if re.search(r"{ens_member(:[0-9]+d)?}", path_fmt):
                if not ens_member_ids:
                    raise ValueError(
                        "input file path contains ensemble member format key, but no "
                        "ensemble member ids have been passed",
                        path_fmt,
                        ens_member_ids,
                    )
                assert ens_member_ids is not None  # mypy
                path_lst = [path_fmt.format(ens_member=id_) for id_ in ens_member_ids]
            elif not ens_member_ids:
                path_lst = [path_fmt]
            else:
                raise ValueError(
                    "input file path missing format key", path_fmt, ens_member_ids,
                )
            return path_lst

        def group_setups(setups: SetupCollection) -> List[SetupCollection]:
            """Group the setups by plot type and time step.

            Return a list of setup collections, each of which defines one plot type
            (which may be based on on multiple fields if it has multiple panels) at
            different time steps.

            """
            setups_field_lst: List[SetupCollection] = []
            for (
                (
                    input_variable,
                    combine_levels,
                    combine_deposition_types,
                    combine_species,
                ),
                sub_setups,
            ) in setups.group(
                [
                    "input_variable",
                    "combine_levels",
                    "combine_deposition_types",
                    "combine_species",
                ]
            ).items():
                skip = ["outfile", "dimensions.time", "ens_member_id"]
                if input_variable == "concentration":
                    if combine_levels:
                        skip.append("dimensions.level")
                elif input_variable == "deposition":
                    if combine_deposition_types:
                        skip.append("dimensions.deposition_type")
                if combine_species:
                    skip.append("dimensions.species_id")
                setups_plots = sub_setups.decompress_partially(None, skip=skip)
                # SR_TMP < SR_TODO Adapt for multipanel plots
                setups_plots = [
                    SetupCollection([setup])
                    for setups in setups_plots
                    for setup in setups
                ]
                # SR_TMP >
                setups_field_lst.extend(setups_plots)
            return setups_field_lst

        ens_member_ids = setups.collect_equal("ens_member_id") or None
        in_file_path_lst = prep_in_file_path_lst(self.in_file_path_fmt, ens_member_ids)

        first_in_file_path = next(iter(in_file_path_lst))
        with nc4.Dataset(first_in_file_path) as fi:
            # SR_TMP < Find better solution! Maybe class NcMetaDataReader?
            nc_meta_data = InputFileEnsemble.read_nc_meta_data(
                cast(InputFileEnsemble, self), fi
            )
            # SR_TMP >
        setups = setups.complete_dimensions(nc_meta_data)
        setups_for_plots_over_time = group_setups(setups)

        files = InputFileEnsemble(
            in_file_path_lst,
            ens_member_ids or None,
            add_ts0=self.add_ts0,
            dry_run=self.dry_run,
            cache_on=self.cache_on,
            cls_fixer=self.cls_fixer,
        )
        fields_for_plots: List[List[Field]] = []
        for setups_for_same_plot_over_time in setups_for_plots_over_time:
            fields_for_plots.extend(
                files.read_fields_over_time(setups_for_same_plot_over_time)
            )

        return fields_for_plots


@dataclass
class InputFileEnsemble:
    def __init__(
        self,
        paths: Sequence[str],
        ens_member_ids: Optional[Sequence[int]] = None,
        *,
        add_ts0: bool = False,
        dry_run: bool = False,
        cache_on: bool = False,
        cls_fixer: Optional[Type["FlexPartDataFixer"]] = None,
    ) -> None:
        self.paths = paths
        self.ens_member_ids = ens_member_ids
        self.add_ts0 = add_ts0
        self.dry_run = dry_run
        self.cache_on = cache_on

        if len(ens_member_ids or [0]) != len(paths):
            raise ValueError(
                "must pass same number of ens_member_ids and paths"
                ", or omit ens_member_ids for single path",
                paths,
                ens_member_ids,
            )

        self.cache = FileReaderCache(self)

        self.fixer: Optional["FlexPartDataFixer"] = (
            cls_fixer(self) if cls_fixer else None
        )

        # Declare some attributes
        self.fld_time_mem: np.ndarray
        self.lat: np.ndarray
        self.lon: np.ndarray
        self.mdata_time: np.ndarray
        self.nc_meta_data: Dict[str, Dict[str, Any]]
        self.setups_lst_time: List[SetupCollection]
        self.time: np.ndarray

    # SR_TMP <<<
    # pylint: disable=R0914  # too-many-locals
    def read_fields_over_time(self, setups: SetupCollection) -> List[List[Field]]:
        """Read fields for the same plot at multiple time steps.

        Return a list of lists of fields, whereby:
        - each field corresponds to a panel in a plot;
        - each inner list of fields to a plot -- multipanel in case of multiple
          fields -- at a specific time step; and
        - the outer list to multiple plots of the same type at different time
          steps.

        """

        first_path = next(iter(self.paths))
        with nc4.Dataset(first_path) as fi:
            # SR_TMP < Find better solution! Maybe class NcMetaDataReader?
            self.nc_meta_data = self.read_nc_meta_data(fi)

        # Create individual setups at each requested time step
        self.setups_lst_time = setups.decompress_partially(
            ["dimensions.time"], skip=["outfile"]
        )
        n_requested_times = len(self.setups_lst_time)

        self.fld_time_mem = np.full(self._get_shape_mem_time(), np.nan, np.float32)
        self.mdata_time = np.full(n_requested_times, None, object)
        for idx_mem, (ens_member_id, file_path) in enumerate(
            zip((self.ens_member_ids or [None]), self.paths)  # type: ignore
        ):
            setups_mem = setups.derive({"ens_member_id": ens_member_id})
            timeless_setups_mem = setups_mem.derive({"dimensions": {"time": None}})
            self._read_data(file_path, idx_mem, timeless_setups_mem)

        # Compute single field from all ensemble members
        fld_time: np.ndarray = self._reduce_ensemble(setups)

        # Compute some statistics across all time steps
        time_stats = FieldTimeProperties(fld_time)

        # Create Field objects at requested time steps
        field_lst_lst: List[List[Field]] = []
        for field_setups, mdata in zip(self.setups_lst_time, self.mdata_time):
            time_idx = field_setups.collect_equal("dimensions.time")
            fld: np.ndarray = fld_time[time_idx]
            field = Field(
                fld=fld,
                lat=self.lat,
                lon=self.lon,
                rotated_pole=self.nc_meta_data["derived"]["rotated_pole"],
                var_setups=field_setups,
                time_props=time_stats,
                nc_meta_data=self.nc_meta_data,
                mdata=mdata,
            )
            field_lst_lst.append([field])  # SR_TMP TODO SR_MULTIPANEL

        return field_lst_lst

    def _read_data(
        self, file_path: str, idx_mem: int, timeless_setups_mem: SetupCollection,
    ) -> None:
        n_mem = len(self.paths)

        # Try to fetch the necessary data from cache
        if not self.cache_on:
            cache_key = cast(Hashable, None)  # Prevent "potentially unbound"
        else:
            cache_key = self.cache.create_key(file_path, timeless_setups_mem)
            try:
                self.cache.get(cache_key, idx_mem)
            except MissingCacheEntryError:
                pass
            else:
                log(dbg=f"get from cache ({idx_mem + 1}/{n_mem}): {file_path}")
                return

        # Read the data from disk
        log(dbg=f"reading file ({idx_mem + 1}/{n_mem}): {file_path}")
        with nc4.Dataset(file_path, "r") as fi:
            self._read_grid(fi)

            if idx_mem > 0:
                # Ensure that meta data is the same for all members
                self.nc_meta_data = self.read_nc_meta_data(fi, check=True)

            if not self.dry_run:

                # Read fields for all members at all time steps
                fld_time_i = self._read_member_fields_over_time(fi, timeless_setups_mem)
                self.fld_time_mem[idx_mem][:] = fld_time_i[:]

                # Read meta data at requested time steps
                mdata_time_i = self._collect_meta_data(fi)
                if idx_mem == 0:
                    self.mdata_time[:] = mdata_time_i
                elif mdata_time_i != self.mdata_time.tolist():
                    raise Exception("meta data differ across members")

                if self.cache_on:
                    self.cache.add(cache_key, fld_time_i, mdata_time_i)

    def read_nc_meta_data(self, fi: nc4.Dataset, check: bool = False) -> Dict[str, Any]:
        nc_meta_data = read_meta_data(fi)
        if self.add_ts0:
            old_size = nc_meta_data["dimensions"]["time"]["size"]
            new_size = old_size + 1
            nc_meta_data["dimensions"]["time"]["size"] = new_size
            nc_meta_data["variables"]["time"]["shape"] = (new_size,)
        if check and nc_meta_data != self.nc_meta_data:
            raise Exception("meta data differs", nc_meta_data, self.nc_meta_data)
        return nc_meta_data

    def _read_member_fields_over_time(
        self, fi: nc4.Dataset, timeless_setups: SetupCollection
    ) -> np.ndarray:
        """Read field over all time steps for each member."""

        fld_time_lst = []
        for sub_setups in timeless_setups.decompress_partially(None, skip=["outfile"]):
            for sub_setup in sub_setups:
                fld_time_lst.append(self._read_fld(fi, sub_setup))
        fld_time = merge_fields(fld_time_lst)

        if self.fixer and self.nc_meta_data["derived"]["model"] in ["IFS"]:
            self.fixer.fix_global_grid(self.lon, fld_time)

        if self.add_ts0:
            fld_time = self._add_ts0_to_fld_time(fld_time)

        return fld_time

    # pylint: disable=R0914  # too-many-locals
    def _collect_meta_data(self, fi: nc4.Dataset) -> List[MetaData]:
        """Collect time-step-specific data meta data."""
        mdata_lst: List[MetaData] = []
        for setups in self.setups_lst_time:
            mdata_i_lst: List[MetaData] = []
            for sub_setups in setups.decompress_partially(None, skip=["outfile"]):
                for sub_setup in sub_setups:
                    mdata_ij: MetaData = collect_meta_data(
                        fi, sub_setup, self.nc_meta_data, add_ts0=self.add_ts0
                    )
                    mdata_i_lst.append(mdata_ij)
            mdata_i = mdata_i_lst[0].merge_with(mdata_i_lst[1:])

            # Fix some known issues with the NetCDF input data
            if self.fixer:
                model_name = self.nc_meta_data["derived"]["model"]
                self.fixer.fix_meta_data(model_name, mdata_i)

            mdata_lst.append(mdata_i)
        return mdata_lst

    def _get_shape_mem_time(self) -> Tuple[int, int, int, int]:
        """Get the shape of an array of fields across members and time steps."""
        dim_names = self._dim_names(self.nc_meta_data["derived"]["model"])
        nlat = self.nc_meta_data["dimensions"][dim_names["lat"]]["size"]
        nlon = self.nc_meta_data["dimensions"][dim_names["lon"]]["size"]
        nts = self.nc_meta_data["dimensions"][dim_names["time"]]["size"]
        n_mem = len(self.ens_member_ids) if self.ens_member_ids else 1
        self.lat = np.full((nlat,), np.nan)
        self.lon = np.full((nlon,), np.nan)
        return (n_mem, nts, nlat, nlon)

    def _read_grid(self, fi: nc4.Dataset) -> None:
        """Read and prepare grid variables."""
        dim_names = self._dim_names(self.nc_meta_data["derived"]["model"])
        lat = fi.variables[dim_names["lat"]][:]
        lon = fi.variables[dim_names["lon"]][:]
        time = fi.variables[dim_names["time"]][:]
        time = self._prepare_time(fi, time)
        self.lat = lat
        self.lon = lon
        self.time = time

    def _prepare_time(self, fi: nc4.Dataset, time: np.ndarray) -> np.ndarray:
        if self.add_ts0:
            dts = time[1] - time[0]
            ts0 = time[0] - dts
            time = np.r_[ts0, time]

        # Convert seconds to hours
        dim_names = self._dim_names(self.nc_meta_data["derived"]["model"])
        time_unit = fi.variables[dim_names["time"]].units
        if time_unit.startswith("seconds since"):
            time = time / 3600.0
        else:
            raise NotImplementedError("unexpected time unit", time_unit)

        return time

    @staticmethod
    def _add_ts0_to_fld_time(fld_time: np.ndarray) -> np.ndarray:
        old_shape = fld_time.shape
        new_shape = tuple([old_shape[0] + 1] + list(old_shape[1:]))
        new_fld_time = np.zeros(new_shape, fld_time.dtype)
        new_fld_time[1:] = fld_time
        return new_fld_time

    def _reduce_ensemble(self, var_setups: SetupCollection,) -> np.ndarray:
        """Reduce the ensemble to a single field (time, lat, lon)."""

        fld_time_mem = self.fld_time_mem

        if not self.ens_member_ids or self.dry_run:
            return fld_time_mem[0]

        ens_variable = var_setups.collect_equal("ens_variable")
        ens_param_thr = var_setups.collect_equal("ens_param_thr")
        ens_param_mem_min = var_setups.collect_equal("ens_param_mem_min")
        ens_param_time_win = var_setups.collect_equal("ens_param_time_win")
        n_ens_mem = len(var_setups.collect_equal("ens_member_id"))

        fld_time: np.ndarray
        if ens_variable == "mean":
            fld_time = np.nanmean(fld_time_mem, axis=0)
        elif ens_variable == "median":
            fld_time = np.nanmedian(fld_time_mem, axis=0)
        elif ens_variable == "minimum":
            fld_time = np.nanmin(fld_time_mem, axis=0)
        elif ens_variable == "maximum":
            fld_time = np.nanmax(fld_time_mem, axis=0)
        elif ens_variable == "probability":
            fld_time = ensemble_probability(fld_time_mem, ens_param_thr, n_ens_mem)
        elif ens_variable.startswith("cloud_"):
            cloud = EnsembleCloud(arr=fld_time_mem, time=self.time, thr=ens_param_thr)
            if ens_variable == "cloud_arrival_time":
                fld_time = cloud.arrival_time(ens_param_mem_min)
            elif ens_variable == "cloud_departure_time":
                fld_time = cloud.departure_time(ens_param_mem_min)
            elif ens_variable == "cloud_occurrence_probability":
                fld_time = cloud.occurrence_probability(ens_param_time_win)
            else:
                raise NotImplementedError("ens_variable", ens_variable)
        else:
            raise NotImplementedError("ens_variable", ens_variable)
        return fld_time

    @staticmethod
    def _dim_names(model: str) -> Dict[str, str]:
        """Model-specific dimension names."""
        if model in ["COSMO-2", "COSMO-1", "COSMO-2E", "COSMO-1E"]:
            return {
                "lat": "rlat",
                "lon": "rlon",
                "time": "time",
                "level": "level",
                "nageclass": "nageclass",
                "noutrel": "noutrel",
                "numpoint": "numpoint",
            }
        elif model in ["IFS", "IFS-HRES"]:
            return {
                "lat": "latitude",
                "lon": "longitude",
                "time": "time",
                "level": "height",
                "nageclass": "nageclass",
                "noutrel": "noutrel",
                "numpoint": "pointspec",
            }
        raise NotImplementedError("dimension names for model", model)

    # SR_TODO refactor to reduce branching and locals!
    # pylint: disable=R0912,R0914  # too-many-branches, too-many-locals
    def _read_fld(self, fi: nc4.Dataset, setup: Setup) -> np.ndarray:
        """Read an individual 2D field at a given time step from disk."""

        # SR_TMP <
        dimensions = setup.core.dimensions
        input_variable = setup.core.input_variable
        integrate = setup.core.integrate
        var_name = nc_var_name(setup, self.nc_meta_data["derived"]["model"])
        # SR_TMP >

        # Indices of field along NetCDF dimensions
        dim_names = self._dim_names(self.nc_meta_data["derived"]["model"])
        dim_idcs_by_name = {
            dim_names["lat"]: slice(None),
            dim_names["lon"]: slice(None),
        }
        for dim_name in ["nageclass", "noutrel", "numpoint", "time", "level"]:
            if dim_name in ["lat", "lon", "time"]:
                idcs = slice(None)
            else:
                idcs = getattr(dimensions, dim_name)
            dim_idcs_by_name[dim_names[dim_name]] = idcs

        # Select variable in file
        nc_var = fi.variables[var_name]

        # Assemble indices for slicing
        indices: List[Any] = [None] * len(nc_var.dimensions)
        for dim_name, dim_idx in dim_idcs_by_name.items():
            err_dct = {
                "dim_idx": dim_idx,
                "dim_name": dim_name,
                "fi.filepath": fi.filepath(),
            }
            # Get the index of the dimension for this variable
            try:
                idx = nc_var.dimensions.index(dim_name)
            except ValueError:
                # Potential issue: Dimension is not among the variable dimensions!
                if dim_idx in (None, 0):
                    continue  # Zero-index: We're good after all!
                raise Exception(
                    "dimension with non-zero index missing",
                    {**err_dct, "dimensions": nc_var.dimensions, "var_name": var_name},
                )

            # Check that the index along the dimension is valid
            if dim_idx is None:
                raise Exception("dimension is None", idx, err_dct)

            indices[idx] = dim_idx

        # Check that all variable dimensions have been identified
        try:
            idx = indices.index(None)
        except ValueError:
            pass  # All good!
        else:
            raise Exception(
                "unknown variable dimension",
                nc_var.dimensions[idx],
                (idx, var_name, indices, nc_var.dimensions, dim_idcs_by_name),
            )
        check_array_indices(nc_var.shape, indices)
        fld = nc_var[indices]

        # Fix known issues with NetCDF input data
        model = self.nc_meta_data["derived"]["model"]
        var_ncattrs = {attr: nc_var.getncattr(attr) for attr in nc_var.ncattrs()}
        if self.fixer:
            self.fixer.fix_nc_var_fld(fld, model, var_ncattrs)

        fld = self._handle_time_integration(fi, fld, input_variable, integrate)
        return fld

    def _handle_time_integration(
        self, fi: nc4.Dataset, fld: np.ndarray, input_variable: str, integrate: bool
    ) -> np.ndarray:
        """Integrate or desintegrate the field over time."""

        def get_temp_res_hrs(fi: nc4.Dataset) -> float:
            """Determine the (constant) temporal resolution in hours."""
            time = fi.variables["time"]
            dts = set(time[1:] - time[:-1])
            if len(dts) > 1:
                raise Exception(f"Non-uniform time resolution: {sorted(dts)} ({time})")
            dt_min = next(iter(dts))
            dt_hr = dt_min / 3600.0
            return dt_hr

        if input_variable == "concentration":
            if integrate:
                # Integrate field over time
                dt_hr = get_temp_res_hrs(fi)
                return np.cumsum(fld, axis=0) * dt_hr
            else:
                # Field is already instantaneous
                return fld
        elif input_variable == "deposition":
            if integrate:
                # Field is already time-integrated
                return fld
            else:
                # Revert time integration of field
                dt_hr = get_temp_res_hrs(fi)
                fld[1:] = (fld[1:] - fld[:-1]) / dt_hr
                return fld
        raise NotImplementedError("unknown variable", input_variable)


class FileReaderCache:
    def __init__(self, files: InputFileEnsemble) -> None:
        self.files = files
        self._cache: Dict[Hashable, Any] = {}

    # SR_TMP <<< TODO Derive better key from setups! Make hashable?
    @staticmethod
    def create_key(file_path: str, setups: SetupCollection) -> Hashable:
        params = ["input_variable", "integrate", "dimensions"]
        return tuple([file_path] + [repr(setups.collect(param)) for param in params])

    def add(
        self, cache_key: Hashable, fld_time_i: np.ndarray, mdata_time_i: np.ndarray
    ) -> None:
        self._cache[cache_key] = {
            "grid": deepcopy((self.files.lat, self.files.lon, self.files.time)),
            "nc_meta_data": deepcopy(self.files.nc_meta_data),
            "fld_time_i": deepcopy(fld_time_i),
            "mdata_time_i": deepcopy(mdata_time_i),
        }

    def get(self, key: Hashable, idx_mem: int,) -> None:
        try:
            entry = self._cache[key]
        except KeyError:
            raise MissingCacheEntryError(key)
        entry = deepcopy(entry)
        self.files.lat, self.files.lon, self.files.time = entry["grid"]
        self.files.nc_meta_data = entry["nc_meta_data"]
        self.files.fld_time_mem[idx_mem][:] = entry["fld_time_i"]
        self.files.mdata_time[:] = entry["mdata_time_i"]

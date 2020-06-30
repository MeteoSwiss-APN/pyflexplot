# -*- coding: utf-8 -*-
"""
Data input.
"""
# Standard library
import re
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type

# Third-party
import netCDF4 as nc4
import numpy as np

# First-party
from srutils.various import check_array_indices

# Local
from .data import ensemble_probability
from .data import EnsembleCloud
from .data import Field
from .data import merge_fields
from .fix_nc_input import FlexPartDataFixer
from .meta_data import collect_meta_data
from .meta_data import MetaData
from .meta_data import nc_var_name
from .nc_meta_data import read_meta_data
from .setup import Setup
from .setup import SetupCollection
from .utils.logging import log


# pylint: disable=R0914  # too-many-locals
def read_fields(
    in_file_path: str,
    setups: SetupCollection,
    *,
    add_ts0: bool = False,
    dry_run: bool = False,
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

    """
    log(dbg=f"reading fields from {in_file_path}")
    reader = FileReader(
        in_file_path, add_ts0=add_ts0, dry_run=dry_run, cls_fixer=FlexPartDataFixer,
    )
    field_lst_lst = reader.run(setups)
    n_plt = len(field_lst_lst)
    n_tot = sum([len(field_lst) for field_lst in field_lst_lst])
    log(dbg=f"don reading {in_file_path}: read {n_tot} fields for {n_plt} plots")
    return field_lst_lst


# pylint: disable=R0902  # too-many-instance-attributes
class FileReader:
    """Reader of NetCDF files containing FLEXPART data.

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

        """
        self.in_file_path_fmt = in_file_path
        self.add_ts0 = add_ts0
        self.dry_run = dry_run

        # Declare some attributes
        self.ens_member_ids: Optional[List[int]]
        self.nc_meta_data: Dict[str, Any]
        self.file_path_lst: Sequence[str]
        self.lat: np.ndarray
        self.lon: np.ndarray
        self.time: np.ndarray

        # SR_TMP <
        self._n_members: int = -1
        # SR_TMP >

        self.fixer: Optional["FlexPartDataFixer"] = None
        if cls_fixer is not None:
            self.fixer = cls_fixer(self)

    # pylint: disable=R0914  # too-many-locals
    # pylint: disable=R1702  # too-many-nested-blocks
    def run(self, setups: SetupCollection) -> List[List[Field]]:

        self.ens_member_ids = setups.collect_equal("ens_member_id") or None
        self._n_members = 1 if not self.ens_member_ids else len(self.ens_member_ids)
        self._prepare_in_file_path_lst()

        with nc4.Dataset(next(iter(self.file_path_lst))) as fi:
            self.nc_meta_data = self._read_nc_meta_data(fi)

        setups = setups.complete_dimensions(self.nc_meta_data)

        setups_field_lst_lst: List[List[SetupCollection]] = []
        for (
            (input_variable, combine_levels, combine_deposition_types, combine_species),
            setups_spc,
        ) in setups.group(
            [
                "input_variable",
                "combine_levels",
                "combine_deposition_types",
                "combine_species",
            ]
        ).items():
            setups_loc = setups_spc  # SR_TMP
            skip = ["dimensions.time", "ens_member_id"]
            if input_variable == "concentration":
                if combine_levels:
                    skip.append("dimensions.level")
            elif input_variable == "deposition":
                if combine_deposition_types:
                    skip.append("dimensions.deposition_type")
            if combine_species:
                skip.append("dimensions.species_id")
            setups_field_lst = setups_loc.decompress_partially(None, skip=skip)
            # SR_TMP <
            setups_field_lst = [
                SetupCollection([setup])
                for setups in setups_field_lst
                for setup in setups
            ]
            # SR_TMP >
            setups_field_lst_lst.append(setups_field_lst)

        field_lst_lst: List[List[Field]] = []
        for setups_field_lst in setups_field_lst_lst:
            for setups_field in setups_field_lst:
                # Each list of fields corresponds to one plot
                # Each field therein corresponds to one panel in the plot
                field_lst_lst.extend(self._run_core(setups_field))

        return field_lst_lst

    # SR_TMP <<<
    # pylint: disable=R0914  # too-many-locals
    def _run_core(self, setups: SetupCollection) -> List[List[Field]]:
        """Read one or more fields from a file from disc."""

        # Create individual setups at each requested time step
        setups_lst_time = setups.decompress_partially(["dimensions.time"])
        n_requested_times = len(setups_lst_time)

        shape_mem_time = self._get_shape_mem_time()
        flds_time_mem: np.ndarray = np.full(shape_mem_time, np.nan, np.float32)
        mdata_time: np.ndarray = np.full(n_requested_times, None, object)
        for idx_mem, (ens_member_id, file_path) in enumerate(
            zip((self.ens_member_ids or [None]), self.file_path_lst)  # type: ignore
        ):
            with nc4.Dataset(file_path, "r") as fi:
                setups_mem = setups.derive({"ens_member_id": ens_member_id})
                self._read_grid(fi)

                if idx_mem > 1:
                    # Ensure that meta data is the same for all members
                    self.nc_meta_data = self._read_nc_meta_data(fi, check=True)

                if not self.dry_run:

                    # Read fields for all members at all time steps
                    fld_time_i = self._read_member_fields_over_time(fi, setups_mem)
                    flds_time_mem[idx_mem][:] = fld_time_i[:]

                    # Read meta data at requested time steps
                    mdata_time_i = self._collect_meta_data(fi, setups_lst_time)
                    if idx_mem == 0:
                        mdata_time[:] = mdata_time_i
                    elif mdata_time_i != mdata_time.tolist():
                        raise Exception("meta data differ across members")

        # Compute single field from all ensemble members
        fld_time: np.ndarray
        if self._n_members == 1 or self.dry_run:
            fld_time = flds_time_mem[0]
        else:
            fld_time = self._reduce_ensemble(flds_time_mem, setups)

        # Compute some statistics across all time steps
        time_stats: Dict[str, np.ndarray] = self._collect_time_stats(fld_time)

        # Create Field objects at requested time steps
        field_lst_lst: List[List[Field]] = []
        for field_setups, mdata in zip(setups_lst_time, mdata_time):
            time_idx = field_setups.collect_equal("dimensions.time")
            fld: np.ndarray = fld_time[time_idx]
            field = Field(
                fld=fld,
                lat=self.lat,
                lon=self.lon,
                rotated_pole=self.nc_meta_data["analysis"]["rotated_pole"],
                var_setups=field_setups,
                time_stats=time_stats,
                nc_meta_data=self.nc_meta_data,
                mdata=mdata,
            )
            field_lst_lst.append([field])  # SR_TMP TODO SR_MULTIPANEL

        return field_lst_lst

    def _read_nc_meta_data(
        self, fi: nc4.Dataset, check: bool = False
    ) -> Dict[str, Any]:
        nc_meta_data = read_meta_data(fi)
        if self.add_ts0:
            self._insert_ts0_in_nc_meta_data(nc_meta_data)
        if check and nc_meta_data != self.nc_meta_data:
            raise Exception("meta data differs", nc_meta_data, self.nc_meta_data)
        return nc_meta_data

    def _read_member_fields_over_time(
        self, fi: nc4.Dataset, setups: SetupCollection
    ) -> np.ndarray:
        """Read field over all time steps for each member."""

        timeless_setups = setups.derive({"dimensions": {"time": None}})

        fld_time_lst = []
        for sub_setups in timeless_setups.decompress():
            for sub_setup in sub_setups:
                fld_time_lst.append(self._read_fld(fi, sub_setup))
        fld_time = merge_fields(fld_time_lst)

        if self.fixer and self.nc_meta_data["analysis"]["model"] in ["ifs"]:
            self.fixer.fix_global_grid(self.lon, fld_time)

        if self.add_ts0:
            fld_time = self._add_ts0_to_fld_time(fld_time)

        return fld_time

    # pylint: disable=R0914  # too-many-locals
    def _collect_meta_data(
        self, fi: nc4.Dataset, setups_lst_time: Sequence[SetupCollection]
    ) -> List[MetaData]:
        """Collect time-step-specific data meta data."""
        mdata_lst: List[MetaData] = []
        for setups in setups_lst_time:
            mdata_i_lst = []
            for sub_setups in setups.decompress():
                for sub_setup in sub_setups:
                    mdata_ij = collect_meta_data(
                        fi, sub_setup, self.nc_meta_data, add_ts0=self.add_ts0
                    )
                    mdata_i_lst.append(mdata_ij)
            mdata_i = mdata_i_lst[0].merge_with(mdata_i_lst[1:])

            # Fix some known issues with the NetCDF input data
            if self.fixer:
                model_name = self.nc_meta_data["analysis"]["model"]
                self.fixer.fix_meta_data(model_name, mdata_i)

            mdata_lst.append(mdata_i)
        return mdata_lst

    def _prepare_in_file_path_lst(self) -> None:
        ens_member_ids = self.ens_member_ids
        path_fmt = self.in_file_path_fmt
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
        self.file_path_lst = path_lst

    def _get_shape_mem_time(self) -> Tuple[int, int, int, int]:
        """Get the shape of an array of fields across members and time steps."""
        dim_names = self._dim_names()
        nlat = self.nc_meta_data["dimensions"][dim_names["lat"]]["size"]
        nlon = self.nc_meta_data["dimensions"][dim_names["lon"]]["size"]
        nts = self.nc_meta_data["dimensions"][dim_names["time"]]["size"]
        self.lat = np.full((nlat,), np.nan)
        self.lon = np.full((nlon,), np.nan)
        return (self._n_members, nts, nlat, nlon)

    def _read_grid(self, fi: nc4.Dataset) -> None:
        """Read and prepare grid variables."""
        dim_names = self._dim_names()
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
        dim_names = self._dim_names()
        time_unit = fi.variables[dim_names["time"]].units
        if time_unit.startswith("seconds since"):
            time = time / 3600.0
        else:
            raise NotImplementedError("unexpected time unit", time_unit)

        return time

    def _insert_ts0_in_nc_meta_data(self, nc_meta_data: Dict[str, Any]) -> None:
        old_size = nc_meta_data["dimensions"]["time"]["size"]
        new_size = old_size + 1
        nc_meta_data["dimensions"]["time"]["size"] = new_size
        nc_meta_data["variables"]["time"]["shape"] = (new_size,)

    def _add_ts0_to_time(self, time: np.ndarray):
        delta = time[1] - time[0]
        if time[0] != delta:
            raise ValueError("expecting first time to equal delta", time)
        return np.array([0.0] + time.tolist(), time.dtype)

    def _add_ts0_to_fld_time(self, fld_time: np.ndarray) -> np.ndarray:
        old_shape = fld_time.shape
        new_shape = tuple([old_shape[0] + 1] + list(old_shape[1:]))
        new_fld_time = np.zeros(new_shape, fld_time.dtype)
        new_fld_time[1:] = fld_time
        return new_fld_time

    def _reduce_ensemble(
        self, fld_time_mem: np.ndarray, var_setups: SetupCollection,
    ) -> np.ndarray:
        """Reduce the ensemble to a single field (time, lat, lon)."""

        ens_variable = var_setups.collect_equal("ens_variable")
        ens_param_thr = var_setups.collect_equal("ens_param_thr")
        ens_param_mem_min = var_setups.collect_equal("ens_param_mem_min")
        ens_param_time_win = var_setups.collect_equal("ens_param_time_win")
        n_ens_mem = len(var_setups.collect_equal("ens_member_id"))

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
        return fld_time

    def _collect_time_stats(self, fld_time: np.ndarray,) -> Dict[str, np.ndarray]:
        data = fld_time[np.isfinite(fld_time)]
        data_nz = data[data > 0]
        # Avoid zero-size errors below
        if data.size == 0:
            data = np.full([1], np.nan)
        if data_nz.size == 0:
            data_nz = np.full([1], np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
            )
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Mean of empty slice"
            )
            return {
                "mean": np.nanmean(data),
                "median": np.nanmedian(data),
                "mean_nz": np.nanmean(data_nz),
                "median_nz": np.nanmedian(data_nz),
                "max": np.nanmax(data),
            }

    def _dim_names(self) -> Dict[str, str]:
        """Model-specific dimension names."""
        model = self.nc_meta_data["analysis"]["model"]
        if model.startswith("cosmo"):
            return {
                "lat": "rlat",
                "lon": "rlon",
                "time": "time",
                "level": "level",
                "nageclass": "nageclass",
                "noutrel": "noutrel",
                "numpoint": "numpoint",
            }
        elif model == "ifs":
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
        var_name = nc_var_name(setup, self.nc_meta_data["analysis"]["model"])
        # SR_TMP >

        # Indices of field along NetCDF dimensions
        dim_names = self._dim_names()
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
        model = self.nc_meta_data["analysis"]["model"]
        var_ncattrs = {attr: nc_var.getncattr(attr) for attr in nc_var.ncattrs()}
        if self.fixer:
            self.fixer.fix_nc_var_fld(fld, model, var_ncattrs)

        fld = self._handle_time_integration(fi, fld, input_variable, integrate)
        return fld

    def _handle_time_integration(
        self, fi: nc4.Dataset, fld: np.ndarray, input_variable: str, integrate: bool
    ) -> np.ndarray:
        """Integrate, or desintegrate, field over time."""
        if input_variable == "concentration":
            if integrate:
                # Integrate field over time
                dt_hr = self._compute_temporal_resolution(fi)
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
                dt_hr = self._compute_temporal_resolution(fi)
                fld[1:] = (fld[1:] - fld[:-1]) / dt_hr
                return fld
        raise NotImplementedError("unknown variable", input_variable)

    def _compute_temporal_resolution(self, fi: nc4.Dataset) -> float:
        time = fi.variables["time"]
        dts = set(time[1:] - time[:-1])
        if len(dts) > 1:
            raise Exception(f"Non-uniform time resolution: {sorted(dts)} ({time})")
        dt_min = next(iter(dts))
        dt_hr = dt_min / 3600.0
        return dt_hr

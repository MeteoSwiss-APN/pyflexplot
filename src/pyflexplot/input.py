# -*- coding: utf-8 -*-
"""
IO.
"""
# Standard library
import re
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import netCDF4 as nc4
import numpy as np

# First-party
from srutils.various import check_array_indices

# Local
from .data import EnsembleCloud
from .data import Field
from .data import ensemble_probability
from .data import merge_fields
from .meta_data import MetaData
from .meta_data import collect_meta_data
from .meta_data import get_integr_type
from .meta_data import nc_var_name
from .nc_meta_data import read_meta_data
from .setup import InputSetup
from .setup import InputSetupCollection


# pylint: disable=R0914  # too-many-locals
def read_fields(
    in_file_path: str,
    setups: InputSetupCollection,
    *,
    add_ts0: bool = False,
    dry_run: bool = False,
) -> Tuple[List[List[Field]], Union[List[List[MetaData]], List[List[None]]]]:
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
    field_lst_lst: List[List[Field]] = []
    mdata_lst_lst: List[List[Any]] = []
    ens_mem_ids: List[int]
    setups_ens: InputSetupCollection
    for ens_mem_ids, setups_ens in setups.group("ens_member_id").items():
        reader = FileReader(in_file_path, ens_mem_ids, add_ts0=add_ts0, dry_run=dry_run)
        reader.prepare()
        setups_ens = setups_ens.complete_dimensions(reader.nc_meta_data)

        setups_lst_lst: List[List[InputSetupCollection]] = []
        for combine_species, setups_spec in setups_ens.group("combine_species").items():
            for setup_spec in setups_spec:
                skip = ["ens_member_id"]
                setups_time = setup_spec.decompress_partially(["time"], skip)
                setups_lst_i = []
                for sub_setup in setups_time:
                    setups_lst_i.append(sub_setup.decompress(skip))
                setups_lst_lst.append(setups_lst_i)

        for setups_lst_i in setups_lst_lst:
            field_lst_lst.append([])
            mdata_lst_lst.append([])
            for sub_setups in setups_lst_i:
                field_lst_i, mdata_lst_i = reader.run(sub_setups)
                assert len(field_lst_i) == 1  # SR_TMP  SR_MULTIPANEL
                field = next(iter(field_lst_i))
                mdata = next(iter(mdata_lst_i))
                field_lst_lst[-1].append(field)  # SR_TMP  SR_MULTIPANEL
                mdata_lst_lst[-1].append(mdata)  # SR_TMP  SR_MULTIPANEL

    # breakpoint()
    return field_lst_lst, mdata_lst_lst


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
        ens_member_ids: Optional[List[int]] = None,
        *,
        add_ts0: bool = False,
        dry_run: bool = False,
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

        """
        self.in_file_path_fmt = in_file_path
        self.ens_member_ids = ens_member_ids
        self.add_ts0 = add_ts0
        self.dry_run = dry_run

        # Declare some attributes
        self.prepared: bool = False
        self.nc_meta_data: Dict[str, Any]
        self.in_file_path_lst: Sequence[str]
        self.lat: np.ndarray
        self.lon: np.ndarray
        self.time: np.ndarray

        self.fixer: FlexPartDataFixer = FlexPartDataFixer(self)

    def prepare(self):
        if self.prepared:
            raise Exception("file reader has already been prepared")
        self.prepared = True

        self._prepare_in_file_path_lst()
        self._read_nc_meta_data()

    def run(
        self, var_setups: InputSetupCollection
    ) -> Tuple[List[Field], Union[List[MetaData], List[None]]]:
        """Read one or more fields from a file from disc."""
        if not self.prepared:
            self.prepare()
        return self._create_fields(var_setups)

    # SR_TMP <<< TODO eliminate or implement properly
    @property
    def n_members(self):
        return 1 if not self.ens_member_ids else len(self.ens_member_ids)

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
        self.in_file_path_lst = path_lst

    def _create_fields(
        self, var_setups: InputSetupCollection,
    ) -> Tuple[List[Field], Union[List[MetaData], List[None]]]:
        fld_time: np.ndarray
        if self.dry_run:
            fld_time = self._create_dummy_fld()
        else:
            fld_time_mem = self._read_fld_time_mem(var_setups)
            fld_time = self._reduce_ensemble(fld_time_mem, var_setups)
        time_stats: Dict[str, np.ndarray] = self._collect_time_stats(fld_time)
        field_lst: List[Field] = self._create_field_objs(
            var_setups, fld_time, time_stats
        )
        mdata_lst: Union[List[MetaData], List[None]]
        if self.dry_run:
            mdata_lst = [None] * len(field_lst)
        else:
            mdata_lst = self._collect_meta_data(var_setups)
        return field_lst, mdata_lst

    def _read_nc_meta_data(self):
        nc_meta_data: Dict[str, Any]
        for i_mem, in_file_path in enumerate(self.in_file_path_lst or []):
            with nc4.Dataset(in_file_path, "r") as fi:
                nc_meta_data_i = read_meta_data(fi)
                if i_mem == 0:
                    nc_meta_data = nc_meta_data_i
                elif nc_meta_data_i != nc_meta_data:
                    raise Exception(
                        "meta data differs", nc_meta_data_i, nc_meta_data,
                    )
        if self.add_ts0:
            self._insert_ts0_in_nc_meta_data(nc_meta_data)
        self.nc_meta_data = nc_meta_data

    def _create_dummy_fld(self):
        dim_names = self._dim_names()
        nlat = self.nc_meta_data["dimensions"][dim_names["lat"]]["size"]
        nlon = self.nc_meta_data["dimensions"][dim_names["lon"]]["size"]
        nts = self.nc_meta_data["dimensions"][dim_names["time"]]["size"]
        self.lat = np.full((nlat,), np.nan)
        self.lon = np.full((nlon,), np.nan)
        fld_time = np.full((nts, nlat, nlon), np.nan)
        return fld_time

    def _read_fld_time_mem(self, setups: InputSetupCollection) -> np.ndarray:
        """Read field over all time steps for each member."""
        fld_time_mem: Optional[np.ndarray] = None
        for i_mem, in_file_path in enumerate(self.in_file_path_lst or []):
            with nc4.Dataset(in_file_path, "r") as fi:

                # Read grid variables
                self._read_grid(fi)

                # Read field (all time steps)
                fld_time = self._read_fld_time(fi, setups)
                if self.add_ts0:
                    fld_time = self._add_ts0_to_fld_time(fld_time)

                # Store field for current member
                if fld_time_mem is None:
                    shape = tuple([self.n_members] + list(fld_time.shape))
                    fld_time_mem = np.full(shape, np.nan, np.float32)
                fld_time_mem[i_mem] = fld_time

        assert fld_time_mem is not None  # mypy
        return fld_time_mem

    def _read_grid(self, fi):
        dim_names = self._dim_names()
        lat = fi.variables[dim_names["lat"]][:]
        lon = fi.variables[dim_names["lon"]][:]
        time = fi.variables[dim_names["time"]][:]
        # SR_TMP <
        time_unit = fi.variables[dim_names["time"]].units
        if time_unit.startswith("seconds since"):
            time = time / 3600.0
        else:
            raise NotImplementedError("unexpected time unit", time_unit)
        # SR_TMP >
        if self.add_ts0:
            time = self._add_ts0_to_time(time)
        try:
            old_lat = self.lat
            old_lon = self.lon
            old_time = self.time
        except AttributeError:
            self.lat = lat
            self.lon = lon
            self.time = time
        else:
            if not (lat == old_lat).all():
                raise Exception("inconsistent latitude", lat, old_lat)
            if not (lon == old_lon).all():
                raise Exception("inconsistent longitude", lon, old_lon)
            if not (time == old_time).all():
                raise Exception("inconsistent time", time, old_time)

    def _read_fld_time(self, fi, setups):
        """Read field at all time steps."""

        expand = ["lat", "lon", "time"]
        flds_time = [self._read_nc_var(fi, setup, expand) for setup in setups]
        fld_time: np.ndarray = merge_fields(flds_time)

        model = self.nc_meta_data["analysis"]["model"]
        if model in ["ifs"]:
            self.fixer.fix_global_grid(self.lon, fld_time)

        return fld_time

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
        self, fld_time_mem: np.ndarray, var_setups: InputSetupCollection,
    ) -> np.ndarray:
        """Reduce the ensemble to a single field (time, lat, lon)."""

        if self.n_members == 1:
            return fld_time_mem[0]

        plot_type = var_setups.collect_equal("plot_type")
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
            raise NotImplementedError("plot var", plot_type)
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

    # pylint: disable=R0914  # too-many-locals
    def _collect_meta_data(self, var_setups: InputSetupCollection,) -> List[MetaData]:
        """Collect time-step-specific data meta data."""

        # Collect meta data at requested time steps for all members
        n_ts = len(var_setups.collect_equal("time"))
        shape = (n_ts, self.n_members)
        mdata_by_reqtime_mem: np.ndarray = np.full(shape, None)
        for idx_mem, in_file_path in enumerate(self.in_file_path_lst or []):
            with nc4.Dataset(in_file_path, "r") as fi:
                for idx_time, sub_setups in enumerate(
                    var_setups.decompress_partially(["time"])
                ):
                    mdata_lst_i = []
                    for sub_setup in sub_setups:
                        mdata_lst_i.append(
                            collect_meta_data(
                                fi, sub_setup, self.nc_meta_data, add_ts0=self.add_ts0
                            )
                        )
                    mdata = mdata_lst_i[0].merge_with(mdata_lst_i[1:])
                    mdata_by_reqtime_mem[idx_time, idx_mem] = mdata

        # Merge meta data across members
        for i_mem in range(1, self.n_members or 1):
            for i_reqtime, mdata in enumerate(mdata_by_reqtime_mem[:, i_mem]):
                mdata_ref = mdata_by_reqtime_mem[i_reqtime, 0]
                if mdata != mdata_ref:
                    raise Exception(
                        f"meta data differ between members 0 and {i_mem}",
                        mdata_ref,
                        mdata,
                    )
        mdata_lst: List[MetaData] = mdata_by_reqtime_mem[:, 0].tolist()

        # Fix some known issues with the NetCDF input data
        self.fixer.fix_meta_data(self.nc_meta_data["analysis"]["model"], mdata_lst)

        return mdata_lst

    def _create_field_objs(
        self,
        var_setups: InputSetupCollection,
        fld_time: np.ndarray,
        time_stats: Mapping[str, np.ndarray],
    ) -> List[Field]:
        """Create fields at requested time steps for all members."""
        rotated_pole = self.nc_meta_data["analysis"]["rotated_pole"]
        time_idcs = var_setups.collect_equal("time")
        field_lst: List[Field] = []
        for time_idx in time_idcs:
            fld: np.ndarray = fld_time[time_idx]
            field = Field(
                fld=fld,
                lat=self.lat,
                lon=self.lon,
                rotated_pole=rotated_pole,
                var_setups=var_setups,
                time_stats=time_stats,
                nc_meta_data=self.nc_meta_data,
            )
            field_lst.append(field)  # SR_TMP SR_MULTIPANEL
        return field_lst

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
    def _read_nc_var(
        self, fi: nc4.Dataset, setup: InputSetup, expand: Optional[List[str]] = None,
    ) -> np.ndarray:
        if expand is None:
            expand = []

        model = self.nc_meta_data["analysis"]["model"]

        # Get dimension names (model-specific)
        dim_names = self._dim_names()

        # Select variable in file
        var_name = nc_var_name(setup, model)
        nc_var = fi.variables[var_name]

        if setup.level is None:
            level = None
        else:
            assert len(setup.level) == 1
            level = next(iter(setup.level))

        # Indices of field along NetCDF dimensions
        dim_idcs_by_name = {
            dim_names["time"]: slice(None) if "time" in expand else level,
            dim_names["level"]: slice(None) if "level" in expand else level,
            dim_names["lat"]: slice(None),
            dim_names["lon"]: slice(None),
        }

        # SR_TMP <
        for dim_name in ["nageclass", "noutrel", "numpoint"]:
            if dim_name in expand:
                idcs = slice(None)
            else:
                idcs = getattr(setup, dim_name)
                if idcs is not None:
                    # SR_TMP <
                    assert isinstance(idcs, Sequence)  # mypy
                    assert len(idcs) == 1, f"len(setup.{dim_name}) > 1: {idcs}"
                    idcs = next(iter(idcs))
                    # SR_TMP >
            dim_idcs_by_name[dim_names[dim_name]] = idcs
        # SR_TMP >

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
        self.fixer.fix_nc_var(nc_var, fld, model)

        fld = self._handle_time_integration(fi, fld, setup)
        return fld

    def _handle_time_integration(
        self, fi: nc4.Dataset, fld: np.ndarray, setup: InputSetup,
    ) -> np.ndarray:
        """Integrate, or desintegrate, field over time."""
        if setup.input_variable == "concentration":
            if setup.integrate:
                # Integrate field over time
                dt_hr = self._compute_temporal_resolution(fi)
                return np.cumsum(fld, axis=0) * dt_hr
            else:
                # Field is already instantaneous
                return fld
        elif setup.input_variable == "deposition":
            if setup.integrate:
                # Field is already time-integrated
                return fld
            else:
                # Revert time integration of field
                dt_hr = self._compute_temporal_resolution(fi)
                fld[1:] = (fld[1:] - fld[:-1]) / dt_hr
                return fld
        raise NotImplementedError("unknown variable", setup.input_variable)

    def _compute_temporal_resolution(self, fi: nc4.Dataset) -> float:
        time = fi.variables["time"]
        dts = set(time[1:] - time[:-1])
        if len(dts) > 1:
            raise Exception(f"Non-uniform time resolution: {sorted(dts)} ({time})")
        dt_min = next(iter(dts))
        dt_hr = dt_min / 3600.0
        return dt_hr


class FlexPartDataFixer:
    """Fix issues with FlexPart NetCDF output."""

    def __init__(self, file_reader):
        self.file_reader = file_reader

    possible_var_names: List[Union[str, Tuple[str, ...]]] = [
        "Cs-137",
        "I-131a",
        ("Cs-137", "I-131a"),
        ("I-131a", "Cs-137"),
    ]
    conversion_factor_by_unit: Dict[str, float] = {
        "ng kg-1": 1.0e-12,
        "1e-12 kg m-2": 1.0e-12,
    }

    def fix_nc_var(self, nc_var: nc4.Variable, fld: np.ndarray, model: str) -> None:
        if model in ["cosmo2", "cosmo1"]:
            self._fix_nc_var_cosmo(nc_var, fld)
        elif model == "ifs":
            pass
        else:
            raise NotImplementedError("model", model)

    def _fix_nc_var_cosmo(self, nc_var, fld):
        name = nc_var.getncattr("long_name").split("_")[0]
        unit = nc_var.getncattr("units")
        if name not in self.possible_var_names:
            raise NotImplementedError("input_variable", name)
        try:
            fact = self.conversion_factor_by_unit[unit]
        except KeyError:
            raise NotImplementedError("conversion factor", name, unit)
        fld[:] *= fact

    def fix_meta_data(
        self, model: str, mdata: Union[MetaData, Sequence[MetaData]],
    ) -> None:
        if model in ["cosmo2", "cosmo1"]:
            self._fix_meta_data_cosmo(mdata)
        elif model == "ifs":
            pass
        else:
            raise NotImplementedError("model", model)

    def _fix_meta_data_cosmo(self, mdata):
        if isinstance(mdata, Sequence):
            for mdata_i in mdata:
                self._fix_meta_data_cosmo(mdata_i)
            return
        assert isinstance(mdata, MetaData)  # mypy

        # Variable unit
        var_name = mdata.species_name.value
        if var_name not in self.possible_var_names:
            raise NotImplementedError("input_variable", var_name)
        integr_type = get_integr_type(mdata.setup)
        old_unit = mdata.variable_unit.value
        new_unit = "Bq"
        if integr_type == "mean":
            pass
        elif integr_type in ["sum", "accum"]:
            new_unit += " h"
        else:
            raise NotImplementedError(
                "integration type for variable", integr_type, var_name
            )
        if old_unit == "ng kg-1":
            new_unit += " m-3"
        elif old_unit == "1e-12 kg m-2":
            new_unit += " m-2"
        else:
            raise NotImplementedError("unit for variable", old_unit, var_name)
        mdata.variable_unit.value = new_unit

    def fix_global_grid(self, lon, fld_time, idx_lon=-1):
        """Shift global grid longitudinally to fit into (-180..180) range."""

        # Check longitude dimension index
        if (
            (idx_lon < 0 and -idx_lon > len(fld_time.shape))
            or (idx_lon >= 0 and idx_lon >= len(fld_time.shape))
            or fld_time.shape[idx_lon] != lon.size
        ):
            raise ValueError("invalid idx_lon", idx_lon, fld_time.shape, lon.size)

        # Check longitudinal range
        if lon.max() - lon.min() > 360.0:
            raise ValueError("longitutinal range too large", lon.max() - lon.min())

        # Check that longitude is evenly spaced and seamless across date line
        dlons_raw = np.r_[lon, lon[0]] - np.r_[lon[-1], lon]
        dlons = np.abs(np.stack([dlons_raw, 360 - np.abs(dlons_raw)])).min(axis=0)
        if np.unique(dlons).size > 1:
            raise ValueError("longitude not evenly spaced/seamless", np.unique(dlons))
        dlon = next(iter(dlons))

        # Shift the grid
        if lon[-1] > 180.0:
            # Eastward shift
            n_shift = 0
            while lon[-1] > 180.0:
                n_shift += 1
                lon[:] = np.r_[lon[0] - dlon, lon[:-1]]
                if lon[0] < -180.0 or n_shift >= lon.size:
                    raise Exception(
                        "unexpected error while shifting lon eastward", lon, n_shift,
                    )
                idcs = np.arange(fld_time.shape[idx_lon] - 1)
                fld_time[:] = np.concatenate(
                    [
                        np.take(fld_time, [-1], idx_lon),
                        np.take(fld_time, idcs, idx_lon),
                    ],
                    axis=idx_lon,
                )
            warnings.warn(f"fix global data: shift eastward by {n_shift} * {dlon} deg")
            return

        elif lon[0] < -180.0:
            # Westward shift
            n_shift = 0
            while lon[0] < -180.0:
                n_shift += 1
                lon[:] = np.r_[lon[1:], lon[-1] + dlon]
                print(n_shift, lon)
                if lon[-1] < -180.0 or n_shift >= lon.size:
                    raise Exception(
                        "unexpected error while shifting lon eastward", lon, n_shift,
                    )
                idcs = np.arange(1, fld_time.shape[idx_lon])
                fld_time[:] = np.concatenate(
                    [
                        np.take(fld_time, idcs, idx_lon),
                        np.take(fld_time, [0], idx_lon),
                    ],
                    axis=idx_lon,
                )
            warnings.warn(f"fix global data: shift westward by {n_shift} * {dlon} deg")
            return
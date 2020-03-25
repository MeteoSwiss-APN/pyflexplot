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
from .data import Field
from .data import cloud_arrival_time
from .data import merge_fields
from .data import threshold_agreement
from .io_meta_data import read_meta_data
from .meta_data import MetaDataCollection
from .meta_data import collect_meta_data
from .meta_data import nc_var_name
from .setup import InputSetup
from .setup import InputSetupCollection
from .specs import FldSpecs


def read_files(in_file_path, fld_specs_lst):
    return FileReader(in_file_path).run(fld_specs_lst)


class FileReader:
    """Reader of NetCDF files containing FLEXPART data.

    It represents a single input file for deterministic FLEXPART runs, or an
    ensemble of input files for ensemble FLEXPART runs (one file per ensemble
    member).

    """

    choices_ens_var = ["mean", "median", "min", "max"]

    def __init__(self, in_file_path: str):
        """Create an instance of ``FileReader``.

        Args:
            in_file_path: File path. In case of ensemble data, it must
                contain the format key '{ens_member[:0?d]}'.

        """
        self.in_file_path_fmt = in_file_path

        self.n_members: Optional[int] = None
        self.in_file_path_lst: Optional[Sequence[str]] = None
        self.lat: Optional[np.ndarray] = None
        self.lon: Optional[np.ndarray] = None

        self.rotated_pole: bool

        self.fixer: FlexPartDataFixer = FlexPartDataFixer(self)

    def run(
        self, fld_specs_lst: Sequence[FldSpecs]
    ) -> Tuple[List[Field], List[MetaDataCollection]]:
        """Read one or more fields from a file from disc.

        Args:
            fld_specs_lst: List of field specifications.

        """
        # Collect ensemble member ids
        ens_member_ids = self._collect_ens_member_ids(fld_specs_lst)
        self.n_members = 1 if not ens_member_ids else len(ens_member_ids)

        # Create input file paths
        self.in_file_path_lst = self._prepare_in_file_path_lst(ens_member_ids)

        # Collect input file meta data
        nc_meta_data = self._read_nc_meta_data()
        for fld_specs in fld_specs_lst:
            fld_specs.var_setups.replace_nones(nc_meta_data, decompress_skip=["time"])

        # Collect fields and attrs
        fields: List[Field] = []
        mdata_lst: List[MetaDataCollection] = []
        for fld_specs in fld_specs_lst:
            fields_i, mdata_lst_i = self._create_fields(fld_specs, nc_meta_data)
            fields.extend(fields_i)
            mdata_lst.extend(mdata_lst_i)

        return fields, mdata_lst

    def _collect_ens_member_ids(
        self, fld_specs_lst: Sequence[FldSpecs]
    ) -> Optional[List[int]]:
        """Collect the ensemble member ids from field specifications."""
        ens_member_ids: Optional[List[int]] = None
        for fld_specs in fld_specs_lst:
            ens_member_ids_i = fld_specs.collect_equal("ens_member_id")
            if not ens_member_ids:
                ens_member_ids = ens_member_ids_i
            else:
                # Should be the same for all!
                assert ens_member_ids_i == ens_member_ids
        return ens_member_ids

    def _prepare_in_file_path_lst(self, ens_member_ids: Sequence[int]) -> List[str]:
        path_fmt = self.in_file_path_fmt
        if re.search(r"{ens_member(:[0-9]+d)?}", path_fmt):
            if not ens_member_ids:
                raise ValueError(
                    f"input file path contains ensemble member format key, but no "
                    f"ensemble member ids have been passed",
                    path_fmt,
                    ens_member_ids,
                )
            assert ens_member_ids is not None  # mypy
            return [path_fmt.format(ens_member=id_) for id_ in ens_member_ids]
        elif not ens_member_ids:
            return [path_fmt]
        else:
            raise ValueError(
                f"input file path missing format key", path_fmt, ens_member_ids,
            )

    def _create_fields(
        self, fld_specs: FldSpecs, nc_meta_data: Mapping["str", Any],
    ) -> Tuple[List[Field], List[MetaDataCollection]]:

        # Read fields of all members at all time steps
        fld_time_mem: np.ndarray = self._read_fld_time_mem(
            fld_specs.var_setups, nc_meta_data,
        )

        # Reduce fields array along member dimension
        # In other words: Compute single field from ensemble
        fld_time: np.ndarray = self._reduce_ensemble(fld_time_mem, fld_specs.fld_setup)

        # Collect time stats
        time_stats: Dict[str, np.ndarray] = {
            "mean": np.nanmean(fld_time),
            "median": np.nanmedian(fld_time),
            "mean_nz": np.nanmean(fld_time[fld_time > 0]),
            "median_nz": np.nanmedian(fld_time[fld_time > 0]),
            "max": np.nanmax(fld_time),
        }

        # Collect time-step-specific data meta data
        mdata_lst: List[MetaDataCollection] = self._collect_meta_data(
            fld_specs, nc_meta_data,
        )

        # Create fields at requested time steps for all members
        fields: List[Field] = self._create_field_objs(
            fld_specs, fld_time, time_stats, nc_meta_data,
        )

        return fields, mdata_lst

    def _read_nc_meta_data(self) -> Dict[str, Any]:
        nc_meta_data = None
        for i_mem, in_file_path in enumerate(self.in_file_path_lst or []):
            with nc4.Dataset(in_file_path, "r") as fi:
                if nc_meta_data is not None:
                    nc_meta_data_i = read_meta_data(fi)
                    if nc_meta_data_i != nc_meta_data:
                        raise Exception(
                            f"meta data differs", nc_meta_data_i, nc_meta_data,
                        )
                else:
                    nc_meta_data = read_meta_data(fi)
        assert nc_meta_data is not None  # mypy
        return nc_meta_data

    def _read_fld_time_mem(
        self, setups: InputSetupCollection, nc_meta_data: Mapping[str, Any],
    ) -> np.ndarray:
        """Read field over all time steps for each member."""
        fld_time_mem: Optional[np.ndarray] = None
        for i_mem, in_file_path in enumerate(self.in_file_path_lst or []):
            with nc4.Dataset(in_file_path, "r") as fi:

                # Read grid variables
                self.lat, self.lon = self._read_grid(fi, nc_meta_data)

                # Read field (all time steps)
                fld_time = self._read_fld_time(fi, setups, nc_meta_data)

                # Store field for current member
                if fld_time_mem is None:
                    shape = [self.n_members] + list(fld_time.shape)
                    fld_time_mem = np.full(shape, np.nan, np.float32)
                fld_time_mem[i_mem] = fld_time

        assert fld_time_mem is not None  # mypy
        return fld_time_mem

    def _read_grid(self, fi, nc_meta_data):
        dim_names = self._dim_names(nc_meta_data)
        lat = fi.variables[dim_names["lat"]][:]
        lon = fi.variables[dim_names["lon"]][:]

        # Ensure consistent grid across all input files
        if self.lat is not None:
            if not (lat == self.lat).all():
                raise Exception("inconsistent latitude", lat, self.lat)
            if not (lon == self.lon).all():
                raise Exception("inconsistent longitude", lon, self.lon)

        return lat, lon

    def _read_fld_time(self, fi, setups, nc_meta_data):
        """Read field at all time steps."""

        expand = ["lat", "lon", "time"]
        flds_time = [
            self._read_nc_var(fi, setup, nc_meta_data, expand) for setup in setups
        ]
        fld_time: np.ndarray = merge_fields(flds_time)

        if nc_meta_data["analysis"]["model"] in ["ifs"]:
            self.fixer.fix_global_grid(self.lon, fld_time)

        return fld_time

    def _reduce_ensemble(
        self, fld_time_mem: np.ndarray, setup: InputSetup,
    ) -> np.ndarray:
        """Reduce the ensemble to a single field (time, lat, lon)."""
        if self.n_members == 1:
            return fld_time_mem[0]
        plot_type = setup.plot_type
        if plot_type == "ens_mean":
            fld_time = np.nanmean(fld_time_mem, axis=0)
        elif plot_type == "ens_median":
            fld_time = np.nanmedian(fld_time_mem, axis=0)
        elif plot_type == "ens_min":
            fld_time = np.nanmin(fld_time_mem, axis=0)
        elif plot_type == "ens_max":
            fld_time = np.nanmax(fld_time_mem, axis=0)
        elif plot_type == "ens_thr_agrmt":
            fld_time = threshold_agreement(fld_time_mem, setup.ens_param_thr, axis=0)
        elif plot_type == "ens_cloud_arrival_time":
            fld_time = cloud_arrival_time(
                fld_time_mem, setup.ens_param_thr, setup.ens_param_mem_min, mem_axis=0,
            )
        else:
            raise NotImplementedError(f"plot var '{plot_type}'")
        return fld_time

    def _collect_meta_data(
        self, fld_specs: FldSpecs, nc_meta_data: Mapping[str, Any],
    ) -> List[MetaDataCollection]:
        """Collect time-step-specific data meta data."""

        # Collect meta data at requested time steps for all members
        n_ts = len(fld_specs.collect_equal("time"))
        shape = (n_ts, self.n_members)
        mdata_by_reqtime_mem: np.ndarray = np.full(shape, None)
        for idx_mem, in_file_path in enumerate(self.in_file_path_lst or []):
            with nc4.Dataset(in_file_path, "r") as fi:
                for idx_time, fld_specs_i in enumerate(fld_specs.decompress_time()):
                    mdata_lst_i = []
                    for var_setup in fld_specs_i.var_setups:
                        mdata_lst_i.append(
                            collect_meta_data(
                                fi, var_setup, nc_meta_data["analysis"]["model"],
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
        mdata_lst: List[MetaDataCollection] = mdata_by_reqtime_mem[:, 0].tolist()

        # Fix some known issues with the NetCDF input data
        self.fixer.fix_meta_data(nc_meta_data["analysis"]["model"], mdata_lst)

        return mdata_lst

    def _create_field_objs(
        self,
        fld_specs: FldSpecs,
        fld_time: np.ndarray,
        time_stats: Mapping[str, np.ndarray],
        nc_meta_data: Mapping[str, Any],
    ) -> List[Field]:
        """Create fields at requested time steps for all members."""
        time_idcs = fld_specs.collect_equal("time")
        fields: List[Field] = []
        for time_idx in time_idcs:
            fld: np.ndarray = fld_time[time_idx]
            rotated_pole: bool = nc_meta_data["analysis"]["rotated_pole"]
            field = Field(fld, self.lat, self.lon, rotated_pole, fld_specs, time_stats)
            fields.append(field)
        return fields

    def _dim_names(self, nc_meta_data: Mapping[str, Any]) -> Dict[str, str]:
        """Model-specific dimension names."""
        model = nc_meta_data["analysis"]["model"]
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

    def _read_nc_var(
        self,
        fi: nc4.Dataset,
        setup: InputSetup,
        nc_meta_data: Mapping["str", Any],
        expand: Optional[List[str]] = None,
    ) -> np.ndarray:
        if expand is None:
            expand = []

        # Model-specific dimension names
        dim_names = self._dim_names(nc_meta_data)

        # Select variable in file
        var_name = nc_var_name(setup, nc_meta_data["analysis"]["model"])
        nc_var = fi.variables[var_name]

        # SR_TMP < TODO proper solution
        if setup.level is None:
            level = None
        else:
            assert len(setup.level) == 1
            level = next(iter(setup.level))
        # SR_TMP >

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
        self.fixer.fix_nc_var(nc_var, fld, nc_meta_data["analysis"]["model"])

        fld = self._handle_time_integration(fi, fld, setup)
        return fld

    def _handle_time_integration(
        self, fi: nc4.Dataset, fld: np.ndarray, setup: InputSetup,
    ) -> np.ndarray:
        """Integrate, or desintegrate, field over time."""
        if setup.variable == "concentration":
            if setup.integrate:
                # Integrate field over time
                dt_hr = self._compute_temporal_resolution(fi)
                return np.cumsum(fld, axis=0) * dt_hr
            else:
                # Field is already instantaneous
                return fld
        elif setup.variable == "deposition":
            if setup.integrate:
                # Field is already time-integrated
                return fld
            else:
                # Revert time integration of field
                dt_hr = self._compute_temporal_resolution(fi)
                fld[1:] = (fld[1:] - fld[:-1]) / dt_hr
                return fld
        raise NotImplementedError("unknown variable", setup.variable)

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
            raise NotImplementedError("variable", {"name": name})
        try:
            fact = self.conversion_factor_by_unit[unit]
        except KeyError:
            raise NotImplementedError(
                "conversion factor", {"name": name, "unit": unit},
            )
        fld[:] *= fact

    def fix_meta_data(
        self,
        model: str,
        mdata: Union[MetaDataCollection, Sequence[MetaDataCollection]],
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
        assert isinstance(mdata, MetaDataCollection)  # mypy

        name = mdata.species.name.value
        if name not in self.possible_var_names:
            raise NotImplementedError("variable", {"name": name})
        integr_type = mdata.simulation.integr_type.value
        old_unit = mdata.variable.unit.value

        new_unit = "Bq"
        if integr_type == "mean":
            pass
        elif integr_type in ["sum", "accum"]:
            new_unit += " h"
        else:
            raise NotImplementedError(
                "unknown integration type", {"integr_type": integr_type, "name": name},
            )
        if old_unit == "ng kg-1":
            new_unit += " m-3"
        elif old_unit == "1e-12 kg m-2":
            new_unit += " m-2"
        else:
            raise NotImplementedError(
                "unknown unit", {"name": name, "unit": old_unit},
            )
        mdata.variable.unit.value = new_unit

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

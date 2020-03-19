# -*- coding: utf-8 -*-
"""
IO.
"""
# Standard library
from copy import deepcopy
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
from .meta_data import MetaDataCollection
from .meta_data import collect_meta_data
from .meta_data import nc_var_name
from .setup import Setup
from .specs import FldSpecs


def read_files(in_file_path, setup, fld_specs_lst):
    return FileReader(in_file_path, setup).run(fld_specs_lst)


class FileReader:
    """Reader of NetCDF files containing FLEXPART data.

    It represents a single input file for deterministic FLEXPART runs, or an
    ensemble of input files for ensemble FLEXPART runs (one file per ensemble
    member).

    """

    choices_ens_var = ["mean", "median", "min", "max"]

    def __init__(self, in_file_path: str, setup: Setup):
        """Create an instance of ``FileReader``.

        Args:
            in_file_path: File path. In case of ensemble data, it must
                contain the format key '{ens_member[:0?d]}'.

            setup: Setup.

        """
        self.in_file_path_fmt = in_file_path
        self.setup = setup

        self.n_members: Optional[int] = None
        self.in_file_path_lst: Optional[Sequence[str]] = None
        self.model: str = None
        self.lat: Optional[np.ndarray] = None
        self.lon: Optional[np.ndarray] = None

        self.fixer: FlexPartDataFixer = FlexPartDataFixer(self)

    def run(
        self, fld_specs_lst: Sequence[FldSpecs]
    ) -> Tuple[List[Field], List[MetaDataCollection]]:
        """Read one or more fields from a file from disc.

        Args:
            fld_specs_lst: List of field specifications.

        """
        timeless_fld_specs_lst: List[FldSpecs]
        time_idcs_lst: List[List[int]]
        timeless_fld_specs_lst, time_idcs_lst = self._extract_time_idcs_from_fld_specs(
            fld_specs_lst,
        )

        # Prepare array for fields
        self.n_reqtime = self._determine_n_reqtime(time_idcs_lst)

        # Collect fields and attrs
        fields: List[Field] = []
        mdata_lst: List[MetaDataCollection] = []
        for timeless_fld_specs, time_idcs in zip(timeless_fld_specs_lst, time_idcs_lst):
            # SR_TMP < TODO cleaner  mypy-compatible solution
            ens_member_ids: Optional[Sequence[int]]
            assert timeless_fld_specs.fld_setup.ens_member_id is None or all(
                isinstance(i, int) for i in timeless_fld_specs.fld_setup.ens_member_id
            )  # mypy
            ens_member_ids = timeless_fld_specs.fld_setup.ens_member_id  # type: ignore
            # SR_TMP >
            self.n_members = 1 if not ens_member_ids else len(ens_member_ids)
            self.in_file_path_lst = self._prepare_in_file_path_lst(ens_member_ids)

            fields_i, mdata_lst_i = self._create_fields(timeless_fld_specs, time_idcs,)
            fields.extend(fields_i)
            mdata_lst.extend(mdata_lst_i)

        return fields, mdata_lst

    def _extract_time_idcs_from_fld_specs(
        self, fld_specs_lst: Sequence[FldSpecs]
    ) -> Tuple[List[FldSpecs], List[List[int]]]:
        """Group field specs that differ only in their time dimension.

        Steps:
         -  Group field specifications objects such that all in one group only
            differ in time.
         -  Collect the respective time indices.
         -  Merge the group into one field specifications instance with time
            dimension 'slice(None)'.

        """

        timeless_fld_specs_lst: List[FldSpecs] = []
        time_idcs_lst: List[List[int]] = []
        for fld_specs in fld_specs_lst:

            # Extract time index (shoult be the same for all)
            time_idcs = fld_specs.collect_equal("time")
            assert len(time_idcs) == 1  # SR_DBG
            time_idx: int = next(iter(time_idcs))

            # Reset time index
            dummy_time_idx = -999  # SR_TMP TODO find proper solution
            timeless_fld_specs: FldSpecs = deepcopy(fld_specs)
            timeless_fld_specs.var_setups = [
                var_setup.derive({"time": [dummy_time_idx]})
                for var_setup in timeless_fld_specs.var_setups
            ]

            for idx, timeless_fld_specs_i in enumerate(timeless_fld_specs_lst):
                if timeless_fld_specs == timeless_fld_specs_i:
                    if time_idx in time_idcs_lst[idx]:
                        raise Exception(
                            "duplicate fld_specs", time_idx, timeless_fld_specs,
                        )
                    time_idcs_lst[idx].append(time_idx)
                    break
            else:
                timeless_fld_specs_lst.append(timeless_fld_specs)
                time_idcs_lst.append([time_idx])

        return timeless_fld_specs_lst, time_idcs_lst

    def _prepare_in_file_path_lst(
        self, ens_member_ids: Union[Optional[Tuple[None]], Sequence[int]]
    ) -> List[str]:
        fmt_keys = ["{ens_member}", "{ens_member:"]
        fmt_key_in_path = any(k in self.in_file_path_fmt for k in fmt_keys)
        if ens_member_ids in [None, (None,)]:  # SR_TMP
            if fmt_key_in_path:
                raise ValueError(
                    f"input file path contains format key '{fmt_keys[0]}', but no "
                    f"ensemble member ids ('ens_member_ids') have been passed"
                )
            return [self.in_file_path_fmt]
        if not fmt_key_in_path:
            raise ValueError(
                f"input file path missing format key",
                fmt_keys[0],
                self.in_file_path_fmt,
            )
        assert ens_member_ids is not None  # mypy
        return [self.in_file_path_fmt.format(ens_member=id_) for id_ in ens_member_ids]

    def _determine_n_reqtime(self, time_idcs_lst: Sequence[Sequence[int]]) -> int:
        """Determine the number of selected time steps."""
        n_reqtime_per_mem = [len(idcs) for idcs in time_idcs_lst]
        n_reqtime = n_reqtime_per_mem.pop(0)
        if any(n != n_reqtime for n in n_reqtime_per_mem[1:]):
            raise Exception(
                "numbers of timesteps differ across members",
                [n_reqtime] + n_reqtime_per_mem,
            )
        return n_reqtime

    def _create_fields(
        self, timeless_fld_specs: FldSpecs, time_idcs: Sequence[int],
    ) -> Tuple[List[Field], List[MetaDataCollection]]:
        # SR_TMP <
        fld_setup = timeless_fld_specs.fld_setup
        var_setups = timeless_fld_specs.var_setups
        # SR_TMP >

        # Read fields of all members at all time steps
        fld_time_mem: np.ndarray = self._read_fld_time_mem(var_setups)

        # Reduce fields array along member dimension
        # In other words: Compute single field from ensemble
        fld_time: np.ndarray = self._reduce_ensemble(fld_time_mem, fld_setup)

        # Collect time stats
        time_stats: Dict[str, np.ndarray] = {
            "mean": np.nanmean(fld_time),
            "median": np.nanmedian(fld_time),
            "mean_nz": np.nanmean(fld_time[fld_time > 0]),
            "median_nz": np.nanmedian(fld_time[fld_time > 0]),
            "max": np.nanmax(fld_time),
        }

        # Create time-step-specific field specifications
        fld_specs_lst: List[FldSpecs] = self._expand_timeless_fld_specs_in_time(
            timeless_fld_specs, time_idcs,
        )

        # Collect time-step-specific data meta data
        mdata_lst: List[MetaDataCollection] = self._collect_meta_data(fld_specs_lst)

        # Create fields at requested time steps for all members
        fields: List[Field] = self._create_field_objs(
            fld_specs_lst, fld_time, time_stats,
        )

        return fields, mdata_lst

    def _read_fld_time_mem(self, setups: Sequence[Setup]) -> np.ndarray:
        """Read field over all time steps for each member."""
        fld_time_mem: Optional[np.ndarray] = None
        for i_mem, in_file_path in enumerate(self.in_file_path_lst or []):
            with nc4.Dataset(in_file_path, "r") as fi:

                # Determine model
                model = self._determine_model(fi)
                if self.model is None:
                    # SR_TMP <
                    if model in ["cosmo1", "cosmo2"]:
                        self.rotated_pole = True
                    else:
                        self.rotated_pole = False
                    self.model = model
                    # SR_TMP >
                elif model != self.model:
                    raise Exception("inconsistent model", model, self.model)

                # Read grid variables
                if not self.rotated_pole:
                    lat = fi.variables["latitude"][:]
                    lon = fi.variables["longitude"][:]
                else:
                    lat = fi.variables["rlat"][:]
                    lon = fi.variables["rlon"][:]
                if self.lat is None:
                    self.lat = lat
                    self.lon = lon
                else:
                    if not (lat == self.lat).all():
                        raise Exception("inconsistent latitude", lat, self.lat)
                    if not (lon == self.lon).all():
                        raise Exception("inconsistent longitude", lon, self.lon)

                # Read field (all time steps)
                fld_time: np.ndarray = merge_fields(
                    [self._read_nc_var(fi, setup) for setup in setups],
                )

                # Store field for current member
                if fld_time_mem is None:
                    shape = [self.n_members] + list(fld_time.shape)
                    fld_time_mem = np.full(shape, np.nan, np.float32)
                fld_time_mem[i_mem] = fld_time

        return fld_time_mem

    # SR_TODO Add class representing model, storing info like rotated pole etc.
    def _determine_model(self, fi):
        """Determine the model from the NetCDF meta data.

        For lack of an explicit NetCDF attribute, use the grid resolution.

        """
        dxout = fi.getncattr("dxout")
        choices = {
            type(dxout)(0.25): "ifs",
            type(dxout)(0.02): "cosmo2",
            type(dxout)(0.01): "cosmo1",
        }
        try:
            return choices[dxout]
        except KeyError:
            raise Exception("no model defined for dxout", dxout, choices)

    def _reduce_ensemble(self, fld_time_mem: np.ndarray, setup: Setup) -> np.ndarray:
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

    def _expand_timeless_fld_specs_in_time(
        self, timeless_fld_specs: FldSpecs, time_idcs: Sequence[int]
    ) -> List[FldSpecs]:
        fld_specs_lst: List[FldSpecs] = []
        for time_idx in time_idcs:
            fld_specs_i: FldSpecs = deepcopy(timeless_fld_specs)
            # SR_TMP <
            fld_specs_i.var_setups = [
                var_setup.derive({"time": [time_idx]})
                for var_setup in fld_specs_i.var_setups
            ]
            # SR_TMP >
            fld_specs_lst.append(fld_specs_i)
        return fld_specs_lst

    def _collect_meta_data(
        self, fld_specs_lst: Sequence[FldSpecs]
    ) -> List[MetaDataCollection]:
        """Collect time-step-specific data meta data."""

        # Collect meta data at requested time steps for all members
        shape = (self.n_reqtime, self.n_members)
        mdata_by_reqtime_mem: np.ndarray = np.full(shape, None)
        for idx_mem, in_file_path in enumerate(self.in_file_path_lst or []):
            with nc4.Dataset(in_file_path, "r") as fi:
                for idx_time, fld_specs in enumerate(fld_specs_lst):
                    mdata_lst_i = [
                        collect_meta_data(fi, var_setup, self.model)
                        for var_setup in fld_specs.var_setups
                    ]
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
        assert isinstance(next(iter(mdata_lst)), MetaDataCollection)  # SR_DBG

        # Fix some known issues with the NetCDF input data
        self.fixer.fix_meta_data(mdata_lst)

        return mdata_lst

    def _create_field_objs(
        self,
        fld_specs_lst: Sequence[FldSpecs],
        fld_time: np.ndarray,
        time_stats: Mapping[str, np.ndarray],
    ) -> List[Field]:
        """Create fields at requested time steps for all members."""
        fields: List[Field] = []
        for fld_specs in fld_specs_lst:
            time_idcs = fld_specs.collect_equal("time")
            assert len(time_idcs) == 1  # SR_TMP
            time_idx = next(iter(time_idcs))
            fld: np.ndarray = fld_time[time_idx]
            field = Field(
                fld, self.lat, self.lon, self.rotated_pole, fld_specs, time_stats,
            )
            fields.append(field)
        return fields

    def _read_nc_var(self, fi: nc4.Dataset, setup: Setup) -> np.ndarray:

        # Select variable in file
        var_name = nc_var_name(setup, self.model)
        nc_var = fi.variables[var_name]

        # SR_TMP < TODO remove once CoreSetup implemented
        assert setup.level is None or len(setup.level) == 1
        assert len(setup.nageclass) == 1
        assert len(setup.noutrel) == 1
        assert len(setup.numpoint) == 1
        # SR_TMP >

        # SR_TMP < TODO proper solution
        if setup.level is None:
            level = None
        else:
            level = next(iter(setup.level))
        # SR_TMP >

        # Indices of field along NetCDF dimensions
        dim_idcs_by_name = {
            "nageclass": next(iter(setup.nageclass)),
            "noutrel": next(iter(setup.noutrel)),
            "numpoint": next(iter(setup.numpoint)),
            "level": level,
            "time": slice(None),  # SR_TMP
            "rlat": slice(None),  # SR_TMP
            "rlon": slice(None),  # SR_TMP
        }
        # SR_TMP < TODO proper implementation
        if self.model == "ifs":
            dim_idcs_by_name["pointspec"] = dim_idcs_by_name.pop("numpoint")  # SR_TMP
            dim_idcs_by_name["height"] = dim_idcs_by_name.pop("level")  # SR_TMP
            assert not self.rotated_pole  # SR_TMP
            dim_idcs_by_name["latitude"] = dim_idcs_by_name.pop("rlat")  # SR_TMP
            dim_idcs_by_name["longitude"] = dim_idcs_by_name.pop("rlon")  # SR_TMP
        # SR_TMP >

        # Assemble indices for slicing
        idcs: List[Any] = [None] * len(nc_var.dimensions)
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

            idcs[idx] = dim_idx

        # Check that all variable dimensions have been identified
        try:
            idx = idcs.index(None)
        except ValueError:
            pass  # All good!
        else:
            raise Exception(
                "unknown variable dimension",
                nc_var.dimensions[idx],
                {
                    "idx": idx,
                    "var_name": var_name,
                    "idcs": idcs,
                    "dimensions": nc_var.dimensions,
                    "dim_idcs_by_name": dim_idcs_by_name,
                },
            )
        check_array_indices(nc_var.shape, idcs)
        fld = nc_var[idcs]

        # Fix known issues with NetCDF input data
        self.fixer.fix_nc_var(nc_var, fld)

        fld = self._handle_time_integration(fi, fld, setup)
        return fld

    def _handle_time_integration(
        self, fi: nc4.Dataset, fld: np.ndarray, setup: Setup,
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

    def fix_nc_var(self, nc_var: nc4.Variable, fld: np.ndarray) -> None:
        if self.file_reader.model in ["cosmo2", "cosmo1"]:
            self._fix_nc_var_cosmo(nc_var, fld)
        elif self.file_reader.model == "ifs":
            pass
        else:
            raise NotImplementedError("model", self.file_reader.model)

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
        self, mdata: Union[MetaDataCollection, Sequence[MetaDataCollection]],
    ) -> None:
        if self.file_reader.model in ["cosmo2", "cosmo1"]:
            self._fix_meta_data_cosmo(mdata)
        elif self.file_reader.model == "ifs":
            pass
        else:
            raise NotImplementedError("model", self.file_reader.model)

    def _fix_meta_data_cosmo(self, mdata):
        if isinstance(mdata, Sequence):
            for mdata_i in mdata:
                self.fix_meta_data(mdata_i)
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

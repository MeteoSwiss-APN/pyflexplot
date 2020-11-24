"""Data input."""
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
import scipy.stats as sp_stats
from cartopy.crs import PlateCarree
from cartopy.crs import RotatedPole

# First-party
from srutils.various import check_array_indices

# Local
from ..setup import Setup
from ..setup import SetupCollection
from ..utils.exceptions import MissingCacheEntryError
from ..utils.logging import log
from .data import Cloud
from .data import ensemble_probability
from .data import EnsembleCloud
from .data import Field
from .data import FieldTimeProperties
from .data import merge_fields
from .fix_nc_input import FlexPartDataFixer
from .meta_data import MetaData
from .nc_meta_data import derive_variable_name
from .nc_meta_data import read_nc_meta_data

AFFECTED_AREA_THRESHOLD = 0.0
CLOUD_THRESHOLD = 0.0


@dataclass
class FieldInputConfig:
    """Field input configuration.

    Args:
        add_ts0 (optional): Insert an additional time step 0 in the beginning
            with empty fields, given that the first data time step may not
            correspond to the beginning of the simulation, but constitute the
            sum over the first few hours of the simulation.

        dry_run (optional): Perform dry run without reading the fields from disk
            (meta data are still read).

        cache_on (optional): Cache input fields to avoid reading the same field
            multiple times.

        missing_ok (optional): When fields are missing in the input, instead of
            failing, return an empty field.

        cls_fixer (optional): Class providing methods to fix issues with the
            input data. Is instatiated with this instance of ``FileReader``.

    """

    add_ts0: bool = False
    dry_run: bool = False
    cache_on: bool = False
    missing_ok: bool = False
    cls_fixer: Optional[Type["FlexPartDataFixer"]] = None


# pylint: disable=R0914  # too-many-locals
def read_fields(
    in_file_path: str,
    setups: SetupCollection,
    **config_kwargs: Any,
) -> List[List[Field]]:
    """Read fields from an input file, or multiple files derived from one path.

    Args:
        in_file_path: Input file path. In case of ensemble data, it must contain
            the format key '{ens_member[:0?d]}', in which case a separate path
            is derived for each member.

        setups: Collection variable setups, containing among other things the
            ensemble member IDs in case of an ensemble simulation.

        config_kwargs (optional): Keyword arguments used to create an instance
            of ``FieldInputConfig``.

    """
    log(dbg=f"reading fields from {in_file_path}")
    if "cls_fixer" not in config_kwargs:
        config_kwargs["cls_fixer"] = FlexPartDataFixer
    reader = FieldInputOrganizer(in_file_path, **config_kwargs)
    field_lst_lst = reader.run(setups)
    n_plt = len(field_lst_lst)
    n_tot = sum([len(field_lst) for field_lst in field_lst_lst])
    log(dbg=f"done reading {in_file_path}: read {n_tot} fields for {n_plt} plots")
    return field_lst_lst


# pylint: disable=R0902  # too-many-instance-attributes
class FieldInputOrganizer:
    """Organize input of fields from FLEXPART NetCDF files.

    It represents a single input file for deterministic FLEXPART runs, or an
    ensemble of input files for ensemble FLEXPART runs (one file per ensemble
    member).

    """

    def __init__(self, in_file_path: str, **config_kwargs: Any):
        """Create an instance of ``FieldInputOrganizer``.

        Args:
            in_file_path: File path. In case of ensemble data, it must contain the
                format key '{ens_member[:0?d]}'.

            config_kwargs (optional): Keyword arguments used to create an instance
                of ``FieldInputConfig``.

        """
        self.in_file_path = in_file_path
        self.config = FieldInputConfig(**config_kwargs)

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
                    "input file path missing format key", path_fmt, ens_member_ids
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
                    _,  # ens_member_id,
                    input_variable,
                    combine_levels,
                    combine_deposition_types,
                    combine_species,
                ),
                sub_setups,
            ) in setups.group(
                [
                    "ens_member_id",
                    "input_variable",
                    "combine_levels",
                    "combine_deposition_types",
                    "combine_species",
                ]
            ).items():
                skip = ["outfile", "dimensions.time", "ens_member_id"]
                if input_variable in [
                    "concentration",
                    "cloud_arrival_time",
                    "cloud_departure_time",
                ]:
                    if combine_levels:
                        skip.append("dimensions.level")
                elif input_variable in ["deposition", "affected_area"]:
                    if combine_deposition_types:
                        skip.append("dimensions.deposition_type")
                if input_variable in [
                    "affected_area",
                    "cloud_arrival_time",
                    "cloud_departure_time",
                ]:
                    skip.append("input_variable")
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

        all_ens_member_ids = (
            setups.collect("ens_member_id", flatten=True, exclude_nones=True) or None
        )
        in_file_path_lst = prep_in_file_path_lst(self.in_file_path, all_ens_member_ids)

        first_in_file_path = next(iter(in_file_path_lst))
        with nc4.Dataset(first_in_file_path) as fi:
            # SR_TMP < Find better solution! Maybe class NcMetaDataReader?
            nc_meta_data = InputFileEnsemble.read_nc_meta_data(
                cast(InputFileEnsemble, self), fi
            )
            # SR_TMP >
        setups = setups.complete_dimensions(nc_meta_data)
        setups_for_plots_over_time = group_setups(setups)

        files = InputFileEnsemble(in_file_path_lst, self.config, all_ens_member_ids)
        fields_for_plots: List[List[Field]] = []
        for setups_for_same_plot_over_time in setups_for_plots_over_time:
            ens_mem_ids = setups_for_same_plot_over_time.collect_equal("ens_member_id")
            fields_for_plots.extend(
                files.read_fields_over_time(setups_for_same_plot_over_time, ens_mem_ids)
            )

        return fields_for_plots


@dataclass
class InputFileEnsemble:
    """An ensemble (of size one or more) of input files."""

    paths: Sequence[str]
    config: FieldInputConfig
    ens_member_ids: Optional[Sequence[int]] = None

    def __post_init__(self):
        # SR_TMP < TODO Fix the cache!!!
        if self.config.cache_on:
            raise NotImplementedError("input file cache is currently broken!")

        if len(self.ens_member_ids or [0]) != len(self.paths):
            raise ValueError(
                "must pass same number of ens_member_ids and paths"
                ", or omit ens_member_ids for single path",
                self.paths,
                self.ens_member_ids,
            )

        self.cache = FileReaderCache(self)

        self.fixer: Optional["FlexPartDataFixer"] = (
            self.config.cls_fixer(self) if self.config.cls_fixer else None
        )

        # Declare some attributes
        self.fld_time_mem: np.ndarray
        self.lat: np.ndarray
        self.lon: np.ndarray
        self.mdata_tss: List[MetaData]
        self.nc_meta_data: Dict[str, Dict[str, Any]]
        self.setups_lst_time: List[SetupCollection]
        self.time: np.ndarray

    # SR_TMP <<<
    # pylint: disable=R0914  # too-many-locals
    def read_fields_over_time(
        self, setups: SetupCollection, ens_member_ids: Optional[Sequence[int]] = None
    ) -> List[List[Field]]:
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
            ts_hrs = self.get_temp_res_hrs(fi)

        # Create individual setups at each requested time step
        select = ["dimensions.time"]
        skip = ["outfile"]
        skip.append("input_variable")  # For "affected_area"
        self.setups_lst_time = setups.decompress_partially(select, skip=skip)

        model = setups.collect_equal("model")
        self.fld_time_mem = np.full(self._get_shape_mem_time(model), np.nan, np.float32)
        for idx_mem, (ens_member_id, file_path) in enumerate(
            zip((self.ens_member_ids or [None]), self.paths)  # type: ignore
        ):
            if ens_member_ids is not None and ens_member_id not in ens_member_ids:
                continue
            setups_mem = setups.derive({"ens_member_id": ens_member_id})
            timeless_setups_mem = setups_mem.derive({"dimensions": {"time": None}})
            self._read_data(file_path, idx_mem, timeless_setups_mem)

        # Compute single field from all ensemble members
        fld_time: np.ndarray = self._reduce_ensemble(setups, ts_hrs)

        # Compute some statistics across all time steps
        time_stats = FieldTimeProperties(fld_time)

        # Create Field objects at requested time steps
        field_lst_lst: List[List[Field]] = []
        for field_setups, mdata in zip(self.setups_lst_time, self.mdata_tss):
            time_idx = field_setups.collect_equal("dimensions.time")
            fld: np.ndarray = fld_time[time_idx]
            if self.nc_meta_data["derived"]["rotated_pole"]:
                ncattrs = self.nc_meta_data["variables"]["rotated_pole"]["ncattrs"]
                proj = RotatedPole(
                    pole_latitude=ncattrs["grid_north_pole_latitude"],
                    pole_longitude=ncattrs["grid_north_pole_longitude"],
                )
            else:
                proj = PlateCarree()
            field = Field(
                fld=fld,
                lat=self.lat,
                lon=self.lon,
                proj=proj,
                var_setups=field_setups,
                time_props=time_stats,
                nc_meta_data=self.nc_meta_data,
                mdata=mdata,
            )
            field_lst_lst.append([field])  # SR_TMP TODO SR_MULTIPANEL

        return field_lst_lst

    def _read_data(
        self, file_path: str, idx_mem: int, timeless_setups_mem: SetupCollection
    ) -> None:
        n_mem = len(self.paths)
        model = timeless_setups_mem.collect_equal("model")

        # Try to fetch the necessary data from cache
        if not self.config.cache_on:
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
            self._read_grid(fi, model)

            if idx_mem > 0:
                # Ensure that meta data is the same for all members
                self.nc_meta_data = self.read_nc_meta_data(fi, check=True)

            # Read meta data at requested time steps
            mdata_tss_i = self._collect_meta_data_tss(fi)
            if idx_mem == 0:
                self.mdata_tss = mdata_tss_i
            elif mdata_tss_i != self.mdata_tss:
                raise Exception("meta data differ across members")

            if not self.config.dry_run:
                # Read fields for all members at all time steps
                fld_time_i = self._read_member_fields_over_time(fi, timeless_setups_mem)
                self.fld_time_mem[idx_mem][:] = fld_time_i[:]
            else:
                fld_time_i = np.empty(self.fld_time_mem.shape[1:])

            if self.config.cache_on:
                self.cache.add(cache_key, fld_time_i, mdata_tss_i)

    def read_nc_meta_data(self, fi: nc4.Dataset, check: bool = False) -> Dict[str, Any]:
        nc_meta_data = read_nc_meta_data(fi)
        if self.config.add_ts0:
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
        input_variable = timeless_setups.collect_equal("input_variable")
        for sub_setups in timeless_setups.decompress_partially(None, skip=["outfile"]):
            for sub_setup in sub_setups:
                fld_time_lst.append(self._read_fld_over_time(fi, sub_setup))
        if input_variable == "affected_area":
            shapes = [fld.shape for fld in fld_time_lst]
            # SR_TMP < TODO proper check
            assert len(set(shapes)) == 1, f"shapes differ: {shapes}"
            # SR_TMP >
            fld_time = (np.array(fld_time_lst) > AFFECTED_AREA_THRESHOLD).any(axis=0)
        else:
            fld_time = merge_fields(fld_time_lst)
        if self.fixer and timeless_setups.collect_equal("model") == "IFS-HRES":
            # Note: IFS-HRES-EU is not global, so doesn't need this fix
            self.fixer.fix_global_grid(self.lon, fld_time)
        if self.config.add_ts0:
            fld_time = self._add_ts0_to_fld_time(fld_time)
        if input_variable in ["cloud_arrival_time", "cloud_departure_time"]:
            ts_hrs = self.get_temp_res_hrs(fi)
            cloud = Cloud(mask=np.array(fld_time) > CLOUD_THRESHOLD, ts=ts_hrs)
            if input_variable == "cloud_arrival_time":
                fld_time = cloud.arrival_time()
            else:
                fld_time = cloud.departure_time()
        return fld_time

    # pylint: disable=R0914  # too-many-locals
    def _collect_meta_data_tss(self, fi: nc4.Dataset) -> List[MetaData]:
        """Collect time-step-specific data meta data."""
        mdata_lst: List[MetaData] = []
        skip = ["outfile"]
        skip.append("input_variable")  # For "affected_area"
        for setups in self.setups_lst_time:
            mdata_i_lst: List[MetaData] = []
            for sub_setups in setups.decompress_partially(None, skip=skip):
                for sub_setup in sub_setups:
                    mdata_ij: MetaData = MetaData.collect(
                        fi, sub_setup, add_ts0=self.config.add_ts0
                    )
                    mdata_i_lst.append(mdata_ij)
            mdata_i = mdata_i_lst[0].merge_with(mdata_i_lst[1:])

            # Fix some known issues with the NetCDF input data
            if self.fixer:
                model_name = setups.collect_equal("model")
                input_variable = setups.collect_equal("input_variable")
                integrate = setups.collect_equal("integrate")
                self.fixer.fix_meta_data(model_name, input_variable, integrate, mdata_i)

            mdata_lst.append(mdata_i)
        return mdata_lst

    def _get_shape_mem_time(self, model: str) -> Tuple[int, int, int, int]:
        """Get the shape of an array of fields across members and time steps."""
        dim_names = self._dim_names(model)
        nlat = self.nc_meta_data["dimensions"][dim_names["lat"]]["size"]
        nlon = self.nc_meta_data["dimensions"][dim_names["lon"]]["size"]
        nts = self.nc_meta_data["dimensions"][dim_names["time"]]["size"]
        n_mem = len(self.ens_member_ids) if self.ens_member_ids else 1
        self.lat = np.full((nlat,), np.nan)
        self.lon = np.full((nlon,), np.nan)
        return (n_mem, nts, nlat, nlon)

    def _read_grid(self, fi: nc4.Dataset, model: str) -> None:
        """Read and prepare grid variables."""
        dim_names = self._dim_names(model)
        lat = fi.variables[dim_names["lat"]][:]
        lon = fi.variables[dim_names["lon"]][:]
        time = fi.variables[dim_names["time"]][:]
        time = self._prepare_time(fi, time, model)
        self.lat = lat
        self.lon = lon
        self.time = time

    def _prepare_time(
        self, fi: nc4.Dataset, time: np.ndarray, model: str
    ) -> np.ndarray:
        if self.config.add_ts0:
            dts = time[1] - time[0]
            ts0 = time[0] - dts
            time = np.r_[ts0, time]
        # Convert seconds to hours
        dim_names = self._dim_names(model)
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

    # pylint: disable=R0912  # too-many-branches
    def _reduce_ensemble(
        self, var_setups: SetupCollection, ts_hrs: float
    ) -> np.ndarray:
        """Reduce the ensemble to a single field (time, lat, lon)."""
        fld_time_mem = self.fld_time_mem

        if not self.ens_member_ids or self.config.dry_run:
            return fld_time_mem[0]

        ens_param_mem_min = var_setups.collect_equal("ens_param_mem_min")
        ens_param_pctl = var_setups.collect_equal("ens_param_pctl")
        ens_param_thr = var_setups.collect_equal("ens_param_thr")
        ens_param_thr_type = var_setups.collect_equal("ens_param_thr_type")
        ens_variable = var_setups.collect_equal("ens_variable")

        fld_time: np.ndarray
        if ens_variable == "minimum":
            fld_time = np.nanmin(fld_time_mem, axis=0)
        elif ens_variable == "maximum":
            fld_time = np.nanmax(fld_time_mem, axis=0)
        elif ens_variable == "median":
            fld_time = np.nanmedian(fld_time_mem, axis=0)
        elif ens_variable == "mean":
            fld_time = np.nanmean(fld_time_mem, axis=0)
        elif ens_variable == "std_dev":
            fld_time = np.nanstd(fld_time_mem, axis=0)
        elif ens_variable == "med_abs_dev":
            fld_time = sp_stats.median_abs_deviation(fld_time_mem, axis=0)
        elif ens_variable == "percentile":
            assert ens_param_pctl is not None  # mypy
            fld_time = np.percentile(fld_time_mem, ens_param_pctl, axis=0)
        elif ens_variable == "probability":
            fld_time = ensemble_probability(
                fld_time_mem, ens_param_thr, ens_param_thr_type
            )
        elif ens_variable.startswith("ens_cloud_"):
            cloud = EnsembleCloud(
                mask=fld_time_mem > ens_param_thr, mem_min=ens_param_mem_min, ts=ts_hrs
            )
            if ens_variable == "ens_cloud_arrival_time":
                fld_time = cloud.arrival_time()
            elif ens_variable == "ens_cloud_departure_time":
                fld_time = cloud.departure_time()
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
        elif model in ["IFS-HRES", "IFS-HRES-EU"]:
            return {
                "lat": "latitude",
                "lon": "longitude",
                "time": "time",
                "level": "height",
                "nageclass": "nageclass",
                "noutrel": "noutrel",
                "numpoint": "pointspec",
            }
        else:
            raise NotImplementedError("dimension names for model", model)

    # pylint: disable=R0912,R0914  # too-many-branches, too-many-locals
    def _read_fld_over_time(self, fi: nc4.Dataset, setup: Setup) -> np.ndarray:
        """Read a 2D field at all time steps from disk."""
        # Indices of field along NetCDF dimensions
        dim_names = self._dim_names(setup.model)
        dim_idcs_by_name = {
            dim_names["lat"]: slice(None),
            dim_names["lon"]: slice(None),
        }
        for dim_name in ["nageclass", "noutrel", "numpoint", "time", "level"]:
            if dim_name in ["lat", "lon", "time"]:
                idcs = slice(None)
            else:
                idcs = getattr(setup.core.dimensions, dim_name)
            dim_idcs_by_name[dim_names[dim_name]] = idcs

        # Select variable in file
        assert setup.core.dimensions.species_id is not None  # mypy
        var_name = derive_variable_name(
            model=setup.model,
            input_variable=setup.core.input_variable,
            species_id=setup.core.dimensions.species_id,
            deposition_type=setup.deposition_type_str,
        )
        try:
            nc_var = fi.variables[var_name]
        except KeyError as e:
            if not self.config.missing_ok:
                raise Exception(f"missing variable '{var_name}'") from e
            shape = (
                fi.dimensions[dim_names["time"]].size,
                fi.dimensions[dim_names["lat"]].size,
                fi.dimensions[dim_names["lon"]].size,
            )
            return np.zeros(shape, np.float32)

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
            except ValueError as e:
                # Potential issue: Dimension is not among the variable dimensions!
                if dim_idx in (None, 0):
                    continue  # Zero-index: We're good after all!
                raise Exception(
                    "dimension with non-zero index missing",
                    {**err_dct, "dimensions": nc_var.dimensions, "var_name": var_name},
                ) from e

            # Check that the index along the dimension is valid
            if dim_idx is None:
                raise Exception(
                    f"value of dimension '{err_dct['dim_name']} is None", idx, err_dct
                )

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
        var_ncattrs = {attr: nc_var.getncattr(attr) for attr in nc_var.ncattrs()}
        if self.fixer:
            self.fixer.fix_nc_var_fld(fld, setup.model, var_ncattrs)

        return self._handle_time_integration(
            fi, fld, setup.core.input_variable, setup.core.integrate
        )

    def _handle_time_integration(
        self, fi: nc4.Dataset, fld: np.ndarray, input_variable: str, integrate: bool
    ) -> np.ndarray:
        """Integrate or desintegrate the field over time."""
        if input_variable == "concentration":
            if integrate:
                # Integrate field over time
                dt_hr = self.get_temp_res_hrs(fi)
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
                dt_hr = self.get_temp_res_hrs(fi)
                fld[1:] = (fld[1:] - fld[:-1]) / dt_hr
                return fld
        raise NotImplementedError("unknown variable", input_variable)

    @staticmethod
    def get_temp_res_hrs(fi: nc4.Dataset) -> float:
        """Determine the (constant) temporal resolution in hours."""
        time = fi.variables["time"]
        dts = set(time[1:] - time[:-1])
        if len(dts) > 1:
            raise Exception(f"Non-uniform time resolution: {sorted(dts)} ({time})")
        dt_min = next(iter(dts))
        dt_hr = dt_min / 3600.0
        return dt_hr


class FileReaderCache:
    """Cache for file reader."""

    def __init__(self, files: InputFileEnsemble) -> None:
        """Create an instance of ``FileReaderCache``."""
        self.files = files
        self._cache: Dict[Hashable, Any] = {}

    # SR_TMP <<< TODO Derive better key from setups! Make hashable?
    @staticmethod
    def create_key(file_path: str, setups: SetupCollection) -> Hashable:
        params = ["input_variable", "integrate", "dimensions"]
        return tuple([file_path] + [repr(setups.collect(param)) for param in params])

    def add(
        self, cache_key: Hashable, fld_time_i: np.ndarray, mdata_tss_i: List[MetaData]
    ) -> None:
        self._cache[cache_key] = {
            "grid": deepcopy((self.files.lat, self.files.lon, self.files.time)),
            "nc_meta_data": deepcopy(self.files.nc_meta_data),
            "fld_time_i": deepcopy(fld_time_i),
            "mdata_tss_i": deepcopy(mdata_tss_i),
        }

    def get(self, key: Hashable, idx_mem: int) -> None:
        try:
            entry = self._cache[key]
        except KeyError as e:
            raise MissingCacheEntryError(key) from e
        entry = deepcopy(entry)
        self.files.lat, self.files.lon, self.files.time = entry["grid"]
        self.files.nc_meta_data = entry["nc_meta_data"]
        self.files.fld_time_mem[idx_mem][:] = entry["fld_time_i"]
        self.files.mdata_tss = entry["mdata_tss_i"]

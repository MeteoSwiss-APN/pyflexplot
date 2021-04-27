"""Data input."""
# Standard library
import dataclasses as dc
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

# Third-party
import netCDF4 as nc4  # types: ignore  # pylance
import numpy as np
import scipy.stats as sp_stats

# First-party
from srutils.various import check_array_indices

# Local
from ..setups.dimensions import Dimensions
from ..setups.model_setup import ModelSetup
from ..setups.plot_panel_setup import PlotPanelSetup
from ..setups.plot_panel_setup import PlotPanelSetupGroup
from ..setups.plot_setup import PlotSetup
from ..setups.plot_setup import PlotSetupGroup
from ..utils.logging import log
from .data import Cloud
from .data import ensemble_probability
from .data import EnsembleCloud
from .data import merge_fields
from .field import Field
from .field import FieldGroup
from .field import FieldGroupAttrs
from .field import FieldTimeProperties
from .fix_nc_input import FlexPartDataFixer
from .meta_data import derive_variable_name
from .meta_data import MetaData
from .meta_data import read_dimensions
from .meta_data import read_species_ids
from .meta_data import read_time_steps

AFFECTED_AREA_THRESHOLD = 0.0
CLOUD_THRESHOLD = 0.0


@dc.dataclass
class InputConfig:
    """Input configuration.

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

    """

    add_ts0: bool = True
    dry_run: bool = False
    cache_on: bool = False
    missing_ok: bool = False
    cls_fixer: Optional[Type["FlexPartDataFixer"]] = FlexPartDataFixer


# pylint: disable=R0914  # too-many-locals
def read_fields(
    setup_group: PlotSetupGroup,
    config: Optional[Union[InputConfig, Dict[str, Any]]] = None,
    *,
    only: Optional[int] = None,
    _override_infile: Optional[str] = None,
) -> List[FieldGroup]:
    """Read fields from an input file, or multiple files derived from one path.

    Args:
        setup_group: Group of setups, containing among other things the input
            file name and the ensemble member ids (if there are any).

        config (optional): Instance of ``InputConfig``, or dict with parameters
            used to create one.

        only (optional): Restrict the number of fields that are read.

        _override_infile (optional): Override ``setups.files.input``; should not be
            used outside of tests.

    """
    setup_group = setup_group.copy()
    model_setup = setup_group.collect_equal("model")
    ens_member_ids = setup_group.ens_member_ids

    if not isinstance(config, InputConfig):
        config = InputConfig(**(config or {}))

    files = InputFileEnsemble(
        raw_path=setup_group.infile,
        config=config,
        model_setup=model_setup,
        ens_member_ids=ens_member_ids,
        override_raw_path=_override_infile,
    )

    # SR_TMP < TODO improve log message
    log(dbg=f"reading fields from {files}")
    # SR_TMP >

    first_path = next(iter(files.paths))
    with nc4.Dataset(first_path) as fi:
        raw_dimensions = read_dimensions(fi, add_ts0=config.add_ts0)
        species_ids = read_species_ids(fi)
        time_steps = read_time_steps(fi)
    if model_setup.base_time is None:
        model_setup.base_time = int(time_steps[0])
    for plot_type_setup in setup_group:
        plot_type_setup.model.base_time = model_setup.base_time

    # SR_TODO Consider moving group_setups_by_plot_type into complete_dimensions
    setup_group = setup_group.complete_dimensions(
        raw_dimensions=raw_dimensions,
        species_ids=species_ids,
    )

    # Limit number of plots
    if only and len(setup_group) > only:
        log(dbg=f"[only:{only}] skip {len(setup_group) - only}/{len(setup_group)}")
        setup_group = PlotSetupGroup(list(setup_group)[:only])

    # Separate setups of different plot types
    # Each setup may still lead to multiple plots (multiple outfiles/time steps)
    plot_type_setups = group_setups_by_plot_type(setup_group)

    # Read fields for plots (all time steps for each plot type at once)
    field_groups: List[FieldGroup] = []
    for plot_type_setup in plot_type_setups:
        field_groups_i = files.read_fields_over_time(plot_type_setup)
        if only and len(field_groups) + len(field_groups_i) > only:
            log(dbg=f"[only:{only}] skip reading remaining fields after {only}")
            field_groups.extend(field_groups_i[: only - len(field_groups)])
            break
        field_groups.extend(field_groups_i)

    n_plt = len(field_groups)
    n_tot = sum([len(field_group) for field_group in field_groups])
    log(dbg=(f"done reading infiles: read {n_tot} fields for {n_plt} plots"))
    return field_groups


def prepare_paths(path: str, ens_member_ids: Optional[Sequence[int]]) -> List[str]:
    """Prepare paths: one for deterministic, multiple for ensemble run."""
    key = "ens_member"
    if re.search(f"{{{key}" + r"(:[0-9]+d?)?}", path):
        if not ens_member_ids:
            raise ValueError(
                "input file path contains ensemble member format key, but no "
                f"ensemble member ids have been passed: {path}"
            )
        assert ens_member_ids is not None  # mypy
        paths = [path.format(**{key: id_}) for id_ in ens_member_ids]
    elif not ens_member_ids:
        paths = [path]
    else:
        raise ValueError(f"format key '{key}' missing in input file path {path}")
    if ens_member_ids and len(ens_member_ids) != len(paths):
        raise ValueError(
            f"must pass same number of ens_member_ids ({len(ens_member_ids)}"
            f" and paths ({len(paths)}), or omit ens_member_ids (single path)"
        )
    return paths


def group_setups_by_plot_type(setups: PlotSetupGroup) -> PlotSetupGroup:
    """Group the setups by plot type and time step.

    Return a group of setups, each of which defines one plot type (based on
    multiple fields in case of multiple panels) at different time steps.

    """
    setup_lst: List[PlotSetup] = []
    for (
        _,
        plot_type,
        multipanel_param,
        plot_variable,
        combine_levels,
        combine_species,
    ), sub_setups in setups.group(
        [
            "model.ens_member_id",
            "layout.plot_type",
            "layout.multipanel_param",
            "plot_variable",
            "combine_levels",
            "combine_species",
        ]
    ).items():
        skip = ["files.output", "dimensions.time", "model.ens_member_id"]
        if plot_variable in [
            "affected_area",
            "concentration",
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            if combine_levels:
                skip.append("dimensions.level")
        if combine_species:
            skip.append("dimensions.species_id")
        if plot_type == "multipanel":
            skip.append(multipanel_param)
        for plot_setup in sub_setups.decompress(None, skip=skip):
            setup_lst.append(plot_setup)
    return PlotSetupGroup(setup_lst)


# pylint: disable=R0902  # too-many-instance-attributes (>7)
class InputFileEnsemble:
    """An ensemble (of size one or more) of input files."""

    # pylint: disable=R0913  # too-many-arguments
    def __init__(
        self,
        raw_path: str,
        config: InputConfig,
        model_setup: ModelSetup,
        ens_member_ids: Optional[Sequence[int]] = None,
        *,
        override_raw_path: Optional[str] = None,
    ):
        """Create an instance of ``InputFileEnsemble``.

        Args:
            raw_path: Raw file path, with format key for member ID in case of
                an ensemble simulation.

            config: Input configuration.

            model_setup: Model setup.

            ens_member_ids (optional): Ensemble member IDs; required for
                ensemble simulations, in which case ``raw_path`` must contain
                a corresponding format key.

            override_raw_path (optional): Raw path from which to derive the
                paths of the files on disk; if passed, the paths derived from
                ``raw_path`` are only passed to the returned ``FieldGroup``
                object as attributes; useful in tests to read files from a
                temporary directory that should not be stored in the attributes,
                in which case a simplified, reproducible path can be passed to
                ``raw_path``, and the actual path to ``override_raw_path``.

        """
        self.config: InputConfig = config
        self.model_setup: ModelSetup = model_setup

        self.paths: List[str] = prepare_paths(
            override_raw_path or raw_path, ens_member_ids
        )
        self.public_paths: List[str] = prepare_paths(raw_path, ens_member_ids)
        if len(self.paths) != len(self.public_paths):
            raise Exception(
                f"unequal number of paths ({len(self.paths)}) and public_paths"
                f" ({len(self.public_paths)}) derived from raw_path '{raw_path}'"
                f" and override_raw_path '{override_raw_path}'"
            )

        # SR_TMP TODO Fix the cache!!!
        if self.config.cache_on:
            raise NotImplementedError("input file cache")

        self.fixer: Optional["FlexPartDataFixer"] = None
        if self.config.cls_fixer:
            self.fixer = self.config.cls_fixer(self)

        # Declare some attributes
        self.lat: np.ndarray
        self.lon: np.ndarray
        self.mdata_tss: List[MetaData]
        self.time: np.ndarray

    # pylint: disable=R0914  # too-many-locals
    def read_fields_over_time(self, plot_setup: PlotSetup) -> List[FieldGroup]:
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
            nc_dimensions_dct = read_dimensions(fi, add_ts0=self.config.add_ts0)
            ts_hrs = self.get_temp_res_hrs(fi)

        field_lst_by_ts: Dict[int, List[Field]] = {}
        for panel_setup_i in plot_setup.panels:

            # Create individual setups at each requested time step
            panel_setups_req_time: PlotPanelSetupGroup = panel_setup_i.decompress(
                ["dimensions.time"]
            )

            fld_time_mem = np.full(
                self._get_shape_mem_time(nc_dimensions_dct), np.nan, np.float32
            )
            for idx_mem, file_path in enumerate(self.paths):
                timeless_panel_setup = panel_setup_i.derive(
                    {"dimensions": {"time": None}}
                )

                # Read the data from disk
                n_mem = len(self.paths)
                log(dbg=f"reading file ({idx_mem + 1}/{n_mem}): {file_path}")
                with nc4.Dataset(file_path, "r") as fi:
                    self._read_grid(fi)

                    # Read meta data at requested time steps
                    mdata_tss_i = self._collect_meta_data_tss(fi, panel_setups_req_time)
                    if idx_mem == 0:
                        self.mdata_tss = mdata_tss_i
                    elif mdata_tss_i != self.mdata_tss:
                        raise Exception("meta data differ between members")

                    if not self.config.dry_run:
                        # Read fields for all members at all time steps
                        fld_time_i = self._read_member_fields_over_time(
                            fi, timeless_panel_setup
                        )
                    else:
                        fld_time_i = np.empty(fld_time_mem.shape[1:], np.float32)
                    fld_time_mem[idx_mem][:] = fld_time_i[:]

            # Compute single field from all ensemble members
            fld_time: np.ndarray = self._reduce_ensemble(
                fld_time_mem, panel_setup_i, ts_hrs
            )

            # Compute some statistics across all time steps
            time_stats = FieldTimeProperties(fld_time)

            # Create Field objects at requested time steps
            for panel_setup_req_time, mdata_req_time in zip(
                panel_setups_req_time, self.mdata_tss
            ):
                time_idx: int = panel_setup_req_time.dimensions.time
                field = Field(
                    fld=fld_time[time_idx],
                    lat=self.lat,
                    lon=self.lon,
                    mdata=mdata_req_time,
                    time_props=time_stats,
                    panel_setup=panel_setup_req_time,
                    model_setup=plot_setup.model,
                )
                if time_idx not in field_lst_by_ts:
                    field_lst_by_ts[time_idx] = []
                field_lst_by_ts[time_idx].append(field)

        # Create one FieldGroup per plot (possibly with multiple outfiles)
        field_groups: List[FieldGroup] = []
        for time_idx, field_lst_i in field_lst_by_ts.items():
            # Derive plot setup for given time step
            panel_params = {"dimensions": {"time": time_idx}}
            plot_setup_i = plot_setup.derive(
                {"panels": [panel_params] * len(plot_setup.panels)}
            )
            group_attrs = FieldGroupAttrs(
                raw_path=plot_setup_i.files.input,
                paths=self.public_paths,
                ens_member_ids=plot_setup.model.ens_member_id,
            )
            field_group = FieldGroup(
                field_lst_i, plot_setup=plot_setup_i, attrs=group_attrs
            )
            field_groups.append(field_group)
        return field_groups

    def _read_member_fields_over_time(
        self, fi: nc4.Dataset, timeless_panel_setup: PlotPanelSetup
    ) -> np.ndarray:
        """Read field over all time steps for each member."""
        fld_time_lst = []
        plot_variable = timeless_panel_setup.plot_variable
        for dimensions in timeless_panel_setup.dimensions.decompress():
            fld_time_lst.append(
                self._read_fld_over_time(fi, dimensions, timeless_panel_setup.integrate)
            )
        if plot_variable == "affected_area":
            shapes = [fld.shape for fld in fld_time_lst]
            if len(set(shapes)) != 1:
                raise Exception(f"field shapes differ: {shapes}")
            fld_time = (np.array(fld_time_lst) > AFFECTED_AREA_THRESHOLD).any(axis=0)
        else:
            fld_time = merge_fields(fld_time_lst)
        if self.fixer and self.model_setup.name == "IFS-HRES":
            # Note: IFS-HRES-EU is not global, so doesn't need this fix
            self.fixer.fix_global_grid(self.lon, fld_time)
        if self.config.add_ts0:
            fld_time = self._add_ts0_to_fld_time(fld_time)
        if plot_variable in ["cloud_arrival_time", "cloud_departure_time"]:
            ts_hrs = self.get_temp_res_hrs(fi)
            cloud = Cloud(mask=np.array(fld_time) > CLOUD_THRESHOLD, ts=ts_hrs)
            if plot_variable == "cloud_arrival_time":
                fld_time = cloud.arrival_time()
            else:
                fld_time = cloud.departure_time()
        return fld_time

    # pylint: disable=R0914  # too-many-locals
    def _collect_meta_data_tss(
        self,
        fi: nc4.Dataset,
        panel_setups_req_time: PlotPanelSetupGroup,
    ) -> List[MetaData]:
        """Collect time-step-specific data meta data."""
        mdata_lst: List[MetaData] = []
        for panel_setup in panel_setups_req_time:
            mdata_i_lst: List[MetaData] = []
            for dimensions in panel_setup.dimensions.decompress():
                mdata_ij: MetaData = MetaData.collect(
                    fi,
                    self.model_setup,
                    dimensions,
                    plot_variable=panel_setup.plot_variable,
                    integrate=panel_setup.integrate,
                    add_ts0=self.config.add_ts0,
                )
                mdata_i_lst.append(mdata_ij)
            mdata_i = mdata_i_lst[0].merge_with(mdata_i_lst[1:])
            if self.fixer:
                # Fix some known issues with the NetCDF input data
                model_name = self.model_setup.name
                self.fixer.fix_meta_data(
                    model_name,
                    panel_setup.plot_variable,
                    panel_setup.integrate,
                    mdata_i,
                )
            mdata_lst.append(mdata_i)
        return mdata_lst

    def _get_shape_mem_time(
        self, raw_dimensions: Mapping[str, Mapping[str, Any]]
    ) -> Tuple[int, int, int, int]:
        """Get the shape of an array of fields across members and time steps."""
        renamed_dims = self._renamed_dims()
        if self.config.dry_run:
            nlat = 1
            nlon = 1
        else:
            nlat = raw_dimensions[renamed_dims.get("lat", "lat")]["size"]
            nlon = raw_dimensions[renamed_dims.get("lon", "lon")]["size"]
        nts = raw_dimensions[renamed_dims.get("time", "time")]["size"]
        n_mem = len(self.paths)
        self.lat = np.full((nlat,), np.nan)
        self.lon = np.full((nlon,), np.nan)
        return (n_mem, nts, nlat, nlon)

    def _read_grid(self, fi: nc4.Dataset) -> None:
        """Read and prepare grid variables."""
        renamed_dims = self._renamed_dims()
        if self.config.dry_run:
            lat = np.zeros(1, np.float32)
            lon = np.zeros(1, np.float32)
        else:
            lat = fi.variables[renamed_dims.get("lat", "lat")][:]
            lon = fi.variables[renamed_dims.get("lon", "lon")][:]
        time = fi.variables[renamed_dims.get("time", "time")][:]
        time = self._prepare_time(fi, time)
        self.lat = lat
        self.lon = lon
        self.time = time

    def _prepare_time(self, fi: nc4.Dataset, time: np.ndarray) -> np.ndarray:
        if self.config.add_ts0:
            dts = time[1] - time[0]
            ts0 = time[0] - dts
            time = np.r_[ts0, time]
        # Convert seconds to hours
        renamed_dims = self._renamed_dims()
        time_unit = fi.variables[renamed_dims.get("time", "time")].units
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
    # pylint: disable=R1702  # too-many-nested-blocks (>5)
    def _reduce_ensemble(
        self, fld_time_mem: np.ndarray, panel_setup: PlotPanelSetup, ts_hrs: float
    ) -> np.ndarray:
        """Reduce the ensemble to a single field (time, lat, lon)."""
        if len(self.paths) == 1 or self.config.dry_run:
            # Create copy of subarray, otherwise whole array kept in memory
            return fld_time_mem[0].copy()

        ens_params = panel_setup.ens_params
        ens_variable = panel_setup.ens_variable

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
            if ens_params.pctl is None:
                raise Exception("ens_params.pctl is None")
            fld_time = np.percentile(fld_time_mem, ens_params.pctl, axis=0)
        elif ens_variable == "probability":
            if ens_params.thr is None:
                raise Exception("ens_params.thr is None")
            fld_time = ensemble_probability(
                fld_time_mem, ens_params.thr, ens_params.thr_type
            )
        elif ens_variable.startswith("cloud_"):
            if ens_params.thr is None:
                raise Exception("ens_params.thr is None")
            if ens_params.mem_min is None:
                raise Exception("ens_params.mem_min is None")
            cloud = EnsembleCloud(
                mask=fld_time_mem > ens_params.thr,
                mem_min=ens_params.mem_min,
                ts=ts_hrs,
            )
            if ens_variable == "cloud_arrival_time":
                fld_time = cloud.arrival_time()
            elif ens_variable == "cloud_departure_time":
                fld_time = cloud.departure_time()
            else:
                raise NotImplementedError("ens_variable", ens_variable)
        else:
            raise NotImplementedError("ens_variable", ens_variable)
        return fld_time

    def _renamed_dims(self) -> Dict[str, str]:
        """Model-specific dimension names."""
        if self.model_setup.name in [
            "COSMO-2",
            "COSMO-1",
            "COSMO-E",
            "COSMO-2E",
            "COSMO-1E",
        ]:
            return {
                "lat": "rlat",
                "lon": "rlon",
            }
        elif self.model_setup.name in ["IFS-HRES", "IFS-HRES-EU"]:
            return {
                "lat": "latitude",
                "lon": "longitude",
                "level": "height",
            }
        raise NotImplementedError(
            f"dimension names for model '{self.model_setup.name}'"
        )

    # pylint: disable=R0912,R0914  # too-many-branches, too-many-locals
    def _read_fld_over_time(
        self, fi: nc4.Dataset, dimensions: Dimensions, integrate: bool
    ) -> np.ndarray:
        """Read a 2D field at all time steps from disk."""
        # Indices of field along NetCDF dimensions
        renamed_dims = self._renamed_dims()
        dim_idcs_by_name = {
            renamed_dims.get("lat", "lat"): slice(None),
            renamed_dims.get("lon", "lon"): slice(None),
        }
        for dim_name in ["nageclass", "release", "time", "level"]:
            if dim_name == "release":
                # SR_TMP < TODO cleaner colution
                idcs = getattr(dimensions, dim_name)
                for var in fi.variables.values():
                    if var.name.startswith("spec"):
                        dim_name = var.dimensions[1]
                        break
                else:
                    raise Exception(
                        f"no variable 'spec*' found among {list(fi.variables)} in"
                        f" {fi.filepath()}"
                    )
                # SR_TMP >
            elif dim_name == "time":
                idcs = slice(None)
            else:
                idcs = getattr(dimensions, dim_name)
            dim_idcs_by_name[renamed_dims.get(dim_name, dim_name)] = idcs

        # Select variable in file
        assert dimensions.species_id is not None  # mypy
        # SR_TPM <
        variable = dimensions.variable
        if set(variable) == {"dry_deposition", "wet_deposition"}:
            variable = "dry_deposition"
        assert isinstance(variable, str)
        # SR_TPM >
        var_name = derive_variable_name(
            model=self.model_setup.name,
            variable=variable,
            species_id=dimensions.species_id,
        )
        try:
            nc_var = fi.variables[var_name]
        except KeyError as e:
            if not self.config.missing_ok:
                raise Exception(f"missing variable '{var_name}'") from e
            shape = (
                fi.dimensions[renamed_dims.get("time", "time")].size,
                fi.dimensions[renamed_dims.get("lat", "lat")].size,
                fi.dimensions[renamed_dims.get("lon", "lon")].size,
            )
            return np.zeros(shape, np.float32)

        # Assemble indices for slicing
        indices: List[Any] = [None] * len(nc_var.dimensions)
        for dim_name, dim_idx in dim_idcs_by_name.items():
            # Get the index of the dimension for this variable
            try:
                idx = nc_var.dimensions.index(dim_name)
            except ValueError as e:
                # Potential issue: Dimension is not among the variable dimensions!
                if dim_idx in (None, 0):
                    continue  # Zero-index: We're good after all!
                raise Exception(
                    f"'{dim_name}' (#{dim_idx}) not among dimensions"
                    f" {list(nc_var.dimensions)} of variable '{var_name}' in"
                    f" {fi.filepath()}"
                ) from e

            # Check that the index along the dimension is valid
            if dim_idx is None:
                raise Exception(
                    f"value of dimension #{idx} '{dim_name}' (#{dim_idx}) of variable"
                    f" '{var_name}' is None in {fi.filepath()}"
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
            self.fixer.fix_nc_var_fld(fld, self.model_setup.name, var_ncattrs)

        return self._handle_time_integration(fi, fld, dimensions.variable, integrate)

    def _handle_time_integration(
        self, fi: nc4.Dataset, fld: np.ndarray, variable: str, integrate: bool
    ) -> np.ndarray:
        """Integrate or desintegrate the field over time."""
        if variable == "concentration":
            if integrate:
                # Integrate field over time
                dt_hr = self.get_temp_res_hrs(fi)
                return np.cumsum(fld, axis=0) * dt_hr
            else:
                # Field is already instantaneous
                return fld
        elif variable.endswith("_deposition"):
            if integrate:
                # Field is already time-integrated
                return fld
            else:
                # Revert time integration of field
                dt_hr = self.get_temp_res_hrs(fi)
                fld[1:] = (fld[1:] - fld[:-1]) / dt_hr
                return fld
        raise NotImplementedError("unknown variable", variable)

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

"""Meta data."""
# Standard library
import dataclasses as dc
import re
import warnings
from copy import copy
from copy import deepcopy
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from pprint import pformat
from typing import Any
from typing import cast
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Union

# Third-party
import netCDF4 as nc4
import numpy as np

# First-party
from srutils.dataclasses import cast_field_value
from srutils.dataclasses import dataclass_repr
from srutils.dataclasses import get_dataclass_fields
from srutils.datetime import datetime_range
from srutils.datetime import init_datetime
from srutils.dict import compress_multival_dicts
from srutils.format import sfmt

# Local
from ..setups.dimensions import Dimensions
from ..setups.model_setup import ModelSetup
from ..words import SYMBOLS
from .species import get_species
from .species import Species

CoreMetaDatumType = Union[int, float, str, datetime, timedelta]
MetaDatumType = Union[CoreMetaDatumType, Tuple[CoreMetaDatumType, ...]]


@overload
def format_meta_datum(
    value: MetaDatumType,
    unit: Optional[Union[str, Tuple[str, ...]]] = None,
    *,
    join_values: str = ...,
) -> str:
    ...


@overload
def format_meta_datum(
    value: None = None,
    unit: Union[str, Tuple[str, ...]] = ...,
    *,
    join_values: str = ...,
) -> str:
    ...


# pylint: disable=R0911  # too-many-return-statements
# pylint: disable=R0912  # too-many-branches
def format_meta_datum(value=None, unit=None, *, join_values=" / "):
    if value is None and unit is None:
        raise ValueError("value and unit cannot both be None")
    elif value is None and unit is not None:
        if isinstance(unit, tuple):
            raise NotImplementedError(f"multiple units without value: {unit}")
        else:
            return _format_unit(unit)
    else:
        assert value is not None  # mypy
        if unit is not None:
            return _format_meta_datum_with_unit(value, unit, join_values=join_values)
        if isinstance(value, tuple):
            # SR_TODO make sure this is covered by a test (it currently isn't)!
            return join_values.join([format_meta_datum(v) for v in value])
        elif isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M %Z")
        elif isinstance(value, timedelta):
            hours = int(value.total_seconds() / 3600)
            minutes = int((value.total_seconds() / 60) % 60)
            return f"{hours:d}:{minutes:02d}$\\,$h"
        elif isinstance(value, (float, int)):
            if 1e-5 < value < 1e6:
                return f"{value:g}"
            else:
                return f"{value:.2g}"
        else:
            return str(value)


def _format_meta_datum_with_unit(
    value: MetaDatumType,
    unit: Union[str, Tuple[str, ...]],
    *,
    join_values: str = " / ",
    join_unit: str = r"$\,$",
) -> str:
    if not isinstance(value, tuple):
        if isinstance(unit, tuple):
            raise ValueError(f"multiple units for single value: {value}, {unit}")
        # Single value, single unit
        value = (value,)
        unit = (unit,)
    elif not isinstance(unit, tuple):
        # Create a copy of the unit for each of the multiple values
        unit = tuple([unit] * len(value))
    elif len(value) != len(unit):
        raise ValueError(
            f"different number of values ({len(value)}) and units ({len(unit)})"
            f": {value}, {unit}"
        )
    kwargs = {"join_values": join_values}
    value_fmtd = tuple(map(lambda v: format_meta_datum(v, **kwargs), value))
    unit_fmtd = tuple(map(lambda u: _format_unit(format_meta_datum(u, **kwargs)), unit))
    return format_meta_datum(tuple(map(join_unit.join, zip(value_fmtd, unit_fmtd))))


def _format_unit(s: str) -> str:
    """Auto-format the unit by elevating superscripts etc."""
    s = str(s)
    old_new = [
        ("m-2", "m$^{-2}$"),
        ("m-3", "m$^{-3}$"),
        ("s-1", "s$^{-1}$"),
    ]
    for old, new in old_new:
        s = s.replace(old, new)
    return s


@dc.dataclass
class MetaData:
    release: "ReleaseMetaData"
    simulation: "SimulationMetaData"
    species: "SpeciesMetaData"
    variable: "VariableMetaData"

    def __repr__(self) -> str:
        return dataclass_repr(self, fmt=lambda obj: dataclass_repr(obj, nested=1))

    def dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            field: getattr(self, field).dict()
            for field in get_dataclass_fields(type(self))
        }

    # pylint: disable=R0914  # too-many-locals
    def merge_with(self, others: Collection["MetaData"]) -> "MetaData":
        """Merge this ``MetaData`` instance with one or more others.

        The nested functions are former methods that have been moved in here in
        order to eventually eliminate them.

        """
        unique_others: List["MetaData"] = []
        for other in others:
            if other != self and not any(
                unique_other == other for unique_other in unique_others
            ):
                unique_others.append(other)
        others = unique_others

        if not others:
            return type(self)(
                release=self.release,
                simulation=self.simulation,
                variable=self.variable,
                species=self.species,
            )

        # Extract parameter values from objects
        grouped_params: Dict[str, List[Dict[str, Any]]] = {
            group: [getattr(obj, group).dict() for obj in [self] + list(others)]
            for group in get_dataclass_fields(type(self))
        }

        #
        # The meta data may differ in two aspects:
        # - the input variable, specifically the vertical level range; and
        # - the release meta data, i.e., which substance has been released.
        #
        # Release meta data values (for given set of input variable meta data)
        # are retained in a tuple (e.g., `"name": ("Cs-137", "I-131a")`).
        #
        # Adjacent level ranges, on the other hand, are merged (provided the
        # release meta data don't differ, otherwise an exception is raised).
        # This reduces the number of parameters in the merged MetaData object.
        #

        # Group meta data indices by level ranges
        params_lst = grouped_params["variable"]
        level_ranges: Dict[Tuple[float, float], List[int]] = {}
        for idx, params in enumerate(params_lst):
            level_range = (params["bottom_level"], params["top_level"])
            if level_range not in level_ranges:
                level_ranges[level_range] = []
            level_ranges[level_range].append(idx)
        level_ranges_lst = sorted(level_ranges.items())

        # SR_TMP < TODO implement non-adjacent level ranges (with test)
        # Make sure that multiple level ranges are adjacent
        bottoms = [bottom for (bottom, _), _ in level_ranges_lst]
        tops = [top for (_, top), _ in level_ranges_lst]
        if bottoms[1:] != tops[:-1]:
            raise NotImplementedError("non-adjacent level ranges: {list(level_ranges)}")
        # SR_TMP >

        # SR_TMP < TODO implement different meta data per level (if required)
        # Ensure the number of meta data per level range are all equal
        if len(set(map(len, level_ranges.values()))) != 1:
            raise NotImplementedError(
                f"number of meta data differ betwenen level ranges: {level_ranges}"
            )
        # SR_TMP >

        params_ref = None
        for idx, params in enumerate(grouped_params["variable"][1:]):
            params = {
                **grouped_params["variable"][0],
                "bottom_level": None,
                "top_level": None,
            }
            if idx == 0:
                params_ref = params
            elif params != params_ref:
                raise Exception(
                    f"variable meta data differ:\n\n{pformat(params)}"
                    f"\n\n!=\n\n{params_ref}\n\n(idx={idx})"
                )
        variable_params: Dict[str, Any] = {
            **grouped_params["variable"][0],
            "bottom_level": level_ranges_lst[0][0][0],
            "top_level": level_ranges_lst[-1][0][1],
        }
        variable_mdata = VariableMetaData(**variable_params)

        # For each merged levels range, collect all non-variable meta data
        merged_params_lst: List[Dict[str, Dict[str, Any]]] = []
        for idx_idx in range(len(level_ranges_lst[0][1])):
            merged_params_lst.append({})
            idx_ref = level_ranges_lst[0][1][idx_idx]
            for group, params_lst in grouped_params.items():
                if group == "variable":
                    continue
                params_ref = params_lst[idx_ref]
                for level_range, idcs in level_ranges_lst[1:]:
                    idx = idcs[idx_idx]
                    params = params_lst[idx]
                    # SR_TMP < TODO implement support for reordering (with test)
                    # Make sure for each range the other meta data are identical
                    if params != params_ref:
                        raise Exception(
                            f"{group} meta data differ:"
                            f"\n\n{pformat(params)}\n\n!=\n\n{pformat(params_ref)}"
                            f"\n\n(idx_idx={idx_idx}, idx_ref={idx_ref}, idx={idx}"
                            f", group='{group}')"
                        )
                    # SR_TMP >
                if group not in merged_params_lst[-1]:
                    merged_params_lst[-1][group] = params_ref

        # Simulation meta data
        simulation_params = compress_multival_dicts(
            [params["simulation"] for params in merged_params_lst],
            tuple,
        )
        simulation_mdata = SimulationMetaData(**simulation_params)

        # Species meta data
        species_params = compress_multival_dicts(
            [params["species"] for params in merged_params_lst], tuple, skip_compr=True
        )
        species_mdata = SpeciesMetaData(**species_params)

        # Release meta data
        release_params = compress_multival_dicts(
            [params["release"] for params in merged_params_lst],
            tuple,
            skip_compr_keys=["mass", "mass_unit", "rate", "rate_unit"],
            expect_equal=True,
        )
        release_mdata = ReleaseMetaData(**release_params)

        return type(self)(
            release=release_mdata,
            simulation=simulation_mdata,
            variable=variable_mdata,
            species=species_mdata,
        )

    def __deepcopy__(self, memo):
        return type(self)(
            release=copy(self.release),
            simulation=copy(self.simulation),
            variable=copy(self.variable),
            species=deepcopy(self.species, memo),
        )

    @classmethod
    def collect(
        cls,
        fi: nc4.Dataset,
        model_setup: ModelSetup,
        dimensions: Dimensions,
        *,
        plot_variable: str,
        integrate: bool,
        add_ts0: bool = True,
    ) -> "MetaData":
        """Collect meta data from file."""
        assert isinstance(dimensions.variable, str)  # SR_DBG
        return cls(
            release=ReleaseMetaData.from_file(fi, dimensions),
            simulation=SimulationMetaData.from_file(
                fi, model_setup, dimensions, integrate, add_ts0
            ),
            variable=VariableMetaData.from_file(
                fi, model_setup, dimensions, plot_variable
            ),
            species=SpeciesMetaData.from_file(fi, model_setup, dimensions),
        )


def getncattr(nc_obj: Union[nc4.Dataset, nc4.Variable], attr: str) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="tostring() is deprecated. Use tobytes() instead.",
        )
        return nc_obj.getncattr(attr)


class _MetaDataBase:
    def __repr__(self) -> str:
        return dataclass_repr(self)

    def dict(self) -> Dict[str, Any]:
        dct: Dict[str, Any] = {}
        for field in get_dataclass_fields(type(self)):
            dct[field] = getattr(self, field)
        return dct


@dc.dataclass(repr=False)
class VariableMetaData(_MetaDataBase):
    unit: str
    bottom_level: float
    top_level: float
    level_unit: str

    @classmethod
    def from_file(
        cls,
        fi: nc4.Dataset,
        model_setup: ModelSetup,
        dimensions: Dimensions,
        plot_variable: str,
    ) -> "VariableMetaData":
        if plot_variable == "affected_area":
            unit = ""
        elif plot_variable in [
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            unit = "h"
        else:
            assert dimensions.species_id is not None  # mypy
            var_name = derive_variable_name(
                model=model_setup.name,
                variable=dimensions.variable,
                species_id=dimensions.species_id,
            )
            try:
                var = fi.variables[var_name]
            except KeyError:
                unit = "N/A"
            else:
                unit = getncattr(var, "units")
        idx: int
        if dimensions.level is None:
            level_unit = ""
            level_bot = -1.0
            level_top = -1.0
        else:
            idx = dimensions.level
            try:  # SR_TMP IFS
                var = fi.variables["level"]
            except KeyError:  # SR_TMP IFS
                var = fi.variables["height"]  # SR_TMP IFS
            try:
                level_bot = 0.0 if idx == 0 else float(var[idx - 1])
            except IndexError as e:
                raise Exception(
                    f"index of bottom level out of bounds: {idx} > {var.size - 1}"
                ) from e
            try:
                level_top = float(var[idx])
            except IndexError as e:
                raise Exception(
                    f"index of top level out of bounds: {idx} > {var.size - 1}"
                ) from e
            level_unit = getncattr(var, "units")
        return cls(
            unit=unit,
            bottom_level=level_bot,
            top_level=level_top,
            level_unit=level_unit,
        )


# pylint: disable=R0902  # too-many-instance-attributes
# pylint: disable=R0914  # too-many-instance-locals (>15)
@dc.dataclass(repr=False)
class SimulationMetaData(_MetaDataBase):
    start: datetime
    end: datetime
    now: datetime
    now_rel: timedelta
    base_time: datetime
    lead_time: timedelta
    reduction_start: datetime
    reduction_start_rel: timedelta
    time_steps: List[int]
    grid_is_rotated: bool
    grid_north_pole_lat: float
    grid_north_pole_lon: float

    @overload
    def get_duration(self, unit: None = None) -> timedelta:
        ...

    @overload
    def get_duration(self, unit: str) -> float:
        ...

    def get_duration(self, unit=None):
        if unit is None:
            return self.end - self.start
        elif unit in ["h", "hours"]:
            return self.get_duration().total_seconds() / 3600
        else:
            choices = [None, "h", "hours"]
            raise ValueError(
                f"unit {sfmt(unit)} not among [{', '.join(map(sfmt, choices))}]"
            )

    # pylint: disable=R0913  # too-many-arguments (>5)
    @classmethod
    def from_file(
        cls,
        fi: nc4.Dataset,
        model_setup: ModelSetup,
        dimensions: Dimensions,
        integrate: bool,
        add_ts0: bool,
    ) -> "SimulationMetaData":

        # Start and end timesteps of simulation
        start = init_datetime(
            str(getncattr(fi, "ibdate")) + str(getncattr(fi, "ibtime"))
        )
        end = init_datetime(str(getncattr(fi, "iedate")) + str(getncattr(fi, "ietime")))
        step = int(getncattr(fi, "loutstep"))

        # Formatted time steps
        time_steps: List[int] = datetime_range(
            start=start,
            end=end,
            step=step,
            convert=int,
            fmt="%Y%m%d%H%M",
        )

        # Current time step and start time step of current integration period
        collector = TimeStepMetaDataCollector(
            fi, dimensions, integrate, add_ts0=add_ts0
        )
        now = collector.now()
        reduction_start = collector.integration_start()
        now_rel: timedelta = collector.now_rel()
        reduction_start_rel = collector.integration_start_rel()

        base_time = init_datetime(cast(int, model_setup.base_time))
        lead_time = now - base_time

        # Grid
        try:
            var = fi.variables["rotated_pole"]
        except KeyError:
            grid_is_rotated = False
            grid_north_pole_lat = 90.0
            grid_north_pole_lon = 180.0
        else:
            grid_is_rotated = True
            grid_north_pole_lat = getncattr(var, "grid_north_pole_latitude")
            grid_north_pole_lon = getncattr(var, "grid_north_pole_longitude")

        return cls(
            start=start,
            end=end,
            now=now,
            now_rel=now_rel,
            base_time=base_time,
            lead_time=lead_time,
            time_steps=time_steps,
            reduction_start=reduction_start,
            reduction_start_rel=reduction_start_rel,
            grid_is_rotated=grid_is_rotated,
            grid_north_pole_lat=grid_north_pole_lat,
            grid_north_pole_lon=grid_north_pole_lon,
        )


@dc.dataclass(repr=False)
# pylint: disable=R0902  # too-many-instance-attrbutes
class ReleaseMetaData(_MetaDataBase):
    duration: timedelta
    duration_unit: str
    end_rel: timedelta
    height: float
    height_unit: str
    lat: float
    lon: float
    mass: Union[float, Tuple[float, ...]]
    mass_unit: str
    rate: Union[float, Tuple[float, ...]]
    rate_unit: str
    raw_site_name: str
    site_name: str
    start_rel: timedelta

    @classmethod
    def from_file(cls, fi: nc4.Dataset, dimensions: Dimensions) -> "ReleaseMetaData":
        """Read information on a release from open file."""
        raw = RawReleaseMetaData.from_file(fi, dimensions)
        raw_site_name = raw.site
        site_name = (
            raw_site_name.replace("ae", SYMBOLS["ae"].s)
            .replace("oe", SYMBOLS["oe"].s)
            .replace("ue", SYMBOLS["ue"].s)
        )
        start_rel = raw.rel_start
        end_rel = raw.rel_end
        duration = end_rel - start_rel
        duration_unit = "s"  # SR_HC
        assert raw.zbot_unit == raw.ztop_unit
        height_unit = raw.zbot_unit
        mass = raw.ms_parts
        mass_unit = "Bq"  # SR_HC
        return cls(
            duration=duration,
            duration_unit=duration_unit,
            end_rel=end_rel,
            height=np.mean([raw.zbot, raw.ztop]),
            height_unit=height_unit,
            lat=np.mean([raw.lllat, raw.urlat]),
            lon=np.mean([raw.lllon, raw.urlon]),
            mass=mass,
            mass_unit=mass_unit,
            rate=mass / duration.total_seconds() if duration else np.nan,
            rate_unit=f"{mass_unit} {duration_unit}-1",
            raw_site_name=raw_site_name,
            site_name=site_name,
            start_rel=start_rel,
        )


@dc.dataclass(repr=False)
# pylint: disable=R0902  # too-many-instance-attributes
class SpeciesMetaData(_MetaDataBase):
    name: Union[str, Tuple[str, ...]]
    half_life: Union[float, Tuple[float, ...]]
    half_life_unit: Union[str, Tuple[str, ...]]
    deposition_velocity: Union[float, Tuple[float, ...]]
    deposition_velocity_unit: Union[str, Tuple[str, ...]]
    sedimentation_velocity: Union[float, Tuple[float, ...]]
    sedimentation_velocity_unit: Union[str, Tuple[str, ...]]
    washout_coefficient: Union[float, Tuple[float, ...]]
    washout_coefficient_unit: Union[str, Tuple[str, ...]]
    washout_exponent: Union[float, Tuple[float, ...]]

    @classmethod
    def from_file(
        cls,
        fi: nc4.Dataset,
        model_setup: ModelSetup,
        dimensions: Dimensions,
    ) -> "SpeciesMetaData":
        name: str
        assert dimensions.species_id is not None  # mypy
        var_name = derive_variable_name(
            model=model_setup.name,
            variable=dimensions.variable,
            species_id=dimensions.species_id,
        )
        try:
            var: nc4.Variable = fi.variables[var_name]
            name = getncattr(var, "long_name")
        except (KeyError, AttributeError):
            if model_setup.name.startswith("IFS"):
                name = cls._get_species_name_ifs(fi, var_name)
            elif dimensions.variable.endswith("_deposition"):
                # Deposition field may be missing
                return cls.from_file(
                    fi,
                    model_setup,
                    dimensions.derive({"variable": "concentration"}),
                )
            else:
                name = "N/A"
        else:
            name = name.split("_")[0]
        species: Species = get_species(name=name)
        return cls(
            name=species.name,
            half_life=species.half_life.value,
            half_life_unit=cast(str, species.half_life.unit),
            deposition_velocity=max([0.0, species.deposition_velocity.value]),
            deposition_velocity_unit=cast(str, species.deposition_velocity.unit),
            sedimentation_velocity=max([0.0, species.sedimentation_velocity.value]),
            sedimentation_velocity_unit=cast(str, species.sedimentation_velocity.unit),
            washout_coefficient=species.washout_coefficient.value,
            washout_coefficient_unit=cast(str, species.washout_coefficient.unit),
            washout_exponent=species.washout_exponent.value,
        )

    @staticmethod
    def _get_species_name_ifs(fi: nc4.Dataset, var_name: str) -> str:
        """Get the species name from an IFS simulation.

        In the IFS NetCDF files, the deposition variables are missing the basic
        meta data on species, like "long_name". Therefore, try to obtain the
        name from the activity variable of the same species.

        """
        if not var_name.startswith("DD_") and not var_name.startswith("WD_"):
            return "N/A"
        alt_name = f"{var_name[3:]}_mr"
        try:
            alt_var = fi.variables[alt_name]
            name = getncattr(alt_var, "long_name")
        except (KeyError, AttributeError):
            return "N/A"
        else:
            return name.split("_")[0]


@dc.dataclass
class RawReleaseMetaData:
    age_id: int
    kind: str
    lllat: float
    lllon: float
    ms_parts: float
    n_parts: int
    rel_end: timedelta
    rel_start: timedelta
    site: str
    urlat: float
    urlon: float
    zbot: float
    zbot_unit: str
    ztop: float
    ztop_unit: str

    @classmethod
    def create(cls, params: Dict[str, Any]) -> "RawReleaseMetaData":
        params = {
            param: cast_field_value(
                cls,
                param,
                value,
                auto_wrap=True,
                bool_mode="intuitive",
                timedelta_unit="seconds",
                unpack_str=False,
            )
            for param, value in params.items()
        }
        return cls(**params)

    # pylint: disable=R0914  # too-many-locals
    @classmethod
    def from_file(cls, fi: nc4.Dataset, dimensions: Dimensions) -> "RawReleaseMetaData":
        """Read information on a release from open file."""
        # Fetch release
        assert dimensions.release is not None  # mypy
        idx_release = dimensions.release

        # Fetch species_id
        assert dimensions.species_id is not None  # mypy
        n_species = fi.dimensions["numspec"].size
        if dimensions.species_id < 1:
            raise Exception(f"invalid species_id: {dimensions.species_id} <= 0")
        elif dimensions.species_id > n_species:
            raise Exception(
                f"invalid species_id: {dimensions.species_id} > {n_species}"
                f" (no. species in {Path(fi.filepath()).name})"
            )
        idx_spec = dimensions.species_id - 1

        # Fetch nageclass
        assert dimensions.nageclass is not None  # mypy
        idx_age = dimensions.nageclass

        var_name: str = "RELCOM"  # SR_HC TODO un-hardcode
        var = fi.variables[var_name]

        # Check index against no. release point and set it if necessary
        n = var.shape[0]
        if n == 0:
            raise ValueError(f"file '{fi.name}': no release points ('{var_name}')")
        elif n == 1:
            if idx_release is None:
                idx_release = 0
        elif n > 1:
            if idx_release is None:
                raise ValueError(
                    f"file '{fi.name}': idx is None despite {n} release points"
                )
        assert idx_release is not None  # mypy
        if idx_release < 0 or idx_release >= n:
            raise ValueError(
                f"file '{fi.name}': invalid index {idx_release} for {n} release points"
            )

        # Name: convert from byte character array
        site = (
            var[idx_release][~var[idx_release].mask].tobytes().decode("utf-8").rstrip()
        )

        # Other attributes
        key_pairs = [
            ("age_id", "LAGE"),
            ("kind", "RELKINDZ"),
            ("lllat", "RELLAT1"),
            ("lllon", "RELLNG1"),
            ("ms_parts", "RELXMASS"),
            ("n_parts", "RELPART"),
            ("rel_end", "RELEND"),
            ("rel_start", "RELSTART"),
            ("urlat", "RELLAT2"),
            ("urlon", "RELLNG2"),
            ("zbot", "RELZZ1"),
            ("ztop", "RELZZ2"),
        ]
        store_units = ["zbot", "ztop"]
        params = {"site": site}
        for key_out, key_in in key_pairs:
            var = fi.variables[key_in]
            # SR_TMP < TODO more flexible solution w/o hardcoding so much
            if var.dimensions == ("numpoint",):
                params[key_out] = var[idx_release].tolist()
            elif var.dimensions == ("numspec", "numpoint"):
                params[key_out] = var[idx_spec, idx_release].tolist()
            elif var.dimensions == ("nageclass",):
                params[key_out] = var[idx_age].tolist()
            else:
                raise NotImplementedError(
                    f"dimensions {var.dimensions} (variable {var.name})"
                )
            # SR_TMP >
            if key_out in store_units:
                unit = getncattr(fi.variables[key_in], "units")
                params[f"{key_out}_unit"] = unit
        return cls.create(params)


class TimeStepMetaDataCollector:
    """Collect time step meta data from file."""

    def __init__(
        self,
        fi: nc4.Dataset,
        dimensions: Dimensions,
        integrate: bool,
        *,
        add_ts0: bool = True,
    ) -> None:
        """Create an instance of ``TimeStepMetaDataCollector``."""
        self.fi = fi
        self.dimensions = dimensions
        self.integrate = integrate
        self.add_ts0 = add_ts0

    def start(self) -> datetime:
        """Compute the time step when the simulation started."""
        var = self.fi.variables["time"]
        rx = re.compile(
            r"seconds since "
            r"(?P<year>[12][0-9][0-9][0-9])-"
            r"(?P<month>[01][0-9])-"
            r"(?P<day>[0-3][0-9]) "
            r"(?P<hour>[0-2][0-9]):"
            r"(?P<minute>[0-6][0-9])"
        )
        match = rx.match(var.units)
        if not match:
            raise Exception(f"cannot extract start from units '{var.units}'")
        return datetime(
            year=int(match["year"]),
            month=int(match["month"]),
            day=int(match["day"]),
            hour=int(match["hour"]),
            minute=int(match["minute"]),
            tzinfo=timezone.utc,
        )

    def now(self) -> datetime:
        """Return the current time step."""
        return self.start() + self.now_rel()

    def now_rel(self) -> timedelta:
        """Return the time since start."""
        var = self.fi.variables["time"]
        idx = self.time_step_idx()
        if idx < 0:
            return timedelta(0)
        return timedelta(seconds=int(var[idx]))

    def integration_start(self) -> datetime:
        """Return start time step of integration period."""
        return self.now() - self.integration_duration()

    def integration_start_rel(self) -> timedelta:
        """Return time between starts of simulation and integration period."""
        return self.integration_start() - self.start()

    def integration_duration(self) -> timedelta:
        """Compute timestep delta of integration period."""
        if self.integrate:
            return self.now_rel()
        n = self.time_step_idx() + 1
        if n == 0:
            return timedelta(0)
        return self.now_rel() / n

    def time_step_idx(self) -> int:
        """Index of current time step of current field."""
        assert self.dimensions.time is not None  # mypy
        if self.add_ts0:
            return self.dimensions.time - 1
        return self.dimensions.time


def derive_variable_name(model: str, variable: str, species_id: int) -> str:
    """Derive the NetCDF variable name given some attributes."""
    cosmo_models = ["COSMO-2", "COSMO-1", "COSMO-E", "COSMO-2E", "COSMO-1E"]
    ifs_models = ["IFS-HRES", "IFS-HRES-EU"]
    if variable == "concentration":
        if model in cosmo_models:
            return f"spec{species_id:03d}"
        elif model in ifs_models:
            return f"spec{species_id:03d}_mr"
        else:
            raise ValueError("unknown model", model)
    elif variable.endswith("_deposition"):
        prefix = {"wet": "WD", "dry": "DD"}[variable[:3]]
        return f"{prefix}_spec{species_id:03d}"
    raise ValueError(f"unknown variable '{variable}'")


def read_dimensions(file_handle: nc4.Dataset, add_ts0: bool = True) -> Dict[str, Any]:
    """Read dimensions from a NetCDF file.

    Args:
        file_handle: Open NetCDF file handle.

        add_ts0 (optional): Insert an additional time step 0 in the beginning
            with empty fields, given that the first data time step may not
            correspond to the beginning of the simulation, but constitute the
            sum over the first few hours of the simulation.

    """
    dimensions: Dict[str, Any] = {}
    for dim_handle in file_handle.dimensions.values():
        dimensions[dim_handle.name] = {
            "name": dim_handle.name,
            "size": dim_handle.size,
        }

    if add_ts0:
        dimensions["time"]["size"] += 1

    return dimensions


def read_time_steps(file_handle: nc4.Dataset) -> List[str]:
    """Derive the formatted time steps from the NetCDF global attributes.

    Args:
        file_handle: Open NetCDF file handle.

    """
    attrs_select: List[str] = ["ibdate", "ibtime", "iedate", "ietime", "loutstep"]
    attrs_try_select: List[str] = []
    ncattrs: Dict[str, Any] = {}
    for attr in attrs_select:
        ncattrs[attr] = file_handle.getncattr(attr)
    for attr in attrs_try_select:
        try:
            ncattrs[attr] = file_handle.getncattr(attr)
        except AttributeError:
            continue
    return datetime_range(
        start=ncattrs["ibdate"] + ncattrs["ibtime"],
        end=ncattrs["iedate"] + ncattrs["ietime"],
        step=ncattrs["loutstep"],
        convert=str,
        fmt="%Y%m%d%H%M",
    )


def read_species_ids(file_handle: nc4.Dataset) -> Tuple[int, ...]:
    """Derive the species ids from the NetCDF variable names.

    Args:
        file_handle: Open NetCDF file handle.

    """
    rx = re.compile(r"\A([WD]D_)?spec(?P<species_id>[0-9][0-9][0-9])(_mr)?\Z")
    species_ids = set()
    for var_name in file_handle.variables:
        match = rx.match(var_name)
        if match:
            species_id = int(match.group("species_id"))
            species_ids.add(species_id)
    if not species_ids:
        raise Exception("could not identify species ids")
    return tuple(sorted(species_ids))

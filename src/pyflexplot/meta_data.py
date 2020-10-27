# pylint: disable=C0302  # too-many-lines (>1000)
"""
Meta data.

Note that these meta data should eventually be merged with the raw ones in
module ``pyflexplot.nc_meta_data`` because the two very different data
structures serve very similar purposes in parallel.

"""
# Standard library
import re
import warnings
from copy import copy
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
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
from pydantic import BaseModel

# First-party
from srutils.dataclasses import dataclass_repr
from srutils.dataclasses import get_dataclass_fields
from srutils.dict import compress_multival_dicts

# Local
from .setup import Setup
from .species import get_species
from .species import Species
from .utils.datetime import init_datetime

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


@dataclass
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
        """Merge this `MetaData` instance with one or more others.

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

        # Variable meta data: merge across adjacent level ranges
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
        cls, fi: nc4.Dataset, setup: Setup, *, add_ts0: bool = False
    ) -> "MetaData":
        """Collect meta data from file."""
        return cls(
            release=ReleaseMetaData.from_file(fi, setup),
            simulation=SimulationMetaData.from_file(fi, setup, add_ts0),
            variable=VariableMetaData.from_file(fi, setup),
            species=SpeciesMetaData.from_file(fi, setup),
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


@dataclass
class VariableMetaData(_MetaDataBase):
    unit: str
    bottom_level: float
    top_level: float
    level_unit: str

    @classmethod
    def from_file(cls, fi: nc4.Dataset, setup: Setup) -> "VariableMetaData":
        if setup.core.input_variable == "affected_area":
            unit = ""
        else:
            var_name = nc_var_name(setup, setup.model)
            try:
                var = fi.variables[var_name]
            except KeyError:
                unit = "N/A"
            else:
                unit = getncattr(var, "units")
        idx: int
        if setup.core.dimensions.level is None:
            level_unit = ""
            level_bot = -1.0
            level_top = -1.0
        else:
            idx = setup.core.dimensions.level
            try:  # SR_TMP IFS
                var = fi.variables["level"]
            except KeyError:  # SR_TMP IFS
                var = fi.variables["height"]  # SR_TMP IFS
            level_bot = 0.0 if idx == 0 else float(var[idx - 1])
            level_top = float(var[idx])
            level_unit = getncattr(var, "units")
        return cls(
            unit=unit,
            bottom_level=level_bot,
            top_level=level_top,
            level_unit=level_unit,
        )


# pylint: disable=R0902  # too-many-instance-attributes
@dataclass
class SimulationMetaData(_MetaDataBase):
    start: datetime
    end: datetime
    now: datetime
    now_rel: timedelta
    base_time: datetime
    lead_time: timedelta
    reduction_start: datetime
    reduction_start_rel: timedelta

    @classmethod
    def from_file(
        cls, fi: nc4.Dataset, setup: Setup, add_ts0: bool
    ) -> "SimulationMetaData":

        # Start and end timesteps of simulation
        start = init_datetime(getncattr(fi, "ibdate") + getncattr(fi, "ibtime"))
        end = init_datetime(getncattr(fi, "iedate") + getncattr(fi, "ietime"))

        # Current time step and start time step of current integration period
        collector = TimeStepMetaDataCollector(fi, setup, add_ts0=add_ts0)
        now = collector.now()
        reduction_start = collector.reduction_start()
        now_rel: timedelta = collector.now_rel()
        reduction_start_rel = collector.reduction_start_rel()

        base_time = init_datetime(cast(int, setup.base_time))
        lead_time = now - base_time

        return cls(
            start=start,
            end=end,
            now=now,
            now_rel=now_rel,
            base_time=base_time,
            lead_time=lead_time,
            reduction_start=reduction_start,
            reduction_start_rel=reduction_start_rel,
        )


@dataclass
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
    site: str
    rate: Union[float, Tuple[float, ...]]
    rate_unit: str
    start_rel: timedelta

    @classmethod
    def from_file(cls, fi: nc4.Dataset, setup: Setup) -> "ReleaseMetaData":
        """Read information on a release from open file."""
        raw = RawReleaseMetaData.from_file(fi, setup)
        site = raw.site
        # SR_HC <
        if site == "Goesgen":
            site = r"G$\mathrm{\"o}$sgen"
        # SR_HC >
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
            site=site,
            rate=mass / duration.total_seconds() if duration else np.nan,
            rate_unit=f"{mass_unit} {duration_unit}-1",
            start_rel=start_rel,
        )


@dataclass
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
    def from_file(cls, fi: nc4.Dataset, setup: Setup) -> "SpeciesMetaData":
        model: str = setup.model
        name: str
        if setup.core.input_variable == "affected_area":
            alt_setup = setup.derive({"input_variable": "concentration"})
            return cls.from_file(fi, alt_setup)
        else:
            var_name: str = nc_var_name(setup, model)
            try:
                var: nc4.Variable = fi.variables[var_name]
                name = getncattr(var, "long_name")
            except (KeyError, AttributeError):
                if model.startswith("IFS"):
                    name = cls._get_species_name_ifs(fi, var_name)
                elif setup.core.input_variable == "deposition":
                    # Deposition field may be missing
                    alt_setup = setup.derive({"input_variable": "concentration"})
                    return cls.from_file(fi, alt_setup)
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


class RawReleaseMetaData(BaseModel):
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

    class Config:  # noqa
        arbitrary_types_allowed = True
        extra = "forbid"
        validate_all = True
        validate_assigment = True

    # pylint: disable=R0914  # too-many-locals
    @classmethod
    def from_file(cls, fi: nc4.Dataset, setup: Setup) -> "RawReleaseMetaData":
        """Read information on a release from open file."""

        assert setup.core.dimensions.numpoint is not None  # mypy
        idx_point = setup.core.dimensions.numpoint

        assert setup.core.dimensions.species_id is not None  # mypy
        # SR_TMP <
        idx_spec = setup.core.dimensions.species_id - 1
        assert 0 <= idx_spec < fi.dimensions["numspec"].size
        # SR_TMP >

        assert setup.core.dimensions.nageclass is not None  # mypy
        idx_age = setup.core.dimensions.nageclass

        var_name: str = "RELCOM"  # SR_HC TODO un-hardcode
        var = fi.variables[var_name]

        # Check index against no. release point and set it if necessary
        n = var.shape[0]
        if n == 0:
            raise ValueError(f"file '{fi.name}': no release points ('{var_name}')")
        elif n == 1:
            if idx_point is None:
                idx_point = 0
        elif n > 1:
            if idx_point is None:
                raise ValueError(
                    f"file '{fi.name}': idx is None despite {n} release points"
                )
        assert idx_point is not None  # mypy
        if idx_point < 0 or idx_point >= n:
            raise ValueError(
                f"file '{fi.name}': invalid index {idx_point} for {n} release points"
            )

        # Name: convert from byte character array
        site = var[idx_point][~var[idx_point].mask].tobytes().decode("utf-8").rstrip()

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
                params[key_out] = var[idx_point].tolist()
            elif var.dimensions == ("numspec", "numpoint"):
                params[key_out] = var[idx_spec, idx_point].tolist()
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
        return cls(**params)


class TimeStepMetaDataCollector:
    def __init__(self, fi: nc4.Dataset, setup: Setup, *, add_ts0: bool = False) -> None:
        self.fi = fi
        self.setup = setup
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
        """Current time step."""
        return self.start() + self.now_rel()

    def now_rel(self) -> timedelta:
        """Time since start."""
        var = self.fi.variables["time"]
        idx = self.time_step_idx()
        if idx < 0:
            return timedelta(0)
        return timedelta(seconds=int(var[idx]))

    def reduction_start(self) -> datetime:
        """Time step when reduction (mean/sum) started."""
        return self.now() - self.integration_duration()

    def reduction_start_rel(self) -> timedelta:
        """Time between simulation start and start of reduction (mean/sum)."""
        return self.reduction_start() - self.start()

    def integration_duration(self) -> timedelta:
        """Compute timestep delta of integration period."""
        if self.setup.core.integrate:
            return self.now_rel()
        n = self.time_step_idx() + 1
        if n == 0:
            return timedelta(0)
        return self.now_rel() / n

    def time_step_idx(self) -> int:
        """Index of current time step of current field."""
        # Default to timestep of current field
        assert self.setup.core.dimensions.time is not None  # mypy
        if self.add_ts0:
            return self.setup.core.dimensions.time - 1
        return self.setup.core.dimensions.time


def nc_var_name(setup: Setup, model: str) -> str:
    cosmo_models = ["COSMO-2", "COSMO-1", "COSMO-2E", "COSMO-1E"]
    ifs_models = ["IFS-HRES", "IFS-HRES-EU"]
    # SR_TMP <
    dimensions = setup.core.dimensions
    input_variable = setup.core.input_variable
    deposition_type = setup.deposition_type_str
    # SR_TMP >
    assert dimensions.species_id is not None  # mypy
    species_id = dimensions.species_id
    if input_variable == "concentration":
        if model in cosmo_models:
            return f"spec{species_id:03d}"
        elif model in ifs_models:
            return f"spec{species_id:03d}_mr"
        else:
            raise ValueError("unknown model", model)
    elif input_variable == "deposition":
        prefix = {"wet": "WD", "dry": "DD"}[deposition_type]
        return f"{prefix}_spec{species_id:03d}"
    raise ValueError("unknown variable", input_variable)

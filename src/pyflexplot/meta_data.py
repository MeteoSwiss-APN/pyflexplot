# -*- coding: utf-8 -*-
# pylint: disable=C0302  # too-many-lines (>1000)
"""
Attributes.
"""
# Standard library
import re
import time
import warnings
from copy import copy
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generic
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

# Third-party
import netCDF4 as nc4
import numpy as np
from pydantic import BaseModel
from pydantic import validator
from pydantic.generics import GenericModel

# Local
from .setup import Setup
from .species import get_species
from .species import Species
from .utils.summarize import summarizable

ValueT = TypeVar("ValueT", int, float, str, datetime, timedelta)


@summarizable
# pylint: disable=E0213  # no-self-argument (validator)
class MetaDatum(GenericModel, Generic[ValueT]):
    """Individual piece of meta data."""

    name: str
    value: ValueT
    attrs: Dict[str, Any] = {}

    class Config:  # noqa
        extra = "forbid"
        validate_all = True
        validate_assigment = True

    @validator("attrs", pre=True, always=True)
    def _init_attrs(cls, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        return value

    @property
    def type_(self) -> type:
        return type(self.value)

    @property
    def cls_combo(self) -> type:
        try:
            return {
                int: MetaDatumCombo[int],
                float: MetaDatumCombo[float],
                str: MetaDatumCombo[str],
                datetime: MetaDatumCombo[datetime],
                timedelta: MetaDatumCombo[timedelta],
            }[self.type_]
        except KeyError:
            raise NotImplementedError(f"{type(self).__name__}.cls_combo")

    def __str__(self):
        return format_meta_datum(self.value)

    def derive(
        self,
        *,
        value: Optional[ValueT] = None,
        name: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "MetaDatum[ValueT]":
        if name is None:
            name = self.name
        if value is None:
            value = self.value
        if attrs is None:
            attrs = self.attrs
        return type(self)(name=name, value=value, attrs=attrs)

    def derive_combo(
        self,
        *,
        value: Sequence[ValueT],
        name: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "MetaDatumCombo[ValueT]":
        if name is None:
            name = self.name
        if attrs is None:
            attrs = self.attrs
        return self.cls_combo(name=name, value=value, attrs=attrs)

    # SR_TODO Extract methods into separate class (e.g., AttrMerger)!
    def merge_with(
        self,
        others: Collection["MetaDatum[ValueT]"],
        replace: Optional[Mapping[str, Any]] = None,
    ) -> Union["MetaDatum[ValueT]", "MetaDatumCombo[ValueT]"]:
        if replace is None:
            replace = {}

        # Reduce names
        try:
            name = replace["name"]
        except KeyError:
            names = sorted(set([self.name] + [other.name for other in others]))
            if len(names) > 1:
                raise ValueError("names differ", names)
            name = self.name

        # Reduce values
        try:
            reduced_value = replace["value"]
        except KeyError:
            reduced_value = self._reduce_values(others)

        # Reduce attrs dicts
        attrs = self._merge_attrs(others)

        if isinstance(reduced_value, Sequence) and not isinstance(reduced_value, str):
            combo_value: Sequence[ValueT] = reduced_value  # mypy
            return self.derive_combo(name=name, value=combo_value, attrs=attrs)
        else:
            value: ValueT = reduced_value  # type: ignore  # mypy
            return self.derive(name=name, value=value, attrs=attrs)

    def _reduce_values(
        self, others: Collection["MetaDatum[ValueT]"],
    ) -> Union[ValueT, Tuple[ValueT, ...]]:

        # Collect unique values
        values = [self.value]
        for other in others:
            if other.value not in values:
                values.append(other.value)

        if len(values) == 1:
            return next(iter(values))

        return next(iter(values)) if len(values) == 1 else tuple(values)

    def _merge_attrs(self, others: Collection["MetaDatum"]) -> Dict[str, Any]:
        assert all(o.attrs.keys() == self.attrs.keys() for o in others)
        attrs = {}
        for attr, value in self.attrs.items():
            other_values = [o.attrs[attr] for o in others]
            if all(other_value == value for other_value in other_values):
                attrs[attr] = value
            else:
                raise NotImplementedError("values differ", attr, value, other_values)
        return attrs

    def combine_with(self, other: "MetaDatum", join: str = " ") -> "MetaDatum":
        if not isinstance(other, MetaDatum):
            raise ValueError(
                f"wrong type to be combined with {type(self).__name__}", type(other)
            )
        combo_value = join.join([str(self), str(other)])
        return MetaDatum[str](name=self.name, value=combo_value, attrs=self.attrs)


@summarizable
class MetaDatumCombo(GenericModel, Generic[ValueT]):
    """Meta datum with multiple values."""

    name: str
    value: Tuple[ValueT, ...]
    attrs: Dict[str, Any] = {}

    class Config:  # noqa
        extra = "forbid"
        validate_all = True
        validate_assignment = True

    @property
    def type_(self):
        return type(next(iter(self.value)))

    def __str__(self):
        return format_meta_datum(self.value, self.attrs.get("join"))

    def derive(
        self,
        name: Optional[str] = None,
        value: Optional[Sequence[ValueT]] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "MetaDatumCombo[ValueT]":
        if name is None:
            name = self.name
        if value is None:
            value = self.value
        if attrs is None:
            attrs = self.attrs
        return type(self)(name=name, value=value, attrs=attrs)

    def as_datums(self) -> List[MetaDatum[ValueT]]:
        datums: List[MetaDatum] = []
        for value in self.value:
            cls = MetaDatum[type(value)]  # type: ignore
            datum = cls(name=self.name, value=value, attrs=self.attrs)
            datums.append(datum)
        return datums

    def merge_with(self, others, replace=None):
        raise NotImplementedError(f"{type(self).__name__}.merge_with")

    def combine_with(
        self, other: Union[MetaDatum, "MetaDatumCombo"], join: str = " "
    ) -> "MetaDatumCombo":
        if isinstance(other, MetaDatum):
            combo_value = [self.value] * len(self.value)
            other = other.derive_combo(value=combo_value)
        elif not isinstance(other, MetaDatumCombo):
            raise ValueError(
                "wrong type to be combined with {type(self).__name__}", type(other)
            )
        elif len(self.value) != len(other.value):
            raise ValueError(
                "number of combo values differs", len(self.value), len(other.value)
            )
        assert isinstance(other, MetaDatumCombo)  # mypy
        combo_values = []
        for self_datum, other_datum in zip(self.as_datums(), other.as_datums()):
            combined_datum = self_datum.combine_with(other_datum, join=join)
            combo_values.append(combined_datum.value)
        return MetaDatumCombo[str](
            name=f"{self.name}+{other.name}",
            value=combo_values,
            attrs={**other.attrs, **self.attrs},
        )


def format_meta_datum(value: Any, join: Optional[str] = None) -> str:
    if isinstance(value, Collection) and not isinstance(value, str):
        # SR_TODO make sure this is covered by a test (it currently isn't)!
        return (join or " / ").join([format_meta_datum(v) for v in value])
    elif isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M %Z")
    elif isinstance(value, timedelta):
        hours = int(value.total_seconds() / 3600)
        minutes = int((value.total_seconds() / 60) % 60)
        return f"{hours:d}:{minutes:02d}$\\,$h"
    fmt = "g" if isinstance(value, (float, int)) else ""
    return f"{{:{fmt}}}".format(value)


@dataclass
class MetaData:
    release: "ReleaseMetaData"
    simulation: "SimulationMetaData"
    variable: "VariableMetaData"
    species: Union[Species, Tuple[Species, ...]]

    def merge_with(self, others: Collection["MetaData"],) -> "MetaData":
        if not others:
            return self

        def merge_release():
            if not all(other.release == self.release for other in others):
                raise ValueError("release meta data differ")
            return self.release

        def merge_simulation():
            if not all(other.simulation == self.simulation for other in others):
                raise ValueError("simulation meta data differ")
            return self.simulation

        def merge_variable():
            unit = self.variable.unit
            level_unit = self.variable.level_unit
            bottom_level = self.variable.bottom_level
            top_level = self.variable.top_level
            for obj in others:
                assert unit == obj.variable.unit
                assert level_unit == obj.variable.level_unit
                bottom_level = min([bottom_level, obj.variable.bottom_level])
                top_level = max([top_level, obj.variable.top_level])
            return type(self.variable)(
                unit=unit,
                level_unit=level_unit,
                bottom_level=bottom_level,
                top_level=top_level,
            )

        def merge_species():
            species_lst: List[Species] = []
            for obj in [self] + list(others):
                for species in (
                    [obj.species] if isinstance(obj.species, Species) else obj.species
                ):
                    if species not in species_lst:
                        species_lst.append(species)
            return species_lst[0] if len(species_lst) == 1 else tuple(species_lst)

        return type(self)(
            release=merge_release(),
            simulation=merge_simulation(),
            variable=merge_variable(),
            species=merge_species(),
        )

    def __deepcopy__(self, memo):
        return type(self)(
            release=copy(self.release),
            simulation=copy(self.simulation),
            variable=copy(self.variable),
            species=deepcopy(self.species, memo),
        )

    def format(
        self,
        param: str,
        *,
        add_unit: bool = False,
        auto_format_unit: bool = True,
        join_combo: Optional[str] = None,
    ) -> str:
        """Format a parameter, optionally adding the unit (`~_unit`)."""
        if add_unit and param.endswith("_unit"):
            raise ValueError("cannot add unit to param ending in '_unit'", param)
        try:
            datum = getattr(self, param)
        except AttributeError:
            raise ValueError("invalid param", param)
        if auto_format_unit and param.endswith("_unit"):
            contains_unit = True
        elif add_unit:
            unit_param = f"{param}_unit"
            try:
                unit_datum = getattr(self, unit_param)
            except AttributeError:
                raise Exception("no unit found for parameter", param, unit_param)
            datum = datum.combine_with(unit_datum, join=r"$\,$")
            contains_unit = True
        else:
            contains_unit = False
        if isinstance(datum, MetaDatumCombo) and join_combo is not None:
            datum = datum.derive(attrs={**datum.attrs, "join": join_combo})
        # SR_TMP <
        if isinstance(datum, MetaDatumCombo):
            datums = datum.as_datums()
            if all(datum == datums[0] for datum in datums[1:]):
                datum = datums[0]
        # SR_TMP >
        if contains_unit:
            return format_unit(str(datum))
        return str(datum)

    def __getattr__(self, attr):
        try:
            value = {
                "release_end_rel": self.release.end_rel,
                "release_height": self.release.height,
                "release_height_unit": self.release.height_unit,
                "release_mass": self.release.mass,
                "release_mass_unit": self.release.mass_unit,
                "release_rate": self.release.rate,
                "release_rate_unit": self.release.rate_unit,
                "release_site_lat": self.release.lat,
                "release_site_lon": self.release.lon,
                "release_site_name": self.release.site,
                "release_start_rel": self.release.start_rel,
                "simulation_end": self.simulation.end,
                "simulation_integr_start_rel": self.simulation.integration_start_rel,
                "simulation_integr_start": self.simulation.integration_start,
                "simulation_now_rel": self.simulation.now_rel,
                "simulation_now": self.simulation.now,
                "simulation_start": self.simulation.start,
                "variable_level_bot": self.variable.bottom_level,
                "variable_level_bot_unit": self.variable.level_unit,
                "variable_level_top": self.variable.top_level,
                "variable_level_top_unit": self.variable.level_unit,
                "variable_unit": self.variable.unit,
            }[attr]
        except KeyError:
            if not attr.startswith("species_"):
                raise AttributeError(attr)
        else:
            return MetaDatum[type(value)](name=attr, value=value)
        values = []
        for species in (
            [self.species] if isinstance(self.species, Species) else self.species
        ):
            try:
                value = {
                    "species_deposit_vel": species.deposition_velocity.value,
                    "species_deposit_vel_unit": species.deposition_velocity.unit,
                    "species_half_life": species.half_life.value,
                    "species_half_life_unit": species.half_life.unit,
                    "species_name": species.name,
                    "species_sediment_vel": species.sedimentation_velocity.value,
                    "species_sediment_vel_unit": species.sedimentation_velocity.unit,
                    "species_washout_coeff": species.washout_coefficient.value,
                    "species_washout_coeff_unit": species.washout_coefficient.unit,
                    "species_washout_exponent": species.washout_exponent.value,
                }[attr]
            except KeyError:
                raise AttributeError(attr)
            else:
                values.append(value)
        if len(values) == 1:
            value = values[0]
            return MetaDatum[type(value)](name=attr, value=value)
        return MetaDatumCombo[type(values[0])](name=attr, value=values)


def format_unit(s: str) -> str:
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


def collect_meta_data(
    fi: nc4.Dataset,
    setup: Setup,
    nc_meta_data: Mapping[str, Mapping[str, Any]],
    *,
    add_ts0: bool = False,
) -> MetaData:
    assert issubclass(MetaData, MetaData)  # SR_TMP
    return MetaData(
        release=ReleaseMetaData.from_file(fi, setup),
        simulation=SimulationMetaData.from_file(fi, setup, add_ts0),
        variable=VariableMetaData.from_file(fi, setup, nc_meta_data),
        species=species_from_file(fi, setup, nc_meta_data),
    )


def getncattr(nc_obj: Union[nc4.Dataset, nc4.Variable], attr: str) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="tostring() is deprecated. Use tobytes() instead.",
        )
        return nc_obj.getncattr(attr)


@dataclass
class VariableMetaData:
    unit: str
    bottom_level: float
    top_level: float
    level_unit: str

    @classmethod
    def from_file(
        cls,
        fi: nc4.Dataset,
        setup: Setup,
        nc_meta_data: Mapping[str, Mapping[str, Any]],
    ) -> "VariableMetaData":
        name = nc_var_name(setup, nc_meta_data["derived"]["model"])
        var = fi.variables[name]
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


@dataclass
class SimulationMetaData:
    start: datetime
    end: datetime
    now: datetime
    now_rel: timedelta
    integration_start: datetime
    integration_start_rel: timedelta

    @classmethod
    def from_file(
        cls, fi: nc4.Dataset, setup: Setup, add_ts0: bool
    ) -> "SimulationMetaData":

        # Start and end timesteps of simulation
        start = datetime(
            *time.strptime(
                getncattr(fi, "ibdate") + getncattr(fi, "ibtime"), "%Y%m%d%H%M%S"
            )[:6],
            tzinfo=timezone.utc,
        )
        end = datetime(
            *time.strptime(
                getncattr(fi, "iedate") + getncattr(fi, "ietime"), "%Y%m%d%H%M%S",
            )[:6],
            tzinfo=timezone.utc,
        )

        # Current time step and start time step of current integration period
        collector = TimeStepMetaDataCollector(fi, setup, add_ts0=add_ts0)
        now = collector.now()
        integration_start = collector.integration_start()
        now_rel: timedelta = collector.now_rel()
        integration_start_rel = collector.integration_start_rel()
        return cls(
            start=start,
            end=end,
            now=now,
            now_rel=now_rel,
            integration_start=integration_start,
            integration_start_rel=integration_start_rel,
        )


@dataclass
# pylint: disable=R0902  # too-many-instance-attrbutes
class ReleaseMetaData:
    duration: timedelta
    duration_unit: str
    end_rel: timedelta
    height: float
    height_unit: str
    lat: float
    lon: float
    mass: float
    mass_unit: str
    site: str
    rate: float
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
        assert len(raw.ms_parts) == 1
        mass = next(iter(raw.ms_parts))
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


class RawReleaseMetaData(BaseModel):
    age_id: int
    kind: str
    lllat: float
    lllon: float
    ms_parts: Tuple[int, ...]
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

    @classmethod
    def from_file(cls, fi: nc4.Dataset, setup: Setup) -> "RawReleaseMetaData":
        """Read information on a release from open file."""

        assert setup.core.dimensions.numpoint is not None  # mypy
        idx = setup.core.dimensions.numpoint

        var_name: str = "RELCOM"  # SR_HC TODO un-hardcode
        var = fi.variables[var_name]

        # Check index against no. release point and set it if necessary
        n = var.shape[0]
        if n == 0:
            raise ValueError(f"file '{fi.name}': no release points ('{var_name}')")
        elif n == 1:
            if idx is None:
                idx = 0
        elif n > 1:
            if idx is None:
                raise ValueError(
                    f"file '{fi.name}': idx is None despite {n} release points"
                )
        assert idx is not None  # mypy
        if idx < 0 or idx >= n:
            raise ValueError(
                f"file '{fi.name}': invalid index {idx} for {n} release points"
            )

        # Name: convert from byte character array
        site = var[idx][~var[idx].mask].tostring().decode("utf-8").rstrip()

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
            params[key_out] = fi.variables[key_in][idx].tolist()
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

    def integration_start(self) -> datetime:
        """Time step when integration started."""
        return self.now() - self.integration_duration()

    def integration_start_rel(self) -> timedelta:
        """Time between start of simulation and start of integration."""
        return self.integration_start() - self.start()

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


def species_from_file(
    fi: nc4.Dataset, setup: Setup, nc_meta_data: Mapping[str, Mapping[str, Any]],
) -> Species:
    model: str = nc_meta_data["derived"]["model"]
    name: str = nc_var_name(setup, model)
    var: nc4.Variable = fi.variables[name]
    try:  # SR_TMP
        name = getncattr(var, "long_name")
    except AttributeError:
        # SR_TMP <
        # name = "N/A"
        if model.startswith("ifs"):
            # In the IFS NetCDF files, the deposition variables are missing
            # the basic meta data on species, like "long_name". Therefore,
            # try to # obtain the name from the activity variable of the
            # same species.
            if name.startswith("DD_") or name.startswith("WD_"):
                alternative_name = f"{name[3:]}_mr"
                try:
                    alternative_var = fi.variables[alternative_name]
                    name = getncattr(alternative_var, "long_name")
                except (KeyError, AttributeError):
                    name = "N/A"
                else:
                    name = name.split("_")[0]
        else:
            name = "N/A"
        # SR_TMP >
    else:
        name = name.split("_")[0]
    return get_species(name=name)


def nc_var_name(setup: Setup, model: str) -> str:
    # SR_TMP <
    dimensions = setup.core.dimensions
    input_variable = setup.core.input_variable
    deposition_type = setup.deposition_type_str
    # SR_TMP >
    assert dimensions.species_id is not None  # mypy
    species_id = dimensions.species_id
    if input_variable == "concentration":
        if model in ["cosmo2", "cosmo1"]:
            return f"spec{species_id:03d}"
        elif model in ["ifs", "ifs-hres"]:
            return f"spec{species_id:03d}_mr"
        else:
            raise ValueError("unknown model", model)
    elif input_variable == "deposition":
        prefix = {"wet": "WD", "dry": "DD"}[deposition_type]
        return f"{prefix}_spec{species_id:03d}"
    raise ValueError("unknown variable", input_variable)

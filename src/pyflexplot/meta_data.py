# -*- coding: utf-8 -*-
# pylint: disable=C0302  # too-many-lines (>1000)
"""
Attributes.
"""
# Standard library
import re
import time
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
from .setup import InputSetup
from .setup import InputSetupCollection
from .summarize import summarizable

ValueT = TypeVar("ValueT", int, float, str, datetime, timedelta)


@summarizable
# pylint: disable=E0213  # no-self-argument (@validator)
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
        return [
            MetaDatum[ValueT](name=self.name, value=value, attrs=self.attrs)
            for value in self.value
        ]

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
            name=f"self.name+other.name",
            value=combo_values,
            attrs={**other.attrs, **self.attrs},
        )


def format_meta_datum(value: str, join: Optional[str] = None) -> str:
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


def init_mdatum(type_: type, name: str, attrs: Optional[Mapping[str, Any]] = None):
    """Pydantic validator to initialize ``MetaData`` attributes."""

    def f(value):
        if not isinstance(value, (MetaDatum, MetaDatumCombo)):
            if isinstance(value, Collection) and not isinstance(value, str):
                cls = MetaDatumCombo
            else:
                cls = MetaDatum
            value = cls[type_](name=name, value=value, attrs=attrs)
        return value

    return validator(name, pre=True, allow_reuse=True)(f)


class MetaData(BaseModel):
    """Meta data.

    Attributes:
        release_end: End of the release.

        release_end_rel: End of the release relative to the start of the simulation.

        release_height: Release height value(s).

        release_height_unit: Release height unit

        release_mass: Release mass value(s).

        release_mass_unit: Release mass unit.

        release_rate: Release rate value(s).

        release_rate_unit: Release rate unit.

        release_site_lat: Latitude of release site.

        release_site_lon: Longitude of release site.

        release_site_name: Name of release site.

        release_start_rel:  Release start relative to the simulation start.

        release_start: Release start.

        simulation_end: Simulation end.

        simulation_integr_start_rel: Integration period start relative to the
            simulation start.

        simulation_integr_start: Integration period start.

        simulation_model_name: Model name.

        simulation_now: Current timestep.

        simulation_now_rel: Current timestep relative to the simulation start.

        simulation_start: Simulation start.

        species_deposit_vel: Deposition velocity value(s).

        species_deposit_vel_unit: Deposition velocity unit.

        species_half_life: Half life value(s).

        species_half_life_unit: Half life unit.

        species_name: Species name.

        species_sediment_vel: Sedimentation velocity value(s).

        species_sediment_vel_unit: Sedimentation velocity unit.

        species_washout_coeff_unit: Washout coefficient unit.

        species_washout_coeff: Washout coefficient value(s).

        species_washout_exponent: Washout exponent value(s).

        variable_level_bot: Bottom level value(s).

        variable_level_bot_unit: Bottom level unit.

        variable_level_top: Top level value(s).

        variable_level_top_unit: Bottom level unit.

        variable_name: Name of variable.

        variable_unit: Unit of variable as a string (e.g., 'm-3' for m^3).

    """

    setup: InputSetup
    release_end_rel: Union[MetaDatum[timedelta], MetaDatumCombo[timedelta]]
    release_end: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    release_height: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_height_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_mass: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_mass_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_rate: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_rate_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_site_lat: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_site_lon: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_site_name: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_start_rel: Union[MetaDatum[timedelta], MetaDatumCombo[timedelta]]
    release_start: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    simulation_end: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    simulation_integr_start_rel: Union[MetaDatum[timedelta], MetaDatumCombo[timedelta]]
    simulation_integr_start: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    simulation_model_name: Union[MetaDatum[str], MetaDatumCombo[str]]
    simulation_now_rel: Union[MetaDatum[timedelta], MetaDatumCombo[timedelta]]
    simulation_now: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    simulation_start: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    species_deposit_vel: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_deposit_vel_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_half_life: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_half_life_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_name: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_sediment_vel: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_sediment_vel_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_washout_coeff: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_washout_coeff_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_washout_exponent: Union[MetaDatum[float], MetaDatumCombo[float]]
    variable_level_bot: Union[MetaDatum[float], MetaDatumCombo[float]]
    variable_level_bot_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    variable_level_top: Union[MetaDatum[float], MetaDatumCombo[float]]
    variable_level_top_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    variable_unit: Union[MetaDatum[str], MetaDatumCombo[str]]

    _init_release_end = init_mdatum(datetime, "release_end")
    _init_release_end_rel = init_mdatum(timedelta, "release_end_rel")
    _init_release_height = init_mdatum(float, "release_height")
    _init_release_height_unit = init_mdatum(str, "release_height_unit")
    _init_release_mass = init_mdatum(float, "release_mass")
    _init_release_mass_unit = init_mdatum(str, "release_mass_unit")
    _init_release_rate = init_mdatum(float, "release_rate")
    _init_release_rate_unit = init_mdatum(str, "release_rate_unit")
    _init_release_site_lat = init_mdatum(float, "release_site_lat")
    _init_release_site_lon = init_mdatum(float, "release_site_lon")
    _init_release_site_name = init_mdatum(str, "release_site_name")
    _init_release_start = init_mdatum(datetime, "release_start")
    _init_release_start_rel = init_mdatum(timedelta, "release_start_rel")
    _init_simulation_end = init_mdatum(datetime, "simulation_end")
    _init_simulation_integr_start = init_mdatum(datetime, "simulation_integr_start")
    _init_simulation_integr_start_rel = init_mdatum(
        timedelta, "simulation_integr_start_rel",
    )
    _init_simulation_model_name = init_mdatum(str, "simulation_model_name")
    _init_simulation_now = init_mdatum(datetime, "simulation_now")
    _init_simulation_now_rel = init_mdatum(timedelta, "simulation_now_rel")
    _init_simulation_start = init_mdatum(datetime, "simulation_start")
    _init_species_deposit_vel = init_mdatum(float, "species_deposit_vel")
    _init_species_deposit_vel_unit = init_mdatum(str, "species_deposit_vel_unit")
    _init_species_half_life = init_mdatum(float, "species_half_life")
    _init_species_half_life_unit = init_mdatum(str, "species_half_life_unit")
    _init_species_name = init_mdatum(str, "species_name", attrs={"join": " + "})
    _init_species_sediment_vel = init_mdatum(float, "species_sediment_vel")
    _init_species_sediment_vel_unit = init_mdatum(str, "species_sediment_vel_unit")
    _init_species_washout_coeff = init_mdatum(float, "species_washout_coeff")
    _init_species_washout_coeff_unit = init_mdatum(str, "species_washout_coeff_unit")
    _init_species_washout_exponent = init_mdatum(float, "species_washout_exponent")
    _init_variable_level_bot = init_mdatum(float, "variable_level_bot")
    _init_variable_level_bot_unit = init_mdatum(str, "variable_level_bot_unit")
    _init_variable_level_top = init_mdatum(float, "variable_level_top")
    _init_variable_level_top_unit = init_mdatum(str, "variable_level_top_unit")
    _init_variable_unit = init_mdatum(str, "variable_unit")

    class Config:  # noqa
        extra = "forbid"

    def merge_with(
        self,
        others: Collection["MetaData"],
        replace: Optional[Mapping[str, Any]] = None,
    ) -> "MetaData":
        """Create a new instance by merging this instance with others.

        Note that neither ``self`` nor ``others`` are changed.

        Args:
            others: Other instances of the same meta data class, to be merged
                with this one.

            replace (optional): Attributes to be replaced in the merged
                instance. Must contain all meta data that differ between any
                of the instances to be merged. Defaults to '{}'.

        Returns:
            Merged instance derived from ``self`` and ``others``.

        """

        if replace is None:
            replace = {}

        # Check setups
        equal_setup_params = ["lang"]  # SR_TMP TODO add more keys
        self_setup_dct = {k: self.setup.dict()[k] for k in equal_setup_params}
        other_setups = [other.setup for other in others]
        other_setup_dcts = [
            {k: setup.dict()[k] for k in equal_setup_params} for setup in other_setups
        ]
        differing = [dct for dct in other_setup_dcts if dct != self_setup_dct]
        if differing:
            raise ValueError(
                "setups of others differ",
                equal_setup_params,
                self_setup_dct,
                other_setup_dcts,
            )
        setup = InputSetupCollection([self.setup] + other_setups).compress()

        kwargs = {}
        for name in self.__fields__:  # pylint: disable=E1101  # no-member
            if name == "setup":
                continue
            datum = getattr(self, name)
            other_data = [getattr(o, name) for o in others]
            kwargs[name] = datum.merge_with(other_data, replace=replace.get(name))

        return type(self)(setup=setup, **kwargs)

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
        if contains_unit:
            return format_unit(str(datum))
        return str(datum)


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
    fi: nc4.Dataset, setup: InputSetup, nc_meta_data: Mapping[str, Any],
) -> MetaData:
    """Collect meta data in open NetCDF file."""
    return MetaDataCollector(fi, setup, nc_meta_data).run()


class MetaDataCollector:
    """Collect meta data for a field from an open NetCDF file."""

    def __init__(
        self, fi: nc4.Dataset, setup: InputSetup, nc_meta_data: Mapping[str, Any],
    ) -> None:
        self.fi = fi
        self.setup = setup
        self.nc_meta_data = nc_meta_data  # SR_TMP

        # Collect all global attributes
        self.ncattrs_global = {
            attr: self.fi.getncattr(attr) for attr in self.fi.ncattrs()
        }

        # Collect all variables attributes
        self.ncattrs_vars: Dict[str, Any] = {}
        for var in self.fi.variables.values():
            self.ncattrs_vars[var.name] = {
                attr: var.getncattr(attr) for attr in var.ncattrs()
            }

        # Select attributes of field variable
        name = nc_var_name(self.setup, self.nc_meta_data["analysis"]["model"])
        assert isinstance(name, str)  # mypy
        self.ncattrs_field = self.ncattrs_vars[name]

    def run(self) -> MetaData:
        """Collect meta data."""
        mdata_raw: Dict[str, Any] = {}
        self.collect_simulation_mdata(mdata_raw)
        self.collect_release_mdata(mdata_raw)
        self.collect_species_mdata(mdata_raw)
        self.collect_variable_mdata(mdata_raw)
        return MetaData(setup=self.setup, **mdata_raw)

    def collect_simulation_mdata(self, mdata_raw: Dict[str, Any]) -> None:
        """Collect simulation meta data."""

        # Model name
        model_name_raw = self.nc_meta_data["analysis"]["model"]
        if model_name_raw == "cosmo1":
            model_name = "COSMO-1"
        elif model_name_raw == "cosmo2":
            model_name = "COSMO-2"
        elif model_name_raw == "ifs":
            model_name = "IFS"
        else:
            raise Exception("unknown raw model name", model_name_raw)

        # Start and end timesteps of simulation
        ts_start = datetime(
            *time.strptime(
                self.ncattrs_global["ibdate"] + self.ncattrs_global["ibtime"],
                "%Y%m%d%H%M%S",
            )[:6],
            tzinfo=timezone.utc,
        )
        ts_end = datetime(
            *time.strptime(
                self.ncattrs_global["iedate"] + self.ncattrs_global["ietime"],
                "%Y%m%d%H%M%S",
            )[:6],
            tzinfo=timezone.utc,
        )

        # Current time step and start time step of current integration period
        tsp = TimeStepMetaDataCollector(self.fi, self.setup)
        ts_now = tsp.ts_now()
        ts_integr_start = tsp.ts_integr_start()
        ts_now_rel: timedelta = ts_now - ts_start
        ts_integr_start_rel: timedelta = ts_integr_start - ts_start

        # Type of integration (or, rather, reduction)

        mdata_raw.update(
            {
                "simulation_model_name": model_name,
                "simulation_start": ts_start,
                "simulation_end": ts_end,
                "simulation_now": ts_now,
                "simulation_now_rel": ts_now_rel,
                "simulation_integr_start": ts_integr_start,
                "simulation_integr_start_rel": ts_integr_start_rel,
            }
        )

    def collect_release_mdata(self, mdata_raw: Dict[str, Any]) -> None:
        """Collect release point meta data."""
        release = ReleaseMetaData.from_file(self.fi, mdata_raw, self.setup)
        mdata_raw.update(
            {
                # "release_duration": release.duration,
                # "release_duration_unit": release.duration_unit,
                "release_end": release.end,
                "release_end_rel": release.end_rel,
                "release_height": release.height,
                "release_height_unit": release.height_unit,
                "release_mass": release.mass,
                "release_mass_unit": release.mass_unit,
                "release_rate": release.rate,
                "release_rate_unit": release.rate_unit,
                "release_site_lat": release.lat,
                "release_site_lon": release.lon,
                "release_site_name": release.name,
                "release_start": release.start,
                "release_start_rel": release.start_rel,
            }
        )

    def collect_variable_mdata(self, mdata_raw: Dict[str, Any]) -> None:
        """Collect variable meta data."""

        unit = self.ncattrs_field["units"]
        # SR_TMP <
        unit = fix_unit_meters_agl(unit, self.setup.lang)
        # SR_TMP >

        # SR_TMP < TODO clean up once CoreInputSetup has been implemented
        assert self.setup.level is None or len(self.setup.level) == 1
        idx = None if self.setup.level is None else next(iter(self.setup.level))
        # idx = self.setup.level
        # SR_TMP >

        if idx is None:
            level_unit = ""
            level_bot = -1.0
            level_top = -1.0
        else:
            try:  # SR_TMP IFS
                var = self.fi.variables["level"]
            except KeyError:  # SR_TMP IFS
                var = self.fi.variables["height"]  # SR_TMP IFS
            level_bot = 0.0 if idx == 0 else float(var[idx - 1])
            level_top = float(var[idx])
            level_unit = var.getncattr("units")
            # SR_TMP <
            level_unit = fix_unit_meters_agl(level_unit, self.setup.lang)
            # SR_TMP >

        mdata_raw.update(
            {
                "variable_unit": unit,
                "variable_level_bot": level_bot,
                "variable_level_bot_unit": level_unit,
                "variable_level_top": level_top,
                "variable_level_top_unit": level_unit,
            }
        )

    def collect_species_mdata(self, mdata_raw: Dict[str, Any]) -> None:
        """Collect species meta data."""

        name_core = nc_var_name(self.setup, self.nc_meta_data["analysis"]["model"])
        if self.setup.input_variable == "deposition":  # SR_TMP
            name_core = name_core[3:]
        try:  # SR_TMP IFS
            deposit_vel = self.ncattrs_vars[f"DD_{name_core}"]["dryvel"]
            washout_coeff = self.ncattrs_vars[f"WD_{name_core}"]["weta"]
            washout_exponent = self.ncattrs_vars[f"WD_{name_core}"]["wetb"]
        except KeyError:  # SR_TMP IFS
            deposit_vel = -1  # SR_TMP IFS
            washout_coeff = -1  # SR_TMP IFS
            washout_exponent = -1  # SR_TMP IFS

        name = self.ncattrs_field["long_name"].split("_")[0]

        if name.startswith("Cs-137"):
            half_life = 30.17  # SR_HC
            half_life_unit = "a"  # SR_HC
        elif name.startswith("I-131a"):
            half_life = 8.02  # SR_HC
            half_life_unit = "d"  # SR_HC
        else:
            raise NotImplementedError(f"half_life of '{name}'")

        sediment_vel = 0.0  # SR_HC
        deposit_vel_unit = "m s-1"  # SR_HC
        sediment_vel_unit = "m s-1"  # SR_HC
        washout_coeff_unit = "s-1"  # SR_HC

        mdata_raw.update(
            {
                "species_name": name,
                "species_half_life": half_life,
                "species_half_life_unit": half_life_unit,
                "species_deposit_vel": deposit_vel,
                "species_deposit_vel_unit": deposit_vel_unit,
                "species_sediment_vel": sediment_vel,
                "species_sediment_vel_unit": sediment_vel_unit,
                "species_washout_coeff": washout_coeff,
                "species_washout_coeff_unit": washout_coeff_unit,
                "species_washout_exponent": washout_exponent,
            }
        )


class TimeStepMetaDataCollector:
    def __init__(self, fi: nc4.Dataset, setup: InputSetup) -> None:
        self.fi = fi
        self.setup = setup

    def comp_ts_start(self) -> datetime:
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

    def ts_now(self) -> datetime:
        """Compute current time step."""
        return self.comp_ts_start() + self.ts_delta_tot()

    def ts_integr_start(self) -> datetime:
        """Compute the timestep when integration started."""
        return self.ts_now() - self.ts_delta_integr()

    def ts_delta_tot(self) -> timedelta:
        """Compute time since start."""
        var = self.fi.variables["time"]
        return timedelta(seconds=int(var[self.ts_idx()]))

    def ts_delta_integr(self) -> timedelta:
        """Compute timestep delta of integration period."""
        delta_tot = self.ts_delta_tot()
        if self.setup.integrate:
            return delta_tot
        delta_prev = delta_tot / (self.ts_idx() + 1)
        return delta_prev

    def ts_idx(self) -> int:
        """Index of current time step of current field."""
        # Default to timestep of current field
        assert self.setup.time is not None  # mypy
        assert len(self.setup.time) == 1  # SR_TMP
        return next(iter(self.setup.time))  # SR_TMP


class RawReleaseMetaData(BaseModel):
    age_id: int
    kind: str
    lllat: float
    lllon: float
    ms_parts: Tuple[int, ...]
    name: str
    n_parts: int
    rel_end: timedelta
    rel_start: timedelta
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
    def from_file(cls, fi: nc4.Dataset, setup: InputSetup) -> "RawReleaseMetaData":
        """Read information on a release from open file."""

        assert setup.numpoint is not None  # mypy
        # SR_TMP < TODO proper implementation
        assert len(setup.numpoint) == 1
        idx: int = next(iter(setup.numpoint))
        # SR_TMP >

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
        name = var[idx][~var[idx].mask].tostring().decode("utf-8").rstrip()

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
        params = {"name": name}
        for key_out, key_in in key_pairs:
            params[key_out] = fi.variables[key_in][idx].tolist()
            if key_out in store_units:
                unit = fi.variables[key_in].getncattr("units")
                # SR_TMP <
                unit = fix_unit_meters_agl(unit, setup.lang)
                # SR_TMP >
                params[f"{key_out}_unit"] = unit
        return cls(**params)


class ReleaseMetaData(BaseModel):
    """Release point information."""

    duration: timedelta
    duration_unit: str
    end: datetime
    end_rel: timedelta
    height: float
    height_unit: str
    lat: float
    lon: float
    mass: float
    mass_unit: str
    name: str
    rate: float
    rate_unit: str
    start: datetime
    start_rel: timedelta

    class Config:  # noqa
        arbitrary_types_allowed = True
        extra = "forbid"
        validate_all = True
        validate_assigment = True

    @classmethod
    def from_file(
        cls, fi: nc4.Dataset, nc_meta_data: Dict[str, Any], setup: InputSetup
    ) -> "ReleaseMetaData":
        """Read information on a release from open file."""

        raw = RawReleaseMetaData.from_file(fi, setup)

        name = raw.name
        if name == "Goesgen":
            name = r"G$\mathrm{\"o}$sgen"

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
            end=nc_meta_data["simulation_start"] + end_rel,
            end_rel=end_rel,
            height=np.mean([raw.zbot, raw.ztop]),
            height_unit=height_unit,
            lat=np.mean([raw.lllat, raw.urlat]),
            lon=np.mean([raw.lllon, raw.urlon]),
            mass=mass,
            mass_unit=mass_unit,
            name=name,
            rate=mass / duration.total_seconds(),
            rate_unit=f"{mass_unit} {duration_unit}-1",
            start=nc_meta_data["simulation_start"] + start_rel,
            start_rel=start_rel,
        )


# SR_TMP <<< TODO Eliminate this dirty workaround!!!
def fix_unit_meters_agl(unit: str, lang: str) -> str:
    """Replace 'meters' by 'm AGL' or its German equivalent.

    This fix is currently easiest to apply here, during meta data collection,
    because later, it is not as easily available stand-alone, but may be part
    of a MetaDataCombo or already concatenated with the value.

    However, WORDS is no longer available during meta data collection, so as a
    compromise (read: dirty workaround), this fix along with the necessary
    import of WORDS has been put in this function.

    TODO: Find a more appropriate solution!!!

    """
    if unit == "meters":
        # pylint: disable=C0415  # import-outside-toplevel
        from .words import WORDS  # isort:skip

        unit = WORDS.get("m_agl", lang=lang).s
    return unit


def nc_var_name(setup: InputSetup, model: str) -> Union[str, List[str]]:
    assert setup.species_id is not None  # mypy
    assert len(setup.species_id) == 1  # SR_TMP
    species_id = next(iter(setup.species_id))
    if setup.input_variable == "concentration":
        if model in ["cosmo2", "cosmo1"]:
            return f"spec{species_id:03d}"
        elif model == "ifs":
            return f"spec{species_id:03d}_mr"
        else:
            raise ValueError("unknown model", model)
    elif setup.input_variable == "deposition":
        assert isinstance(setup.deposition_type, str)  # mypy
        prefix = {"wet": "WD", "dry": "DD"}[setup.deposition_type]
        return f"{prefix}_spec{species_id:03d}"
    raise ValueError("unknown variable", setup.input_variable)


def get_integr_type(setup: InputSetup) -> str:
    if not setup.integrate:
        return "mean"
    elif setup.input_variable == "concentration":
        return "sum"
    elif setup.input_variable == "deposition":
        return "accum"
    else:
        raise NotImplementedError("integration type for variable", setup.input_variable)

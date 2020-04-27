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
from .utils import summarizable
from .words import WORDS
from .words import TranslatedWords

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
    def type_(self):
        return type(self.value)

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

        if isinstance(reduced_value, Collection) and not isinstance(reduced_value, str):
            return MetaDatumCombo[ValueT](name=name, value=reduced_value, attrs=attrs)
        else:
            return MetaDatum[ValueT](name=name, value=reduced_value, attrs=attrs)

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

    def __str__(self):
        return format_meta_datum(self.value)


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

    def type_(self):
        return type(next(iter(self.value)))

    def merge_with(self, others, replace=None):
        raise NotImplementedError(f"{type(self).__name__}.merge_with")

    def __str__(self):
        return format_meta_datum(self.value, self.attrs.get("join"))


def format_meta_datum(value: str, join: Optional[str] = None) -> str:
    if isinstance(value, Collection) and not isinstance(value, str):
        # SR_TODO make sure this is covered by a test (it currently isn't)!
        return (join or " / ").join([format_meta_datum(v) for v in value])
    elif isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M %Z")
    elif isinstance(value, timedelta):
        seconds = value.total_seconds()
        hours = int(seconds / 3600)
        mins = int((seconds / 3600) % 1 * 60)
        return f"{hours:02d}:{mins:02d}$\\,$h"
    fmt = "g" if isinstance(value, (float, int)) else ""
    return f"{{:{fmt}}}".format(value)


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


def format_level_range(
    value_bottom: Union[float, Sequence[float]],
    value_top: Union[float, Sequence[float]],
    unit_bottom: str,
    unit_top: str,
) -> Optional[str]:

    if (value_bottom, value_top) == (-1, -1):
        return None

    def fmt(bot, top):
        if unit_bottom != unit_top:
            raise Exception(f"level units differ: '{unit_bottom}' != '{unit_top}'")
        unit_fmtd = unit_top
        return f"{bot:g}" + r"$-$" + f"{top:g} {unit_fmtd}"

    try:
        # One level range (early exit)
        return fmt(value_bottom, value_top)
    except TypeError:
        pass

    # Multiple level ranges
    assert isinstance(value_bottom, Collection)  # mypy
    assert isinstance(value_top, Collection)  # mypy
    bots = sorted(value_bottom)
    tops = sorted(value_top)
    if len(bots) != len(tops):
        raise Exception(f"inconsistent no. levels: {len(bots)} != {len(tops)}")
    n = len(bots)
    if n == 2:
        # Two level ranges
        if tops[0] == bots[1]:
            return fmt(bots[0], tops[1])
        else:
            return f"{fmt(bots[0], tops[0])} + {fmt(bots[1], tops[1])}"
    elif n == 3:
        # Three level ranges
        if tops[0] == bots[1] and tops[1] == bots[2]:
            return fmt(bots[0], tops[2])
        else:
            raise NotImplementedError(f"3 non-continuous level ranges")
    else:
        raise NotImplementedError(f"{n} sets of levels")


def format_integr_period(mdata: "MetaData"):
    start = mdata.simulation_integr_start.value
    now = mdata.simulation_now.value
    assert isinstance(start, datetime)  # mypy
    assert isinstance(now, datetime)  # mypy
    integr_period = now - start
    return f"{integr_period.total_seconds()/3600:g}$\\,$h"


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


def collect_meta_data(
    fi: nc4.Dataset, setup: InputSetup, nc_meta_data: Mapping[str, Any],
) -> MetaData:
    """Collect meta data in open NetCDF file."""
    words = WORDS
    words.set_default_lang(setup.lang)
    return MetaDataCollector(fi, setup, words, nc_meta_data).run()


class MetaDataCollector:
    """Collect meta data for a field from an open NetCDF file."""

    def __init__(
        self,
        fi: nc4.Dataset,
        setup: InputSetup,
        words: TranslatedWords,
        nc_meta_data: Mapping[str, Any],
    ) -> None:
        self.fi = fi
        self.setup = setup
        self._words = words
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
        release = ReleaseMetaData.from_file(self.fi, mdata_raw, self.setup, self._words)
        mdata_raw.update(
            {
                # "release_duration": release.get_duration(),
                # "release_duration_unit": release.get_duration_unit(),
                "release_end": release.get_end(),
                "release_end_rel": release.get_end_rel(),
                "release_height": release.get_height(),
                "release_height_unit": release.get_height_unit(),
                "release_mass": release.get_mass(),
                "release_mass_unit": release.get_mass_unit(),
                "release_rate": release.get_rate(),
                "release_rate_unit": release.get_rate_unit(),
                "release_site_lat": release.get_lat(),
                "release_site_lon": release.get_lon(),
                "release_site_name": release.get_name(),
                "release_start": release.get_start(),
                "release_start_rel": release.get_start_rel(),
            }
        )

    def collect_variable_mdata(self, mdata_raw: Dict[str, Any]) -> None:
        """Collect variable meta data."""

        unit = self.ncattrs_field["units"]

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
            level_unit = self._words["m_agl"].s
            try:  # SR_TMP IFS
                _var = self.fi.variables["level"]
            except KeyError:  # SR_TMP IFS
                _var = self.fi.variables["height"]  # SR_TMP IFS
            level_bot = 0.0 if idx == 0 else float(_var[idx - 1])
            level_top = float(_var[idx])

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
        if self.setup.variable == "deposition":  # SR_TMP
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


class ReleaseMetaData(BaseModel):
    """Release point information."""

    nc_meta_data: Dict[str, Any]
    setup: InputSetup
    words: TranslatedWords

    raw_age_id: int
    raw_kind: str
    raw_lllat: float
    raw_lllon: float
    raw_ms_parts: Tuple[int, ...]
    raw_name: str
    raw_n_parts: int
    raw_rel_end: timedelta
    raw_rel_start: timedelta
    raw_urlat: float
    raw_urlon: float
    raw_zbot: float
    raw_ztop: float

    class Config:  # noqa
        arbitrary_types_allowed = True
        extra = "forbid"
        validate_all = True
        validate_assigment = True

    def get_name(self) -> str:
        name = self.raw_name
        if name == "Goesgen":
            name = r"G$\mathrm{\"o}$sgen"
        return name

    def get_start_rel(self) -> timedelta:
        return self.raw_rel_start

    def get_end_rel(self) -> timedelta:
        return self.raw_rel_end

    def get_start(self) -> datetime:
        return self.nc_meta_data["simulation_start"] + self.get_start_rel()

    def get_end(self) -> datetime:
        return self.nc_meta_data["simulation_start"] + self.get_end_rel()

    def get_lat(self) -> float:
        return np.mean([self.raw_lllat, self.raw_urlat])

    def get_lon(self) -> float:
        return np.mean([self.raw_lllon, self.raw_urlon])

    def get_height(self) -> float:
        return np.mean([self.raw_zbot, self.raw_ztop])

    def get_height_unit(self) -> str:
        return self.words["m_agl"].s

    def get_duration(self) -> timedelta:
        return self.get_end_rel() - self.get_start_rel()

    def get_duration_unit(self) -> str:
        return "s"  # SR_HC

    def get_mass(self) -> float:
        assert len(self.raw_ms_parts) == 1
        return next(iter(self.raw_ms_parts))

    def get_mass_unit(self) -> str:
        return "Bq"  # SR_HC

    def get_rate(self) -> float:
        return self.get_mass() / self.get_duration().total_seconds()

    def get_rate_unit(self) -> str:
        return f"{self.get_mass_unit()} {self.get_duration_unit()}-1"

    @classmethod
    def from_file(
        cls,
        fi: nc4.Dataset,
        nc_meta_data: Dict[str, Any],
        setup: InputSetup,
        words: TranslatedWords,
    ) -> "ReleaseMetaData":
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
        raw_mdata = {"raw_name": name}
        for key_out, key_in in key_pairs:
            raw_mdata[f"raw_{key_out}"] = fi.variables[key_in][idx].tolist()

        return cls(nc_meta_data=nc_meta_data, setup=setup, words=words, **raw_mdata)


# SR_TMP <<< TODO figure out what to do with this
def nc_var_name(setup: InputSetup, model: str) -> Union[str, List[str]]:
    assert setup.species_id is not None  # mypy
    assert len(setup.species_id) == 1  # SR_TMP
    species_id = next(iter(setup.species_id))
    if setup.variable == "concentration":
        if model in ["cosmo2", "cosmo1"]:
            return f"spec{species_id:03d}"
        elif model == "ifs":
            return f"spec{species_id:03d}_mr"
        else:
            raise ValueError("unknown model", model)
    elif setup.variable == "deposition":
        assert isinstance(setup.deposition_type, str)  # mypy
        prefix = {"wet": "WD", "dry": "DD"}[setup.deposition_type]
        return f"{prefix}_spec{species_id:03d}"
    raise ValueError("unknown variable", setup.variable)


# SR_TMP <<< TODO figure out what to do with this
def get_integr_type(setup: InputSetup) -> str:
    if not setup.integrate:
        return "mean"
    elif setup.variable == "concentration":
        return "sum"
    elif setup.variable == "deposition":
        return "accum"
    else:
        raise NotImplementedError("integration type for variable", setup.variable)

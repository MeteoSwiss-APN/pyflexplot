# -*- coding: utf-8 -*-
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
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

# Third-party
import numpy as np
from pydantic import BaseModel
from pydantic import validator
from pydantic.generics import GenericModel

# First-party
from srutils.dict import format_dictlike

# Local
from .setup import InputSetup
from .setup import InputSetupCollection
from .utils import summarizable
from .words import WORDS

ValueT = TypeVar("ValueT", int, float, str, datetime, timedelta)


@summarizable
class MetaDatum(GenericModel, Generic[ValueT]):
    """Individual piece of meta data."""

    name: str
    value: ValueT
    attrs: Dict[str, Any] = {}

    class Config:  # noqa
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

        if self.name == "variable_long_name":
            val = next(iter(values))
            assert isinstance(val, str)  # mypy
            if val.lower().startswith("beaufschlagtes gebiet"):
                values = [val.replace("nasse", "totale").replace("trockene", "totale")]
            elif val.lower().startswith("affected area"):
                values = [val.replace("wet", "total").replace("dry", "total")]
            elif val.lower().endswith("bodendeposition"):
                values = [val.replace("nasse", "totale").replace("trockene", "totale")]
            elif val.lower().endswith("surface deposition"):
                values = [val.replace("wet", "total").replace("dry", "total")]

        return next(iter(values)) if len(values) == 1 else tuple(values)

    def _merge_attrs(self, others):
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


@summarizable
class BaseMetaData:
    """Base class for meta data."""

    def __init__(self, *args, **kwargs):
        self.setup: InputSetup
        raise ValueError("BaseMetaData must be subclassed!")

    def __eq__(self, other: Any) -> bool:
        return self.setup == other.setup

    # SR_TODO <<< eliminate iterator method
    def iter_objs(self):
        if isinstance(self, BaseModel):  # SR_TMP
            for name in self.__fields__:
                if name != "setup":
                    yield name, getattr(self, name)
        else:
            for name in dir(self):
                if not (
                    name.startswith("_") or name in ["setup", "summarizable_attrs"]
                ):
                    datum = getattr(self, name)
                    if not callable(datum):
                        yield name, datum

    def merge_with(
        self,
        others: Collection["BaseMetaData"],
        replace: Optional[Dict[str, Any]] = None,
    ) -> "BaseMetaData":
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
        for name, datum in self.iter_objs():
            other_data = [getattr(o, name) for o in others]
            kwargs[name] = datum.merge_with(other_data, replace=replace.get(name))

        assert type(self) is not BaseMetaData  # mypy
        return type(self)(setup=setup, **kwargs)


def init_mdatum(type_, name, attrs=None):
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


class MetaData(BaseModel, BaseMetaData):
    """Meta data.

    Attributes:
        variable_name: Name of variable.

        variable_unit: Unit of variable as a string (e.g., 'm-3' for m^3).

        variable_level_bot: Bottom level value(s).

        variable_level_bot_unit: Bottom level unit.

        variable_level_top: Top level value(s).

        variable_level_top_unit: Bottom level unit.

        release_site_name: Name of release site.

        release_site_lat: Latitude of release site.

        release_site_lon: Longitude of release site.

        release_height: Release height value(s).

        release_rate: Release rate value(s).

        release_mass: Release mass value(s).

        release_height_unit: Release height unit

        release_rate_unit: Release rate unit.

        release_mass_unit: Release mass unit.

        release_start: Start of the release.

        release_start_rel: Start of the release relative to the start of the
            simulation.

        release_end: End of the release.

        release_end_rel: End of the release relative to the start of the
            simulation.

        species_name: Species name.

        species_half_life: Half life value(s).

        species_half_life_unit: Half life unit.

        species_deposit_vel: Deposition velocity value(s).

        species_deposit_vel_unit: Deposition velocity unit.

        species_sediment_vel: Sedimentation velocity value(s).

        species_sediment_vel_unit: Sedimentation velocity unit.

        species_washout_coeff: Washout coefficient value(s).

        species_washout_coeff_unit: Washout coefficient unit.

        species_washout_exponent: Washout exponent value(s).

        simulation_model_name: Name of the model.

        simulation_start: Start of the simulation.

        simulation_end: End of the simulation.

        simulation_now: Current timestep.

        simulation_now_rel: Current timestep relative to the start of the
            simulation.

        simulation_integr_start: Start of the integration period.

        simulation_integr_start_rel: Start of the integration period relative
            to the simulation start..

        simulation_integr_type: Type of integration (or reduction).

    """

    setup: InputSetup
    variable_long_name: Union[MetaDatum[str], MetaDatumCombo[str]]
    variable_short_name: Union[MetaDatum[str], MetaDatumCombo[str]]
    variable_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    variable_level_bot: Union[MetaDatum[float], MetaDatumCombo[float]]
    variable_level_top: Union[MetaDatum[float], MetaDatumCombo[float]]
    variable_level_bot_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    variable_level_top_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_site_name: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_site_lat: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_site_lon: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_height: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_rate: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_mass: Union[MetaDatum[float], MetaDatumCombo[float]]
    release_height_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_rate_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_mass_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    release_start: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    release_start_rel: Union[MetaDatum[timedelta], MetaDatumCombo[timedelta]]
    release_end: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    release_end_rel: Union[MetaDatum[timedelta], MetaDatumCombo[timedelta]]
    species_name: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_half_life: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_deposit_vel: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_sediment_vel: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_washout_coeff: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_washout_exponent: Union[MetaDatum[float], MetaDatumCombo[float]]
    species_half_life_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_deposit_vel_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_sediment_vel_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    species_washout_coeff_unit: Union[MetaDatum[str], MetaDatumCombo[str]]
    simulation_model_name: Union[MetaDatum[str], MetaDatumCombo[str]]
    simulation_start: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    simulation_end: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    simulation_now: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    simulation_now_rel: Union[MetaDatum[timedelta], MetaDatumCombo[timedelta]]
    simulation_integr_start: Union[MetaDatum[datetime], MetaDatumCombo[datetime]]
    simulation_integr_start_rel: Union[MetaDatum[timedelta], MetaDatumCombo[timedelta]]
    simulation_integr_type: Union[MetaDatum[str], MetaDatumCombo[str]]

    _init_variable_long_name = init_mdatum(str, "variable_long_name")
    _init_variable_short_name = init_mdatum(str, "variable_short_name")
    _init_variable_unit = init_mdatum(str, "variable_unit")
    _init_variable_level_bot = init_mdatum(float, "variable_level_bot")
    _init_variable_level_top = init_mdatum(float, "variable_level_top")
    _init_variable_level_bot_unit = init_mdatum(str, "variable_level_bot_unit")
    _init_variable_level_top_unit = init_mdatum(str, "variable_level_top_unit")
    _init_release_site_name = init_mdatum(str, "release_site_name")
    _init_release_site_lat = init_mdatum(float, "release_site_lat")
    _init_release_site_lon = init_mdatum(float, "release_site_lon")
    _init_release_height = init_mdatum(float, "release_height")
    _init_release_rate = init_mdatum(float, "release_rate")
    _init_release_mass = init_mdatum(float, "release_mass")
    _init_release_height_unit = init_mdatum(str, "release_height_unit")
    _init_release_rate_unit = init_mdatum(str, "release_rate_unit")
    _init_release_mass_unit = init_mdatum(str, "release_mass_unit")
    _init_release_start = init_mdatum(datetime, "release_start")
    _init_release_start_rel = init_mdatum(timedelta, "release_start_rel")
    _init_release_end = init_mdatum(datetime, "release_end")
    _init_release_end_rel = init_mdatum(timedelta, "release_end_rel")
    _init_species_name = init_mdatum(str, "species_name", attrs={"join": " + "})
    _init_species_half_life = init_mdatum(float, "species_half_life")
    _init_species_deposit_vel = init_mdatum(float, "species_deposit_vel")
    _init_species_sediment_vel = init_mdatum(float, "species_sediment_vel")
    _init_species_washout_coeff = init_mdatum(float, "species_washout_coeff")
    _init_species_washout_exponent = init_mdatum(float, "species_washout_exponent")
    _init_species_half_life_unit = init_mdatum(str, "species_half_life_unit")
    _init_species_deposit_vel_unit = init_mdatum(str, "species_deposit_vel_unit")
    _init_species_sediment_vel_unit = init_mdatum(str, "species_sediment_vel_unit")
    _init_species_washout_coeff_unit = init_mdatum(str, "species_washout_coeff_unit")
    _init_simulation_model_name = init_mdatum(str, "simulation_model_name")
    _init_simulation_start = init_mdatum(datetime, "simulation_start")
    _init_simulation_end = init_mdatum(datetime, "simulation_end")
    _init_simulation_now = init_mdatum(datetime, "simulation_now")
    _init_simulation_now_rel = init_mdatum(timedelta, "simulation_now_rel")
    _init_simulation_integr_start = init_mdatum(datetime, "simulation_integr_start")
    _init_simulation_integr_start_rel = init_mdatum(
        timedelta, "simulation_integr_start_rel"
    )
    _init_simulation_integr_type = init_mdatum(str, "simulation_integr_type")

    class Config:  # noqa
        extra = "forbid"

    def simulation_fmt_integr_period(self):
        integr_period = self.simulation_now.value - self.simulation_integr_start.value
        return f"{integr_period.total_seconds()/3600:g}$\\,$h"


def collect_meta_data(fi, setup, nc_meta_data):
    """Collect meta data in open NetCDF file."""
    words = WORDS
    words.set_default_lang(setup.lang)
    return MetaDataCollector(fi, setup, words, nc_meta_data).run()


class MetaDataCollector:
    """Collect meta data for a field from an open NetCDF file."""

    def __init__(self, fi, setup, words, nc_meta_data):
        self.fi = fi
        self.setup = setup
        self._words = words
        self.nc_meta_data = nc_meta_data  # SR_TMP

        # Collect all global attributes
        self.ncattrs_global = {
            attr: self.fi.getncattr(attr) for attr in self.fi.ncattrs()
        }

        # Collect all variables attributes
        self.ncattrs_vars = {}
        for var in self.fi.variables.values():
            self.ncattrs_vars[var.name] = {
                attr: var.getncattr(attr) for attr in var.ncattrs()
            }

        # Select attributes of field variable
        model = self.nc_meta_data["analysis"]["model"]
        self.ncattrs_field = self.ncattrs_vars[nc_var_name(self.setup, model)]

    def run(self):
        """Collect meta data."""
        mdata_raw = {}
        self.collect_simulation_mdata(mdata_raw)
        self.collect_release_mdata(mdata_raw)
        self.collect_species_mdata(mdata_raw)
        self.collect_variable_mdata(mdata_raw)
        return MetaData(setup=self.setup, **mdata_raw)

    def collect_simulation_mdata(self, mdata_raw):
        """Collect simulation meta data."""

        # Model name
        model_name = {"cosmo1": "COSMO-1", "cosmo2": "COSMO-2", "ifs": "IFS"}[
            self.nc_meta_data["analysis"]["model"]
        ]

        # Start and end timesteps of simulation
        _ga = self.ncattrs_global
        ts_start = datetime(
            *time.strptime(_ga["ibdate"] + _ga["ibtime"], "%Y%m%d%H%M%S")[:6],
            tzinfo=timezone.utc,
        )
        ts_end = datetime(
            *time.strptime(_ga["iedate"] + _ga["ietime"], "%Y%m%d%H%M%S")[:6],
            tzinfo=timezone.utc,
        )

        # Current time step and start time step of current integration period
        ts_now, ts_integr_start = self._get_current_timestep_etc()
        ts_now_rel = ts_now - ts_start
        ts_integr_start_rel = ts_integr_start - ts_start

        # Type of integration (or, rather, reduction)
        if self.setup.variable == "concentration":  # SR_TMP
            integr_type = "sum" if self.setup.integrate else "mean"
        elif self.setup.variable == "deposition":  # SR_TMP
            integr_type = "accum" if self.setup.integrate else "mean"
        else:
            raise NotImplementedError(
                f"no integration type specified for '{self.name}'"
            )

        mdata_raw.update(
            {
                "simulation_model_name": model_name,
                "simulation_start": ts_start,
                "simulation_end": ts_end,
                "simulation_now": ts_now,
                "simulation_now_rel": ts_now_rel,
                "simulation_integr_start": ts_integr_start,
                "simulation_integr_start_rel": ts_integr_start_rel,
                "simulation_integr_type": integr_type,
            }
        )

    def _get_current_timestep_etc(self, idx=None):
        """Get the current timestep, or a specific one by index."""

        if idx is None:
            # Default to timestep of current field
            assert len(self.setup.time) == 1  # SR_TMP
            idx = next(iter(self.setup.time))  # SR_TMP

        if not isinstance(idx, int):
            raise Exception(f"expect type 'int', not '{type(idx).__name__}': {idx}")

        var = self.fi.variables["time"]

        # Obtain start from time unit
        rx = re.compile(
            r"seconds since "
            r"(?P<yyyy>[12][0-9][0-9][0-9])-"
            r"(?P<mm>[01][0-9])-"
            r"(?P<dd>[0-3][0-9]) "
            r"(?P<HH>[0-2][0-9]):"
            r"(?P<MM>[0-6][0-9])"
        )
        match = rx.match(var.units)
        if not match:
            raise Exception(f"cannot extract start from units '{var.units}'")
        start = datetime(
            int(match["yyyy"]),
            int(match["mm"]),
            int(match["dd"]),
            int(match["HH"]),
            int(match["MM"]),
            tzinfo=timezone.utc,
        )

        # Determine time since start
        delta_tot = timedelta(seconds=int(var[idx]))

        # Determine current timestep
        now = start + delta_tot

        # Determine start timestep of integration period
        if self.setup.integrate:
            delta_integr = delta_tot
        else:
            delta_prev = delta_tot / (idx + 1)
            delta_integr = delta_prev
        ts_integr_start = now - delta_integr

        return now, ts_integr_start

    def collect_release_mdata(self, mdata_raw):
        """Collect release point meta data."""

        # Collect release point information
        # SR_TMP < TODO clean up once CoreInputSetup has been implemented
        assert len(self.setup.numpoint) == 1
        idx = next(iter(self.setup.numpoint))
        # idx = self.setup.numpoint
        # SR_TMP >
        numpoint = ReleasePoint.from_file(self.fi, idx)

        sim_start = mdata_raw["simulation_start"]
        start_rel = timedelta(seconds=numpoint.rel_start)
        end_rel = timedelta(seconds=numpoint.rel_end)
        start = sim_start + start_rel
        end = sim_start + end_rel

        site_lat = np.mean([numpoint.lllat, numpoint.urlat])
        site_lon = np.mean([numpoint.lllon, numpoint.urlon])
        site_name = numpoint.name
        if site_name == "Goesgen":
            site_name = r"G$\mathrm{\"o}$sgen"

        height = np.mean([numpoint.zbot, numpoint.ztop])
        height_unit = self._words["m_agl"].s

        assert len(numpoint.ms_parts) == 1
        mass = next(iter(numpoint.ms_parts))
        mass_unit = "Bq"  # SR_HC

        duration = numpoint.rel_end - numpoint.rel_start
        duration_unit = "s"  # SR_HC

        rate = mass / duration
        rate_unit = f"{mass_unit} {duration_unit}-1"

        mdata_raw.update(
            {
                "release_start": start,
                "release_start_rel": start_rel,
                "release_end": end,
                "release_end_rel": end_rel,
                "release_site_lat": site_lat,
                "release_site_lon": site_lon,
                "release_site_name": site_name,
                "release_height": height,
                "release_height_unit": height_unit,
                "release_rate": rate,
                "release_rate_unit": rate_unit,
                "release_mass": mass,
                "release_mass_unit": mass_unit,
            }
        )

    def collect_variable_mdata(self, mdata_raw):
        """Collect variable meta data."""

        # Variable names
        long_name = self._long_name()
        short_name = self._short_name()

        # Variable unit
        unit = self.ncattrs_field["units"]

        # SR_TMP < TODO clean up once CoreInputSetup has been implemented
        assert self.setup.level is None or len(self.setup.level) == 1
        idx = None if self.setup.level is None else next(iter(self.setup.level))
        # idx = self.setup.level
        # SR_TMP >
        if idx is None:
            level_unit = ""
            level_bot = -1
            level_top = -1
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
                "variable_long_name": long_name,
                "variable_short_name": short_name,
                "variable_unit": unit,
                "variable_level_bot": level_bot,
                "variable_level_bot_unit": level_unit,
                "variable_level_top": level_top,
                "variable_level_top_unit": level_unit,
            }
        )

    def collect_species_mdata(self, mdata_raw):
        """Collect species meta data."""

        substance = self._get_substance()

        # Get deposition and washout data
        model = self.nc_meta_data["analysis"]["model"]
        name_core = nc_var_name(self.setup, model)
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

        # Get half life information
        try:
            half_life, half_life_unit = {
                "Cs-137": (30.17, "a"),  # SR_HC
                "I-131a": (8.02, "d"),  # SR_HC
            }[substance]
        except KeyError:
            raise NotImplementedError(f"half_life of '{substance}'")

        deposit_vel_unit = "m s-1"  # SR_HC
        sediment_vel_unit = "m s-1"  # SR_HC
        washout_coeff_unit = "s-1"  # SR_HC

        mdata_raw.update(
            {
                "species_name": substance,
                "species_half_life": half_life,
                "species_half_life_unit": half_life_unit,
                "species_deposit_vel": deposit_vel,
                "species_deposit_vel_unit": deposit_vel_unit,
                "species_sediment_vel": 0.0,
                "species_sediment_vel_unit": sediment_vel_unit,
                "species_washout_coeff": washout_coeff,
                "species_washout_coeff_unit": washout_coeff_unit,
                "species_washout_exponent": washout_exponent,
            }
        )

    def _get_substance(self):
        substance = self.ncattrs_field["long_name"]
        if self.setup.variable == "deposition":  # SR_TMP
            substance = substance.replace(
                f"_{self.setup.deposition_type}_deposition", ""
            )  # SR_HC
        return substance

    def _long_name(self, *, variable=None, plot_type=None):
        setup = self.setup
        words = self._words
        if (variable, plot_type) == (None, None):
            variable = setup.variable
            plot_type = setup.plot_type
        dep = self._deposition_type_word()
        if plot_type and plot_type.startswith("affected_area"):
            super_name = self._long_name(variable="deposition")
            return f"{words['affected_area']} ({super_name})"
        elif plot_type == "ens_thr_agrmt":
            super_name = self._short_name(variable="deposition")
            return f"{words['threshold_agreement']} ({super_name})"
        elif plot_type == "ens_cloud_arrival_time":
            return f"{words['cloud_arrival_time']}"
        if variable == "deposition":
            if plot_type == "ens_min":
                return (
                    f"{words['ensemble_minimum']} {dep} {words['surface_deposition']}"
                )
            elif plot_type == "ens_max":
                return (
                    f"{words['ensemble_maximum']} {dep} {words['surface_deposition']}"
                )
            elif plot_type == "ens_median":
                return f"{words['ensemble_median']} {dep} {words['surface_deposition']}"
            elif plot_type == "ens_mean":
                return f"{words['ensemble_mean']} {dep} {words['surface_deposition']}"
            else:
                return f"{dep} {words['surface_deposition']}"
        if variable == "concentration":
            if plot_type == "ens_min":
                return f"{words['ensemble_minimum']} {words['concentration']}"
            elif plot_type == "ens_max":
                return f"{words['ensemble_maximum']} {words['concentration']}"
            elif plot_type == "ens_median":
                return f"{words['ensemble_median']} {words['concentration']}"
            elif plot_type == "ens_mean":
                return f"{words['ensemble_mean']} {words['concentration']}"
            else:
                ctx = "abbr" if setup.integrate else "*"
                return words["activity_concentration", None, ctx].s
        raise NotImplementedError(
            f"long_name for variable '{variable}' and plot_type '{plot_type}'"
        )

    def _short_name(self, *, variable=None, plot_type=None):
        setup = self.setup
        words = self._words
        if (variable, plot_type) == (None, None):
            variable = setup.variable
            plot_type = setup.plot_type
        if variable == "concentration":
            if plot_type == "ens_cloud_arrival_time":
                return f"{words['arrival'].c} ({words['hour', None, 'pl']}??)"
            else:
                if setup.integrate:
                    return (
                        f"{words['integrated', None, 'abbr']} "
                        f"{words['concentration', None, 'abbr']}"
                    )
                return words["concentration"].s
        if variable == "deposition":
            if plot_type == "ens_thr_agrmt":
                return (
                    f"{words['number_of', None, 'abbr'].c} "
                    f"{words['member', None, 'pl']}"
                )
            else:
                return words["deposition"].s
        raise NotImplementedError(
            f"short_name for variable '{variable}' and plot_type '{plot_type}'"
        )

    def _deposition_type_word(self):
        setup = self.setup
        words = self._words
        if setup.variable == "deposition":
            type_ = setup.deposition_type
            word = {"tot": "total"}.get(type_, type_)
            return words[word, None, "f"].s
        return "none"


class ReleasePoint:
    """Release point information."""

    # Define the init arguments with target types
    # Done here to avoid duplication
    attr_types = {
        "name": str,
        "age_id": int,
        "kind": str,
        "lllat": float,
        "lllon": float,
        "urlat": float,
        "urlon": float,
        "zbot": float,
        "ztop": float,
        "rel_start": int,
        "rel_end": int,
        "n_parts": int,
        "ms_parts": list,
    }

    def __init__(self, **kwargs):
        """Create an instance of ``ReleasePoint``."""

        # Keep track of passed keyword arguments
        attr_keys_todo = [k for k in self.attr_types]

        # Process keyword arguments
        for key, val in kwargs.items():

            # Check existence of argument while fetching target type
            try:
                type_ = self.attr_types[key]
            except KeyError:
                raise ValueError(f"unexpected argument {key}='{val}'")

            # Cross valid argument off the todo list
            attr_keys_todo.remove(key)

            # Check type while converting to target type
            try:
                val = type_(val)
            except TypeError:
                raise ValueError(
                    f"argument {key}={val} of type '{type(val).__name__}' incompatible "
                    f"with expected type '{type_.__name__}'"
                )

            # Set as attribute
            setattr(self, key, val)

        # Check that all arguments have been passed
        if attr_keys_todo:
            n = len(attr_keys_todo)
            raise ValueError(
                f"missing {n} argument{'' if n == 1 else 's'}: " f"{attr_keys_todo}"
            )

    def __repr__(self):
        return format_dictlike(self)

    def __str__(self):
        return format_dictlike(self)

    def __iter__(self):
        for key in self.attr_types:
            yield key, getattr(self, key)

    @classmethod
    def from_file(cls, fi, i=None, var_name="RELCOM"):
        """Read information on single release point from open file.

        Args:
            fi (netCDF4.Dataset): Open NetCDF file handle.

            i (int, optional): Release point index. Mandatory if ``fi``
                contains multiple release points. Defaults to None.

            var_name (str, optional): Variable name of release point. Defaults
                to 'RELCOM'.

        Returns:
            ReleasePoint: Release point object.

        """
        var = fi.variables[var_name]

        # Check index against no. release point and set it if necessary
        n = var.shape[0]
        if n == 0:
            raise ValueError(f"file '{fi.name}': no release points ('{var_name}')")
        elif n == 1:
            if i is None:
                i = 0
        elif n > 1:
            if i is None:
                raise ValueError(
                    f"file '{fi.name}': i is None despite {n} release points"
                )
        if i < 0 or i >= n:
            raise ValueError(
                f"file '{fi.name}': invalid index {i} for {n} release points"
            )

        kwargs = {}

        # Name -- byte character array
        kwargs["name"] = var[i][~var[i].mask].tostring().decode("utf-8").rstrip()

        # Other attributes
        key_pairs = [
            ("age_id", "LAGE"),
            ("kind", "RELKINDZ"),
            ("lllat", "RELLAT1"),
            ("lllon", "RELLNG1"),
            ("urlat", "RELLAT2"),
            ("urlon", "RELLNG2"),
            ("zbot", "RELZZ1"),
            ("ztop", "RELZZ2"),
            ("rel_start", "RELSTART"),
            ("rel_end", "RELEND"),
            ("n_parts", "RELPART"),
            ("ms_parts", "RELXMASS"),
        ]
        for key_out, key_in in key_pairs:
            kwargs[key_out] = fi.variables[key_in][i].tolist()

        return cls(**kwargs)

    @classmethod
    def multiple_from_file(cls, fi, var_name="RELCOM"):
        """Read information on multiple release points from open file.

        Args:
            fi (netCDF4.Dataset): Open NetCDF file handle.

            var_name (str, optional): Variable name of release point. Defaults
                to 'RELCOM'.

        Returns:
            list[ReleasePoint]: List of release points objects.

        """
        n = fi.variables[var_name].shape[0]
        return [cls.from_file(fi, i, var_name) for i in range(n)]


# SR_TMP <<< TODO figure out what to do with this
def nc_var_name(setup, model):
    result = []
    for species_id in setup.species_id:
        if setup.variable == "concentration":
            if model in ["cosmo2", "cosmo1"]:
                result.append(f"spec{species_id:03d}")
            elif model == "ifs":
                result.append(f"spec{species_id:03d}_mr")
            else:
                raise ValueError("unknown model", model)
        elif setup.variable == "deposition":
            prefix = {"wet": "WD", "dry": "DD"}[setup.deposition_type]
            result.append(f"{prefix}_spec{species_id:03d}")
    return result[0] if len(result) == 1 else result

# -*- coding: utf-8 -*-
"""
Attributes.
"""
# Standard library
import datetime
import re
import time
from typing import Any
from typing import Dict
from typing import Generic
from typing import Optional
from typing import TypeVar
from typing import Union

# Third-party
import numpy as np
from pydantic import BaseModel
from pydantic import root_validator
from pydantic import validator
from pydantic.generics import GenericModel

# First-party
from srutils.dict import format_dictlike
from srutils.iter import isiterable

# Local
from .setup import Setup
from .setup import SetupCollection
from .utils import summarizable
from .words import WORDS

ValueT = TypeVar("ValueT", int, float, str, datetime.datetime)


@summarizable
class Attr(GenericModel, Generic[ValueT]):
    """Individual attribute."""

    name: str
    value: ValueT
    attrs: Dict[str, Any] = {}  # need Optional[] despite validator?

    class Config:  # noqa
        # allow_mutation = False
        validate_all = True
        validate_assignment = True

    @validator("attrs", pre=True, always=True)
    def _init_attrs(cls, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        return value

    @property
    def type_(self):
        return type(self.value)

    @classmethod
    def create(cls, type_, *, name, value, attrs=None):
        return Attr[type_](name=name, value=value, attrs=attrs)

    # SR_TODO Extract methods into separate class (e.g., AttrMerger)!
    def merge_with(self, others, replace=None):
        assert all(issubclass(o.type_, self.type_) for o in others)  # noqa SR_DBG
        if replace is None:
            replace = {}
        name = replace.get("name", self.name)
        reduced_value = self._reduce_values(others, replace)
        attrs = self._merge_attrs(others)
        if isiterable(reduced_value, str_ok=False):
            return AttrMult(self.type_, name=name, values=reduced_value, attrs=attrs)
        return Attr(name=name, value=reduced_value, attrs=attrs)

    def _reduce_values(self, others, replace):
        values = [self.value]
        for other in others:
            if "name" not in replace:
                if other.name != self.name:
                    raise ValueError("names differ", self.name, other.name)
                if type(other) is not type(self):  # noqa
                    raise ValueError("types differ", type(self), type(other))
            if other.value not in values:
                values.append(other.value)
        if len(values) == 1:
            reduced_value = next(iter(values))
        else:
            # SR_TMP<
            if self.name == "long_name":
                name0 = next(iter(values))
                if name0.lower().startswith("beaufschlagtes gebiet"):
                    values = name0.replace("nasse", "totale").replace(
                        "trockene", "totale"
                    )
                elif name0.lower().startswith("affected area"):
                    values = name0.replace("wet", "total").replace("dry", "total")
                elif name0.lower().endswith("bodendeposition"):
                    values = name0.replace("nasse", "totale").replace(
                        "trockene", "totale"
                    )
                elif name0.lower().endswith("surface deposition"):
                    values = name0.replace("wet", "total").replace("dry", "total")
            # SR_TMP>
            reduced_value = next(iter(values)) if len(values) == 1 else values
        reduced_value = replace.get("value", reduced_value)
        return reduced_value

    def _merge_attrs(self, others):
        assert all(o.attrs.keys() == self.attrs.keys() for o in others)
        attrs = {}
        for attr, value in self.attrs.items():
            other_values = [o.attrs[attr] for o in others]
            if all(other_value == value for other_value in other_values):
                attrs[attr] = value
            else:
                raise NotImplementedError("differing values", attr, value, other_values)
        return attrs

    def format(
        self, fmt=None, *, escape_format=False, join=None, **kwargs_datetime,
    ):
        if issubclass(self.type_, datetime.datetime):
            return self._format_datetime(**kwargs_datetime)

        if fmt is None:
            fmt = "g" if issubclass(self.type_, (float, int)) else ""
        s = f"{{:{fmt}}}".format(self.value)
        if escape_format:
            s = s.replace("{", "{{").replace("}", "}}")
        return s

    def _format_datetime(self, *, rel=False, rel_start=None, rel_neg_ok=True):
        if not rel:
            return self.value.strftime("%Y-%m-%d %H:%M %Z")
        if rel_start is None:
            rel_start = self.attrs["start"]
        seconds = (self.value - rel_start).total_seconds()
        if not rel_neg_ok and seconds < 0:
            seconds = -seconds
        hours = int(seconds / 3600)
        mins = int((seconds / 3600) % 1 * 60)
        return f"{hours:02d}:{mins:02d}$\\,$h"


# SR_TMP <<<
def format_unit(s):
    """Auto-format the unit by elevating superscripts etc."""
    old_new = [
        ("m-2", "m$^{-2}$"),
        ("m-3", "m$^{-3}$"),
        ("s-1", "s$^{-1}$"),
    ]
    for old, new in old_new:
        s = s.replace(old, new)
    return s


@summarizable(attrs=["type_", "name", "attrs", "_attr_lst"])
class AttrMult:
    def __init__(self, type_, *, name, values, attrs=None):
        assert attrs is not None  # SR_TMP

        self.type_ = type_
        self.name = name
        self.attrs = attrs or {}

        assert isiterable(values, str_ok=False)

        # Initialize individual attributes
        self._attr_lst = [
            Attr[type_](name=name, value=value, attrs=attrs) for value in values
        ]

    @property
    def value(self):
        return [a.value for a in self._attr_lst]

    def format(self, fmt=None, *, join=" / ", **kwargs):
        return join.join([a.format(fmt, **kwargs) for a in self._attr_lst])

    # SR_TMP <<< quick'n'dirty
    def __repr__(self):
        from pprint import pformat  # isort:skip

        return pformat(self.summarize())

    def __eq__(self, other):
        if type(self) is not type(other):  # noqa
            return False
        if self.type_ is not other.type_:  # noqa
            return False
        if self.name != other.name:
            return False
        if self.attrs != other.attrs:
            return False
        if self._attr_lst != other._attr_lst:
            return False
        return True


@summarizable
class AttrGroup:
    """Base class for attributes."""

    def __init__(self, setup, **kwargs):
        if type(self) is AttrGroup:
            raise ValueError(
                f"{type(self).__name__} should be subclassed, not instatiated"
            )
        self.setup = setup

    def __eq__(self, other):
        return self.setup == other.setup

    # SR_DBG <<<
    def __iter__(self):
        raise DeprecationWarning()

    # SR_TODO <<< eliminate iterator method
    def iter_attr_items(self):
        result = []  # SR_DBG
        if isinstance(self, BaseModel):  # SR_TMP
            for attr in self.__fields__:
                if attr != "setup":
                    # yield attr, getattr(self, attr)
                    result.append((attr, getattr(self, attr)))  # SR_DBG
        else:
            for attr in dir(self):
                if not (
                    attr.startswith("_") or attr in ["setup", "summarizable_attrs"]
                ):
                    value = getattr(self, attr)
                    if not callable(value):
                        # yield attr, value
                        result.append((attr, value))  # SR_DBG
        return result  # SR_DBG

    def merge_with(self, others, replace=None):
        """Create new instance by merging self and others.

        Args:
            others (list[AttrGroup]): Other instances of the same attributes
                class, to be merged with this one.

            replace (dict, optional): Attributes to be replaced in the merged
                instance. Must contain all attributes that differ between any
                of the instances to be merged. Defaults to '{}'.

        Returns:
            AttrGroup: Merged instance derived from ``self`` and ``others``.
                Note that no input instance is changed.

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
        setup = Setup.compress(SetupCollection([self.setup] + other_setups))

        kwargs = {}
        for attr, value in self.iter_attr_items():
            other_values = [getattr(o, attr) for o in others]
            kwargs[attr] = value.merge_with(other_values, replace=replace.get(attr))

        return type(self)(setup=setup, **kwargs)


def init_attr(type_, name, attrs=None):
    def f(value):
        if not isinstance(value, (Attr, AttrMult)):
            value = Attr[type_](name=name, value=value, attrs=attrs)
        return value

    return validator(name, pre=True, allow_reuse=True)(f)


class AttrGroupGrid(BaseModel, AttrGroup):
    """Grid attributes.

    Attributes:
        north_pole_lat: Latitude of rotated north pole.

        north_pole_lon: Longitude of rotated north pole.

    """

    setup: Setup
    north_pole_lat: Attr[float]
    north_pole_lon: Attr[float]

    _init_north_pole_lat = init_attr(float, "north_pole_lat")
    _init_north_pole_lon = init_attr(float, "north_pole_lon")


class AttrGroupVariable(BaseModel, AttrGroup):
    """Variable attributes.

    Attributes:
        name: Name of variable.

        unit: Unit of variable as a string (e.g., 'm-3' for cubic meters).

        level_bot: Bottom level value(s).

        level_bot_unit: Bottom level unit.

        level_top: Top level value(s).

        level_top_unit: Bottom level unit.

    """

    setup: Setup
    # long_name: Attr[str]
    # short_name: Attr[str]
    # unit: Attr[str]
    # level_bot: Attr[float]
    # level_top: Attr[float]
    # level_bot_unit: Attr[str]
    # level_top_unit: Attr[str]
    long_name: Union[Attr[str], AttrMult]
    short_name: Union[Attr[str], AttrMult]
    unit: Union[Attr[str], AttrMult]
    level_bot: Union[Attr[float], AttrMult]
    level_top: Union[Attr[float], AttrMult]
    level_bot_unit: Union[Attr[str], AttrMult]
    level_top_unit: Union[Attr[str], AttrMult]

    _init_long_name = init_attr(str, "long_name")
    _init_short_name = init_attr(str, "short_name")
    _init_unit = init_attr(str, "unit")
    _init_level_bot = init_attr(float, "level_bot")
    _init_level_top = init_attr(float, "level_top")
    _init_level_bot_unit = init_attr(str, "level_bot_unit")
    _init_level_top_unit = init_attr(str, "level_top_unit")

    class Config:  # noqa
        arbitrary_types_allowed = True  # SR_TMP AttrMult
        extra = "forbid"

    def fmt_level_unit(self):
        unit_bottom = format_unit(self.level_bot_unit.value)
        unit_top = format_unit(self.level_top_unit.value)
        if unit_bottom != unit_top:
            raise Exception(
                f"bottom and top level units differ: '{unit_bottom}' != '{unit_top}'"
            )
        return unit_top

    def fmt_level_range(self):

        if (self.level_bot.value, self.level_top.value) == (-1, -1):
            return None

        def fmt(bot, top, unit_fmtd=self.fmt_level_unit()):
            s = f"{bot:g}" + r"$-$" + f"{top:g}"
            if unit_fmtd:
                s += f" {unit_fmtd}"
            return s

        try:
            # Single level range
            return fmt(self.level_bot.value, self.level_top.value)
        except TypeError:
            pass
        # -- Multiple level ranges

        bots, tops, n = self._get_range_params()

        if n == 2:
            if tops[0] == bots[1]:
                return fmt(bots[0], tops[1])
            else:
                return f"{fmt(bots[0], tops[0], None)} + {fmt(bots[1], tops[1])}"
                raise NotImplementedError(f"2 non-continuous level ranges")

        elif n == 3:
            if tops[0] == bots[1] and tops[1] == bots[2]:
                return fmt(bots[0], tops[2])
            else:
                raise NotImplementedError(f"3 non-continuous level ranges")

        else:
            raise NotImplementedError(f"{n} sets of levels")

    def _get_range_params(self):
        try:
            bots = sorted(self.level_bot.value)
            tops = sorted(self.level_top.value)
        except TypeError:
            raise  # SR_TMP TODO proper error message
        else:
            if len(bots) != len(tops):
                raise Exception(f"inconsistent no. levels: {len(bots)} != {len(tops)}")
            n = len(bots)
        return bots, tops, n


class AttrGroupRelease(BaseModel, AttrGroup):
    """Release attributes.

    Attributes:
        site_name: Name of release site.

        site_lat: Latitude of release site.

        site_lon: Longitude of release site.

        height: Release height value(s).

        rate: Release rate value(s).

        mass: Release mass value(s).

        height_unit: Release height unit

        rate_unit: Release rate unit.

        mass_unit: Release mass unit.

        start: Start of the release.

        end: End of the release.

    """

    setup: Setup
    # site_name: Attr[str]
    # site_lat: Attr[float]
    # site_lon: Attr[float]
    # height: Attr[float]
    # rate: Attr[float]
    # mass: Attr[float]
    # height_unit: Attr[str]
    # rate_unit: Attr[str]
    # mass_unit: Attr[str]
    # start: Attr[datetime.datetime]
    # end: Attr[datetime.datetime]
    site_name: Union[Attr[str], AttrMult]
    site_lat: Union[Attr[float], AttrMult]
    site_lon: Union[Attr[float], AttrMult]
    height: Union[Attr[float], AttrMult]
    rate: Union[Attr[float], AttrMult]
    mass: Union[Attr[float], AttrMult]
    height_unit: Union[Attr[str], AttrMult]
    rate_unit: Union[Attr[str], AttrMult]
    mass_unit: Union[Attr[str], AttrMult]
    start: Union[Attr[datetime.datetime], AttrMult]
    end: Union[Attr[datetime.datetime], AttrMult]

    _init_site_name = init_attr(str, "site_name")
    _init_site_lat = init_attr(float, "site_lat")
    _init_site_lon = init_attr(float, "site_lon")
    _init_height = init_attr(float, "height")
    _init_rate = init_attr(float, "rate")
    _init_mass = init_attr(float, "mass")
    _init_height_unit = init_attr(str, "height_unit")
    _init_rate_unit = init_attr(str, "rate_unit")
    _init_mass_unit = init_attr(str, "mass_unit")
    _init_start = init_attr(datetime.datetime, "start")
    _init_end = init_attr(datetime.datetime, "end")

    @root_validator
    def _set_start(cls, values):
        values["start"].attrs["start"] = values["start"].value
        values["end"].attrs["start"] = values["start"].value
        return values

    class Config:  # noqa
        arbitrary_types_allowed = True  # SR_TMP AttrMult
        extra = "forbid"


class AttrGroupSpecies(BaseModel, AttrGroup):
    """Species attributes.

    Attributes:
        name: Species name.

        half_life: Half life value(s).

        half_life_unit: Half life unit.

        deposit_vel: Deposition velocity value(s).

        deposit_vel_unit: Deposition velocity unit.

        sediment_vel: Sedimentation velocity value(s).

        sediment_vel_unit: Sedimentation velocity unit.

        washout_coeff: Washout coefficient value(s).

        washout_coeff_unit: Washout coefficient unit.

        washout_exponent: Washout exponent value(s).

    """

    setup: Setup
    # name: Attr[str]
    # half_life: Attr[float]
    # deposit_vel: Attr[float]
    # sediment_vel: Attr[float]
    # washout_coeff: Attr[float]
    # washout_exponent: Attr[float]
    # half_life_unit: Attr[str]
    # deposit_vel_unit: Attr[str]
    # sediment_vel_unit: Attr[str]
    # washout_coeff_unit: Attr[str]
    name: Union[Attr[str], AttrMult]
    half_life: Union[Attr[float], AttrMult]
    deposit_vel: Union[Attr[float], AttrMult]
    sediment_vel: Union[Attr[float], AttrMult]
    washout_coeff: Union[Attr[float], AttrMult]
    washout_exponent: Union[Attr[float], AttrMult]
    half_life_unit: Union[Attr[str], AttrMult]
    deposit_vel_unit: Union[Attr[str], AttrMult]
    sediment_vel_unit: Union[Attr[str], AttrMult]
    washout_coeff_unit: Union[Attr[str], AttrMult]

    _init_name = init_attr(str, "name")
    _init_half_life = init_attr(float, "half_life")
    _init_deposit_vel = init_attr(float, "deposit_vel")
    _init_sediment_vel = init_attr(float, "sediment_vel")
    _init_washout_coeff = init_attr(float, "washout_coeff")
    _init_washout_exponent = init_attr(float, "washout_exponent")
    _init_half_life_unit = init_attr(str, "half_life_unit")
    _init_deposit_vel_unit = init_attr(str, "deposit_vel_unit")
    _init_sediment_vel_unit = init_attr(str, "sediment_vel_unit")
    _init_washout_coeff_unit = init_attr(str, "washout_coeff_unit")

    class Config:  # noqa
        arbitrary_types_allowed = True  # SR_TMP AttrMult
        extra = "forbid"


class AttrGroupSimulation(BaseModel, AttrGroup):
    """Simulation attributes.

    Attributes:
        model_name: Name of the model.

        start: Start of the simulation.

        end: End of the simulation.

        now: Current timestep.

        integr_start: Start of the integration period.

        integr_type: Type of integration (or reduction).

    """

    setup: Setup
    # model_name: Attr[str]
    # start: Attr[datetime.datetime]
    # end: Attr[datetime.datetime]
    # now: Attr[datetime.datetime]
    # integr_start: Attr[datetime.datetime]
    # integr_type: Attr[str]
    model_name: Union[Attr[str], AttrMult]
    start: Union[Attr[datetime.datetime], AttrMult]
    end: Union[Attr[datetime.datetime], AttrMult]
    now: Union[Attr[datetime.datetime], AttrMult]
    integr_start: Union[Attr[datetime.datetime], AttrMult]
    integr_type: Union[Attr[str], AttrMult]

    _init_model_name = init_attr(str, "model_name")
    _init_start = init_attr(datetime.datetime, "start")
    _init_end = init_attr(datetime.datetime, "end")
    _init_now = init_attr(datetime.datetime, "now")
    _init_integr_start = init_attr(datetime.datetime, "integr_start")
    _init_integr_type = init_attr(str, "integr_type")

    @root_validator
    def _set_start(cls, values):
        values["start"].attrs["start"] = values["start"].value
        values["end"].attrs["start"] = values["start"].value
        values["now"].attrs["start"] = values["start"].value
        values["integr_start"].attrs["start"] = values["start"].value
        return values

    class Config:  # noqa
        arbitrary_types_allowed = True  # SR_TMP AttrMult
        extra = "forbid"

    def fmt_integr_period(self):
        integr_period = self.now.value - self.integr_start.value
        return f"{integr_period.total_seconds()/3600:g}$\\,$h"


class AttrGroupCollection:
    """Collection of FLEXPART attributes."""

    def __init__(self, setup, *, grid, variable, release, species, simulation):
        """Initialize an instance of ``AttrGroupCollection``.

        Args:
            setup (Setup): Setup.

        Kwargs:
            grid (dict): Kwargs passed to ``AttrGroupGrid``.

            variable (dict): Kwargs passed to ``AttrGroupVariable``.

            release (dict): Kwargs passed to ``AttrGroupRelease``.

            species (dict): Kwargs passed to ``AttrGroupSpecies``.

            simulation (dict): Kwargs passed to ``AttrGroupSimulation``.

        """
        self.setup = setup
        self._attrs = {}
        self.add("grid", grid)
        self.add("variable", variable)
        self.add("release", release)
        self.add("species", species)
        self.add("simulation", simulation)

    def __eq__(self, other):
        return self._attrs == other._attrs

    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__}.{name}")

    def add(self, name, attrs):

        if not isinstance(attrs, AttrGroup):
            cls_by_name = {
                "grid": AttrGroupGrid,
                "variable": AttrGroupVariable,
                "release": AttrGroupRelease,
                "species": AttrGroupSpecies,
                "simulation": AttrGroupSimulation,
            }
            try:
                cls_group = cls_by_name[name]
            except KeyError:
                raise ValueError(f"missing AttrGroup class for name '{name}'")
            attrs = cls_group(setup=self.setup, **attrs)

        self._attrs[name] = attrs

    def merge_with(self, others, **replace):
        """Create a new instance by merging this and others.

        Args:
            others (list[AttrGroupCollection]): Other instances to be merged
                with this.

            **replace (dicts): Collections of attributes to be replaced in the
                merged ``AttrGroup*`` instances of the shared collection
                instance. Must contain all attributes that differ between any
                of the collections.

        Returns:
            AttrGroupCollection: New attributes collection instance derived
                from ``self`` and ``others``. Note that no input collection is
                changed.

        Example:
            In this example, all attributes collection in a list are merged and
            the variable and release site names replaced, implying those differ
            between the collections. Wether that is the case or not, all other
            attributes must be the same, otherwise an error is issued.

            attrs_coll = attrs_colls[0].merge_with(
                attrs_colls[1:],
                variable={
                    'long_name': 'X Sum',
                    'short_name': 'x_sum',
                },
                release={
                    'site_name': 'source',
                },
            )

        """
        kwargs = {"setup": self.setup}
        for name in sorted(self._attrs.keys()):
            other_attrs = [o._attrs[name] for o in others]
            merged = self._attrs[name].merge_with(other_attrs, replace.get(name))
            kwargs[name] = dict(merged.iter_attr_items())
        return type(self)(**kwargs)

    def __iter__(self):
        for name, attr in sorted(self._attrs.items()):
            yield name, attr

    def asdict(self):
        return {name: dict(attrs) for name, attrs in self}


def collect_attrs(fi, setup):
    return AttrsCollector(fi, setup, WORDS).run()


class AttrsCollector:
    """Collect attributes for a field from an open NetCDF file."""

    def __init__(self, fi, setup, words):
        """Create an instance of ``AttrsCollector``.

        Args:
            fi (netCDF4.Dataset): An open FLEXPART NetCDF file.

            setup (Setup): Setup.

            words (Words): Words.

        """
        self.fi = fi
        self.setup = setup
        self._words = words

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
        self.ncattrs_field = self.ncattrs_vars[nc_var_name(self.setup)]

    def run(self):
        """Collect attributes."""

        attrs = {}
        attrs["simulation"] = self.collect_simulation_attrs(attrs)
        attrs["grid"] = self.collect_grid_attrs(attrs)
        attrs["release"] = self.collect_release_attrs(attrs)
        attrs["species"] = self.collect_species_attrs(attrs)
        attrs["variable"] = self.collect_variable_attrs(attrs)

        return AttrGroupCollection(self.setup, **attrs)

    def collect_simulation_attrs(self, attrs):
        """Collect simulation attributes."""

        model_name = "COSMO-?"  # SR_HC

        # Start and end timesteps of simulation
        _ga = self.ncattrs_global
        ts_start = datetime.datetime(
            *time.strptime(_ga["ibdate"] + _ga["ibtime"], "%Y%m%d%H%M%S")[:6],
            tzinfo=datetime.timezone.utc,
        )
        ts_end = datetime.datetime(
            *time.strptime(_ga["iedate"] + _ga["ietime"], "%Y%m%d%H%M%S")[:6],
            tzinfo=datetime.timezone.utc,
        )

        # Current time step and start time step of current integration period
        ts_now, ts_integr_start = self._get_current_timestep_etc()

        # Type of integration (or, rather, reduction)
        if self.setup.variable == "concentration":  # SR_TMP
            integr_type = "sum" if self.setup.integrate else "mean"
        elif self.setup.variable == "deposition":  # SR_TMP
            integr_type = "accum" if self.setup.integrate else "mean"
        else:
            raise NotImplementedError(
                f"no integration type specified for '{self.name}'"
            )

        return {
            "model_name": model_name,
            "start": ts_start,
            "end": ts_end,
            "now": ts_now,
            "integr_start": ts_integr_start,
            "integr_type": integr_type,
        }

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
        start = datetime.datetime(
            int(match["yyyy"]),
            int(match["mm"]),
            int(match["dd"]),
            int(match["HH"]),
            int(match["MM"]),
            tzinfo=datetime.timezone.utc,
        )

        # Determine time since start
        delta_tot = datetime.timedelta(seconds=int(var[idx]))

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

    def collect_grid_attrs(self, attrs):
        """Collect grid attributes."""

        np_lat = self.ncattrs_vars["rotated_pole"]["grid_north_pole_latitude"]
        np_lon = self.ncattrs_vars["rotated_pole"]["grid_north_pole_longitude"]

        return {
            "north_pole_lat": np_lat,
            "north_pole_lon": np_lon,
        }

    def collect_release_attrs(self, attrs):
        """Collect release point attributes."""

        # Collect release point information
        # SR_TMP < TODO clean up once CoreSetup has been implemented
        assert len(self.setup.numpoint) == 1
        idx = next(iter(self.setup.numpoint))
        # idx = self.setup.numpoint
        # SR_TMP >
        numpoint = ReleasePoint.from_file(self.fi, idx)

        sim_start = attrs["simulation"]["start"]
        start = sim_start + datetime.timedelta(seconds=numpoint.rel_start)
        end = sim_start + datetime.timedelta(seconds=numpoint.rel_end)

        site_lat = np.mean([numpoint.lllat, numpoint.urlat])
        site_lon = np.mean([numpoint.lllon, numpoint.urlon])
        site_name = numpoint.name
        site_name = {"Goesgen": r"G$\mathrm{\"o}$sgen"}.get(site_name)

        height = np.mean([numpoint.zbot, numpoint.ztop])
        height_unit = self._words["m_agl", self.setup.lang].s

        assert len(numpoint.ms_parts) == 1
        mass = next(iter(numpoint.ms_parts))
        mass_unit = "Bq"  # SR_HC

        duration = numpoint.rel_end - numpoint.rel_start
        duration_unit = "s"  # SR_HC

        rate = mass / duration
        rate_unit = f"{mass_unit} {duration_unit}-1"

        return {
            "start": start,
            "end": end,
            "site_lat": site_lat,
            "site_lon": site_lon,
            "site_name": site_name,
            "height": height,
            "height_unit": height_unit,
            "rate": rate,
            "rate_unit": rate_unit,
            "mass": mass,
            "mass_unit": mass_unit,
        }

    def collect_variable_attrs(self, attrs):
        """Collect variable attributes."""

        # Variable names
        long_name = self._long_name()
        short_name = self._short_name()

        # Variable unit
        unit = self.ncattrs_field["units"]

        # SR_TMP < TODO clean up once CoreSetup has been implemented
        assert self.setup.level is None or len(self.setup.level) == 1
        idx = None if self.setup.level is None else next(iter(self.setup.level))
        # idx = self.setup.level
        # SR_TMP >
        if idx is None:
            level_unit = ""
            level_bot = -1
            level_top = -1
        else:
            level_unit = self._words["m_agl", self.setup.lang].s
            _var = self.fi.variables["level"]
            level_bot = 0.0 if idx == 0 else float(_var[idx - 1])
            level_top = float(_var[idx])

        return {
            "long_name": long_name,
            "short_name": short_name,
            "unit": unit,
            "level_bot": level_bot,
            "level_bot_unit": level_unit,
            "level_top": level_top,
            "level_top_unit": level_unit,
        }

    def collect_species_attrs(self, attrs):
        """Collect species attributes."""

        substance = self._get_substance()

        # Get deposition and washout data
        name_core = nc_var_name(self.setup)
        if self.setup.variable == "deposition":  # SR_TMP
            name_core = name_core[3:]
        deposit_vel = self.ncattrs_vars[f"DD_{name_core}"]["dryvel"]
        washout_coeff = self.ncattrs_vars[f"WD_{name_core}"]["weta"]
        washout_exponent = self.ncattrs_vars[f"WD_{name_core}"]["wetb"]

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

        return {
            "name": substance,
            "half_life": half_life,
            "half_life_unit": half_life_unit,
            "deposit_vel": deposit_vel,
            "deposit_vel_unit": deposit_vel_unit,
            "sediment_vel": 0.0,
            "sediment_vel_unit": sediment_vel_unit,
            "washout_coeff": washout_coeff,
            "washout_coeff_unit": washout_coeff_unit,
            "washout_exponent": washout_exponent,
        }

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
        if plot_type in ["affected_area", "affected_area_mono"]:
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
def nc_var_name(setup):
    result = []
    for species_id in setup.species_id:
        if setup.variable == "concentration":
            result.append(f"spec{species_id:03d}")
        elif setup.variable == "deposition":
            prefix = {"wet": "WD", "dry": "DD"}[setup.deposition_type]
            result.append(f"{prefix}_spec{species_id:03d}")
    return result[0] if len(result) == 1 else result

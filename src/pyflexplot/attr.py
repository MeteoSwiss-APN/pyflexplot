# -*- coding: utf-8 -*-
"""
Attributes.
"""
# Standard library
import datetime
import re
import time
from pprint import pformat

# Third-party
import numpy as np

# First-party
from srutils.dict import format_dictlike
from srutils.various import isiterable

# Local
from .utils import summarizable


@summarizable
class Attr:
    """Individual attribute."""

    def __init__(self, name, value, type_=None, unit=None, group=None):

        self.name = name
        self.type_ = type(value) if type_ is None else type_
        self.unit = unit

        self.set_group(group)

        # SR_TMP <
        assert not isinstance(value, type(self)), f"value type {type(value).__name__}"
        assert not isiterable(value, str_ok=False), f"iterable value: {value}"
        # SR_TMP >

        # Ensure consistency of value and type_
        if isinstance(value, type_):
            self.value = value
        else:
            try:
                self.value = type_(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"value='{value}' of type '{type(value).__name__}' incompatible "
                    f"with type_='{type_.__name__}' ({type(e).__name__}: {e})"
                )

    @classmethod
    def multiple(cls, *args, **kwargs):
        return AttrMult(*args, cls_attr=cls, **kwargs)

    def set_group(self, group):
        self._group = group

    # SR_TMP <<<
    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        if isinstance(self.value, str):
            value = f"'{self.value}'"
        else:
            value = str(self.value)
        if self.unit is None:
            unit = "None"
        else:
            unit = f"'{self.unit}'"
        return (
            f"{type(self).__name__}("
            + ", ".join(
                [
                    f"name='{self.name}'",
                    f"value={value}",
                    f"type_={self.type_.__name__}",
                    f"unit={unit}",
                ]
            )
            + f")"
        )

    def merge_with(self, others, replace=None):
        if replace is None:
            replace = {}
        name = replace.get("name", self.name)
        type_ = replace.get("type_", self.type_)
        values, units = self._collect_values_units(others, replace)
        value, unit = self._reduce_values_units(values, units)
        value = replace.get("value", value)
        unit = replace.get("unit", unit)
        kwargs = {
            "name": name,
            "type_": type_,
            "value": value,
            "unit": unit,
        }
        if isiterable(value, str_ok=False):
            return type(self).multiple(**kwargs)
        return type(self)(**kwargs)

    def _collect_values_units(self, others, replace):
        values_units = [(self.value, self.unit)]
        values, units = [self.value], [self.unit]
        for other in others:
            if "name" not in replace and other.name != self.name:
                raise ValueError(f"names differ: {other.name} != {self.name}")
            if "name" not in replace and other.type_ != self.type_:
                raise ValueError(f"types differ: {other.type_} != {self.type_}")
            if (other.value, other.unit) not in values_units:
                values_units.append((other.value, other.unit))
                values.append(other.value)
                units.append(other.unit)
        return values, units

    def _reduce_values_units(self, values, units):
        if len(units) == 1:
            # One value, one unit
            return next(iter(values)), next(iter(units))
        # Multiple values, one or equally many units
        # SR_TMP<
        if self.name == "long_name":
            name0 = next(iter(values))
            if name0.lower().startswith("beaufschlagtes gebiet"):
                values = name0.replace("nasse", "totale").replace("trockene", "totale")
            elif name0.lower().startswith("affected area"):
                values = name0.replace("wet", "total").replace("dry", "total")
            elif name0.lower().endswith("bodendeposition"):
                values = name0.replace("nasse", "totale").replace("trockene", "totale")
            elif name0.lower().endswith("surface deposition"):
                values = name0.replace("wet", "total").replace("dry", "total")
        # SR_TMP>
        value = values
        unit = next(iter(units)) if len(set(units)) == 1 else units
        return value, unit

    def format(self, fmt=None, *, escape_format=False, skip_unit=False, join=None):
        if fmt is None:
            if issubclass(self.type_, (float, int)):
                fmt = "g"
            else:
                fmt = ""
        s = f"{{:{fmt}}}".format(self.value)
        if self.name == "unit":
            s = self.fmt_unit(unit=s)
        if self.unit and not skip_unit:
            s += f" {self.fmt_unit()}"
        if escape_format:
            s = s.replace("{", "{{").replace("}", "}}")
        return s

    def fmt_unit(self, unit=None):
        """Auto-format the unit by elevating superscripts etc."""
        if unit is None:
            unit = self.unit
        old_new = [
            ("m-2", "m$^{-2}$"),
            ("m-3", "m$^{-3}$"),
            ("s-1", "s$^{-1}$"),
        ]
        for old, new in old_new:
            unit = unit.replace(old, new)
        return unit


class AttrMult(Attr):
    def __init__(self, name, value, type_=None, unit=None, cls_attr=Attr):

        self.name = name

        if not isiterable(value, str_ok=False):
            raise ValueError(f"value not iterable: {value}")

        if not isiterable(unit, str_ok=False):
            unit = [unit for _ in value]

        if len(unit) != len(value):
            raise ValueError(
                f"numbers of values and units differ: "
                f"{len(value)} != {len(unit)}"
                f"\nvalue: {value}\nunit: {unit}"
            )

        if type_ is None:
            types = sorted(set([type(v) for v in value]))
            if len(types) > 1:
                raise ValueError(
                    f"'type_' omitted when types of values differ: {types}"
                )
            type_ = next(iter(types))

        self.type_ = type_

        values, units = value, unit
        del value, unit

        # Initialize individual attributes
        self._attr_lst = []
        for value, unit in zip(values, units):
            attr = cls_attr(name, value, type_=type_, unit=unit)
            self._attr_lst.append(attr)

    def __repr__(self):
        s = f"{type(self).__name__}("
        s += f"name='{self.name}', "
        s += f"value={self.value}, "
        s += f"type_={self.type_.__name__}, "
        if isinstance(self.unit, str):
            s += f"unit='{self.unit}')"
        else:
            s += f"unit={self.unit})"
        return s

    @property
    def value(self):
        return [a.value for a in self._attr_lst]

    @property
    def unit(self):
        units = [a.unit for a in self._attr_lst]
        if len(set(units)) == 1:
            return units[0]
        return units

    def format(self, fmt=None, *, join=" / ", **kwargs):
        return join.join([a.format(fmt, **kwargs) for a in self._attr_lst])

    def fmt_unit(self, unit=None):
        units = [a.fmt_unit(unit=unit) for a in self._attr_lst]
        if len(set(units)) == 1:
            return next(iter(units))
        return units


class AttrDatetime(Attr):
    """Individual datetime attribute."""

    def __init__(self, name, value, *, type_=None, start=None, **kwargs):

        if type_ is None:
            type_ = datetime.datetime

        if type_ is not datetime.datetime:
            raise ValueError(
                f"invalid type_ '{type_.__name__}' (not datetime.datetime)"
            )

        if not isinstance(value, type_):
            raise ValueError(
                f"value has wrong type: expected '{type_.__name__}', got "
                f"'{type(value).__name__}'"
            )

        super().__init__(name, value, type_=type_, **kwargs)

        self.start = start.value if isinstance(start, Attr) else start

    def __repr__(self):
        s = super().__repr__()
        s = s[:-1] + f", start={self.start}" + s[-1]
        return s

    def merge_with(self, others, **kwargs):
        attr = super().merge_with(others, **kwargs)
        starts = sorted(set([self.start] + [o.start for o in others]))
        if len(starts) != 1:
            raise ValueError(
                f"cannot merge with {len(others)} other instances of "
                f"{type(self).__name__}: starts differ: {starts}"
            )
        attr.start = next(iter(starts))
        return attr

    def format(self, rel=False, rel_start=None, rel_neg_ok=True):
        """Format a datetime object to a string."""
        if not rel:
            return self.value.strftime("%Y-%m-%d %H:%M %Z")
        if rel_start is None:
            if self.start is None:
                raise ValueError(
                    f"{self.name}: relative formatting failed: missing start"
                )
            rel_start = self.start
        seconds = (self.value - rel_start).total_seconds()
        if not rel_neg_ok and seconds < 0:
            seconds = -seconds
        hours = int(seconds / 3600)
        mins = int((seconds / 3600) % 1 * 60)
        return f"{hours:02d}:{mins:02d}$\\,$h"


class AttrGroup:
    """Base class for attributes."""

    def __init__(self, *args, lang=None, **kwargs):
        if type(self) is AttrGroup:
            raise ValueError(
                f"{type(self).__name__} should be subclassed, not instatiated"
            )
        self.lang = lang or "en"
        self.reset()

    def reset(self):
        self._attrs = {}

    def __eq__(self, other):
        return dict(self) == dict(other)

    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__}.{name}")

    # SR_TMP <<< TODO eliminate
    def set(self, name, value, type_, **kwargs):
        if isinstance(value, Attr):
            attr = value
            attr.set_group(self)
        else:
            cls = {datetime.datetime: AttrDatetime}.get(type_, Attr)
            kwargs.update({"name": name, "value": value, "type_": type_, "group": self})
            try:
                attr = cls(**kwargs)
            except Exception as e:
                raise Exception(
                    f"error creating instance of {cls.__name__} "
                    f"({type(e).__name__}: {e})"
                )
        self._attrs[name] = attr

    def __iter__(self):
        for name, attr in sorted(self._attrs.items()):
            yield name, attr

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

        # Language
        langs = sorted(set([self.lang] + [o.lang for o in others]))
        if len(langs) == 1:
            lang = next(iter(langs))
        else:
            raise Exception(f"multiple languages: {langs}")

        kwargs = {"lang": lang}
        for key, attr0 in iter(self):
            attrs1 = [getattr(o, key) for o in others]
            attr = attr0.merge_with(attrs1, replace=replace.get(key))
            kwargs[key] = attr
            if attr.unit is not None:
                kwargs[f"{key}_unit"] = attr.unit

        try:
            return type(self)(**kwargs)
        except Exception as e:
            raise Exception(
                f"error creating instance of {type(self).__name__} "
                f"({type(e).__name__}: {e})\n\n{len(kwargs)} kwargs:\n{pformat(kwargs)}"
            )


class AttrGroupGrid(AttrGroup):
    """Grid attributes."""

    group_name = "grid"

    def __init__(self, *, north_pole_lat, north_pole_lon, **kwargs):
        """Initialize instance of ``AttrGroupGrid``.

        Kwargs:
            north_pole_lat (float): Latitude of rotated north pole.

            north_pole_lon (float): Longitude of rotated north pole.

        """
        super().__init__(**kwargs)
        self.set("north_pole_lat", north_pole_lat, float)
        self.set("north_pole_lon", north_pole_lon, float)


class AttrGroupVariable(AttrGroup):
    """Variable attributes."""

    group_name = "variable"

    def __init__(
        self,
        *,
        long_name,
        short_name,
        unit,
        level_bot,
        level_bot_unit,
        level_top,
        level_top_unit,
        **kwargs,
    ):
        """Initialize an instance of ``AttrGroupVariable``.

        Kwargs:
            name (str): Name of variable.

            unit (str): Unit of variable as a regular string (e.g., 'm-3' for
                cubic meters).

            level_bot (float or list[float]): Bottom level value(s).

            level_bot_unit (str): Bottom level unit.

            level_top (float or list[float]): Top level value(s).

            level_top_unit (str): Bottom level unit.

        """
        super().__init__(**kwargs)
        self.set("long_name", long_name, str)
        self.set("short_name", short_name, str)
        self.set("unit", unit, str)
        self.set("level_bot", level_bot, float, unit=level_bot_unit)
        self.set("level_top", level_top, float, unit=level_top_unit)

    def fmt_level_unit(self):
        unit_bottom = self.level_bot.fmt_unit()
        unit_top = self.level_top.fmt_unit()
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


class AttrGroupRelease(AttrGroup):
    """Release attributes."""

    group_name = "release"

    def __init__(
        self,
        *,
        start,
        end,
        site_lat,
        site_lon,
        site_name,
        height,
        height_unit,
        rate,
        rate_unit,
        mass,
        mass_unit,
        **kwargs,
    ):
        """Initialize an instance of ``AttrGroupRelease``.

        Kwargs:
            start (datetime): Start of the release.

            end (datetime): End of the release.

            site_lat (float): Latitude of release site.

            site_lon (float): Longitude of release site.

            site_name (str): Name of release site.

            height (float or list[float]): Release height value(s).

            height_unit (str): Release height unit

            rate (float or list[float]): Release rate value(s).

            rate_unit (str): Release rate unit.

            mass (float or list[float]): Release mass value(s).

            mass_unit (str): Release mass unit.

        """
        super().__init__(**kwargs)
        self.set("start", start, datetime.datetime, start=start)
        self.set("end", end, datetime.datetime, start=start)
        self.set("site_lat", site_lat, float)
        self.set("site_lon", site_lon, float)
        self.set("site_name", site_name, str)
        self.set("height", height, float, unit=height_unit)
        self.set("rate", rate, float, unit=rate_unit)
        self.set("mass", mass, float, unit=mass_unit)


class AttrGroupSpecies(AttrGroup):
    """Species attributes."""

    group_name = "species"

    def __init__(
        self,
        *,
        name,
        half_life,
        half_life_unit,
        deposit_vel,
        deposit_vel_unit,
        sediment_vel,
        sediment_vel_unit,
        washout_coeff,
        washout_coeff_unit,
        washout_exponent,
        **kwargs,
    ):
        """Create an instance of ``AttrGroupSpecies``.

        Kwargs:
            name (str): Species name.

            half_life (float or list[float]): Half life value(s).

            half_life_unit (str): Half life unit.

            deposit_vel (float or list[float]): Deposition velocity value(s).

            deposit_vel_unit (str): Deposition velocity unit.

            sediment_vel (float or list[float]): Sedimentation velocity
                value(s).

            sediment_vel_unit (str): Sedimentation velocity unit.

            washout_coeff (float or list[float]): Washout coefficient value(s).

            washout_coeff_unit (str): Washout coefficient unit.

            washout_exponent (float or list[float]): Washout exponent value(s).

        """
        super().__init__(**kwargs)
        self.set("name", name, str)
        self.set("half_life", half_life, float, unit=half_life_unit)
        self.set("deposit_vel", deposit_vel, float, unit=deposit_vel_unit)
        self.set("sediment_vel", sediment_vel, float, unit=sediment_vel_unit)
        self.set("washout_coeff", washout_coeff, float, unit=washout_coeff_unit)
        self.set("washout_exponent", washout_exponent, float)


class AttrGroupSimulation(AttrGroup):
    """Simulation attributes."""

    group_name = "simulation"

    def __init__(
        self, *, model_name, start, end, now, integr_start, integr_type, **kwargs
    ):
        """Create an instance of ``AttrGroupSimulation``.

        Kwargs:
            model_name (str): Name of the model.

            start (datetime): Start of the simulation.

            end (datetime): End of the simulation.

            now (datetime): Current timestep.

            integr_start (datetime): Start of the integration period.

            integr_type (str): Type of integration (or reduction).

        """
        super().__init__(**kwargs)
        self.set("model_name", model_name, str)
        self.set("start", start, datetime.datetime, start=start)
        self.set("end", end, datetime.datetime, start=start)
        self.set("now", now, datetime.datetime, start=start)
        self.set("integr_start", integr_start, datetime.datetime, start=start)
        self.set("integr_type", integr_type, str)

    def fmt_integr_period(self):
        integr_period = self.now.value - self.integr_start.value
        return f"{integr_period.total_seconds()/3600:g}$\\,$h"


class AttrGroupCollection:
    """Collection of FLEXPART attributes."""

    def __init__(self, *, grid, variable, release, species, simulation, lang=None):
        """Initialize an instance of ``AttrGroupCollection``.

        Kwargs:
            grid (dict): Kwargs passed to ``AttrGroupGrid``.

            variable (dict): Kwargs passed to ``AttrGroupVariable``.

            release (dict): Kwargs passed to ``AttrGroupRelease``.

            species (dict): Kwargs passed to ``AttrGroupSpecies``.

            simulation (dict): Kwargs passed to ``AttrGroupSimulation``.

        """
        self.lang = lang or "en"
        self.reset()
        self.add("grid", grid)
        self.add("variable", variable)
        self.add("release", release)
        self.add("species", species)
        self.add("simulation", simulation)

    def reset(self):
        self._attrs = {}

    def __eq__(self, other):
        return dict(self) == dict(other)

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
                cls = cls_by_name[name]
            except KeyError:
                raise ValueError(f"missing AttrGroup class for name '{name}'")
            # SR_TMP <
            for key, attr in [(k, v) for k, v in attrs.items()]:
                if isinstance(attr, Attr) and attr.unit is not None:
                    attrs[f"{key}_unit"] = attr.unit
            # SR_TMP >
            attrs = cls(lang=self.lang, **attrs)

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
        kwargs = {"lang": self.lang}
        for name in sorted(self._attrs.keys()):
            kwargs[name] = dict(
                self._attrs[name].merge_with(
                    [o._attrs[name] for o in others], replace.get(name),
                )
            )
        return type(self)(**kwargs)

    def __iter__(self):
        for name, attr in sorted(self._attrs.items()):
            yield name, attr

    def asdict(self):
        return {name: dict(attrs) for name, attrs in self}


def collect_attrs(fi, var_specs, **kwargs):
    return AttrsCollector(fi, var_specs).run(**kwargs)


class AttrsCollector:
    """Collect attributes for a field from an open NetCDF file."""

    def __init__(self, fi, var_specs):
        """Create an instance of ``AttrsCollector``.

        Args:
            fi (netCDF4.Dataset): An open FLEXPART NetCDF file.

            var_specs (VarSpecs): Input field specifications.

        """
        self.fi = fi
        self._setup = var_specs._setup
        self._words = var_specs._words
        self.var_specs = var_specs

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
        self.ncattrs_field = self.ncattrs_vars[nc_var_name(self._setup)]

    def run(self, lang=None):
        """Collect attributes."""

        self.lang = lang or "en"

        attrs = {}
        attrs["simulation"] = self.collect_simulation_attrs(attrs)
        attrs["grid"] = self.collect_grid_attrs(attrs)
        attrs["release"] = self.collect_release_attrs(attrs)
        attrs["species"] = self.collect_species_attrs(attrs)
        attrs["variable"] = self.collect_variable_attrs(attrs)

        return AttrGroupCollection(lang=lang, **attrs)

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
        if self._setup.variable == "concentration":  # SR_TMP
            integr_type = "sum" if self._setup.integrate else "mean"
        elif self._setup.variable == "deposition":  # SR_TMP
            integr_type = "accum" if self._setup.integrate else "mean"
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
            assert len(self._setup.time_idcs) == 1  # SR_TMP
            idx = next(iter(self._setup.time_idcs))  # SR_TMP

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
        if self._setup.integrate:
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
        release_point = ReleasePoint.from_file(self.fi, self._setup.release_point_idx,)

        sim_start = attrs["simulation"]["start"]
        start = sim_start + datetime.timedelta(seconds=release_point.rel_start)
        end = sim_start + datetime.timedelta(seconds=release_point.rel_end)

        site_lat = np.mean([release_point.lllat, release_point.urlat])
        site_lon = np.mean([release_point.lllon, release_point.urlon])
        site_name = release_point.name
        site_name = {"Goesgen": r"G$\mathrm{\"o}$sgen"}.get(site_name)

        height = np.mean([release_point.zbot, release_point.ztop])
        height_unit = self._words["m_agl", self.lang].s

        assert len(release_point.ms_parts) == 1
        mass = next(iter(release_point.ms_parts))
        mass_unit = "Bq"  # SR_HC

        duration = release_point.rel_end - release_point.rel_start
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

        try:
            _i = self._setup.level_idx
        except AttributeError:
            # SR_TMP <
            level_unit = ""
            level_bot = -1
            level_top = -1
            # SR_TMP >
        else:
            level_unit = self._words["m_agl", self.lang].s
            _var = self.fi.variables["level"]
            level_bot = 0.0 if _i == 0 else float(_var[_i - 1])
            level_top = float(_var[_i])

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
        name_core = nc_var_name(self._setup)
        if self._setup.variable == "deposition":  # SR_TMP
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
        if self._setup.variable == "deposition":  # SR_TMP
            substance = substance.replace(
                f"_{self._setup.deposition_type}_deposition", ""
            )  # SR_HC
        return substance

    def _long_name(self, *, variable=None, plot_type=None):
        setup = self._setup
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
        setup = self._setup
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
        setup = self._setup
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
    if setup.variable == "concentration":
        if isinstance(setup.species_id, int):
            return f"spec{setup.species_id:03d}"
        return [f"spec{sid:03d}" for sid in setup.species_id]
    elif setup.variable == "deposition":
        prefix = {"wet": "WD", "dry": "DD"}[setup.deposition_type]
        return f"{prefix}_spec{setup.species_id:03d}"

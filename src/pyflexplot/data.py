# -*- coding: utf-8 -*-
"""
Data structures.
"""
import datetime
import logging as log
import numpy as np

from collections import namedtuple
from copy import copy, deepcopy

from .utils import isiterable

from .utils_dev import ipython  #SR_DEV


class FlexFieldRotPole:
    """FLEXPART data on rotated-pole grid."""

    def __init__(self, rlat, rlon, fld, attrs, field_specs, time_stats):
        """Create an instance of ``FlexFieldRotPole``.

        Args:
            rlat (ndarray[float]): Rotated latitude array (1D).

            rlon (ndarray[float]): Rotated longitude array (1D).

            fld (ndarray[float, float]): Field array (2D).

            attrs (FlexAttrsCollection): Attributes collection.

            field_specs (FlexFieldSpecs): Input field specifications.

            time_stats (dict): Some statistics across all time steps.

        """
        self.rlat = rlat
        self.rlon = rlon
        self.fld = fld
        self.attrs = attrs
        self.field_specs = field_specs
        self.time_stats = time_stats


class FlexAttr:
    """Individual attribute."""

    def __init__(self, name, value, type_=None, unit=None):
        self.name = name
        self.type_ = type(value) if type_ is None else type_
        self.unit = unit

        #SR_TMP< TODO move support for multiple values out of FlexAttr
        assert not isinstance(value, self.__class__)
        value_in = value
        values = value
        if not isiterable(value, str_ok=False):
            #SR_KEEP< TODO: remove TMP around it, but keep this
            if isinstance(value, type_):
                self.value = value
            else:
                try:
                    self.value = type_(value)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"value='{value}' of type '{type(value).__name__}' "
                        f"incompatible with type_='{type_.__name__}' "
                        f"({type(e).__name__}: {e})")
            #SR_KEEP>
        else:
            values_in = copy(value)
            self.value = []
            for value in values_in:
                #SR_TMP<
                if isinstance(value, self.__class__):
                    value = value.value
                #SR_TMP>
                if isinstance(value, type_):
                    self.value.append(value)
                else:
                    try:
                        value = type_(value)
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"value='{value}' of type '{type(value).__name__}' "
                            f"incompatible with type_='{type_.__name__}' "
                            f"({type(e).__name__}: {e})")
                    else:
                        self.value.append(value)
        #SR_TMP>

    #SR_TMP<<<
    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        if isinstance(self.value, str):
            value = f"'{self.value}'"
        else:
            value = str(self.value)
        if self.unit is None:
            unit = 'None'
        else:
            unit = f"'{self.unit}'"
        return (
            f"{type(self).__name__}("
            f"name='{self.name}', "
            f"value={value}, "
            f"type_={self.type_.__name__}, "
            f"unit={unit}"
            f")")

    def merge_with(self, others, replace=None):
        if replace is None:
            replace = {}

        # Name and type must be the same for all
        name = replace.get('name', self.name)
        type_ = replace.get('type_', self.type_)

        # Value and unit might differ
        values_units = [(self.value, self.unit)]
        values, units = [self.value], [self.unit]

        # Collect values and units
        for other in others:
            if 'name' not in replace and other.name != name:
                raise ValueError(f"names differ: {other.name} != {name}")
            if 'name' not in replace and other.type_ != type_:
                raise ValueError(f"types differ: {other.type_} != {type_}")

            if (other.value, other.unit) not in values_units:
                values_units.append((other.value, other.unit))
                values.append(other.value)
                units.append(other.unit)

        # Reduce values and units
        if len(units) == 1:
            # One value, one unit
            value = values[0]
            unit = units[0]
        else:
            # Multiple values, one or equally many units
            value = values
            unit = units[0] if len(set(units)) == 1 else units

        return self.__class__(
            name=name,
            type_=type_,
            value=replace.get('value', value),
            unit=replace.get('unit', unit),
        )

    def format_unit(self, unit=None):
        """Auto-format the unit by elevating superscripts etc."""
        if unit is None:
            unit = self.unit
        if isiterable(unit, str_ok=False):
            return [self.format_unit(u) for u in unit]
        old_new = [
            ('m-2', 'm$^{-2}$'),
            ('m-3', 'm$^{-3}$'),
            ('s-1', 's$^{-1}$'),
        ]
        for old, new in old_new:
            unit = unit.replace(old, new)
        return unit


class FlexAttrs:
    """Base class for attributes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._attrs = {}

    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(name) from None

    #SR_TMP<<< TODO eliminate
    def set(self, name, val, type_, unit=None):
        if not isinstance(val, FlexAttr):
            val = FlexAttr(name, val, type_, unit=unit)
        self._attrs[name] = val

    def __iter__(self):
        for name, attr in sorted(self._attrs.items()):
            yield name, attr

    def merge_with(self, others, replace=None):
        """Create new instance by merging self and others.

        Args:
            others (list[FlexAttrs]): Other instances of the same
                attributes class, to be merged with this one.

            replace (dict, optional): Attributes to be replaced in the
                merged instance. Must contain all attributes that
                differ between any of the instances to be merged.
                Defaults to '{}'.

        Returns:
            FlexAttrs: Merged instance derived from ``self`` and
                ``others``. Note that no input instance is changed.

        """
        if replace is None:
            replace = {}
        kwargs = {}
        for key, attr0 in iter(self):
            attrs1 = [getattr(o, key) for o in others]
            attr = attr0.merge_with(attrs1, replace=replace.get(key))
            kwargs[key] = attr
            if attr.unit is not None:
                kwargs[f'{key}_unit'] = attr.unit
        return self.__class__(**kwargs)


class FlexAttrsGrid(FlexAttrs):
    """Grid attributes."""

    def __init__(self, *, north_pole_lat, north_pole_lon):
        """Initialize instance of ``FlexAttrsGrid``.

        Kwargs:
            north_pole_lat (float): Latitude of rotated north pole.

            north_pole_lon (float): Longitude of rotated north pole.

        """
        super().__init__()
        self.set('north_pole_lat', north_pole_lat, float)
        self.set('north_pole_lon', north_pole_lon, float)


class FlexAttrsVariable(FlexAttrs):
    """Variable attributes."""

    def __init__(
            self, *, long_name, short_name, unit, level_bot, level_bot_unit,
            level_top, level_top_unit):
        """Initialize an instance of ``FlexAttrsVariable``.

        Kwargs:
            name (str): Name of variable.

            unit (str): Unit of variable as a regular string
                (e.g., 'm-3' for cubic meters).

            level_bot (float or list[float]): Bottom level value(s).

            level_bot_unit (str): Bottom level unit.

            level_top (float or list[float]): Top level value(s).

            level_top_unit (str): Bottom level unit.

        """
        super().__init__()
        self.set('long_name', long_name, str)
        self.set('short_name', short_name, str)
        self.set('unit', unit, str)
        self.set('level_bot', level_bot, float, unit=level_bot_unit)
        self.set('level_top', level_top, float, unit=level_top_unit)

    def format_unit(self):
        return self.unit.format_unit(self.unit.value)

    def format_short_name(self):
        return f'{self.short_name.value} ({self.format_unit()})'  #SR_ATTR

    def format_level_bot_unit(self):
        return self.level_bot.format_unit()

    def format_level_top_unit(self):
        return self.level_top.format_unit()

    def format_level_unit(self):
        unit_bottom = self.format_level_bot_unit()
        unit_top = self.format_level_top_unit()
        if unit_bottom != unit_top:
            raise Exception(
                f"bottom and top level units differ: "
                f"'{unit_bottom}' != '{unit_top}'")
        return unit_top

    def format_level_range(self):

        if (self.level_bot.value, self.level_top.value) == (-1, -1):  #SR_ATTR
            return None

        def fmt(bot, top, unit_fmtd=self.format_level_unit()):
            s = f"{bot:g}$\\,\\endash\\,${top:g}"
            if unit_fmtd:
                s += f" {unit_fmtd}"
            return s

        try:
            # Single level range
            return fmt(self.level_bot.value, self.level_top.value)  #SR_ATTR
        except TypeError:
            pass
        #-- Multiple level ranges

        try:
            bots = sorted(self.level_bot.value)  #SR_ATTR
            tops = sorted(self.level_top.value)  #SR_ATTR
        except TypeError:
            raise  #SR_TMP TODO proper error message
        else:
            if len(bots) != len(tops):
                raise Exception(
                    f"inconsistent no. levels: {len(bots)} != {len(tops)}")
            n = len(bots)

        if n == 2:
            if tops[0] == bots[1]:
                return fmt(bots[0], tops[1])
            else:
                return (
                    f"{fmt(bots[0], tops[0], None)} + {fmt(bots[1], tops[1])}")
                raise NotImplementedError(f"2 non-continuous level ranges")

        elif n == 3:
            if tops[0] == bots[1] and tops[1] == bots[2]:
                return fmt(bots[0], tops[2])
            else:
                raise NotImplementedError(f"3 non-continuous level ranges")

        else:
            raise NotImplementedError(f"{n} sets of levels")


class FlexAttrsRelease(FlexAttrs):
    """Release attributes."""

    def __init__(
            self, *, site_lat, site_lon, site_name, height, height_unit, rate,
            rate_unit, mass, mass_unit):
        """Initialize an instance of ``FlexAttrsRelease``.

        Kwargs:
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
        super().__init__()
        self.set('site_lat', site_lat, float)
        self.set('site_lon', site_lon, float)
        self.set('site_name', site_name, str)
        self.set('height', height, float, unit=height_unit)
        self.set('rate', rate, float, unit=rate_unit)
        self.set('mass', mass, float, unit=mass_unit)

    def format_height(self):
        return f'{self.height.value} {self.height.unit}'  #SR_ATTR

    def format_rate_unit(self):
        return self.rate.format_unit()  #SR_ATTR

    def format_rate(self):
        return f'{self.rate.value:g} {self.format_rate_unit()}'  #SR_ATTR

    def format_mass_unit(self):
        return self.mass.format_unit()

    def format_mass(self):
        return f'{self.mass.value:g} {self.format_mass_unit()}'  #SR_ATTR


class FlexAttrsSpecies(FlexAttrs):
    """Species attributes."""

    def __init__(
            self, *, name, half_life, half_life_unit, deposit_vel,
            deposit_vel_unit, sediment_vel, sediment_vel_unit, washout_coeff,
            washout_coeff_unit, washout_exponent):
        """Create an instance of ``FlexAttrsSpecies``.

        Kwargs:
            name (str): Species name.

            half_life (float or list[float]): Half life value(s).

            half_life_unit (str): Half life unit.

            deposit_vel (float or list[float]): Deposition velocity
                value(s).

            deposit_vel_unit (str): Deposition velocity unit.

            sediment_vel (float or list[float]): Sedimentation velocity
                value(s).

            sediment_vel_unit (str): Sedimentation velocity unit.

            washout_coeff (float or list[float]): Washout coefficient
                value(s).

            washout_coeff_unit (str): Washout coefficient unit.

            washout_exponent (float or list[float]): Washout exponent
                value(s).

        """
        super().__init__()
        self.set('name', name, str)
        self.set('half_life', half_life, float, unit=half_life_unit)
        self.set('deposit_vel', deposit_vel, float, unit=deposit_vel_unit)
        self.set('sediment_vel', sediment_vel, float, unit=sediment_vel_unit)
        self.set('washout_coeff', washout_coeff, float, unit=washout_coeff_unit)
        self.set('washout_exponent', washout_exponent, float)

    def format_name(self, join='/'):
        name = self.name.value  #SR_ATTR
        if isinstance(name, str):
            return name
        return f' {join} '.join(name)

    def format_half_life_unit(self):
        return self.half_life.format_unit()

    def format_half_life(self, join='/'):

        def fmt(attr):
            return f'{attr.value:g} {attr.format_unit()}'

        if not isiterable(self.half_life.value):  #SR_ATTR
            return fmt(self.half_life)  #SR_ATTR
        else:
            assert len(self.half_life.value) == len(
                self.half_life.unit)  #SR_ATTR
            s_lst = []
            for val, unit in zip(self.half_life.value,
                                 self.half_life.format_unit()):  #SR_ATTR
                s_lst.append(f'{val} {unit}')
            return f' {join} '.join(s_lst)

    def format_deposit_vel_unit(self):
        return self.deposit_vel.format_unit()

    def format_deposit_vel(self):
        return (
            f'{self.deposit_vel.value:g} '
            f'{self.format_deposit_vel_unit()}')  #SR_ATTR

    def format_sediment_vel_unit(self):
        return self.sediment_vel.format_unit()

    def format_sediment_vel(self):
        return (
            f'{self.sediment_vel.value:g} '
            f'{self.format_sediment_vel_unit()}')  #SR_ATTR

    def format_washout_coeff_unit(self):
        return self.washout_coeff.format_unit()

    def format_washout_coeff(self):
        return f'{self.washout_coeff.value:g} {self.format_washout_coeff_unit()}'  #SR_ATTR


class FlexAttrsSimulation(FlexAttrs):
    """Simulation attributes."""

    def __init__(self, *, model_name, start, end, now, integr_start):
        """Create an instance of ``FlexAttrsSimulation``.

        Kwargs:
            model_name (str): Name of the model.

            start (datetime): Start of the simulation.

            end (datetime): End of the simulation.

            now (datetime): Current timestep.

            integr_start (datetime): Start of the integration period.

        """
        super().__init__()
        self.set('model_name', model_name, str)
        self.set('start', start, datetime.datetime)
        self.set('end', end, datetime.datetime)
        self.set('now', now, datetime.datetime)
        self.set('integr_start', integr_start, datetime.datetime)

    def _format_dt(self, dt, relative):
        """Format a datetime object to a string."""
        dt = dt.value  #SR_ATTR
        if not relative:
            return dt.strftime('%Y-%m-%d %H:%M %Z')
        delta = dt - self.start.value  #SR_ATTR
        s = f"T$_{0}$"
        if delta.total_seconds() > 0:
            hours = int(delta.total_seconds()/3600)
            mins = int((delta.total_seconds()/3600)%1*60)
            s = f"{s}$\\,+\\,${hours:02d}:{mins:02d}$\\,$h"
        return s

    def format_start(self, relative=False):
        return self._format_dt(self.start, relative)

    def format_end(self, relative=False):
        return self._format_dt(self.end, relative)

    def format_now(self, relative=False):
        return self._format_dt(self.now, relative)

    def format_integr_start(self, relative=False):
        return self._format_dt(self.integr_start, relative)

    @property
    def integr_period(self):
        return self.now.value - self.integr_start.value  #SR_ATTR

    def format_integr_period(self):
        return f'{self.integr_period.total_seconds()/3600:g}$\\,$h'


class FlexAttrsCollection:
    """Collection of FLEXPART attributes."""

    def __init__(self, *, grid, variable, release, species, simulation):
        """Initialize an instance of ``FlexAttrsCollection``.

        Kwargs:
            grid (dict): Kwargs passed to ``FlexAttrsGrid``.

            variable (dict): Kwargs passed to ``FlexAttrsVariable``.

            release (dict): Kwargs passed to ``FlexAttrsRelease``.

            species (dict): Kwargs passed to ``FlexAttrsSpecies``.

            simulation (dict): Kwargs passed to ``FlexAttrsSimulation``.

        """
        self.reset()
        self.add('grid', grid)
        self.add('variable', variable)
        self.add('release', release)
        self.add('species', species)
        self.add('simulation', simulation)

    def reset(self):
        self._attrs = {}

    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(name) from None

    def add(self, name, attrs):

        if not isinstance(attrs, FlexAttrs):
            cls_by_name = {
                'grid': FlexAttrsGrid,
                'variable': FlexAttrsVariable,
                'release': FlexAttrsRelease,
                'species': FlexAttrsSpecies,
                'simulation': FlexAttrsSimulation,
            }
            try:
                cls = cls_by_name[name]
            except KeyError:
                raise ValueError(f"missing FlexAttrs class for name '{name}'")
            #SR_TMP<
            for key, attr in [(k, v) for k, v in attrs.items()]:
                if isinstance(attr, FlexAttr) and attr.unit is not None:
                    attrs[f'{key}_unit'] = attr.unit
            #SR_TMP>
            attrs = cls(**attrs)

        self._attrs[name] = attrs

    def merge_with(self, others, **replace):
        """Create a new instance by merging this and others.

        Args:
            others (list[FlexAttrsCollection]): Other instances to be
                merged with this.

            **replace (dicts): Collections of attributes to be replaced
                in the merged ``FlexAttrs*`` instances of the shared
                collection instance. Must contain all attributes that
                differ between any of the collections.

        Returns:
            FlexAttrsCollection: New attributes collection instance
                derived from ``self`` and ``others``. Note that no
                input collection is changed.

        Example:
            In this example, all attributes collection in a list are
            merged and the variable and release site names replaced,
            implying those differ between the collections. Wether that
            is the case or not, all other attributes must be the same,
            otherwise an error is issued.

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
        kwargs = {}
        for name in sorted(self._attrs.keys()):
            kwargs[name] = dict(
                self._attrs[name].merge_with(
                    [o._attrs[name] for o in others],
                    replace.get(name),
                ))
        return self.__class__(**kwargs)

    def __iter__(self):
        for name, attr in sorted(self._attrs.items()):
            yield name, attr

    def asdict(self):
        return {name: dict(attrs) for name, attrs in self}

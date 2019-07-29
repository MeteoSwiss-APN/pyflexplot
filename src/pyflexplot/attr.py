# -*- coding: utf-8 -*-
"""
Attributes.
"""
import datetime
import logging as log

from pprint import pformat

from .utils import isiterable

from .utils_dev import ipython  #SR_DEV


class FlexAttr:
    """Individual attribute."""

    def __init__(self, name, value, type_=None, unit=None, group=None):

        self.name = name
        self.type_ = type(value) if type_ is None else type_
        self.unit = unit

        self.set_group(group)

        #SR_TMP<
        assert not isinstance(
            value, self.__class__), f"value type {type(value).__name__}"
        assert not isiterable(value, str_ok=False), f"iterable value: {value}"
        #SR_TMP>

        # Ensure consistency of value and type_
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

    @classmethod
    def multiple(cls, *args, **kwargs):
        return FlexAttrMult(*args, cls_attr=cls, **kwargs)

    def set_group(self, group):
        self._group = group

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

        value = replace.get('value', value)
        unit = replace.get('unit', unit)

        kwargs = {'name': name, 'type_': type_, 'value': value, 'unit': unit}
        if isiterable(value, str_ok=False):
            return self.__class__.multiple(**kwargs)
        return self.__class__(**kwargs)

    def format(self, fmt=None, *, skip_unit=False, join=None):
        if fmt is None:
            if issubclass(self.type_, (float, int)):
                fmt = 'g'
            else:
                fmt = ''
        s = f'{{:{fmt}}}'.format(self.value)
        if self.unit and not skip_unit:
            s += f' {self.unit}'
        return s

    def format_unit(self, unit=None):
        """Auto-format the unit by elevating superscripts etc."""
        if unit is None:
            unit = self.unit
        old_new = [
            ('m-2', 'm$^{-2}$'),
            ('m-3', 'm$^{-3}$'),
            ('s-1', 's$^{-1}$'),
        ]
        for old, new in old_new:
            unit = unit.replace(old, new)
        return unit

    @property
    def label(self):
        labels = self._group._labels
        try:
            label = getattr(labels, self.name)
        except AttributeError:
            raise AttributeError(
                f"{type(labels).__name__}.{self.name}")
        return label


class FlexAttrMult(FlexAttr):

    def __init__(self, name, value, type_=None, unit=None, cls_attr=FlexAttr):

        self.name = name

        if not isiterable(value, str_ok=False):
            raise ValueError(f"value not iterable: {value}")

        if not isiterable(unit):
            unit = [unit for _ in value]

        if len(unit) != len(value):
            raise ValueError(
                f"numbers of values and units differ: "
                f"{len(value)} != {len(unit)}")

        if type_ is None:
            types = sorted(set([type(v) for v in value]))
            if len(types) > 1:
                raise ValueError(
                    f"'type_' omitted when types of values differ: {types}")
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

    def format(self, fmt=None, *, join=' / ', **kwargs):
        return join.join([a.format(fmt, **kwargs) for a in self._attr_lst])

    def format_unit(self, unit=None):
        return [a.format_unit(unit=unit) for a in self._attr_lst]


class FlexAttrDatetime(FlexAttr):
    """Individual datetime attribute."""

    def __init__(
            self, name, value, *, type_=None, start=None, **kwargs):

        if type_ is None:
            type_ = datetime.datetime

        if type_ is not datetime.datetime:
            raise ValueError(
                f"invalid type_ '{type_.__name__}' (not datetime.datetime)")

        if not isinstance(value, type_):
            raise ValueError(
                f"value has wrong type: expected '{type_.__name__}', "
                f"got '{type(value).__name__}'")

        super().__init__(name, value, type_=type_, **kwargs)

        self.start = start.value if isinstance(start, FlexAttr) else start

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
                f"{type(self).__name__}: starts differ: {starts}")
        attr.start = next(iter(starts))
        return attr

    def format(self, relative=False):
        """Format a datetime object to a string."""

        if not relative:
            return self.value.strftime('%Y-%m-%d %H:%M %Z')

        if self.start is None:
            ipython(globals(), locals())
            raise ValueError(
                f"{self.name}: relative formatting failed: missing start")

        delta = self.value - self.start

        s = f"T$_{0}$"

        if delta.total_seconds() > 0:
            hours = int(delta.total_seconds()/3600)
            mins = int((delta.total_seconds()/3600)%1*60)
            s = f"{s}$\\,+\\,${hours:02d}:{mins:02d}$\\,$h"

        return s


class FlexAttrGroup:
    """Base class for attributes."""

    def __init__(self, *args, **kwargs):
        if type(self) is FlexAttrGroup:
            raise ValueError(
                f"{type(self).__name__} should be subclassed, not instatiated")
        self.reset()

    def reset(self):
        self._attrs = {}

    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__}.{name}")

    #SR_TMP<<< TODO eliminate
    def set(self, name, value, type_, **kwargs):
        if isinstance(value, FlexAttr):
            attr = value
            attr.set_group(self)
        else:
            cls = {datetime.datetime: FlexAttrDatetime}.get(type_, FlexAttr)
            kwargs.update(
                {'name': name, 'value': value, 'type_': type_, 'group': self})
            try:
                attr = cls(**kwargs)
            except Exception as e:
                raise Exception(
                    f"error creating instance of {cls.__name__} "
                    f"({type(e).__name__}: {e})")
        self._attrs[name] = attr

    def __iter__(self):
        for name, attr in sorted(self._attrs.items()):
            yield name, attr

    def merge_with(self, others, replace=None):
        """Create new instance by merging self and others.

        Args:
            others (list[FlexAttrGroup]): Other instances of the same
                attributes class, to be merged with this one.

            replace (dict, optional): Attributes to be replaced in the
                merged instance. Must contain all attributes that
                differ between any of the instances to be merged.
                Defaults to '{}'.

        Returns:
            FlexAttrGroup: Merged instance derived from ``self`` and
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
        try:
            return self.__class__(**kwargs)
        except Exception as e:
            raise Exception(
                f"error creating instance of {self.__class__.__name__} "
                f"({type(e).__name__}: {e})\n\n{len(kwargs)} kwargs:"
                f"\n{pformat(kwargs)}")


class FlexAttrGroupGrid(FlexAttrGroup):
    """Grid attributes."""

    def __init__(self, *, north_pole_lat, north_pole_lon):
        """Initialize instance of ``FlexAttrGroupGrid``.

        Kwargs:
            north_pole_lat (float): Latitude of rotated north pole.

            north_pole_lon (float): Longitude of rotated north pole.

        """
        super().__init__()
        self.set('north_pole_lat', north_pole_lat, float)
        self.set('north_pole_lon', north_pole_lon, float)


class FlexAttrGroupVariable(FlexAttrGroup):
    """Variable attributes."""

    def __init__(
            self, *, long_name, short_name, unit, level_bot, level_bot_unit,
            level_top, level_top_unit):
        """Initialize an instance of ``FlexAttrGroupVariable``.

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

    def format_level_unit(self):
        unit_bottom = self.level_bot.format_unit()
        unit_top = self.level_top.format_unit()
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


class FlexAttrGroupRelease(FlexAttrGroup):
    """Release attributes."""

    def __init__(
            self, *, lat, lon, site_name, height, height_unit, rate,
            rate_unit, mass, mass_unit):
        """Initialize an instance of ``FlexAttrGroupRelease``.

        Kwargs:
            lat (float): Latitude of release site.

            lon (float): Longitude of release site.

            site_name (str): Name of release site.

            height (float or list[float]): Release height value(s).

            height_unit (str): Release height unit

            rate (float or list[float]): Release rate value(s).

            rate_unit (str): Release rate unit.

            mass (float or list[float]): Release mass value(s).

            mass_unit (str): Release mass unit.

        """
        super().__init__()
        self.set('lat', lat, float)
        self.set('lon', lon, float)
        self.set('site_name', site_name, str)
        self.set('height', height, float, unit=height_unit)
        self.set('rate', rate, float, unit=rate_unit)
        self.set('mass', mass, float, unit=mass_unit)


class FlexAttrGroupSpecies(FlexAttrGroup):
    """Species attributes."""

    def __init__(
            self, *, name, half_life, half_life_unit, deposit_vel,
            deposit_vel_unit, sediment_vel, sediment_vel_unit, washout_coeff,
            washout_coeff_unit, washout_exponent):
        """Create an instance of ``FlexAttrGroupSpecies``.

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
        self.set(
            'washout_coeff', washout_coeff, float, unit=washout_coeff_unit)
        self.set('washout_exponent', washout_exponent, float)


class FlexAttrGroupSimulation(FlexAttrGroup):
    """Simulation attributes."""

    def __init__(self, *, model_name, start, end, now, integr_start):
        """Create an instance of ``FlexAttrGroupSimulation``.

        Kwargs:
            model_name (str): Name of the model.

            start (datetime): Start of the simulation.

            end (datetime): End of the simulation.

            now (datetime): Current timestep.

            integr_start (datetime): Start of the integration period.

        """
        super().__init__()
        self.set('model_name', model_name, str)
        self.set('start', start, datetime.datetime, start=start)
        self.set('end', end, datetime.datetime, start=start)
        self.set('now', now, datetime.datetime, start=start)
        self.set('integr_start', integr_start, datetime.datetime, start=start)

    @property
    def integr_period(self):
        return self.now.value - self.integr_start.value  #SR_ATTR

    def format_integr_period(self):
        return f'{self.integr_period.total_seconds()/3600:g}$\\,$h'


class FlexAttrGroupCollection:
    """Collection of FLEXPART attributes."""

    def __init__(self, *, grid, variable, release, species, simulation):
        """Initialize an instance of ``FlexAttrGroupCollection``.

        Kwargs:
            grid (dict): Kwargs passed to ``FlexAttrGroupGrid``.

            variable (dict): Kwargs passed to ``FlexAttrGroupVariable``.

            release (dict): Kwargs passed to ``FlexAttrGroupRelease``.

            species (dict): Kwargs passed to ``FlexAttrGroupSpecies``.

            simulation (dict): Kwargs passed to ``FlexAttrGroupSimulation``.

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

        if not isinstance(attrs, FlexAttrGroup):
            cls_by_name = {
                'grid': FlexAttrGroupGrid,
                'variable': FlexAttrGroupVariable,
                'release': FlexAttrGroupRelease,
                'species': FlexAttrGroupSpecies,
                'simulation': FlexAttrGroupSimulation,
            }
            try:
                cls = cls_by_name[name]
            except KeyError:
                raise ValueError(
                    f"missing FlexAttrGroup class for name '{name}'")
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
            others (list[FlexAttrGroupCollection]): Other instances to be
                merged with this.

            **replace (dicts): Collections of attributes to be replaced
                in the merged ``FlexAttrGroup*`` instances of the shared
                collection instance. Must contain all attributes that
                differ between any of the collections.

        Returns:
            FlexAttrGroupCollection: New attributes collection instance
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

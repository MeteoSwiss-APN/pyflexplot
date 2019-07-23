# -*- coding: utf-8 -*-
"""
Data structures.
"""
import datetime
import logging as log
import numpy as np

from collections import namedtuple
from copy import copy, deepcopy

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


class FlexAttrs:
    """Base class for attributes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.reset_names()

    def reset_names(self):
        if hasattr(self, 'names'):
            try:
                iter_names = iter(self.names)
            except TypeError:
                pass
            for name in iter_names:
                try:
                    delattr(self, name)
                except AttributeError:
                    pass
        self.__dict__['names'] = set()

    def __setattr__(self, key, val):
        if key.startswith('_') or key in self.names:
            self.__dict__[key] = val
        else:
            raise ValueError(
                f"invalid public attribute '{key}': "
                f"not among {sorted(self.names)}")

    def set(self, name, val, type_=None, mult_vals_ok=False):
        """Set an attribute, optionally checking its type.

        Args:
            name (str or list[str]): Name of attribute. Can be a list
                of multiple names if ``val`` is a list of values to be
                distributed over multiple attributes (e.g., a number-
                unit-pair). In lists, tilde ('~') can be used in all
                but the first name to be replace by the previous name
                in the list (e.g., ``['velocity', '~_unit']``).

            val (object or list[object]): Value of attribute. If is a
                list of multiple values and ``name`` is a list as well,
                then the values are distributed over multiple values.
                If ``mult_vals_ok=True``, each value in turn is allowed
                to be a list of multiple values.

            type_ (type or list[type], optional): Type of attribute.
                If passed, ``val`` must either be an instance of or
                convertible to it. If ``val`` is a list of multiple
                values, a separate type can be passed for each.
                Defaults to None.

            mult_vals_ok (bool, optional): Whether each value is
                allowed to be a list of multiple values. Defaults to
                False.

        """

        if not isinstance(name, str):
            # Handle multiple names
            names, vals, types = self._prepare_multi_attrs(name, val, type_)
            for name, val, type_ in zip(names, vals, types):
                self.set(name, val, type_, mult_vals_ok=True)

        # Check type(s) of value(s)
        if type_ is not None and not isinstance(val, type_):

            # Check whether the value is a list of multiple values
            try:
                iter(val)
            except TypeError:
                # There's only one value; check its type
                try:
                    val = type_(val)
                except TypeError:
                    raise ValueError(
                        f"attr '{name}': value '{val}' has wrong type: "
                        f"expected '{type_.__name__}', got "
                        f"'{type(val).__name__}'")
            else:
                # There are multiple values
                if mult_vals_ok:
                    # That's OK; check types of values
                    try:
                        val = [type_(v) for v in val]
                    except TypeError:
                        raise ValueError(
                            f"attr '{name}': one or more values in {val} "
                            f"has wrong type: expected '{type_.__name__}', "
                            f"got {[type(v).__name__ for v in val]}")
                else:
                    # That's not OK!
                    raise ValueError(
                        f"attr '{name}': expected single value, got list of "
                        f"{len(val)}: {val}")

        self.__dict__[name] = val
        self.names.add(name)

    def set_value_unit(
            self, val_name, val, unit, val_type, mult_vals_ok=False):
        """Set a (value, unit) attribute pair.

        Args:
            val_name (str): Name of the value attribute, from which
                the name of the unit attribute is derived by appending
                '_unit'.

            val (<val_type> or list[<val_type>]): Value(s) compatible
                with type ``val_type``. A list of multiple values is
                only allowed for ``mult_vals_ok=True``.

            unit (str): Unit.

            val_type (type): Type compatible with ``val``.

            mult_vals_ok (bool, optional): Whether a list of multiple
                values is allowed for ``val``. Defaults to False.

        """
        self.set(
            [val_name, '~_unit'],
            (val, unit),
            [val_type, str],
            mult_vals_ok=mult_vals_ok,
        )

    def _prepare_multi_attrs(self, names, vals, types):

        # Ensure that one value has been passed for each name
        class UnequalLenError(Exception):
            pass

        try:
            if len(names) != len(vals):
                raise UnequalLenError
        except (TypeError, UnequalLenError):
            raise ValueError(
                f"attrs '{names}' ({len(names)}): "
                f"need one value for each name: {vals}")

        # Tilde-notation: insert previous name in place of '~'
        for i, name in enumerate(copy(names)):
            if '~' in name:
                if i == 0:
                    raise ValueError(
                        f"'~' notation not possible in first name: "
                        f"{names}")
                names[i] = name.replace('~', names[i - 1])

        # Prepare types (same for all, or one for each)
        if types is None or isinstance(types, type):
            types = [types]*len(names)
        elif len(types) != len(names):
            raise ValueError(
                f"attrs '{names}' ({len(names)}): "
                f"need one type_ for all values, or one for each: "
                f"{type_}")

        return names, vals, types

        raise Exception(
            f"'{cname}' has no attribute 'attrs' -- maybe forgot "
            f"`super().__init__()` in `{cname}.__init__`?")

    def _format_unit(self, unit):
        """Auto-format a unit by elevating superscripts etc."""
        old_new = [
            ('m-2', 'm$^{-2}$'),
            ('m-3', 'm$^{-3}$'),
            ('s-1', 's$^{-1}$'),
        ]
        for old, new in old_new:
            unit = unit.replace(old, new)
        return unit

    def __iter__(self):
        for name in sorted(self.names):
            yield name, getattr(self, name)

    def merge(self, others, replace=None):
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
        for key, val0 in iter(self):
            try:
                kwargs[key] = replace[key]
            except KeyError:
                vals = [val0]
                for other in others:
                    val = getattr(other, key)
                    if val not in vals:
                        vals.append(val)
                if len(vals) == 1:
                    kwargs[key] = next(iter(vals))
                else:
                    kwargs[key] = tuple(vals)
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
            self,
            *,
            long_name,
            short_name,
            unit,
            level_bot,
            level_bot_unit,
            level_top,
            level_top_unit):
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
        self.set_value_unit('level_bot', level_bot, level_bot_unit, float)
        self.set_value_unit('level_top', level_top, level_top_unit, float)

    def format_unit(self):
        return self._format_unit(self.unit)

    def format_short_name(self):
        return f'{self.short_name} ({self.format_unit()})'

    def format_level_bot_unit(self):
        return self._format_unit(self.level_bot_unit)

    def format_level_top_unit(self):
        return self._format_unit(self.level_top_unit)

    def format_level_unit(self):
        unit_bottom = self.format_level_bot_unit()
        unit_top = self.format_level_top_unit()
        if unit_bottom != unit_top:
            raise Exception(
                f"bottom and top level units differ: "
                f"'{unit_bottom}' != '{unit_top}'")
        return unit_top

    def format_level_range(self):
        if (self.level_bot, self.level_top) == (-1, -1):
            return None
        return (
            f"{self.level_bot:g}$\\,\\endash\\,${self.level_top:g} "
            f"{self.format_level_unit()}")


class FlexAttrsRelease(FlexAttrs):
    """Release attributes."""

    def __init__(
            self,
            *,
            site_lat,
            site_lon,
            site_name,
            height,
            height_unit,
            rate,
            rate_unit,
            mass,
            mass_unit):
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
        self.set_value_unit('height', height, height_unit, float)
        self.set_value_unit('rate', rate, rate_unit, float)
        self.set_value_unit('mass', mass, mass_unit, float)

    def format_height(self):
        return f'{self.height} {self.height_unit}'

    def format_rate_unit(self):
        return self._format_unit(self.rate_unit)

    def format_rate(self):
        return f'{self.rate:g} {self.format_rate_unit()}'

    def format_mass_unit(self):
        return self._format_unit(self.mass_unit)

    def format_mass(self):
        return f'{self.mass:g} {self.format_mass_unit()}'


class FlexAttrsSpecies(FlexAttrs):
    """Species attributes."""

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
            washout_exponent):
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
        self.set('name', name, str, mult_vals_ok=True)
        self.set_value_unit(
            'half_life', half_life, half_life_unit, float, mult_vals_ok=True)
        self.set_value_unit(
            'deposit_vel', deposit_vel, deposit_vel_unit, float)
        self.set_value_unit(
            'sediment_vel', sediment_vel, sediment_vel_unit, float)
        self.set_value_unit(
            'washout_coeff', washout_coeff, washout_coeff_unit, float)
        self.set('washout_exponent', washout_exponent, float)

    def format_name(self):
        if isinstance(self.name, str):
            return self.name
        return ' / '.join(self.name)

    def format_half_life_unit(self):
        return self._format_unit(self.half_life_unit)

    def format_half_life(self):
        def fmt(val, unit):
            return f'{val:g} {self._format_unit(unit)}'
        try:
            iter(self.half_life)
        except TypeError:
            return fmt(self.half_life, self.half_life_unit)
        else:
            assert len(self.half_life) == len(self.half_life_unit)
            s_lst = []
            for val, unit in zip(self.half_life, self.half_life_unit):
                s_lst.append(fmt(val, unit))
            return ' / '.join(s_lst)

    def format_deposit_vel_unit(self):
        return self._format_unit(self.deposit_vel_unit)

    def format_deposit_vel(self):
        return (f'{self.deposit_vel:g} ' f'{self.format_deposit_vel_unit()}')

    def format_sediment_vel_unit(self):
        return self._format_unit(self.sediment_vel_unit)

    def format_sediment_vel(self):
        return (f'{self.sediment_vel:g} ' f'{self.format_sediment_vel_unit()}')

    def format_washout_coeff_unit(self):
        return self._format_unit(self.washout_coeff_unit)

    def format_washout_coeff(self):
        return f'{self.washout_coeff:g} {self.format_washout_coeff_unit()}'


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
        if not relative:
            return dt.strftime('%Y-%m-%d %H:%M %Z')
        delta = dt - self.start
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
        return self.now - self.integr_start

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
        FlexAttrs.reset_names(self)

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
            attrs = cls(**attrs)

        setattr(self, name, attrs)
        self.names.add(name)

    def merge(self, others, **replace):
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

            attrs_coll = attrs_colls[0].merge(
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
        for name in sorted(self.names):
            kwargs[name] = dict(
                getattr(self, name).merge(
                    [getattr(o, name) for o in others],
                    replace.get(name),
                ))
        return self.__class__(**kwargs)

    def __iter__(self):
        for name in sorted(self.names):
            yield name, getattr(self, name)

    def asdict(self):
        return {name: dict(attrs) for name, attrs in self}

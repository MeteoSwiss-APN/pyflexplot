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


class FlexDataRotPole:
    """FLEXPART data on rotated-pole grid."""

    def __init__(self, rlat, rlon, field, attrs, field_specs):
        self.rlat = rlat
        self.rlon = rlon
        self.field = field
        self.attrs = attrs
        self.field_specs = field_specs


class FlexAttrsBase:
    """Base class for attributes."""

    def __init__(self):
        self.attrs = {}

    def __getattr__(self, name):
        try:
            return self.__dict__['attrs'][name]
        except KeyError as e:
            if str(e) == 'attrs' and name != 'attrs':
                self._raise_missing_attrs()
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'")

    def set(self, name, val, type_=None):
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

            type_ (type or list[type], optional): Type of attribute.
                If passed, ``val`` must either be an instance of or
                convertible to it. If ``val`` is a list of multiple
                values, a separate type can be passed for each.
                Defaults to None.

        """

        if not isinstance(name, str):
            # Handle multiple names
            names, vals, types = self._prepare_multi_attrs(name, val, type_)
            for name, val, type_ in zip(names, vals, types):
                self.set(name, val, type_)

        # Check type
        if type_ is not None and not isinstance(val, type_):
            try:
                val = type_(val)
            except TypeError:
                raise ValueError(
                    f"attr '{name}': value '{val}' has wrong type: "
                    f"expected '{type_.__name__}', got '{type(val).__name__}'")

        # Set attribute
        try:
            self.attrs[name] = val
        except AttributeError:
            self._raise_missing_attrs()

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

    def _raise_missing_attrs(self):
        cname = type(self).__name__
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


class FlexAttrsGrid(FlexAttrsBase):
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


class FlexAttrsVariable(FlexAttrsBase):
    """Variable attributes."""

    def __init__(self, *, name, unit, level_bot, level_top):
        """Initialize an instance of ``FlexAttrsVariable``.

        Kwargs:
            name (str): Name of variable.

            unit (str): Unit of variable as a regular string
                (e.g., 'm-3' for cubic meters).

            level_bot ((float, str)): Bottom level (value and unit).

            level_top ((float, str)): Top level (value and unit).

        """
        super().__init__()
        self.set('name', name, str)
        self.set('unit', unit, str)
        self.set(['level_bot', '~_unit'], level_bot, [float, str])
        self.set(['level_top', '~_unit'], level_top, [float, str])

    def format_unit(self):
        return self._format_unit(self.unit)

    def format_name(self):
        return f'{self.name} ({self.format_unit()})'

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
        return (
            f"{self.level_bot:g}$\,\endash\,${self.level_top:g} "
            f"{self.format_level_unit()}")


class FlexAttrsRelease(FlexAttrsBase):
    """Release attributes."""

    def __init__(
            self, *, site_lat, site_lon, site_name, site_tz_name, height, rate,
            mass):
        """Initialize an instance of ``FlexAttrsRelease``.

        Kwargs:
            site_lat (float): Latitude of release site.

            site_lon (float): Longitude of release site.

            site_name (str): Name of release site.

            site_tz_name (str): Name of time zone of release site.

            height ((float, str)): Release height (value and unit).

            rate ((float, str)): Release rate (value and unit).

            mass ((float, str)): Release mass (value and unit).

        """
        super().__init__()
        self.set('site_lat', site_lat, float)
        self.set('site_lon', site_lon, float)
        self.set('site_name', site_name, str)
        self.set('site_tz_name', site_tz_name, str)
        self.set(['height', '~_unit'], height, [float, str])
        self.set(['rate', '~_unit'], rate, [float, str])
        self.set(['mass', '~_unit'], mass, [float, str])

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


class FlexAttrsSpecies(FlexAttrsBase):
    """Species attributes."""

    def __init__(
            self, *, name, half_life, deposit_vel, sediment_vel, washout_coeff,
            washout_exponent):
        """Create an instance of ``FlexAttrsSpecies``.

        Kwargs:
            name (str): Species name.

            half_life ((float, str)): Half life (value and unit).

            deposit_vel ((float, str)): Deposition velocity (value
                and unit).

            sediment_vel ((float, str)): Sedimentation velocity (value
                and unit).

            washout_coeff ((float, str)): Washout coefficient (value
                and unit).

            washout_exponent (float): <TODO>

        """
        super().__init__()
        self.set('name', name, str)
        self.set(['half_life', '~_unit'], half_life, [float, str])
        self.set(['deposit_vel', '~_unit'], deposit_vel, [float, str])
        self.set(['sediment_vel', '~_unit'], sediment_vel, [float, str])
        self.set(['washout_coeff', '~_unit'], washout_coeff, [float, str])
        self.set('washout_exponent', washout_exponent, float)

    def format_half_life_unit(self):
        return self._format_unit(self.half_life_unit)

    def format_half_life(self):
        return f'{self.half_life:g} {self.format_half_life_unit()}'

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


class FlexAttrsSimulation(FlexAttrsBase):
    """Simulation attributes."""

    def __init__(self, *, model_name, start, end, now):
        """Create an instance of ``FlexAttrsSimulation``.

        Kwargs:
            model_name (str): Name of the model.

            start (datetime): Start of the simulation.

            end (datetime): End of the simulation.

            now (datetime): Current timestep.

        """
        super().__init__()
        self.set('model_name', model_name, str)
        self.set('start', start, datetime.datetime)
        self.set('end', end, datetime.datetime)
        self.set('now', now, datetime.datetime)

    def _format_dt(self, dt):
        """Format a datetime object to a string."""
        return dt.strftime('%Y-%m-%d %H:%M %Z')

    def format_start(self):
        return self._format_dt(self.start)

    def format_end(self):
        return self._format_dt(self.end)

    def format_now(self):
        return self._format_dt(self.now)


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
        self.grid = FlexAttrsGrid(**grid)
        self.variable = FlexAttrsVariable(**variable)
        self.release = FlexAttrsRelease(**release)
        self.species = FlexAttrsSpecies(**species)
        self.simulation = FlexAttrsSimulation(**simulation)

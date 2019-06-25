# -*- coding: utf-8 -*-
"""
Data structures.
"""
import datetime
import logging as log
import numpy as np

from collections import namedtuple
from copy import copy

from .utils_dev import ipython  #SR_DEV

FieldKey = namedtuple(
    'FieldKey',
    'age_ind relpt_ind time_ind level_ind species_id field_type',
)


class FlexData:
    """Hold FLEXPART output data.

    Args:
        setup (dict): Setup. TODO: Better description!

    """

    def __init__(self, setup):
        self._setup = setup

        self._fields = {}
        self._field_attrs = {}

    def field_key(
            self, *, age_ind, relpt_ind, time_ind, level_ind, species_id,
            field_type):
        """Create key for field variable.

        Kwargs:
            age_ind (int): Index of age class.

            relpt_ind (int): Index of release point.

            time_ind (int): Index of time step.

            level_ind (int): Index of vertical level.

            species_id (int): Id of particle species.

            field_type (str): The type of the field ('3D', 'WD', 'DD').

        Returns:
            FieldKey instance (namedtuple).
        """
        return FieldKey(
            age_ind=age_ind,
            relpt_ind=relpt_ind,
            time_ind=time_ind,
            level_ind=level_ind,
            species_id=species_id,
            field_type=field_type,
        )

    def field_keys(
            self,
            *,
            age_inds=None,
            relpt_inds=None,
            time_inds=None,
            level_inds=None,
            species_ids=None,
            field_types=None):
        """Returns field keys, either all or a subset.

        Kwargs:
            age_inds (list[int], optional): Age class indices.
                Defaults to None. If None, all values are included.

            relpt_inds (list[int], optional): Release point indices.
                Defaults to None. If None, all values are included.

            time_inds (list[int], optional): Time step indices.
                Defaults to None. If None, all values are included.

            level_inds (list[int], optional): Vertical level indices.
                Defaults to None. If None, all values are included.

            species_ids (list[int], optional): Species ids.
                Defaults to None. If None, all values are included.

            field_types (list[int], optional): Field types.
                Defaults to None. If None, all values are included.

        Returns:
            list: List of keys.
        """

        def check_restriction(values, name):
            """Check validity of restriction."""
            if values is not None:
                choices = getattr(self, f'{name}s')()
                for value in values:
                    if value not in choices:
                        raise ValueError(
                            f"invalid {name.replace('_', ' ')} '{value}':"
                            f" not among {choices}")

        check_restriction(species_ids, 'age_ind')
        check_restriction(species_ids, 'relpt_ind')
        check_restriction(species_ids, 'time_ind')
        check_restriction(species_ids, 'level_ind')
        check_restriction(species_ids, 'species_id')
        check_restriction(field_types, 'field_type')

        # Collect keys
        keys = []
        for key in self._fields.keys():
            if (age_inds is not None and key.age_ind not in age_inds):
                continue
            if (relpt_inds is not None and key.relpt_ind not in relpt_inds):
                continue
            if (time_inds is not None and key.time_ind not in time_inds):
                continue
            if (level_inds is not None and key.level_ind not in level_inds):
                continue
            if (species_ids is not None and key.species_id not in species_ids):
                continue
            if (field_types is not None and key.field_type not in field_types):
                continue
            keys.append(key)

        return [key for key in self._fields.keys()]

    def age_inds(self, **restrictions):
        """Returns all age class ids, or a subset thereof."""
        return [key.age_ind for key in self.field_keys(**restrictions)]

    def relpt_inds(self, **restrictions):
        """Returns all relpt class ids, or a subset thereof."""
        return [key.relpt_ind for key in self.field_keys(**restrictions)]

    def time_inds(self, **restrictions):
        """Returns all time class ids, or a subset thereof."""
        return [key.time_ind for key in self.field_keys(**restrictions)]

    def level_inds(self, **restrictions):
        """Returns all level class ids, or a subset thereof."""
        return [key.level_ind for key in self.field_keys(**restrictions)]

    def species_ids(self, **restrictions):
        """Returns all species ids, or a subset thereof."""
        return [key.species_id for key in self.field_keys(**restrictions)]

    def field_types(self, **restrictions):
        """Returns all field types, or a subset thereof."""
        return [key.field_type for key in self.field_keys(**restrictions)]

    def set_grid(self, *, rlat, rlon):
        """Set grid variables.

        Kwargs:
            rlat (ndarray): Rotated latitude array (1D).

            rlon (ndarray): Rotated longitude array (1D).
        """
        self.rlat = rlat
        self.rlon = rlon

    def add_field(self, arr, attrs=None, **key_comps):
        """Add a field array with optional attributes.

        Args:
            arr (ndarray): Field array.

            attrs (dict, optional): Field attributes. Defaults to None.

            **key_comps: Components of field key. Passed on to method
                ``FlexData.field_key``.

        """
        key = self.field_key(**key_comps)
        self._fields[key] = arr
        self._field_attrs[key] = attrs

    def field(self, key_or_comps):
        """Access a field.

        Args:
            key_or_comps (FieldKey or dict): A FieldKey instance, or
                a dict containing the key components passed on to
                ``FlexData.field_key`` to create a FieldKey.

        Returns:
            ndarray: Field array.

        """
        if isinstance(key_or_comps, FieldKey):
            key = key_or_comps
        else:
            key = self.field_key(**key_or_comps)
        return self._fields[key]

    def field_attrs(self, **key_components):
        """Return field attributes.

        Args:
            **key_comps: Components of field key. Passed on to method
                ``FlexData.field_key``.

        Returns:
            Field attributes dict.

        """
        return self._field_attrs[self.field_key(**key_comps)]

    def get_attrs(self, key):
        """Return attributes collection for a given key."""
        #SR_TMP<
        attrs = FlexAttrsCollection(
            grid={
                'north_pole_lat': 43.0,
                'north_pole_lon': -170.0,
            },
            release={
                'site_lat': 47.37,
                'site_lon': 7.97,
                'site_name': 'Goesgen',
                'site_tz_name': 'Europe/Zurich',
                'height': (100, 'm AGL'),
                'rate': (34722.2, 'Bq s-1'),
                'mass': (1e9, 'Bq'),
            },
            variable={
                'name': 'Concentration',
                'unit': 'Bq m-3',
                'level_bot': (500, 'm AGL'),
                'level_top': (2000, 'm AGL'),
            },
            species={
                'name': 'Cs-137',
                'half_life': (30.0, 'years'),
                'deposit_vel': (1.5e-3, 'm s-1'),
                'sediment_vel': (0.0, 'm s-1'),
                'washout_coeff': (7.0e-5, 's-1'),
                'washout_exponent': 0.8,
            },
            simulation={
                'model_name':
                'COSMO-1',
                'start':
                datetime.datetime(
                    year=2019,
                    month=5,
                    day=28,
                    hour=0,
                    minute=0,
                    tzinfo=datetime.timezone.utc,
                ),
                'end':
                datetime.datetime(
                    year=2019,
                    month=5,
                    day=28,
                    hour=8,
                    minute=0,
                    tzinfo=datetime.timezone.utc,
                ),
                'now':
                datetime.datetime(
                    year=2019,
                    month=5,
                    day=28,
                    hour=9,
                    minute=0,
                    tzinfo=datetime.timezone.utc,
                ),
            },
        )
        #SR_TMP>

        return attrs


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
        class UnequalLenError(Exception): pass
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

    def __init__(
            self, *, name, unit, level_bot, level_top):
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
            self, *, site_lat, site_lon, site_name, site_tz_name, height,
            rate, mass):
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
        return (
            f'{self.deposit_vel:g} '
            f'{self.format_deposit_vel_unit()}')

    def format_sediment_vel_unit(self):
        return self._format_unit(self.sediment_vel_unit)

    def format_sediment_vel(self):
        return (
            f'{self.sediment_vel:g} '
            f'{self.format_sediment_vel_unit()}')

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



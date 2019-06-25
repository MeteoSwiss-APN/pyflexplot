# -*- coding: utf-8 -*-
"""
Plots.
"""
import cartopy
import datetime
import geopy.distance
import logging as log
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pytz
import re

from collections import namedtuple
from matplotlib import ticker
from textwrap import dedent

from .utils import Degrees
from .utils import MaxIterationError
from .utils import merge_dicts
from .utils_dev import ipython  #SR_DEV

mpl.use('Agg')  # Prevent ``couldn't connect to display`` error


class FlexPlotter:
    """Create one or more FLEXPLART plots of a certain type.

    Attributes:
        <TODO>

    Methods:
        <TODO>

    """

    def __init__(self, type_):
        """Initialize instance of FlexPlotter.

        Args:
            type_ (str): Type of plot.

        """
        self.type_ = type_

    def run(self, data, file_path_fmt):
        """Create plots.

        Args:
            data (FlexData): All necessary data (grid, fields, attrs)
                required for the plot, read from a FLEXPART file.

            file_path_fmt (str): Format string of output file path.
                Must contain all necessary format keys to avoid that
                multiple files have the same name, but can be a plain
                string if no variable assumes more than one value.

        Yields:
            str: Output file paths.

        """
        self.data = data
        self.file_path_fmt = file_path_fmt

        if self.type_ == 'concentration':
            for path in self._run_concentration():
                yield path
        else:
            raise NotImplementedError(f"plot type '{self.type_}'")

    def _run_concentration(self):
        """Create one or more concentration plots."""

        # Check availability of required field type
        field_type = '3D'
        field_types_all = self.data.field_types()
        if field_type not in field_types_all:
            raise Exception(
                f"missing field type '{field_type}' among {field_types_all}")

        # Collect field keys
        restrictions = {"field_types": [field_type]}
        keys = self.data.field_keys(**restrictions)

        # Check output file path for format keys
        self.check_file_path_fmt(restrictions)

        # Create plots
        file_paths = []
        _s = 's' if len(keys) > 1 else ''
        print(f"create {len(keys)} concentration plot{_s}")
        for i_key, key in enumerate(keys):
            file_path = self.format_file_path(key)
            _w = len(str(len(keys)))
            print(f" {i_key+1:{_w}}/{len(keys)}  {file_path}")

            kwargs = {
                'rlat': self.data.rlat,
                'rlon': self.data.rlon,
                'fld': self.data.field(key),
                'attrs_raw': {},  #SR_TMP
            }

            FlexPlotConcentration(**kwargs).save(file_path)

            yield file_path

    def check_file_path_fmt(self, restrictions):
        """Check output file path for necessary variables format keys.

        If a variable (e.g., species id, level index) assumes multiple
        values, the file path must contain a corresponding format key,
        otherwise multiple output files will share the same name and
        overwrite each other.

        Args:
            restrictions (dict): Restrictions to field keys, i.e.,
                explicitly selected values of some variables.

        """

        def _check_file_path_fmt__core(name):
            """Check if the file path contains a necessary variable."""
            values = set(getattr(self.data, f'{name}s')(**restrictions))
            rx = re.compile(r'\{' + name + r'(:.*)?\}')
            if len(values) > 1 and not rx.search(self.file_path_fmt):
                raise Exception(
                    f"output file path '{self.file_path_fmt}' must contain"
                    f" format key '{fmt_key}' to plot {len(values)} different"
                    f" {name.replace('_', ' ')}s {sorted(values)}")

        _check_file_path_fmt__core('age_ind')
        _check_file_path_fmt__core('relpt_ind')
        _check_file_path_fmt__core('time_ind')
        _check_file_path_fmt__core('level_ind')
        _check_file_path_fmt__core('species_id')
        _check_file_path_fmt__core('field_type')

    def format_file_path(self, key):
        """Create output file path for a given field key.

        Args:
            key (namedtuple): Field key.

        """
        return self.file_path_fmt.format(**key._asdict())


#SR_TMP<
class ColorStr:

    def __init__(self, s, c):
        self.s = s
        self.c = c

    def __repr__(self):
        return f"{self.__class__.__name__}({self.s}, {self.c})"

    def __str__(self):
        return self.s

    def split(self, *args, **kwargs):
        return [
            self.__class__(s, self.c)
            for s in str.split(self.s, *args, **kwargs)
        ]

    def strip(self, *args, **kwargs):
        s = str.strip(self.s, *args, **kwargs)
        return self.__class__(s, self.c)


#SR_TMP>


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
                unit-pair).

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

            level_bot (float): Bottom level.

            level_top_unit (str): Unit of bottom level.

            level_top (float): Top level.

            level_top_unit (str): Unit of top level.

        """
        super().__init__()
        self.set('name', name, str)
        self.set('unit', unit, str)
        self.set(['level_bot', 'level_bot_unit'], level_bot, [float, str])
        self.set(['level_top', 'level_top_unit'], level_top, [float, str])

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
        self.set(['height', 'height_unit'], height, [float, str])
        self.set(['rate', 'rate_unit'], rate, [float, str])
        self.set(['mass', 'mass_unit'], mass, [float, str])

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
        self.set(['half_life', 'half_life_unit'], half_life, [float, str])
        self.set(['deposit_vel', 'deposit_vel_unit'], deposit_vel, [float, str])
        self.set(['sediment_vel', 'sediment_vel_unit'], sediment_vel, [float, str])
        self.set(['washout_coeff', 'washout_coeff_unit'], washout_coeff, [float, str])
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


class FlexPlotConcentration:
    """FLEXPART plot of particle concentration at a certain level.

    Attributes:
        <TODO>

    Methods:
        <TODO>

    """

    def __init__(self, rlat, rlon, fld, attrs_raw, conf=None):
        """Initialize instance of FlexPlotConcentration.

        Args:
            rlat (ndarray[float]): Rotated latitude (1d).

            rlon (ndarray[float]): Rotated longitude (1d).

            fld (ndarray[float, float]): Concentration field (2d).

            attrs_raw (dict): Attributes from the FLEXPART NetCDF file
                (gloabl, variable-specific, etc.).

            conf (dict, optional): Plot configuration. Defaults to None.

        """
        self.rlat = rlat
        self.rlon = rlon
        self.fld = np.where(fld > 0, fld, np.nan)
        self.conf = {} if conf is None else conf

        self.levels = None

        # Formatting arguments
        self._max_marker_kwargs = {
            'marker': '+',
            'color': 'black',
            'markersize': 10,
            'markeredgewidth': 1.5,
        }
        self._site_marker_kwargs = {
            'marker': '^',
            'markeredgecolor': 'red',
            'markerfacecolor': 'white',
            'markersize': 7.5,
            'markeredgewidth': 1.5,
        }

        # Map plot configuration
        self.map_conf = {
            'bbox_pad_rel': -0.01,
            'geogr_res': '10m',
            #'geogr_res': '50m',
            'ref_dist_x0': 0.046,
            'ref_dist_y0': 0.96,
        }

        #SR_TMP<
        self.attrs = FlexAttrsCollection(
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

        #SR_TMP>
        self.levels_log10 = np.arange(-9, -2 + 0.1, 1)
        self.levels = 10**self.levels_log10
        self.extend = 'max'
        #SR_TMP>

        self._run()

    def _run(self):

        # Prepare plot
        self.fig = plt.figure(figsize=(12, 9))
        pollat = self.attrs.grid.north_pole_lat
        pollon = self.attrs.grid.north_pole_lon
        self.ax_map = AxesMapRotPole(
            self.fig, self.rlat, self.rlon, pollat, pollon, **self.map_conf)

        # Plot particle concentration field
        self.map_add_particle_concentrations()

        # Add text boxes around plot
        self.fig_add_text_boxes()
        self.fill_box_top()
        self.fill_box_right_top()
        self.fill_box_right_bottom()
        self.fill_box_bottom()

    def map_add_particle_concentrations(self):
        """Plot the particle concentrations onto the map."""

        # Define colors
        # yapf: disable
        self.colors = (np.array([
            (255, 155, 255),  # -> under
            (224, 196, 172),  # \
            (221, 127, 215),  # |
            ( 99,   0, 255),  # |
            (100, 153, 199),  #  > range
            ( 93, 255,   2),  # |
            (199, 255,   0),  # |
            (255, 239,  57),  # /
            (200, 200, 200),  # -> over
        ], float)/255).tolist()
        # yapf: enable
        if self.extend in ['none', 'max']:
            self.colors.pop(0)  # Remove `under`
        if self.extend in ['none', 'min']:
            self.colors.pop(-1)  # Remove `over`

        # Add reference distance indicator
        self.ax_map.add_ref_dist_indicator()

        # Plot concentrations
        fld_log10 = np.log10(self.fld)
        handle = self.ax_map.contourf(
            fld_log10,
            levels=self.levels_log10,
            colors=self.colors,
            extend=self.extend,
        )

        # Add marker at location of maximum value
        self.ax_map.mark_max(self.fld, **self._max_marker_kwargs)

        # Add marker at release site
        self.ax_map.marker(
            self.attrs.release.site_lon,
            self.attrs.release.site_lat,
            **self._site_marker_kwargs,
        )

        return handle

    def fig_add_text_boxes(
            self,
            h_rel_t=0.1,
            h_rel_b=0.03,
            w_rel_r=0.25,
            pad_hor_rel=0.015,
            h_rel_box_rt=0.44):
        """Add empty text boxes to the figure around the map plot.

        Args:
            h_rel_t (float, optional): Height of top box as a fraction
                of the height of the map plot. Defaults to <TODO>.

            h_rel_b (float, optional): Height of bottom box as a
                fraction of the height of the map plot. Defaults to
                <TODO>.

            w_rel_r (float, optional): Width of the right boxes as a
                fraction of the width of the map plot. Default to <TODO>.

            pad_hor_rel (float, optional): Padding between map plot and
                the text boxes as a fraction of the map plot width. The
                same absolute padding is used in the horizontal and
                vertical direction. Defaults to <TODO>.

            h_rel_box_rt (float, optional): Height of the top box to
                the right of the map plot as a fraction of the combined
                height of both right boxees. Defaults to <TODO>.

        """

        # Freeze the map plot in order to fix it's coordinates (bbox)
        self.fig.canvas.draw()

        # Obtain aspect ratio of figure
        fig_pxs = self.fig.get_window_extent()
        fig_aspect = fig_pxs.width/fig_pxs.height

        # Get map dimensions in figure coordinates
        w_map, h_map = ax_dims_fig_coords(self.fig, self.ax_map.ax)

        # Relocate the map close to the lower left corner
        x0_map, y0_map = 0.05, 0.05
        self.ax_map.ax.set_position([x0_map, y0_map, w_map, h_map])

        # Determine height of top box and width of right boxes
        w_box = w_rel_r*w_map
        h_box_t = h_rel_t*h_map
        h_box_b = h_rel_b*h_map

        # Determine padding between plot and boxes
        pad_hor = pad_hor_rel*w_map
        pad_ver = pad_hor*fig_aspect

        # Add axes for text boxes (one on top, two to the right)
        self.axs_box = np.array([
            # Top
            AxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map,
                    y0_map + pad_ver + h_map,
                    w_map + pad_hor + w_box,
                    h_box_t,
                ]),
            # Right/top
            AxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map + pad_hor + w_map,
                    y0_map + 0.5*pad_ver + (1.0 - h_rel_box_rt)*h_map,
                    w_box,
                    h_rel_box_rt*h_map - 0.5*pad_ver,
                ]),
            # Right/bottom
            AxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map + pad_hor + w_map,
                    y0_map,
                    w_box,
                    (1.0 - h_rel_box_rt)*h_map - 0.5*pad_ver,
                ]),
            # Bottom
            AxesTextBox(
                self.fig,
                self.ax_map.ax, [
                    x0_map,
                    y0_map - h_box_b,
                    w_map + pad_hor + w_box,
                    h_box_b,
                ],
                show_border=False),
        ])

    def fill_box_top(self):
        """Fill the box above the map plot."""
        box = self.axs_box[0]

        # Top left: variable and level
        s = (
            f"{self.attrs.variable.name} "
            f"{self.attrs.variable.format_level_range()}")
        box.text('tl', s, size='xx-large')

        # Top center: species
        s = f"{self.attrs.species.name}"
        box.text('tc', s, size='xx-large')

        # Top right: datetime
        timestep_fmtd = self.attrs.simulation.format_now()
        s = f"{timestep_fmtd}"
        box.text('tr', s, size='xx-large')

        # Bottom left: release site
        s = f"Release site: {self.attrs.release.site_name}"
        box.text('bl', s, size='large')

        # Bottom right: time zone
        tz = pytz.timezone(self.attrs.release.site_tz_name)
        offset_str = self.attrs.simulation.now.astimezone(tz).strftime('%z')
        s = f"T$_{0}$ {offset_str[0:1]} {offset_str[1:3]}:{offset_str[3:5]}"
        box.text('br', s, size='large')

    def fill_box_right_top(self):
        """Fill the top box to the right of the map plot."""
        box = self.axs_box[1]

        #SR_TMP<
        fld_max = np.nanmax(self.fld)
        fld_max_fmtd = f"Max.: {fld_max:.2E} {self.attrs.variable.format_unit()}"
        #SR_TMP>

        # Add box title
        s = f"{self.attrs.variable.format_name()}"
        #box.text('tc', s=s, size='large')
        box.text('tc', s=s, dy=1, size='large')  #SR_DBG

        # Format level ranges (contour plot legend)
        labels = self._format_level_ranges()

        # Positioning parameters
        dx = 1.2
        dy_line = 3.0
        dy0_labels = 8.0
        w_box, h_box = 4, 2
        dy0_boxes = dy0_labels - 0.4

        # Add level labels
        box.text_block(
            'br',
            labels,
            dy0=dy0_labels,
            dy_line=dy_line,
            dx=-dx,
            reverse=True,
            size='small',
            #family='monospace',
        )

        # Add color boxes
        dy = dy0_boxes
        for color in self.colors[::-1]:
            box.color_rect(
                loc='bl',
                fc=color,
                ec='black',
                dx=dx,
                dy=dy,
                w=w_box,
                h=h_box,
                lw=1.0,
            )
            dy += dy_line

        # Spacing between colors and markers
        dy_spacing_markers = 0.5*dy_line

        # Add maximum value marker
        dy_max = dy0_labels - dy_spacing_markers - dy_line
        box.marker(
            loc='bl',
            dx=dx + 1.0,
            dy=dy_max + 0.7,
            **self._max_marker_kwargs,
        )
        box.text(
            loc='bl',
            s=fld_max_fmtd,
            dx=5.5,
            dy=dy_max,
            size='small',
        )

        # Add release site marker
        dy_site = dy0_labels - dy_spacing_markers - 2*dy_line
        box.marker(
            loc='bl',
            dx=dx + 1.0,
            dy=dy_site + 0.7,
            **self._site_marker_kwargs,
        )
        s = f"Release Site: {self.attrs.release.site_name}"
        box.text(loc='bl', s=s, dx=5.5, dy=dy_site, size='small')

    def _format_level_ranges(self):
        """Format the levels ranges for the contour plot legend."""

        labels = []

        # Under range
        if self.extend in ('min', 'both'):
            labels.append(self._format_label(None, self.levels[0]))

        # In range
        for lvl0, lvl1 in zip(self.levels[:-1], self.levels[1:]):
            labels.append(self._format_label(lvl0, lvl1))

        # Over range
        if self.extend in ('max', 'both'):
            labels.append(self._format_label(self.levels[-1], None))

        #SR_TMP<
        _n = len(self.colors)
        assert len(labels) == _n, f'{len(labels)} != {_n}'
        #SR_TMP>

        return labels

    def _format_label(self, lvl0, lvl1):

        def format_level(lvl):
            return '' if lvl is None else f"{lvl:.0E}".strip()

        if lvl0 is not None and lvl1 is not None:
            op = '-'
            return f"{format_level(lvl0):>6} {op} {format_level(lvl1):<6}"
        else:
            if lvl0 is not None:
                op = '>'
                lvl = lvl0
            elif lvl1 is not None:
                op = '<'
                lvl = lvl1

            return f"{op+' '+format_level(lvl):^15}"

    def fill_box_right_bottom(self):
        """Fill the bottom box to the right of the map plot."""
        box = self.axs_box[2]

        # Add box title
        #box.text('tc', 'Release', size='large')
        box.text('tc', 'Release', dy=-1.5, size='large')

        # Release site coordinates
        lat = Degrees(self.attrs.release.site_lat)
        lat_fmtd = (
            f"{lat.degs()}$^\circ\,${lat.mins()}'$\,$N"
            f" ({lat.frac()}$^\circ\,$N)")
        lon = Degrees(self.attrs.release.site_lon)
        lon_fmtd = (
            f"{lon.degs()}$^\circ\,${lon.mins()}'$\,$E"
            f" ({lon.frac()}$^\circ\,$E)")

        info_blocks = dedent(
            f"""\
            Latitude:\t{lat_fmtd}
            Longitude:\t{lon_fmtd}
            Height:\t{self.attrs.release.format_height()}

            Start:\t{self.attrs.simulation.format_start()}
            End:\t{self.attrs.simulation.format_end()}
            Rate:\t{self.attrs.release.format_rate()}
            Total Mass:\t{self.attrs.release.format_mass()}

            Substance:\t{self.attrs.species.name}
            Half-Life:\t{self.attrs.species.format_half_life()}
            Deposit. Vel.:\t{self.attrs.species.format_deposit_vel()}
            Sediment. Vel.:\t{self.attrs.species.format_sediment_vel()}
            Washout Coeff.:\t{self.attrs.species.format_washout_coeff()}
            Washout Exponent:\t{self.attrs.species.washout_exponent}
            """)

        # Add lines bottom-up (to take advantage of baseline alignment)
        dy0 = 3
        dy = 2.5
        box.text_blocks_hfill(
            'b',
            dy0=dy0,
            dy_line=dy,
            blocks=info_blocks,
            reverse=True,
            size='small')

    def fill_box_bottom(self):
        """Fill the box to the bottom of the map plot."""
        box = self.axs_box[3]

        # FLEXPART/model info
        _model = self.attrs.simulation.model_name
        _simstart_fmtd = self.attrs.simulation.format_start()
        s = f"FLEXPART based on {_model} {_simstart_fmtd}"
        box.text('tl', dx=-0.7, dy=0.5, s=s, size='small')

        # MeteoSwiss Copyright
        cpright_fmtd = u"\u00a9MeteoSwiss"
        box.text('tr', dx=0.7, dy=0.5, s=cpright_fmtd, size='small')

    def save(self, file_path, format=None):
        """Save the plot to disk.

        Args:
            file_path (str): Output file name, incl. path.

            format (str): Plot format (e.g., 'png', 'pdf'). Defaults to
                None. If ``format`` is None, the plot format is derived
                from the extension of ``file_path``.

        """
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.pdf', '.png']:
                raise ValueError(
                    f"Cannot derive format from extension '{ext}'"
                    f"derived from '{os.path.basename(file_path)}'")
            format = ext[1:]
        self.fig.savefig(
            file_path,
            facecolor=self.fig.get_facecolor(),
            edgecolor=self.fig.get_edgecolor(),
            bbox_inches='tight',
            pad_inches=0.15,
        )
        plt.close(self.fig)


class AxesMap:

    def __init__(self, fig):
        """Initialize instance of AxesMap.

        Args:
            fig (Figure): Figure to which to map axes is added.
        """
        self.fig = fig


class AxesConfMap:
    pass


class AxesConfMapRotPole(AxesConfMap):

    def __init__(
            self,
            *,
            bbox_pad_rel=0.01,
            geogr_res='50m',
            ref_dist=100,
            ref_dist_unit='km',
            ref_dist_dir='east',
            ref_dist_x0=0.05,
            ref_dist_y0=0.95,
    ):
        """

        Kwargs:
            bbox_pad_rel (float, optional): Relative padding applied
                to the bounding box of the input data (derived from
                rotated lat/lon), as a fraction of its size in both
                directions. If positive/zero/negative, the shown map
                is bigger/equal/smaller in size than the data domain.
                Defaults to 0.01.

            geogr_res (str, optional): Resolution of geographic map
                elements.  Defaults to '50m'.

            ref_dist (float, optional): Reference distance in
                ``ref_dist_unit``. Defaults to 100.

            ref_dist_unit (str, optional): Unit of reference distance
                (``ref_dist``). Defaults to 'km'.

            ref_dist_dir (str, optional): Direction in which the
                reference distance indicator is drawn (starting from
                (``ref_dist_x0``, ``ref_dist_y0``), relative to the
                plot itself, NOT to the underlying map (so 'east'
                means straight to the right, regardless of the
                projection of the map plot). Defaults to 'east'.

            ref_dist_x0 (float, optional): Horizontal starting point
                of reference distance indicator in axes coordinates.
                Defaults to 0.05.

            ref_dist_y0 (float, optional): Vertical starting point
                of reference distance indicator in axes coordinates.
                Defaults to 0.95.

        """
        self.bbox_pad_rel = bbox_pad_rel
        self.geogr_res = geogr_res
        self.ref_dist = ref_dist
        self.ref_dist_unit = ref_dist_unit
        self.ref_dist_dir = ref_dist_dir
        self.ref_dist_x0 = ref_dist_x0
        self.ref_dist_y0 = ref_dist_y0


#SR_TODO Push non-rotated-pole specific code up into AxesMap
class AxesMapRotPole(AxesMap):
    """Map plot axes for FLEXPART plot for rotated-pole data.

    Attributes:
        <TODO>

    Methods:
        <TODO>

    """

    def __init__(self, fig, rlat, rlon, pollat, pollon, **conf):
        """Initialize instance of AxesMapRotPole.

        Args:
            fig (Figure): Figure to which to map axes is added.

            rlat (ndarray[float]): Rotated latitude coordinates.

            rlon (ndarray[float]): Rotated longitude coordinates.

            pollat (float): Latitude of rotated pole.

            pollon (float): Longitude of rotated pole.

            **conf: Keyword arguments to create a configuration object
                of type ``AxesConfMapRotPole``.

        """
        self.fig = fig
        self.rlat = rlat
        self.rlon = rlon
        self.conf = AxesConfMapRotPole(**conf)

        # Determine zorder of unique plot elements, from low to high
        zorders_const = [
            'map',
            'grid',
            'fld',
            'marker',
        ]
        d0, dz = 1, 1
        self.zorder = {e: d0 + i*dz for i, e in enumerate(zorders_const)}

        # Prepare the map projections (input, plot, geographic)
        self.prepare_projections(pollat, pollon)

        # Initialize plot
        self.ax = self.fig.add_subplot(projection=self.proj_plot)

        # Set extent of map
        bbox = [self.rlon[0], self.rlon[-1], self.rlat[0], self.rlat[-1]]
        bbox = pad_bbox(*bbox, pad_rel=self.conf.bbox_pad_rel)
        self.ax.set_extent(bbox, self.proj_data)

        # Activate grid lines
        gl = self.ax.gridlines(
            linestyle=':',
            linewidth=1,
            color='black',
            zorder=self.zorder['grid'],
        )
        gl.xlocator = mpl.ticker.FixedLocator(np.arange(-2, 18.1, 2))
        gl.ylocator = mpl.ticker.FixedLocator(np.arange(40, 52.1, 2))

        # Add geographical elements (coasts etc.)
        self.add_geography(self.conf.geogr_res)

        # Show data domain outline
        self.add_data_domain_outline()

    def prepare_projections(self, pollat, pollon):
        """Prepare projections to transform the data for plotting.

        Args:
            pollat (float): Lattitude of rorated pole.

            pollon (float): Longitude of rotated pole.

        """

        # Projection of input data: Rotated Pole
        self.proj_data = cartopy.crs.RotatedPole(
            pole_latitude=pollat, pole_longitude=pollon)

        # Projection of plot
        clon = 180 + pollon
        self.proj_plot = cartopy.crs.TransverseMercator(central_longitude=clon)

        # Geographical lat/lon arrays
        self.proj_geo = cartopy.crs.PlateCarree()
        rlat2d, rlon2d = np.meshgrid(self.rlat, self.rlon)
        self.lon2d, self.lat2d, _ = self.proj_geo.transform_points(
            self.proj_data, rlat2d, rlon2d).T

    def add_geography(self, scale):
        """Add geographic elements: coasts, countries, colors, ...

        Args:
            scale (str): Spatial scale of elements, e.g., '10m', '50m'.

        """

        self.ax.coastlines(resolution=scale)

        self.ax.background_patch.set_facecolor(cartopy.feature.COLORS['water'])

        self.ax.add_feature(
            cartopy.feature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_countries_lakes',
                scale=scale,
                edgecolor='black',
                facecolor='white',
            ),
            zorder=self.zorder['map'],
        )

    def add_data_domain_outline(self):
        """Add domain outlines to map plot."""

        rlon0, rlon1 = self.rlon[[0, -1]]
        rlat0, rlat1 = self.rlat[[0, -1]]
        xs = [rlon0, rlon1, rlon1, rlon0, rlon0]
        ys = [rlat0, rlat0, rlat1, rlat1, rlat0]

        self.ax.plot(xs, ys, transform=self.proj_data, c='black', lw=1)

    def contourf(self, fld, **kwargs):
        """Plot a color contour field on the map.

        Args:
            fld (ndarray[float, float]): Field to plot.

            **kwargs: Arguments passed to ax.contourf().

        Returns:
            Plot handle.

        """
        handle = self.ax.contourf(
            self.rlon,
            self.rlat,
            fld,
            transform=self.proj_data,
            zorder=self.zorder['fld'],
            **kwargs,
        )
        return handle

    def marker(self, lon, lat, marker, **kwargs):
        """Add a marker at a location in natural coordinates."""
        rlon, rlat = self.proj_data.transform_point(lon, lat, self.proj_geo)
        handle = self.marker_rot(rlon, rlat, marker, **kwargs)
        return handle

    def marker_rot(self, rlon, rlat, marker, **kwargs):
        """Add a marker at a location in rotated coordinates."""
        handle = self.ax.plot(
            rlon,
            rlat,
            marker=marker,
            transform=self.proj_data,
            zorder=self.zorder['marker'],
            **kwargs,
        )
        return handle

    def mark_max(self, fld, marker, **kwargs):
        """Mark the location of the field maximum."""
        jmax, imax = np.unravel_index(np.nanargmax(fld), fld.shape)
        rlon, rlat = self.rlon[imax], self.rlat[jmax]
        handle = self.marker_rot(rlon, rlat, marker, **kwargs)
        return handle

    def transform_axes_to_geo(self, x, y):
        """Transform axes coordinates to geographic coordinates."""

        # Convert `Axes` to `Display`
        xy_disp = self.ax.transAxes.transform((x, y))

        # Convert `Display` to `Data`
        x_data, y_data = self.ax.transData.inverted().transform(xy_disp)

        # Convert `Data` to `Geographic`
        xy_geo = self.proj_geo.transform_point(x_data, y_data, self.proj_plot)

        return xy_geo

    def add_ref_dist_indicator(self):
        """Add a reference distance indicator.

        The configuration is obtained from an ``AxesConfMapRotPole``
        instance.

        Returns:
            float: Actual distance within specified relative tolerance.

        """
        # Obtain setup
        dist = self.conf.ref_dist
        unit = self.conf.ref_dist_unit
        dir = self.conf.ref_dist_dir
        x0 = self.conf.ref_dist_x0
        y0 = self.conf.ref_dist_y0

        # Determine end point (axes coordinates)
        x1, y1, _ = MapPlotGeoDist(self, unit).measure(x0, y0, dist, dir)

        # Draw line
        self.ax.plot(
            [x0, x1],
            [y0, y1],
            transform=self.ax.transAxes,
            linestyle='-',
            linewidth=2.0,
            color='k',
        )

        # Add label
        self.ax.text(
            x=0.5*(x1 + x0),
            y=0.5*(y1 + y0) + 0.01,
            s=f"{dist:g} {unit}",
            transform=self.ax.transAxes,
            horizontalalignment='center',
            fontsize='large',
        )


def pad_bbox(lon0, lon1, lat0, lat1, pad_rel):
    """Add relative padding to a bounding box.

    Args:
        lon0 (float): Longitude of lower-left domain corner.

        lon1 (float): Longitude of upper-right domain corner.

        lat0 (float): Latitude of lower-left domain corner.

        lat1 (float): Latitude of upper-right domain corner.

        pad_rel (float): Relative padding applied to box defined
            by the lower-left and upper-right domain corners, as a
            fraction of the width/height of the box in the
            horizontal/vertical. Can be negative.

    Returns:
        ndarray[float, n=4]: Padded bounding box comprised of the
            following four elements: [lon0, lon1, lat0, lat1].

    """

    pad_fact_x, pad_fact_y = [pad_rel]*2

    dlon = lon1 - lon0
    dlat = lat1 - lat0

    padx = dlon*pad_fact_x
    pady = dlon*pad_fact_y

    bbox = np.array([
        lon0 - padx,
        lon1 + padx,
        lat0 - pady,
        lat1 + pady,
    ], float)

    return bbox


class MapPlotGeoDist:
    """Measture geo. distance along a line on a map plot."""

    def __init__(self, ax_map, unit='km', p=0.001):
        """Initialize an instance of MapPlotGeoDist.

        Args:
            ax_map (AxesMap*): Map plot object providing the
                projections etc. [TODO reformulate!]

            unit (str, optional): Unit of ``dist``. Defaults to 'km'.

            p (float, optional): Required precision as a fraction of
                ``dist``. Defaults to 0.001.

        """
        self.ax_map = ax_map
        self.unit = unit
        self.p = p

    def reset(self, dist=None, dir=None):
        self._set_dist(dist)
        self._set_dir(dir)

        self._step_ax_rel = 0.1

    def _set_dist(self, dist):
        """Check and set the target distance."""
        self.dist = dist
        if dist is not None:
            if dist <= 0.0:
                raise ValueError(f"dist not above zero: {dist}")

    def _set_dir(self, dir):
        """Check and set the direction."""
        self.dir = dir
        if dir is not None:
            if dir == 'east':
                self._dx_unit = 1
                self._dy_unit = 0
            else:
                raise NotImplementedError(
                    f"dir '{direction}' not among {dir_choices}")

    def measure(self, x0, y0, dist, dir='east'):
        """Measure geo. distance along a straight line on the plot."""
        self.reset(dist=dist, dir=dir)

        #SR_DBG<
        debug = False
        #SR_DBG>

        step_ax_rel = 0.1
        refine_quot = 3

        dist0 = 0.0
        x, y = x0, y0

        iter_max = 99999
        for iter_i in range(iter_max):

            # Step away from point until target distance exceeded
            path, dists = self._overstep(x, y, dist0, step_ax_rel)

            # Select the largest distance
            i_sel = -1
            dist = dists[i_sel]
            # Note: We could also check `dists[-2]` and return that
            # if it were sufficiently close (based on the relative
            # error) and closer to the targest dist than `dist[-1]`.

            # Compute the relative error
            err = abs(dist - self.dist)/self.dist
            # Note: `abs` only necessary if `dist` could be `dist[-2]`

            #SR_DBG<
            if debug:
                print(
                    f"{iter_i:2d}"
                    f" ({x0:.2f}, {y0:.2f})"
                    f"--{{{dist0:6.2f} {self.unit}}}"
                    f"->({x:.2f}, {y:.2f})"
                    f"--{{{dists[-1] - dist0:6.2f} {self.unit}}}"
                    f"->({path[-1][0]:.2f}, {path[-1][1]:.2f})"
                    f" : {dist:6.2f} {self.unit}"
                    f" : {err:10.5%}")
            #SR_DBG>

            if err < self.p:
                # Error sufficiently small: We're done!
                x1, y1 = path[i_sel]
                break

            dist0 = dists[-2]
            x, y = path[-2]
            step_ax_rel /= refine_quot

        else:
            raise MaxIterationError(iter_max)

        return x1, y1, dist

    def _overstep(self, x0_ax, y0_ax, dist0, step_ax_rel):
        """Move stepwise until the target distance is exceeded."""

        #SR_DBG<
        debug = False
        #SR_DBG>

        # Transform starting point to geographical coordinates
        x0_geo, y0_geo = self.ax_map.transform_axes_to_geo(x0_ax, y0_ax)

        path_ax = [(x0_ax, y0_ax)]
        dists = [dist0]

        # Step away until target distance exceeded
        dist = 0.0
        x1_ax, y1_ax = x0_ax, y0_ax
        while dist < self.dist - dist0:

            # Move one step farther away from starting point
            x1_ax += self._dx_unit*step_ax_rel
            y1_ax += self._dy_unit*step_ax_rel

            # Transform current point to gegographical coordinates
            x1_geo, y1_geo = self.ax_map.transform_axes_to_geo(x1_ax, y1_ax)

            # Compute geographical distance from starting point
            dist = self.comp_dist(x0_geo, y0_geo, x1_geo, y1_geo)

            path_ax.append((x1_ax, y1_ax))
            dists.append(dist + dist0)

            #SR_DBG<
            if debug:
                print(
                    f"({x0_ax:.2f}, {y0_ax:.2f})"
                    f"=({x1_geo:.2f}, {y1_geo:.2f})"
                    f"--{{{dist:6.2f}"
                    f"/{self.dist - dist0:6.2f} {self.unit}}}"
                    f"->({x1_ax:.2f}, {y1_ax:.2f})"
                    f"=({x1_geo:.2f}, {y1_geo:.2f})")
            #SR_DBG>

        return path_ax, dists

    def comp_dist(self, lon0, lat0, lon1, lat1):
        """Compute the great circle distance between two points."""
        dist_obj = geopy.distance.great_circle((lat0, lon0), (lat1, lon1))
        if self.unit == 'km':
            return dist_obj.kilometers
        else:
            raise NotImplementedError(f"great circle distance in {self.unit}")


class AxesTextBox:
    """Text box axes for FLEXPART plot.

    Attributes:
        <TODO>

    Methods:
        <TODO>

    """

    def __init__(self, fig, ax_ref, rect, show_border=True):
        """Initialize instance of AxesTextBox.

        Args:
            fig (Figure): Figure to which to add the text box axes.

            ax_ref (Axis): Reference axes.

            rect (list): Rectangle [left, bottom, width, height].

            show_border (bool, optional): Show the border of the box.
                Default to True.

        """

        self.fig = fig
        self.ax_ref = ax_ref

        self.ax = self.fig.add_axes(rect)
        self.ax.axis('off')

        if show_border:
            self.draw_border()

        self.compute_unit_distances()

        # Text base line settings
        # (line below text for debugging)
        self._show_baselines = False
        self._baseline_kwargs_default = {
            'color': 'black',
            'linewidth': 0.5,
        }
        self._baseline_kwargs = self._baseline_kwargs_default

    def draw_border(self, x=0.0, y=0.0, w=1.0, h=1.0, fc='white', ec='black'):
        """Draw a box onto the axes."""
        self.ax.add_patch(
            mpl.patches.Rectangle(
                xy=(x, y),
                width=w,
                height=h,
                transform=self.ax.transAxes,
                fc=fc,
                ec=ec,
            ))

    def compute_unit_distances(self, unit_w_map_rel=0.01):
        """Compute unit distances in x and y for text positioning.

        To position text nicely inside a box, it is handy to have
        unit distances of absolute length to work with that are
        independent of the size of the box (i.e., axes). This method
        computes such distances as a fraction of the width of the
        map plot.

        Args:
            unit_w_map_rel (float, optional): Fraction of the width
                of the map plot that corresponds to one unit distance.
                Defaults to 0.01.

        """
        w_map_fig, _ = ax_dims_fig_coords(self.fig, self.ax_ref)
        w_box_fig, h_box_fig = ax_dims_fig_coords(self.fig, self.ax)

        self.dx = unit_w_map_rel*w_map_fig/w_box_fig
        self.dy = unit_w_map_rel*w_map_fig/h_box_fig

    def text(self, loc, s, dx=None, dy=None, **kwargs):
        """Add text positioned relative to a reference location.

        Args:
            loc (int or str): Reference location parameter used to
                initialize an instance of ``BoxLocation``.

            s (str): Text string.

            dx (float, optional): Horizontal offset in number of unit
                distances. Can be negative. Defaults to None.

            dy (float, optional): Vertical offset in number of unit
                distances. Can be negative. Defaults to None.

            **kwargs: Formatting options passed to ax.text().

        """

        # Derive location variables from parameter
        loc = BoxLocation(loc, self.dx, self.dy)
        x = loc.get_x(dx)
        y = loc.get_y(dy)
        ha = loc.get_ha()
        va = loc.get_va()

        # Add alignment parameters, unless specified in input kwargs
        kwargs['ha'] = kwargs.get('horizontalalignment', kwargs.get('ha', ha))
        kwargs['va'] = kwargs.get('verticalalignment', kwargs.get('va', va))

        if kwargs['va'] == 'top_baseline':
            # SR_NOTE: [2019-06-11]
            # Ideally, we would like to align text by a `top_baseline`,
            # analogous to baseline and center_baseline, which does not
            # depend on the height of the letters (e.g., '$^\circ$'
            # lifts the top of the text, like 'g' at the bottom). This
            # does not exist, however, and attempts to emulate it by
            # determining the line height (e.g., draw an 'M') and then
            # shifting y accordingly (with `baseline` alignment) were
            # not successful.
            raise NotImplementedError(f"verticalalignment='{kwargs['vs']}'")

        #SR_TMP<
        if isinstance(s, ColorStr):
            kwargs['color'] = s.c
        #SR_TMP>

        # Add text
        self.ax.text(x=x, y=y, s=s, **kwargs)

        if self._show_baselines:
            # Draw a horizontal line at the text baseline
            self.ax.axhline(y, **self._baseline_kwargs)

    def text_block(self, loc, block, colors=None, **kwargs):
        """Add a text block comprised of multiple lines.

        Args:
            loc (int or str): Reference location. For details see
                ``AxesTextBox.text``.

            block (list[str]): Text block.

            colors (list[<color>], optional): Line-specific colors.
                Defaults to None. If not None, must have same length
                as ``block``. Omit individual lines with None.

            **kwargs: Positioning and formatting options passed to
                ``AxesTextBox.text_blocks``.

        """
        self.text_blocks(loc, [block], colors=[colors], **kwargs)

    def text_blocks(
            self,
            loc,
            blocks,
            *,
            dy0=None,
            dy_line=None,
            dy_block=None,
            reverse=False,
            colors=None,
            **kwargs):
        """Add multiple text blocks.

        Args:
            loc (int or str): Reference location. For details see
                ``AxesTextBox.text``.

            blocks (list[list[str]]): List of text blocks, each of
                which constitutes a list of lines.

            dy0 (float, optional): Initial vertical offset in number
                of unit distances. Can be negative. Defaults to
                ``dy_line``.

            dy_line (float, optional): Incremental vertical offset
                between lines. Can be negative. Defaults to 2.5.

            dy_block (float, optional): Incremental vertical offset
                between blocks of lines. Can be negative. Defaults to
                ``dy_line``.

            dx (float, optional): Horizontal offset in number
                of unit distances. Can be negative. Defaults to 0.0.

            reverse (bool, optional): If True, revert the blocka and
                line order. Defaults to False. Note that if line-
                specific colors are passed, they must be in the same
                order as the unreversed blocks.

            colors (list[list[<color>]], optional): Line-specific
                colors in each block. Defaults to None. If not None,
                must have same shape as ``blocks``. Omit individual
                blocks or lines in blocks with None.

            **kwargs: Formatting options passed to ``ax.text``.

        """
        if dy_line is None:
            dy_line = 2.5
        if dy0 is None:
            dy0 = dy_line
        if dy_block is None:
            dy_block = dy_line

        # Fetch text color (fall-back if no line-specific color)
        default_color = kwargs.pop('color', kwargs.pop('c', 'black'))

        # Rename colors variable
        colors_blocks = colors
        del colors

        # Prepare line colors
        if colors_blocks is None:
            colors_blocks = [None]*len(blocks)
        elif len(colors_blocks) != len(blocks):
            raise ValueError(
                f"colors must have same length as blocks:"
                f"  {len(colors)} != {len(blocks)}")
        for i, block in enumerate(blocks):
            if colors_blocks[i] is None:
                colors_blocks[i] = [None]*len(block)
            elif len(colors_blocks) != len(blocks):
                ith = f"{i}{({1: 'st', 2: 'nd', 3: 'rd'}.get(i, 'th'))}"
                raise ValueError(
                    f"colors of {ith} block must have same length as block:"
                    f"  {len(colors_blocks[i])} != {len(block)}")
            for j in range(len(block)):
                if colors_blocks[i][j] is None:
                    colors_blocks[i][j] = default_color

        if reverse:
            # Revert order of blocks and lines
            def revert(lsts):
                return [[l for l in lst[::-1]] for lst in lsts[::-1]]

            blocks = revert(blocks)
            colors_blocks = revert(colors_blocks)

        dy = dy0
        for i, block in enumerate(blocks):
            for j, line in enumerate(block):
                self.text(
                    loc,
                    s=line,
                    dy=dy,
                    color=colors_blocks[i][j],
                    **kwargs,
                )
                dy += dy_line
            dy += dy_block

    def text_block_hfill(self, loc_y, block, **kwargs):
        """Single block of horizontally filled lines.

        See ``AxesTextBox.text_blocks_hfill`` for details.
        """
        self.text_blocks_hfill(loc_y, [block], **kwargs)

    def text_blocks_hfill(self, loc_y, blocks, **kwargs):
        """Add blocks of horizontally-filling lines.

        Lines are split at a tab character ('\t'), with the text before
        the tab left-aligned, and the text after right-aligned.

        Args:
            locy (int or str): Vertical reference location. For details
                see ``AxesTextBox.text`` (vertical component
                only).

            blocks (str or list[str] or list[list[str]] or
                list[list[tuple]]): Text blocks, each of which consists
                of lines, each of which in turn consists of a left and
                right part. Possible formats:

                  - The blocks can be a multiline string, with empty
                    lines separating the individual blocks; or a list.

                  - In case of list blocks, each block can in turn
                    constitute a multiline string, or a list of lines.

                  - In case of a list block, each line can in turn
                    constitute a string, or a two-element string tuple.

                  - Lines represented by a string are split into a left
                    and right part at the first tab character ('\t').

            **kwargs: Location and formatting options passed to
                ``AxesTextBox.text_blocks``.
        """

        if isinstance(blocks, str):
            # Whole blocks is a multiline string
            blocks = blocks.strip().split('\n\n')

        # Handle case where a multiblock string is embedded
        # in a blocks list alongside string or list blocks
        blocks_orig, blocks = blocks, []
        for block in blocks_orig:
            if isinstance(block, str):
                # Possible multiblock string (if with empty line)
                for subblock in block.strip().split('\n\n'):
                    blocks.append(subblock)
            else:
                # List block
                blocks.append(block)

        # Separate left and right parts of lines
        blocks_l, blocks_r = [], []
        for block in blocks:

            if isinstance(block, str):
                # Turn multiline block into list block
                block = block.strip().split('\n')

            blocks_l.append([])
            blocks_r.append([])
            for line in block:

                # Obtain left and right part of line
                if isinstance(line, str):
                    str_l, str_r = line.split('\t', 1)
                elif len(line) == 2:
                    str_l, str_r = line
                else:
                    raise ValueError(f"invalid line: {line}")

                blocks_l[-1].append(str_l)
                blocks_r[-1].append(str_r)

        dx_l = kwargs.pop('dx', None)
        dx_r = None if dx_l is None else -dx_l

        # Add lines to box
        self.text_blocks('bl', blocks_l, dx=dx_l, **kwargs)
        self.text_blocks('br', blocks_r, dx=dx_r, **kwargs)

    def sample_labels(self):
        """Add sample text labels in corners etc."""
        kwargs = dict(fontsize=9)
        self.text('bl', 'bot. left', **kwargs)
        self.text('bc', 'bot. center', **kwargs)
        self.text('br', 'bot. right', **kwargs)
        self.text('cl', 'center left', **kwargs)
        self.text('cc', 'center', **kwargs)
        self.text('cr', 'center right', **kwargs)
        self.text('tl', 'top left', **kwargs)
        self.text('tc', 'top center', **kwargs)
        self.text('tr', 'top right', **kwargs)

    def show_baselines(self, val=True, **kwargs):
        """Show the base line of a text command (for debugging).

        Args:
            val (bool, optional): Whether to show or hide the baseline.
                Defaults to True.

            **kwargs: Keyword arguments passed to ax.axhline().

        """
        self._show_baselines = val
        self._baseline_kwargs = self._baseline_kwargs_default
        self._baseline_kwargs.update(kwargs)

    def color_rect(
            self, loc, fc, ec=None, dx=None, dy=None, w=3, h=2, **kwargs):
        """Add a colored rectangle.

        Args:
            loc (int or str): Reference location parameter used to
                initialize an instance of ``BoxLocation``.

            fc (<color>): Face color.

            ec (<color>, optional): Edge color. Defaults to face color.

            dx (float, optional): Horizontal offset in number of unit
                distances. Can be negative. Defaults to None.

            dy (float, optional): Vertical offset in number of unit
                distances. Can be negative. Defaults to None.

            w (float, optional): Width in number of unit distances.
                Defaults to <TODO>.

            h (float, optional): Height in number of unit distances.
                Defaults to <TODO>.

            **kwargs: Keyword arguments passed to
                ``matplotlib.patches.Rectangle``.

        """
        if ec is None:
            ec = fc

        # Derive location variables from parameter
        loc = BoxLocation(loc, self.dx, self.dy)
        x = loc.get_x(dx)
        y = loc.get_y(dy)

        # Transform box dimensions to axes coordinates
        w = w*self.dx
        h = h*self.dy

        # Draw rectangle
        p = mpl.patches.Rectangle(
            (x, y),
            w,
            h,
            fill=True,
            fc=fc,
            ec=ec,
            **kwargs,
        )
        self.ax.add_patch(p)

        if self._show_baselines:
            self.ax.axhline(y, **self._baseline_kwargs)

    def marker(self, loc, marker, dx=None, dy=None, **kwargs):
        """Add a marker symbol.

        Args:
            loc (int or str): Reference location parameter used to
                initialize an instance of ``BoxLocation``.

            marker (str or int) Marker style passed to ``mpl.plot``.
                See ``matplotlib.markers`` for more information.

            dx (float, optional): Horizontal offset in number of unit
                distances. Can be negative. Defaults to None.

            dy (float, optional): Vertical offset in number of unit
                distances. Can be negative. Defaults to None.

            **kwargs: Keyword arguments passed to ``mpl.plot``.

        """

        # Derive location variables from parameter
        loc = BoxLocation(loc, self.dx, self.dy)
        x = loc.get_x(dx)
        y = loc.get_y(dy)

        # Add marker
        self.ax.plot([x], [y], marker=marker, **kwargs)

        if self._show_baselines:
            self.ax.axhline(y, **self._baseline_kwargs)


class BoxLocation:
    """Represents reference location inside a box on a 3x3 grid."""

    def __init__(self, loc, dx, dy):
        """Initialize an instance of BoxLocation.

        Args:
            loc (int or str): Location parameter. Takes one of three
                formats: integer, short string, or long string.

                Choices:

                    int     short   long
                    00      bl      bottom left
                    01      bc      bottom center
                    02      br      bottom right
                    10      cl      center left
                    11      cc      center
                    12      cr      center right
                    20      tl      top left
                    21      tc      top center
                    22      tr      top right

        """
        self.loc = loc
        self.loc_y, self.loc_x = self._prepare_loc()
        self.dx, self.dy = dx, dy

    def _prepare_loc(self):
        """Split and evaluate components of location parameter."""

        loc = str(self.loc)

        # Split location into vertical and horizontal part
        if len(loc) == 2:
            loc_y, loc_x = loc
        elif loc == 'center':
            loc_y, loc_x = loc, loc
        else:
            loc_y, loc_x = line.split(' ', 1)

        # Evaluate location components
        loc_y = self._eval_loc_vert(loc_y)
        loc_x = self._eval_loc_horz(loc_x)

        return loc_y, loc_x

    def _eval_loc_vert(self, loc):
        """Evaluate vertical location component."""
        if loc in (0, '0', 'b', 'bottom'):
            return 'b'
        elif loc in (1, '1', 'c', 'center'):
            return 'c'
        elif loc in (2, '2', 't', 'top'):
            return 't'
        raise ValueError(f"invalid vertical location component '{loc}'")

    def _eval_loc_horz(self, loc):
        """Evaluate horizontal location component."""
        if loc in (0, '0', 'l', 'left'):
            return 'l'
        elif loc in (1, '1', 'c', 'center'):
            return 'c'
        elif loc in (2, '2', 'r', 'right'):
            return 'r'
        raise ValueError(f"invalid horizontal location component '{loc}'")

    def get_va(self):
        """Derive the vertical alignment variable."""
        return {
            'b': 'baseline',
            'c': 'center_baseline',
            #'t': 'top_baseline',  # unfortunately nonexistent
            't': 'top',
        }[self.loc_y]

    def get_ha(self):
        """Derive the horizontal alignment variable."""
        return {
            'l': 'left',
            'c': 'center',
            'r': 'right',
        }[self.loc_x]

    def get_y0(self):
        """Derive the vertical baseline variable."""
        return {
            'b': 0.0 + self.dy,
            'c': 0.5,
            't': 1.0 - self.dy,
        }[self.loc_y]

    def get_x0(self):
        """Derive the horizontal baseline variable."""
        return {
            'l': 0.0 + self.dx,
            'c': 0.5,
            'r': 1.0 - self.dx,
        }[self.loc_x]

    def get_x(self, dx=None):
        """Derive the horizontal position."""
        if dx is None:
            dx = 0.0
        return self.get_x0() + dx*self.dx

    def get_y(self, dy=None):
        """Derive the vertical position."""
        if dy is None:
            dy = 0.0
        return self.get_y0() + dy*self.dy


def ax_dims_fig_coords(fig, ax):
    """Get the dimensions of an axes in figure coords."""
    trans = fig.transFigure.inverted()
    x, y, w, h = ax.bbox.transformed(trans).bounds
    return w, h


def colors_from_cmap(cmap, n_levels, extend):
    """Get colors from cmap for given no. levels and extend param."""
    colors = cmap(np.linspace(0, 1, n_levels + 1))
    if extend == 'both':
        return colors
    elif extend == 'min':
        return colors[:-1]
    elif extend == 'max':
        return colors[1:]
    else:
        return colors[1:-1]

# -*- coding: utf-8 -*-
"""
Plots.
"""
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import logging as log
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
import re

from matplotlib import ticker
from .utils_dev import ipython  #SRU_DEV


class FlexPlotter:
    """Create one or more FLEXPLART plots of a certain type.

    Args:
        type_ (str): Type of plot.

    """

    def __init__(self, type_):
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

        Returns:
            list[str]: Output file paths.

        """
        self.data = data
        self.file_path_fmt = file_path_fmt

        if self.type_ == 'concentration':
            return self._run_concentration()
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
                'field': self.data.field(key),
                'attrs': {},  #SRU_TMP
            }

            FlexPlotConcentration(**kwargs).save(file_path)

            file_paths.append(file_path)

        return file_paths

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


class FlexPlotConcentration:
    """FLEXPART plot of particle concentration at a certain level.

    Args:
        TODO

    """

    def __init__(self, rlat, rlon, field, attrs):

        self.rlat = rlat
        self.rlon = rlon
        self.field = field
        self.attrs = attrs

        self._create()

    def _create(self):
        """Create the plot."""

        #SRU_TMP< TODO Extract from NetCDF file
        self.attrs['rotated_pole'] = {
            'grid_north_pole_latitude': 43.0,
            'grid_north_pole_longitude': -170.0,
        }
        #SRU_TMP>

        #SRU_TMP>
        pollat = self.attrs['rotated_pole']['grid_north_pole_latitude']
        pollon = self.attrs['rotated_pole']['grid_north_pole_longitude']
        bbox = [0.0, 17.0, 41.7, 50.6]
        contour_levels = 10**np.arange(-1, 9.1, 1)
        #SRU_TMP>

        # Define projections
        self.proj_data = ccrs.RotatedPole(
            pole_latitude=pollat,
            pole_longitude=pollon,
        )
        self.proj_plot = ccrs.EuroPP()

        self.fig = plt.figure()
        self.ax = plt.subplot(projection=self.proj_plot)

        self.ax.set_extent(bbox)
        self.ax.gridlines()
        #self.ax.coastlines()

        self.ax.add_image(cimgt.Stamen('terrain-background'), 5)

        # Plot particle concentration field
        arr = np.where(self.field > 0, self.field, np.nan)
        p = self.ax.contourf(
            self.rlon,
            self.rlat,
            arr,
            locator=ticker.LogLocator(),
            levels=contour_levels,
            extend='both',
            transform=self.proj_data,
        )
        self.fig.colorbar(p)

        self.add_data_domain_outline()

    def add_data_domain_outline(self):
        """Add domain outlines to plot."""

        lon0, lon1 = self.rlon[[0, -1]]
        lat0, lat1 = self.rlat[[0, -1]]
        xs = [lon0, lon1, lon1, lon0, lon0]
        ys = [lat0, lat0, lat1, lat1, lat0]

        self.ax.plot(xs, ys, transform=self.proj_data, c='black', lw=1)

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
        self.fig.savefig(file_path, bbox_inches='tight')
        plt.close(self.fig)

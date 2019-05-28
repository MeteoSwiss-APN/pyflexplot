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
            data (FlexData): All data (grid, fields, attributes) required
                for the plot, read from a FLEXPART output file.

        file_path_fmt (str): Format string to create output file path.
            Must contain all necessary format keys in case ``data``
            contains enough data (e.g., multiple fields or levels)
            to create multiple plots.

        """
        self.data = data
        self.file_path_fmt = file_path_fmt

        if self.type_ == 'concentration':
            self._run_concentration()
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
        _s = 's' if len(keys) > 1 else ''
        print(f"create {len(keys)} concentration plot{_s}")
        for i_key, key in enumerate(keys):
            file_path = self.format_file_path(key)
            print(f"({i_key+1}/{len(keys)}) {file_path}")

            kwargs = {
                'rlat': self.data.rlat,
                'rlon': self.data.rlon,
                'field': self.data.field(key),
                'attrs': {},  #SRU_TMP
            }

            FlexPlotConcentration(**kwargs).save(file_path)

    def check_file_path_fmt(self, restrictions):
        """Check output file path for necessary format keys.

        Args:
            restrictions (dict): Restrictions to field keys.

        """

        def check_file_path_fmt_core(name):
            values = set(getattr(self.data, f'{name}s')(**restrictions))
            fmt_key = f'{{{name}}}'
            if (len(values) > 1 and fmt_key not in self.file_path_fmt):
                raise Exception(
                    f"output file path '{self.file_path_fmt}' must contain"
                    f" format key '{fmt_key}' to plot {len(values)} different"
                    f" {name.replace('_', ' ')}s {sorted(values)}")

        check_file_path_fmt_core('age_ind')
        check_file_path_fmt_core('relpt_ind')
        check_file_path_fmt_core('time_ind')
        check_file_path_fmt_core('level_ind')
        check_file_path_fmt_core('species_id')
        check_file_path_fmt_core('field_type')

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

        self._create()

    def _create(self):
        """Create the plot."""

        #SR_TMP<
        pollat = 43.0
        pollon = -170.0
        bbox = [0, 15, 40, 50]
        #SR_TMP>

        grid_rot = ccrs.RotatedPole(
            pole_latitude=pollat,
            pole_longitude=pollon,
        )
        #grid_std = ccrs.Geodetic()
        grid_std = ccrs.EuroPP()

        self.fig = plt.figure()
        self.ax = plt.subplot(projection=grid_std)

        #self.ax.coastlines()
        self.ax.gridlines()

        #self.ax.stock_img()
        self.ax.add_image(cimgt.Stamen('terrain-background'), 5)

        #self.ax.set_global()
        self.ax.set_extent(bbox)

        arr = np.where(self.field > 0, self.field, np.nan)

        p = self.ax.contourf(
            self.rlon,
            self.rlat,
            arr,
            locator=ticker.LogLocator(),
            levels=10**np.arange(5, 15.1),
            extend='both',
            transform=grid_rot,
        )

        self.fig.colorbar(p)

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

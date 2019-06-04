# -*- coding: utf-8 -*-
"""
Plots.
"""
import cartopy
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
                'attrs': {},  #SRU_TMP
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


class FlexPlotConcentration:
    """FLEXPART plot of particle concentration at a certain level.

    Args:
        rlat (ndarray[float]): Rotated latitude (1d).

        rlon (ndarray[float]): Rotated longitude (1d).

        fld (ndarray[float, float]): Concentration field (2d).

        attrs (dict): Attributes from the FLEXPART NetCDF file
            (gloabl, variable-specific, etc.).

        conf (dict, optional): Plot configuration. Defaults to None.

    """

    def __init__(self, rlat, rlon, fld, attrs, conf=None):

        self.rlat = rlat
        self.rlon = rlon
        self.fld = np.where(fld > 0, fld, np.nan)
        self.attrs = attrs
        self.conf = {} if conf is None else conf

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
        contour_levels = 10**np.arange(-1, 9.1, 1)
        #SRU_TMP>

        self.prepare_projections()

        # Initialize plot
        self.init_fig_axs()

        self.set_extent()
        gls = self.ax.gridlines(linestyle=':')
        self.add_geography('50m')
        #self.add_geography('10m')

        #ipython(globals(), locals())

        # Plot particle concentration field
        p = self.ax.contourf(
            self.rlon,
            self.rlat,
            self.fld,
            locator=ticker.LogLocator(),
            levels=contour_levels,
            extend='both',
            transform=self.proj_data,
        )
        #self.fig.colorbar(p)

        self.add_data_domain_outline()

    def prepare_projections(self):
        """Prepare projections to transform the data for plotting."""

        # Projection of input data: Rotated Pole
        pollat = self.attrs['rotated_pole']['grid_north_pole_latitude']
        pollon = self.attrs['rotated_pole']['grid_north_pole_longitude']
        self.proj_data = cartopy.crs.RotatedPole(
            pole_latitude=pollat,
            pole_longitude=pollon,
        )

        # Projection of plot
        clon = 180 + pollon
        #self.proj_plot = cartopy.crs.EuroPP()
        self.proj_plot = cartopy.crs.TransverseMercator(central_longitude=clon)

        # Geographical lat/lon arrays
        self.proj_geo = cartopy.crs.PlateCarree()
        rlat2d, rlon2d = np.meshgrid(self.rlat, self.rlon)
        self.lon2d, self.lat2d, _ = self.proj_geo.transform_points(
            self.proj_data, rlat2d, rlon2d).T

    def init_fig_axs(self):
        """Initialize figure and axes."""

        self.fig = plt.figure(
            #constrained_layout=True,
        )

        gs = self.fig.add_gridspec(
            ncols=2,
            nrows=3,
            width_ratios=(8, 2),
            height_ratios=(2, 4, 4),
        )

        self.axs = np.array([
            self.fig.add_subplot(gs[1:, 0], projection=self.proj_plot),
            self.fig.add_subplot(gs[0, :]),  # Top box
            self.fig.add_subplot(gs[1, 1]),  # Middle-right box
            self.fig.add_subplot(gs[2, 1]),  # Bottom-right box
        ])
        self.ax = self.axs[0]

        for ax in self.axs[1:]:
            ax.axis('off')

        self.axs[1].text(0.5, 0.5, 'top')
        self.axs[2].text(0.5, 0.5, 'middle-right')
        self.axs[3].text(0.5, 0.5, 'bottom-right')

    def set_extent(self):
        """Set the extent of the plot (bounding box)."""

        # Default: data domain
        bbox = [self.rlon[0], self.rlon[-1], self.rlat[0], self.rlat[-1]]

        # Get padding factor -- either a single number, or a (x, y) tuple
        bbox_pad_rel = self.conf.get('bbox_pad_rel', 0.01)
        try:
            pad_fact_x, pad_fact_y = bbox_pad_rel
        except TypeError:
            pad_fact_x, pad_fact_y = [bbox_pad_rel]*2

        # Add padding: grow (or shrink) bbox by a factor
        dlon = bbox[1] - bbox[0]
        dlat = bbox[3] - bbox[2]
        padx = dlon*pad_fact_x
        pady = dlon*pad_fact_y
        bbox_pad = np.array([-padx, padx, -pady, pady])
        bbox += bbox_pad

        # Apply to plot
        self.ax.set_extent(bbox, self.proj_data)

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
            ))

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
        self.fig.savefig(file_path)
        plt.close(self.fig)

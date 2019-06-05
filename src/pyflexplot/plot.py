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

mpl.use('Agg')  # Prevent ``couldn't connect to display`` error


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

        #SRU_TMP< TODO Extract from NetCDF file
        self.attrs['rotated_pole'] = {
            'grid_north_pole_latitude': 43.0,
            'grid_north_pole_longitude': -170.0,
        }
        #SRU_TMP>

        # Prepare plot
        self.fig = plt.figure(figsize=(12, 9))
        pollat = self.attrs['rotated_pole']['grid_north_pole_latitude']
        pollon = self.attrs['rotated_pole']['grid_north_pole_longitude']
        self.ax_map = FlexAxesMapRotatedPole(
            self.fig, self.rlat, self.rlon, pollat, pollon)

        # Plot particle concentration field
        self.map_add_particle_concentrations()

        # Add text boxes around plot
        self.fig_add_text_boxes()
        self.fill_text_boxes()

    def map_add_particle_concentrations(self):
        """Plot the particle concentrations onto the map."""

        #SRU_TMP>
        contour_levels = 10**np.arange(-1, 9.1, 1)
        #SRU_TMP>

        p = self.ax_map.plot_contourf(
            self.fld,
            locator=ticker.LogLocator(),
            levels=contour_levels,
            extend='both',
        )
        #self.fig.colorbar(p)

        return p

    def fig_add_text_boxes(self, h_rel=0.1, w_rel=0.25, pad_hor_rel=0.015):
        """Add empty text boxes to the figure around the map plot.

        Args:
            h_rel (float, optional): Height of top box as a fraction of
                the height of the map plot. Defaults to <TODO>.

            w_rel (float, optional): Width of the right boxes as a
                fraction of the width of the map plot. Default to <TODO>.

            pad_hor_rel (float, optional): Padding between map plot and
                the text boxes as a fraction of the map plot width. The
                same absolute padding is used in the horizontal and
                vertical direction. Defaults to <TODO>.

        """
        self.ax_ref = self.ax_map.ax  #SRU_TMP

        # Freeze the map plot in order to fix it's coordinates (bbox)
        self.fig.canvas.draw()

        # Obtain aspect ratio of figure
        fig_pxs = self.fig.get_window_extent()
        fig_aspect = fig_pxs.width/fig_pxs.height

        # Get map dimensions in figure coordinates
        w_map, h_map = ax_dims_fig_coords(self.fig, self.ax_ref)

        # Relocate the map close to the lower left corner
        x0_map, y0_map = 0.05, 0.05
        self.ax_ref.set_position([x0_map, y0_map, w_map, h_map])

        # Determine height of top box and width of right boxes
        w_box = w_rel*w_map
        h_box = h_rel*h_map

        # Determine padding between plot and boxes
        pad_hor = pad_hor_rel*w_map
        pad_ver = pad_hor*fig_aspect

        # Add axes for text boxes (one on top, two to the right)
        self.axs_box = np.array([
            FlexAxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map,
                    y0_map + pad_ver + h_map,
                    w_map + pad_hor + w_box,
                    h_box,
                ]),
            FlexAxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map + pad_hor + w_map,
                    y0_map + pad_ver/2 + h_map/2,
                    w_box,
                    h_map/2 - pad_ver/2,
                ]),
            FlexAxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map + pad_hor + w_map,
                    y0_map,
                    w_box,
                    h_map/2 - pad_ver/2,
                ]),
        ])

    def fill_text_boxes(self):
        """Add text etc. to the boxes around the map plot."""
        self.fill_box_top()
        self.fill_box_top_right()
        self.fill_box_bottom_right()

    def fill_box_top(self):
        """Fill the box above the map plot."""
        box = self.axs_box[0]

        box.add_sample_labels()  #SRU_TMP

    def fill_box_top_right(self):
        """Fill the box to the top-right of the map plot."""
        box = self.axs_box[1]

        box.add_sample_labels()  #SRU_TMP

    def fill_box_bottom_right(self):
        """Fill the box to the bottom-right of the map plot."""
        box = self.axs_box[2]

        box.add_sample_labels()  #SRU_TMP

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


class FlexAxesMapRotatedPole():
    """Map plot axes for FLEXPART plot for rotated-pole data.

    Args:
        fig (Figure): Figure to which to map axes is added.

        rlat (ndarray[float]): Rotated latitude coordinates.

        rlon (ndarray[float]): Rotated longitude coordinates.

        pollat (float): Latitude of rotated pole.

        pollon (float): Longitude of rotated pole.

        **conf: Various plot configuration parameters.

    """

    def __init__(self, fig, rlat, rlon, pollat, pollon, **conf):
        self.fig = fig
        self.rlat = rlat
        self.rlon = rlon
        self.conf = conf

        # Determine zorder of unique plot elements, from low to high
        zorders_const = [
            'map',
            'grid',
            'fld',
        ]
        d0, dz = 1, 1
        self.zorder = {e: d0 + i*dz for i, e in enumerate(zorders_const)}

        self.prepare_projections(pollat, pollon)

        self.ax = self.fig.add_subplot(projection=self.proj_plot)

        self.ax.set_extent(
            self.padded_bbox(pad_rel=0.01),
            self.proj_data,
        )

        self.ax.gridlines(
            linestyle=':',
            linewidth=1,
            color='black',
            zorder=self.zorder['grid'],
        )

        self.add_geography('50m')
        #self.add_geography('10m')

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

    def padded_bbox(self, pad_rel=0.0):
        """Compute the bounding box based on rlat/rlon with padding.

        Args:
            pad_rel (float, optional): Padding between the bounding box
                of the data and that of the plot, specified as a
                fraction of the extent of the bounding box of the data
                in the respective direction (horizontal or vertical).
                Can be negative. Defaults to 0.0.

        """
        # Default: data domain
        bbox = [self.rlon[0], self.rlon[-1], self.rlat[0], self.rlat[-1]]

        # Get padding factor -- either a single number, or a (x, y) tuple
        bbox_pad_rel = self.conf.get('bbox_pad_rel', pad_rel)
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

        return bbox

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

        lon0, lon1 = self.rlon[[0, -1]]
        lat0, lat1 = self.rlat[[0, -1]]
        xs = [lon0, lon1, lon1, lon0, lon0]
        ys = [lat0, lat0, lat1, lat1, lat0]

        self.ax.plot(xs, ys, transform=self.proj_data, c='black', lw=1)

    def plot_contourf(self, fld, **kwargs):
        """Plot a color contour field on the map.

        Args:
            fld (ndarray[float, float]): Field to plot.

            **kwargs: Arguments passed to ax.contourf().

        Returns:
            Plot handle.

        """
        p = self.ax.contourf(
            self.rlon,
            self.rlat,
            fld,
            transform=self.proj_data,
            zorder=self.zorder['fld'],
            **kwargs,
        )

        return p


class FlexAxesTextBox:
    """Text box axes for FLEXPART plot.

    Args:
        fig (Figure): Figure to which to add the text box axes.

        ax_ref (Axis): Reference axes.

        rect (list): Rectangle [left, bottom, width, height].

    """

    def __init__(self, fig, ax_ref, rect):

        self.fig = fig
        self.ax_ref = ax_ref

        self.ax = self.fig.add_axes(rect)
        self.ax.axis('off')

        self.draw_box()

        self.compute_unit_distances()

    def draw_box(self, x=0.0, y=0.0, w=1.0, h=1.0, fc='white', ec='black'):
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

    def add_sample_labels(self):
        """Add sample text labels in corners etc."""
        dx, dy = self.dx, self.dy

        def txt(x, y, **kwargs):
            self.ax.text(x, y, transform=self.ax.transAxes, **kwargs)

        # yapf: disable
        txt(0.0 + dx, 0.0 + dy, ha='left',   va='bottom', s='bot left'  )
        txt(0.0 + dx, 0.5,      ha='left',   va='center', s='mid left'  )
        txt(0.0 + dx, 1.0 - dy, ha='left',   va='top',    s='top left'  )
        txt(0.5,      0.0 + dy, ha='center', va='bottom', s='bot center')
        txt(0.5,      0.5,      ha='center', va='center', s='mid center')
        txt(0.5,      1.0 - dy, ha='center', va='top',    s='top center')
        txt(1.0 - dx, 0.0 + dy, ha='right',  va='bottom', s='bot right' )
        txt(1.0 - dx, 0.5,      ha='right',  va='center', s='mid right' )
        txt(1.0 - dx, 1.0 - dy, va='top',    ha='right',  s='top right' )
        # yapf: enable


def ax_dims_fig_coords(fig, ax):
    """Get the dimensions of an axes in figure coords."""
    trans = fig.transFigure.inverted()
    x, y, w, h = ax.bbox.transformed(trans).bounds
    return w, h

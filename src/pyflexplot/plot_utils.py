# -*- coding: utf-8 -*-
"""
Plots.
"""
import cartopy
import geopy.distance
import logging as log
import matplotlib as mpl
import matplotlib.patches
import matplotlib.ticker
import numpy as np

from copy import copy

from .utils import MaxIterationError
from .utils import SummarizableClass
from .utils_dev import ipython  #SR_DEV

#======================================================================
# Summarize Plot-Related Classes
#======================================================================


class SummarizablePlotClass(SummarizableClass):
    """Extend SummarizableClass with plotting-related functionality."""

    def summarize(self, *, add=None, skip=None):
        data = super().summarize(add=add, skip=skip)
        cls_fig = mpl.figure.Figure
        cls_axs = mpl.axes.Axes
        cls_bbox = (mpl.transforms.Bbox, mpl.transforms.TransformedBbox)
        for key, val in data.items():
            if isinstance(val, cls_fig):
                data[key] = summarize_mpl_figure(val)
            elif isinstance(val, cls_axs):
                data[key] = summarize_mpl_axes(val)
            elif isinstance(val, cls_bbox):
                data[key] = summarize_mpl_bbox(val)
        return data


def summarize_mpl_figure(obj):
    """Summarize a matplotlib ``Figure`` instance in a dict."""
    data = {
        'type': type(obj).__name__,
        #SR_TODO if necessary, add option for shallow summary (avoid loops)
        'axes': [summarize_mpl_axes(a) for a in obj.get_axes()],
        'bbox': summarize_mpl_bbox(obj.bbox),
        'dpi': obj.dpi,
    }
    return data


def summarize_mpl_axes(obj):
    """Summarize a matplotlib ``Axes`` instance in a dict."""
    data = {
        'type': type(obj).__name__,
        'bbox': summarize_mpl_bbox(obj.bbox),
    }
    return data


def summarize_mpl_bbox(obj):
    """Summarize a matplotlib ``Bbox`` instance in a dict."""
    data = {
        'type': type(obj).__name__,
        'bounds': obj.bounds,
    }
    return data


#======================================================================
# Map Plot Axes
#======================================================================


class AxesConfMap(SummarizableClass):

    summarizable_attrs = [
        'bbox_pad_rel', 'geogr_res', 'ref_dist', 'ref_dist_unit',
        'ref_dist_dir', 'ref_dist_x0', 'ref_dist_y0'
    ]

    def __init__(
            self,
            *,
            bbox_pad_rel=0.01,
            geogr_res='50m',
            ref_dist=100,
            ref_dist_unit='km',
            ref_dist_dir='east',
            ref_dist_x0=0.05,
            ref_dist_y0=0.95):
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
class AxesMap(SummarizablePlotClass):
    """Map plot axes for FLEXPART plot for rotated-pole data."""

    summarizable_attrs = []  #SR_TMP

    def __init__(self, fig, rlat, rlon, pollat, pollon, **conf):
        """Initialize instance of AxesMap.

        Args:
            fig (Figure): Figure to which to map axes is added.

            rlat (ndarray[float]): Rotated latitude coordinates.

            rlon (ndarray[float]): Rotated longitude coordinates.

            pollat (float): Latitude of rotated pole.

            pollon (float): Longitude of rotated pole.

            **conf: Keyword arguments to create a configuration object
                of type ``AxesConfMap``.

        """
        self.fig = fig
        self.rlat = rlat
        self.rlon = rlon
        self.conf = AxesConfMap(**conf)

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

    def contour(self, fld, **kwargs):
        """Plot a contour field on the map.

        Args:
            fld (ndarray[float, float]): Field to plot.

            **kwargs: Arguments passed to ax.contourf().

        Returns:
            Plot handle.

        """
        if np.isnan(fld).all():
            log.warning('skip contour plot (all-nan field)')
            return

        handle = self.ax.contour(
            self.rlon,
            self.rlat,
            fld,
            transform=self.proj_data,
            zorder=self.zorder['fld'],
            **kwargs,
        )
        return handle

    def contourf(self, fld, **kwargs):
        """Plot a filled-contour field on the map.

        Args:
            fld (ndarray[float, float]): Field to plot.

            **kwargs: Arguments passed to ax.contourf().

        Returns:
            Plot handle.

        """
        if np.isnan(fld).all():
            log.warning('skip filled contour plot (all-nan field)')
            return

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
        if np.isnan(fld).all():
            log.warning('skip maximum marker (all-nan field)')
            return
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

        The configuration is obtained from an ``AxesConfMap``
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


#======================================================================
# Text Box Elements
#======================================================================


class TextBoxElement(SummarizablePlotClass):
    """Base class for elements in text box."""

    def __init__(self, *args, **kwargs):
        raise Exception(f"{type(self).__name__} must be subclassed")


class TextBoxElement_Text(TextBoxElement):
    """Text element in text box."""

    summarizable_attrs = [
        'loc', 's', 'replace_edge_spaces', 'edge_spaces_replacement_char',
        'kwargs'
    ]

    def __init__(
            self,
            box,
            loc,
            *,
            s,
            replace_edge_spaces=False,
            edge_spaces_replacement_char=u'\u2423',
            **kwargs):
        """Create an instance of ``TextBoxElement_Text``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (BoxLocation): Location in parent text box.

            s (str): Text.

            replace_edge_spaces (bool): Replace the first and/and last
                character in ``s`` by ``edge_spaces_replacement_char``
                if it is a space. This can be useful for debugging, as
                trailing spaces may be dropped during rendering.
                Defaults to False.

            edge_spaces_replacement_char (str): Replacement character
                for space if ``edge_spaces_replacement_char == True``.
                Defaults to u'\u2423' (the 'open-box' character below
                the text baseline that commonly represents a space).

            **kwargs: Additional keyword arguments for ``ax.text``.

        """
        self.box = box
        self.loc = loc
        self.s = s
        self.replace_edge_spaces = replace_edge_spaces
        self.edge_spaces_replacement_char = edge_spaces_replacement_char
        self.kwargs = kwargs

        #SR_TMP< TODO consider removing this
        # Add alignment parameters, unless specified in input kwargs
        self.kwargs['ha'] = self.kwargs.get(
            'horizontalalignment', self.kwargs.get('ha', self.loc.ha))
        self.kwargs['va'] = self.kwargs.get(
            'verticalalignment', self.kwargs.get('va', self.loc.va))
        #SR_TMP>

        #SR_TMP<
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
        #SR_TMP>

    def draw(self):
        """Draw text element onto text bot axes."""
        s = self.s

        if self.replace_edge_spaces:
            # Preserve trailing whitespace by replacing the first and
            # last space by a visible character
            if self.edge_spaces_replacement_char == ' ':
                raise Exception("edge_spaces_replacement_char == ' '")
            if self.s[0] == ' ':
                s = self.edge_spaces_replacement_char + s[1:]
            if self.s[-1] == ' ':
                s = s[:-1] + self.edge_spaces_replacement_char

        self.box.ax.text(x=self.loc.x, y=self.loc.y, s=s, **self.kwargs)


class TextBoxElement_ColorRect(TextBoxElement):
    """A colored box element inside a text box axes."""

    summarizable_attrs = ['loc', 'w', 'h', 'fc', 'ec', 'x_anker', 'kwargs']

    def __init__(self, box, loc, *, w, h, fc, ec, x_anker=None, **kwargs):
        """Create an instance of ``TextBoxElement_BolorBox``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (BoxLocation): Location in parent text box.

            w (float): Width (box coordinates).

            h (float): Height (box coordinates).

            fc (str or typle[float]): Face color.

            ec (str or typle[float]): Edge color.

            x_anker (str): Horizontal anker. Options: 'l' or 'left';
                'c' or 'center'; 'r' or 'right'; and None, in which
                case it is derived from the horizontal location in
                ``loc``. Defaults to None.

            **kwargs: Additional keyword arguments for
                ``mpl.patches.Rectangle``.

        """
        self.box = box
        self.loc = loc
        self.w = w
        self.h = h
        self.fc = fc
        self.ec = ec
        self.x_anker = x_anker
        self.kwargs = kwargs

    def draw(self):
        x = self.loc.x
        y = self.loc.y
        w = self.w*self.loc.dx0
        h = self.h*self.loc.dy0

        # Adjust horizontal position
        if self.x_anker in ['l', 'left']:
            pass
        elif self.x_anker in ['c', 'center']:
            x -= 0.5*w
        elif self.x_anker in ['r', 'right']:
            x -= w
        elif self.x_anker is None:
            x -= w*{'l': 0.0, 'c': 0.5, 'r': 1.0}[self.loc.loc_x]
        else:
            raise Exception(f"invalid x_anker '{self.x_anker}'")

        p = mpl.patches.Rectangle(
            xy=(x, y),
            width=w,
            height=h,
            fill=True,
            fc=self.fc,
            ec=self.ec,
            **self.kwargs,
        )
        self.box.ax.add_patch(p)


class TextBoxElement_Marker(TextBoxElement):
    """A marker element in a text box axes."""

    summarizable_attrs = ['loc', 'm', 'kwargs']

    def __init__(self, box, loc, *, m, **kwargs):
        """Create an instance of ``TextBoxElement_Marker``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (BoxLocation): Position in parent text box.

            m (str or int): Marker type.

            **kwargs: Additional keyword arguments for ``ax.plot``.

        """
        self.box = box
        self.loc = loc
        self.m = m
        self.kwargs = kwargs

    def draw(self):
        self.box.ax.plot([self.loc.x], [self.loc.y],
                         marker=self.m,
                         **self.kwargs)


class TextBoxElement_HLine(TextBoxElement):
    """Horizontal line in a text box axes."""

    summarizable_attrs = ['loc', 'c', 'lw']

    def __init__(self, box, loc, *, c='k', lw=1.0):
        """Create an instance of ``TextBoxElement_HLine``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (BoxLocation): Location in parent text box.

            c (<color>, optional): Line color. Defaults to 'k' (black).

            lw (float, optional): Line width. Defaults to 1.0.

        """
        self.box = box
        self.loc = loc
        self.c = c
        self.lw = lw

    def draw(self):
        self.box.ax.axhline(self.loc.y, color=self.c, linewidth=self.lw)


#======================================================================
# Text Boxes
#======================================================================


class TextBoxAxes(SummarizablePlotClass):
    """Text box axes for FLEXPART plot."""

    # Show text base line (useful for debugging)
    _show_baselines = False

    def __init__(self, fig, ax_ref, rect, show_border=True):
        """Initialize instance of TextBoxAxes.

        Args:
            fig (Figure): Figure to which to add the text box axes.

            ax_ref (Axis): Reference axes.

            rect (list): Rectangle [left, bottom, width, height].

            show_border (bool, optional): Show the border of the box.
                Default to True.

        """
        self.fig = fig
        self.ax_ref = ax_ref
        self.rect = rect
        self.show_border = show_border

        self.elements = []
        self.ax = self.fig.add_axes(self.rect)
        self.ax.axis('off')
        self.compute_unit_distances()

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

        self.dx0 = unit_w_map_rel*w_map_fig/w_box_fig
        self.dy0 = unit_w_map_rel*w_map_fig/h_box_fig

    def draw(self):
        """Draw the defined text boxes onto the plot axes."""

        if self.show_border:
            self.draw_border()

        for element in self.elements:
            element.draw()

    summarizable_attrs = ['rect', 'show_border', 'dx0', 'dy0']

    def summarize(self, *, add=None, skip=None):
        """Summarize the text box to a JSON dict."""
        data = super().summarize(add=add, skip=skip)
        data['elements'] = [e.summarize() for e in self.elements]
        return data

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
        loc = BoxLocation(loc, self.dx0, self.dy0, dx, dy)
        self.elements.append(TextBoxElement_Text(self, loc=loc, s=s, **kwargs))
        if self._show_baselines:
            self.elements.append(TextBoxElement_HLine(self, loc))

    def text_block(self, loc, block, colors=None, **kwargs):
        """Add a text block comprised of multiple lines.

        Args:
            loc (int or str): Reference location. For details see
                ``TextBoxAxes.text``.

            block (list[str]): Text block.

            colors (list[<color>], optional): Line-specific colors.
                Defaults to None. If not None, must have same length
                as ``block``. Omit individual lines with None.

            **kwargs: Positioning and formatting options passed to
                ``TextBoxAxes.text_blocks``.

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
                ``TextBoxAxes.text``.

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

        See ``TextBoxAxes.text_blocks_hfill`` for details.
        """
        self.text_blocks_hfill(loc_y, [block], **kwargs)

    def text_blocks_hfill(self, loc_y, blocks, **kwargs):
        """Add blocks of horizontally-filling lines.

        Lines are split at a tab character ('\t'), with the text before
        the tab left-aligned, and the text after right-aligned.

        Args:
            locy (int or str): Vertical reference location. For details
                see ``TextBoxAxes.text`` (vertical component
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
                ``TextBoxAxes.text_blocks``.
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

            fc (str or typle[float]): Face color.

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
        loc = BoxLocation(loc, self.dx0, self.dy0, dx, dy)
        self.elements.append(
            TextBoxElement_ColorRect(
                self, loc, w=w, h=h, fc=fc, ec=ec, **kwargs))
        if self._show_baselines:
            self.elements.append(TextBoxElement_HLine(self, loc))

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
        loc = BoxLocation(loc, self.dx0, self.dy0, dx, dy)
        self.elements.append(
            TextBoxElement_Marker(self, loc, m=marker, **kwargs))
        if self._show_baselines:
            self.elements.append(TextBoxElement_HLine(self, loc))


class BoxLocation(SummarizablePlotClass):
    """Represents reference location inside a box on a 3x3 grid."""

    summarizable_attrs = [
        'loc', 'loc_y', 'loc_x', 'dx0', 'dy0', 'dx', 'dy', 'x0', 'y0', 'x',
        'y', 'va', 'ha'
    ]

    def __init__(self, loc, dx0, dy0, dx=None, dy=None):
        """Initialize an instance of BoxLocation.

        Args:
            loc (int or str): Location parameter. Takes one of three
                formats: integer, short string, or long string.

                Choices:

                    int     short   long
                    00      bl      bottom left
                    01      bc      bottom center
                    02      br      bottom right
                    10      ml      middle left
                    11      mc      middle
                    12      mr      middle right
                    20      tl      top left
                    21      tc      top center
                    22      tr      top right

            dx0 (float): Horizontal baseline offset.

            dy0 (float): Vertical baseline offset.

            dx (float, optional): Horizontal offset. Defaults to 0.0.

            dx (float, optional): Vertical offset. Defaults to 0.0.

        """
        self.dx0 = dx0
        self.dy0 = dy0
        self.dx = 0.0 if dx is None else dx
        self.dy = 0.0 if dy is None else dy

        self._determine_loc_components(loc)

    def _determine_loc_components(self, loc):
        """Split and evaluate components of location parameter."""

        loc = str(loc)

        # Split location into vertical and horizontal part
        if len(loc) == 2:
            loc_y, loc_x = loc
        elif loc == 'center':
            loc_y, loc_x = loc, loc
        else:
            loc_y, loc_x = line.split(' ', 1)

        # Evaluate location components
        self.loc_y = self._standardize_loc_y(loc_y)
        self.loc_x = self._standardize_loc_x(loc_x)
        self.loc = f'{self.loc_y}{self.loc_x}'

    def _standardize_loc_y(self, loc):
        """Standardize vertical location component."""
        if loc in (0, '0', 'b', 'bottom'):
            return 'b'
        elif loc in (1, '1', 'm', 'middle'):
            return 'm'
        elif loc in (2, '2', 't', 'top'):
            return 't'
        raise ValueError(f"invalid vertical location component '{loc}'")

    def _standardize_loc_x(self, loc):
        """Standardize horizontal location component."""
        if loc in (0, '0', 'l', 'left'):
            return 'l'
        elif loc in (1, '1', 'c', 'center'):
            return 'c'
        elif loc in (2, '2', 'r', 'right'):
            return 'r'
        raise ValueError(f"invalid horizontal location component '{loc}'")

    @property
    def va(self):
        """Vertical alignment variable."""
        return {
            'b': 'baseline',
            'm': 'center_baseline',
            #'t': 'top_baseline',  # unfortunately nonexistent
            't': 'top',
        }[self.loc_y]

    @property
    def ha(self):
        """Horizontal alignment variable."""
        return {
            'l': 'left',
            'c': 'center',
            'r': 'right',
        }[self.loc_x]

    @property
    def y0(self):
        """Vertical baseline position."""
        return {
            'b': 0.0 + self.dy0,
            'm': 0.5,
            't': 1.0 - self.dy0,
        }[self.loc_y]

    @property
    def x0(self):
        """Horizontal baseline position."""
        return {
            'l': 0.0 + self.dx0,
            'c': 0.5,
            'r': 1.0 - self.dx0,
        }[self.loc_x]

    @property
    def x(self):
        """Horizontal position."""
        return self.x0 + self.dx*self.dx0

    @property
    def y(self):
        """Vertical position."""
        return self.y0 + self.dy*self.dy0


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

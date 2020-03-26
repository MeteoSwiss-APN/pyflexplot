# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Tuple

# Third-party
import cartopy
import geopy.distance
import matplotlib as mpl
import matplotlib.patches
import matplotlib.ticker
import numpy as np

# First-party
from srutils.iter import isiterable

# Local
from .utils import MaxIterationError
from .utils import summarizable


def post_summarize_plot(self, summary):
    """Method to post-process summary dict of plot class.

    Convert elements like figures and axes into human-readable summaries of
    themselves, provided that they have been specified as to be summarized.

    """
    cls_fig = mpl.figure.Figure
    cls_axs = mpl.axes.Axes
    cls_bbox = (mpl.transforms.Bbox, mpl.transforms.TransformedBbox)
    for key, val in summary.items():
        if isinstance(val, cls_fig):
            summary[key] = summarize_mpl_figure(val)
        elif isinstance(val, cls_axs):
            summary[key] = summarize_mpl_axes(val)
        elif isinstance(val, cls_bbox):
            summary[key] = summarize_mpl_bbox(val)
    return summary


def summarize_mpl_figure(obj):
    """Summarize a matplotlib ``Figure`` instance in a dict."""
    summary = {
        "type": type(obj).__name__,
        # SR_TODO if necessary, add option for shallow summary (avoid loops)
        "axes": [summarize_mpl_axes(a) for a in obj.get_axes()],
        "bbox": summarize_mpl_bbox(obj.bbox),
    }
    return summary


def summarize_mpl_axes(obj):
    """Summarize a matplotlib ``Axes`` instance in a dict."""
    summary = {
        "type": type(obj).__name__,
        "bbox": summarize_mpl_bbox(obj.bbox),
    }
    return summary


def summarize_mpl_bbox(obj):
    """Summarize a matplotlib ``Bbox`` instance in a dict."""
    summary = {
        "type": type(obj).__name__,
        "bounds": obj.bounds,
    }
    return summary


@summarizable
@dataclass
class RefDistIndConf:
    """
    Configuration of ``ReferenceDistanceIndicator``.

    Args:
        dist: Reference distance in ``unit``.

        pos: Position of reference distance indicator box (corners of the
            plot). Options: "tl" (top-left), "tr" (top-right), "bl"
            (bottom-left), "br" (bottom-right).

        unit: Unit of reference distance ``val``.

    """

    dist: int = 100
    pos: str = "bl"
    unit: str = "km"


@summarizable
@dataclass
class MapAxesConf:
    """
    Configuration of ``MapAxesPlot``.

    Args:
        geo_res: Resolution of geographic map elements.

        geo_res_cities: Scale for cities shown on map. Defaults to ``geo_res``.

        geo_res_rivers: Scale for rivers shown on map. Defaults to ``geo_res``.

        lang: Language ('en' for English, 'de' for German).

        lw_frame: Line width of frames.

        min_city_pop: Minimum population of cities shown.

        ref_dist_conf: Reference distance indicator setupuration.

        ref_dist_on: Whether to add a reference distance indicator.

        rel_offset: Relative offset in x and y direction as a fraction of the
            respective domain extent.

        zoom_fact: Zoom factor. Use values above/below 1.0 to zoom in/out.

    """

    geo_res: str = "50m"
    geo_res_cities: Optional[str] = None
    geo_res_rivers: Optional[str] = None
    lang: str = "en"
    lw_frame: float = 1.0
    min_city_pop: int = 0
    ref_dist_conf: RefDistIndConf = field(default_factory=RefDistIndConf)
    ref_dist_on: bool = True
    rel_offset: Tuple[float, float] = (0.0, 0.0)
    zoom_fact: float = 1.0

    def __post_init__(self):
        if self.geo_res_cities is None:
            self.geo_res_cities = self.geo_res
        if self.geo_res_rivers is None:
            self.geo_res_rivers = self.geo_res


@summarizable(post_summarize=post_summarize_plot)
class MapAxes:
    """Map plot axes for regular lat/lon data."""

    # water_color = cartopy.feature.COLORS["water"]
    water_color = "lightskyblue"

    def __init__(self, *, fig, lat, lon, conf):
        """Initialize instance of MapAxesRotatedPole.

        Args:
            fig (Figure): Figure to which to map axes is added.

            lat (ndarray[float]): Latitude coordinates.

            lon (ndarray[float]): Longitude coordinates.

            conf (MapAxesConf): Map axes setup.

        """
        self.fig = fig
        self.lat = lat
        self.lon = lon
        self.conf = conf

        # Determine zorder of unique plot elements, from low to high
        zorders_const = [
            "lowest",
            "geo_lower",
            "fld",
            "geo_upper",
            "grid",
            "marker",
            "frames",
        ]
        d0, dz = 1, 1
        self.zorder = {e: d0 + i * dz for i, e in enumerate(zorders_const)}

        # Prepare the map projections (input, plot, geographic)
        self.prepare_projections()

        # Initialize plot
        self.ax = self.fig.add_subplot(projection=self.proj_map)
        self.ax.outline_patch.set_edgecolor("none")

        # Set extent of map
        bbox = (
            MapAxesBoundingBox(
                self, "data", self.lon[0], self.lon[-1], self.lat[0], self.lat[-1],
            )
            .to_axes()
            .zoom(self.conf.zoom_fact, self.conf.rel_offset)
            .to_data()
        )
        self.ax.set_extent(bbox, self.proj_data)

        # Activate grid lines
        gl = self.ax.gridlines(
            linestyle=":", linewidth=1, color="black", zorder=self.zorder["grid"],
        )
        gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180, 180.1, 2))
        gl.ylocator = mpl.ticker.FixedLocator(np.arange(-90, 90.1, 2))

        # Define reference distance indicator
        if not self.conf.ref_dist_on:
            self.ref_dist_box = None
        else:
            self.ref_dist_box = ReferenceDistanceIndicator(
                ax=self.ax,
                axes_to_geo=self._transform_xy_axes_to_geo,
                conf=self.conf.ref_dist_conf,
                zorder=self.zorder["grid"],
            )

        # Add geographical elements (coasts etc.)
        self.add_geography()

        # # Show data domain outline
        # self.add_data_domain_outline()

        # Redraw plot frame on top of all other elements
        self.ax.add_patch(
            mpl.patches.Rectangle(
                xy=(0, 0),
                width=1,
                height=1,
                transform=self.ax.transAxes,
                zorder=self.zorder["frames"],
                facecolor="none",
                edgecolor="black",
                linewidth=self.conf.lw_frame,
                clip_on=False,
            ),
        )

    def __repr__(self):
        return f"{type(self).__name__}(<TODO>)"  # SR_TODO

    def prepare_projections(self):
        """Prepare projections to transform the data for plotting."""

        # Projection of input data
        self.proj_data = cartopy.crs.PlateCarree()

        # Projection of plot
        self.proj_map = cartopy.crs.PlateCarree()  # SR_TMP

        # Geographical projection
        self.proj_geo = cartopy.crs.PlateCarree()

    def add_data_domain_outline(self):
        """Add domain outlines to map plot."""
        lon0, lon1 = self.lon[[0, -1]]
        lat0, lat1 = self.lat[[0, -1]]
        xs = [lon0, lon1, lon1, lon0, lon0]
        ys = [lat0, lat0, lat1, lat1, lat0]
        self.ax.plot(xs, ys, transform=self.proj_data, c="black", lw=1)

    def add_geography(self):
        """Add geographic elements: coasts, countries, colors, ...
        """
        self.ax.coastlines(resolution=self.conf.geo_res)
        self.ax.background_patch.set_facecolor(self.water_color)
        self.add_countries("lowest")
        self.add_lakes("lowest")
        self.add_rivers("lowest")
        self.add_countries("geo_upper")
        self.add_cities()

    def add_countries(self, zorder_key):
        facecolor = "white" if zorder_key == "lowest" else "none"
        linewidth = 1 if zorder_key == "lowest" else 1 / 3
        self.ax.add_feature(
            cartopy.feature.NaturalEarthFeature(
                category="cultural",
                name="admin_0_countries_lakes",
                scale=self.conf.geo_res,
                edgecolor="black",
                facecolor=facecolor,
                linewidth=linewidth,
            ),
            zorder=self.zorder[zorder_key],
        )

    def add_lakes(self, zorder_key):
        self.ax.add_feature(
            cartopy.feature.NaturalEarthFeature(
                category="physical",
                name="lakes",
                scale=self.conf.geo_res,
                edgecolor="none",
                facecolor=self.water_color,
            ),
            zorder=self.zorder[zorder_key],
        )
        self.ax.add_feature(
            cartopy.feature.NaturalEarthFeature(
                category="physical",
                name="lakes_europe",
                scale=self.conf.geo_res,
                edgecolor="none",
                facecolor=self.water_color,
            ),
            zorder=self.zorder[zorder_key],
        )

    def add_rivers(self, zorder_key):
        linewidth = {"lowest": 1, "geo_lower": 1, "geo_upper": 2 / 3}[zorder_key]

        # SR_DBG <
        # Note:
        #  - Bug in Cartopy with recent shapefiles triggers errors (NULL geometry)
        #  - Issue fixed in a branch but pull request still pending
        #       -> Branch: https://github.com/shevawen/cartopy/tree/patch-1
        #       -> PR: https://github.com/SciTools/cartopy/pull/1411
        #  - Fixed it in our fork MeteoSwiss-APN/cartopy
        #       -> Fork fixes setup depencies issue (with  pyproject.toml)
        #  - For now, check validity of rivers geometry objects when adding
        #  - Once it works with the master branch, remove these workarounds
        # SR_DBG >

        major_rivers = cartopy.feature.NaturalEarthFeature(
            category="physical",
            name="rivers_lake_centerlines",
            scale=self.conf.geo_res,
            edgecolor=self.water_color,
            facecolor=(0, 0, 0, 0),
            linewidth=linewidth,
        )
        # SR_DBG < TODO revert once bugfix in Cartopy master
        # self.ax.add_feature(major_rivers, zorder=self.zorder[zorder_key])
        # SR_DBG >

        if self.conf.geo_res_rivers == "10m":
            minor_rivers = cartopy.feature.NaturalEarthFeature(
                category="physical",
                name="rivers_europe",
                scale=self.conf.geo_res,
                edgecolor=self.water_color,
                facecolor=(0, 0, 0, 0),
                linewidth=linewidth,
            )
            # SR_DBG < TODO revert once bugfix in Cartopy master
            # self.ax.add_feature(minor_rivers, zorder=self.zorder[zorder_key])
            # SR_DBG >

        # SR_DBG <<< TODO remove once bugfix in Cartopy master
        try:
            major_rivers.geometries()
        except Exception:
            warnings.warn(
                f"cannot add major rivers due to shapely issue with "
                "'rivers_lake_centerline; pending bugfix: "
                "https://github.com/SciTools/cartopy/pull/1411; workaround: use "
                "https://github.com/shevawen/cartopy/tree/patch-1"
            )
        else:
            warnings.warn(
                f"successfully added major rivers; "
                "TODO: remove workaround and pin minimum Cartopy version!"
            )
            self.ax.add_feature(major_rivers, zorder=self.zorder[zorder_key])
            if self.conf.geo_res_rivers == "10m":
                self.ax.add_feature(minor_rivers, zorder=self.zorder[zorder_key])
        # SR_DBG >

    def add_cities(self):
        """Add major cities, incl. all capitals."""

        def is_visible(city):
            """Check if a point is inside the domain."""

            plon, plat = city.geometry.x, city.geometry.y

            def is_in_box(x, y, x0, x1, y0, y1):
                return x0 <= x <= x1 and y0 <= y <= y1

            # In domain
            plon, plat = self.proj_data.transform_point(
                plon, plat, self.proj_geo, trap=True,
            )
            in_domain = is_in_box(
                plon, plat, self.lon[0], self.lon[-1], self.lat[0], self.lat[-1],
            )

            # Not behind reference distance indicator box
            pxa, pya = self._transform_xy_geo_to_axes(plon, plat)
            rdb = self.ref_dist_box
            behind_rdb = is_in_box(
                pxa, pya, rdb.x0_box, rdb.x1_box, rdb.y0_box, rdb.y1_box,
            )

            return in_domain and not behind_rdb

        def is_of_interest(city):
            """Check if a city fulfils certain importance criteria."""
            is_capital = city.attributes["FEATURECLA"].startswith("Admin-0 capital")
            is_large = city.attributes["GN_POP"] > self.conf.min_city_pop
            return is_capital or is_large

        def get_name(city):
            """Fetch city name in current language, hand-correcting some."""
            name = city.attributes[f"name_{self.conf.lang}"]
            if name.startswith("Freiburg im ") and name.endswith("echtland"):
                name = "Freiburg"
            return name

        # src: https://www.naturalearthdata.com/downloads/50m-cultural-vectors/50m-populated-places/lk  # noqa
        cities = cartopy.io.shapereader.Reader(
            cartopy.io.shapereader.natural_earth(
                category="cultural",
                name="populated_places",
                resolution=self.conf.geo_res_cities,
            )
        ).records()

        for city in cities:
            x, y = city.geometry.x, city.geometry.y
            if is_visible(city) and is_of_interest(city):
                self.marker(
                    x,
                    y,
                    marker="o",
                    color="black",
                    fillstyle="none",
                    markeredgewidth=1,
                    markersize=3,
                    zorder=self.zorder["geo_upper"],
                )
                name = get_name(city)
                self.text(
                    x, y, name, (0.01, 0), va="center", size="small", clip_on=True,
                )

    def contour(self, fld, **kwargs):
        """Plot a contour field on the map.

        Args:
            fld (ndarray[float, float]): Field to plot.

            **kwargs: Arguments passed to ax.contourf().

        Returns:
            Plot handle.

        """
        if np.isnan(fld).all():
            warnings.warn("skip contour plot (all-nan field)")
            return

        handle = self.ax.contour(
            self.lon,
            self.lat,
            fld,
            transform=self.proj_data,
            zorder=self.zorder["fld"],
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
            warnings.warn("skip filled contour plot (all-nan field)")
            return

        handle = self.ax.contourf(
            self.lon,
            self.lat,
            fld,
            transform=self.proj_data,
            zorder=self.zorder["fld"],
            **kwargs,
        )
        return handle

    def marker(self, lon, lat, marker, *, zorder=None, **kwargs):
        """Add a marker at a location in natural coordinates."""
        lon, lat = self.proj_data.transform_point(lon, lat, self.proj_geo, trap=True)
        handle = self.add_marker(lon, lat, marker, zorder=zorder, **kwargs)
        return handle

    def add_marker(self, lon, lat, marker, *, zorder=None, **kwargs):
        """Add a marker at a location in rotated coordinates."""
        if zorder is None:
            zorder = self.zorder["marker"]
        handle = self.ax.plot(
            lon, lat, marker=marker, transform=self.proj_data, zorder=zorder, **kwargs,
        )
        return handle

    def text(self, lon, lat, s, xy_offset_axs=None, *, zorder=None, **kwargs):
        """Add text at a geographical point.

        Args:
            lon (float): Point longitude.

            lat (float): Point latitude.

            s (str): Text string.

            xy_offset_axs (tuple[float], optional): Horizontal and vertical
                offset in Axes coordinates. Defaults to None.

            dy (float, optional): Vertical offset in unit distances. Defaults
                to None.

            zorder (int, optional): Vertical order. Defaults to "geo_lower".

            **kwargs: Additional keyword arguments for ``ax.text``.

        """
        if zorder is None:
            zorder = self.zorder["geo_lower"]
        x, y = self._transform_xy_geo_to_axes(lon, lat)
        if xy_offset_axs is not None:
            x += xy_offset_axs[0]
            y += xy_offset_axs[1]
        handle = self.ax.text(
            x, y, s, zorder=zorder, transform=self.ax.transAxes, **kwargs
        )
        return handle

    def mark_max(self, fld, marker, **kwargs):
        """Mark the location of the field maximum."""
        if np.isnan(fld).all():
            warnings.warn("skip maximum marker (all-nan field)")
            return
        jmax, imax = np.unravel_index(np.nanargmax(fld), fld.shape)
        lon, lat = self.lon[imax], self.lat[jmax]
        handle = self.add_marker(lon, lat, marker, **kwargs)
        return handle

    # SR_TODO create tranformator class

    def _transform_xy_axes_to_geo(self, x, y):
        return transform_xy_axes_to_geo(
            x, y, self.ax.transAxes, self.ax.transData, self.proj_geo, self.proj_map,
        )

    def _transform_xy_geo_to_axes(self, x, y):
        return transform_xy_geo_to_axes(
            x, y, self.proj_map, self.proj_geo, self.ax.transData, self.ax.transAxes,
        )


class MapAxesRotatedPole(MapAxes):
    """Map plot axes for rotated-pole data."""

    def __init__(self, *, pollon, pollat, **kwargs):
        self.pollon = pollon
        self.pollat = pollat
        super().__init__(**kwargs)

    def prepare_projections(self):
        """Prepare projections to transform the data for plotting."""

        # Projection of input data: Rotated Pole
        self.proj_data = cartopy.crs.RotatedPole(
            pole_latitude=self.pollat, pole_longitude=self.pollon
        )

        # Projection of plot
        clon = 180 + self.pollon
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.proj_map = cartopy.crs.TransverseMercator(
                central_longitude=clon, approx=True,
            )

        # Geographical projection
        self.proj_geo = cartopy.crs.PlateCarree()


class ReferenceDistanceIndicator:
    """Reference distance indicator on a map plot."""

    def __init__(self, ax, axes_to_geo, conf, zorder):
        """Create an instance of ``ReferenceDistanceIndicator``.


        """
        self.conf = conf

        # Position in the plot (one of the corners)
        pos_choices = ["tl", "bl", "br", "tr"]
        if conf.pos not in pos_choices:
            s_choices = ", ".join([f"'{p}'" for p in pos_choices])
            raise ValueError(f"invalid position '{conf.pos}' (choices: {s_choices}")
        self.pos_y, self.pos_x = conf.pos

        self.h_box = 0.06
        self.xpad_box = 0.2 * self.h_box
        self.ypad_box = 0.2 * self.h_box

        self._calc_box_y_params()
        self._calc_box_x_params(axes_to_geo)

        self._add_to(ax, zorder)

    @property
    def x_text(self):
        return self.x0_box + 0.5 * self.w_box

    @property
    def y_text(self):
        return self.y1_box - self.ypad_box

    def _add_to(self, ax, zorder):

        # Draw box
        ax.add_patch(
            mpl.patches.Rectangle(
                xy=(self.x0_box, self.y0_box),
                width=self.w_box,
                height=self.h_box,
                transform=ax.transAxes,
                zorder=zorder,
                fill=True,
                facecolor="white",
                edgecolor="black",
                linewidth=1,
            )
        )

        # Draw line
        ax.plot(
            [self.x0_line, self.x1_line],
            [self.y_line] * 2,
            transform=ax.transAxes,
            zorder=zorder,
            linestyle="-",
            linewidth=2.0,
            color="k",
        )

        # Add label
        ax.text(
            x=self.x_text,
            y=self.y_text,
            s=f"{self.conf.dist:g} {self.conf.unit}",
            transform=ax.transAxes,
            zorder=zorder,
            ha="center",
            va="top",
            fontsize="large",
        )

    def _calc_box_y_params(self):
        if self.pos_y == "t":
            self.y1_box = 1.0
            self.y0_box = self.y1_box - self.h_box
        elif self.pos_y == "b":
            self.y0_box = 0.0
            self.y1_box = self.y0_box + self.h_box
        else:
            raise ValueError(f"invalid y-position '{self.pos_y}'")
        self.y_line = self.y0_box + self.ypad_box

    def _calc_box_x_params(self, axes_to_geo):
        if self.pos_x == "l":
            self.x0_box = 0.0
            self.x0_line = self.x0_box + self.xpad_box
            self.x1_line = self._calc_horiz_dist(self.x0_line, "east", axes_to_geo)
        elif self.pos_x == "r":
            self.x1_box = 1.0
            self.x1_line = self.x1_box - self.xpad_box
            self.x0_line = self._calc_horiz_dist(self.x1_line, "west", axes_to_geo)
        else:
            raise ValueError(f"invalid x-position '{self.pos_x}'")

        self.w_box = self.x1_line - self.x0_line + 2 * self.xpad_box

        if self.pos_x == "l":
            self.x1_box = self.x0_box + self.w_box
        elif self.pos_x == "r":
            self.x0_box = self.x1_box - self.w_box

    def _calc_horiz_dist(self, x0, direction, axes_to_geo):
        calc = MapDistanceCalculator(axes_to_geo, self.conf.unit)
        x1, _, _ = calc.run(x0, self.y_line, self.conf.dist, direction)
        return x1


def transform_xy_geo_to_axes(
    x, y, proj_map, proj_geo, transData, transAxes, invalid_ok=True, invalid_warn=True,
):
    """Transform geographic coordinates to axes coordinates."""

    def recurse(xi, yi):
        return transform_xy_geo_to_axes(
            xi, yi, proj_map, proj_geo, transData, transAxes, invalid_ok, invalid_warn,
        )

    if isiterable(x) or isiterable(y):
        check_same_sized_iterables(x, y)
        return tuple(np.array([recurse(xi, yi) for xi, yi in zip(x, y)]).T)

    check_valid_coords((x, y), invalid_ok, invalid_warn)

    # Geo -> Plot
    xy_plt = proj_map.transform_point(x, y, proj_geo, trap=True)
    check_valid_coords(xy_plt, invalid_ok, invalid_warn)

    # Plot -> Display
    xy_dis = transData.transform(xy_plt)
    check_valid_coords(xy_dis, invalid_ok, invalid_warn)

    # Display -> Axes
    xy_axs = transAxes.inverted().transform(xy_dis)
    check_valid_coords(xy_axs, invalid_ok, invalid_warn)

    return xy_axs


def transform_xy_axes_to_geo(
    x, y, transAxes, transData, proj_geo, proj_map, invalid_ok=True, invalid_warn=True,
):
    """Transform axes coordinates to geographic coordinates."""

    def recurse(xi, yi):
        return transform_xy_axes_to_geo(
            xi, yi, transAxes, transData, proj_geo, proj_map, invalid_ok, invalid_warn,
        )

    if isiterable(x) or isiterable(y):
        check_same_sized_iterables(x, y)
        return tuple(np.array([recurse(xi, yi) for xi, yi in zip(x, y)]).T)

    check_valid_coords((x, y), invalid_ok, invalid_warn)

    # Axes -> Display
    xy_dis = transAxes.transform((x, y))
    check_valid_coords(xy_dis, invalid_ok, invalid_warn)

    # Display -> Plot
    x_plt, y_plt = transData.inverted().transform(xy_dis)
    check_valid_coords((x_plt, y_plt), invalid_ok, invalid_warn)

    # Plot -> Geo
    xy_geo = proj_geo.transform_point(x_plt, y_plt, proj_map, trap=True)
    check_valid_coords(xy_geo, invalid_ok, invalid_warn)

    return xy_geo


def check_same_sized_iterables(x, y):
    """Check that x and y are iterables of the same size."""
    if isiterable(x) and not isiterable(y):
        raise ValueError(f"x is iterable but y is not", (x, y))
    if isiterable(y) and not isiterable(x):
        raise ValueError(f"y is iterable but x is not", (x, y))
    if len(x) != len(y):
        raise ValueError(f"x and y differ in length", (len(x), len(y)), (x, y))


def check_valid_coords(xy, allow, warn):
    """Check that xy coordinate is valid."""
    if np.isnan(xy).any() or np.isinf(xy).any():
        if not allow:
            raise ValueError("invalid coordinates", xy)
        elif warn:
            warnings.warn(f"invalid coordinates: {xy}")


class MapDistanceCalculator:
    """Calculate geographic distance along a line on a map plot."""

    def __init__(self, axes_to_geo, unit="km", p=0.001):
        """Initialize an instance of MapDistanceCalculator.

        Args:
            axes_to_geo (callable): Function to transform a point
                (x, y) from axes coordinates to geographic coordinates (x, y
                may be arrays).

            unit (str, optional): Unit of ``dist``. Defaults to 'km'.

            p (float, optional): Required precision as a fraction of ``dist``.
                Defaults to 0.001.

        """
        self.axes_to_geo = axes_to_geo
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

    def _set_dir(self, direction):
        """Check and set the direction."""
        if direction is not None:
            if direction in "east":
                self._dx_unit = 1
                self._dy_unit = 0
            elif direction in "west":
                self._dx_unit = -1
                self._dy_unit = 0
            elif direction in "north":
                self._dx_unit = 0
                self._dy_unit = 1
            elif direction in "south":
                self._dx_unit = 0
                self._dy_unit = -1
            else:
                raise NotImplementedError(f"direction='{direction}'")
        self.direction = direction

    def run(self, x0, y0, dist, dir="east"):
        """Measure geo. distance along a straight line on the plot."""
        self.reset(dist=dist, dir=dir)

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
            err = abs(dist - self.dist) / self.dist
            # Note: `abs` only necessary if `dist` could be `dist[-2]`

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

        # Transform starting point to geographical coordinates
        x0_geo, y0_geo = self.axes_to_geo(x0_ax, y0_ax)

        path_ax = [(x0_ax, y0_ax)]
        dists = [dist0]

        # Step away until target distance exceeded
        dist = 0.0
        x1_ax, y1_ax = x0_ax, y0_ax
        while dist < self.dist - dist0:

            # Move one step farther away from starting point
            x1_ax += self._dx_unit * step_ax_rel
            y1_ax += self._dy_unit * step_ax_rel

            # Transform current point to gegographical coordinates
            x1_geo, y1_geo = self.axes_to_geo(x1_ax, y1_ax)

            # Compute geographical distance from starting point
            dist = self.comp_dist(x0_geo, y0_geo, x1_geo, y1_geo)

            path_ax.append((x1_ax, y1_ax))
            dists.append(dist + dist0)

        return path_ax, dists

    def comp_dist(self, lon0, lat0, lon1, lat1):
        """Compute the great circle distance between two points."""
        dist_obj = geopy.distance.great_circle((lat0, lon0), (lat1, lon1))
        if self.unit == "km":
            return dist_obj.kilometers
        else:
            raise NotImplementedError(f"great circle distance in {self.unit}")


@summarizable(post_summarize=post_summarize_plot)
class TextBoxElement:
    """Base class for elements in text box."""

    def __init__(self, *args, **kwargs):
        raise Exception(f"{type(self).__name__} must be subclassed")


@summarizable(
    attrs=["loc", "s", "replace_edge_spaces", "edge_spaces_replacement_char", "kwargs"],
    overwrite=True,
)
class TextBoxElement_Text(TextBoxElement):
    """Text element in text box."""

    def __init__(
        self,
        box,
        loc,
        *,
        s,
        replace_edge_spaces=False,
        edge_spaces_replacement_char="\u2423",
        **kwargs,
    ):
        """Create an instance of ``TextBoxElement_Text``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (TextBoxLocation): Location in parent text box.

            s (str): Text.

            replace_edge_spaces (bool): Replace the first and/and last
                character in ``s`` by ``edge_spaces_replacement_char`` if it is
                a space. This can be useful for debugging, as trailing spaces
                may be dropped during rendering. Defaults to False.

            edge_spaces_replacement_char (str): Replacement character for space
                if ``edge_spaces_replacement_char == True``. Defaults to
                u'\u2423' (the 'open-box' character below the text baseline
                that commonly represents a space).

            **kwargs: Additional keyword arguments for ``ax.text``.

        """
        self.box = box
        self.loc = loc
        self.s = s
        self.replace_edge_spaces = replace_edge_spaces
        self.edge_spaces_replacement_char = edge_spaces_replacement_char
        self.kwargs = kwargs

        # SR_TMP < TODO consider removing this
        # Add alignment parameters, unless specified in input kwargs
        self.kwargs["ha"] = self.kwargs.get(
            "horizontalalignment", self.kwargs.get("ha", self.loc.ha)
        )
        self.kwargs["va"] = self.kwargs.get(
            "verticalalignment", self.kwargs.get("va", self.loc.va)
        )
        # SR_TMP >

        # SR_TMP <
        if kwargs["va"] == "top_baseline":
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
        # SR_TMP >

    def draw(self):
        """Draw text element onto text bot axes."""
        s = self.s

        if self.replace_edge_spaces:
            # Preserve trailing whitespace by replacing the first and
            # last space by a visible character
            if self.edge_spaces_replacement_char == " ":
                raise Exception("edge_spaces_replacement_char == ' '")
            if self.s[0] == " ":
                s = self.edge_spaces_replacement_char + s[1:]
            if self.s[-1] == " ":
                s = s[:-1] + self.edge_spaces_replacement_char

        self.box.ax.text(x=self.loc.x, y=self.loc.y, s=s, **self.kwargs)


@summarizable(attrs=["loc", "w", "h", "fc", "ec", "x_anker", "kwargs"], overwrite=True)
class TextBoxElement_ColorRect(TextBoxElement):
    """A colored box element inside a text box axes."""

    def __init__(self, box, loc, *, w, h, fc, ec, x_anker=None, **kwargs):
        """Create an instance of ``TextBoxElement_BolorBox``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (TextBoxLocation): Location in parent text box.

            w (float): Width (box coordinates).

            h (float): Height (box coordinates).

            fc (str or typle[float]): Face color.

            ec (str or typle[float]): Edge color.

            x_anker (str): Horizontal anker. Options: 'l' or 'left'; 'c' or
                'center'; 'r' or 'right'; and None, in which case it is derived
                from the horizontal location in ``loc``. Defaults to None.

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
        w = self.w * self.loc.dx_unit
        h = self.h * self.loc.dy_unit

        # Adjust horizontal position
        if self.x_anker in ["l", "left"]:
            pass
        elif self.x_anker in ["c", "center"]:
            x -= 0.5 * w
        elif self.x_anker in ["r", "right"]:
            x -= w
        elif self.x_anker is None:
            x -= w * {"l": 0.0, "c": 0.5, "r": 1.0}[self.loc.loc_x]
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


@summarizable(attrs=["loc", "m", "kwargs"], overwrite=True)
class TextBoxElement_Marker(TextBoxElement):
    """A marker element in a text box axes."""

    def __init__(self, box, loc, *, m, **kwargs):
        """Create an instance of ``TextBoxElement_Marker``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (TextBoxLocation): Position in parent text box.

            m (str or int): Marker type.

            **kwargs: Additional keyword arguments for ``ax.plot``.

        """
        self.box = box
        self.loc = loc
        self.m = m
        self.kwargs = kwargs

    def draw(self):
        self.box.ax.plot([self.loc.x], [self.loc.y], marker=self.m, **self.kwargs)


@summarizable(attrs=["loc", "c", "lw"], overwrite=True)
class TextBoxElement_HLine(TextBoxElement):
    """Horizontal line in a text box axes."""

    def __init__(self, box, loc, *, c="k", lw=1.0):
        """Create an instance of ``TextBoxElement_HLine``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (TextBoxLocation): Location in parent text box.

            c (<color>, optional): Line color. Defaults to 'k' (black).

            lw (float, optional): Line width. Defaults to 1.0.

        """
        self.box = box
        self.loc = loc
        self.c = c
        self.lw = lw

    def draw(self):
        self.box.ax.axhline(self.loc.y, color=self.c, linewidth=self.lw)


@summarizable(
    attrs=["name", "rect", "lw_frame", "dx_unit", "dy_unit"],
    post_summarize=lambda self, summary: {
        **post_summarize_plot(self, summary),
        "elements": [e.summarize() for e in self.elements],
    },
)
class TextBoxAxes:
    """Text box axes for FLEXPART plot."""

    # Show text base line (useful for debugging)
    _show_baselines = False

    def __init__(
        self,
        fig,
        ax_ref,
        rect,
        name=None,
        lw_frame=1.0,
        ec="black",
        fc="none",
        f_pad=None,
    ):
        """Initialize instance of TextBoxAxes.

        Args:
            fig (Figure): Figure to which to add the text box axes.

            ax_ref (Axis): Reference axes.

            rect (list): Rectangle [left, bottom, width, height].

            name (list, optional): Name. Defaults to None.

            lw_frame (float, optional): Line width of frame around box. Frame
                is omitted if ``lw_frame`` is None. Defaults to 1.0.

            ec (str, optional): Edge color. Defaults to "black".

            fc (str, optional): Face color. Defaults to "none".

            f_pad (float or tuple[float, float], optional): Padding of text box
                elements in unit distances, either one value in both directions
                or separate values in the x and y direction. Defaults to 1.0.

        """
        self.fig = fig
        self.ax_ref = ax_ref
        self.rect = rect
        self.name = name
        self.lw_frame = lw_frame
        self.ec = ec
        self.fc = fc

        self.elements = []
        self.ax = self.fig.add_axes(self.rect)
        self.ax.axis("off")
        self._set_unit_distances()
        self._set_pad(1.0 if f_pad is None else f_pad)

    def _set_unit_distances(self, unit_w_map_rel=0.01):
        """Compute unit distances in x and y for text positioning.

        To position text nicely inside a box, it is handy to have unit
        distances of absolute length to work with that are independent of the
        size of the box (i.e., axes). This method computes such distances as a
        fraction of the width of the map plot.

        Args:
            unit_w_map_rel (float, optional): Fraction of the width of the map
                plot that corresponds to one unit distance. Defaults to 0.01.

        """
        w_map_fig, _ = ax_w_h_in_fig_coords(self.fig, self.ax_ref)
        w_box_fig, h_box_fig = ax_w_h_in_fig_coords(self.fig, self.ax)

        self.dx_unit = unit_w_map_rel * w_map_fig / w_box_fig
        self.dy_unit = unit_w_map_rel * w_map_fig / h_box_fig

    def _set_pad(self, f_pad):
        """Set padding in x and y direction."""

        if not isiterable(f_pad, str_ok=False):
            try:
                f_pad_x = float(f_pad)
                f_pad_y = float(f_pad)
            except Exception:
                raise ValueError(f"f_pad is not a float nor a pair of floats: {f_pad}")
        else:
            try:
                f_pad_x, f_pad_y = [float(i) for i in f_pad]
            except Exception:
                raise ValueError(f"f_pad is not a float nor a pair of floats: {f_pad}")

        self.pad_x = f_pad_x * self.dx_unit
        self.pad_y = f_pad_y * self.dy_unit

    def draw(self):
        """Draw the defined text boxes onto the plot axes."""

        # Create box without frame
        p = mpl.patches.Rectangle(
            xy=(0, 0),
            width=1,
            height=1,
            transform=self.ax.transAxes,
            facecolor=self.fc,
            edgecolor="none",
            clip_on=False,
        )
        self.ax.add_patch(p)

        if self.lw_frame and self.lw_frame != "none":
            # Enable frame around box
            p.set(edgecolor=self.ec, linewidth=self.lw_frame)

        for element in self.elements:
            element.draw()

    def text(self, loc, s, dx=0.0, dy=0.0, **kwargs):
        """Add text positioned relative to a reference location.

        Args:
            loc (int or str): Reference location parameter used to initialize
                an instance of ``TextBoxLocation``.

            s (str): Text string.

            dx (float, optional): Horizontal offset in unit distances. May be
                negative. Defaults to 0.0.

            dy (float, optional): Vertical offset in unit distances. May be
                negative. Defaults to 0.0.

            **kwargs: Formatting options passed to ax.text().

        """
        loc = TextBoxLocation(self, loc, dx, dy)
        self.elements.append(TextBoxElement_Text(self, loc=loc, s=s, **kwargs))
        if self._show_baselines:
            self.elements.append(TextBoxElement_HLine(self, loc))

    def text_block(self, loc, block, colors=None, **kwargs):
        """Add a text block comprised of multiple lines.

        Args:
            loc (int or str): Reference location. For details see
                ``TextBoxAxes.text``.

            block (list[str]): Text block.

            colors (list[<color>], optional): Line-specific colors. Defaults to
                None. If not None, must have same length as ``block``. Omit
                individual lines with None.

            **kwargs: Positioning and formatting options passed to
                ``TextBoxAxes.text_blocks``.

        """
        self.text_blocks(loc, [block], colors=[colors], **kwargs)

    def text_blocks(
        self,
        loc,
        blocks,
        *,
        dy_unit=None,
        dy_line=None,
        dy_block=None,
        reverse=False,
        colors=None,
        **kwargs,
    ):
        """Add multiple text blocks.

        Args:
            loc (int or str): Reference location. For details see
                ``TextBoxAxes.text``.

            blocks (list[list[str]]): List of text blocks, each of which
                constitutes a list of lines.

            dy_unit (float, optional): Initial vertical offset in unit distances.
                May be negative. Defaults to ``dy_line``.

            dy_line (float, optional): Incremental vertical offset between
                lines. Can be negative. Defaults to 2.5.

            dy_block (float, optional): Incremental vertical offset between
                blocks of lines. Can be negative. Defaults to ``dy_line``.

            dx (float, optional): Horizontal offset in unit distances. May be
                negative. Defaults to 0.0.

            reverse (bool, optional): If True, revert the block and line order.
                Defaults to False. Note that if line-specific colors are
                passed, they must be in the same order as the unreversed
                blocks.

            colors (list[list[<color>]], optional): Line-specific colors in
                each block. Defaults to None. If not None, must have same shape
                as ``blocks``. Omit individual blocks or lines in blocks with
                None.

            **kwargs: Formatting options passed to ``ax.text``.

        """
        if dy_line is None:
            dy_line = 2.5
        if dy_unit is None:
            dy_unit = dy_line
        if dy_block is None:
            dy_block = dy_line

        default_color = kwargs.pop("color", kwargs.pop("c", "black"))
        colors_blocks = self._prepare_line_colors(blocks, colors, default_color)

        if reverse:
            # Revert order of blocks and lines
            def revert(lsts):
                return [[l for l in lst[::-1]] for lst in lsts[::-1]]

            blocks = revert(blocks)
            colors_blocks = revert(colors_blocks)

        dy = dy_unit
        for i, block in enumerate(blocks):
            for j, line in enumerate(block):
                self.text(loc, s=line, dy=dy, color=colors_blocks[i][j], **kwargs)
                dy += dy_line
            dy += dy_block

    @staticmethod
    def _prepare_line_colors(blocks, colors, default_color):

        if colors is None:
            colors_blocks = [None] * len(blocks)
        elif len(colors) == len(blocks):
            colors_blocks = colors
        else:
            raise ValueError(
                f"different no. colors than blocks: {len(colors)} != {len(blocks)}"
            )

        for i, block in enumerate(blocks):
            if colors_blocks[i] is None:
                colors_blocks[i] = [None] * len(block)
            elif len(colors_blocks) != len(blocks):
                ith = f"{i}{({1: 'st', 2: 'nd', 3: 'rd'}.get(i, 'th'))}"
                raise ValueError(
                    f"colors of {ith} block must have same length as block: "
                    f"{len(colors_blocks[i])} != {len(block)}"
                )
            for j in range(len(block)):
                if colors_blocks[i][j] is None:
                    colors_blocks[i][j] = default_color

        return colors_blocks

    def text_block_hfill(self, loc_y, block, **kwargs):
        """Single block of horizontally filled lines.

        Args:
            loc_y (int or str): Vertical reference location. For details see
                ``TextBoxAxes.text`` (vertical component only).

            block (str or list[str] or ...): Text block. See docstring of
                method ``text_blocks_hfill`` for details.

            **kwargs: Additional keyword arguments passed on to method
                ``text_blocks_hfill``.

        """
        self.text_blocks_hfill(loc_y, [block], **kwargs)

    def text_blocks_hfill(self, loc_y, blocks, **kwargs):
        r"""Add blocks of horizontally-filling lines.

        Lines are split at a tab character ('\t'), with the text before the tab
        left-aligned, and the text after right-aligned.

        Args:
            loc_y (int or str): Vertical reference location. For details see
                ``TextBoxAxes.text`` (vertical component only).

            blocks (str or list[str] or ...): Text blocks, each of which
                consists of lines, each of which in turn consists of a left and
                right part. Possible formats:

                 - The blocks can be a multiline string, with empty lines
                   separating the individual blocks; or a list.

                 - In case of list blocks, each block can in turn constitute a
                   multiline string, or a list of lines.

                 - In case of a list block, each line can in turn constitute a
                   string, or a two-element string tuple.

                 - Lines represented by a string are split into a left and
                   right part at the first tab character ('\t').

            **kwargs: Location and formatting options passed to
                ``TextBoxAxes.text_blocks``.

        """

        if isinstance(blocks, str):
            # Whole blocks is a multiline string
            blocks = blocks.strip().split("\n\n")

        # Handle case where a multiblock string is embedded
        # in a blocks list alongside string or list blocks
        blocks_orig, blocks = blocks, []
        for block in blocks_orig:
            if isinstance(block, str):
                # Possible multiblock string (if with empty line)
                for subblock in block.strip().split("\n\n"):
                    blocks.append(subblock)
            else:
                # List block
                blocks.append(block)

        # Separate left and right parts of lines
        blocks_l, blocks_r = [], []
        for block in blocks:

            if isinstance(block, str):
                # Turn multiline block into list block
                block = block.strip().split("\n")

            blocks_l.append([])
            blocks_r.append([])
            for line in block:

                # Obtain left and right part of line
                if isinstance(line, str):
                    str_l, str_r = line.split("\t", 1)
                elif len(line) == 2:
                    str_l, str_r = line
                else:
                    raise ValueError(f"invalid line: {line}")

                blocks_l[-1].append(str_l)
                blocks_r[-1].append(str_r)

        dx_l = kwargs.pop("dx", None)
        dx_r = None if dx_l is None else -dx_l

        # Add lines to box
        self.text_blocks("bl", blocks_l, dx=dx_l, **kwargs)
        self.text_blocks("br", blocks_r, dx=dx_r, **kwargs)

    def sample_labels(self):
        """Add sample text labels in corners etc."""
        kwargs = dict(fontsize=9)
        self.text("bl", "bot. left", **kwargs)
        self.text("bc", "bot. center", **kwargs)
        self.text("br", "bot. right", **kwargs)
        self.text("ml", "middle left", **kwargs)
        self.text("mc", "middle center", **kwargs)
        self.text("mr", "middle right", **kwargs)
        self.text("tl", "top left", **kwargs)
        self.text("tc", "top center", **kwargs)
        self.text("tr", "top right", **kwargs)

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

    def color_rect(self, loc, fc, ec=None, dx=0.0, dy=0.0, w=3.0, h=2.0, **kwargs):
        """Add a colored rectangle.

        Args:
            loc (int or str): Reference location parameter used to initialize
                an instance of ``TextBoxLocation``.

            fc (str or typle[float]): Face color.

            ec (<color>, optional): Edge color. Defaults to face color.

            dx (float, optional): Horizontal offset in unit distances. May be
                negative. Defaults to 0.0.

            dy (float, optional): Vertical offset in unit distances. May be
                negative. Defaults to 0.0.

            w (float, optional): Width in unit distances. Defaults to 3.0.

            h (float, optional): Height in unit distances. Defaults to 2.0.

            **kwargs: Keyword arguments passed to
                ``matplotlib.patches.Rectangle``.

        """
        if ec is None:
            ec = fc
        loc = TextBoxLocation(self, loc, dx, dy)
        self.elements.append(
            TextBoxElement_ColorRect(self, loc, w=w, h=h, fc=fc, ec=ec, **kwargs)
        )
        if self._show_baselines:
            self.elements.append(TextBoxElement_HLine(self, loc))

    def marker(self, loc, marker, dx=0.0, dy=0.0, **kwargs):
        """Add a marker symbol.

        Args:
            loc (int or str): Reference location parameter used to initialize
                an instance of ``TextBoxLocation``.

            marker (str or int): Marker style passed to ``mpl.plot``. See
                ``matplotlib.markers`` for more information.

            dx (float, optional): Horizontal offset in unit distances. May be
                negative. Defaults to 0.0.

            dy (float, optional): Vertical offset in unit distances. May be
                negative. Defaults to 0.0.

            **kwargs: Keyword arguments passed to ``mpl.plot``.

        """
        loc = TextBoxLocation(self, loc, dx, dy)
        self.elements.append(TextBoxElement_Marker(self, loc, m=marker, **kwargs))
        if self._show_baselines:
            self.elements.append(TextBoxElement_HLine(self, loc))

    def fit_text(self, s, size, **kwargs):
        return TextFitter(self.ax, dx_unit=self.dx_unit, **kwargs).fit(s, size)


class MinFontSizeReachedError(Exception):
    """Font size cannot be reduced further."""


class MinStrLenReachedError(Exception):
    """String cannot be further truncated."""


class TextFitter:
    """Fit a text string into the box by shrinking and/or truncation."""

    sizes = ["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]

    def __init__(self, ax, *, dx_unit, n_shrink_max=None, pad_rel=None, dots=".."):
        """Create an instance of ``TextFitter``.

        Args:
            ax (Axes): Axes.

            dx_unit (float): Horizontal unit distance.

            n_shrink_max (int, optional): Maximum number of times the font size
                can be reduced before the string is truncated. If it is None or
                negative, the font size is reduced all the way to "xx-small"
                if necessary. Defaults to None.

            pad_rel (float, optional): Total horizontal padding as a fraction
                of the box width. Defaults to twice the default horizontal
                offset ``2 * dx_unit``.

            dots (str, optional): String replacing the end of the retained part
                of ``s`` in case it must be truncated. Defaults to "..".

        """
        self.ax = ax
        self.dx_unit = dx_unit

        if n_shrink_max is not None:
            try:
                n_shrink_max = int(n_shrink_max)
            except ValueError:
                raise ValueError(
                    f"n_shrink_max of type {type(n_shrink_max).__name__} not "
                    f"int-compatible: {n_shrink_max}"
                )
            if n_shrink_max < 0:
                n_shrink_max = None
        self.n_shrink_max = n_shrink_max

        if pad_rel is None:
            pad_rel = 2 * self.dx_unit
        self.pad_rel = pad_rel

        self.dots = dots

    def fit(self, s, size):
        """
        Fit a string with a certain target size into the box.

        Args:
            s (str): Text string to fit into the box.

            size (str): Initial font size (e.g., "medium", "x-large").

        """
        if size not in self.sizes:
            raise ValueError(f"unknown font size '{size}'; must be one of {self.sizes}")

        w_rel_max = 1.0 - self.pad_rel
        while len(s) >= len(self.dots) and self.w_rel(s, size) > w_rel_max:
            try:
                size = self.shrink(size)
            except MinFontSizeReachedError:
                try:
                    s = self.truncate(s)
                except MinStrLenReachedError:
                    break

        return s, size

    def w_rel(self, s, size):
        """Returns the width of a string as a fraction of the box width."""

        # Determine width of text in display coordinates
        # src: https://stackoverflow.com/a/36959454
        renderer = self.ax.get_figure().canvas.get_renderer()
        txt = self.ax.text(0, 0, s, size=size)
        w_disp = txt.get_window_extent(renderer=renderer).width

        # Remove the text again from the axes
        self.ax.texts.pop()

        return w_disp / self.ax.bbox.width

    def shrink(self, size, _n=[0]):
        """Shrink the relative font size by one increment."""
        i = self.sizes.index(size)
        if i == 0 or (self.n_shrink_max is not None and _n[0] >= self.n_shrink_max):
            raise MinFontSizeReachedError(size)
        size = self.sizes[i - 1]
        _n[0] += 1
        return size

    def truncate(self, s, _n=[0]):
        """Truncate a string by one character and end with ``self.dots``."""
        if len(s) <= len(self.dots):
            raise MinStrLenReachedError(s)
        _n[0] += 1
        return s[: -(len(self.dots) + 1)] + self.dots


class MapAxesBoundingBox:
    """Bounding box of a ``MapAxes``."""

    def __init__(self, map_axes, coord_type, lon0, lon1, lat0, lat1):
        """Create an instance of ``MapAxesBoundingBox``.

        Args:
            map_axes (MapAxes): Parent map axes object.

            coord_type (str): Coordinates type.

            lon0 (float): Longitude of south-western corner.

            lon1 (float): Longitude of north-eastern corner.

            lat0 (float): Latitude of south-western corner.

            lat1 (float): Latitude of north-eastern corner.

        """
        # SR_TMP <
        self.coord_types = ["data", "geo"]
        # SR_TMP >
        self.map_axes = map_axes
        self.set(coord_type, lon0, lon1, lat0, lat1)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f'map_axes={self.map_axes}, coord_type="{self.coord_type}", '
            f"lon0={self.lon0:.2f}, lon1={self.lon1:.2f}, "
            f"lat0={self.lat0:.2f}, lat1={self.lat1:.2f})"
        )

    @property
    def coord_type(self):
        return self._curr_coord_type

    @property
    def lon0(self):
        return self._curr_lon0

    @property
    def lon1(self):
        return self._curr_lon1

    @property
    def lat0(self):
        return self._curr_lat0

    @property
    def lat1(self):
        return self._curr_lat1

    def set(self, coord_type, lon0, lon1, lat0, lat1):
        assert not any(np.isnan(c) for c in (lon0, lon1, lat0, lat1))
        self._curr_coord_type = coord_type
        self._curr_lon0 = lon0
        self._curr_lon1 = lon1
        self._curr_lat0 = lat0
        self._curr_lat1 = lat1

    def __iter__(self):
        """Iterate over the rotated corner coordinates."""
        yield self._curr_lon0
        yield self._curr_lon1
        yield self._curr_lat0
        yield self._curr_lat1

    def __len__(self):
        return len(list(iter(self)))

    def __getitem__(self, idx):
        return list(iter(self))[idx]

    def to_data(self):
        if self.coord_type == "geo":
            coords = np.concatenate(
                self._proj_data.transform_points(self._proj_geo, self.lon, self.lat)[
                    :, :2
                ].T
            )
        elif self.coord_type == "axes":
            return self.to_geo().to_data()
        else:
            self._error("to_data")
        self.set("data", *coords)
        return self

    def to_geo(self):
        if self.coord_type == "data":
            coords = np.concatenate(
                self._proj_geo.transform_points(self._proj_data, self.lon, self.lat)[
                    :, :2
                ].T
            )
        elif self.coord_type == "axes":
            coords = np.concatenate(
                transform_xy_axes_to_geo(
                    self.lon,
                    self.lat,
                    self._transAxes,
                    self._transData,
                    self._proj_geo,
                    self._proj_map,
                    invalid_ok=False,
                )
            )
        else:
            self._error("to_geo")
        self.set("geo", *coords)
        return self

    def to_axes(self):
        if self.coord_type == "geo":
            coords = np.concatenate(
                transform_xy_geo_to_axes(
                    self.lon,
                    self.lat,
                    self._proj_map,
                    self._proj_geo,
                    self._transData,
                    self._transAxes,
                    invalid_ok=False,
                )
            )
        elif self.coord_type == "data":
            return self.to_geo().to_axes()
        else:
            self._error("to_axes")
        self.set("axes", *coords)
        return self

    def _error(self, method):
        raise NotImplementedError(
            f"{type(self).__name__}.{method} from '{self.coord_type}'"
        )

    def zoom(self, fact, rel_offset):
        """Zoom into or out of the domain.

        Args:
            fact (float): Zoom factor, > 1.0 to zoom in, < 1.0 to zoom out.

            rel_offset (tuple[float, float], optional): Relative offset in x
                and y direction as a fraction of the respective domain extent.
                Defaults to (0.0, 0.0).

        Returns:
            ndarray[float, n=4]: Zoomed bounding box.

        """
        try:
            rel_x_offset, rel_y_offset = [float(i) for i in rel_offset]
        except Exception:
            raise ValueError(
                f"rel_offset expected to be a pair of floats, not {rel_offset}"
            )
        lon0, lon1, lat0, lat1 = iter(self)

        dlon = lon1 - lon0
        dlat = lat1 - lat0

        clon = lon0 + (0.5 + rel_x_offset) * dlon
        clat = lat0 + (0.5 + rel_y_offset) * dlat

        dlon_zm = dlon / fact
        dlat_zm = dlat / fact

        coords = np.array(
            [
                clon - 0.5 * dlon_zm,
                clon + 0.5 * dlon_zm,
                clat - 0.5 * dlat_zm,
                clat + 0.5 * dlat_zm,
            ],
            float,
        )

        self.set(self.coord_type, *coords)
        return self

    @property
    def lon(self):
        return np.asarray(self)[:2]

    @property
    def lat(self):
        return np.asarray(self)[2:]

    @property
    def _proj_data(self):
        return self.map_axes.proj_data

    @property
    def _proj_geo(self):
        return self.map_axes.proj_geo

    @property
    def _proj_map(self):
        return self.map_axes.proj_map

    @property
    def _transAxes(self):
        return self.map_axes.ax.transAxes

    @property
    def _transData(self):
        return self.map_axes.ax.transData


@summarizable(
    attrs=[
        "loc",
        "loc_y",
        "loc_x",
        "dx_unit",
        "dy_unit",
        "dx",
        "dy",
        "x0",
        "y0",
        "x",
        "y",
        "va",
        "ha",
    ],
    post_summarize=post_summarize_plot,
)
class TextBoxLocation:
    """A reference location (like bottom-left) inside a box on a 3x3 grid."""

    def __init__(self, parent, loc, dx=None, dy=None):
        """Initialize an instance of TextBoxLocation.

        Args:
            parent (TextBoxAxes): Parent text box axes.

            loc (int or str): Location parameter. Takes one of three formats:
                integer, short string, or long string.

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

            dx (float, optional): Horizontal offset in unit distances. Defaults
                to 0.0.

            dy (float, optional): Vertical offset in unit distances. Defaults
                to 0.0.

        """
        self.dx = dx or 0.0
        self.dy = dy or 0.0

        self._determine_loc_components(loc)

        self.dx_unit = parent.dx_unit
        self.dy_unit = parent.dy_unit
        self.pad_x = parent.pad_x
        self.pad_y = parent.pad_y

    def _determine_loc_components(self, loc):
        """Split and evaluate components of location parameter."""

        loc = str(loc)

        # Split location into vertical and horizontal part
        if len(loc) == 2:
            loc_y, loc_x = loc
        elif loc == "center":
            loc_y, loc_x = loc, loc
        else:
            loc_y, loc_x = loc.split(" ", 1)

        # Evaluate location components
        self.loc_y = self._standardize_loc_y(loc_y)
        self.loc_x = self._standardize_loc_x(loc_x)
        self.loc = f"{self.loc_y}{self.loc_x}"

    def _standardize_loc_y(self, loc):
        """Standardize vertical location component."""
        if loc in (0, "0", "b", "bottom"):
            return "b"
        elif loc in (1, "1", "m", "middle"):
            return "m"
        elif loc in (2, "2", "t", "top"):
            return "t"
        raise ValueError(f"invalid vertical location component '{loc}'")

    def _standardize_loc_x(self, loc):
        """Standardize horizontal location component."""
        if loc in (0, "0", "l", "left"):
            return "l"
        elif loc in (1, "1", "c", "center"):
            return "c"
        elif loc in (2, "2", "r", "right"):
            return "r"
        raise ValueError(f"invalid horizontal location component '{loc}'")

    @property
    def va(self):
        """Vertical alignment variable."""
        return {"b": "baseline", "m": "center_baseline", "t": "top"}[self.loc_y]

    @property
    def ha(self):
        """Horizontal alignment variable."""
        return {"l": "left", "c": "center", "r": "right"}[self.loc_x]

    @property
    def y0(self):
        """Vertical baseline position."""
        if self.loc_y == "b":
            return 0.0 + self.pad_y
        elif self.loc_y == "m":
            return 0.5
        elif self.loc_y == "t":
            return 1.0 - self.pad_y
        else:
            raise Exception(
                f"invalid {type(self).__name__} instance attr loc_y: '{self.loc_y}'"
            )

    @property
    def x0(self):
        """Horizontal baseline position."""
        if self.loc_x == "l":
            return 0.0 + self.pad_x
        elif self.loc_x == "c":
            return 0.5
        elif self.loc_x == "r":
            return 1.0 - self.pad_x
        else:
            raise Exception(
                f"invalid {type(self).__name__} instance attr loc_x: '{self.loc_x}'"
            )

    @property
    def x(self):
        """Horizontal position."""
        return self.x0 + self.dx * self.dx_unit

    @property
    def y(self):
        """Vertical position."""
        return self.y0 + self.dy * self.dy_unit


def ax_w_h_in_fig_coords(fig, ax):
    """Get the dimensions of an axes in figure coords."""
    trans = fig.transFigure.inverted()
    x, y, w, h = ax.bbox.transformed(trans).bounds
    return w, h


def colors_from_cmap(cmap, n_levels, extend):
    """Get colors from cmap for given no. levels and extend param."""
    colors = cmap(np.linspace(0, 1, n_levels + 1))
    if extend == "both":
        return colors
    elif extend == "min":
        return colors[:-1]
    elif extend == "max":
        return colors[1:]
    else:
        return colors[1:-1]

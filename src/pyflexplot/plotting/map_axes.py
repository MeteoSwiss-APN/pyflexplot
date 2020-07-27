# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

# Third-party
import cartopy
import matplotlib as mpl
import numpy as np
from cartopy.crs import Projection  # type: ignore
from cartopy.io.shapereader import Record  # type: ignore
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from pydantic import BaseModel
from pydantic import root_validator
from pydantic import validator

# First-party
from pyflexplot.data import Field

# Local
from ..utils.summarize import post_summarize_plot
from ..utils.summarize import summarizable
from ..utils.typing import ColorType
from ..utils.typing import RectType
from .coord_trans import CoordinateTransformer
from .ref_dist_indicator import RefDistIndConf
from .ref_dist_indicator import ReferenceDistanceIndicator


@summarizable
# pylint: disable=E0213  # no-self-argument (validators)
class MapAxesConf(BaseModel):
    """
    Configuration of ``MapAxesPlot``.

    Args:
        geo_res: Resolution of geographic map elements.

        geo_res_cities: Scale for cities shown on map. Defaults to ``geo_res``.

        geo_res_rivers: Scale for rivers shown on map. Defaults to ``geo_res``.

        lang: Language ('en' for English, 'de' for German).

        lllat: Latitude of lower-left corner.

        lllon: Longitude of lower-left corner.

        lw_frame: Line width of frames.

        min_city_pop: Minimum population of cities shown.

        urlat: Latitude of upper-right corner.

        urlon: Longitude of upper-right corner.

        ref_dist_conf: Reference distance indicator setup.

        ref_dist_on: Whether to add a reference distance indicator.

        rel_offset: Relative offset in x and y direction as a fraction of the
            respective domain extent.

        zoom_fact: Zoom factor. Use values above/below 1.0 to zoom in/out.

    """

    geo_res: str = "50m"
    geo_res_cities: str = "none"
    geo_res_rivers: str = "none"
    lang: str = "en"
    lllat: Optional[float] = None
    lllon: Optional[float] = None
    lw_frame: float = 1.0
    min_city_pop: int = 0
    urlat: Optional[float] = None
    urlon: Optional[float] = None
    ref_dist_conf: RefDistIndConf = RefDistIndConf()
    ref_dist_on: bool = True
    rel_offset: Tuple[float, float] = (0.0, 0.0)
    zoom_fact: float = 1.0

    class Config:  # noqa
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def _init_ref_dist_conf(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        param = "ref_dist_conf"
        type_ = RefDistIndConf
        try:
            value = values[param]
        except KeyError:
            values[param] = type_()
        else:
            if isinstance(value, dict):
                values[param] = type_(**value)
            elif not isinstance(value, type_):
                raise TypeError(
                    f"ref_dist_conf: expected dict or {type_.__name__}, got "
                    f"{type(value).__name__}",
                    value,
                )
        return values

    @validator("geo_res_cities", always=True, allow_reuse=True)
    def _init_geo_res_cities(cls, value: str, values: Dict[str, Any]) -> str:
        if value == "none":
            value = values["geo_res"]
        return value

    _init_geo_res_rivers = validator("geo_res_rivers", always=True)(
        _init_geo_res_cities  # type: ignore
    )


# pylint: disable=R0902  # too-many-instance-attributes
@summarizable(post_summarize=post_summarize_plot)
@dataclass
class MapAxes:
    """Map plot axes for regular lat/lon data.

    Args:
        fig: Figure to which to map axes is added.

        rect: Position of map plot panel in figure coordinates.

        field: Field.

        conf: Map axes setup.

    """

    fig: Figure
    rect: RectType
    field: Field
    conf: MapAxesConf

    def __post_init__(self) -> None:
        self._water_color: ColorType = "lightskyblue"

        self.zorder: Dict[str, int]
        self._init_zorder()

        self.proj_data: Projection
        self.proj_map: Projection
        self.proj_geo: Projection
        self._init_projs()

        self.ax: Axes
        self._init_ax()

        self.trans = CoordinateTransformer(
            trans_axes=self.ax.transAxes,
            trans_data=self.ax.transData,
            proj_geo=self.proj_geo,
            proj_map=self.proj_map,
            proj_data=self.proj_data,
        )

        self.ref_dist_box: Optional[ReferenceDistanceIndicator]
        self._init_ref_dist_box()

        self._ax_add_grid()
        self._ax_add_geography()
        self._ax_add_data_domain_outline()
        self._ax_add_frame()

    @classmethod
    def create(
        cls, conf: MapAxesConf, *, fig: Figure, rect: RectType, field: np.ndarray,
    ) -> "MapAxes":
        if field.rotated_pole:
            return MapAxesRotatedPole.create(conf, fig=fig, rect=rect, field=field)
        else:
            return cls(fig=fig, rect=rect, field=field, conf=conf)

    def marker(
        self,
        *,
        p_lon: float,
        p_lat: float,
        marker: str,
        zorder: Optional[int] = None,
        **kwargs,
    ) -> Sequence[Line2D]:
        """Add a marker at a location in natural coordinates."""
        p_lon, p_lat = self.trans.geo_to_data(p_lon, p_lat)
        if zorder is None:
            zorder = self.zorder["marker"]
        handle = self.ax.plot(
            p_lon,
            p_lat,
            marker=marker,
            transform=self.proj_data,
            zorder=zorder,
            **kwargs,
        )
        return handle

    def text(
        self,
        p_lon: float,
        p_lat: float,
        s: str,
        *,
        zorder: Optional[int] = None,
        **kwargs,
    ) -> Text:
        """Add text at a geographical point.

        Args:
            p_lon: Point longitude.

            p_lat: Point latitude.

            s: Text string.

            dx: Horizontal offset in unit distances.

            dy (optional): Vertical offset in unit distances.

            zorder (optional): Vertical order. Defaults to "geo_lower".

            **kwargs: Additional keyword arguments for ``ax.text``.

        """
        if zorder is None:
            zorder = self.zorder["geo_lower"]
        kwargs_default = {"xytext": (5, 1), "textcoords": "offset points"}
        kwargs = {**kwargs_default, **kwargs}
        # pylint: disable=W0212  # protected-access
        transform = self.proj_geo._as_mpl_transform(self.ax)
        # -> see https://stackoverflow.com/a/25421922/4419816
        handle = self.ax.annotate(
            s, xy=(p_lon, p_lat), xycoords=transform, zorder=zorder, **kwargs,
        )
        return handle

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<TODO>)"  # SR_TODO

    def _init_zorder(self) -> None:
        """Determine zorder of unique plot elements, from low to high."""
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
        self.zorder = {name: d0 + idx * dz for idx, name in enumerate(zorders_const)}

    def _init_projs(self) -> None:
        """Prepare projections to transform the data for plotting."""

        # Projection of input data
        self.proj_data = cartopy.crs.PlateCarree()

        # Projection of plot
        self.proj_map = cartopy.crs.PlateCarree()  # SR_TMP

        # Geographical projection
        self.proj_geo = cartopy.crs.PlateCarree()

    def _init_ax(self) -> None:
        """Initialize Axes."""
        ax: Axes = self.fig.add_axes(self.rect, projection=self.proj_map)
        ax.set_adjustable("datalim")
        ax.outline_patch.set_edgecolor("none")
        self.ax = ax

        # Set geographical extent
        conf: MapAxesConf = self.conf
        field: Field = self.field
        domain_type: str = field.var_setups.collect_equal("domain")
        if domain_type == "data":
            assert (field.lat.size, field.lon.size) == field.time_props.mask_nz.shape
            mask_lon = field.time_props.mask_nz.any(axis=0)
            mask_lat = field.time_props.mask_nz.any(axis=1)
            lllon = field.lon[mask_lon].min()
            urlon = field.lon[mask_lon].max()
            lllat = field.lat[mask_lat].min()
            urlat = field.lat[mask_lat].max()
        else:
            lllon = conf.lllon if conf.lllon is not None else field.lon[0]
            urlon = conf.urlon if conf.urlon is not None else field.lon[-1]
            lllat = conf.lllat if conf.lllat is not None else field.lat[0]
            urlat = conf.urlat if conf.urlat is not None else field.lat[-1]
        bbox = (
            MapAxesBoundingBox(self, "data", lllon, urlon, lllat, urlat)
            .to_axes()
            .zoom(conf.zoom_fact, conf.rel_offset)
            .to_data()
        )
        self.ax.set_extent(bbox, self.proj_data)

    def _init_ref_dist_box(self) -> None:
        """Initialize the reference distance indicator (if activated)."""
        if not self.conf.ref_dist_on:
            self.ref_dist_box = None
        else:
            self.ref_dist_box = ReferenceDistanceIndicator(
                ax=self.ax,
                axes_to_geo=self.trans.axes_to_geo,
                conf=self.conf.ref_dist_conf,
                zorder=self.zorder["grid"],
            )

    def _ax_add_grid(self) -> None:
        """Show grid lines on map."""
        gl = self.ax.gridlines(
            linestyle=":", linewidth=1, color="black", zorder=self.zorder["grid"],
        )
        gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180, 180.1, 2))
        gl.ylocator = mpl.ticker.FixedLocator(np.arange(-90, 90.1, 2))

    def _ax_add_geography(self) -> None:
        """Add geographic elements: coasts, countries, colors, etc."""
        self.ax.coastlines(resolution=self.conf.geo_res)
        self.ax.background_patch.set_facecolor(self._water_color)
        self._ax_add_countries("lowest")
        self._ax_add_lakes("lowest")
        self._ax_add_rivers("lowest")
        self._ax_add_countries("geo_upper")
        self._ax_add_cities()

    def _ax_add_countries(self, zorder_key: str) -> None:
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

    def _ax_add_lakes(self, zorder_key: str) -> None:
        self.ax.add_feature(
            cartopy.feature.NaturalEarthFeature(
                category="physical",
                name="lakes",
                scale=self.conf.geo_res,
                edgecolor="none",
                facecolor=self._water_color,
            ),
            zorder=self.zorder[zorder_key],
        )
        if self.conf.geo_res == "10m":
            self.ax.add_feature(
                cartopy.feature.NaturalEarthFeature(
                    category="physical",
                    name="lakes_europe",
                    scale=self.conf.geo_res,
                    edgecolor="none",
                    facecolor=self._water_color,
                ),
                zorder=self.zorder[zorder_key],
            )

    def _ax_add_rivers(self, zorder_key: str) -> None:
        linewidth = {"lowest": 1, "geo_lower": 1, "geo_upper": 2 / 3}[zorder_key]

        # SR_WORKAROUND <
        # Note:
        #  - Bug in Cartopy with recent shapefiles triggers errors (NULL geometry)
        #  - Issue fixed in a branch but pull request still pending
        #       -> Branch: https://github.com/shevawen/cartopy/tree/patch-1
        #       -> PR: https://github.com/SciTools/cartopy/pull/1411
        #  - Fixed it in our fork MeteoSwiss-APN/cartopy
        #       -> Fork fixes setup depencies issue (with  pyproject.toml)
        #  - For now, check validity of rivers geometry objects when adding
        #  - Once it works with the master branch, remove these workarounds
        # SR_WORKAROUND >

        major_rivers = cartopy.feature.NaturalEarthFeature(
            category="physical",
            name="rivers_lake_centerlines",
            scale=self.conf.geo_res,
            edgecolor=self._water_color,
            facecolor=(0, 0, 0, 0),
            linewidth=linewidth,
        )
        # SR_WORKAROUND < TODO revert once bugfix in Cartopy master
        # self.ax.add_feature(major_rivers, zorder=self.zorder[zorder_key])
        # SR_WORKAROUND >

        if self.conf.geo_res_rivers == "10m":
            minor_rivers = cartopy.feature.NaturalEarthFeature(
                category="physical",
                name="rivers_europe",
                scale=self.conf.geo_res,
                edgecolor=self._water_color,
                facecolor=(0, 0, 0, 0),
                linewidth=linewidth,
            )
            # SR_WORKAROUND < TODO revert once bugfix in Cartopy master
            # self.ax.add_feature(minor_rivers, zorder=self.zorder[zorder_key])
            # SR_WORKAROUND >

        # SR_WORKAROUND <<< TODO remove once bugfix in Cartopy master
        try:
            major_rivers.geometries()
        except Exception:  # pylint: disable=W0703  # broad-except
            warnings.warn(
                "cannot add major rivers due to shapely issue with "
                "'rivers_lake_centerline; pending bugfix: "
                "https://github.com/SciTools/cartopy/pull/1411; workaround: use "
                "https://github.com/shevawen/cartopy/tree/patch-1"
            )
        else:
            # warnings.warn(
            #     f"successfully added major rivers; "
            #     "TODO: remove workaround and pin minimum Cartopy version!"
            # )
            self.ax.add_feature(major_rivers, zorder=self.zorder[zorder_key])
            if self.conf.geo_res_rivers == "10m":
                self.ax.add_feature(minor_rivers, zorder=self.zorder[zorder_key])
        # SR_WORKAROUND >

    def _ax_add_cities(self) -> None:
        """Add major cities, incl. all capitals."""

        # pylint: disable=R0913  # too-many-arguments
        def is_in_box(
            x: float, y: float, x0: float, x1: float, y0: float, y1: float
        ) -> bool:
            return x0 <= x <= x1 and y0 <= y <= y1

        def is_visible(city: Record) -> bool:
            """Check if a point is inside the domain."""
            p_lon: float = city.geometry.x
            p_lat: float = city.geometry.y

            # In domain
            p_lon, p_lat = self.proj_data.transform_point(
                p_lon, p_lat, self.proj_geo, trap=True,
            )
            in_domain = is_in_box(
                p_lon,
                p_lat,
                self.field.lon[0],
                self.field.lon[-1],
                self.field.lat[0],
                self.field.lat[-1],
            )

            # Not behind reference distance indicator box
            pxa, pya = self.trans.geo_to_axes(p_lon, p_lat)
            rdb = self.ref_dist_box
            assert rdb is not None  # mypy
            behind_rdb = is_in_box(
                pxa, pya, rdb.x0_box, rdb.x1_box, rdb.y0_box, rdb.y1_box,
            )

            return in_domain and not behind_rdb

        def is_of_interest(city: Record) -> bool:
            """Check if a city fulfils certain importance criteria."""
            is_capital = city.attributes["FEATURECLA"].startswith("Admin-0 capital")
            is_large = city.attributes["GN_POP"] > self.conf.min_city_pop
            return is_capital or is_large

        def get_name(city: Record) -> str:
            """Fetch city name in current language, hand-correcting some."""
            name = city.attributes[f"name_{self.conf.lang}"]
            if name.startswith("Freiburg im ") and name.endswith("echtland"):
                name = "Freiburg"
            return name

        # src: https://www.naturalearthdata.com/downloads/50m-cultural-vectors/50m-populated-places/lk  # noqa
        cities: Sequence[Record] = cartopy.io.shapereader.Reader(
            cartopy.io.shapereader.natural_earth(
                category="cultural",
                name="populated_places",
                resolution=self.conf.geo_res_cities,
            )
        ).records()

        for city in cities:
            lon, lat = city.geometry.x, city.geometry.y
            if is_visible(city) and is_of_interest(city):
                self.marker(
                    p_lat=lat,
                    p_lon=lon,
                    marker="o",
                    color="black",
                    fillstyle="none",
                    markeredgewidth=1,
                    markersize=3,
                    zorder=self.zorder["geo_upper"],
                )
                name = get_name(city)
                self.text(lon, lat, name, va="center", size="small", clip_on=True)

    def _ax_add_data_domain_outline(self) -> None:
        """Add domain outlines to map plot."""
        lon0, lon1 = self.field.lon[[0, -1]]
        lat0, lat1 = self.field.lat[[0, -1]]
        xs = [lon0, lon1, lon1, lon0, lon0]
        ys = [lat0, lat0, lat1, lat1, lat0]
        self.ax.plot(xs, ys, transform=self.proj_data, c="black", lw=1)

    def _ax_add_frame(self) -> None:
        """Draw frame around map plot."""
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


@dataclass
class MapAxesRotatedPole(MapAxes):
    """Map plot axes for rotated-pole data."""

    pollon: float
    pollat: float

    @classmethod
    def create(
        cls, conf: MapAxesConf, *, fig: Figure, rect: RectType, field: Field
    ) -> "MapAxesRotatedPole":
        if not field.rotated_pole:
            raise ValueError("not a rotated-pole field", field)
        rotated_pole = field.nc_meta_data["variables"]["rotated_pole"]["ncattrs"]
        pollat = rotated_pole["grid_north_pole_latitude"]
        pollon = rotated_pole["grid_north_pole_longitude"]
        return cls(
            fig=fig, rect=rect, field=field, pollat=pollat, pollon=pollon, conf=conf,
        )

    def _init_projs(self) -> None:
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


class MapAxesBoundingBox:
    """Bounding box of a ``MapAxes``."""

    # pylint: disable=R0913  # too-many-arguments
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
        self.coord_types = ["data", "geo"]  # SR_TMP
        self.map_axes = map_axes
        self.set(coord_type, lon0, lon1, lat0, lat1)
        self.trans = CoordinateTransformer(
            trans_axes=map_axes.ax.transAxes,
            trans_data=map_axes.ax.transData,
            proj_geo=map_axes.proj_geo,
            proj_map=map_axes.proj_map,
            proj_data=map_axes.proj_data,
            invalid_ok=False,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f'map_axes={self.map_axes}, coord_type="{self.coord_type}", '
            f"lon0={self.lon0:.2f}, lon1={self.lon1:.2f}, "
            f"lat0={self.lat0:.2f}, lat1={self.lat1:.2f})"
        )

    @property
    def lon(self):
        return np.asarray(self)[:2]

    @property
    def lat(self):
        return np.asarray(self)[2:]

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

    # pylint: disable=R0913  # too-many-arguments
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
            coords = np.concatenate(self.trans.geo_to_data(self.lon, self.lat))
        elif self.coord_type == "axes":
            return self.to_geo().to_data()
        else:
            self._error("to_data")
        self.set("data", *coords)
        return self

    def to_geo(self):
        if self.coord_type == "data":
            coords = np.concatenate(self.trans.data_to_geo(self.lon, self.lat))
        elif self.coord_type == "axes":
            coords = np.concatenate(self.trans.axes_to_geo(self.lon, self.lat))
        else:
            self._error("to_geo")
        self.set("geo", *coords)
        return self

    def to_axes(self):
        if self.coord_type == "geo":
            coords = np.concatenate(self.trans.geo_to_axes(self.lon, self.lat))
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

    # pylint: disable=R0914  # too-many-locals
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

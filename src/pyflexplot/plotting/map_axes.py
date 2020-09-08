# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Tuple

# Third-party
import cartopy
import matplotlib as mpl
import numpy as np
from cartopy.crs import PlateCarree  # type: ignore
from cartopy.crs import Projection
from cartopy.crs import RotatedPole
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
from .ref_dist_indicator import RefDistIndConfig
from .ref_dist_indicator import ReferenceDistanceIndicator


@summarizable
@dataclass
class Domain:
    """Plot domain.

    Args:
        field: Field to be plotted on the domain.

        zoom_fact (optional): Zoom factor. Use values above/below 1.0 to zoom
            in/out.

        rel_offset (optional): Relative offset in x and y direction as a
            fraction of the respective domain extent.

    """

    field: Field
    zoom_fact: float = 1.0
    rel_offset: Tuple[float, float] = (0.0, 0.0)

    def get_bbox(self, map_axes: "MapAxes") -> "MapAxesBoundingBox":
        """Get bounding box of domain."""
        lllon, urlon, lllat, urlat = self._get_bbox_corners(map_axes)
        bbox = MapAxesBoundingBox(map_axes, "data", lllon, urlon, lllat, urlat)
        if self.zoom_fact != 1.0:
            bbox = bbox.to_axes().zoom(self.zoom_fact, self.rel_offset).to_data()
        return bbox

    # pylint: disable=W0613  # unused-argument (map_axes)
    def _get_bbox_corners(
        self, map_axes: "MapAxes"
    ) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        lllat = self.field.lat[0]
        urlat = self.field.lat[-1]
        lllon = self.field.lon[0]
        urlon = self.field.lon[-1]
        return lllon, urlon, lllat, urlat


class CloudDomain(Domain):
    """Domain derived from spatial distribution of cloud over time."""

    def _get_bbox_corners(
        self, map_axes: "MapAxes"
    ) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        lat: np.ndarray = self.field.lat
        lon: np.ndarray = self.field.lon
        assert (lat.size, lon.size) == self.field.time_props.mask_nz.shape
        mask_lat = self.field.time_props.mask_nz.any(axis=1)
        mask_lon = self.field.time_props.mask_nz.any(axis=0)
        if not any(mask_lat):
            lllat = lat.min()
            urlat = lat.max()
        else:
            lllat = lat[mask_lat].min()
            urlat = lat[mask_lat].max()
        if not any(mask_lon):
            lllon = lon.min()
            urlon = lon.max()
        else:
            lllon = lon[mask_lon].min()
            urlon = lon[mask_lon].max()

        # Increase latitudinal size if minimum specified
        d_lat_min = self.field.var_setups.collect_equal("domain_size_lat")
        if d_lat_min is not None:
            d_lat = urlat - lllat
            if d_lat < d_lat_min:
                lllat -= 0.5 * (d_lat_min - d_lat)
                urlat += 0.5 * (d_lat_min - d_lat)

        # Increase latitudinal size if minimum specified
        d_lon_min = self.field.var_setups.collect_equal("domain_size_lon")
        if d_lon_min is not None:
            d_lon = urlon - lllon
            if d_lon < d_lon_min:
                lllon -= 0.5 * (d_lon_min - d_lon)
                urlon += 0.5 * (d_lon_min - d_lon)

        # Adjust aspect ratio to avoid distortion
        d_lat = urlat - lllat
        d_lon = urlon - lllon
        aspect = map_axes.get_aspect_ratio()
        if d_lon < d_lat * aspect:
            lllon -= 0.5 * (d_lat * aspect - d_lon)
            urlon += 0.5 * (d_lat * aspect - d_lon)
        elif d_lat < d_lon / aspect:
            lllat -= 0.5 * (d_lon / aspect - d_lat)
            urlat += 0.5 * (d_lon / aspect - d_lat)

        return lllon, urlon, lllat, urlat


class ReleaseSiteDomain(Domain):
    """Domain relative to release point."""

    def _get_bbox_corners(
        self, map_axes: "MapAxes"
    ) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        assert self.field.mdata is not None  # mypy
        release_lat: float = self.field.mdata.release.lat
        release_lon: float = self.field.mdata.release.lon
        d_lat = self.field.var_setups.collect_equal("domain_size_lat")
        d_lon = self.field.var_setups.collect_equal("domain_size_lon")
        if d_lat is None and d_lon is None:
            raise Exception(
                "domain type 'release_site': setup params 'domain_size_(lat|lon)'"
                " are both None; one or both is required"
            )
        elif d_lat is None:
            d_lat = d_lon / map_axes.get_aspect_ratio()
        elif d_lon is None:
            d_lon = d_lat / map_axes.get_aspect_ratio()
        assert self.field.mdata is not None  # mypy
        if isinstance(self.field.proj, RotatedPole):
            c_lon, c_lat = self.field.proj.transform_point(
                release_lon, release_lat, PlateCarree(),
            )
            lllat = c_lat - 0.5 * d_lat
            lllon = c_lon - 0.5 * d_lon
            urlat = c_lat + 0.5 * d_lat
            urlon = c_lon + 0.5 * d_lon
        else:
            lllat = release_lat - 0.5 * d_lat
            lllon = release_lon - 0.5 * d_lon
            urlat = release_lat + 0.5 * d_lat
            urlon = release_lon + 0.5 * d_lon
            lllon, lllat = self.field.proj.transform_point(lllon, lllat, PlateCarree())
            urlon, urlat = self.field.proj.transform_point(urlon, urlat, PlateCarree())
        return lllon, urlon, lllat, urlat


@summarizable
# pylint: disable=E0213  # no-self-argument (validators)
class MapAxesConfig(BaseModel):
    """
    Configuration of ``MapAxesPlot``.

    Args:
        domain: Plot domain.

        geo_res: Resolution of geographic map elements.

        geo_res_cities: Scale for cities shown on map. Defaults to ``geo_res``.

        geo_res_rivers: Scale for rivers shown on map. Defaults to ``geo_res``.

        lang: Language ('en' for English, 'de' for German).

        lw_frame: Line width of frames.

        projection: Map projection. Defaults to that of the input data.

        min_city_pop: Minimum population of cities shown.

        ref_dist_config: Reference distance indicator setup.

        ref_dist_on: Whether to add a reference distance indicator.

        scale_fact: Scaling factor for plot elements (fonts, lines, etc.)

    """

    domain: Domain
    geo_res: str = "50m"
    geo_res_cities: str = "none"
    geo_res_rivers: str = "none"
    lang: str = "en"
    lw_frame: float = 1.0
    projection: str = "data"
    min_city_pop: int = 0
    ref_dist_config: RefDistIndConfig
    ref_dist_on: bool = True
    scale_fact: float = 1.0

    class Config:  # noqa
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def _init_ref_dist_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            value = values["ref_dist_config"]
        except KeyError:
            # Not passed as argument; use default
            values["ref_dist_config"] = RefDistIndConfig()
        else:
            if isinstance(value, MutableMapping):
                # Passed a dict as argument; turn into ``RefDistIndConfig``
                values["ref_dist_config"] = RefDistIndConfig(**value)
            elif not isinstance(value, RefDistIndConfig):
                # Passed neither a dict nor a ``RefDistIndConfig`` as argument
                raise TypeError(
                    f"ref_dist_config: expected dict or RefDistIndConfig, got "
                    f"{type(value).__name__}",
                    value,
                )
        scale_fact = values.get("scale_fact", 1.0)
        values["ref_dist_config"] = values["ref_dist_config"].scale(scale_fact)
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
@summarizable
@dataclass
class MapAxes:
    """Map plot axes for regular lat/lon data.

    Args:
        fig: Figure to which to map axes is added.

        rect: Position of map plot panel in figure coordinates.

        field: Field.

        config: Map axes setup.

    """

    fig: Figure
    rect: RectType
    field: Field
    config: MapAxesConfig

    def __post_init__(self) -> None:
        self.elements: List[Tuple[str, Any]] = []
        self._summarized_elements: List[Dict[str, Any]] = []

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
        cls, config: MapAxesConfig, *, fig: Figure, rect: RectType, field: np.ndarray,
    ) -> "MapAxes":
        if isinstance(field.proj, RotatedPole):
            return MapAxesRotatedPole.create(config, fig=fig, rect=rect, field=field)
        else:
            return cls(fig=fig, rect=rect, field=field, config=config)

    def post_summarize(
        self, summary: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        summary = post_summarize_plot(self, summary)
        summary["elements"] = self._summarized_elements
        return summary

    def add_marker(
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
        self.elements.append(handle)
        self._summarized_elements.append(
            {
                "element_type": "marker",
                "p_lon": p_lon,
                "p_lat": p_lat,
                "marker": marker,
                "transform": f"{type(self.proj_data).__name__} instance",
                "zorder": zorder,
                **kwargs,
            }
        )
        return handle

    def add_text(
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
        self.elements.append(handle)
        self._summarized_elements.append(
            {
                "element_type": "text",
                "s": s,
                "xy": (p_lon, p_lat),
                "xycoords": f"{type(transform).__name__} instance",
                "zorder": zorder,
                **kwargs,
            }
        )
        return handle

    def get_aspect_ratio(self) -> float:
        """Get aspect ratio (height by width) of map plot."""
        bbox = self.ax.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted()
        )
        return bbox.width / bbox.height

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
        self._init_proj_data()
        self._init_proj_map()
        self.proj_geo = cartopy.crs.PlateCarree()

    def _init_proj_data(self) -> None:
        """Initialize projection of input data."""
        self.proj_data = cartopy.crs.PlateCarree()

    def _init_proj_map(self) -> None:
        """Initialize projection of map plot."""
        projection: str = self.config.projection
        if projection == "data":
            self.proj_map = self.proj_data
        elif projection == "mercator":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.proj_map = cartopy.crs.TransverseMercator(
                    central_longitude=self.field.lon.mean(), approx=True,
                )
        else:
            choices = ["data", "mercator"]
            raise NotImplementedError(f"projection '{projection}'; choices: {choices}")

    # pylint: disable=R0912  # too-many-branches
    # pylint: disable=R0914  # too-many-locals
    # pylint: disable=R0915  # too-many-statements
    def _init_ax(self) -> None:
        """Initialize Axes."""
        ax: Axes = self.fig.add_axes(self.rect, projection=self.proj_map)
        ax.set_adjustable("datalim")
        ax.outline_patch.set_edgecolor("none")
        self.ax = ax

        # Set geographical extent
        self.ax.set_aspect("auto")
        bbox: MapAxesBoundingBox = self.config.domain.get_bbox(self)
        self.ax.set_extent(bbox, self.proj_data)

    def _init_ref_dist_box(self) -> None:
        """Initialize the reference distance indicator (if activated)."""
        if not self.config.ref_dist_on:
            self.ref_dist_box = None
        else:
            self.ref_dist_box = ReferenceDistanceIndicator(
                ax=self.ax,
                axes_to_geo=self.trans.axes_to_geo,
                config=self.config.ref_dist_config,
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
        self.ax.coastlines(resolution=self.config.geo_res)
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
                scale=self.config.geo_res,
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
                scale=self.config.geo_res,
                edgecolor="none",
                facecolor=self._water_color,
            ),
            zorder=self.zorder[zorder_key],
        )
        if self.config.geo_res == "10m":
            self.ax.add_feature(
                cartopy.feature.NaturalEarthFeature(
                    category="physical",
                    name="lakes_europe",
                    scale=self.config.geo_res,
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
            scale=self.config.geo_res,
            edgecolor=self._water_color,
            facecolor=(0, 0, 0, 0),
            linewidth=linewidth,
        )
        # SR_WORKAROUND < TODO revert once bugfix in Cartopy master
        # self.ax.add_feature(major_rivers, zorder=self.zorder[zorder_key])
        # SR_WORKAROUND >

        if self.config.geo_res_rivers == "10m":
            minor_rivers = cartopy.feature.NaturalEarthFeature(
                category="physical",
                name="rivers_europe",
                scale=self.config.geo_res,
                edgecolor=self._water_color,
                facecolor=(0, 0, 0, 0),
                linewidth=linewidth,
            )
            # SR_WORKAROUND < TODO revert once bugfix in Cartopy master
            # self.ax.add_feature(minor_rivers, zorder=self.zorder[zorder_key])
            # SR_WORKAROUND >
        else:
            minor_rivers = None

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
            if self.config.geo_res_rivers == "10m":
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
            behind_rdb = (
                False
                if rdb is None
                else is_in_box(
                    pxa, pya, rdb.x0_box, rdb.x1_box, rdb.y0_box, rdb.y1_box,
                )
            )

            return in_domain and not behind_rdb

        def is_of_interest(city: Record) -> bool:
            """Check if a city fulfils certain importance criteria."""
            is_capital = city.attributes["FEATURECLA"].startswith("Admin-0 capital")
            is_large = city.attributes["GN_POP"] > self.config.min_city_pop
            return is_capital or is_large

        def get_name(city: Record) -> str:
            """Fetch city name in current language, hand-correcting some."""
            name = city.attributes[f"name_{self.config.lang}"]
            if name.startswith("Freiburg im ") and name.endswith("echtland"):
                name = "Freiburg"
            return name

        # src: https://www.naturalearthdata.com/downloads/50m-cultural-vectors/50m-populated-places/lk  # noqa
        cities: Sequence[Record] = cartopy.io.shapereader.Reader(
            cartopy.io.shapereader.natural_earth(
                category="cultural",
                name="populated_places",
                resolution=self.config.geo_res_cities,
            )
        ).records()

        for city in cities:
            lon, lat = city.geometry.x, city.geometry.y
            if is_visible(city) and is_of_interest(city):
                self.add_marker(
                    p_lat=lat,
                    p_lon=lon,
                    marker="o",
                    color="black",
                    fillstyle="none",
                    markeredgewidth=1 * self.config.scale_fact,
                    markersize=3 * self.config.scale_fact,
                    zorder=self.zorder["geo_upper"],
                )
                self.add_text(
                    lon,
                    lat,
                    get_name(city),
                    va="center",
                    size=9 * self.config.scale_fact,
                    clip_on=True,
                )

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
                linewidth=self.config.lw_frame,
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
        cls, config: MapAxesConfig, *, fig: Figure, rect: RectType, field: Field
    ) -> "MapAxesRotatedPole":
        if not isinstance(field.proj, RotatedPole):
            raise ValueError("not a rotated-pole field", field)
        rotated_pole = field.nc_meta_data["variables"]["rotated_pole"]["ncattrs"]
        pollat = rotated_pole["grid_north_pole_latitude"]
        pollon = rotated_pole["grid_north_pole_longitude"]
        return cls(
            fig=fig,
            rect=rect,
            field=field,
            pollat=pollat,
            pollon=pollon,
            config=config,
        )

    def _init_proj_data(self) -> None:
        """Initialize projection of input data."""
        self.proj_data = cartopy.crs.RotatedPole(
            pole_latitude=self.pollat, pole_longitude=self.pollon
        )


class MapAxesBoundingBox:
    """Bounding box of a ``MapAxes``."""

    # pylint: disable=R0913  # too-many-arguments
    def __init__(
        self,
        map_axes: MapAxes,
        coord_type: str,
        lon0: float,
        lon1: float,
        lat0: float,
        lat1: float,
    ) -> None:
        """Create an instance of ``MapAxesBoundingBox``.

        Args:
            map_axes: Parent map axes object.

            coord_type: Coordinates type.

            lon0: Longitude of south-western corner.

            lon1: Longitude of north-eastern corner.

            lat0: Latitude of south-western corner.

            lat1: Latitude of north-eastern corner.

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
            f"{type(self).__name__}(\n  "
            + ",\n  ".join(
                [
                    f"map_axes={self.map_axes}",
                    f"coord_type='{self.coord_type}'",
                    f"lon0={self.lon0:.2f}",
                    f"lon1={self.lon1:.2f}",
                    f"lat0={self.lat0:.2f}",
                    f"lat1={self.lat1:.2f}",
                ]
            )
            + ",\n)"
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

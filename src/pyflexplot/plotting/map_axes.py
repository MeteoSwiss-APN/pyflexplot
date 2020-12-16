"""Map axes."""
# Standard library
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import cartopy
import matplotlib as mpl
import numpy as np
from cartopy.io.shapereader import Record  # type: ignore
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

# Local
from ..input.data import Field
from ..utils.summarize import summarizable
from ..utils.typing import ColorType
from ..utils.typing import RectType
from .coord_trans import CoordinateTransformer
from .domain import Domain
from .proj_bbox import MapAxesProjections
from .ref_dist_indicator import RefDistIndConfig
from .ref_dist_indicator import ReferenceDistanceIndicator


# pylint: disable=E0213  # no-self-argument (validators)
# pylint: disable=?R0902  # too-many-instance-attributes
@summarizable
@dataclass
class MapAxesConfig:
    """Configuration of ``MapAxesPlot``.

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
    ref_dist_config: Optional[Union[RefDistIndConfig, Mapping[str, Any]]] = None
    ref_dist_on: bool = True
    scale_fact: float = 1.0

    def __post_init__(self) -> None:
        # geo_res*
        if self.geo_res_cities == "none":
            self.geo_res_cities = self.geo_res
        if self.geo_res_rivers == "none":
            self.geo_res_rivers = self.geo_res

        # ref_dist_config
        if self.ref_dist_config is None:
            self.ref_dist_config = RefDistIndConfig()
        elif isinstance(self.ref_dist_config, Mapping):
            # Passed a dict as argument; turn into ``RefDistIndConfig``
            self.ref_dist_config = RefDistIndConfig(**self.ref_dist_config)
        assert isinstance(self.ref_dist_config, RefDistIndConfig)
        self.ref_dist_config = self.ref_dist_config.scale(self.scale_fact)


# pylint: disable=R0902  # too-many-instance-attributes
@summarizable(
    attrs=["fig", "rect", "field", "config", "trans"],
    post_summarize=lambda self, summary: {
        **summary,
        "elements": self._summarized_elements,
    },
)
class MapAxes:
    """Map plot axes for regular lat/lon data."""

    def __init__(
        self,
        *,
        config: MapAxesConfig,
        field: Field,
        fig: Figure,
        rect: RectType,
    ) -> None:
        """Create an instance of ``MapAxes``.

        Args:
            fig: Figure to which to map axes is added.

            rect: Position of map plot panel in figure coordinates.

            field: Field.

            config: Map axes setup.

        """
        self.config: MapAxesConfig = config
        self.field: Field = field
        self.fig: Figure = fig
        self.rect: RectType = rect

        self.elements: List[Tuple[str, Any]] = []
        self._summarized_elements: List[Dict[str, Any]] = []

        self._water_color: ColorType = "lightskyblue"

        self.zorder: Dict[str, int]
        self._init_zorder()

        # SR_TMP <<< TODO Clean this up!
        def _create_ax(
            fig: Figure,
            rect: RectType,
            projs: MapAxesProjections,
            config: MapAxesConfig,
        ) -> Axes:
            """Initialize Axes."""
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="numpy.ufunc size changed",
                )
                ax: Axes = fig.add_axes(rect, projection=projs.map)
            ax.set_adjustable("datalim")
            ax.spines["geo"].set_edgecolor("none")

            # Set geographical extent
            ax.set_aspect("auto")
            domain = config.domain
            bbox = domain.get_bbox(ax, projs)
            ax.set_extent(bbox, projs.data)

            return ax

        projs: MapAxesProjections = field.get_projs()

        self.ax: Axes = _create_ax(self.fig, self.rect, projs, self.config)

        self.trans = CoordinateTransformer(
            trans_axes=self.ax.transAxes,
            trans_data=self.ax.transData,
            proj_geo=projs.geo,
            proj_map=projs.map,
            proj_data=projs.data,
        )

        self.ref_dist_box: Optional[ReferenceDistanceIndicator]
        self._init_ref_dist_box()

        self._ax_add_grid()
        self._ax_add_geography()
        self._ax_add_data_domain_outline()
        self._ax_add_frame()

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
            transform=self.trans.proj_data,
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
                "transform": f"{type(self.trans.proj_data).__name__} instance",
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
        transform = self.trans.proj_geo._as_mpl_transform(self.ax)
        # -> see https://stackoverflow.com/a/25421922/4419816
        handle = self.ax.annotate(
            s, xy=(p_lon, p_lat), xycoords=transform, zorder=zorder, **kwargs
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<TODO>)"  # SR_TODO

    def _init_zorder(self) -> None:
        """Determine zorder of unique plot elements, from low to high."""
        zorders_const = [
            "frames",
            "marker",
            "grid",
            "geo_upper",
            "fld",
            "geo_lower",
            "lowest",
        ][::-1]
        d0, dz = 1, 1
        self.zorder = {name: d0 + idx * dz for idx, name in enumerate(zorders_const)}

    def _init_ref_dist_box(self) -> None:
        """Initialize the reference distance indicator (if activated)."""
        if not self.config.ref_dist_on:
            self.ref_dist_box = None
        else:
            assert isinstance(self.config.ref_dist_config, RefDistIndConfig)  # mypy
            self.ref_dist_box = ReferenceDistanceIndicator(
                ax=self.ax,
                axes_to_geo=self.trans.axes_to_geo,
                config=self.config.ref_dist_config,
                zorder=self.zorder["grid"],
            )

    def _ax_add_grid(self) -> None:
        """Show grid lines on map."""
        gl = self.ax.gridlines(
            linestyle=":", linewidth=1, color="black", zorder=self.zorder["grid"]
        )
        gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180, 180, 2))
        gl.ylocator = mpl.ticker.FixedLocator(np.arange(-90, 90.1, 2))

    def _ax_add_geography(self) -> None:
        """Add geographic elements: coasts, countries, colors, etc."""
        self.ax.coastlines(resolution=self.config.geo_res)
        self.ax.patch.set_facecolor(self._water_color)
        self._ax_add_countries("lowest", rasterized=True)
        self._ax_add_lakes("lowest", rasterized=True)
        self._ax_add_rivers("lowest", rasterized=True)
        self._ax_add_countries("geo_upper", rasterized=True)
        self._ax_add_cities(rasterized=False)

    def _ax_add_countries(self, zorder_key: str, rasterized: bool = False) -> None:
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
                rasterized=rasterized,
            ),
            zorder=self.zorder[zorder_key],
            rasterized=rasterized,
        )

    def _ax_add_lakes(self, zorder_key: str, rasterized: bool = False) -> None:
        self.ax.add_feature(
            cartopy.feature.NaturalEarthFeature(
                category="physical",
                name="lakes",
                scale=self.config.geo_res,
                edgecolor="none",
                facecolor=self._water_color,
                rasterized=rasterized,
            ),
            zorder=self.zorder[zorder_key],
            rasterized=rasterized,
        )
        if self.config.geo_res == "10m":
            self.ax.add_feature(
                cartopy.feature.NaturalEarthFeature(
                    category="physical",
                    name="lakes_europe",
                    scale=self.config.geo_res,
                    edgecolor="none",
                    facecolor=self._water_color,
                    rasterized=rasterized,
                ),
                zorder=self.zorder[zorder_key],
                rasterized=rasterized,
            )

    def _ax_add_rivers(self, zorder_key: str, rasterized: bool = False) -> None:
        linewidth = {"lowest": 1, "geo_lower": 1, "geo_upper": 2 / 3}[zorder_key]
        # Note:
        #  - Bug in Cartopy with recent shapefiles triggers errors (NULL geometry)
        #    -> PR: https://github.com/SciTools/cartopy/pull/1411
        #  - Issue fixed in Cartopy 0.18.0

        major_rivers = cartopy.feature.NaturalEarthFeature(
            category="physical",
            name="rivers_lake_centerlines",
            scale=self.config.geo_res,
            edgecolor=self._water_color,
            facecolor=(0, 0, 0, 0),
            linewidth=linewidth,
            rasterized=rasterized,
        )
        self.ax.add_feature(major_rivers, zorder=self.zorder[zorder_key])

        if self.config.geo_res_rivers == "10m":
            minor_rivers = cartopy.feature.NaturalEarthFeature(
                category="physical",
                name="rivers_europe",
                scale=self.config.geo_res,
                edgecolor=self._water_color,
                facecolor=(0, 0, 0, 0),
                linewidth=linewidth,
                rasterized=rasterized,
            )
            self.ax.add_feature(minor_rivers, zorder=self.zorder[zorder_key])

    def _ax_add_cities(self, rasterized: bool = False) -> None:
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
            p_lon, p_lat = self.trans.proj_data.transform_point(
                p_lon, p_lat, self.trans.proj_geo, trap=True
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
                else is_in_box(pxa, pya, rdb.x0_box, rdb.x1_box, rdb.y0_box, rdb.y1_box)
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

        # src: https://www.naturalearthdata.com/downloads/50m-cultural-vectors/...
        # .../50m-populated-places/lk
        cities: Sequence[Record] = cartopy.io.shapereader.Reader(
            cartopy.io.shapereader.natural_earth(
                category="cultural",
                name="populated_places",
                resolution=self.config.geo_res_cities,
            )
        ).records()

        plot_domain = mpl.patches.Rectangle(
            xy=(0, 0), width=1.0, height=1.0, transform=self.ax.transAxes
        )
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
                    rasterized=rasterized,
                )
                text = self.add_text(
                    lon,
                    lat,
                    get_name(city),
                    va="center",
                    size=9 * self.config.scale_fact,
                    rasterized=rasterized,
                )
                # Note: `clip_on=True` doesn't work in cartopy v0.18
                text.set_clip_path(plot_domain)

    def _ax_add_data_domain_outline(self) -> None:
        """Add domain outlines to map plot."""
        lon0, lon1 = self.field.lon[[0, -1]]
        lat0, lat1 = self.field.lat[[0, -1]]
        xs = [lon0, lon1, lon1, lon0, lon0]
        ys = [lat0, lat0, lat1, lat1, lat0]
        self.ax.plot(xs, ys, transform=self.trans.proj_data, c="black", lw=1)

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

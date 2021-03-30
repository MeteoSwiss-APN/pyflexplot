"""Boxed plots."""
# Standard library
import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

# Local
from ..input.field import Field
from ..plot_layouts import BoxedPlotLayout
from ..setups.plot_setup import PlotSetup
from ..utils.exceptions import FieldAllNaNError
from ..utils.summarize import summarizable
from ..utils.typing import ColorType
from ..utils.typing import FontSizeType
from ..utils.typing import RectType
from .map_axes import MapAxes
from .map_axes import MapAxesConfig
from .text_box_axes import TextBoxAxes


@summarizable
@dataclass
class FontSizes:
    title_large: FontSizeType = 14.0
    title_medium: FontSizeType = 12.0
    title_small: FontSizeType = 12.0
    content_large: FontSizeType = 12.0
    content_medium: FontSizeType = 10.0
    content_small: FontSizeType = 9.0

    def scale(self, factor: float) -> "FontSizes":
        # pylint: disable=E1101  # no-member (__dataclass_fields__)
        params = list(self.__dataclass_fields__)  # type: ignore
        return type(self)(**{param: getattr(self, param) * factor for param in params})


@summarizable
@dataclass
class FontConfig:
    name: str = "Liberation Sans"
    sizes: FontSizes = FontSizes()


@summarizable
@dataclass
class ContourLevelsLegendConfig:
    range_align: str = "center"
    range_style: str = "base"
    range_widths: Tuple[int, int, int] = (5, 3, 5)
    rstrip_zeros: bool = True
    labels: List[str] = dataclasses.field(default_factory=list)


@summarizable
@dataclass
class ContourLevelsConfig:
    extend: str = "max"
    include_lower: bool = True
    legend: ContourLevelsLegendConfig = ContourLevelsLegendConfig()
    levels: Optional[np.ndarray] = None
    n: int = 0  # SR_TMP TODO eliminate
    scale: str = "log"


@summarizable
@dataclass
class MarkersConfig:
    markers: Optional[Dict[str, Dict[str, Any]]] = None
    mark_field_max: bool = True
    mark_release_site: bool = True


# pylint: disable=R0902  # too-many-instance-attributes (>7)
@summarizable
@dataclass
class BoxedPlotConfig:
    setup: PlotSetup  # SR_TODO consider removing this
    layout: BoxedPlotLayout
    # ---
    font: FontConfig = FontConfig()
    levels: ContourLevelsConfig = ContourLevelsConfig()
    markers: MarkersConfig = MarkersConfig()
    # ---
    colors: Optional[List[ColorType]] = None
    fig_size: Optional[Tuple[float, float]] = None
    labels: Dict[str, Any] = dataclasses.field(default_factory=dict)
    lw_frame: float = 1.0
    model_info: str = "N/A"


@summarizable(attrs=["config", "axes"])
class BoxedPlot:
    """A FLEXPART dispersion plot."""

    def __init__(
        self,
        config: BoxedPlotConfig,
    ) -> None:
        """Create an instance of ``BoxedPlot``."""
        self.config = config
        self.axes: Dict[str, Union[MapAxes, TextBoxAxes]] = {}
        self._fig: Optional[Figure] = None

    @property
    def fig(self) -> Figure:
        if self._fig is None:
            self._fig = plt.figure(figsize=self.config.fig_size)
        return self._fig

    def write(self, file_path: str) -> None:
        self.fig.savefig(
            file_path,
            facecolor=self.fig.get_facecolor(),
            edgecolor=self.fig.get_edgecolor(),
            bbox_inches="tight",
            pad_inches=0.15,
            dpi=90,
        )

    def clean(self) -> None:
        plt.close(self.fig)

    def add_map_plot(
        self, name: str, field: Field, map_config: MapAxesConfig, rect: RectType
    ) -> MapAxes:
        ax = MapAxes(
            config=map_config,
            field=field,
            fig=self.fig,
            rect=rect,
        )
        assert self.config.colors is not None  # SR_TMP
        _draw_colors_contours(
            ax,
            field,
            levels_config=self.config.levels,
            colors=self.config.colors,
        )
        _add_markers(ax, field, self.config.markers)
        self.axes[name] = ax
        return ax

    def add_text_box(
        self,
        name: str,
        rect: RectType,
        fill: Callable[[TextBoxAxes, "BoxedPlot"], None],
        *,
        frame_on: bool = True,
    ) -> TextBoxAxes:
        lw_frame: Optional[float] = self.config.lw_frame if frame_on else None
        box = TextBoxAxes(name=name, rect=rect, fig=self.fig, lw_frame=lw_frame)
        fill(box, self)
        box.draw()
        self.axes[name] = box
        return box


# pylint: disable=R0912  # too-many-branches
def _draw_colors_contours(
    ax: MapAxes,
    field: Field,
    levels_config: ContourLevelsConfig,
    colors: List[ColorType],
) -> None:
    arr = np.asarray(field.fld)
    levels = np.asarray(levels_config.levels)
    extend = levels_config.extend
    if levels_config.scale == "log":
        with np.errstate(divide="ignore"):
            arr = np.log10(np.where(arr > 0, arr, np.nan))
        levels = np.log10(levels)

    if not levels_config.include_lower:
        # Turn levels from lower- to upped-bound inclusive
        # by infitesimally increasing them
        levels = levels + np.finfo(np.float32).eps

    # Replace infs (apparently ignored by contourf)
    arr = np.where(np.isneginf(arr), np.finfo(np.float32).min, arr)
    arr = np.where(np.isposinf(arr), np.finfo(np.float32).max, arr)

    try:
        contours = ax.ax.contourf(
            field.lon,
            field.lat,
            arr,
            transform=ax.trans.proj_data,
            levels=levels,
            extend=extend,
            zorder=ax.zorder["fld"],
            colors=colors,
        )
    except ValueError as e:
        if str(e) == "'bboxes' cannot be empty":
            # Expected error when there are no contours to plot
            # (Easier to catch error than explicitly detect 'empty' array)
            return
        raise e
    else:
        for contour in contours.collections:
            contour.set_rasterized(True)


def _add_markers(ax: MapAxes, field: Field, markers_config: MarkersConfig) -> None:
    mdata = field.mdata
    if markers_config.mark_release_site:
        assert markers_config.markers is not None  # mypy
        ax.add_marker(
            p_lat=mdata.release.lat,
            p_lon=mdata.release.lon,
            **markers_config.markers["site"],
        )
    if markers_config.mark_field_max:
        assert markers_config.markers is not None  # mypy
        try:
            max_lat, max_lon = field.locate_max()
        except FieldAllNaNError:
            warnings.warn("skip maximum marker (all-nan field)")
        else:
            ax.add_marker(p_lat=max_lat, p_lon=max_lon, **markers_config.markers["max"])

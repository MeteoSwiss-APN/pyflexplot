"""Boxed plots."""
# Standard library
import dataclasses
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# Third-party
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

# Local
from ..data import Field
from ..meta_data import MetaData
from ..setup import Setup
from ..utils.summarize import summarizable
from ..utils.typing import ColorType
from ..utils.typing import FontSizeType
from ..utils.typing import RectType
from .map_axes import MapAxes
from .map_axes import MapAxesConfig
from .map_axes import post_summarize_plot
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
    ranges_align: str = "center"
    range_style: str = "base"
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
    setup: Setup  # SR_TODO consider removing this
    mdata: MetaData  # SR_TODO consider removing this
    # ---
    font: FontConfig = FontConfig()
    levels: ContourLevelsConfig = ContourLevelsConfig()
    markers: MarkersConfig = MarkersConfig()
    # ---
    colors: Optional[List[ColorType]] = None
    # SR_NOTE Figure size may change when boxes etc. are added
    # SR_TODO Specify plot size in a robust way (what you want is what you get)
    fig_size: Tuple[float, float] = (12.5, 8.0)
    labels: Dict[str, Any] = dataclasses.field(default_factory=dict)
    lw_frame: float = 1.0
    model_info: str = ""  # SR_TODO sensible default

    @property
    def fig_aspect(self):
        return np.divide(*self.fig_size)


@summarizable
@dataclass
class DummyBoxedPlot:
    """Dummy for dry runs."""


@summarizable(
    attrs=["ax_map", "boxes", "field", "fig", "map_config"],
    post_summarize=post_summarize_plot,
)
# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlot:
    """A FLEXPART dispersion plot."""

    def __init__(
        self, field: Field, config: BoxedPlotConfig, map_config: MapAxesConfig
    ) -> None:
        """Create an instance of ``BoxedPlot``."""
        self.field = field
        self.config = config
        self.map_config = map_config
        self.boxes: Dict[str, TextBoxAxes] = {}
        self._fig: Optional[Figure] = None
        self.ax_map: MapAxes  # SR_TMP TODO eliminate single centralized Map axes

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
        self.clean()

    def clean(self) -> None:
        plt.close(self.fig)

    def add_map_plot(self, rect: RectType) -> MapAxes:
        ax = MapAxes.create(self.map_config, fig=self.fig, rect=rect, field=self.field)
        self.ax_map = ax  # SR_TMP
        self._draw_colors_contours()
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
        self.boxes[name] = box
        return box

    # SR_TODO Pull out of BoxedPlot class to MapAxes or some MapAxesContent class
    # SR_TODO Replace checks with plot-specific config/setup object
    # pylint: disable=R0912  # too-many-branches
    def _draw_colors_contours(self) -> None:
        arr = np.asarray(self.field.fld)
        levels = np.asarray(self.config.levels.levels)
        colors = self.config.colors
        assert colors is not None  # SR_TMP
        extend = self.config.levels.extend
        if self.config.levels.scale == "log":
            with np.errstate(divide="ignore"):
                arr = np.log10(np.where(arr > 0, arr, np.nan))
            levels = np.log10(levels)

        if not self.config.levels.include_lower:
            # Turn levels from lower- to upped-bound inclusive
            # by infitesimally increasing them
            levels = levels + np.finfo(np.float32).eps

        # Replace infs (apparently ignored by contourf)
        arr = np.where(np.isneginf(arr), np.finfo(np.float32).min, arr)
        arr = np.where(np.isposinf(arr), np.finfo(np.float32).max, arr)

        try:
            contours = self.ax_map.ax.contourf(
                self.field.lon,
                self.field.lat,
                arr,
                transform=self.ax_map.proj_data,
                levels=levels,
                extend=extend,
                zorder=self.ax_map.zorder["fld"],
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

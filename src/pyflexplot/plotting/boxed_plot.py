# -*- coding: utf-8 -*-
"""
Boxed plots.
"""
# Standard library
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
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from pydantic import BaseModel

# Local
from ..data import Field
from ..data import FieldStats
from ..meta_data import MetaData
from ..setup import Setup
from ..utils.summarize import summarizable
from ..utils.typing import ColorType
from ..utils.typing import FontSizeType
from ..utils.typing import RectType
from .map_axes import MapAxes
from .map_axes import MapAxesConf
from .map_axes import post_summarize_plot
from .text_box_axes import TextBoxAxes


@dataclass
class FontSizes:
    title_large: FontSizeType = 14.0
    title_medium: FontSizeType = 12.0
    title_small: FontSizeType = 12.0
    content_large: FontSizeType = 12.0
    content_medium: FontSizeType = 10.0
    content_small: FontSizeType = 9.0


# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlotConfig(BaseModel):
    setup: Setup  # SR_TODO consider removing this
    mdata: MetaData  # SR_TODO consider removing this
    cmap: Union[str, Colormap] = "flexplot"
    colors: List[ColorType]
    color_above_closed: Optional[ColorType] = None  # SR_TODO better approach
    d_level: Optional[int] = None  # SR_TODO sensible default
    extend: str = "max"
    # SR_NOTE Figure size may change when boxes etc. are added
    # SR_TODO Specify plot size in a robust way (what you want is what you get)
    fig_size: Tuple[float, float] = (12.5, 8.0)
    font_name: str = "Liberation Sans"
    font_sizes: FontSizes = FontSizes()
    labels: Dict[str, Any] = {}
    legend_rstrip_zeros: bool = True
    level_ranges_align: str = "center"
    level_range_style: str = "base"
    levels_scale: str = "log"
    lw_frame: float = 1.0
    mark_field_max: bool = True
    mark_release_site: bool = True
    markers: Optional[Dict[str, Dict[str, Any]]] = None
    model_info: str = ""  # SR_TODO sensible default
    n_levels: Optional[int] = None  # SR_TODO sensible default

    class Config:  # noqa
        allow_extra = False
        arbitrary_types_allowed = True

    @property
    def fig_aspect(self):
        return np.divide(*self.fig_size)


@dataclass
class DummyBoxedPlot:
    """Dummy for dry runs."""


@summarizable(
    attrs=["ax_map", "boxes", "field", "fig", "map_conf"],
    post_summarize=post_summarize_plot,
)
# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlot:
    """A FLEXPART dispersion plot."""

    def __init__(
        self, field: Field, config: BoxedPlotConfig, map_conf: MapAxesConf
    ) -> None:
        """Create an instance of ``BoxedPlot``."""

        self.field = field
        self.config = config
        self.map_conf = map_conf

        self.boxes: Dict[str, TextBoxAxes] = {}
        self.levels: np.ndarray = levels_from_time_stats(
            self.config, self.field.time_props.stats
        )

        # Declarations
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
        )
        self.clean()

    def clean(self) -> None:
        plt.close(self.fig)

    def add_map_plot(self, rect: RectType,) -> MapAxes:
        axs = MapAxes.create(self.map_conf, fig=self.fig, rect=rect, field=self.field)
        self.ax_map = axs  # SR_TMP
        self._draw_colors_contours()
        return axs

    def add_text_box(
        self,
        name: str,
        rect: RectType,
        fill: Callable[[TextBoxAxes, "BoxedPlot"], None],
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
    def _draw_colors_contours(self) -> None:
        arr = self.field.fld
        levels = self.levels
        colors = self.config.colors
        extend = self.config.extend
        color_above_closed = self.config.color_above_closed
        if self.config.levels_scale == "log":
            with np.errstate(divide="ignore"):
                arr = np.log10(arr)
            levels = np.log10(levels)
        elif extend in ["none", "min"]:
            if color_above_closed:
                # Areas beyond the closed upper bound are colored black
                colors = list(colors) + [color_above_closed]
                if extend == "none":
                    extend = "max"
                elif extend == "min":
                    extend = "both"
        # SR_TMP < TODO move this logic out of here!
        if self.config.cmap != "flexplot":
            if extend in ["min", "both"]:
                levels = np.r_[-np.inf, levels]
            if extend in ["max", "both"]:
                levels = np.r_[levels, np.inf]
            extend = "none"
        # SR_TMP >

        arr = replace_infs(arr, levels)

        def nothing_to_plot(arr, levels):
            """Check if there's anything to plot.

            This allows one to exit before calling contourf if there's nothing
            to plot, and thus prevent an ugly error.
            """
            all_nan = np.nanmin(arr) == np.nanmax(arr)
            all_below = np.nanmax(arr) < levels.min() and extend not in ["min", "both"]
            all_above = np.nanmax(arr) > levels.max() and extend not in ["max", "both"]
            return all_nan or all_below or all_above

        if nothing_to_plot(arr, levels):
            return

        self.ax_map.ax.contourf(
            self.field.lon,
            self.field.lat,
            arr,
            transform=self.ax_map.proj_data,
            levels=levels,
            extend=extend,
            zorder=self.ax_map.zorder["fld"],
            colors=colors,
        )


def replace_infs(fld: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Replace inf by large values outside the data and level range.

    Reason: Contourf apparently ignores inf values.

    """
    large_value = 999.9
    vals = np.r_[fld.flatten(), levels]
    thr = vals[np.isfinite(vals)].max()
    max_ = np.finfo(np.float32).max
    while large_value <= thr:
        large_value = large_value * 10 + 0.9
        if large_value > max_:
            raise Exception("cannot derive large enough value", large_value, max_, thr)
    fld = np.where(np.isneginf(fld), -large_value, fld)
    fld = np.where(np.isposinf(fld), large_value, fld)
    return fld


def levels_from_time_stats(
    plot_config: BoxedPlotConfig, time_stats: FieldStats
) -> np.ndarray:
    def _auto_levels_log10(n_levels: int, val_max: float) -> List[float]:
        if not np.isfinite(val_max):
            raise ValueError("val_max not finite", val_max)
        log10_max = int(np.floor(np.log10(val_max)))
        log10_d = 1
        return 10 ** np.arange(
            log10_max - (n_levels - 1) * log10_d, log10_max + 0.5 * log10_d, log10_d
        )

    assert plot_config.n_levels is not None  # mypy
    if plot_config.setup.get_simulation_type() == "ensemble":
        if plot_config.setup.core.ens_variable.endswith("probability"):
            assert plot_config.d_level is not None  # mypy
            n_max = 90
            return np.arange(
                n_max - plot_config.d_level * (plot_config.n_levels - 1),
                n_max + plot_config.d_level,
                plot_config.d_level,
            )
        elif plot_config.setup.core.ens_variable in [
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            assert plot_config.d_level is not None  # mypy
            return np.arange(0, plot_config.n_levels) * plot_config.d_level
    elif plot_config.setup.core.plot_variable == "affected_area_mono":
        levels = _auto_levels_log10(n_levels=9, val_max=time_stats.max)
        return np.array([levels[0], np.inf])
    return _auto_levels_log10(plot_config.n_levels, val_max=time_stats.max)

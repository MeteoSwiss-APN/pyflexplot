# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.colors import LinearSegmentedColormap
from pydantic import BaseModel

# Local
from .data import Field
from .meta_data import MetaData
from .plot_elements import MapAxes
from .plot_elements import MapAxesConf
from .plot_elements import TextBoxAxes
from .plot_elements import post_summarize_plot
from .setup import InputSetup
from .summarize import summarizable
from .typing import ColorType
from .typing import FontSizeType
from .typing import RectType


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
    setup: InputSetup  # SR_TODO consider removing this
    mdata: MetaData  # SR_TODO consider removing this
    cmap: Union[str, Colormap] = "flexplot"
    colors: List[ColorType]
    d_level: Optional[int] = None  # SR_TODO sensible default
    draw_colors: bool = True
    draw_contours: bool = False
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
# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlotLayout:
    aspect: float
    x0_tot: float = 0.0
    x1_tot: float = 1.0
    y0_tot: float = 0.0
    y1_tot: float = 1.0
    y_pad: float = 0.02
    h_top: float = 0.08
    w_right: float = 0.2
    h_rigtop: float = 0.15
    h_rigbot: float = 0.42
    h_bottom: float = 0.05

    def __post_init__(self):
        self.x_pad: float = self.y_pad / self.aspect
        self.h_tot: float = self.y1_tot - self.y0_tot
        self.y1_top: float = self.y1_tot
        self.y0_top: float = self.y1_top - self.h_top
        self.y1_center: float = self.y0_top - self.y_pad
        self.y0_center: float = self.y0_tot + self.h_bottom
        self.h_center: float = self.y1_center - self.y0_center
        self.w_tot: float = self.x1_tot - self.x0_tot
        self.x1_right: float = self.x1_tot
        self.x0_right: float = self.x1_right - self.w_right
        self.x1_left: float = self.x0_right - self.x_pad
        self.x0_left: float = self.x0_tot
        self.w_left: float = self.x1_left - self.x0_left
        self.y0_rigbot: float = self.y0_center
        self.y1_rigtop: float = self.y1_tot
        self.y0_rigtop: float = self.y1_rigtop - self.h_rigtop
        self.y1_rigbot: float = self.y0_rigbot + self.h_rigbot
        self.y1_rigmid: float = self.y0_rigtop - self.y_pad
        self.y0_rigmid: float = self.y1_rigbot + self.y_pad
        self.h_rigmid: float = self.y1_rigmid - self.y0_rigmid
        self.rect_top: RectType = (self.x0_left, self.y0_top, self.w_left, self.h_top)
        self.rect_center: RectType = (
            self.x0_left,
            self.y0_center,
            self.w_left,
            self.h_center,
        )
        self.rect_right_top: RectType = (
            self.x0_right,
            self.y0_rigtop,
            self.w_right,
            self.h_rigtop,
        )
        self.rect_right_middle: RectType = (
            self.x0_right,
            self.y0_rigmid,
            self.w_right,
            self.h_rigmid,
        )
        self.rect_right_bottom: RectType = (
            self.x0_right,
            self.y0_rigbot,
            self.w_right,
            self.h_rigbot,
        )
        self.rect_bottom: RectType = (
            self.x0_tot,
            self.y0_tot,
            self.w_tot,
            self.h_bottom,
        )


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
        self.layout = BoxedPlotLayout(aspect=self.config.fig_aspect)
        self.levels = levels_from_time_stats(self.config, self.field.time_stats)
        self.fig = plt.figure(figsize=self.config.fig_size)
        self.ax_map = MapAxes.create(
            self.map_conf, fig=self.fig, rect=self.layout.rect_center, field=self.field,
        )
        self._draw_colors_contours()

    def save(self, file_path: str, *, write: bool = True) -> None:
        """Save the plot to disk.

        Args:
            file_path: Output file name, incl. path.

            write (optional): Whether to actually write the plot to disk.

        """
        if write:
            self.fig.savefig(
                file_path,
                facecolor=self.fig.get_facecolor(),
                edgecolor=self.fig.get_edgecolor(),
                bbox_inches="tight",
                pad_inches=0.15,
            )
        plt.close(self.fig)

    # SR_TODO Replace checks with plot-specific config/setup object
    def _draw_colors_contours(self) -> None:
        arr = self.field.fld
        levels = self.levels
        colors = self.config.colors
        extend = self.config.extend
        if self.config.levels_scale == "log":
            arr = np.log10(arr)
            levels = np.log10(levels)
        elif extend in ["none", "min"]:
            # Areas beyond the closed upper bound are colored black
            colors = list(colors) + ["black"]
            extend = {"none": "max", "min": "both"}[extend]
        if self.config.draw_colors:
            self.ax_map.contourf(arr, levels=levels, colors=colors, extend=extend)
            # SR_TMP <
            cmap_black = LinearSegmentedColormap.from_list("black", ["black"])
            self.ax_map.contourf(arr, levels=np.array([-0.01, 0.01]), colors=cmap_black)
            # SR_TMP >
        if self.config.draw_contours:
            self.ax_map.contour(arr, levels=levels, colors="black", linewidths=1)

    def add_text_box(
        self,
        name: str,
        rect: RectType,
        fill: Callable[[TextBoxAxes, "BoxedPlot"], None],
        frame_on: bool = True,
    ) -> None:
        lw_frame: Optional[float] = self.config.lw_frame if frame_on else None
        box = TextBoxAxes(name=name, rect=rect, fig=self.fig, lw_frame=lw_frame)
        fill(box, self)
        box.draw()
        self.boxes[name] = box

    def add_marker(self, lat: float, lon: float, marker: str, **kwargs) -> None:
        self.ax_map.marker(lat=lat, lon=lon, marker=marker, **kwargs)


def levels_from_time_stats(
    plot_config: BoxedPlotConfig, time_stats: Mapping[str, float]
) -> List[float]:
    def _auto_levels_log10(n_levels: int, val_max: float) -> List[float]:
        if not np.isfinite(val_max):
            raise ValueError("val_max not finite", val_max)
        log10_max = int(np.floor(np.log10(val_max)))
        log10_d = 1
        return 10 ** np.arange(
            log10_max - (n_levels - 1) * log10_d, log10_max + 0.5 * log10_d, log10_d
        )

    assert plot_config.n_levels is not None  # mypy
    if plot_config.setup.plot_type.startswith(
        "ensemble_"
    ) and plot_config.setup.plot_type.endswith("_probability"):
        assert plot_config.d_level is not None  # mypy
        n_max = 90
        return np.arange(
            n_max - plot_config.d_level * (plot_config.n_levels - 1),
            n_max + plot_config.d_level,
            plot_config.d_level,
        )
    elif plot_config.setup.plot_type in [
        "ensemble_cloud_arrival_time",
        "ensemble_cloud_departure_time",
    ]:
        assert plot_config.d_level is not None  # mypy
        return np.arange(0, plot_config.n_levels) * plot_config.d_level
    elif plot_config.setup.plot_variable == "affected_area_mono":
        levels = _auto_levels_log10(n_levels=9, val_max=time_stats["max"])
        return np.array([levels[0], np.inf])
    else:
        return _auto_levels_log10(plot_config.n_levels, val_max=time_stats["max"])

# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

# Third-party
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Local
from .data import Field
from .plot_lib import MapAxes
from .plot_lib import MapAxesConf
from .plot_lib import TextBoxAxes
from .plot_lib import post_summarize_plot
from .plot_types import PlotConfig
from .plot_types import PlotLayout
from .plot_types import levels_from_time_stats
from .summarize import summarizable

# Custom types
RectType = Tuple[float, float, float, float]


@summarizable(
    attrs=["ax_map", "boxes", "field", "fig", "map_conf"],
    post_summarize=post_summarize_plot,
)
# pylint: disable=R0902  # too-many-instance-attributes
class Plot:
    """A FLEXPART dispersion plot."""

    def __init__(self, field: Field, config: PlotConfig, map_conf: MapAxesConf) -> None:
        """Create an instance of ``Plot``."""
        self.field = field
        self.config = config
        self.map_conf = map_conf
        self.boxes: Dict[str, TextBoxAxes] = {}
        self.levels: np.ndarray = levels_from_time_stats(
            self.config, self.field.time_stats
        )
        self.layout = PlotLayout(aspect=self.config.fig_aspect)
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
        fill: Callable[[TextBoxAxes, "Plot"], None],
        frame_on: bool = True,
    ) -> None:
        lw_frame: Optional[float] = self.config.lw_frame if frame_on else None
        box = TextBoxAxes(name=name, rect=rect, fig=self.fig, lw_frame=lw_frame)
        fill(box, self)
        box.draw()
        self.boxes[name] = box

    def add_marker(self, lat: float, lon: float, marker: str, **kwargs) -> None:
        self.ax_map.marker(lat=lat, lon=lon, marker=marker, **kwargs)

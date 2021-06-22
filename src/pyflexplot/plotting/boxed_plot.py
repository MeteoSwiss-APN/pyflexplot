"""Boxed plots."""
# Standard library
import dataclasses as dc
import warnings
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

# Local
from ..input.field import Field
from ..input.field import FieldGroup
from ..plot_layouts import BoxedPlotLayout
from ..setups.plot_panel_setup import PlotPanelSetup
from ..setups.plot_setup import PlotSetup
from ..utils.exceptions import FieldAllNaNError
from ..utils.summarize import summarizable
from ..utils.typing import ColorType
from ..utils.typing import FontSizeType
from ..utils.typing import RectType
from .domain import Domain
from .map_axes import MapAxes
from .map_axes import MapAxesConfig
from .text_box_axes import TextBoxAxes


@summarizable
@dc.dataclass
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
@dc.dataclass
class FontConfig:
    name: str = "Liberation Sans"
    sizes: FontSizes = dc.field(default_factory=FontSizes)


@summarizable
@dc.dataclass
class ContourLevelsLegendConfig:
    range_align: str = "center"
    range_style: str = "base"
    range_widths: Tuple[int, int, int] = (5, 3, 5)
    rstrip_zeros: bool = True
    labels: List[str] = dc.field(default_factory=list)


@summarizable
@dc.dataclass
class ContourLevelsConfig:
    extend: str = "max"
    include_lower: bool = True
    legend: ContourLevelsLegendConfig = dc.field(
        default_factory=ContourLevelsLegendConfig
    )
    levels: Optional[np.ndarray] = None
    n: int = 0  # SR_TMP TODO eliminate
    scale: str = "log"


@summarizable
@dc.dataclass
class MarkersConfig:
    markers: Optional[Dict[str, Dict[str, Any]]] = None
    mark_field_max: bool = True
    mark_release_site: bool = True


@summarizable
@dc.dataclass
class BoxedPlotPanelConfig:
    setup: PlotPanelSetup  # SR_TODO consider removing this
    # ---
    colors: Sequence[ColorType] = dc.field(default_factory=list)
    label: Optional[str] = None
    levels: ContourLevelsConfig = dc.field(default_factory=ContourLevelsConfig)
    markers: MarkersConfig = dc.field(default_factory=MarkersConfig)


# pylint: disable=R0902  # too-many-instance-attributes (>7)
@summarizable
@dc.dataclass
class BoxedPlotConfig:
    setup: PlotSetup  # SR_TODO consider removing this
    layout: BoxedPlotLayout
    panels: Sequence[BoxedPlotPanelConfig]
    # ---
    font: FontConfig = dc.field(default_factory=FontConfig)
    labels: Dict[str, Any] = dc.field(default_factory=dict)
    # ---
    fig_size: Optional[Tuple[float, float]] = None
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

    def write(self, file_path: Union[Path, str]) -> None:
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

    def add_map_plot_panels(
        self,
        field_group: FieldGroup,
        domains: Sequence[Domain],
        map_configs: List[MapAxesConfig],
    ) -> None:
        n_panels = len(field_group)
        # SR_TMP < TODO proper check
        assert n_panels == len(domains) == len(map_configs) == len(self.config.panels)
        # SR_TMP >
        if n_panels == 1:
            field = next(iter(field_group))
            panel_config = next(iter(self.config.panels))
            domain = next(iter(domains))
            map_config = next(iter(map_configs))
            rect = self.config.layout.get_rect("center")
            self.add_map_plot("map", field, domain, map_config, rect, panel_config)
        elif n_panels == 4:
            # SR_TODO Define sub-rects in layout (new layout type with four panels)
            def derive_sub_rects(
                rect: RectType,
            ) -> Tuple[RectType, RectType, RectType, RectType]:
                x0, y0, w, h = rect
                rel_pad = 0.025
                w_pad = rel_pad * w
                h_pad = w_pad  # SR_TMP
                w = 0.5 * w - 0.5 * w_pad
                h = 0.5 * h - 0.5 * h_pad
                x1 = x0 + w + w_pad
                y1 = y0 + h + h_pad
                return ((x0, y1, w, h), (x1, y1, w, h), (x0, y0, w, h), (x1, y0, w, h))

            sub_rects = derive_sub_rects(self.config.layout.get_rect("center"))
            for idx, (field, panel_config, domain, map_config, sub_rect) in enumerate(
                zip(field_group, self.config.panels, domains, map_configs, sub_rects)
            ):
                name = f"map{idx}"
                self.add_map_plot(
                    name, field, domain, map_config, sub_rect, panel_config
                )
        else:
            raise NotImplementedError(f"{n_panels} number of panels")

    # pylint: disable=R0913  # too-many-arguments (>5)
    def add_map_plot(
        self,
        name: str,
        field: Field,
        domain: Domain,
        map_config: MapAxesConfig,
        rect: RectType,
        panel_config: BoxedPlotPanelConfig,
    ) -> MapAxes:
        ax = MapAxes(
            config=map_config,
            field=field,
            domain=domain,
            fig=self.fig,
            rect=rect,
        )
        if panel_config.label:
            _add_panel_label(ax, panel_config, self.config.layout)
        _draw_colors_contours(
            ax,
            field,
            levels_config=panel_config.levels,
            colors=panel_config.colors,
        )
        _add_markers(ax, field, panel_config.markers)
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


def _add_panel_label(
    ax: MapAxes, panel_config: BoxedPlotPanelConfig, layout: BoxedPlotLayout
) -> None:
    multipanel_param = layout.setup.multipanel_param
    height = 0.10
    if multipanel_param == "ens_params.pctl":
        width = 0.10
    elif multipanel_param == "ens_variable":
        width = 0.20
    elif multipanel_param == "time":
        width = 0.42
    else:
        raise NotImplementedError(f"label for multipanel param '{multipanel_param}'")
    zorder = ax.zorder["frames"]
    ax.ax.add_patch(
        mpl.patches.Rectangle(
            xy=(0.0, 1.0 - height),
            width=width,
            height=height,
            transform=ax.ax.transAxes,
            zorder=zorder,
            fill=True,
            facecolor="white",
            edgecolor="black",
            linewidth=1,
        )
    )
    ax.ax.text(
        x=0.5 * width,
        y=1.0 - 0.55 * height,
        s=panel_config.label,
        transform=ax.ax.transAxes,
        zorder=zorder,
        ha="center",
        va="center",
        fontsize=12,
    )


# pylint: disable=R0912  # too-many-branches
def _draw_colors_contours(
    ax: MapAxes,
    field: Field,
    levels_config: ContourLevelsConfig,
    colors: Sequence[ColorType],
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

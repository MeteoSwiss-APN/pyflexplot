# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import re
from textwrap import dedent
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

# Third-party
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# First-party
from srutils.geo import Degrees
from srutils.iter import isiterable

# Local
# TextBoxAxes Local
from .data import Field
from .formatting import format_level_ranges
from .meta_data import MetaData
from .plot_lib import MapAxes
from .plot_lib import MapAxesConf
from .plot_lib import TextBoxAxes
from .plot_lib import post_summarize_plot
from .plot_types import PlotConfig
from .plot_types import PlotLayout
from .plot_types import colors_from_plot_config
from .plot_types import create_map_conf
from .plot_types import create_plot_config
from .plot_types import levels_from_time_stats
from .summarize import summarizable
from .words import SYMBOLS
from .words import WORDS

# Custom types
RectType = Tuple[float, float, float, float]


def fill_box_top(box: TextBoxAxes, plot: "Plot") -> None:
    for position, label in plot.config.labels.get("top", {}).items():
        if position == "tl":
            font_size = plot.config.font_sizes.title_large
        else:
            font_size = plot.config.font_sizes.content_large
        box.text(label, loc=position, fontname=plot.config.font_name, size=font_size)


def fill_box_right_top(box: TextBoxAxes, plot: "Plot") -> None:
    try:
        lines = plot.config.labels["right_top"]["lines"]
    except KeyError:
        return
    else:
        box.text_block_hfill(
            lines,
            # dy_unit=-4.0,
            dy_unit=0.0,
            dy_line=2.5,
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_small,
        )


# pylint: disable=R0912   # too-many-branches
# pylint: disable=R0913   # too-many-arguments
# pylint: disable=R0914   # too-many-locals
# pylint: disable=R0915   # too-many-statements
def fill_box_right_middle(box: TextBoxAxes, plot: "Plot") -> None:
    """Fill the top box to the right of the map plot."""

    labels = plot.config.labels["right_middle"]

    # Box title
    box.text(
        labels["title_unit"],
        loc="tc",
        fontname=plot.config.font_name,
        size=plot.config.font_sizes.title_small,
    )

    dy_line: float = 3.0
    w_box: float = 4.0
    h_box: float = 2.0

    dx_box: float = -10
    dx_label: float = -3

    dx_marker: float = dx_box + 0.5 * w_box
    dx_marker_label: float = dx_label

    # Vertical position of legend (depending on number of levels)
    dy0_labels = -5.0
    dy0_boxes = dy0_labels - 0.8 * h_box

    # Format level ranges (contour plot legend)
    legend_labels = format_level_ranges(
        levels=plot.levels,
        style=plot.config.level_range_style,
        extend=plot.config.extend,
        rstrip_zeros=plot.config.legend_rstrip_zeros,
        align=plot.config.level_ranges_align,
    )

    # Legend labels (level ranges)
    box.text_block(
        legend_labels,
        loc="tc",
        dy_unit=dy0_labels,
        dy_line=dy_line,
        dx=dx_label,
        ha="left",
        fontname=plot.config.font_name,
        size=plot.config.font_sizes.content_medium,
        family="monospace",
    )

    # Legend color boxes
    colors = colors_from_plot_config(plot.config)
    dy = dy0_boxes
    for color in colors:
        box.color_rect(
            loc="tc",
            x_anker="left",
            dx=dx_box,
            dy=dy,
            w=w_box,
            h=h_box,
            fc=color,
            ec="black",
            lw=1.0,
        )
        dy -= dy_line

    dy0_markers = dy0_boxes - dy_line * (len(legend_labels) - 0.3)
    dy0_marker = dy0_markers

    # Release site marker
    if plot.config.mark_release_site:
        dy_site_label = dy0_marker
        dy0_marker -= dy_line
        dy_site_marker = dy_site_label - 0.7
        box.marker(
            loc="tc", dx=dx_marker, dy=dy_site_marker, **plot.config.markers["site"],
        )
        box.text(
            s=labels["release_site"],
            loc="tc",
            dx=dx_marker_label,
            dy=dy_site_label,
            ha="left",
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_medium,
        )

    # Field maximum marker
    if plot.config.mark_field_max:
        dy_marker_label_max = dy0_marker
        dy0_marker -= dy_line
        dy_max_marker = dy_marker_label_max - 0.7
        box.marker(
            loc="tc", dx=dx_marker, dy=dy_max_marker, **plot.config.markers["max"],
        )
        s = f"{labels['max']}: "
        if np.isnan(plot.field.fld).all():
            s += "NaN"
        else:
            fld_max = np.nanmax(plot.field.fld)
            if 0.001 <= fld_max < 0.01:
                s += f"{fld_max:.5f}"
            elif 0.01 <= fld_max < 0.1:
                s += f"{fld_max:.4f}"
            elif 0.1 <= fld_max < 1:
                s += f"{fld_max:.3f}"
            elif 1 <= fld_max < 10:
                s += f"{fld_max:.2f}"
            elif 10 <= fld_max < 100:
                s += f"{fld_max:.1f}"
            elif 100 <= fld_max < 1000:
                s += f"{fld_max:.0f}"
            else:
                s += f"{fld_max:.2E}"
        box.text(
            s=s,
            loc="tc",
            dx=dx_marker_label,
            dy=dy_marker_label_max,
            ha="left",
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_medium,
        )


def fill_box_right_bottom(box: TextBoxAxes, plot: "Plot") -> None:
    """Fill the bottom box to the right of the map plot."""

    labels = plot.config.labels["right_bottom"]
    mdata = plot.config.mdata

    # Box title
    box.text(
        s=labels["title"],
        loc="tc",
        fontname=plot.config.font_name,
        size=plot.config.font_sizes.title_small,
    )

    # Release site coordinates
    lat = Degrees(mdata.release_site_lat.value)
    lon = Degrees(mdata.release_site_lon.value)
    lat_deg = labels["lat_deg_fmt"].format(d=lat.degs(), m=lat.mins(), f=lat.frac())
    lon_deg = labels["lon_deg_fmt"].format(d=lon.degs(), m=lon.mins(), f=lon.frac())

    # SR_TMP < TODO clean this up, especially for ComboMetaData (units messed up)!
    height = mdata.format("release_height", add_unit=True)
    rate = mdata.format("release_rate", add_unit=True)
    mass = mdata.format("release_mass", add_unit=True)
    substance = mdata.format("species_name", join_combo=" / ")
    half_life = mdata.format("species_half_life", add_unit=True)
    deposit_vel = mdata.format("species_deposit_vel", add_unit=True)
    sediment_vel = mdata.format("species_sediment_vel", add_unit=True)
    washout_coeff = mdata.format("species_washout_coeff", add_unit=True)
    washout_exponent = mdata.format("species_washout_exponent")
    # SR_TMP >

    info_blocks = dedent(
        f"""\
        {labels['site']}:\t{mdata.release_site_name}
        {labels['latitude']}:\t{lat_deg}
        {labels['longitude']}:\t{lon_deg}
        {labels['height']}:\t{height}

        {labels['start']}:\t{mdata.release_start}
        {labels['end']}:\t{mdata.release_end}
        {labels['rate']}:\t{rate}
        {labels['mass']}:\t{mass}

        {labels['name']}:\t{substance}
        {labels['half_life']}:\t{half_life}
        {labels['deposit_vel']}:\t{deposit_vel}
        {labels['sediment_vel']}:\t{sediment_vel}
        {labels['washout_coeff']}:\t{washout_coeff}
        {labels['washout_exponent']}:\t{washout_exponent}
        """
    )

    # Add lines bottom-up (to take advantage of baseline alignment)
    box.text_blocks_hfill(
        info_blocks,
        dy_unit=-4.0,
        dy_line=2.5,
        fontname=plot.config.font_name,
        size=plot.config.font_sizes.content_small,
    )


def fill_box_bottom(box: TextBoxAxes, plot: "Plot") -> None:
    """Fill the box to the bottom of the map plot."""

    labels = plot.config.labels["bottom"]

    # FLEXPART/model info
    s = plot.config.model_info
    box.text(
        s=s,
        loc="tl",
        dx=-0.7,
        dy=0.5,
        fontname=plot.config.font_name,
        size=plot.config.font_sizes.content_small,
    )

    # MeteoSwiss Copyright
    box.text(
        s=labels["copyright"],
        loc="tr",
        dx=0.7,
        dy=0.5,
        fontname=plot.config.font_name,
        size=plot.config.font_sizes.content_small,
    )


@summarizable(
    attrs=["ax_map", "boxes", "field", "fig", "map_conf"],
    post_summarize=post_summarize_plot,
)
# pylint: disable=R0902  # too-many-instance-attributes
class Plot:
    """A FLEXPART dispersion plot."""

    def __init__(self, field: Field, config: PlotConfig, map_conf: MapAxesConf):
        """Create an instance of ``Plot``."""
        self.field = field
        self.config = config
        self.map_conf = map_conf
        self.boxes: Dict[str, TextBoxAxes] = {}
        self.levels: np.ndarray = levels_from_time_stats(
            self.config, self.field.time_stats
        )
        self.create_plot()

    def save(self, file_path: str, *, write: bool = True):
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

    def create_plot(self):
        layout = PlotLayout(aspect=self.config.fig_aspect)
        self.fig = plt.figure(figsize=self.config.fig_size)
        self.ax_map = MapAxes.create(
            self.map_conf, fig=self.fig, rect=layout.rect_center, field=self.field,
        )
        self._add_text_boxes(layout)
        self._draw_colors_contours()
        self._add_markers()

    def _add_markers(self):
        if self.config.mark_release_site:
            self.ax_map.marker(
                self.config.mdata.release_site_lon.value,
                self.config.mdata.release_site_lat.value,
                **self.config.markers["site"],
            )

        if self.config.mark_field_max:
            self.ax_map.mark_max(self.field.fld, **self.config.markers["max"])

    # SR_TODO Replace checks with plot-specific config/setup object
    def _draw_colors_contours(self):
        arr = self.field.fld
        levels = self.levels
        colors = colors_from_plot_config(self.config)
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

    # pylint: disable=R0914  # too-many-locals
    def _add_text_boxes(self, layout):
        self.add_text_box("top", layout.rect_top, fill_box_top)
        self.add_text_box("right_top", layout.rect_right_top, fill_box_right_top)
        self.add_text_box(
            "right_middle", layout.rect_right_middle, fill_box_right_middle
        )
        self.add_text_box(
            "right_bottom", layout.rect_right_bottom, fill_box_right_bottom
        )
        self.add_text_box("bottom", layout.rect_bottom, fill_box_bottom, frame_on=False)

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


def plot_fields(
    fields: Sequence[Field],
    mdata_lst: Sequence[MetaData],
    dry_run: bool = False,
    *,
    write: bool = True,
):

    # Create plots one-by-one
    for field, mdata in zip(fields, mdata_lst):
        setup = field.var_setups.compress()
        out_file_path = format_out_file_path(setup)
        map_conf = create_map_conf(field)

        if dry_run:
            plot = None
        else:
            assert mdata is not None  # mypy
            config = create_plot_config(setup, WORDS, SYMBOLS, mdata)
            plot = Plot(field, config, map_conf)
            plot.save(out_file_path, write=write)

        yield out_file_path, plot


# pylint: disable=W0102  # dangerious-default-value ([])
def format_out_file_path(setup, _previous=[]):
    template = setup.outfile
    path = _format_out_file_path_core(template, setup)
    while path in _previous:
        template = derive_unique_path(template)
        path = _format_out_file_path_core(template, setup)
    _previous.append(path)
    return path


def _format_out_file_path_core(template, setup):

    variable = setup.variable
    if setup.variable == "deposition":
        variable += f"_{setup.deposition_type}"

    kwargs = {
        "nageclass": setup.nageclass,
        "domain": setup.domain,
        "lang": setup.lang,
        "level": setup.level,
        "noutrel": setup.noutrel,
        "species_id": setup.species_id,
        "time": setup.time,
        "variable": variable,
    }

    # Format the file path
    # Don't use str.format in order to handle multival elements
    path = _replace_format_keys(template, kwargs)

    return path


def _replace_format_keys(path, kwargs):
    for key, val in kwargs.items():
        if not isiterable(val, str_ok=False):
            val = [val]
        # Iterate over relevant format keys
        rxs = r"{" + key + r"(:[^}]*)?}"
        re.finditer(rxs, path)
        for m in re.finditer(rxs, path):

            # Obtain format specifier (if there is one)
            try:
                f = m.group(1)
            except IndexError:
                f = None
            if not f:
                f = ""

            # Format the string that replaces this format key in the path
            formatted_key = "+".join([f"{{{f}}}".format(v) for v in val])

            # Replace format key in the path by the just formatted string
            start, end = path[: m.span()[0]], path[m.span()[1] :]
            path = f"{start}{formatted_key}{end}"

    # Check that all keys have been formatted
    if "{" in path or "}" in path:
        raise Exception(
            f"formatted output file path still contains format keys", path,
        )

    return path


def derive_unique_path(path):
    """Add/increment a trailing number to a file path."""

    # Extract suffix
    if path.endswith(".png"):
        suffix = ".png"
    else:
        raise NotImplementedError(f"unknown suffix: {path}")
    path_base = path[: -len(suffix)]

    # Reuse existing numbering if present
    match = re.search(r"-(?P<i>[0-9]+)$", path_base)
    if match:
        i = int(match.group("i")) + 1
        w = len(match.group("i"))
        path_base = path_base[: -w - 1]
    else:
        i = 1
        w = 1

    # Add numbering and suffix
    path = path_base + f"-{{i:0{w}}}{suffix}".format(i=i)

    return path

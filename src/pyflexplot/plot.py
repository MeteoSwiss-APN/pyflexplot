# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict
from typing import Sequence

# Third-party
import numpy as np
from matplotlib import pyplot as plt

# First-party
from srutils.geo import Degrees
from srutils.iter import isiterable

# Local
# TextBoxAxes Local
from .data import Field
from .meta_data import MetaData
from .plot_lib import MapAxes
from .plot_lib import MapAxesConf
from .plot_lib import TextBoxAxes
from .plot_lib import post_summarize_plot
from .plot_types import PlotConfig
from .plot_types import colors_from_plot_config
from .plot_types import create_map_conf
from .plot_types import create_plot_config
from .plot_types import levels_from_time_stats
from .utils import format_level_ranges
from .utils import summarizable
from .words import SYMBOLS
from .words import WORDS


@summarizable(
    attrs=["ax_map", "boxes", "field", "fig", "map_conf"],
    post_summarize=post_summarize_plot,
)
# pylint: disable=R0902  # too-many-instance-attributes
class Plot:
    """A FLEXPART dispersion plot."""

    def __init__(self, field: Field, plot_config: PlotConfig, map_conf: MapAxesConf):
        """Create an instance of ``Plot``."""
        self.field = field
        self.plot_config = plot_config
        self.map_conf = map_conf

        # Formatting arguments
        self._max_marker_kwargs = {
            "marker": "+",
            "color": "black",
            "markersize": 10,
            "markeredgewidth": 1.5,
        }
        self._site_marker_kwargs = {
            "marker": "^",
            "markeredgecolor": "red",
            "markerfacecolor": "white",
            "markersize": 7.5,
            "markeredgewidth": 1.5,
        }

        # Declare attributes
        self.boxes: Dict[str, TextBoxAxes]

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
        """Create plot."""

        # Prepare plot
        layout = PlotLayout(aspect=self.plot_config.fig_aspect)
        self.fig = plt.figure(figsize=self.plot_config.fig_size)
        self.ax_map = MapAxes.create(
            self.map_conf,
            fig=self.fig,
            rect=layout.rect_center_left(),
            field=self.field,
        )
        self.add_text_boxes(layout)
        self.draw_map_plot()
        self.draw_boxes()

    def fld_nonzero(self):
        return np.where(self.field.fld == 0, np.nan, self.field.fld)

    def draw_map_plot(self):
        """Plot the particle concentrations onto the map."""

        # Plot concentrations
        self.draw_colors_contours()

        if self.plot_config.mark_release_site:
            # Marker at release site
            self.ax_map.marker(
                self.plot_config.mdata.release_site_lon.value,
                self.plot_config.mdata.release_site_lat.value,
                **self._site_marker_kwargs,
            )

        if self.plot_config.mark_field_max:
            # Marker at location of maximum value
            self.ax_map.mark_max(self.field.fld, **self._max_marker_kwargs)

    # SR_TODO Replace checks with plot-specific config/setup object
    def draw_colors_contours(self):
        field = self.fld_nonzero()
        levels = levels_from_time_stats(self.plot_config, self.field.time_stats)
        colors = colors_from_plot_config(self.plot_config)
        extend = self.plot_config.extend
        if self.plot_config.levels_scale == "log":
            field = np.log10(field)
            levels = np.log10(levels)
        elif extend in ["none", "min"]:
            # Areas beyond the closed upper bound are colored black
            colors = list(colors) + ["black"]
            extend = {"none": "max", "min": "both"}[extend]
        if self.plot_config.draw_colors:
            self.ax_map.contourf(field, levels=levels, colors=colors, extend=extend)
        if self.plot_config.draw_contours:
            self.ax_map.contour(field, levels=levels, colors="black", linewidths=1)

    def draw_boxes(self):
        for fill_box, box in self.boxes.items():
            fill_box(box)
            box.draw()

    # pylint: disable=R0914  # too-many-locals
    def add_text_boxes(self, layout):
        kwargs = {
            "fig": self.fig,
            "lw_frame": self.plot_config.lw_frame,
        }
        self.boxes = {}
        self.boxes[self.fill_box_top_left] = TextBoxAxes(
            name="top/left", rect=layout.rect_top_left(), **kwargs,
        )
        self.boxes[self.fill_box_top_right] = TextBoxAxes(
            name="top/right", rect=layout.rect_top_right(), **kwargs,
        )
        self.boxes[self.fill_box_right_top] = TextBoxAxes(
            name="right/top", rect=layout.rect_center_right_top(), **kwargs,
        )
        self.boxes[self.fill_box_right_bottom] = TextBoxAxes(
            name="right/bottom", rect=layout.rect_center_right_bottom(), **kwargs,
        )
        self.boxes[self.fill_box_bottom] = TextBoxAxes(
            name="bottom", rect=layout.rect_bottom(), **{**kwargs, "lw_frame": None},
        )

    def fill_box_top_left(self, box: TextBoxAxes) -> None:
        sizes = {"tl": "x-large"}
        for position, label in self.plot_config.labels.get("top_left", {}).items():
            size = sizes.get("position", "large")
            box.text(position, label, size=size)

    def fill_box_top_right(self, box: TextBoxAxes) -> None:
        for position, label in self.plot_config.labels.get("top_right", {}).items():
            box.text(position, label, size="large")

    # pylint: disable=R0912   # too-many-branches
    # pylint: disable=R0913   # too-many-arguments
    # pylint: disable=R0914   # too-many-locals
    # pylint: disable=R0915   # too-many-statements
    def fill_box_right_top(
        self,
        box: TextBoxAxes,
        dy_line: float = 3.0,
        dy0_markers: float = 0.25,
        w_box: float = 4.0,
        h_box: float = 2.0,
    ) -> None:
        """Fill the top box to the right of the map plot."""

        labels = self.plot_config.labels["right_top"]

        # font_size = 'small'
        font_size = "medium"

        dx_box: float = -10
        dx_label: float = -3

        dx_marker: float = dx_box + 0.5 * w_box
        dx_marker_label: float = dx_label

        # Color boxes (legend)

        # Vertical position of legend (depending on number of levels)
        assert self.plot_config.n_levels is not None  # mypy
        _f = (
            self.plot_config.n_levels
            + int(self.plot_config.extend in ["min", "both"])
            + int(self.plot_config.extend in ["max", "both"])
        )
        dy0_labels = 22.5 - 1.5 * _f
        dy0_boxes = dy0_labels - 0.2 * h_box

        # Box title
        s = self.plot_config.legend_box_title
        box.text("tc", s=s, dy=1.5, size="large")

        # Format level ranges (contour plot legend)
        levels = levels_from_time_stats(self.plot_config, self.field.time_stats)
        legend_labels = format_level_ranges(
            levels=levels,
            style=self.plot_config.level_range_style,
            extend=self.plot_config.extend,
            rstrip_zeros=self.plot_config.legend_rstrip_zeros,
            align=self.plot_config.level_ranges_align,
        )

        # Legend labels (level ranges)
        box.text_block(
            "bc",
            legend_labels,
            dy_unit=dy0_labels,
            dy_line=dy_line,
            dx=dx_label,
            reverse=self.plot_config.reverse_legend,
            ha="left",
            size=font_size,
            family="monospace",
        )

        # Legend color boxes
        colors = colors_from_plot_config(self.plot_config)
        if self.plot_config.reverse_legend:
            colors = colors[::-1]
        dy = dy0_boxes
        for color in colors:
            box.color_rect(
                loc="bc",
                x_anker="left",
                dx=dx_box,
                dy=dy,
                w=w_box,
                h=h_box,
                fc=color,
                ec="black",
                lw=1.0,
            )
            dy += dy_line

        # self.Markers

        n_markers = self.plot_config.mark_release_site + self.plot_config.mark_field_max
        dy0_marker_i = dy0_markers + (2 - n_markers) * dy_line / 2

        # Release site marker
        if self.plot_config.mark_release_site:
            dy_site_label = dy0_marker_i
            dy0_marker_i += dy_line
            dy_site_marker = dy_site_label + 0.7
            box.marker(
                loc="bc", dx=dx_marker, dy=dy_site_marker, **self._site_marker_kwargs,
            )
            s = labels["release_site"]
            box.text(
                loc="bc",
                s=s,
                dx=dx_marker_label,
                dy=dy_site_label,
                ha="left",
                size=font_size,
            )

        # Field maximum marker
        if self.plot_config.mark_field_max:
            dy_marker_label_max = dy0_marker_i
            dy0_marker_i += dy_line
            dy_max_marker = dy_marker_label_max + 0.7
            box.marker(
                loc="bc", dx=dx_marker, dy=dy_max_marker, **self._max_marker_kwargs,
            )
            s = f"{labels['max']}: "
            if np.isnan(self.field.fld).all():
                s += "NaN"
            else:
                fld_max = np.nanmax(self.field.fld)
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
                loc="bc",
                s=s,
                dx=dx_marker_label,
                dy=dy_marker_label_max,
                ha="left",
                size=font_size,
            )

    def fill_box_right_bottom(self, box: TextBoxAxes) -> None:
        """Fill the bottom box to the right of the map plot."""

        labels = self.plot_config.labels["right_bottom"]
        mdata = self.plot_config.mdata

        # Box title
        # box.text('tc', labels['title'], size='large')
        box.text("tc", labels["title"], dy=-1.0, size="large")

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
        dy_unit = 1.5
        dy = 2.5
        box.text_blocks_hfill(
            "b",
            dy_unit=dy_unit,
            dy_line=dy,
            blocks=info_blocks,
            reverse=True,
            # size="small",
            size="small",
        )

    def fill_box_bottom(self, box: TextBoxAxes) -> None:
        """Fill the box to the bottom of the map plot."""

        labels = self.plot_config.labels["bottom"]

        # FLEXPART/model info
        s = self.plot_config.model_info
        box.text("tl", dx=-0.7, dy=0.5, s=s, size="small")

        # MeteoSwiss Copyright
        box.text("tr", dx=0.7, dy=0.5, s=labels["copyright"], size="small")


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
            plot_config = create_plot_config(setup, WORDS, SYMBOLS, mdata)
            plot = Plot(field, plot_config, map_conf)
            plot.save(out_file_path, write=write)

        yield out_file_path, plot


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
# pylint: disable=R0904  # too-many-public-methods
class PlotLayout:
    aspect: float
    x0_tot: float = 0.0
    x1_tot: float = 1.0
    y0_tot: float = 0.0
    y1_tot: float = 1.0
    y_pad: float = 0.02
    h_top: float = 0.1
    w_right: float = 0.2
    h_crbot: float = 0.45
    h_bottom: float = 0.05

    @property
    def x_pad(self):
        return self.y_pad / self.aspect

    @property
    def h_tot(self):
        return self.y1_tot - self.y0_tot

    @property
    def y1_top(self):
        return self.y1_tot

    @property
    def y0_top(self):
        return self.y1_top - self.h_top

    @property
    def y1_center(self):
        return self.y0_top - self.y_pad

    @property
    def y0_center(self):
        return self.y0_tot + self.h_bottom

    @property
    def h_center(self):
        return self.y1_center - self.y0_center

    @property
    def w_tot(self):
        return self.x1_tot - self.x0_tot

    @property
    def x1_right(self):
        return self.x1_tot

    @property
    def x0_right(self):
        return self.x1_right - self.w_right

    @property
    def x1_left(self):
        return self.x0_right - self.x_pad

    @property
    def x0_left(self):
        return self.x0_tot

    @property
    def w_left(self):
        return self.x1_left - self.x0_left

    @property
    def y0_crbot(self):
        return self.y0_center

    @property
    def y1_crbot(self):
        return self.y0_crbot + self.h_crbot

    @property
    def y1_crtop(self):
        return self.y1_center

    @property
    def y0_crtop(self):
        return self.y1_crbot + self.y_pad

    @property
    def h_crtop(self):
        return self.y1_crtop - self.y0_crtop

    def rect_top_left(self):
        return (self.x0_left, self.y0_top, self.w_left, self.h_top)

    def rect_top_right(self):
        return (self.x0_right, self.y0_top, self.w_right, self.h_top)

    def rect_center_left(self):
        return (self.x0_left, self.y0_center, self.w_left, self.h_center)

    def rect_center_right_top(self):
        return (self.x0_right, self.y0_crtop, self.w_right, self.h_crtop)

    def rect_center_right_bottom(self):
        return (self.x0_right, self.y0_crbot, self.w_right, self.h_crbot)

    def rect_bottom(self):
        return (self.x0_tot, self.y0_tot, self.w_tot, self.h_bottom)


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

# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import os
import re
from textwrap import dedent
from typing import Collection
from typing import Optional

# Third-party
import numpy as np
from matplotlib import pyplot as plt

# First-party
from srutils.geo import Degrees
from srutils.iter import isiterable

# Local
# TextBoxAxes Local
from .data import Field
from .plot_lib import MapAxes
from .plot_lib import TextBoxAxes
from .plot_lib import ax_w_h_in_fig_coords
from .plot_lib import post_summarize_plot
from .plot_types import MapAxesConf
from .plot_types import MapAxesConf_Cosmo1
from .plot_types import MapAxesConf_Cosmo1_CH
from .plot_types import PlotConfig
from .utils import fmt_float
from .utils import format_level_ranges
from .utils import summarizable


@summarizable(
    attrs=[
        "ax_map",
        "boxes",
        "draw_colors",
        "draw_contours",
        "field",
        "fig",
        "map_conf",
        "mark_release_site",
        "plot_config",
    ],
    post_summarize=post_summarize_plot,
)
class Plot:
    """A FLEXPART dispersion plot."""

    cmap = "flexplot"
    draw_colors = True
    draw_contours = False
    mark_release_site = True
    lw_frame = 1.0

    # SR_TMP TODO consider moving mdat (at least mdata.grid) to Field
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

        self.create_plot()

    def save(self, file_path: str, format: Optional[str] = None):
        """Save the plot to disk.

        Args:
            file_path: Output file name, incl. path.

            format (optional): Plot format (e.g., 'png', 'pdf'). If ``format``
                is None, the plot format is derived from the extension of
                ``file_path``.

        """
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in [".pdf", ".png"]:
                raise ValueError(
                    f"Cannot derive format from extension '{ext}' derived "
                    f"from '{os.path.basename(file_path)}'"
                )
            format = ext[1:]
        assert format is not None  # mypy
        self.format: str = format

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
        self.fig = plt.figure(figsize=self.plot_config.figsize)
        self.ax_map = MapAxes.create(self.map_conf, fig=self.fig, field=self.field)

        # Plot particle concentration field
        self.draw_map_plot()

        # Text boxes around plot
        self.add_text_boxes()
        self.draw_boxes()

    def fld_nonzero(self):
        return np.where(self.field.fld > 0, self.field.fld, np.nan)

    def draw_map_plot(self):
        """Plot the particle concentrations onto the map."""

        # Plot concentrations
        self.draw_colors_contours()

        if self.mark_release_site:
            # Marker at release site
            self.ax_map.marker(
                self.plot_config.mdata.release.site_lon.value,
                self.plot_config.mdata.release.site_lat.value,
                **self._site_marker_kwargs,
            )

        if self.plot_config.mark_field_max:
            # Marker at location of maximum value
            self.ax_map.mark_max(self.field.fld, **self._max_marker_kwargs)

    # SR_TODO Replace checks with plot-specific config/setup object
    def draw_colors_contours(self):
        field = self.fld_nonzero()
        levels = self.plot_config.get_levels(self.field.time_stats)
        colors = self.plot_config.get_colors(self.cmap)
        extend = self.plot_config.extend
        if self.plot_config.levels_scale == "log":
            field = np.log10(field)
            levels = np.log10(levels)
        elif extend in ["none", "min"]:
            # Areas beyond the closed upper bound are colored black
            colors = [c for c in colors] + ["black"]
            extend = {"none": "max", "min": "both"}[extend]
        if self.draw_colors:
            self.ax_map.contourf(field, levels=levels, colors=colors, extend=extend)
        if self.draw_contours:
            self.ax_map.contour(field, levels=levels, colors="black", linewidths=1)

    def draw_boxes(self):
        for fill_box, box in self.boxes.items():
            fill_box(box)
            box.draw()

    def add_text_boxes(self):
        text_box_setup = self.plot_config.text_box_setup
        h_rel_t = text_box_setup["h_rel_t"]
        h_rel_b = text_box_setup["h_rel_b"]
        w_rel_r = text_box_setup["w_rel_r"]
        pad_hor_rel = text_box_setup["pad_hor_rel"]
        h_rel_box_rt = text_box_setup["h_rel_box_rt"]

        # Freeze the map plot in order to fix it's coordinates (bbox)
        try:
            self.fig.canvas.draw()
        except Exception as e:
            # Avoid long library tracebacks (matplotlib/cartopy/urllib/...)
            raise RuntimeError(e)

        # Obtain aspect ratio of figure
        fig_pxs = self.fig.get_window_extent()
        fig_aspect = fig_pxs.width / fig_pxs.height

        # Get map dimensions in figure coordinates
        w_map, h_map = ax_w_h_in_fig_coords(self.fig, self.ax_map.ax)

        # Relocate the map close to the lower left corner
        x0_map, y0_map = 0.05, 0.05
        self.ax_map.ax.set_position([x0_map, y0_map, w_map, h_map])

        # Determine height of top box and width of right boxes
        w_box = w_rel_r * w_map
        h_box_t = h_rel_t * h_map
        h_box_b = h_rel_b * h_map

        # Determine padding between plot and boxes
        pad_hor = pad_hor_rel * w_map
        pad_ver = pad_hor * fig_aspect

        text_box_conf = {
            "fig": self.fig,
            "ax_ref": self.ax_map.ax,
            "f_pad": 1.2,
            "lw_frame": self.lw_frame,
        }

        # Axes for text boxes (one on top, two to the right)
        self.boxes = {
            self.fill_box_top_left: TextBoxAxes(
                name="top/left",
                rect=[x0_map, y0_map + pad_ver + h_map, w_map, h_box_t],
                **text_box_conf,
            ),
            self.fill_box_top_right: TextBoxAxes(
                name="top/right",
                rect=[
                    x0_map + w_map + pad_hor,
                    y0_map + pad_ver + h_map,
                    w_box,
                    h_box_t,
                ],
                **text_box_conf,
            ),
            self.fill_box_right_top: TextBoxAxes(
                name="right/top",
                rect=[
                    x0_map + pad_hor + w_map,
                    y0_map + 0.5 * pad_ver + (1.0 - h_rel_box_rt) * h_map,
                    w_box,
                    h_rel_box_rt * h_map - 0.5 * pad_ver,
                ],
                **text_box_conf,
            ),
            self.fill_box_right_bottom: TextBoxAxes(
                name="right/bottom",
                rect=[
                    x0_map + pad_hor + w_map,
                    y0_map,
                    w_box,
                    (1.0 - h_rel_box_rt) * h_map - 0.5 * pad_ver,
                ],
                **text_box_conf,
            ),
            self.fill_box_bottom: TextBoxAxes(
                name="bottom",
                rect=[x0_map, y0_map - h_box_b, w_map + pad_hor + w_box, h_box_b],
                **{**text_box_conf, "lw_frame": None},
            ),
        }

    def fill_box_top_left(
        self, box: TextBoxAxes, *, skip_pos: Optional[Collection[str]] = None,
    ) -> None:
        """Fill the box above the map plot."""

        labels = self.plot_config.labels.top_left

        if skip_pos is None:
            skip_pos = {}

        # Top-left: Variable name etc.
        if "tl" not in skip_pos:
            box.text("tl", labels["variable"], size="large")

        # Center-left: Additional information
        if "ml" not in skip_pos:
            subtitle = self.plot_config.top_box_subtitle
            if subtitle:
                box.text("ml", subtitle, size="large")

        # Bottom-left: Integration time etc.
        if "bl" not in skip_pos:
            s = labels["period"]
            box.text("bl", s, size="large")

        # Top-right: Time step
        if "tr" not in skip_pos:
            box.text("tr", labels["timestep"], size="large")

        # Bottom-right: Time since release start
        if "br" not in skip_pos:
            box.text("br", labels["time_since_release_start"], size="large")

    def fill_box_top_right(
        self, box: TextBoxAxes, *, skip_pos: Optional[Collection[str]] = None,
    ) -> None:
        """Fill the box to the top-right of the map plot."""

        labels = self.plot_config.labels.top_right

        if skip_pos is None:
            skip_pos = []

        if "tc" not in skip_pos:
            # Top center: species
            box.text("tc", labels["species"], size="large")

        if "bc" not in skip_pos:
            # Bottom center: release site (shrunk/truncated to fit box)
            s, size = box.fit_text(labels["site"], "large", n_shrink_max=1)
            box.text("bc", s, size=size)

    def fill_box_right_top(
        self,
        box: TextBoxAxes,
        dy_line: float = 3.0,
        dy0_markers: float = 0.25,
        w_box: float = 4.0,
        h_box: float = 2.0,
    ) -> None:
        """Fill the top box to the right of the map plot."""

        labels = self.plot_config.labels.right_top

        # font_size = 'small'
        font_size = "medium"

        dx_box: float = -10
        dx_label: float = -3

        dx_marker: float = dx_box + 0.5 * w_box
        dx_marker_label: float = dx_label

        # Color boxes (legend)

        # Vertical position of legend (depending on number of levels)
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
        levels = self.plot_config.get_levels(time_stats=self.field.time_stats)
        legend_labels = format_level_ranges(
            levels=levels,
            style=self.plot_config.level_range_style,
            extend=self.plot_config.extend,
            rstrip_zeros=True,
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
        colors = self.plot_config.get_colors(self.cmap)
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

        # Markers

        n_markers = self.mark_release_site + self.plot_config.mark_field_max
        dy0_marker_i = dy0_markers + (2 - n_markers) * dy_line / 2

        # Release site marker
        if self.mark_release_site:
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
                s += fmt_float(
                    np.nanmax(self.field.fld), fmt_e0="{f:.3E}", fmt_f1="{f:,.3f}"
                )
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

        labels = self.plot_config.labels.right_bottom  # noqa:E741
        mdata = self.plot_config.mdata

        # Box title
        # box.text('tc', labels['title'], size='large')
        box.text("tc", labels["title"], dy=-1.0, size="large")

        # Release site coordinates
        lat = Degrees(mdata.release.site_lat.value)
        lon = Degrees(mdata.release.site_lon.value)
        lat_deg = labels["lat_deg_fmt"].format(d=lat.degs(), m=lat.mins(), f=lat.frac())
        lon_deg = labels["lon_deg_fmt"].format(d=lon.degs(), m=lon.mins(), f=lon.frac())

        info_blocks = dedent(
            f"""\
            {labels['site']}:\t{mdata.release.site_name.format()}
            {labels['latitude']}:\t{lat_deg}
            {labels['longitude']}:\t{lon_deg}
            {labels['height']}:\t{mdata.release.height.format()}

            {labels['start']}:\t{mdata.release.start.format()}
            {labels['end']}:\t{mdata.release.end.format()}
            {labels['rate']}:\t{mdata.release.rate.format()}
            {labels['mass']}:\t{mdata.release.mass.format()}

            {labels['name']}:\t{mdata.species.name.format()}
            {labels['half_life']}:\t{mdata.species.half_life.format()}
            {labels['deposit_vel']}:\t{mdata.species.deposit_vel.format()}
            {labels['sediment_vel']}:\t{mdata.species.sediment_vel.format()}
            {labels['washout_coeff']}:\t{mdata.species.washout_coeff.format()}
            {labels['washout_exponent']}:\t{mdata.species.washout_exponent.format()}
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
            size="small",
        )

    def fill_box_bottom(self, box: TextBoxAxes) -> None:
        """Fill the box to the bottom of the map plot."""

        labels = self.plot_config.labels.bottom

        # FLEXPART/model info
        s = self.plot_config.model_info
        box.text("tl", dx=-0.7, dy=0.5, s=s, size="small")

        # MeteoSwiss Copyright
        box.text("tr", dx=0.7, dy=0.5, s=labels["copyright"], size="small")


def plot_fields(fields, mdata_lst, dry_run=False):
    print(f"create {len(fields)} plot{'s' if len(fields) > 1 else ''}")

    # Create plots one-by-one
    for field, mdata in zip(fields, mdata_lst):
        setup = field.var_setups.compress()
        out_file_path = format_out_file_path(setup)

        # SR_TMP < TODO Find less hard-coded solution
        model = field.nc_meta_data["analysis"]["model"]
        domain = setup.domain
        map_conf = None
        if domain == "auto":
            domain = model
        # if model == "cosmo1":
        if model.startswith("cosmo"):
            # if domain == "cosmo1":
            if domain.startswith("cosmo"):
                map_conf = MapAxesConf_Cosmo1(lang=setup.lang)
            elif domain == "ch":
                map_conf = MapAxesConf_Cosmo1_CH(lang=setup.lang)
        if not map_conf:
            raise NotImplementedError(
                f"map_conf for model '{model}' and domain '{domain}'"
            )
        # SR_TMP >

        yield out_file_path

        if not dry_run:
            assert mdata is not None  # mypy
            plot_config = PlotConfig(setup, mdata)
            Plot(field, plot_config, map_conf).save(out_file_path)


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

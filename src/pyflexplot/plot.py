# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import os
import re
from textwrap import dedent
from typing import Optional

# Third-party
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

# First-party
from srutils.geo import Degrees
from srutils.iter import isiterable

# Local
from .attr import AttrMult
from .plot_lib import MapAxesConf_Cosmo1
from .plot_lib import MapAxesConf_Cosmo1_CH
from .plot_lib import MapAxesRotatedPole
from .plot_lib import TextBoxAxes
from .plot_lib import ax_w_h_in_fig_coords
from .plot_lib import post_summarize_plot
from .setup import Setup
from .utils import fmt_float
from .utils import format_level_ranges
from .utils import summarizable
from .words import SYMBOLS
from .words import WORDS


# SR_TMP TODO Turn into dataclass or the like.
@summarizable
class PlotLabels:

    words = WORDS
    symbols = SYMBOLS

    def __init__(self, lang, attrs):
        """Create an instance of ``PlotLabels``."""
        self.attrs = attrs

        w = self.words
        s = self.symbols
        a = self.attrs

        w.set_default_lang(lang)

        groups = {}

        # Top-left box
        level = a.variable.fmt_level_range()
        s_level = f" {w['at', None, 'level']} {level}" if level else ""
        integr_op = w[
            {
                "sum": "summed_over",
                "mean": "averaged_over",
                "accum": "accumulated_over",
            }[a.simulation.integr_type.value]
        ].s
        ts = a.simulation.now
        time_rels = a.simulation.now.format(rel=True, rel_start=a.release.start.value)
        period = a.simulation.fmt_integr_period()
        start = a.simulation.integr_start.format(rel=True)
        unit_escaped = a.variable.unit.format(escape_format=True)
        groups["top_left"] = {
            "variable": f"{a.variable.long_name.value}{s_level}",
            "period": f"{integr_op} {period} ({w['since']} +{start})",
            "subtitle_thr_agrmt_fmt": f"Cloud: {s['geq']} {{thr}} {unit_escaped}",
            "subtitle_cloud_arrival_time": (
                f"Cloud: {s['geq']} {{thr}} {unit_escaped}; "
                f"members: {s['geq']} {{mem}}"
            ),
            "timestep": f"{ts.format(rel=False)} (+{ts.format(rel=True)})",
            "time_since_release_start": (
                f"{time_rels} {w['since']} {w['release_start']}"
            ),
        }

        # Top-right box
        groups["top_right"] = {
            "species": f"{a.species.name.format(join=' + ')}",
            "site": f"{w['site']}: {a.release.site_name.value}",
        }

        # Right-top box
        name = a.variable.short_name.format()
        unit = a.variable.unit.format()
        unit_escaped = a.variable.unit.format(escape_format=True)
        groups["right_top"] = {
            "title": f"{name}",
            "title_unit": f"{name} ({unit})",
            "release_site": w["release_site"].s,
            "max": w["max"].s,
        }

        # Right-bottom box
        deg_ = f"{s['deg']}{s['short_space']}"
        _N = f"{s['short_space']}{w['north', None, 'abbr']}"
        _E = f"{s['short_space']}{w['east', None, 'abbr']}"
        groups["right_bottom"] = {
            "title": w["release"].t,
            "start": w["start"].s,
            "end": w["end"].s,
            "latitude": w["latitude"].s,
            "longitude": w["longitude"].s,
            "lat_deg_fmt": f"{{d}}{deg_}{{m}}'{_N} ({{f:.2f}}{w['degN']})",
            "lon_deg_fmt": f"{{d}}{deg_}{{m}}'{_E} ({{f:.2f}}{w['degE']})",
            "height": w["height"].s,
            "rate": w["rate"].s,
            "mass": w["total_mass"].s,
            "site": w["site"].s,
            "release_site": w["release_site"].s,
            "max": w["max"].s,
            "name": w["substance"].s,
            "half_life": w["half_life"].s,
            "deposit_vel": w["deposition_velocity", None, "abbr"].s,
            "sediment_vel": w["sedimentation_velocity", None, "abbr"].s,
            "washout_coeff": w["washout_coeff"].s,
            "washout_exponent": w["washout_exponent"].s,
        }

        # Bottom box
        model = a.simulation.model_name.value
        n_members = 21  # SR_TMP SR_HC TODO un-hardcode
        ens_member_id = "{:03d}-{:03d}".format(0, 20)  # SR_TMP SR_HC TODO un-hardcode
        model_ens = (
            f"{model} {w['ensemble']} ({n_members} {w['member', None, 'pl']}: "
            f"{ens_member_id}"
        )
        start = a.simulation.start.format()
        model_info_fmt = f"{w['flexpart']} {w['based_on']} {model}, {start}"
        groups["bottom"] = {
            "model_info_det": model_info_fmt.format(m=model),
            "model_info_ens": model_info_fmt.format(m=model_ens),
            "copyright": f"{s['copyright']}{w['meteoswiss']}",
        }

        # Format all labels (hacky!)
        for group_name, group in groups.items():
            setattr(self, group_name, group)  # SR_TMP
            for name, s in group.items():
                if isinstance(s, str):
                    # Capitalize first letter only (even if it's a space!)
                    group[name] = list(s)[0].capitalize() + s[1:]


def colors_flexplot(n_levels, extend):

    # color_under = [i/255.0 for i in (255, 155, 255)]
    color_under = [i / 255.0 for i in (200, 200, 200)]
    color_over = [i / 255.0 for i in (200, 200, 200)]

    colors_core_8 = (
        np.array(
            [
                (224, 196, 172),  #
                (221, 127, 215),  #
                (99, 0, 255),  #
                (100, 153, 199),  #
                (34, 139, 34),  #
                (93, 255, 2),  #
                (199, 255, 0),  #
                (255, 239, 57),  #
            ],
            float,
        )
        / 255
    ).tolist()

    colors_core_7 = [colors_core_8[i] for i in (0, 1, 2, 3, 5, 6, 7)]
    colors_core_6 = [colors_core_8[i] for i in (1, 2, 3, 4, 5, 7)]
    colors_core_5 = [colors_core_8[i] for i in (1, 2, 4, 5, 7)]

    try:
        colors_core = {
            6: colors_core_5,
            7: colors_core_6,
            8: colors_core_7,
            9: colors_core_8,
        }[n_levels]
    except KeyError:
        raise ValueError(f"n_levels={n_levels}")

    if extend == "none":
        return colors_core
    elif extend == "min":
        return [color_under] + colors_core
    elif extend == "max":
        return colors_core + [color_over]
    elif extend == "both":
        return [color_under] + colors_core + [color_over]
    raise ValueError(f"extend='{extend}'")


class PlotSetup:
    """Class to pull code specific to individual plots out of Plot.

    For the moment, assume this class is only temporary!

    """

    def __init__(
        self, setup: Setup, attrs: AttrMult, labels: Optional[PlotLabels] = None
    ) -> None:
        self.setup = setup
        self.attrs = attrs
        self.labels: PlotLabels = labels or PlotLabels(setup.lang, attrs)

        self.figsize = (12.0, 9.0)
        self.reverse_legend = False

    # SR_TMP <<<
    @property
    def text_box_setup(self):
        return {
            "deterministic": {
                "h_rel_t": 0.1,
                "h_rel_b": 0.03,
                "w_rel_r": 0.25,
                "pad_hor_rel": 0.015,
                "h_rel_box_rt": 0.45,
            },
            "ensemble": {
                "h_rel_t": 0.14,
                "h_rel_b": 0.03,
                "w_rel_r": 0.25,
                "pad_hor_rel": 0.015,
                "h_rel_box_rt": 0.46,
            },
        }[self.setup.simulation_type]

    # SR_TMP
    @property
    def extend(self):
        return {
            "ens_thr_agrmt": "min",
            "ens_cloud_arrival_time": "max",
            "affected_area_mono": "none",
        }.get(self.setup.plot_type, "max")

    # SR_TMP
    @property
    def n_levels(self):
        return {
            "ens_thr_agrmt": 7,
            "ens_cloud_arrival_time": 9,
            "affected_area_mono": 1,
        }.get(
            self.setup.plot_type,
            {"concentration": 8, "deposition": 9}[self.setup.variable],
        )

    # SR_TMP
    @property
    def d_level(self):
        return {"ens_thr_agrmt": 2, "ens_cloud_arrival_time": 3}.get(
            self.setup.plot_type
        )

    # SR_TMP
    @property
    def level_range_style(self):
        return {"ens_thr_agrmt": "int", "ens_cloud_arrival_time": "int"}.get(
            self.setup.plot_type, "base"
        )

    # SR_TMP
    @property
    def level_ranges_align(self):
        return {"ens_thr_agrmt": "left", "ens_cloud_arrival_time": "left"}.get(
            self.setup.plot_type, "center"
        )

    # SR_TMP
    @property
    def mark_field_max(self):
        return {"ens_thr_agrmt": False, "ens_cloud_arrival_time": False}.get(
            self.setup.plot_type, True
        )

    # SR_TMP
    @property
    def top_box_subtitle(self):
        setup = self.setup
        labels = self.labels.top_left
        if setup.plot_type == "ens_thr_agrmt":
            return labels["subtitle_thr_agrmt_fmt"].format(thr=setup.ens_param_thr,)
        elif setup.plot_type == "ens_cloud_arrival_time":
            return labels["subtitle_cloud_arrival_time"].format(
                thr=setup.ens_param_thr, mem=setup.ens_param_mem_min,
            )
        else:
            return None

    # SR_TMP
    @property
    def legend_box_title(self):
        labels = self.labels.right_top
        if self.setup.plot_type == "ens_thr_agrmt":
            return labels["title"]
        elif self.setup.plot_type == "ens_cloud_arrival_time":
            return labels["title"]
        else:
            return labels["title_unit"]

    # SR_TMP
    @property
    def model_info(self):
        labels = self.labels.bottom
        if self.setup.simulation_type == "deterministic":
            return labels["model_info_det"]
        elif self.setup.simulation_type == "ensemble":
            return labels["model_info_ens"]

    # SR_TMP
    def get_colors(self, cmap):
        if self.setup.plot_type == "affected_area_mono":
            return (np.array([(200, 200, 200)]) / 255).tolist()
        elif cmap == "flexplot":
            return colors_flexplot(self.n_levels, self.extend)
        else:
            cmap = mpl.cm.get_cmap(cmap)
            return [cmap(i / (self.n_levels - 1)) for i in range(self.n_levels)]

    # SR_TMP
    @property
    def levels_scale(self):
        if self.setup.plot_type in ["ens_thr_agrmt", "ens_cloud_arrival_time"]:
            return "lin"
        return "log"

    # SR_TMP
    def get_levels(self, time_stats):
        n_levels = self.n_levels
        d_level = self.d_level
        if self.setup.plot_type == "ens_thr_agrmt":
            n_max = 20  # SR_TMP SR_HC
            return (
                np.arange(n_max - d_level * (n_levels - 1), n_max + d_level, d_level,)
                + 1
            )
        elif self.setup.plot_type == "ens_cloud_arrival_time":
            return np.arange(0, n_levels) * d_level
        elif self.setup.plot_type == "affected_area_mono":
            levels = self._auto_levels_log10(n_levels=9, val_max=time_stats["max"])
            return np.array([levels[0], np.inf])
        else:
            return self._auto_levels_log10(val_max=time_stats["max"])

    def _auto_levels_log10(self, n_levels=None, val_max=None):
        if n_levels is None:
            n_levels = self.n_levels
        log10_max = int(np.floor(np.log10(val_max)))
        log10_d = 1
        return 10 ** np.arange(
            log10_max - (n_levels - 1) * log10_d, log10_max + 0.5 * log10_d, log10_d,
        )


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
        "plot_setup",
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

    # SR_TMP TODO consider moving attrs (at least attrs.grid) to Field
    def __init__(self, field, plot_setup, map_conf):
        """Create an instance of ``Plot``."""
        self.field = field
        self.plot_setup = plot_setup
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

    def prepare_plot(self):

        self.fig = plt.figure(figsize=self.plot_setup.figsize)

        self.ax_map = MapAxesRotatedPole(
            self.fig,
            self.field.rlat,
            self.field.rlon,
            self.plot_setup.attrs.grid.north_pole_lat.value,
            self.plot_setup.attrs.grid.north_pole_lon.value,
            self.map_conf,
        )

    def save(self, file_path, format=None):
        """Save the plot to disk.

        Args:
            file_path (str): Output file name, incl. path.

            format (str): Plot format (e.g., 'png', 'pdf'). Defaults to None.
                If ``format`` is None, the plot format is derived from the
                extension of ``file_path``.

        """
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in [".pdf", ".png"]:
                raise ValueError(
                    f"Cannot derive format from extension '{ext}' derived "
                    f"from '{os.path.basename(file_path)}'"
                )
            format = ext[1:]

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
        self.prepare_plot()

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
                self.plot_setup.attrs.release.site_lon.value,
                self.plot_setup.attrs.release.site_lat.value,
                **self._site_marker_kwargs,
            )

        if self.plot_setup.mark_field_max:
            # Marker at location of maximum value
            self.ax_map.mark_max(self.field.fld, **self._max_marker_kwargs)

    # SR_TODO Replace checks with plot-specific config/setup object
    def draw_colors_contours(self):
        field = self.fld_nonzero()
        levels = self.plot_setup.get_levels(self.field.time_stats)
        colors = self.plot_setup.get_colors(self.cmap)
        extend = self.plot_setup.extend
        if self.plot_setup.levels_scale == "log":
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
        text_box_setup = self.plot_setup.text_box_setup
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

    def fill_box_top_left(self, box, *, skip_pos=None):
        """Fill the box above the map plot."""

        labels = self.plot_setup.labels.top_left

        if skip_pos is None:
            skip_pos = {}

        # Top-left: Variable name etc.
        if "tl" not in skip_pos:
            box.text("tl", labels["variable"], size="large")

        # Center-left: Additional information
        if "ml" not in skip_pos:
            subtitle = self.plot_setup.top_box_subtitle
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

        return box

    def fill_box_top_right(self, box, *, skip_pos=None):
        """Fill the box to the top-right of the map plot."""

        labels = self.plot_setup.labels.top_right

        if skip_pos is None:
            skip_pos = []

        if "tc" not in skip_pos:
            # Top center: species
            box.text("tc", labels["species"], size="large")

        if "bc" not in skip_pos:
            # Bottom center: release site (shrunk/truncated to fit box)
            s, size = box.fit_text(labels["site"], "large", n_shrink_max=1)
            box.text("bc", s, size=size)

        return box

    def fill_box_right_top(self, box, dy_line=3.0, dy0_markers=0.25, w_box=4, h_box=2):
        """Fill the top box to the right of the map plot."""

        labels = self.plot_setup.labels.right_top

        # font_size = 'small'
        font_size = "medium"

        dx_box = -10
        dx_label = -3

        dx_marker = dx_box + 0.5 * w_box
        dx_marker_label = dx_label

        # Color boxes (legend)

        # Vertical position of legend (depending on number of levels)
        _f = (
            self.plot_setup.n_levels
            + int(self.plot_setup.extend in ["min", "both"])
            + int(self.plot_setup.extend in ["max", "both"])
        )
        dy0_labels = 22.5 - 1.5 * _f
        dy0_boxes = dy0_labels - 0.2 * h_box

        # Box title
        s = self.plot_setup.legend_box_title
        box.text("tc", s=s, dy=1.5, size="large")

        # Format level ranges (contour plot legend)
        levels = self.plot_setup.get_levels(time_stats=self.field.time_stats)
        legend_labels = format_level_ranges(
            levels=levels,
            style=self.plot_setup.level_range_style,
            extend=self.plot_setup.extend,
            rstrip_zeros=True,
            align=self.plot_setup.level_ranges_align,
        )

        # Legend labels (level ranges)
        box.text_block(
            "bc",
            legend_labels,
            dy_unit=dy0_labels,
            dy_line=dy_line,
            dx=dx_label,
            reverse=self.plot_setup.reverse_legend,
            ha="left",
            size=font_size,
            family="monospace",
        )

        # Legend color boxes
        colors = self.plot_setup.get_colors(self.cmap)
        if self.plot_setup.reverse_legend:
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

        n_markers = self.mark_release_site + self.plot_setup.mark_field_max
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
        if self.plot_setup.mark_field_max:
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

    def fill_box_right_bottom(self, box):
        """Fill the bottom box to the right of the map plot."""

        l = self.plot_setup.labels.right_bottom  # noqa:E741
        a = self.plot_setup.attrs

        # Box title
        # box.text('tc', l['title'], size='large')
        box.text("tc", l["title"], dy=-1.0, size="large")

        # Release site coordinates
        lat = Degrees(a.release.site_lat.value)
        lon = Degrees(a.release.site_lon.value)
        lat_deg = l["lat_deg_fmt"].format(d=lat.degs(), m=lat.mins(), f=lat.frac())
        lon_deg = l["lon_deg_fmt"].format(d=lon.degs(), m=lon.mins(), f=lon.frac())

        info_blocks = dedent(
            f"""\
            {l['site']}:\t{a.release.site_name.format()}
            {l['latitude']}:\t{lat_deg}
            {l['longitude']}:\t{lon_deg}
            {l['height']}:\t{a.release.height.format()}

            {l['start']}:\t{a.release.start.format()}
            {l['end']}:\t{a.release.end.format()}
            {l['rate']}:\t{a.release.rate.format()}
            {l['mass']}:\t{a.release.mass.format()}

            {l['name']}:\t{a.species.name.format()}
            {l['half_life']}:\t{a.species.half_life.format()}
            {l['deposit_vel']}:\t{a.species.deposit_vel.format()}
            {l['sediment_vel']}:\t{a.species.sediment_vel.format()}
            {l['washout_coeff']}:\t{a.species.washout_coeff.format()}
            {l['washout_exponent']}:\t{a.species.washout_exponent.format()}
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

    def fill_box_bottom(self, box):
        """Fill the box to the bottom of the map plot."""

        labels = self.plot_setup.labels.bottom

        # FLEXPART/model info
        s = self.plot_setup.model_info
        box.text("tl", dx=-0.7, dy=0.5, s=s, size="small")

        # MeteoSwiss Copyright
        box.text("tr", dx=0.7, dy=0.5, s=labels["copyright"], size="small")


def plot(fields, attrs_lst, setup):
    return Plotter().run(fields, attrs_lst, setup)


class Plotter:
    """Create one or more FLEXPLART plots of a certain type."""

    specs_fmt_keys = {
        "time": "time_idx",
        "nageclass": "age_idx",
        "numpoint": "rel_pt_idx",
        "level": "level",
        "species_id": "species_id",
        "integrate": "integrate",
    }

    def __init__(self):
        self.file_paths = []

    def run(self, fields, attrs_lst, setup):
        """Create one or more plots.

        Args:
            field (List[Field]): A list of fields.

            attrs_lst (List[Attr???]): A list of data attributes of equal
                length as ``fields``.

            setup (Setup): Plot setup.

        Yields:
            str: Output file paths.

        """
        if setup.outfile is None:
            raise ValueError("setup.outfile is None")

        self.setup = setup

        _s = "s" if len(fields) > 1 else ""
        print(f"create {len(fields)} {self.setup.plot_type} plot{_s}")

        # SR_TMP < TODO Find less hard-coded solution
        domain = self.setup.domain
        if domain == "auto":
            domain = "cosmo1"
        if domain == "cosmo1":
            map_conf = MapAxesConf_Cosmo1(lang=self.setup.lang)
        elif domain == "ch":
            map_conf = MapAxesConf_Cosmo1_CH(lang=self.setup.lang)
        else:
            raise ValueError("unknown domain", domain)
        # SR_TMP >

        # Create plots one-by-one
        for i_data, (field, attrs) in enumerate(zip(fields, attrs_lst)):
            out_file_path = self.format_out_file_path(attrs._setup)
            _w = len(str(len(fields)))
            print(f" {i_data+1:{_w}}/{len(fields)}  {out_file_path}")
            plot_setup = PlotSetup(setup, attrs)
            Plot(field, plot_setup, map_conf).save(out_file_path)
            yield out_file_path

    def format_out_file_path(self, setup):

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
        out_file_path = self._fmt_out_file_path(kwargs)

        # Add number if file path not unique
        out_file_path = self.ensure_unique_path(out_file_path)

        return out_file_path

    def _fmt_out_file_path(self, kwargs):
        out_file_path = self.setup.outfile
        for key, val in kwargs.items():
            if not isiterable(val, str_ok=False):
                val = [val]
            # Iterate over relevant format keys
            rxs = r"{" + key + r"(:[^}]*)?}"
            re.finditer(rxs, out_file_path)
            for m in re.finditer(rxs, out_file_path):

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
                start, end = out_file_path[: m.span()[0]], out_file_path[m.span()[1] :]
                out_file_path = f"{start}{formatted_key}{end}"

        # Check that all keys have been formatted
        if "{" in out_file_path or "}" in out_file_path:
            raise Exception(
                f"formatted output file path still contains format keys", out_file_path,
            )

        return out_file_path

    def ensure_unique_path(self, path):
        """If file path has been used before, add/increment trailing number."""
        while path in self.file_paths:
            path = self.derive_unique_path(path)
        self.file_paths.append(path)
        return path

    @staticmethod
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
        path = f"{path_base}-{{i:0{w}}}{suffix}".format(i=i)

        return path

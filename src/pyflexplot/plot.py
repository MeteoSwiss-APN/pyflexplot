# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import os
from copy import copy
from textwrap import dedent

# Third-party
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

# First-party
from srutils.geo import Degrees

# Local
from .plot_utils import MapAxesRotatedPole
from .plot_utils import TextBoxAxes
from .plot_utils import ax_w_h_in_fig_coords
from .plot_utils import post_summarize_plot
from .utils import fmt_float
from .utils import format_level_ranges
from .utils import summarizable
from .words import SYMBOLS
from .words import WORDS


# SR_TMP TODO Turn into dataclass or the like.
@summarizable
class PlotLabels:
    def __init__(self, lang, words, symbols, attrs):

        self.words = words
        self.symbols = symbols

        w = words
        s = symbols
        a = attrs

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
        ens_member_ids = "{:03d}-{:03d}".format(0, 20)  # SR_TMP SR_HC TODO un-hardcode
        model_ens = (
            f"{model} {w['ensemble']} ({n_members} {w['member', None, 'pl']}: "
            f"{ens_member_ids}"
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


@summarizable(
    attrs=[
        "ax_map",
        "boxes",
        "dpi",
        "draw_colors",
        "draw_contours",
        "extend",
        "field",
        "fig",
        "figsize",
        "labels",
        "level_range_style",
        "map_conf",
        "mark_field_max",
        "mark_release_site",
        "name",
        "setup",
        "text_box_setup",
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

    # SR_TMP TODO move to some config/setup class
    @property
    def text_box_setup(self):
        if self.name.startswith("ens_"):
            return {
                "h_rel_t": 0.14,
                "h_rel_b": 0.03,
                "w_rel_r": 0.25,
                "pad_hor_rel": 0.015,
                "h_rel_box_rt": 0.46,
            }
        else:
            return {
                "h_rel_t": 0.1,
                "h_rel_b": 0.03,
                "w_rel_r": 0.25,
                "pad_hor_rel": 0.015,
                "h_rel_box_rt": 0.45,
            }

    # SR_TMP TODO move to some config/setup class
    @property
    def n_levels(self):
        if self.name.startswith("ens_thr_agrmt"):
            return 7
        elif self.name.startswith("ens_cloud_arrival_time"):
            return 9
        elif self.name.endswith("_mono"):
            return 1
        elif "concentration" in self.name:
            return 8
        elif "deposition" in self.name:
            return 9
        else:
            return 9

    # SR_TMP TODO move to some config/setup class
    @property
    def extend(self):
        if self.name.startswith("ens_thr_agrmt"):
            return "min"
        elif self.name.startswith("ens_cloud_arrival_time"):
            return "max"
        elif self.name.endswith("_mono"):
            return "none"
        else:
            return "max"

    # SR_TMP TODO move to some config/setup class
    @property
    def d_level(self):
        if self.name.startswith("ens_thr_agrmt"):
            return 2
        elif self.name.startswith("ens_cloud_arrival_time"):
            return 3
        else:
            return None

    # SR_TMP TODO move to some config/setup class
    @property
    def level_range_style(self):
        if self.name.startswith("ens_thr_agrmt"):
            return "int"
        elif self.name.startswith("ens_cloud_arrival_time"):
            return "int"
        else:
            return "base"

    # SR_TMP TODO move to some config/setup class
    @property
    def level_ranges_align(self):
        if self.name.startswith("ens_thr_agrmt"):
            return "left"
        elif self.name.startswith("ens_cloud_arrival_time"):
            return "left"
        else:
            return "center"

    # SR_TMP TODO move to some config/setup class
    @property
    def mark_field_max(self):
        if self.name.startswith("ens_thr_agrmt"):
            return False
        elif self.name.startswith("ens_cloud_arrival_time"):
            return False
        else:
            return True

    def __init__(
        self, field, setup, map_conf, *, dpi=None, figsize=None, labels=None,
    ):
        """Create an instance of ``Plot``.

        Args:
            name (str): Name of the plot.

            field (Field): Data field.

            setup (Setup): Plot setup.

            map_conf (MapAxesConf): Map plot setupuration object.

            dpi (float, optional): Plot resolution (dots per inch). Defaults to
                100.0.

            figsize (tuple[float, float], optional): Figure size in inches.
                Defaults to (12.0, 9.0).

            labels (PlotLabels, optional): Labels. Defaults to None.

        """
        self.name = setup.tmp_cls_name()  # SR_TMP TODO eliminate
        if type(self).__name__ != "Plot":  # SR_TMP TODO eliminate
            assert self.name == type(self).name  # SR_TMP TODO eliminate
        self.field = field
        self.setup = setup
        self.map_conf = map_conf
        self.dpi = dpi or 100.0
        self.figsize = figsize or (12.0, 9.0)

        self.field.scale(self.setup.scale_fact)

        if labels is None:
            labels = PlotLabels(setup.lang, WORDS, SYMBOLS, field.attrs)
        self.labels = labels

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

        self.fig = plt.figure(figsize=self.figsize)

        self.ax_map = MapAxesRotatedPole(
            self.fig,
            self.field.rlat,
            self.field.rlon,
            self.field.attrs.grid.north_pole_lat.value,
            self.field.attrs.grid.north_pole_lon.value,
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
        h_con = self.draw_colors_contours()

        if self.mark_release_site:
            # Marker at release site
            self.ax_map.marker(
                self.field.attrs.release.site_lon.value,
                self.field.attrs.release.site_lat.value,
                **self._site_marker_kwargs,
            )

        if self.mark_field_max:
            # Marker at location of maximum value
            self.ax_map.mark_max(self.field.fld, **self._max_marker_kwargs)

        return h_con

    # SR_TODO Replace name checks with plot-specific config/setup object
    def draw_colors_contours(self):
        if self.name.startswith("ens_thr_agrmt"):
            self._tmp_draw_colors_contours_ens_thr_agrmt()
        elif self.name.startswith("ens_cloud_arrival_time"):
            self._tmp_draw_colors_contours_ens_thr_agrmt()
        else:
            self._tmp_draw_colors_contours_default()

    # SR_TMP Intermediate step toward eliminating subclasses
    def _tmp_draw_colors_contours_default(self):

        field = np.log10(self.fld_nonzero())

        if not self.draw_colors:
            h_col = None
        else:
            h_col = self.ax_map.contourf(
                field,
                levels=self.levels_log10,
                colors=self.get_colors(),
                extend=self.extend,
            )

        if not self.draw_contours:
            h_con = None
        else:
            h_con = self.ax_map.contour(
                field, levels=self.levels_log10, colors="black", linewidths=1,
            )
            return (h_col, h_con)

    def _tmp_draw_colors_contours_ens_thr_agrmt(self):

        # If upper end of range is closed, color areas beyond black
        colors_plot = copy(self.get_colors())
        if self.extend in ["none", "min"]:
            colors_plot.append("black")
            extend_plot = {"none": "max", "min": "both"}[self.extend]
        else:
            extend_plot = self.extend

        field = self.fld_nonzero()

        if not self.draw_colors:
            h_col = None
        else:
            h_col = self.ax_map.contourf(
                field, levels=self.get_levels(), colors=colors_plot, extend=extend_plot,
            )

        if not self.draw_contours:
            h_con = None
        else:
            h_con = self.ax_map.contour(
                field, levels=self.get_levels(), colors="black", linewidths=1,
            )

        return (h_col, h_con)

    def draw_boxes(self):
        for fill_box, box in self.boxes.items():
            fill_box(box)
            box.draw()

    def add_text_boxes(self):
        h_rel_t = self.text_box_setup["h_rel_t"]
        h_rel_b = self.text_box_setup["h_rel_b"]
        w_rel_r = self.text_box_setup["w_rel_r"]
        pad_hor_rel = self.text_box_setup["pad_hor_rel"]
        h_rel_box_rt = self.text_box_setup["h_rel_box_rt"]

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

        labels = self.labels.top_left

        if skip_pos is None:
            skip_pos = {}

        # Top-left: Variable name etc.
        if "tl" not in skip_pos:
            box.text("tl", labels["variable"], size="large")

        # Center-left: Additional information
        if "ml" not in skip_pos:
            subtitle = self._format_top_box_subtitle()
            if subtitle:
                box.text("ml", subtitle, size="large")

        # Bottom-left: Integration time etc.
        if "bl" not in skip_pos:
            s = labels["period"]
            if self.setup.scale_fact is not None:
                s += f" ({self.setup.scale_fact}x)"
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

        labels = self.labels.top_right

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

        labels = self.labels.right_top

        # font_size = 'small'
        font_size = "medium"

        dx_box = -10
        dx_label = -3

        dx_marker = dx_box + 0.5 * w_box
        dx_marker_label = dx_label

        # Color boxes (legend)

        # Vertical position of legend (depending on number of levels)
        _f = (
            self.n_levels
            + int(self.extend in ["min", "both"])
            + int(self.extend in ["max", "both"])
        )
        dy0_labels = 22.5 - 1.5 * _f
        dy0_boxes = dy0_labels - 0.2 * h_box

        # Box title
        s = self._format_legend_box_title()
        box.text("tc", s=s, dy=1.5, size="large")

        # Format level ranges (contour plot legend)
        legend_labels = format_level_ranges(
            levels=self.get_levels(),
            style=self.level_range_style,
            extend=self.extend,
            rstrip_zeros=True,
            align=self.level_ranges_align,
        )

        # Legend labels (level ranges)
        box.text_block(
            "bc",
            legend_labels,
            dy_unit=dy0_labels,
            dy_line=dy_line,
            dx=dx_label,
            reverse=self.setup.reverse_legend,
            ha="left",
            size=font_size,
            family="monospace",
        )

        # Legend color boxes
        colors = self.get_colors()
        if self.setup.reverse_legend:
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

        n_markers = self.mark_release_site + self.mark_field_max
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
        if self.mark_field_max:
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

    # SR_TODO Move into Labels class
    def _format_top_box_subtitle(self):
        setup = self.setup
        if self.name.startswith("ens_thr_agrmt"):
            return self.labels.top_left["subtitle_thr_agrmt_fmt"].format(
                thr=setup.ens_param_thr,
            )
        elif self.name.startswith("ens_cloud_arrival_time"):
            return self.labels.top_left["subtitle_cloud_arrival_time"].format(
                thr=setup.ens_param_thr, mem=setup.ens_param_mem_min,
            )
        else:
            return None

    # SR_TODO Move into Labels class
    # SR_TODO Replace name checks with plot-specific config/setup object
    def _format_legend_box_title(self):
        if self.name.startswith("ens_thr_agrmt"):
            return self.labels.right_top["title"]
        elif self.name.startswith("ens_cloud_arrival_time"):
            return self.labels.right_top["title"]
        else:
            return self.labels.right_top["title_unit"]

    def fill_box_right_bottom(self, box):
        """Fill the bottom box to the right of the map plot."""

        l = self.labels.right_bottom  # noqa:E741
        a = self.field.attrs

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

        labels = self.labels.bottom

        # FLEXPART/model info
        s = self._model_info()  # SR_TODO move into Labels class
        box.text("tl", dx=-0.7, dy=0.5, s=s, size="small")

        # MeteoSwiss Copyright
        box.text("tr", dx=0.7, dy=0.5, s=labels["copyright"], size="small")

    # SR_TODO Move this into Labels class
    # SR_TODO Replace name checks with plot-specific config/setup object
    def _model_info(self):
        if self.name.startswith("ens_"):
            return self.labels.bottom["model_info_ens"]
        else:
            return self.labels.bottom["model_info_det"]

    # SR_TODO Replace name checks with plot-specific config/setup object
    def get_colors(self):
        if self.name.endswith("_mono"):
            return (np.array([(200, 200, 200)]) / 255).tolist()
        else:
            if self.cmap == "flexplot":
                return colors_flexplot(self.n_levels, self.extend)
            else:
                cmap = mpl.cm.get_cmap(self.cmap)
                return [cmap(i / (self.n_levels - 1)) for i in range(self.n_levels)]

    # SR_TODO Replace name checks with plot-specific config/setup object
    def get_levels(self):
        if self.name.startswith("ens_thr_agrmt"):
            n_max = 20  # SR_TMP SR_HC
            return (
                np.arange(
                    n_max - self.d_level * (self.n_levels - 1),
                    n_max + self.d_level,
                    self.d_level,
                )
                + 1
            )
        elif self.name.startswith("ens_cloud_arrival_time"):
            return np.arange(0, self.n_levels) * self.d_level
        elif self.name.endswith("_mono"):
            levels = self.auto_levels_log10(n_levels=9)
            return np.array([levels[0], np.inf])
        else:
            return self.auto_levels_log10()

    def auto_levels_log10(self, n_levels=None, val_max=None):
        if n_levels is None:
            n_levels = self.n_levels
        if val_max is None:
            val_max = self.field.time_stats["max"]
        log10_max = int(np.floor(np.log10(val_max)))
        log10_d = 1
        levels_log10 = np.arange(
            log10_max - (n_levels - 1) * log10_d, log10_max + 0.5 * log10_d, log10_d,
        )
        return 10 ** levels_log10

    @property
    def levels_log10(self):
        return np.log10(self.get_levels())

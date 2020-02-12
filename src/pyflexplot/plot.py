# -*- coding: utf-8 -*-
"""
Plots.
"""
# Standard library
import logging as log
import os
from copy import copy
from textwrap import dedent

# Third-party
import numpy as np
from matplotlib import pyplot as plt

# First-party
from srutils.geo import Degrees

# Local
from .plot_utils import MapAxesRotatedPole
from .plot_utils import SummarizablePlotClass
from .plot_utils import TextBoxAxes
from .plot_utils import ax_w_h_in_fig_coords
from .utils import ParentClass
from .utils import SummarizableClass
from .utils import fmt_float
from .utils import format_level_ranges
from .words import WORDS

# Plot Labels


# SR_TMP TODO Turn this class into some more adequate type (simple dict?)
class DispersionPlotLabels(SummarizableClass):

    summarizable_attrs = []  # SR_TODO

    def __init__(self, lang, words, attrs):

        self.words = words

        w = words
        s = words.symbols
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
        groups["top_left"] = {
            "variable": f"{a.variable.long_name.value}{s_level}",
            "period": f"{integr_op} {period} ({w['since']} +{start})",
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
            "title": f"{name} ({unit})",
            "title_thr_agrmt_fmt": f"{name}\n({s['geq']} {{thr}} {unit_escaped})",
            "release_site": w["release_site"].s,
            "max": w["max"].s,
        }

        # Right-bottom box
        deg_ = f"{w.symbols['deg']}{w.symbols['short_space']}"
        _N = f"{w.symbols['short_space']}{w['north', None, 'abbr']}"
        _E = f"{w.symbols['short_space']}{w['east', None, 'abbr']}"
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
        member_ids = "{:03d}-{:03d}".format(0, 20)  # SR_TMP SR_HC TODO un-hardcode
        model_ens = (
            model
            + (f"{w['ensemble']} ({n_members} {w['member', None, 'pl']}: {member_ids}"),
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


# Plots


class Plot(SummarizablePlotClass, ParentClass):
    """Base class for FLEXPART plots."""

    name = "__base__"
    summarizable_attrs = [
        "name",
        "field",
        "map_conf",
        "dpi",
        "figsize",
        "fig",
        "ax_map",
    ]

    def __init__(self, field, map_conf, *, dpi=None, figsize=None, scale_fact=None):
        """Create an instance of ``Plot``.

        Args:
            field (Field): Data field.

            map_conf (MapAxesConf): Map plot configuration object.

            dpi (float, optional): Plot resolution (dots per inch). Defaults to
                100.0.

            figsize (tuple[float, float], optional): Figure size in inches.
                Defaults to (12.0, 9.0).

            scale_fact (float, optional): Scale factor applied to field.
                Defaults to 1.0.
        """
        self.field = field
        self.map_conf = map_conf
        self.dpi = dpi or 100.0
        self.figsize = figsize or (12.0, 9.0)
        self.scale_fact = scale_fact
        self.field.scale(scale_fact)

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

    def summarize(self, *args, **kwargs):
        data = super().summarize(*args, **kwargs)
        return data


# SR_TMP class DispersionPlot(Plot):
class DispersionPlot_old(Plot):  # SR_TMP
    """Base class for FLEXPART dispersion plots."""

    name = "__base__dispersion__"
    cmap = "flexplot"
    extend = "max"
    n_levels = 9
    draw_colors = True
    draw_contours = False
    mark_field_max = True
    mark_release_site = True
    text_box_setup = {
        "h_rel_t": 0.1,
        "h_rel_b": 0.03,
        "w_rel_r": 0.25,
        "pad_hor_rel": 0.015,
        "h_rel_box_rt": 0.45,
    }
    level_range_style = "base"  # see ``format_level_ranges``
    level_ranges_align = "center"
    lw_frame = 1.0

    summarizable_attrs = Plot.summarizable_attrs + [
        "lang",
        "labels",
        "extend",
        "level_range_style",
        "draw_colors",
        "draw_contours",
        "mark_field_max",
        "mark_release_site",
        "text_box_setup",
        "boxes",
        "reverse_legend",
    ]

    def __init__(
        self, field, *, lang=None, labels=None, reverse_legend=False, **kwargs
    ):
        """Create an instance of ``DispersionPlot``.

        Args:
            *args: Additional positional arguments passed to ``Plot.__init__``.

            lang (str, optional): Language, e.g., 'de' for German. Defaults to
                'en' (English).

            **kwargs: Additional keyword arguments passed to ``Plot.__init__``.
        """
        super().__init__(field, **kwargs)
        self.lang = lang or "en"
        self.reverse_legend = reverse_legend or False

        if labels is None:
            labels = DispersionPlotLabels(lang, WORDS, field.attrs)
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

        self.define_colors()
        self.define_levels()

        self.create()

    def create(self):
        """Create plot."""

        # Prepare plot
        self.prepare_plot()

        # Plot particle concentration field
        self.draw_map_plot()

        # Text boxes around plot
        self.add_text_boxes()
        self.draw_boxes()

    def summarize(self, *args, **kwargs):
        data = super().summarize(*args, **kwargs)
        # ...  SR_TODO
        return data

    def fld_nonzero(self):
        return np.where(self.field.fld > 0, self.field.fld, np.nan)

    def draw_map_plot(self):
        """Plot the particle concentrations onto the map."""

        # Plot concentrations
        h_con = self._draw_colors_contours()

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

    def _draw_colors_contours(self):

        field = np.log10(self.fld_nonzero())

        if not self.draw_colors:
            h_col = None
        else:
            h_col = self.ax_map.contourf(
                field, levels=self.levels_log10, colors=self.colors, extend=self.extend,
            )

        if not self.draw_contours:
            h_con = None
        else:
            h_con = self.ax_map.contour(
                field, levels=self.levels_log10, colors="black", linewidths=1,
            )
            return (h_col, h_con)

    def draw_boxes(self):
        for fill_box, box in self.boxes.items():
            fill_box(box)
            box.draw()

    # SR_TODO move parameter descriptions from docstring to where they belong!
    def add_text_boxes(self):
        """Add empty text boxes to the figure around the map plot.

        Args:
            h_rel_t (float, optional): Height of top box as a fraction of the
                height of the map plot. Defaults to <TODO>.

            h_rel_b (float, optional): Height of bottom box as a fraction of
                the height of the map plot. Defaults to <TODO>.

            w_rel_r (float, optional): Width of the right boxes as a fraction
                of the width of the map plot. Default to <TODO>.

            pad_hor_rel (float, optional): Padding between map plot and the
                text boxes as a fraction of the map plot width. The same
                absolute padding is used in the horizontal and vertical
                direction. Defaults to <TODO>.

            h_rel_box_rt (float, optional): Height of the top box to the right
                of the map plot as a fraction of the combined height of both
                right boxees. Defaults to <TODO>.

        """

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

        # Add axes for text boxes (one on top, two to the right)
        self.boxes = {
            self.fill_box_top: TextBoxAxes(
                name="top",
                fig=self.fig,
                ax_ref=self.ax_map.ax,
                rect=[
                    x0_map,
                    y0_map + pad_ver + h_map,
                    w_map + pad_hor + w_box,
                    h_box_t,
                ],
                lw_frame=self.lw_frame,
            ),
            self.fill_box_right_top: TextBoxAxes(
                name="right/top",
                fig=self.fig,
                ax_ref=self.ax_map.ax,
                rect=[
                    x0_map + pad_hor + w_map,
                    y0_map + 0.5 * pad_ver + (1.0 - h_rel_box_rt) * h_map,
                    w_box,
                    h_rel_box_rt * h_map - 0.5 * pad_ver,
                ],
                lw_frame=self.lw_frame,
            ),
            self.fill_box_right_bottom: TextBoxAxes(
                name="right/bottom",
                fig=self.fig,
                ax_ref=self.ax_map.ax,
                rect=[
                    x0_map + pad_hor + w_map,
                    y0_map,
                    w_box,
                    (1.0 - h_rel_box_rt) * h_map - 0.5 * pad_ver,
                ],
                lw_frame=self.lw_frame,
            ),
            self.fill_box_bottom: TextBoxAxes(
                name="bottom",
                fig=self.fig,
                ax_ref=self.ax_map.ax,
                rect=[x0_map, y0_map - h_box_b, w_map + pad_hor + w_box, h_box_b,],
                lw_frame=0.0,
            ),
        }

    def fill_box_top(self, box, *, skip_pos=None):
        """Fill the box above the map plot.

        Args:
            box (TextBoxAxes): The box to fill.

            skip_pos (set, optional): Positions (e.g., 'tl' for top-left; see
                ``pyflexplot.plot.TextBoxLocation`` for all options) to be skipped.
                Defaults to ``{}``.

        """

        a = self.field.attrs
        l = self.labels
        w = self.labels.words

        if skip_pos is None:
            skip_pos = {}

        if not "tl" in skip_pos:
            # Top left: variable
            s = a.variable.long_name.value
            _lvl = a.variable.fmt_level_range()
            if _lvl:
                s += f" {w['at', None, 'level']} {_lvl}"
            box.text("tl", s, size="x-large")

        if not "bl" in skip_pos:
            # Bottom left: integration time & level range
            itype = a.simulation.integr_type.value
            s = {
                "sum": w["summed_over"].s,
                "mean": w["averaged_over"].s,
                "accum": w["accumulated_over"].s,
            }[itype]
            since = w["since"].s
            sim = a.simulation
            start = sim.integr_start.format(rel=True)
            s += f" {sim.fmt_integr_period()} ({since} +{start})"
            if self.scale_fact is not None:
                s += f" ({self.scale_fact}x)"
            box.text("bl", s, size="large")

        dx_center = 0

        if not "tc" in skip_pos:
            # Top center: species
            s = f"{w['substance']}: {a.species.name.format(join=' + ')}"
            box.text("tc", s, dx=dx_center, size="large")

        if not "bc" in skip_pos:
            # Bottom center: release site
            s = f"{l['w']['release_site']}: {a.release.site_name.value}"
            box.text("bc", s, dx=dx_center, size="large")

        if not "tr" in skip_pos:
            # Top right: datetime
            timestep_fmtd = a.simulation.now.format()
            s = f"{timestep_fmtd}"
            box.text("tr", s, size="large")

        if not "br" in skip_pos:
            # Bottom right: time into simulation
            _now_rel = a.simulation.now.format(rel=True)
            s = f"{_now_rel}"
            box.text("br", s, size="large")

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
        s = self._format_legend_box_title(labels)
        box.text("tc", s=s, dy=1.5, size="large")

        # Format level ranges (contour plot legend)
        legend_labels = format_level_ranges(
            levels=self.levels,
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
            reverse=self.reverse_legend,
            ha="left",
            size=font_size,
            family="monospace",
        )

        # Legend color boxes
        colors = self.colors
        if self.reverse_legend:
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

    # SR_TODO move into Labels class
    def _format_legend_box_title(self, labels):
        return labels["title"]

    def fill_box_right_bottom(self, box):
        """Fill the bottom box to the right of the map plot."""

        l = self.labels.right_bottom
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

    # SR_TODO move this into Labels class
    def _model_info(self):
        return self.labels.bottom["model_info_det"]

    def define_colors(self):
        if self.cmap == "flexplot":
            self.colors = colors_flexplot(self.n_levels, self.extend)
        else:
            cmap = mpl.cm.get_cmap(self.cmap)
            colors_core = [cmap(i / (self.n_levels - 1)) for i in range(self.n_levels)]

    def define_levels(self):
        self.levels = self.auto_levels_log10()

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
        return np.log10(self.levels)


# SR_TMP < TODO merge back into DispersionPlot[_0] once new layout accepted
class DispersionPlot(DispersionPlot_old):
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
                rect=[x0_map, y0_map + pad_ver + h_map, w_map, h_box_t,],
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
                rect=[x0_map, y0_map - h_box_b, w_map + pad_hor + w_box, h_box_b,],
                **{**text_box_conf, "lw_frame": None},
            ),
        }

    def fill_box_top_left(self, box, *, skip_pos=None):
        """Fill the box above the map plot."""

        labels = self.labels.top_left

        if skip_pos is None:
            skip_pos = {}

        # Top-left: Variable name etc.
        if not "tl" in skip_pos:
            box.text("tl", labels["variable"], size="x-large")

        # Bottom-left: Integration time etc.
        if not "bl" in skip_pos:
            s = labels["period"]
            if self.scale_fact is not None:
                s += f" ({self.scale_fact}x)"
            box.text("bl", s, size="large")

        # Top-right: Time step
        if not "tr" in skip_pos:
            box.text("tr", labels["timestep"], size="large")

        # Bottom-right: Time since release start
        if not "br" in skip_pos:
            box.text("br", labels["time_since_release_start"], size="large")

        return box

    def fill_box_top_right(self, box, *, skip_pos=None):
        """Fill the box to the top-right of the map plot."""

        labels = self.labels.top_right

        if skip_pos is None:
            skip_pos = []

        if not "tc" in skip_pos:
            # Top center: species
            box.text("tc", labels["species"], size="large")

        if not "bc" in skip_pos:
            # Bottom center: release site (shrunk/truncated to fit box)
            s, size = box.fit_text(labels["site"], "large", n_shrink_max=1)
            box.text("bc", s, size=size)

        return box


# SR_TMP >


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


# Deterministic Simulation


class Plot_Concentration(DispersionPlot):
    """FLEXPART plot of particle concentration at a certain level."""

    name = "concentration"
    n_levels = 8


class Plot_Deposition(DispersionPlot):
    """FLEXPART plot of surface deposition."""

    name = "deposition"
    n_levels = 9


class Plot_AffectedArea(DispersionPlot):
    """FLEXPART plot of area affected by surface deposition."""

    name = "affected_area"


class Plot_AffectedAreaMono(Plot_AffectedArea):
    """FLEXPART plot of area affected by surface deposition (mono)."""

    name = "affected_area_mono"
    extend = "none"
    n_levels = 1

    def define_colors(self):
        self.colors = (np.array([(200, 200, 200)]) / 255).tolist()

    def define_levels(self):
        levels = self.auto_levels_log10(n_levels=9)
        self.levels = np.array([levels[0], np.inf])


# Ensemble Simulation


class Plot_Ens(Plot):
    name = "ens"

    text_box_setup = {
        "h_rel_t": 0.12,
        "h_rel_b": 0.03,
        "w_rel_r": 0.25,
        "pad_hor_rel": 0.015,
        "h_rel_box_rt": 0.46,
    }

    # SR_TODO move this into Labels class
    def _model_info(self):
        return self.labels.bottom["model_info_ens"]


class Plot_EnsMean_Concentration(Plot_Ens, Plot_Concentration):
    name = "ens_mean_concentration"


class Plot_EnsMean_Deposition(Plot_Ens, Plot_Deposition):
    name = "ens_mean_deposition"


class Plot_EnsMeanAffectedArea(Plot_Ens, Plot_AffectedArea):
    name = "ens_mean_affected_area"


class Plot_EnsMeanAffectedAreaMono(Plot_Ens, Plot_AffectedAreaMono):
    name = "ens_mean_affected_area_mono"


class Plot_EnsThrAgrmt(Plot_Ens):
    name = "ens_thr_agrmt"

    n_levels = 7
    d_level = 2
    extend = "min"
    level_range_style = "int"  # see ``format_level_ranges``
    level_ranges_align = "left"
    mark_field_max = False

    def define_levels(self):
        n_max = 20  # SR_TMP SR_HC
        d = self.d_level
        self.levels = (
            np.arange(
                n_max - self.d_level * (self.n_levels - 1),
                n_max + self.d_level,
                self.d_level,
            )
            + 1
        )

    def _draw_colors_contours(self):

        # If upper end of range is closed, color areas beyond black
        colors_plot = copy(self.colors)
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
                field, levels=self.levels, colors=colors_plot, extend=extend_plot,
            )

        if not self.draw_contours:
            h_con = None
        else:
            h_con = self.ax_map.contour(
                field, levels=self.levels, colors="black", linewidths=1,
            )

        return (h_col, h_con)

    # SR_TODO move into Labels class
    def _format_legend_box_title(self, labels):
        thr = self.field.field_specs.ens_var_setup["thr"]
        return labels["title_thr_agrmt_fmt"].format(thr=thr)

    # SR_TODO remove if not triggered
    def _format_level(self, lvl):
        raise Exception(f"{type(self).__name__}._format_level used after all!")
        return f"{lvl}"


class Plot_EnsThrAgrmt_Concentration(Plot_EnsThrAgrmt, Plot_Concentration):
    name = "ens_thr_agrmt_concentration"


class Plot_EnsThrAgrmt_Deposition(Plot_EnsThrAgrmt, Plot_Deposition):
    name = "ens_thr_agrmt_deposition"

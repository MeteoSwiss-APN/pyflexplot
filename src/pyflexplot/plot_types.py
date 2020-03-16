# -*- coding: utf-8 -*-
"""
Plot types.
"""
# Standard library
from typing import Optional

# Third-party
import matplotlib as mpl
import numpy as np

# Local
from .attr import AttrMult
from .setup import Setup
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


class PlotConfig:
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

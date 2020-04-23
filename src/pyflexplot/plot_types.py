# -*- coding: utf-8 -*-
"""
Plot types.
"""
# Standard library
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

# Third-party
import matplotlib as mpl
import numpy as np

# Local
from .data import Field
from .meta_data import MetaData
from .meta_data import format_level_range
from .plot_lib import MapAxesConf
from .setup import InputSetup
from .utils import summarizable
from .words import SYMBOLS
from .words import WORDS


def create_map_conf(field: Field) -> MapAxesConf:
    domain = field.var_setups.collect_equal("domain")
    model = field.nc_meta_data["analysis"]["model"]

    conf_base = {"lang": field.var_setups.collect_equal("lang")}

    conf_model_ifs = {"geo_res": "110m"}
    conf_model_cosmo = {"geo_res": "10m"}

    conf_domain_global = {
        "geo_res_cities": "110m",
        "geo_res_rivers": "110m",
        "min_city_pop": 1_000_000,
        "zoom_fact": 1.02,
    }
    conf_domain_eu = {
        "geo_res_cities": "50m",
        "geo_res_rivers": "50m",
        "min_city_pop": 300_000,
        "zoom_fact": 1.02,
    }
    conf_domain_ch = {
        "geo_res_cities": "10m",
        "geo_res_rivers": "10m",
        "min_city_pop": 0,
        "ref_dist_conf": {"dist": 25},
        "rel_offset": (-0.02, 0.045),
    }

    if (model, domain) in [("cosmo1", "auto"), ("cosmo2", "auto")]:
        conf = {**conf_base, **conf_model_cosmo, **conf_domain_eu}
    elif (model, domain) == ("cosmo1", "ch"):
        conf = {**conf_base, **conf_model_cosmo, **conf_domain_ch, "zoom_fact": 3.6}
    elif (model, domain) == ("cosmo2", "ch"):
        conf = {**conf_base, **conf_model_cosmo, **conf_domain_ch, "zoom_fact": 3.2}
    elif (model, domain) == ("ifs", "auto"):
        conf = {**conf_base, **conf_model_ifs, **conf_domain_global}
    else:
        raise Exception(f"unknown domain '{domain}' for model '{model}'")

    return MapAxesConf(**conf)


# SR_TMP TODO Turn into dataclass or the like.
@summarizable
class PlotLabels:

    words = WORDS
    symbols = SYMBOLS

    def __init__(self, lang, mdata):
        """Create an instance of ``PlotLabels``."""
        self.mdata = mdata

        self.words.set_default_lang(lang)

        # Declare groups
        self.top_left: Dict[str, Any] = self._init_top_left()
        self.top_right: Dict[str, Any] = self._init_top_right()
        self.right_top: Dict[str, Any] = self._init_right_top()
        self.right_bottom: Dict[str, Any] = self._init_right_bottom()
        self.bottom: Dict[str, Any] = self._init_bottom()

    def _init_group(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and set group values."""
        for key, val in values.copy().items():
            if isinstance(val, str):
                # Capitalize first letter only (even if it's a space!)
                values[key] = list(val)[0].capitalize() + val[1:]
        return values

    def _init_top_left(self) -> Dict[str, Any]:
        level = format_level_range(
            value_bottom=self.mdata.variable_level_bot.value,
            value_top=self.mdata.variable_level_top.value,
            unit_bottom=self.mdata.variable_level_bot_unit.value,
            unit_top=self.mdata.variable_level_top_unit.value,
        )
        if not level:
            s_level = ""
        else:
            s_level = f" {self.words['at', None, 'level']} {level}"
        integr_op = self.words[
            {
                "sum": "summed_over",
                "mean": "averaged_over",
                "accum": "accumulated_over",
            }[self.mdata.simulation_integr_type.value]
        ].s
        time_rels = self.mdata.simulation_now_rel
        period = self.mdata.simulation_fmt_integr_period()
        start = self.mdata.simulation_integr_start_rel
        unit = self.mdata.variable_unit
        unit_escaped = str(unit).replace("{", "{{").replace("}", "}}")
        ts_abs = self.mdata.simulation_now
        ts_rel = self.mdata.simulation_now_rel
        return self._init_group(
            {
                "variable": f"{self.mdata.variable_long_name.value}{s_level}",
                "period": f"{integr_op} {period} ({self.words['since']} +{start})",
                "subtitle_thr_agrmt_fmt": (
                    f"Cloud: {self.symbols['geq']} {{thr}} {unit_escaped}"
                ),
                "subtitle_cloud_arrival_time": (
                    f"Cloud: {self.symbols['geq']} {{thr}} {unit_escaped}; "
                    f"members: {self.symbols['geq']} {{mem}}"
                ),
                "timestep": f"{ts_abs} (+{ts_rel})",
                "time_since_release_start": (
                    f"{time_rels} {self.words['since']} {self.words['release_start']}"
                ),
            },
        )

    def _init_top_right(self) -> Dict[str, Any]:
        return self._init_group(
            {
                "species": f"{self.mdata.species_name}",
                "site": f"{self.words['site']}: {self.mdata.release_site_name.value}",
            },
        )

    def _init_right_top(self) -> Dict[str, Any]:
        name = self.mdata.variable_short_name
        unit = self.mdata.variable_unit
        return self._init_group(
            {
                "title": f"{name}",
                "title_unit": f"{name} ({unit})",
                "release_site": self.words["release_site"].s,
                "max": self.words["max"].s,
            },
        )

    def _init_right_bottom(self) -> Dict[str, Any]:
        deg_ = f"{self.symbols['deg']}{self.symbols['short_space']}"
        north = f"{self.symbols['short_space']}{self.words['north', None, 'abbr']}"
        east = f"{self.symbols['short_space']}{self.words['east', None, 'abbr']}"
        return self._init_group(
            {
                "title": self.words["release"].t,
                "start": self.words["start"].s,
                "end": self.words["end"].s,
                "latitude": self.words["latitude"].s,
                "longitude": self.words["longitude"].s,
                "lat_deg_fmt": (
                    f"{{d}}{deg_}{{m}}'{north} ({{f:.4f}}{self.words['degN']})"
                ),
                "lon_deg_fmt": (
                    f"{{d}}{deg_}{{m}}'{east} ({{f:.4f}}{self.words['degE']})"
                ),
                "height": self.words["height"].s,
                "rate": self.words["rate"].s,
                "mass": self.words["total_mass"].s,
                "site": self.words["site"].s,
                "release_site": self.words["release_site"].s,
                "max": self.words["max"].s,
                "name": self.words["substance"].s,
                "half_life": self.words["half_life"].s,
                "deposit_vel": self.words["deposition_velocity", None, "abbr"].s,
                "sediment_vel": self.words["sedimentation_velocity", None, "abbr"].s,
                "washout_coeff": self.words["washout_coeff"].s,
                "washout_exponent": self.words["washout_exponent"].s,
            },
        )

    def _init_bottom(self) -> Dict[str, Any]:
        model = self.mdata.simulation_model_name.value
        n_members = 21  # SR_TMP SR_HC TODO un-hardcode
        ens_member_id = "{:03d}-{:03d}".format(0, 20)  # SR_TMP SR_HC TODO un-hardcode
        model_ens = (
            f"{model} {self.words['ensemble']} ({n_members} "
            f"{self.words['member', None, 'pl']}: {ens_member_id}"
        )
        start = self.mdata.simulation_start
        model_info_fmt = (
            f"{self.words['flexpart']} {self.words['based_on']} {model}, {start}"
        )
        return self._init_group(
            {
                "model_info_det": model_info_fmt.format(m=model),
                "model_info_ens": model_info_fmt.format(m=model_ens),
                "copyright": f"{self.symbols['copyright']}{self.words['meteoswiss']}",
            },
        )


def colors_flexplot(n_levels: int, extend: str) -> Sequence[Tuple[int, int, int]]:

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
    colors_core_4 = [colors_core_8[i] for i in (1, 2, 4, 7)]

    try:
        colors_core = {
            5: colors_core_4,
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
        self, setup: InputSetup, mdata: MetaData, labels: Optional[PlotLabels] = None,
    ) -> None:
        self.setup = setup
        self.mdata = mdata
        self.labels: PlotLabels = labels or PlotLabels(setup.lang, mdata)

        self.figsize = (12.0, 9.0)
        self.reverse_legend = False

    # SR_TMP <<<
    @property
    def text_box_setup(self) -> Dict[str, float]:
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
    def extend(self) -> str:
        return {
            "ens_thr_agrmt": "min",
            "ens_cloud_arrival_time": "max",
            "affected_area_mono": "none",
        }.get(self.setup.plot_type, "max")

    # SR_TMP
    @property
    def n_levels(self) -> int:
        return {
            "ens_thr_agrmt": 7,
            "ens_cloud_arrival_time": 5,
            "affected_area_mono": 1,
        }.get(
            self.setup.plot_type,
            {"concentration": 8, "deposition": 9}[self.setup.variable],
        )

    # SR_TMP
    @property
    def d_level(self) -> int:
        try:
            return {"ens_thr_agrmt": 2, "ens_cloud_arrival_time": 3}[
                self.setup.plot_type
            ]
        except KeyError:
            raise NotImplementedError("plot_type", self.setup.plot_type)

    # SR_TMP
    @property
    def level_range_style(self) -> str:
        return {"ens_thr_agrmt": "int", "ens_cloud_arrival_time": "int"}.get(
            self.setup.plot_type, "base"
        )

    # SR_TMP
    @property
    def level_ranges_align(self) -> str:
        return {"ens_thr_agrmt": "left", "ens_cloud_arrival_time": "left"}.get(
            self.setup.plot_type, "center"
        )

    # SR_TMP
    @property
    def mark_field_max(self) -> bool:
        return {"ens_thr_agrmt": False, "ens_cloud_arrival_time": False}.get(
            self.setup.plot_type, True
        )

    # SR_TMP
    @property
    def top_box_subtitle(self) -> Optional[str]:
        setup = self.setup
        labels = self.labels.top_left
        if setup.plot_type == "ens_thr_agrmt":
            return labels["subtitle_thr_agrmt_fmt"].format(thr=setup.ens_param_thr)
        elif setup.plot_type == "ens_cloud_arrival_time":
            return labels["subtitle_cloud_arrival_time"].format(
                thr=setup.ens_param_thr, mem=setup.ens_param_mem_min,
            )
        return None

    # SR_TMP
    @property
    def legend_box_title(self) -> str:
        labels = self.labels.right_top
        if self.setup.plot_type == "ens_thr_agrmt":
            return labels["title"]
        elif self.setup.plot_type == "ens_cloud_arrival_time":
            return labels["title"]
        else:
            return labels["title_unit"]

    # SR_TMP
    @property
    def model_info(self) -> str:
        labels = self.labels.bottom
        if self.setup.simulation_type == "deterministic":
            return labels["model_info_det"]
        elif self.setup.simulation_type == "ensemble":
            return labels["model_info_ens"]
        raise NotImplementedError("simulation_type", self.setup.simulation_type)

    # SR_TMP
    def get_colors(self, cmap) -> Sequence[Tuple[int, int, int]]:
        if self.setup.plot_type == "affected_area_mono":
            return (np.array([(200, 200, 200)]) / 255).tolist()
        elif cmap == "flexplot":
            return colors_flexplot(self.n_levels, self.extend)
        else:
            cmap = mpl.cm.get_cmap(cmap)
            return [cmap(i / (self.n_levels - 1)) for i in range(self.n_levels)]

    # SR_TMP
    @property
    def levels_scale(self) -> str:
        if self.setup.plot_type in ["ens_thr_agrmt", "ens_cloud_arrival_time"]:
            return "lin"
        return "log"

    # SR_TMP
    def get_levels(self, time_stats) -> Sequence[float]:
        n_levels = self.n_levels
        if self.setup.plot_type == "ens_thr_agrmt":
            d_level = self.d_level
            n_max = 20  # SR_TMP SR_HC
            return (
                np.arange(n_max - d_level * (n_levels - 1), n_max + d_level, d_level)
                + 1
            )
        elif self.setup.plot_type == "ens_cloud_arrival_time":
            return np.arange(0, n_levels) * self.d_level
        elif self.setup.plot_type == "affected_area_mono":
            levels = self._auto_levels_log10(n_levels=9, val_max=time_stats["max"])
            return np.array([levels[0], np.inf])
        else:
            return self._auto_levels_log10(val_max=time_stats["max"])

    def _auto_levels_log10(
        self, n_levels: Optional[int] = None, val_max: Optional[float] = None,
    ) -> Sequence[float]:
        if n_levels is None:
            n_levels = self.n_levels
        log10_max = int(np.floor(np.log10(val_max)))
        log10_d = 1
        return 10 ** np.arange(
            log10_max - (n_levels - 1) * log10_d, log10_max + 0.5 * log10_d, log10_d,
        )

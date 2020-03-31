# -*- coding: utf-8 -*-
"""
Plot types.
"""
# Standard library
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

# Third-party
import matplotlib as mpl
import numpy as np

# Local
from .meta_data import MetaDataCollection
from .plot_lib import MapAxesConf
from .setup import InputSetup
from .utils import summarizable
from .words import SYMBOLS
from .words import WORDS


@dataclass
class MapAxesConf_Cosmo1(MapAxesConf):
    geo_res: str = "10m"
    geo_res_cities: str = "50m"
    geo_res_rivers: str = "50m"
    zoom_fact: float = 1.02
    min_city_pop: int = 300_000


@dataclass
class MapAxesConf_Cosmo1_CH(MapAxesConf_Cosmo1):
    geo_res_cities: str = "10m"
    geo_res_rivers: str = "10m"
    min_city_pop: int = 0
    # SR_TODO Determine the model from the data! (e.g., COSMO-1 v. COSMO-2 v. COSMO-E)
    # rel_offset: Tuple[float] = (0.037, 0.106)  # suitable for ensemble (COSMO-2?)
    # zoom_fact: float = 3.2  # suitable for ensemble (i.e., COSMO-2?)
    rel_offset: Tuple[float, float] = (-0.02, 0.045)
    zoom_fact: float = 3.6

    def __post_init__(self):
        super().__post_init__()
        self.ref_dist_conf.dist = 25


# SR_TMP TODO Turn into dataclass or the like.
@summarizable
class PlotLabels:

    words = WORDS
    symbols = SYMBOLS

    def __init__(self, lang, mdata):
        """Create an instance of ``PlotLabels``."""
        self.mdata = mdata

        words = self.words
        symbs = self.symbols
        mdata = self.mdata

        words.set_default_lang(lang)

        # Declare groups
        self.top_left: Dict[str, Any]
        self.top_right: Dict[str, Any]
        self.right_top: Dict[str, Any]
        self.right_bottom: Dict[str, Any]
        self.bottom: Dict[str, Any]

        def set_group(name, values: Dict[str, Any]) -> None:
            """Preprocess and set group values."""
            for key, val in values.copy().items():
                if isinstance(val, str):
                    # Capitalize first letter only (even if it's a space!)
                    values[key] = list(val)[0].capitalize() + val[1:]
            setattr(self, name, values)

        # Top-left box
        level = mdata.variable.fmt_level_range()
        s_level = f" {words['at', None, 'level']} {level}" if level else ""
        integr_op = words[
            {
                "sum": "summed_over",
                "mean": "averaged_over",
                "accum": "accumulated_over",
            }[mdata.simulation.integr_type.value]
        ].s
        ts = mdata.simulation.now
        time_rels = mdata.simulation.now.format(
            rel=True, rel_start=mdata.release.start.value
        )
        period = mdata.simulation.fmt_integr_period()
        start = mdata.simulation.integr_start.format(rel=True)
        unit_escaped = mdata.variable.unit.format(escape_format=True)
        set_group(
            "top_left",
            {
                "variable": f"{mdata.variable.long_name.value}{s_level}",
                "period": f"{integr_op} {period} ({words['since']} +{start})",
                "subtitle_thr_agrmt_fmt": (
                    f"Cloud: {symbs['geq']} {{thr}} {unit_escaped}"
                ),
                "subtitle_cloud_arrival_time": (
                    f"Cloud: {symbs['geq']} {{thr}} {unit_escaped}; "
                    f"members: {symbs['geq']} {{mem}}"
                ),
                "timestep": f"{ts.format(rel=False)} (+{ts.format(rel=True)})",
                "time_since_release_start": (
                    f"{time_rels} {words['since']} {words['release_start']}"
                ),
            },
        )

        # Top-right box
        self.top_right = {
            "species": f"{mdata.species.name.format(join=' + ')}",
            "site": f"{words['site']}: {mdata.release.site_name.value}",
        }

        # Right-top box
        name = mdata.variable.short_name.format()
        unit = mdata.variable.unit.format()
        unit_escaped = mdata.variable.unit.format(escape_format=True)
        set_group(
            "right_top",
            {
                "title": f"{name}",
                "title_unit": f"{name} ({unit})",
                "release_site": words["release_site"].s,
                "max": words["max"].s,
            },
        )

        # Right-bottom box
        deg_ = f"{symbs['deg']}{symbs['short_space']}"
        _N = f"{symbs['short_space']}{words['north', None, 'abbr']}"
        _E = f"{symbs['short_space']}{words['east', None, 'abbr']}"
        set_group(
            "right_bottom",
            {
                "title": words["release"].t,
                "start": words["start"].s,
                "end": words["end"].s,
                "latitude": words["latitude"].s,
                "longitude": words["longitude"].s,
                "lat_deg_fmt": f"{{d}}{deg_}{{m}}'{_N} ({{f:.4f}}{words['degN']})",
                "lon_deg_fmt": f"{{d}}{deg_}{{m}}'{_E} ({{f:.4f}}{words['degE']})",
                "height": words["height"].s,
                "rate": words["rate"].s,
                "mass": words["total_mass"].s,
                "site": words["site"].s,
                "release_site": words["release_site"].s,
                "max": words["max"].s,
                "name": words["substance"].s,
                "half_life": words["half_life"].s,
                "deposit_vel": words["deposition_velocity", None, "abbr"].s,
                "sediment_vel": words["sedimentation_velocity", None, "abbr"].s,
                "washout_coeff": words["washout_coeff"].s,
                "washout_exponent": words["washout_exponent"].s,
            },
        )

        # Bottom box
        model = mdata.simulation.model_name.value
        n_members = 21  # SR_TMP SR_HC TODO un-hardcode
        ens_member_id = "{:03d}-{:03d}".format(0, 20)  # SR_TMP SR_HC TODO un-hardcode
        model_ens = (
            f"{model} {words['ensemble']} ({n_members} {words['member', None, 'pl']}: "
            f"{ens_member_id}"
        )
        start = mdata.simulation.start.format()
        model_info_fmt = f"{words['flexpart']} {words['based_on']} {model}, {start}"
        set_group(
            "bottom",
            {
                "model_info_det": model_info_fmt.format(m=model),
                "model_info_ens": model_info_fmt.format(m=model_ens),
                "copyright": f"{symbs['copyright']}{words['meteoswiss']}",
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
        self,
        setup: InputSetup,
        mdata: MetaDataCollection,
        labels: Optional[PlotLabels] = None,
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
            "ens_cloud_arrival_time": 9,
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

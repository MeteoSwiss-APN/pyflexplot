"""Tests for module ``pyflexplot.plotting.coord_trans``."""
# Standard library
import dataclasses as dc
from typing import Tuple

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cartopy.crs import PlateCarree

# First-party
from pyflexplot.plotting.coord_trans import CoordinateTransformer


@dc.dataclass
class Cfg:
    xy_in: Tuple[float, float]
    xy_out: Tuple[float, float]
    c_map: float = 0.0
    c_data: float = 0.0
    c_geo: float = 0.0


class TestRegLatLon:
    fig, ax = plt.subplots()
    ax.set_xlim(-20, 80)
    ax.set_ylim(20, 60)

    def trans(self, cfg: Cfg) -> CoordinateTransformer:
        return CoordinateTransformer(
            trans_axes=self.ax.transAxes,
            trans_data=self.ax.transData,
            proj_map=PlateCarree(central_longitude=cfg.c_map),
            proj_data=PlateCarree(central_longitude=cfg.c_data),
            proj_geo=PlateCarree(central_longitude=cfg.c_geo),
            invalid_ok=False,
        )

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg((0.2, -0.5), (0, 0)),  # [cfg0]
            Cfg((0.5, 0.5), (30, 40)),  # [cfg1]
            Cfg((1.1, -0.25), (90, 10)),  # [cfg2]
            Cfg((0.2, -0.5), (180, 0), c_map=180),  # [cfg3]
            Cfg((-1.6, -0.5), (0, 0), c_map=180),  # [cfg4]
        ],
    )
    def test_axes_to_geo(self, cfg):
        trans = self.trans(cfg)
        assert np.allclose(trans.axes_to_geo(*cfg.xy_in), cfg.xy_out)

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg((0, 0), (0.2, -0.5)),  # [cfg0]
            Cfg((30, 40), (0.5, 0.5)),  # [cfg1]
            Cfg((90, 10), (1.1, -0.25)),  # [cfg2]
            Cfg((180, 0), (0.2, -0.5), c_map=180),  # [cfg3]
            Cfg((0, 0), (-1.6, -0.5), c_map=180),  # [cfg4]
        ],
    )
    def test_geo_to_axes(self, cfg):
        trans = self.trans(cfg)
        assert np.allclose(trans.geo_to_axes(*cfg.xy_in), cfg.xy_out)

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg((20, 50), (20, 50)),  # [cfg0]
            Cfg((30, 40), (30, 40)),  # [cfg1]
            Cfg((90, 10), (90, 10)),  # [cfg2]
            Cfg((20, 50), (20, 50), c_map=180),  # [cfg3]
        ],
    )
    def test_data_to_geo(self, cfg):
        trans = self.trans(cfg)
        assert np.allclose(trans.data_to_geo(*cfg.xy_in), cfg.xy_out)

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg((20, 50), (20, 50)),  # [cfg0]
            Cfg((30, 40), (30, 40)),  # [cfg1]
            Cfg((90, 10), (90, 10)),  # [cfg2]
            Cfg((20, 50), (20, 50), c_map=180),  # [cfg3]
        ],
    )
    def test_geo_to_data(self, cfg):
        trans = self.trans(cfg)
        assert np.allclose(trans.geo_to_data(*cfg.xy_in), cfg.xy_out)

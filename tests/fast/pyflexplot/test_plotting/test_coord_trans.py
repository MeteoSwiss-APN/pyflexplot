"""Tests for module ``pyflexplot.plotting.coord_trans``."""
# Standard library
from dataclasses import dataclass
from typing import Tuple

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cartopy.crs import PlateCarree

# First-party
from pyflexplot.plotting.coord_trans import CoordinateTransformer


@dataclass
class Cfg:
    xy_in: Tuple[float, float]
    xy_out: Tuple[float, float]
    clon_map: float = 0.0
    clon_data: float = 0.0
    clon_geo: float = 0.0

    @property
    def clons(self):
        return {
            "clon_map": self.clon_map,
            "clon_data": self.clon_data,
            "clon_geo": self.clon_geo,
        }


class TestRegLatLon:
    fig, ax = plt.subplots()
    ax.set_xlim(-20, 80)
    ax.set_ylim(20, 60)

    def trans(self, clon_map=0.0, clon_data=0.0, clon_geo=0.0):
        return CoordinateTransformer(
            trans_axes=self.ax.transAxes,
            trans_data=self.ax.transData,
            proj_map=PlateCarree(central_longitude=clon_map),
            proj_data=PlateCarree(central_longitude=clon_data),
            proj_geo=PlateCarree(central_longitude=clon_geo),
            invalid_ok=False,
        )

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg((20, 50), (0.4, 0.75)),  # [cfg0]
            Cfg((30, 40), (0.5, 0.5)),  # [cfg1]
            Cfg((90, 10), (1.1, -0.25)),  # [cfg2]
        ],
    )
    def test_geo_to_axes(self, cfg):
        trans = self.trans(**cfg.clons)
        assert np.allclose(trans.geo_to_axes(*cfg.xy_in), cfg.xy_out)

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg((0.4, 0.75), (20, 50)),  # [cfg0]
            Cfg((0.5, 0.5), (30, 40)),  # [cfg1]
            Cfg((1.1, -0.25), (90, 10)),  # [cfg2]
        ],
    )
    def test_axes_to_geo(self, cfg):
        trans = self.trans(**cfg.clons)
        assert np.allclose(trans.axes_to_geo(*cfg.xy_in), cfg.xy_out)

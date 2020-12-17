"""Test for module ``pyflexplot.plotting.proj_bbox``."""
# Standard library
from dataclasses import dataclass

# Third-party
import matplotlib.pyplot as plt
import pytest

# First-party
from pyflexplot.plotting.proj_bbox import ProjectedBoundingBox
from pyflexplot.plotting.proj_bbox import Projections


@dataclass
class Cfg:
    clon: float = 0.0
    curr_proj: str = "axes"
    lon0: float = -180.0
    lon1: float = 180.0
    lat0: float = -90.0
    lat1: float = 90.0


@pytest.mark.skip("unfinished")
class TestRegLatLon:
    fig, ax = plt.subplots()
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 45)

    def bbox(self, cfg: Cfg) -> ProjectedBoundingBox:
        return ProjectedBoundingBox(
            ax=self.ax,
            projs=Projections.create_regular(clon=cfg.clon),
            curr_proj=cfg.curr_proj,
            lon0=cfg.lon0,
            lon1=cfg.lon1,
            lat0=cfg.lat0,
            lat1=cfg.lat1,
        )

    def test(self):
        cfg = Cfg(lon0=160, lon1=180, lat0=46, lat1=60, clon=180)
        bbox = self.bbox(cfg)
        bbox.to_data()
        # breakpoint()

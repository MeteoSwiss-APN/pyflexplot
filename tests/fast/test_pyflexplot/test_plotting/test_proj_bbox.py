"""Test for module ``pyflexplot.plotting.proj_bbox``."""
# Standard library
import dataclasses as dc

# Third-party
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest

# First-party
from pyflexplot.plotting.proj_bbox import ProjectedBoundingBox
from pyflexplot.plotting.proj_bbox import Projections


@dc.dataclass
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


def test_global() -> None:
    """Test transformation of global bbox.

    Implemented to reproduce a bug in production.

    """
    ax = plt.figure().add_axes((0.0, 0.05, 0.7872, 0.85), projection=ccrs.PlateCarree())
    bb = ProjectedBoundingBox(ax=ax, projs=Projections.create_regular()).to_axes()
    bb.to_map()  # Here, the bug triggered an exception
    assert np.allclose(bb, (-180, 180, -90, 90))

"""Test for module ``pyflexplot.plotting.proj_bbox``."""
# Standard library
from dataclasses import dataclass

# Third-party
import cartopy
import matplotlib.pyplot as plt

# First-party
from pyflexplot.plotting.proj_bbox import MapAxesProjections
from pyflexplot.plotting.proj_bbox import ProjectedBoundingBox


@dataclass
class Cfg:
    c_lon: float = 0.0
    curr_proj: str = "axes"
    lon0: float = -180.0
    lon1: float = 180.0
    lat0: float = -90.0
    lat1: float = 90.0


class TestRegLatLon:
    fig, ax = plt.subplots()
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 45)

    def projs(self, cfg) -> MapAxesProjections:
        return MapAxesProjections(
            data=cartopy.crs.PlateCarree(central_longitude=0.0),
            map=cartopy.crs.PlateCarree(central_longitude=cfg.c_lon),
            geo=cartopy.crs.PlateCarree(),
        )

    def bbox(self, cfg: Cfg) -> ProjectedBoundingBox:
        return ProjectedBoundingBox(
            ax=self.ax,
            projs=self.projs(cfg),
            curr_proj=cfg.curr_proj,
            lon0=cfg.lon0,
            lon1=cfg.lon1,
            lat0=cfg.lat0,
            lat1=cfg.lat1,
        )

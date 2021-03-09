"""Tests for module ``pyflexplot.plotting.domain``."""
# Standard library
import dataclasses as dc
from typing import List
from typing import Tuple

# Third-party
import numpy as np
import pytest

# First-party
from pyflexplot.plotting.domain import CloudDomain
from pyflexplot.plotting.domain import find_gaps


class Test_CloudDomain:
    """Test cloud domain."""

    class Global:
        """Global domain."""

        lat = np.arange(-85, 85.1, 10)
        lon = np.arange(-175, 175.1, 10)

        def get_mask(self, lllon, urlon, lllat, urlat):
            mask = np.zeros([self.lat.size, self.lon.size], np.bool)
            if lllon < 0:
                mask[lllat:urlat, lllon:] = True
                mask[lllat:urlat, :urlon] = True
            else:
                mask[lllat:urlat, lllon:urlon] = True
            return mask

        def get_domain(self, lllon, urlon, lllat, urlat, **config):
            mask = self.get_mask(lllon, urlon, lllat, urlat)
            return CloudDomain(self.lat, self.lon, mask=mask, config=config)

        def get_bbox(self, lllon, urlon, lllat, urlat, **config):
            domain = self.get_domain(lllon, urlon, lllat, urlat, **config)
            return domain.find_bbox_corners()

    class Test_ZonalNearPole(Global):
        """Zonally extended cloud near a pole."""

        def test_north_no_aspect(self):
            """Near north pole, w/o specified aspect ratio."""
            bbox = self.get_bbox(10, 29, 15, 17)
            # dlon, dlat = 180, 10; aspect = 18
            assert bbox == (-75, 105, 65, 75)

        def test_north_aspect(self):
            """Near north pole, with specified aspect ratio."""
            bbox = self.get_bbox(10, 29, 16, 18, aspect=2)
            # dlat = dlon / aspect = 180 / 2 = 90
            # ddlat = 90 - 10 = 80
            assert bbox == (-75, 105, -5, 85)

        def test_south_no_aspect(self):
            """Near south pole, w/o specified aspect ratio."""
            bbox = self.get_bbox(15, 24, 1, 4)
            # dlon, dlat = 80, 20; aspect = 4
            assert bbox == (-25, 55, -75, -55)

        def test_south_aspect(self):
            """Near south pole, with specified aspect ratio."""
            bbox = self.get_bbox(15, 24, 1, 4, aspect=1)
            # dlat = dlon / aspect = 80 / 1 = 80
            # ddlat = 80 - 20 = 60
            assert bbox == (-25, 55, -85, -5)

    class Test_AcrossDateline(Global):
        """Cloud stretching across dateline."""

        def test_non_periodic_wrong(self):
            """Results are wrong if domain is not specified as periodic."""
            bbox = self.get_bbox(-3, 2, 10, 17, periodic_lon=False)
            assert bbox == (-175, 175, 15, 75)

        def test_periodic_ok(self):
            """Results are correct if domain is specified as periodic."""
            bbox = self.get_bbox(-3, 2, 10, 17, periodic_lon=True)
            assert bbox == (155, -165, 15, 75)

        def test_zonal_no_aspect(self):
            """Zonally extended cloud, w/o specified aspect ratio."""
            bbox = self.get_bbox(-5, 4, 13, 15, periodic_lon=True)
            # dlon, dlat = 80, 10; aspect = 8
            assert bbox == (135, -145, 45, 55)

        def test_zonal_aspect(self):
            """Zonally extended cloud, with specified aspect ratio."""
            bbox = self.get_bbox(-5, 4, 13, 15, periodic_lon=True, aspect=1.5)
            # dlat = dlon / aspect = 80 / 1.5 = 53.333
            # ddlat = 53.333 - 10 = 43.333
            ddlat2 = (43 + 1 / 3) / 2
            assert bbox == (135, -145, 45 - ddlat2, 55 + ddlat2)

        def test_meridional_no_aspect(self):
            """Meridionally extended cloud, w/o specified aspect ratio."""
            bbox = self.get_bbox(-1, 2, 10, 17, periodic_lon=True)
            # dlon, dlat = 20, 60; aspect = 0.333
            assert bbox == (175, -165, 15, 75)

        def test_meridional_aspect(self):
            """Meridionally extended cloud, with specified aspect ratio."""
            bbox = self.get_bbox(-1, 2, 10, 17, periodic_lon=True, aspect=1.5)
            # dlon = aspect * dlat = 1.5 * 60 = 90
            # ddlon = 90 - 20 = 70
            ddlon2 = 70 / 2
            assert bbox == (175 - ddlon2, -165 + ddlon2, 15, 75)


@dc.dataclass
class FindGapsConfig:
    mask: List[int]
    gaps: List[Tuple[int, int, int]]
    periodic: bool = True


@pytest.mark.parametrize(
    "cfg",
    [
        FindGapsConfig(  # [cfg0]
            mask=[0, 0, 0, 0],
            gaps=[],
        ),
        FindGapsConfig(  # [cfg1]
            mask=[0, 0, 0, 0],
            gaps=[(4, 0, 3)],
            periodic=False,
        ),
        FindGapsConfig(  # [cfg2]
            mask=[1, 0, 0, 1],
            gaps=[(2, 1, 2)],
        ),
        FindGapsConfig(  # [cfg3]
            mask=[1, 0, 0, 1],
            gaps=[(2, 1, 2)],
            periodic=False,
        ),
        FindGapsConfig(  # [cfg4]
            mask=[0, 1, 1, 0],
            gaps=[(2, 3, 0)],
        ),
        FindGapsConfig(  # [cfg5]
            mask=[0, 1, 1, 0],
            gaps=[(1, 0, 0), (1, 3, 3)],
            periodic=False,
        ),
        FindGapsConfig(  # [cfg6]
            mask=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            gaps=[(1, 3, 3), (3, 5, 7), (4, 9, 0)],
        ),
        FindGapsConfig(  # [cfg7]
            mask=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            gaps=[(1, 0, 0), (1, 3, 3), (3, 5, 7), (3, 9, 11)],
            periodic=False,
        ),
    ],
)
def test_find_gaps(cfg):
    gaps = find_gaps(cfg.mask, periodic=cfg.periodic)
    assert gaps == cfg.gaps

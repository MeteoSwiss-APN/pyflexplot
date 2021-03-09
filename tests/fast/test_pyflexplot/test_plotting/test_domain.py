"""Tests for module ``pyflexplot.plotting.domain``."""
# Third-party
import numpy as np

# First-party
from pyflexplot.plotting.domain import CloudDomain


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

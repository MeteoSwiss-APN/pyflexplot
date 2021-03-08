"""Tests for module ``pyflexplot.plotting.domain``."""
# Third-party
import numpy as np

# First-party
from pyflexplot.plotting.domain import CloudDomain


class Test_CloudAcrossDateline:
    """Test cloud domain for a cloud positioned across the dateline."""

    lat = np.arange(-85.0, 85.1, 10.0)
    lon = np.arange(-185.0, 185.1, 10.0)
    mask = np.zeros([lat.size, lon.size], np.bool)
    mask[10:17, :2] = True
    mask[10:17, -3:] = True

    def test_dateline_wrong(self):
        """Results are wrong if domain is not specified as periodic."""
        domain = CloudDomain(
            self.lat, self.lon, mask=self.mask, config={"periodic_lon": False}
        )
        bbox = domain.find_bbox_corners()
        assert bbox == (-185.0, 185.0, 15.0, 75.0)

    def test_dateline_ok(self):
        """Results are correct if domain is specified as periodic."""
        domain = CloudDomain(
            self.lat, self.lon, mask=self.mask, config={"periodic_lon": True}
        )
        bbox = domain.find_bbox_corners()
        assert bbox == (165.0, -175.0, 15.0, 75.0)

    def test_aspect(self):
        """Adapt the domain to a given aspect ratio."""
        """Results are correct if domain is specified as periodic."""
        domain = CloudDomain(
            self.lat,
            self.lon,
            mask=self.mask,
            config={"periodic_lon": True, "aspect": 1.5},
        )
        bbox = domain.find_bbox_corners()
        assert bbox == (130.0, -140.0, 15.0, 75.0)

"""Tests for module ``pyflexplot.plotting.domain``."""
# Third-party
import numpy as np

# First-party
from pyflexplot.plotting.domain import CloudDomain


class TestCloudAcrossDateline:
    """Test cloud domain for a cloud positioned across the dateline."""

    lat = np.arange(-85.0, 85.1, 10.0)
    lon = np.arange(-185.0, 185.1, 10.0)
    mask = np.zeros([lat.size, lon.size], np.bool)
    mask[11:15, :4] = True
    mask[11:15, -4:] = True

    def test_dateline_wrong(self):
        """Results are wrong if domain is not specified as periodic."""
        domain = CloudDomain(self.lat, self.lon, mask_nz=self.mask, periodic_lon=False)
        bbox = domain.get_bbox_corners()
        assert bbox == (-185.0, 185.0, 25.0, 55.0)

    def test_dateline_ok(self):
        """Results are correct if domain is specified as periodic."""
        domain = CloudDomain(self.lat, self.lon, mask_nz=self.mask, periodic_lon=True)
        bbox = domain.get_bbox_corners()
        assert bbox == (155.0, -155.0, 25.0, 55.0)

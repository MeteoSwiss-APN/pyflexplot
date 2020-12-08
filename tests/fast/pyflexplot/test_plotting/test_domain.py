"""Tests for module ``pyflexplot.plotting.domain``."""
# Third-party
import numpy as np

# First-party
from pyflexplot.plotting.domain import CloudDomain


def test_dateline():
    lat = np.arange(-85.0, 85.1, 10.0)
    lon = np.arange(-185.0, 185.1, 10.0)
    mask = np.zeros([lat.size, lon.size], np.bool)
    mask[11:15, :4] = True
    mask[11:15, -4:] = True
    domain = CloudDomain(lat, lon, mask_nz=mask, periodic_lon=True)
    bbox = domain.get_bbox_corners()
    assert bbox == (155.0, -155.0, 25.0, 55.0)

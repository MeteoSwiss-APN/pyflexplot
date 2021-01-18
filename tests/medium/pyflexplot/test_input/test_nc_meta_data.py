"""Tests for module ``pyflexplot.input.nc_meta_data``."""
# Standard library
from typing import Any
from typing import Dict
from typing import Optional

# Third-party
import netCDF4 as nc4

# First-party
from pyflexplot.input.nc_meta_data import read_nc_meta_data

# Local  isort:skip
from .shared import datadir_flexpart_reduced as datadir  # noqa:F401 isort:skip


class _TestBase:
    datafilename: Optional[str] = None

    @classmethod
    def read(cls, datadir) -> Dict[str, Any]:
        with nc4.Dataset(f"{datadir}/{cls.datafilename}", "r") as f:
            return read_nc_meta_data(f, add_ts0=False)


class Test_COSMO1(_TestBase):
    datafilename = "flexpart_cosmo-1_2019052800.nc"

    def test_species_ids(self, datadir):
        nc_meta_data = self.read(datadir)
        assert nc_meta_data["derived"]["species_ids"] == (1, 2)

    def test_dimensions(self, datadir):
        nc_meta_data = self.read(datadir)
        dimensions = {
            "time": {"name": "time", "size": 11},
            "rlon": {"name": "rlon", "size": 40},
            "rlat": {"name": "rlat", "size": 30},
            "level": {"name": "level", "size": 3},
            "numspec": {"name": "numspec", "size": 2},
            "numpoint": {"name": "numpoint", "size": 1},
            "nageclass": {"name": "nageclass", "size": 1},
            "nchar": {"name": "nchar", "size": 45},
        }
        assert nc_meta_data["dimensions"] == dimensions


class Test_COSMO2(_TestBase):
    datafilename = "flexpart_cosmo-2e_2019072712_000.nc"

    def test_species_ids(self, datadir):
        nc_meta_data = self.read(datadir)
        assert nc_meta_data["derived"]["species_ids"] == (1, 2)

    def test_dimensions(self, datadir):
        nc_meta_data = self.read(datadir)
        dimensions = {
            "time": {"name": "time", "size": 11},
            "rlon": {"name": "rlon", "size": 50},
            "rlat": {"name": "rlat", "size": 30},
            "level": {"name": "level", "size": 3},
            "numspec": {"name": "numspec", "size": 2},
            "numpoint": {"name": "numpoint", "size": 1},
            "nageclass": {"name": "nageclass", "size": 1},
            "nchar": {"name": "nchar", "size": 45},
        }
        assert nc_meta_data["dimensions"] == dimensions


class Test_IFS(_TestBase):
    datafilename = "flexpart_ifs_20200317000000.nc"

    def test_species_ids(self, datadir):
        nc_meta_data = self.read(datadir)
        assert nc_meta_data["derived"]["species_ids"] == (1,)

    def test_dimensions(self, datadir):
        nc_meta_data = self.read(datadir)
        dimensions = {
            "time": {"name": "time", "size": 16},
            "longitude": {"name": "longitude", "size": 160},
            "latitude": {"name": "latitude", "size": 80},
            "height": {"name": "height", "size": 4},
            "numspec": {"name": "numspec", "size": 1},
            "pointspec": {"name": "pointspec", "size": 1},
            "nageclass": {"name": "nageclass", "size": 1},
            "nchar": {"name": "nchar", "size": 45},
            "numpoint": {"name": "numpoint", "size": 1},
        }
        assert nc_meta_data["dimensions"] == dimensions

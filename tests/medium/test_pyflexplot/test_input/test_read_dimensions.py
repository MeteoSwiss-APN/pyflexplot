"""Tests for functions in module ``pyflexplot.input.meta_data``."""
# Standard library
from typing import Optional

# Third-party
import netCDF4 as nc4

# First-party
from pyflexplot.input.meta_data import read_dimensions
from pyflexplot.input.meta_data import read_species_ids

# Local  isort:skip
from .shared import datadir_flexpart_reduced as datadir  # noqa:F401 isort:skip


class _TestBase:
    datafilename: Optional[str] = None

    @classmethod
    def read_species_ids(cls, datadir, **kwargs):
        with nc4.Dataset(f"{datadir}/{cls.datafilename}", "r") as file_handle:
            return read_species_ids(file_handle, **kwargs)

    @classmethod
    def read_dimensions(cls, datadir, **kwargs):
        with nc4.Dataset(f"{datadir}/{cls.datafilename}", "r") as file_handle:
            return read_dimensions(file_handle, **kwargs)


class Test_COSMO1(_TestBase):
    datafilename = "flexpart_cosmo-1_2019052800.nc"

    def test_species_ids(self, datadir):
        assert self.read_species_ids(datadir) == (1, 2)

    def test_dimensions(self, datadir):
        res = self.read_dimensions(datadir, add_ts0=False)
        sol = {
            "time": {"name": "time", "size": 11},
            "rlon": {"name": "rlon", "size": 40},
            "rlat": {"name": "rlat", "size": 30},
            "level": {"name": "level", "size": 3},
            "numspec": {"name": "numspec", "size": 2},
            "numpoint": {"name": "numpoint", "size": 1},
            "nageclass": {"name": "nageclass", "size": 1},
            "nchar": {"name": "nchar", "size": 45},
        }
        assert res == sol


class Test_COSMO2(_TestBase):
    datafilename = "flexpart_cosmo-e_2019072712_000.nc"

    def test_species_ids(self, datadir):
        assert self.read_species_ids(datadir) == (1, 2)

    def test_dimensions(self, datadir):
        res = self.read_dimensions(datadir, add_ts0=False)
        sol = {
            "time": {"name": "time", "size": 11},
            "rlon": {"name": "rlon", "size": 50},
            "rlat": {"name": "rlat", "size": 30},
            "level": {"name": "level", "size": 3},
            "numspec": {"name": "numspec", "size": 2},
            "numpoint": {"name": "numpoint", "size": 1},
            "nageclass": {"name": "nageclass", "size": 1},
            "nchar": {"name": "nchar", "size": 45},
        }
        assert res == sol


class Test_IFS(_TestBase):
    datafilename = "flexpart_ifs_20200317000000.nc"

    def test_species_ids(self, datadir):
        assert self.read_species_ids(datadir) == (1,)

    def test_dimensions(self, datadir):
        res = self.read_dimensions(datadir, add_ts0=False)
        sol = {
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
        assert res == sol

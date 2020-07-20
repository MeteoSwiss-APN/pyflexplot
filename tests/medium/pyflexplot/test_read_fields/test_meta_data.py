# -*- coding: utf-8 -*-
"""
Tests for <TODO> in module ``pyflexplot.input``.
"""
# Standard library
from typing import Dict
from typing import Optional

# Third-party
import netCDF4 as nc4

# First-party
from pyflexplot.nc_meta_data import read_meta_data

# Local  isort:skip
from shared import datadir_reduced as datadir  # noqa:F401 isort:skip


class _TestBase:
    datafilename: Optional[str] = None
    meta_data: Optional[Dict] = None

    @classmethod
    def set_up(cls, datadir, cache=True):
        if cache:
            if cls.meta_data is not None:
                return None
        with nc4.Dataset(f"{datadir}/{cls.datafilename}", "r") as f:
            cls.meta_data = read_meta_data(f)


class Test_COSMO1(_TestBase):
    datafilename = "flexpart_cosmo-1_2019052800.nc"

    def test_model(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["model"] == "cosmo1"

    def test_rotated_pole(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["rotated_pole"] is True

    def test_species_ids(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["species_ids"] == (1, 2)

    def test_dimensions(self, datadir):
        self.set_up(datadir)
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
        assert self.meta_data["dimensions"] == dimensions


class Test_COSMO2(_TestBase):
    datafilename = "flexpart_cosmo-2e_2019072712_000.nc"

    def test_model(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["model"] == "cosmo2"

    def test_rotated_pole(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["rotated_pole"] is True

    def test_species_ids(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["species_ids"] == (1, 2)

    def test_dimensions(self, datadir):
        self.set_up(datadir)
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
        assert self.meta_data["dimensions"] == dimensions


class Test_IFS(_TestBase):
    datafilename = "flexpart_ifs_20200317000000.nc"

    def test_model(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["model"] == "ifs"

    def test_rotated_pole(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["rotated_pole"] is False

    def test_species_ids(self, datadir):
        self.set_up(datadir)
        assert self.meta_data["derived"]["species_ids"] == (1,)

    def test_dimensions(self, datadir):
        self.set_up(datadir)
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
        assert self.meta_data["dimensions"] == dimensions

# -*- coding: utf-8 -*-
"""
Tests for class ``pyflexplot.setup.InputSetup``.
"""
# Standard library
from typing import Any
from typing import Dict

# Third-party
import pytest  # type: ignore

# First-party
from pyflexplot.setup import InputSetup
from srutils.dict import merge_dicts

# Local
from .shared import DEFAULT_KWARGS
from .shared import DEFAULT_SETUP


def test_default_setup_dict():
    """Check the default setupuration dict."""
    setup1 = InputSetup.create(DEFAULT_KWARGS)
    setup2 = DEFAULT_SETUP.dict()
    assert setup1 == setup2


class Test_ReplaceNoneByAvailable:
    meta_data: Dict[str, Any] = {
        "dimensions": {
            "time": {"name": "time", "size": 11},
            "rlon": {"name": "rlon", "size": 40},
            "rlat": {"name": "rlat", "size": 30},
            "level": {"name": "level", "size": 3},
            "nageclass": {"name": "nageclass", "size": 1},
            "noutrel": {"name": "numpoint", "size": 1},
            "numpoint": {"name": "numpoint", "size": 2},
            "nchar": {"name": "nchar", "size": 45},
        },
        "analysis": {"species_ids": (1, 2)},
    }

    def setup_create(self, params):
        assert "dimensions" not in DEFAULT_KWARGS
        return InputSetup.create(merge_dicts(DEFAULT_KWARGS, params))

    def test_time(self):
        setup = self.setup_create({"dimensions": {"time": "*"}})
        assert setup.core.dimensions.time is None
        setup = setup.complete_dimensions(self.meta_data)
        assert setup.core.dimensions.time == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    def test_level(self):
        setup = self.setup_create({"dimensions": {"level": "*"}})
        assert setup.core.dimensions.level is None
        setup = setup.complete_dimensions(self.meta_data)
        assert setup.core.dimensions.level == (0, 1, 2)

    def test_species_id(self):
        setup = self.setup_create({"dimensions": {"species_id": "*"}})
        assert setup.core.dimensions.species_id is None
        setup = setup.complete_dimensions(self.meta_data)
        assert setup.core.dimensions.species_id == (1, 2)

    def test_others(self):
        setup = self.setup_create(
            {"dimensions": {"nageclass": "*", "noutrel": "*", "numpoint": "*"}}
        )
        assert setup.core.dimensions.nageclass is None
        assert setup.core.dimensions.noutrel is None
        assert setup.core.dimensions.numpoint is None
        setup = setup.complete_dimensions(self.meta_data)
        assert setup.core.dimensions.nageclass == 0
        assert setup.core.dimensions.noutrel == 0
        assert setup.core.dimensions.numpoint == (0, 1)


class Test_WildcardToNone:
    """Wildcard values passed to ``InputSetup.create`` turn into None.

    The wildcards can be used in a set file to explicitly specify that all
    available values of a respective dimension (etc.) shall be read from a
    NetCDF file, as es the case if the setup value is None (the default).

    """

    def setup_create(self, params):
        return InputSetup.create(merge_dicts(DEFAULT_KWARGS, params))

    def test_species_id(self):
        setup = self.setup_create({"dimensions": {"species_id": "*"}})
        assert setup.core.dimensions.species_id is None

    def test_time(self):
        setup = self.setup_create({"dimensions": {"time": "*"}})
        assert setup.core.dimensions.time is None

    def test_level(self):
        setup = self.setup_create({"dimensions": {"level": "*"}})
        assert setup.core.dimensions.level is None

    def test_others(self):
        setup = self.setup_create(
            {"dimensions": {"nageclass": "*", "noutrel": "*", "numpoint": "*"}}
        )
        assert setup.core.dimensions.nageclass is None
        assert setup.core.dimensions.noutrel is None
        assert setup.core.dimensions.numpoint is None


class Test_Cast:
    def test_infile(self):
        assert InputSetup.cast("infile", "foo.nc") == "foo.nc"

    def test_outfile(self):
        assert InputSetup.cast("outfile", "foo.png") == "foo.png"

    def test_lang(self):
        assert InputSetup.cast("lang", "de") == "de"

    def test_ens_member_id(self):
        assert InputSetup.cast("ens_member_id", "004") == 4

    def test_integrate(self):
        assert InputSetup.cast("integrate", "True") is True
        assert InputSetup.cast("integrate", "False") is False

    def test_level(self):
        assert InputSetup.cast("dimensions", {"level": "2"}) == {"level": 2}

    def test_time(self):
        assert InputSetup.cast("dimensions", {"time": "10"}) == {"time": 10}


class Test_CastSequence:
    def test_infile_fail(self):
        with pytest.raises(ValueError):
            InputSetup.cast("infile", ["a.nc", "b.nc"])

    def test_outfile_fail(self):
        with pytest.raises(ValueError):
            InputSetup.cast("outfile", ["a.png", "b.png"])

    def test_lang_fail(self):
        with pytest.raises(ValueError):
            InputSetup.cast("lang", ["en", "de"])

    def test_ens_member_id(self):
        assert InputSetup.cast("ens_member_id", ["01", "02", "03"]) == [1, 2, 3]

    def test_integrate_fail(self):
        with pytest.raises(ValueError):
            InputSetup.cast("integrate", ["True", "False"])

    def test_level(self):
        res = InputSetup.cast("dimensions", {"level": ["1", "2"]})
        assert res == {"level": (1, 2)}

    def test_time(self):
        res = InputSetup.cast("dimensions", {"time": ["0", "1", "2", "3", "4"]})
        assert res == {"time": (0, 1, 2, 3, 4)}


class Test_CastMany:
    def test_dict(self):
        params = {
            "infile": "foo.nc",
            "dimensions": {"species_id": ["1", "2"]},
            "integrate": "False",
        }
        res = InputSetup.cast_many(params)
        sol = {
            "infile": "foo.nc",
            "dimensions": {"species_id": (1, 2)},
            "integrate": False,
        }
        assert res == sol

    def test_dict_comma_separated_fail(self):
        params = {
            "infile": "foo.nc",
            "dimensions": {"species_id": "1,2"},
            "integrate": "False",
        }
        with pytest.raises(ValueError):
            InputSetup.cast_many(params)

    def test_tuple_duplicates_fail(self):
        params = (
            ("infile", "foo.nc"),
            ("dimensions", ("species_id", "1")),
            ("infile", "bar.nc"),
        )
        with pytest.raises(ValueError):
            InputSetup.cast_many(params)

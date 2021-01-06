"""Test casting methods of class ``pyflexplot.setup.Setup``."""
# Third-party
import pytest  # type: ignore

# First-party
from pyflexplot.setup import Setup
from srutils.exceptions import InvalidParameterValueError


class Test_CastSingle:
    def test_infile(self):
        assert Setup.cast("infile", "foo.nc") == "foo.nc"

    def test_outfile(self):
        assert Setup.cast("outfile", "foo.png") == "foo.png"
        assert Setup.cast("outfile", ["foo.png"]) == ("foo.png",)
        assert Setup.cast("outfile", ["foo.png", "bar.png"]) == ("foo.png", "bar.png")

    def test_lang(self):
        assert Setup.cast("lang", "de") == "de"

    def test_ens_member_id(self):
        assert Setup.cast("model", {"ens_member_id": "004"}) == {"ens_member_id": (4,)}

    def test_integrate(self):
        assert Setup.cast("integrate", "True") is True
        assert Setup.cast("integrate", "False") is False

    def test_level(self):
        assert Setup.cast("dimensions", {"level": "2"}) == {"level": 2}

    def test_time(self):
        assert Setup.cast("dimensions", {"time": "10"}) == {"time": 10}


class Test_CastSequence:
    def test_infile_fail(self):
        with pytest.raises(InvalidParameterValueError):
            Setup.cast("infile", ["a.nc", "b.nc"])

    def test_lang_fail(self):
        with pytest.raises(InvalidParameterValueError):
            Setup.cast("lang", ["en", "de"])

    def test_ens_member_id(self):
        res = Setup.cast("model", {"ens_member_id": ["01", "02", "03"]})
        assert res == {"ens_member_id": (1, 2, 3)}

    def test_integrate_fail(self):
        with pytest.raises(InvalidParameterValueError):
            Setup.cast("integrate", ["True", "False"])

    def test_level(self):
        res = Setup.cast("dimensions", {"level": ["1", "2"]})
        assert res == {"level": (1, 2)}

    def test_time(self):
        res = Setup.cast("dimensions", {"time": ["0", "1", "2", "3", "4"]})
        assert res == {"time": (0, 1, 2, 3, 4)}


class Test_CastMany:
    def test_dict(self):
        params = {
            "infile": "foo.nc",
            "dimensions": {"species_id": ["1", "2"]},
            "integrate": "False",
        }
        res = Setup.cast_many(params)
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
        with pytest.raises(InvalidParameterValueError):
            Setup.cast_many(params)

    def test_tuple_duplicates_fail(self):
        params = (
            ("infile", "foo.nc"),
            ("dimensions", ("species_id", "1")),
            ("infile", "bar.nc"),
        )
        with pytest.raises(ValueError):
            Setup.cast_many(params)

"""Test casting methods of class ``pyflexplot.setup.Setup``."""
# Third-party
import pytest  # type: ignore

# First-party
from pyflexplot.setups.plot_setup import PlotSetup
from srutils.exceptions import InvalidParameterValueError


class Test_CastSingle:
    def test_infile(self):
        assert PlotSetup.cast("files.input", "foo.nc") == "foo.nc"

    def test_outfile(self):
        assert PlotSetup.cast("files.output", "foo.png") == "foo.png"
        assert PlotSetup.cast("files.output", ["foo.png"]) == ("foo.png",)
        res = PlotSetup.cast("files.output", ["foo.png", "bar.png"])
        assert res == ("foo.png", "bar.png")

    def test_lang(self):
        assert PlotSetup.cast("lang", "de") == "de"

    def test_ens_member_id(self):
        assert PlotSetup.cast("model.ens_member_id", "004") == (4,)

    def test_integrate(self):
        assert PlotSetup.cast("integrate", "True") is True
        assert PlotSetup.cast("integrate", "False") is False

    def test_level(self):
        assert PlotSetup.cast("dimensions.level", "2") == 2

    def test_time(self):
        assert PlotSetup.cast("dimensions.time", "10") == 10


class Test_CastSequence:
    def test_infile_fail(self):
        with pytest.raises(InvalidParameterValueError):
            PlotSetup.cast("files.input", ["a.nc", "b.nc"])

    def test_lang_fail(self):
        with pytest.raises(InvalidParameterValueError):
            PlotSetup.cast("lang", ["en", "de"])

    def test_ens_member_id(self):
        assert PlotSetup.cast("model.ens_member_id", ["01", "02", "03"]) == (1, 2, 3)

    def test_integrate_fail(self):
        with pytest.raises(InvalidParameterValueError):
            PlotSetup.cast("integrate", ["True", "False"])

    def test_level(self):
        assert PlotSetup.cast("dimensions.level", ["1", "2"]) == (1, 2)

    def test_time(self):
        res = PlotSetup.cast("dimensions.time", ["0", "1", "2", "3", "4"])
        assert res == (0, 1, 2, 3, 4)


class Test_CastMany:
    def test_dict(self):
        params = {
            "files.input": "foo.nc",
            "dimensions.species_id": ["1", "2"],
            "integrate": "False",
        }
        res = PlotSetup.cast_many(params)
        sol = {
            "files.input": "foo.nc",
            "dimensions.species_id": (1, 2),
            "integrate": False,
        }
        assert res == sol

    def test_dict_comma_separated_fail(self):
        params = {
            "files.input": "foo.nc",
            "dimensions.species_id": "1,2",
            "integrate": "False",
        }
        with pytest.raises(InvalidParameterValueError):
            PlotSetup.cast_many(params)

    def test_tuple_duplicates_fail(self):
        params = (
            ("files.input", "foo.nc"),
            ("dimensions.species_id", "1"),
            ("files.input", "bar.nc"),
        )
        with pytest.raises(ValueError):
            PlotSetup.cast_many(params)

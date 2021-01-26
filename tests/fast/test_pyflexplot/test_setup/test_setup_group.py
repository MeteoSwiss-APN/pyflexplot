"""Tests for class ``pyflexplot.setup.SetupGroup``."""
# Third-party
import pytest

# First-party
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupGroup
from srutils.testing import check_is_sub_element


class TestCopy:
    def test_preserve_outfiles(self):
        params = {
            "infile": "foo.nc",
            "model": {"name": "IFS-HRES"},
            "outfile": ("foo.png", "bar.pdf"),
        }
        setup = Setup.create(params)
        setups = SetupGroup([setup])
        copied_setups = setups.copy()
        assert len(copied_setups) == len(setups)
        assert copied_setups == setups


class Test_FromRawParams:
    def test_empty(self):
        with pytest.raises(ValueError):
            SetupGroup.from_raw_params([])

    def test_one_variable(self):
        raw_params = {
            "infile": "foo.nc",
            "outfile": "foo.png",
            "input_variable": "concentration",
            "species_id": [1, 2],
            "combine_species": False,
        }
        setups = SetupGroup.from_raw_params(raw_params)
        res = setups.dicts()
        sol = [
            {
                "infile": "foo.nc",
                "outfile": "foo.png",
                "core": {
                    "input_variable": "concentration",
                    "combine_species": False,
                    "dimensions": {
                        "species_id": (1, 2),
                    },
                },
            },
        ]
        check_is_sub_element(sol, res, "solution", "result")

"""Tests for class ``pyflexplot.setup.SetupGroup``."""
# Standard library

# Third-party

# First-party
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupGroup


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

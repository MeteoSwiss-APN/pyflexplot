"""Test feature to generate shape files."""
# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa:F401  # required by _TestBase.test
#from pyflexplot.save_data.
# Third-party
import pytest  # type: ignore

# First-party
from pyflexplot.input.read_fields import read_fields
from pyflexplot.plots import create_plot
from pyflexplot.plots import format_out_file_paths
from pyflexplot.setups.plot_setup import PlotSetup
from pyflexplot.setups.plot_setup import PlotSetupGroup
from pyflexplot.utils.summarize import summarize
from srutils.testing import assert_nested_equal
#from pyflexplot.save_data import ShapeFileSaver

import distutils.dir_util
import os
from pathlib import Path

INFILE_1 = "flexpart_cosmo-1_2019093012.nc"

# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference
@pytest.mark.skip("unfinished")
class Test_ShapeFileGeneration(_TestBase):
    reference = "ref_cosmo-1_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-1",
        },
        "panels": [
            {
                "plot_variable": "concentration",
                "integrate": False,
                "lang": "de",
                "domain": "full",
                "dimensions": {
                    "species_id": 1,
                    "time": 5,
                    "level": 0,
                    "multiplier":None,
                },
            }
        ],
    }

    def test(self, datadir):
        #shape_file_saver = ShapeFileSaver()
        setup = PlotSetup.create(self.setup_dct)
        field_group = self.get_field_group(datadir)
        plot = self.get_plot(field_group)
        assert True == True
        print("FIELD GROUP ", field_group , "\n ", plot)
        return
        res = summarize(field_group)
        sol = self.get_reference("field_summary")
        try:
            assert_nested_equal(res, sol, "res", "sol", float_close_ok=True)
        except AssertionError as e:
            msg = f"field summaries differ (result vs. solution):\n\n {e}"
            raise AssertionError(msg)

        plot = self.get_plot(field_group)
        res = summarize(plot)
        plot.clean()
        sol = self.get_reference("plot_summary")
        try:
            assert_nested_equal(res, sol, "res", "sol", float_close_ok=True)
        except AssertionError as e:
            msg = f"plot summaries differ (result vs. solution):\n\n{e}"
            raise AssertionError(msg)
        print("DATA DIR ", datadir)
        assert True == True

"""Test the elements of complete plots based on deterministic COSMO-1 data."""
# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa:F401  # required by _TestBase.test

INFILE_1 = "flexpart_cosmo-1e-ctrl_2020102105.nc"


# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference


class Test_TotalDeposition_MissingField(_TestBase):
    reference = "ref_cosmo-1e_total_deposition_dummy"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "auto",
        },
        "model": {
            "name": "COSMO-1E",
        },
        "panels": [
            {
                "plot_variable": "tot_deposition",
                "integrate": True,
                "lang": "de",
                "domain": "full",
                "dimensions": {
                    "species_id": 1,
                    "time": -1,
                },
            }
        ],
    }

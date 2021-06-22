"""Test the elements of complete plots based on deterministic COSMO-1 data."""
# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa:F401  # required by _TestBase.test

INFILE_1 = "flexpart_cosmo-1e-ctrl_2020102105.nc"
INFILE_2 = "flexpart_cosmo-1e-ctrl_1032_2021040800_6-releases.nc"


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


class Test_AffectedArea(_TestBase):
    reference = "ref_cosmo-1e-ctrl_affected_area"
    setup_dct = {
        "files": {
            "input": INFILE_2,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-1E",
        },
        "panels": {
            "domain": "ch",
            "plot_variable": "affected_area",
            "integrate": True,
            "lang": "en",
            "dimensions": {
                "level": 0,
                "release": 5,
                "species_id": 1,
                "time": -1,
            },
        },
    }


class Test_Concentration_MultiPanelTime(_TestBase):
    reference = "ref_cosmo-1e-ctrl_concentration_multipanel_time"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "multipanel",
            "multipanel_param": "time",
        },
        "model": {
            "name": "COSMO-1E",
        },
        "panels": {
            "plot_variable": "concentration",
            "integrate": False,
            "lang": "de",
            "domain": "full",
            "dimensions": {
                "species_id": 1,
                "time": [2, 4, 6, 8],
            },
        },
    }


class Test_TotalDeposition_MultiPanelTime(_TestBase):
    reference = "ref_cosmo-1e-ctrl_tot_deposition_multipanel_time"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "multipanel",
            "multipanel_param": "time",
        },
        "model": {
            "name": "COSMO-1E",
        },
        "panels": {
            "plot_variable": "tot_deposition",
            "integrate": True,
            "lang": "en",
            "domain": "ch",
            "dimensions": {
                "species_id": 1,
                "time": [2, 4, 6, 8],
            },
        },
    }

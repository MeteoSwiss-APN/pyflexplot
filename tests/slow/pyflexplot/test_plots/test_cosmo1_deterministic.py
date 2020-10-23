"""
Test the elements of complete plots based on deterministic COSMO-1 data.
"""
# Local
from .shared import _TestBase
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa:F401  # required by _TestBase.test

INFILE_NAME = "flexpart_cosmo-1_2019093012.nc"


# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference


class Test_Concentration(_TestBase):
    reference = "ref_cosmo1_deterministic_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "COSMO-1",
        "input_variable": "concentration",
        "integrate": False,
        "lang": "de",
        "domain": "full",
        "dimensions": {"species_id": 1, "time": 5, "level": 0},
    }


class Test_IntegratedConcentration(_TestBase):
    reference = "ref_cosmo1_deterministic_integrated_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "COSMO-1",
        "plot_type": "auto",
        "input_variable": "concentration",
        "integrate": True,
        "lang": "en",
        "domain": "ch",
        "dimensions": {"species_id": 1, "time": 10, "level": 0},
    }


class Test_TotalDeposition(_TestBase):
    reference = "ref_cosmo1_deterministic_total_deposition"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "COSMO-1",
        "plot_type": "auto",
        "input_variable": "deposition",
        "combine_deposition_types": True,
        "integrate": True,
        "lang": "de",
        "domain": "full",
        "dimensions": {"species_id": 1, "time": -1},
    }


class Test_AffectedArea(_TestBase):
    reference = "ref_cosmo1_deterministic_affected_area"
    setup_dct = {
        "combine_deposition_types": True,
        "domain": "ch",
        "infile": INFILE_NAME,
        # "input_variable": "deposition",
        # "plot_variable": "affected_area_mono",
        "input_variable": "affected_area",
        "integrate": True,
        "lang": "en",
        "model": "COSMO-1",
        "outfile": f"{reference}.png",
        "dimensions": {
            "deposition_type": ["dry", "wet"],
            "level": 0,
            "species_id": 1,
            "time": -1,
        },
    }

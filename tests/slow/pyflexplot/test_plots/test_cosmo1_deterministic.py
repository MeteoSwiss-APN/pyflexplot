# -*- coding: utf-8 -*-
"""
Test the elements of complete plots based on deterministic COSMO-1 data.
"""
# Third-party
import pytest

# Local
from .shared import _TestBase
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa:F401  # required by _TestBase.test

# Uncomment to recreate all references
# _TestBase = _TestCreateReference

INFILE_NAME = "flexpart_cosmo-1_2019093012.nc"


# class Test_Concentration(_TestCreateReference):
class Test_Concentration(_TestBase):
    reference = "ref_cosmo1_deterministic_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "input_variable": "concentration",
        "integrate": False,
        "lang": "de",
        "domain": "auto",
        "dimensions": {"species_id": 1, "time": 5, "level": 0},
    }


@pytest.mark.skip(f"{__file__.split('/')[-1]}::Test_Concentration2: TODO implment")
class Test_Concentration2(_TestCreateReference):
    # class Test_Concentration2(_TestBase):
    reference = "ref_cosmo1_deterministic_concentration_2"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "input_variable": "concentration",
        "lang": "en",
        "domain": "ch",
        "dimensions": {"species_id": 1, "time": (1, 6, 11), "level": 0},
    }
    n_plots = 3


# class Test_IntegratedConcentration(_TestCreateReference):
class Test_IntegratedConcentration(_TestBase):
    reference = "ref_cosmo1_deterministic_integrated_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "plot_type": "auto",
        "input_variable": "concentration",
        "integrate": True,
        "lang": "en",
        "domain": "ch",
        "dimensions": {"species_id": 1, "time": 10, "level": 0},
    }


# class Test_TotalDeposition(_TestCreateReference):
class Test_TotalDeposition(_TestBase):
    reference = "ref_cosmo1_deterministic_total_deposition"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "plot_type": "auto",
        "input_variable": "deposition",
        "combine_deposition_types": True,
        "integrate": True,
        "lang": "de",
        "domain": "auto",
        "dimensions": {"species_id": 1, "time": -1},
    }


# class Test_AffectedArea(_TestCreateReference):
class Test_AffectedArea(_TestBase):
    reference = "ref_cosmo1_deterministic_affected_area"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "input_variable": "deposition",
        "plot_variable": "affected_area_mono",
        "combine_deposition_types": True,
        "integrate": True,
        "lang": "en",
        "domain": "ch",
        "dimensions": {"species_id": 1, "time": -1, "deposition_type": ["dry", "wet"]},
    }

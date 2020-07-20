# -*- coding: utf-8 -*-
"""
Test the elements of complete plots based on ensemble COSMO-2 data.
"""
# Third-party
import pytest

# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa  # required by _TestBase.test

INFILE_NAME = "flexpart_cosmo-2e_2019072712_{ens_member:03d}.nc"
ENS_MEMBER_IDS = [0, 1, 5, 10, 15, 20]


# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference


class Test_EnsMedian_Concentration(_TestBase):
    reference = "ref_cosmo2e_ens_mean_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "cosmo2e",
        "ens_variable": "median",
        "input_variable": "concentration",
        "integrate": False,
        "combine_species": True,
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "en",
        "domain": "full",
        "dimensions": {"species_id": (1, 2), "time": 5, "level": 0},
    }


class Test_EnsMax_IntegratedConcentration(_TestBase):
    reference = "ref_cosmo2e_ens_max_integrated_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "cosmo2e",
        "ens_variable": "maximum",
        "input_variable": "concentration",
        "integrate": True,
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "de",
        "domain": "ch",
        "dimensions": {"species_id": 2, "time": 10, "level": 0},
    }


class Test_EnsMean_TotalDeposition(_TestBase):
    reference = "ref_cosmo2e_ens_mean_total_deposition"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "cosmo2e",
        "ens_variable": "mean",
        "input_variable": "deposition",
        "combine_deposition_types": True,
        "integrate": True,
        "combine_species": True,
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "en",
        "domain": "full",
        "dimensions": {"species_id": (1, 2), "time": -1},
    }


class Test_EnsProbability_WetDeposition(_TestBase):
    reference = "ref_cosmo2e_ens_probability_wet_deposition"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "cosmo2e",
        "ens_variable": "probability",
        "input_variable": "deposition",
        "integrate": True,
        "combine_species": True,
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "en",
        "domain": "full",
        "dimensions": {"deposition_type": "wet", "species_id": 2, "time": -1},
    }


@pytest.mark.skip("TODO implement plots like ens min affected area")
class Test_EnsMin_AffectedArea(_TestBase):
    reference = "ref_cosmo2e_ens_min_affected_area"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "cosmo2e",
        "input_variable": "deposition",
        "ens_variable": "minimum",
        "plot_variable": "affected_area",
        "combine_deposition_types": True,
        "integrate": True,
        "combine_species": True,
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "de",
        "domain": "ch",
        "dimensions": {
            "species_id": (1, 2),
            "time": -1,
            "deposition_type": ["dry", "wet"],
        },
    }


class Test_CloudArrivalTime(_TestBase):
    reference = "ref_cosmo2e_ens_cloud_arrival_time"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "cosmo2e",
        "ens_variable": "cloud_arrival_time",
        "input_variable": "concentration",
        "integrate": True,
        "ens_member_id": ENS_MEMBER_IDS,
        "ens_param_mem_min": 3,
        "ens_param_thr": 1e-6,
        "lang": "en",
        "domain": "full",
        "dimensions": {"species_id": 1, "time": 0, "level": 0},
    }


class Test_CloudDepartureTime(_TestBase):
    reference = "ref_cosmo2e_ens_cloud_departure_time"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "cosmo2e",
        "ens_variable": "cloud_departure_time",
        "input_variable": "concentration",
        "integrate": True,
        "combine_species": True,
        "combine_levels": True,
        "ens_member_id": ENS_MEMBER_IDS,
        "ens_param_mem_min": 2,
        "ens_param_thr": 1e-9,
        "lang": "de",
        "domain": "ch",
        "dimensions": {"species_id": (1, 2), "time": 3, "level": (0, 1, 2)},
    }


@pytest.mark.skip("WIP")
# class Test_MultipanelEnsStats_Concentration(_TestCreatePlot):
class Test_MultipanelEnsStats_Concentration(_TestBase):
    reference = "ref_cosmo2e_multipanel_ens_stats_integr_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": f"{reference}.png",
        "model": "cosmo2e",
        "ens_member_id": ENS_MEMBER_IDS,
        "input_variable": "concentration",
        "ens_variable": ["minimum", "maximum", "median", "mean"],
        "plot_type": "multipanel",
        "multipanel_param": "ens_variable",
        "integrate": True,
        "combine_species": True,
        "lang": "de",
        "domain": "ch",
        "dimensions": {"species_id": (1, 2), "time": -1, "level": 0},
    }

# -*- coding: utf-8 -*-
"""
Test the elements of complete plots based on ensemble COSMO-2 data.
"""
# Third-party
import pytest

# Local
from .shared import _CreateReference  # noqa:F401
from .shared import _TestBase
from .shared import datadir  # noqa  # required by _TestBase.test

INFILE_NAME = "flexpart_cosmo-2e_2019072712_{ens_member:03d}.nc"
ENS_MEMBER_IDS = [0, 1, 5, 10, 15, 20]


# class Test_EnsMedian_Concentration(_CreateReference):
class Test_EnsMedian_Concentration(_TestBase):
    reference = "ref_cosmo2e_ens_mean_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "dummy.png",
        "plot_type": "ens_median",
        "variable": "concentration",
        "integrate": False,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "de",
        "domain": "auto",
        "species_id": None,
        "time": (5,),
        "level": (0,),
    }


# class Test_EnsMax_IntegratedConcentration(_CreateReference):
class Test_EnsMax_IntegratedConcentration(_TestBase):
    reference = "ref_cosmo2e_ens_max_integrated_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "ens_max",
        "variable": "concentration",
        "integrate": True,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "de",
        "domain": "ch",
        "species_id": None,
        "time": (10,),
        "level": (0,),
    }


# class Test_EnsMean_TotalDeposition(_CreateReference):
class Test_EnsMean_TotalDeposition(_TestBase):
    reference = "ref_cosmo2e_ens_mean_total_deposition"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "ens_mean",
        "variable": "deposition",
        "deposition_type": "tot",
        "integrate": True,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "de",
        "domain": "auto",
        "species_id": None,
        "time": (-1,),
    }


@pytest.mark.skip("WIP")
# class Test_EnsMin_AffectedArea(_CreateReference):
class Test_EnsMin_AffectedArea(_TestBase):
    reference = "ref_cosmo2e_ens_min_affected_area"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "ens_minn",
        "variable": "affected_area",
        "deposition_type": "tot",
        "integrate": True,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "de",
        "domain": "auto",
        "species_id": None,
        "time": (-1,),
    }


@pytest.mark.skip("WIP")
# class Test_CloudArrivalTime(_CreateReference):
class Test_CloudArrivalTime(_TestBase):
    reference = "ref_cosmo2e_ens_cloud_arrival_time"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "ens_cloud_arrival_time",
        "variable": "concentration",
        "integrate": True,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "ens_param_mem_min": 3,
        "ens_param_thr": 1e-9,
        "lang": "de",
        "domain": "ch",
        "species_id": None,
        "time": (0,),
        "level": (0,),
    }

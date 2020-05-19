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

# Uncomment to recreate all references
# _TestBase = _TestCreateReference


# class Test_EnsMedian_Concentration(_TestCreateReference):
class Test_EnsMedian_Concentration(_TestBase):
    reference = "ref_cosmo2e_ens_mean_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "dummy.png",
        "plot_type": "ens_median",
        "input_variable": "concentration",
        "integrate": False,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "en",
        "domain": "auto",
        "species_id": (1, 2),
        "time": (5,),
        "level": (0,),
    }


# class Test_EnsMax_IntegratedConcentration(_TestCreateReference):
class Test_EnsMax_IntegratedConcentration(_TestBase):
    reference = "ref_cosmo2e_ens_max_integrated_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "ens_max",
        "input_variable": "concentration",
        "integrate": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "de",
        "domain": "ch",
        "species_id": (2,),
        "time": (10,),
        "level": (0,),
    }


# class Test_EnsMean_TotalDeposition(_TestCreateReference):
class Test_EnsMean_TotalDeposition(_TestBase):
    reference = "ref_cosmo2e_ens_mean_total_deposition"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "ens_mean",
        "input_variable": "deposition",
        "deposition_type": "tot",
        "integrate": True,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "lang": "en",
        "domain": "auto",
        "species_id": (1, 2),
        "time": (-1,),
    }


# SR_TODO Reactivate once input and derived variable have been separated
# SR_TODO (input: deposition; derived: affected_area; plot: ens_min)
# # @pytest.mark.skip("WIP")
# # class Test_EnsMin_AffectedArea(_TestCreateReference):
# class Test_EnsMin_AffectedArea(_TestBase):
#     reference = "ref_cosmo2e_ens_min_affected_area"
#     setup_dct = {
#         "infile": INFILE_NAME,
#         "outfile": "plot.png",
#         "input_variable": "deposition",
#         "plot_type": "ens_minn",
#         "plot_variable": "affected_area",
#         "deposition_type": "tot",
#         "integrate": True,
#         "combine_species": True,
#         "simulation_type": "ensemble",
#         "ens_member_id": ENS_MEMBER_IDS,
#         "lang": "de",
#         "domain": "ch",
#         "species_id": (1, 2),
#         "time": (-1,),
#     }


# @pytest.mark.skip("WIP")
# class Test_CloudArrivalTime(_TestCreatePlot):
# class Test_CloudArrivalTime(_TestCreateReference):
class Test_CloudArrivalTime(_TestBase):
    reference = "ref_cosmo2e_ens_cloud_arrival_time"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "ens_cloud_arrival_time",
        "input_variable": "concentration",
        "integrate": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "ens_param_mem_min": 3,
        "ens_param_thr": 1e-6,
        "lang": "en",
        "domain": "auto",
        "species_id": (1,),
        "time": (0,),
        "level": (0,),
    }


@pytest.mark.skip("WIP")
# class Test_CloudDepartureTime(_TestCreatePlot):
# class Test_CloudDepartureTime(_TestCreateReference):
class Test_CloudDepartureTime(_TestBase):
    reference = "ref_cosmo2e_ens_cloud_departure_time"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "ens_cloud_departure_time",
        "input_variable": "concentration",
        "integrate": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "ens_param_mem_min": 4,
        "ens_param_thr": 1e-7,
        "lang": "de",
        "domain": "ch",
        "species_id": (1, 2),
        "time": (0,),
        "level": (0, 1, 2),
    }

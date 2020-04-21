# -*- coding: utf-8 -*-
"""
Test the elements of complete plots based on ensemble COSMO-2 data.
"""
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
        "deposition_type": "none",
        "integrate": False,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "ens_param_mem_min": None,
        "ens_param_thr": None,
        "lang": "de",
        "domain": "auto",
        "nageclass": None,
        "noutrel": None,
        "numpoint": None,
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
        "deposition_type": "none",
        "integrate": True,
        "combine_species": True,
        "simulation_type": "ensemble",
        "ens_member_id": ENS_MEMBER_IDS,
        "ens_param_mem_min": None,
        "ens_param_thr": None,
        "lang": "de",
        "domain": "ch",
        "nageclass": None,
        "noutrel": None,
        "numpoint": None,
        "species_id": None,
        "time": (10,),
        "level": (0,),
    }

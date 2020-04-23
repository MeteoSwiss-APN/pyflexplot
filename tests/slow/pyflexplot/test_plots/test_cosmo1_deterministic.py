# -*- coding: utf-8 -*-
"""
Test the elements of complete plots based on deterministic COSMO-1 data.
"""
# Local
from .shared import _CreateReference  # noqa:F401
from .shared import _TestBase
from .shared import datadir  # noqa:F401  # required by _TestBase.test

INFILE_NAME = "flexpart_cosmo-1_2019093012.nc"


# class Test_Concentration(_CreateReference):
class Test_Concentration(_TestBase):
    reference = "ref_cosmo1_deterministic_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "dummy.png",
        "plot_type": "auto",
        "variable": "concentration",
        "integrate": False,
        "simulation_type": "deterministic",
        "lang": "de",
        "domain": "auto",
        "species_id": (1,),
        "time": (5,),
        "level": (0,),
    }

    def test(self, datadir):
        super().test(datadir)


# class Test_IntegratedConcentration(_CreateReference):
class Test_IntegratedConcentration(_TestBase):
    reference = "ref_cosmo1_deterministic_integrated_concentration"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "auto",
        "variable": "concentration",
        "integrate": True,
        "simulation_type": "deterministic",
        "lang": "de",
        "domain": "ch",
        "species_id": (1,),
        "time": (10,),
        "level": (0,),
    }


# class Test_TotalDeposition(_CreateReference):
class Test_TotalDeposition(_TestBase):
    reference = "ref_cosmo1_deterministic_total_deposition"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "auto",
        "variable": "deposition",
        "deposition_type": "tot",
        "integrate": True,
        "simulation_type": "deterministic",
        "lang": "de",
        "domain": "auto",
        "species_id": (1,),
        "time": (-1,),
    }


# class Test_AffectedArea(_CreateReference):
class Test_AffectedArea(_TestBase):
    reference = "ref_cosmo1_deterministic_affected_area"
    setup_dct = {
        "infile": INFILE_NAME,
        "outfile": "plot.png",
        "plot_type": "affected_area_mono",
        "variable": "deposition",
        "deposition_type": "tot",
        "integrate": True,
        "simulation_type": "deterministic",
        "lang": "de",
        "domain": "ch",
        "species_id": (1,),
        "time": (-1,),
    }

"""Test the elements of complete plots based on ensemble COSMO-2 data."""
# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa  # required by _TestBase.test

INFILE_1 = "flexpart_cosmo-e_2019072712_{ens_member:03d}.nc"
ENS_MEMBER_IDS = [0, 1, 5, 10, 15, 20]


# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference


class Test_EnsMedian_Concentration(_TestBase):
    reference = "ref_cosmo-e_ens_mean_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "ens_variable": "median",
                "plot_variable": "concentration",
                "integrate": False,
                "combine_species": True,
                "lang": "en",
                "domain": "full",
                "dimensions": {
                    "species_id": (1, 2),
                    "time": 5,
                    "level": 0,
                },
            }
        ],
    }


class Test_EnsMax_IntegratedConcentration(_TestBase):
    reference = "ref_cosmo-e_ens_max_integrated_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "ens_variable": "maximum",
                "plot_variable": "concentration",
                "integrate": True,
                "lang": "de",
                "domain": "ch",
                "dimensions": {
                    "species_id": 2,
                    "time": 10,
                    "level": 0,
                },
            }
        ],
    }


class Test_EnsMean_TotalDeposition(_TestBase):
    reference = "ref_cosmo-e_ens_mean_total_deposition"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "ens_variable": "mean",
                "plot_variable": "tot_deposition",
                "integrate": True,
                "combine_species": True,
                "lang": "en",
                "domain": "full",
                "dimensions": {
                    "species_id": (1, 2),
                    "time": -1,
                },
            }
        ],
    }


class Test_EnsProbability_WetDeposition(_TestBase):
    reference = "ref_cosmo-e_ens_probability_wet_deposition"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "ens_variable": "probability",
                "plot_variable": "wet_deposition",
                "integrate": True,
                "lang": "en",
                "domain": "full",
                "dimensions": {
                    "species_id": 1,
                    "time": -1,
                },
            }
        ],
    }


class Test_CloudArrivalTime(_TestBase):
    reference = "ref_cosmo-e_ens_cloud_arrival_time"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "ens_variable": "cloud_arrival_time",
                "plot_variable": "concentration",
                "integrate": True,
                "ens_params": {
                    "mem_min": 3,
                    "thr": 1e-6,
                },
                "lang": "en",
                "domain": "full",
                "dimensions": {
                    "species_id": 1,
                    "time": 0,
                    "level": 0,
                },
            }
        ],
    }


class Test_CloudDepartureTime(_TestBase):
    reference = "ref_cosmo-e_ens_cloud_departure_time"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "ens_variable": "cloud_departure_time",
                "plot_variable": "concentration",
                "integrate": True,
                "combine_species": True,
                "combine_levels": True,
                "ens_params": {
                    "mem_min": 2,
                    "thr": 1e-9,
                },
                "lang": "de",
                "domain": "ch",
                "dimensions": {
                    "species_id": (1, 2),
                    "time": 3,
                    "level": (0, 1, 2),
                },
            }
        ],
    }

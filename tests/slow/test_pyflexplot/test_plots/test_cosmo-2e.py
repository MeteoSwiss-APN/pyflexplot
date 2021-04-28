"""Test the elements of complete plots based on ensemble COSMO-2 data."""
# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa  # required by _TestBase.test

INFILE_1 = "flexpart_cosmo-2e_2021030503_{ens_member:03d}_MUE.nc"
ENS_MEMBER_IDS = [0, 5, 10, 15, 20]


# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference


class Test_EnsMin_AffectedArea(_TestBase):
    reference = "ref_cosmo-e_ens_min_affected_area"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-2E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "plot_variable": "affected_area",
                "ens_variable": "minimum",
                "integrate": True,
                "lang": "de",
                "domain": "full",
                "dimensions": {
                    "time": -1,
                    "level": 0,
                },
            }
        ],
    }


class Test_CloudArrivalTime(_TestBase):
    reference = "ref_cosmo-2e_ens_cloud_arrival_time"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-2E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "ens_variable": "cloud_arrival_time",
                "plot_variable": "concentration",
                "integrate": True,
                "ens_params": {
                    "mem_min": 1,
                    "thr": 1e-6,
                },
                "lang": "en",
                "domain": "full",
                "dimensions": {
                    "time": 0,
                    "level": 0,
                },
            }
        ],
    }


class Test_CloudDepartureTime(_TestBase):
    reference = "ref_cosmo-2e_ens_cloud_departure_time"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-2E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": [
            {
                "ens_variable": "cloud_departure_time",
                "plot_variable": "concentration",
                "integrate": True,
                "combine_levels": True,
                "ens_params": {
                    "mem_min": 1,
                    "thr": 0,
                },
                "lang": "de",
                "domain": "ch",
                "dimensions": {
                    "time": 3,
                    "level": 0,
                },
            }
        ],
    }


class Test_MultipanelEnsStats_Concentration(_TestBase):
    reference = "ref_cosmo-2e_multipanel_ens_stats_integr_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "multipanel",
            "multipanel_param": "ens_variable",
        },
        "model": {
            "name": "COSMO-2E",
            "ens_member_id": ENS_MEMBER_IDS,
        },
        "panels": {
            "plot_variable": "concentration",
            "ens_variable": ["minimum", "maximum", "median", "mean"],
            "integrate": True,
            "lang": "de",
            "domain": "ch",
            "dimensions": {
                "time": -1,
                "level": 0,
            },
        },
    }

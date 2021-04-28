"""Test the elements of complete plots based on deterministic COSMO-1 data."""
# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa:F401  # required by _TestBase.test

INFILE_1 = "flexpart_cosmo-1_2019093012.nc"


# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference


class Test_Concentration(_TestBase):
    reference = "ref_cosmo-1_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-1",
        },
        "panels": [
            {
                "plot_variable": "concentration",
                "integrate": False,
                "lang": "de",
                "domain": "full",
                "dimensions": {
                    "species_id": 1,
                    "time": 5,
                    "level": 0,
                },
            }
        ],
    }


class Test_IntegratedConcentration(_TestBase):
    reference = "ref_cosmo-1_integrated_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "auto",
        },
        "model": {
            "name": "COSMO-1",
        },
        "panels": [
            {
                "plot_variable": "concentration",
                "integrate": True,
                "lang": "en",
                "domain": "ch",
                "dimensions": {
                    "species_id": 1,
                    "time": 10,
                    "level": 0,
                },
            }
        ],
    }


class Test_TotalDeposition(_TestBase):
    reference = "ref_cosmo-1_total_deposition"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "auto",
        },
        "model": {
            "name": "COSMO-1",
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
    reference = "ref_cosmo-1_affected_area"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-1",
        },
        "panels": [
            {
                "domain": "ch",
                "plot_variable": "affected_area",
                "integrate": True,
                "lang": "en",
                "dimensions": {
                    "level": 0,
                    "species_id": 1,
                    "time": -1,
                },
            }
        ],
    }


class Test_CloudArrivalTime(_TestBase):
    reference = "ref_cosmo-1_cloud_arrival_time"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "COSMO-1",
        },
        "panels": [
            {
                "plot_variable": "cloud_arrival_time",
                "integrate": False,
                "lang": "en",
                "domain": "ch",
                "dimensions": {
                    "species_id": 1,
                    "time": 0,
                    "level": 0,
                },
            }
        ],
    }

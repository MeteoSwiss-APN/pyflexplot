"""Test the elements of complete plots based on deterministic IFS-HRES data."""
# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa:F401  # required by _TestBase.test

INFILE_1 = "flexpart_ifs_20200317000000.nc"
INFILE_2 = "flexpart_ifs-hres_1018_20200921000000.nc"


# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference


class Test_Concentration(_TestBase):
    reference = "ref_ifs_deterministic_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "IFS-HRES",
        },
        "panels": [
            {
                "plot_variable": "concentration",
                "integrate": False,
                "lang": "de",
                "domain": "cloud",
                "dimensions": {
                    "species_id": 1,
                    "time": 5,
                    "level": 0,
                },
            }
        ],
    }


class Test_IntegratedConcentration(_TestBase):
    reference = "ref_ifs_deterministic_integrated_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "auto",
        },
        "model": {
            "name": "IFS-HRES",
        },
        "panels": [
            {
                "plot_variable": "concentration",
                "integrate": True,
                "lang": "en",
                "domain": "cloud",
                "dimensions": {
                    "species_id": 1,
                    "time": 10,
                    "level": 0,
                },
            }
        ],
    }


class Test_TotalDeposition(_TestBase):
    reference = "ref_ifs_deterministic_total_deposition"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "auto",
        },
        "model": {
            "name": "IFS-HRES",
        },
        "panels": [
            {
                "plot_variable": "tot_deposition",
                "integrate": True,
                "lang": "de",
                "domain": "cloud",
                "dimensions": {
                    "species_id": 1,
                    "time": -1,
                },
            }
        ],
    }


class Test_TotalDeposition_EmptyField(_TestBase):
    reference = "ref_ifs_deterministic_total_deposition_empty_field"
    setup_dct = {
        "files": {
            "input": INFILE_2,
            "output": f"{reference}.png",
        },
        "layout": {
            "plot_type": "auto",
        },
        "model": {
            "name": "IFS-HRES",
        },
        "panels": [
            {
                "plot_variable": "tot_deposition",
                "integrate": True,
                "lang": "de",
                "domain": "cloud",
                "dimensions": {
                    "species_id": 3,
                    "time": 0,
                },
            }
        ],
    }


class Test_AffectedArea(_TestBase):
    reference = "ref_ifs_deterministic_affected_area"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "IFS-HRES",
        },
        "panels": [
            {
                "plot_variable": "affected_area",
                "integrate": True,
                "lang": "en",
                "domain": "cloud",
                "dimensions": {
                    "species_id": 1,
                    "level": 0,
                    "time": -1,
                },
            }
        ],
    }


class Test_CloudDepartureTime(_TestBase):
    reference = "ref_ifs_deterministic_cloud_departure_time"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.png",
        },
        "model": {
            "name": "IFS-HRES",
        },
        "panels": [
            {
                "plot_variable": "cloud_departure_time",
                "integrate": False,
                "lang": "en",
                "domain": "cloud",
                "dimensions": {
                    "species_id": 1,
                    "time": 0,
                    "level": 0,
                },
            }
        ],
    }

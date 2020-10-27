"""Shared resources for setup tests."""
# Standard library
from typing import Any
from typing import Dict

# First-party
from pyflexplot.setup import Setup

DUMMY_PARAMS: Dict[str, Any] = {
    "infile": "none",
    "outfile": "none",
    "model": "none",
}

DEFAULT_PARAMS: Dict[str, Any] = {
    "base_time": None,
    "combine_deposition_types": False,
    "combine_levels": False,
    "combine_species": False,
    "dimensions_default": "all",
    "dimensions": {
        "deposition_type": None,
        "level": None,
        "nageclass": None,
        "noutrel": None,
        "numpoint": None,
        "species_id": None,
        "time": None,
    },
    "domain": "full",
    "domain_size_lat": None,
    "domain_size_lon": None,
    "ens_member_id": None,
    "ens_param_mem_min": None,
    "ens_param_pctl": None,
    "ens_param_thr": None,
    "ens_param_time_win": None,
    "ens_variable": "none",
    "input_variable": "concentration",
    "integrate": False,
    "lang": "en",
    "multipanel_param": None,
    "outfile_time_format": "%Y%m%d%H%M",
    "plot_type": "auto",
    "plot_variable": "auto",
    "scale_fact": 1.0,
}


DEFAULT_SETUP: Setup = Setup.create({**DUMMY_PARAMS, **DEFAULT_PARAMS})

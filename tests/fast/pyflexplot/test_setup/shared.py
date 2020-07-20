# -*- coding: utf-8 -*-
"""
Shared resources for setup tests.
"""
# Standard library
from typing import Any
from typing import Dict

# First-party
from pyflexplot.setup import Setup

DEFAULT_PARAMS: Dict[str, Any] = {
    "combine_deposition_types": False,
    "combine_levels": False,
    "combine_species": False,
    "domain": "full",
    "ens_member_id": None,
    "ens_param_mem_min": None,
    "ens_param_thr": None,
    "ens_param_time_win": None,
    "ens_variable": "none",
    "infile": "none",
    "input_variable": "concentration",
    "integrate": False,
    "lang": "en",
    "model": "none",
    "multipanel_param": None,
    "outfile": "none",
    "plot_type": "auto",
    "plot_variable": "auto",
    "dimensions": {
        "deposition_type": None,
        "level": None,
        "nageclass": None,
        "noutrel": None,
        "numpoint": None,
        "species_id": None,
        "time": None,
    },
}


DEFAULT_SETUP: Setup = Setup.create(DEFAULT_PARAMS)

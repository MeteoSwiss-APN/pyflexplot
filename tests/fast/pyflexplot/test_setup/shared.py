# -*- coding: utf-8 -*-
"""
Shared resources for setup tests.
"""
# Standard library
from typing import Any
from typing import Dict

# First-party
from pyflexplot.setup import InputSetup

DEFAULT_KWARGS: Dict[str, Any] = {
    "infile": "foo.nc",
    "outfile": "bar.png",
}


DEFAULT_SETUP = InputSetup.create(
    {
        **DEFAULT_KWARGS,
        "combine_deposition_types": False,
        "combine_species": False,
        "combine_species": False,
        "deposition_type": None,
        "domain": "auto",
        "ens_member_id": None,
        "ens_variable": "none",
        "input_variable": "concentration",
        "integrate": False,
        "lang": "en",
        "dimensions": {
            "level": None,
            "nageclass": None,
            "noutrel": None,
            "numpoint": None,
            "species_id": None,
            "time": None,
        },
        "plot_type": "auto",
    }
)

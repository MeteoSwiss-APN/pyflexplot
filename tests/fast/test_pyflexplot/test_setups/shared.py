"""Shared resources for setup tests."""
# Standard library
from typing import Any
from typing import Dict

# First-party
from pyflexplot.setups.dimensions import CoreDimensions
from pyflexplot.setups.dimensions import Dimensions
from pyflexplot.setups.plot_panel_setup import EnsembleParams
from pyflexplot.setups.plot_panel_setup import PlotPanelSetup
from pyflexplot.setups.setup import ModelSetup
from pyflexplot.setups.setup import PlotSetup
from srutils.dict import merge_dicts

MANDATORY_RAW_DEFAULT_PARAMS: Dict[str, Any] = {
    "infile": "none",
    "outfile": "none",
    "model": {
        "name": "none",
    },
}

OPTIONAL_RAW_DEFAULT_PARAMS: Dict[str, Any] = {
    "core": {
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
        "ens_params": {
            "mem_min": None,
            "pctl": None,
            "thr": None,
            "thr_type": "lower",
        },
        "ens_variable": "none",
        "input_variable": "concentration",
        "integrate": False,
        "lang": "en",
    },
    "model": {
        "base_time": None,
        "ens_member_id": None,
        "simulation_type": "deterministic",
    },
    "multipanel_param": None,
    "plot_type": "auto",
    "outfile_time_format": "%Y%m%d%H%M",
    "scale_fact": 1.0,
}


RAW_DEFAULT_PARAMS = merge_dicts(
    MANDATORY_RAW_DEFAULT_PARAMS, OPTIONAL_RAW_DEFAULT_PARAMS
)


DEFAULT_PARAMS = merge_dicts(
    RAW_DEFAULT_PARAMS,
    {
        "model": ModelSetup(**RAW_DEFAULT_PARAMS["model"]),
        "core": PlotPanelSetup(
            **merge_dicts(
                RAW_DEFAULT_PARAMS["core"],
                {
                    "dimensions": Dimensions(
                        [CoreDimensions(**RAW_DEFAULT_PARAMS["core"]["dimensions"])]
                    ),
                    "ens_params": EnsembleParams(
                        **RAW_DEFAULT_PARAMS["core"]["ens_params"]
                    ),
                },
            ),
        ),
    },
)


DEFAULT_SETUP: PlotSetup = PlotSetup.create(RAW_DEFAULT_PARAMS)

"""Shared resources for setup tests."""
# Standard library
from copy import deepcopy
from typing import Any
from typing import Dict

# First-party
from pyflexplot.setups.dimensions import CoreDimensions
from pyflexplot.setups.dimensions import Dimensions
from pyflexplot.setups.layout_setup import LayoutSetup
from pyflexplot.setups.model_setup import ModelSetup
from pyflexplot.setups.plot_panel_setup import EnsembleParams
from pyflexplot.setups.plot_panel_setup import PlotPanelSetup
from pyflexplot.setups.plot_panel_setup import PlotPanelSetupGroup
from pyflexplot.setups.plot_setup import PlotSetup
from srutils.dict import merge_dicts

MANDATORY_RAW_DEFAULT_PARAMS: Dict[str, Any] = {
    "infile": "none",
    "outfile": "none",
    "model": {
        "name": "none",
    },
}

OPTIONAL_RAW_DEFAULT_PARAMS: Dict[str, Any] = {
    "multipanel_param": None,
    "outfile_time_format": "%Y%m%d%H%M",
    "scale_fact": 1.0,
    "layout": {
        "plot_type": "auto",
        "type": "post_vintage",
    },
    "model": {
        "base_time": None,
        "ens_member_id": None,
        "simulation_type": "deterministic",
    },
    "panels": [
        {
            "combine_levels": False,
            "combine_species": False,
            "dimensions_default": "all",
            "dimensions": {
                "level": None,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": None,
                "variable": "concentration",
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
            "plot_variable": "concentration",
            "integrate": False,
            "lang": "en",
        }
    ],
}


RAW_DEFAULT_PARAMS = merge_dicts(
    MANDATORY_RAW_DEFAULT_PARAMS, OPTIONAL_RAW_DEFAULT_PARAMS
)


DEFAULT_PARAMS = merge_dicts(
    RAW_DEFAULT_PARAMS,
    {
        "layout": LayoutSetup(**RAW_DEFAULT_PARAMS["layout"]),
        "model": ModelSetup(**RAW_DEFAULT_PARAMS["model"]),
        "panels": PlotPanelSetupGroup(
            [
                PlotPanelSetup(
                    **merge_dicts(
                        RAW_DEFAULT_PARAMS["panels"][0],
                        {
                            "dimensions": Dimensions(
                                [
                                    CoreDimensions(
                                        **RAW_DEFAULT_PARAMS["panels"][0]["dimensions"]
                                    )
                                ]
                            ),
                            "ens_params": EnsembleParams(
                                **RAW_DEFAULT_PARAMS["panels"][0]["ens_params"]
                            ),
                        },
                        overwrite_seqs=True,
                        overwrite_seq_dicts=True,
                    ),
                ),
            ],
        ),
    },
    overwrite_seqs=True,
    overwrite_seq_dicts=True,
)


RAW_DEFAULT_PARAMS_PRE = deepcopy(RAW_DEFAULT_PARAMS)
DEFAULT_SETUP: PlotSetup = PlotSetup.create(RAW_DEFAULT_PARAMS)
assert RAW_DEFAULT_PARAMS == RAW_DEFAULT_PARAMS_PRE

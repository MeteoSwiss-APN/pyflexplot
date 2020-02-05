# -*- coding: utf-8 -*-
"""
Configuration and configuration file.
"""
import tomlkit
import warnings

from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Union


@dataclass
class Config:
    #
    infiles: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Input file path(s). Main contain format keys."},
    )
    member_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "List of ensemble member ids. Omit for deterministic simulations "
            "Use the format key '{member_id}' to embed the member id(s) in ``infiles`` "
            "or ``outfile``."
        },
    )
    outfile: Optional[str] = field(
        default=None, metadata={"help": "Output file path. May contain format keys."},
    )
    #
    variable: Optional[str] = field(
        default="concentration",
        metadata={
            "help": "Input variable to be plotted.",
            "choices": ["concentration", "deposition"],
        },
    )
    simulation_type: Optional[str] = field(
        default="deterministic",
        metadata={
            "help": "Type of the simulation.",
            "choices": ["deterministic", "ensemble"],
        },
    )
    plot_type: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Plot type.",
            "choices": [
                "auto",
                "affected_area",
                "affected_area_mono",
                "ens_mean",
                "ens_max",
                "ens_thr_agrmt",
            ],
        },
    )
    #
    domain: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Plot domain. Defaults to 'data', which derives the domain size "
            "from the input data. Use the format key '{domain}' to embed the domain "
            "name in the plot file path.",
            "choices": ["auto", "ch"],
        },
    )
    lang: Optional[str] = field(
        default="en",
        metadata={
            "help": "Language. Use the format key '{lang}' to embed it into the plot "
            "file path.",
            "choices": ["en", "de"],
        },
    )
    #
    age_class_idx: Optional[int] = field(
        default=0,
        metadata={
            "help": "Index of age class (zero-based). Use the format key "
            "'{age_class_idx}' to embed it into the output file path.",
        },
    )
    deposition_type: Optional[str] = field(
        default="tot",
        metadata={
            "help": "Type of deposition. Part of the plot variable name that may be "
            "embedded in the plot file path with the format key '{variable}'.",
            "choices": ["tot", "wet", "dry"],
        },
    )
    integrate: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Integrate field over time. Use the format key '{integrate}' to "
            "embed '[no-]int' in the plot file path."
        },
    )
    level_idx: Optional[Union[int, List[int]]] = field(
        default=0,
        metadata={
            "help": "Index/indices of vertical level (zero-based, bottom-up). To sum "
            "up multiple levels, combine their indices with '+'. Format key: "
            "'{level_idx}'.",
        },
    )
    nout_rel_idx: Optional[int] = field(
        default=0,
        metadata={
            "help": "Index of noutrel (zero-based). Format key: '{noutrel_idx}'.",
        },
    )
    release_point_idx: Optional[int] = field(
        default=0,
        metadata={
            "help": "Index of release point (zero-based). Format key: '{rls_pt_idx}'."
        },
    )
    species_id: Optional[Union[int, List[int]]] = field(
        default=1,
        metadata={
            "help": "Species id(s) (default: 0). To sum up multiple species, combine "
            "their ids with '+'. Format key: '{species_id}'.",
        },
    )
    time_idx: Optional[int] = field(
        default=0,
        metadata={"help": "Index of time (zero-based). Format key: '{time_idx}'."},
    )

    def __post__init__(self):
        if self.deposition_type == "tot":
            self.deposition_type = ["dry", "wet"]

    def update(self, dct):
        for key, val in dct.items():
            if not hasattr(self, key):
                warnings.warn(f"{type(self).__name__}.update: unknown key: {key}")
            else:
                setattr(self, key, val)

    @classmethod
    def from_file(cls, file):
        s = file.read()
        try:
            data = tomlkit.parse(s)
        except Exception as e:
            raise Exception(
                f"error parsing TOML file {file.name} ({type(e).__name__}: {e})"
            )
        # SR_TMP <
        required_keys = ["plot"]
        optional_keys = []
        unknown_keys = [
            k for k in data.keys() if k not in required_keys + optional_keys
        ]
        if unknown_keys:
            raise NotImplementedError(
                f"{len(unknown_keys)} unknown section(s) in {file.name}", unknown_keys,
            )
        # SR_TMP >
        obj = cls(**data["plot"])
        return obj

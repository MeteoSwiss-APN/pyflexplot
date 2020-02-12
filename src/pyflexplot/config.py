# -*- coding: utf-8 -*-
"""
Configuration and configuration file.
"""
# Standard library
import dataclasses
import warnings
from dataclasses import field
from typing import List
from typing import Optional
from typing import Union

# Third-party
import tomlkit
from pydantic.dataclasses import dataclass as pydantic_dataclass

# First-party
from srutils.dict import decompress_nested_dict


@pydantic_dataclass(frozen=True)
class Config:
    """
    PyFlexPlot configuration.

    Args:

        infiles: Input file path(s). May contain format keys.

        member_ids: List of ensemble member ids. Omit for deterministic
            simulations. Use the format key '{member_id}' to embed the member
            id(s) in ``infiles`` or ``outfile``.

        outfile: Output file path. May contain format keys.

        variable: Input variable to be plotted. Choices: "concentration",
            "deposition".

        simulation_type: Type of the simulation. Choices: "deterministic",
            "ensemble".

        plot_type: Plot type. Choices: "auto", "affected_area",
            "affected_area_mono", "ens_mean", "ens_max", "ens_thr_agrmt".

        domain: Plot domain. Defaults to 'data', which derives the domain size
            from the input data. Use the format key '{domain}' to embed the
            domain name in the plot file path. Choices": "auto", "ch".

        lang: Language. Use the format key '{lang}' to embed it into the plot
            file path. Choices: "en", "de".

        age_class_idx: Index of age class (zero-based). Use the format key
            '{age_class_idx}' to embed it into the output file path.

        deposition_type: Type of deposition. Part of the plot variable name
            that may be embedded in the plot file path with the format key
            '{variable}'. Choices: "tot", "wet", "dry".

        integrate: Integrate field over time. Use the format key '{integrate}'
            to embed '[no-]int' in the plot file path.

        level_idx: Index/indices of vertical level (zero-based, bottom-up). To
            sum up multiple levels, combine their indices with '+'. Use the
            format key '{level_idx}' to embed it in the plot file path.

        nout_rel_idx: Index of noutrel (zero-based). Use the format key
            '{noutrel_idx}' to embed it in the plot file path.

        release_point_idx: Index of release point (zero-based). Use the format
            key '{rls_pt_idx}' to embed it in the plot file path.

        species_id: Species id(s). To sum up multiple species, combine their
            ids with '+'. Use the format key '{species_id}' to embed it in the
            plot file path.

        time_idx: Index of time (zero-based). Use the format key '{time_idx}'
            to embed it in the plot file path.

    """

    infiles: Optional[List[str]] = None
    member_ids: Optional[List[int]] = None
    outfile: Optional[str] = None
    #
    variable: str = "concentration"
    simulation_type: str = "deterministic"
    plot_type: str = "auto"
    #
    domain: str = "auto"
    lang: str = "en"
    #
    age_class_idx: int = 0
    deposition_type: str = "tot"
    integrate: bool = False
    level_idx: Union[int, List[int]] = 0
    nout_rel_idx: int = 0
    release_point_idx: int = 0
    species_id: Union[int, List[int]] = 1
    time_idx: int = 0

    def __post__init_post_parse__(self):
        if isinstance(self.infiles, str):
            self.infiles = [self.infiles]
        if self.deposition_type == "tot":
            self.deposition_type = ["dry", "wet"]

    @classmethod
    def as_config(cls, obj):
        if isinstance(obj, cls):
            return obj
        return cls(**obj)

    def update(self, dct, skip_none=False):
        for key, val in dct.items():
            if not hasattr(self, key):
                warnings.warn(f"{type(self).__name__}.update: unknown key: {key}")
            elif skip_none and val is None:
                continue
            else:
                setattr(self, key, val)

    def as_dict(self):
        return dataclasses.asdict(self)

    def __len__(self):
        return len(self.as_dict())

    def __eq__(self, other):
        try:
            other_as_dict = dataclasses.asdict(other)
        except TypeError:
            try:
                other_as_dict = dict(other)
            except TypeError:
                return False
        return self.as_dict() == other_as_dict


class ConfigSet:
    """
    A set of ``Config`` objects.
    """

    def __init__(self, configs):
        self._configs = [Config.as_config(obj) for obj in configs]

    def __len__(self):
        return len(self._configs)

    def __iter__(self):
        for config in self._configs:
            yield config

    def __eq__(self, other):
        return [c.as_dict() for c in self._configs] == other


class ConfigFile:
    """
    Configuration file to be read from and/or written to disk.
    """

    def __init__(self, path):
        self.path = path

    def read(self):
        """
        Read the configuration from a text file (TOML format).
        """
        with open(self.path, "r") as f:
            s = f.read()
        try:
            raw_data = tomlkit.parse(s)
        except Exception as e:
            raise Exception(f"error parsing TOML file {path} ({type(e).__name__}: {e})")
        values, paths = decompress_nested_dict(
            raw_data, match_end=lambda key: not key.startswith("_"), return_paths=True,
        )
        configs = ConfigSet(values)
        return configs

    def write(self, *args, **kwargs):
        """
        Write the configuration to a text file (TOML format).
        """
        raise NotImplementedError(f"{type(self).__name__}.write")

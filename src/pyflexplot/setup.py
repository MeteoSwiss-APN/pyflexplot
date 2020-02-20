# -*- coding: utf-8 -*-
"""
Plot setup and setup files.
"""
# Standard library
import dataclasses
from typing import List
from typing import Optional
from typing import Union

# Third-party
import toml
from pydantic.dataclasses import dataclass as pydantic_dataclass

# First-party
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards


@pydantic_dataclass(frozen=True)
class Setup:
    """
    PyFlexPlot setup.

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

    infiles: List[str]
    outfile: str
    member_ids: Optional[List[int]] = None
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
    #
    scale_fact: Optional[float] = None
    reverse_legend: bool = False

    def __post_init_post_parse__(self):
        pass

    @classmethod
    def as_setup(cls, obj):
        if isinstance(obj, cls):
            return obj
        return cls(**obj)

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

    def tmp_cls_name(self):
        if self.simulation_type == "deterministic":
            return f"{self.variable}"
        elif self.simulation_type == "ensemble":
            return f"{self.plot_type}_{self.variable}"
        raise NotImplementedError(f"simulation_type='{self.simulation_type}'")


class SetupCollection:
    """
    A set of ``Setup`` objects.
    """

    def __init__(self, setups):
        self._setups = [Setup.as_setup(obj) for obj in setups]

    def __repr__(self):
        s_setups = "\n  ".join([""] + [str(c) for c in self._setups])
        return f"{type(self).__name__}([{s_setups}\n])"

    def __len__(self):
        return len(self._setups)

    def __iter__(self):
        for setup in self._setups:
            yield setup

    def __eq__(self, other):
        return self.as_dicts() == other

    def as_dicts(self):
        return [c.as_dict() for c in self._setups]


class SetupFile:
    """
    Setup file to be read from and/or written to disk.
    """

    def __init__(self, path):
        self.path = path

    def read(self):
        """
        Read the setup from a text file in TOML format.
        """
        with open(self.path, "r") as f:
            try:
                raw_data = toml.load(f)
            except Exception as e:
                raise Exception(
                    f"error parsing TOML file {self.path} ({type(e).__name__}: {e})"
                )
        if not raw_data:
            raise ValueError(f"empty setup file", self.path)
        raw_data = nested_dict_resolve_wildcards(raw_data)
        values = decompress_nested_dict(
            raw_data, branch_end_criterion=lambda key: not key.startswith("_"),
        )
        setups = SetupCollection(values)
        return setups

    def write(self, *args, **kwargs):
        """
        Write the setup to a text file in TOML format.
        """
        raise NotImplementedError(f"{type(self).__name__}.write")

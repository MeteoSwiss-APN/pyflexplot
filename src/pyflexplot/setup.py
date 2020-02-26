# -*- coding: utf-8 -*-
"""
Plot setup and setup files.
"""
# Standard library
import dataclasses
from typing import Any
from typing import Collection
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import toml
from pydantic import BaseModel
from pydantic import validator

# First-party
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards


class Setup(BaseModel):
    """
    PyFlexPlot setup.

    Args:
        age_class_idx: Index of age class (zero-based). Use the format key
            '{age_class_idx}' to embed it into the output file path.

        deposition_type: Type of deposition. Part of the plot variable name
            that may be embedded in the plot file path with the format key
            '{variable}'. Choices: "tot", "wet", "dry".

        domain: Plot domain. Defaults to 'data', which derives the domain size
            from the input data. Use the format key '{domain}' to embed the
            domain name in the plot file path. Choices": "auto", "ch".

        ens_member_ids: Ensemble member ids. Omit for deterministic simulations.

        infiles: Input file path(s). May contain format keys.

        integrate: Integrate field over time.

        lang: Language. Use the format key '{lang}' to embed it into the plot
            file path. Choices: "en", "de".

        level_idx: Index/indices of vertical level (zero-based, bottom-up). To
            sum up multiple levels, combine their indices with '+'. Use the
            format key '{level_idx}' to embed it in the plot file path.

        nout_rel_idx: Index of noutrel (zero-based). Use the format key
            '{noutrel_idx}' to embed it in the plot file path.

        outfile: Output file path. May contain format keys.

        plot_type: Plot type. Choices: "auto", "affected_area",
            "affected_area_mono", "ens_mean", "ens_max", "ens_thr_agrmt".

        release_point_idx: Index of release point (zero-based). Use the format
            key '{rls_pt_idx}' to embed it in the plot file path.

        simulation_type: Type of the simulation. Choices: "deterministic",
            "ensemble".

        species_id: Species id(s). To sum up multiple species, combine their
            ids with '+'. Use the format key '{species_id}' to embed it in the
            plot file path.

        time_idxs: Time step indices.. Use the format key '{time_idx}' to embed
            it in the plot file path.

        variable: Input variable to be plotted. Choices: "concentration",
            "deposition".

    """

    class Config:  # noqa
        allow_mutation = False
        extra = "forbid"

    age_class_idx: int = 0
    deposition_type: Union[str, Tuple[str, str]] = "tot"
    domain: str = "auto"
    ens_member_ids: Optional[Tuple[int, ...]] = None
    infiles: Tuple[str, ...]
    integrate: bool = False
    lang: str = "en"
    level_idx: Union[int, Tuple[int, ...]] = 0
    nout_rel_idx: int = 0
    outfile: str
    plot_type: str = "auto"
    release_point_idx: int = 0
    reverse_legend: bool = False
    scale_fact: Optional[float] = None
    simulation_type: str = "deterministic"
    species_id: Union[int, Tuple[int, ...]] = 1
    time_idxs: Tuple[int, ...] = (0,)
    variable: str = "concentration"

    @validator("deposition_type")
    def _validate_deposition_type(cls, val: Union[str, Tuple[str, str]]) -> str:
        if isinstance(val, tuple):
            if set(val) == {"dry", "wet"}:
                return "tot"
        elif isinstance(val, str):
            if val in ["dry", "wet", "tot"]:
                return val
        raise ValueError("invalid deposition type", val)

    @classmethod
    def as_setup(cls, obj: Union[Mapping[str, Any], "Setup"]) -> "Setup":
        if isinstance(obj, cls):
            return obj
        return cls(**obj)  # type: ignore

    def __repr__(self) -> str:  # type: ignore
        def fmt(obj):
            if isinstance(obj, str):
                return f"'{obj}'"
            return str(obj)

        s_attrs = ",\n  ".join(f"{k}={fmt(v)}" for k, v in self.dict().items())
        return f"{type(self).__name__}(\n  {s_attrs},\n)"

    def __str__(self) -> str:  # type: ignore
        return repr(self)

    def __len__(self) -> int:
        return len(dict(self))

    def __eq__(self, other: object) -> bool:
        try:
            other_dict = dict(other)  # type: ignore
        except TypeError:
            try:
                other_dict = dataclasses.asdict(other)
            except TypeError:
                return False
        return self.dict() == other_dict

    def derive(self, **kwargs: Mapping[str, Any]) -> "Setup":
        return type(self)(**{**self.dict(), **kwargs})

    def tmp_cls_name(self):
        if self.simulation_type == "deterministic":
            return f"{self.variable}"
        elif self.simulation_type == "ensemble":
            return f"{self.plot_type}_{self.variable}"
        raise NotImplementedError(f"simulation_type='{self.simulation_type}'")


class SetupCollection:
    """A set of ``Setup`` objects."""

    def __init__(self, setups: Collection[Union[Mapping[str, Any], Setup]]) -> None:
        self._setups = [Setup.as_setup(obj) for obj in setups]

    def __repr__(self) -> str:
        s_setups = "\n  ".join([""] + [str(c) for c in self._setups])
        return f"{type(self).__name__}([{s_setups}\n])"

    def __len__(self) -> int:
        return len(self._setups)

    def __iter__(self) -> Iterator[Setup]:
        for setup in self._setups:
            yield setup

    def __eq__(self, other: object) -> bool:
        try:
            other_dicts = other.as_dicts()  # type: ignore
        except AttributeError:
            other_dicts = [dict(obj) for obj in other]  # type: ignore
        self_dicts = self.dicts()
        return all(obj in other_dicts for obj in self_dicts) and all(
            obj in self_dicts for obj in other_dicts
        )

    def dicts(self) -> List[Mapping[str, Any]]:
        return [obj.dict() for obj in self._setups]


class SetupFile:
    """Setup file to be read from and/or written to disk."""

    def __init__(self, path: str) -> None:
        self.path: str = path

    def read(self) -> SetupCollection:
        """Read the setup from a text file in TOML format."""
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

    def write(self, *args, **kwargs) -> None:
        """Write the setup to a text file in TOML format."""
        raise NotImplementedError(f"{type(self).__name__}.write")

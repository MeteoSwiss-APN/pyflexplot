# -*- coding: utf-8 -*-
"""
Plot setup and setup files.
"""
# Standard library
import dataclasses
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import overload

# Third-party
import toml
from pydantic import BaseModel
from pydantic import validator

# First-party
from srutils.dict import decompress_multival_dict
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards

# Some plot-specific default values
ENS_THR_AGRMT_THR_DEFAULT: float = 1e-9
ENS_CLOUD_ARRIVAL_TIME_MEM_MIN_DEFAULT = 1
ENS_CLOUD_ARRIVAL_TIME_THR_DEFAULT: float = 1e-9


class Setup(BaseModel):
    """
    PyFlexPlot setup.

    Args:
        age_class_idx: Index of age class (zero-based). Use the format key
            '{age_class}' to embed it into the output file path.

        combine_species: Sum up over all specified species. Otherwise, each is
            plotted separately.

        deposition_type: Type of deposition. Part of the plot variable name
            that may be embedded in the plot file path with the format key
            '{variable}'. Choices: "tot", "wet", "dry", "none".

        domain: Plot domain. Defaults to 'data', which derives the domain size
            from the input data. Use the format key '{domain}' to embed the
            domain name in the plot file path. Choices": "auto", "ch".

        ens_member_ids: Ensemble member ids. Use the format key '{ens_member}'
            to embed it into the input file path. Omit for deterministic
            simulations.

        ens_param_mem_min: Minimum number of ensemble members used to compute
            some ensemble variables. Its precise meaning depends on the
            variable.

        ens_param_thr: Threshold used to compute some ensemble variables. Its
            precise meaning depends on the variable.

        infiles: Input file path(s). May contain format keys.

        integrate: Integrate field over time.

        lang: Language. Use the format key '{lang}' to embed it into the plot
            file path. Choices: "en", "de".

        level_idx: Index/indices of vertical level (zero-based, bottom-up). To
            sum up multiple levels, combine their indices with '+'. Use the
            format key '{level}' to embed it in the plot file path.

        nout_rel_idx: Index of noutrel (zero-based). Use the format key
            '{nout_rel}' to embed it in the plot file path.

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

        time_idcs: Time step indices.. Use the format key '{time}' to embed it
            in the plot file path.

        variable: Input variable to be plotted. Choices: "concentration",
            "deposition".

    """

    class Config:  # noqa
        allow_mutation = False
        extra = "forbid"

    age_class_idx: int = 0
    combine_species: bool = False
    deposition_type: Union[str, Tuple[str, str]] = "none"
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
    time_idcs: Tuple[int, ...] = (0,)
    variable: str = "concentration"

    # Derived parameters
    ens_param_mem_min: Optional[int] = None
    ens_param_thr: Optional[float] = None

    @validator("deposition_type", always=True)
    def _validate_deposition_type(cls, value: Union[str, Tuple[str, str]]) -> str:
        if isinstance(value, tuple):
            if set(value) == {"dry", "wet"}:
                return "tot"
        elif isinstance(value, str):
            if value in ["dry", "wet", "tot", "none"]:
                return value
        raise ValueError("deposition_type is invalid", value)

    @validator("ens_param_mem_min", always=True)
    def _validate_ens_param_mem_min(
        cls, value: Optional[int], values: Dict[str, Any],
    ) -> Optional[int]:
        if value is not None:
            if values["simulation_type"] != "ensemble":
                raise ValueError(
                    "ens_param_mem_min incompatible with simulation_type",
                    value,
                    values["simulation_type"],
                )
        elif values["plot_type"] == "ens_cloud_arrival_time":
            value = ENS_CLOUD_ARRIVAL_TIME_MEM_MIN_DEFAULT
        return value

    @validator("ens_param_thr", always=True)
    def _validate_ens_param_thr(
        cls, value: Optional[float], values: Dict[str, Any],
    ) -> Optional[float]:
        if value is not None:
            if values["simulation_type"] != "ensemble":
                raise ValueError(
                    "ens_param_thr incompatible with simulation_type",
                    value,
                    values["simulation_type"],
                )
        elif values["plot_type"] == "ens_thr_agrmt":
            value = ENS_THR_AGRMT_THR_DEFAULT
        elif values["plot_type"] == "ens_cloud_arrival_time":
            value = ENS_CLOUD_ARRIVAL_TIME_THR_DEFAULT
        return value

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

    @overload
    def derive(self, params: Mapping[str, Any]) -> "Setup":
        ...

    @overload
    def derive(self, params: Sequence[Mapping[str, Any]]) -> List["Setup"]:
        ...

    def derive(
        self, params: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    ) -> Union["Setup", List["Setup"]]:
        """Derive ``Setup`` object(s) with adapted parameters."""
        if isinstance(params, Sequence):
            return [self.derive(params_i) for params_i in params]
        return type(self)(**{**self.dict(), **params})

    def decompress(self) -> List["Setup"]:
        """Create multiple ``Setup`` objects with one-value parameters only."""

        def create(dct):
            return Setup(**{**dct, "time_idcs": [dct["time_idcs"]]})

        # SR_TMP < TODO cleaner solution
        skip = ["infiles", "ens_member_ids"]
        dcts = decompress_multival_dict(self.dict(), skip=skip)
        if self.variable == "deposition" and self.deposition_type == "tot":
            sub_setups = []
            for dct in dcts:
                for deposition_type in ["dry", "wet"]:
                    sub_setups.append(
                        create(dct).derive({"deposition_type": deposition_type})
                    )
        else:
            sub_setups = [create(dct) for dct in dcts]
        # SR_TMP >

        return sub_setups

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

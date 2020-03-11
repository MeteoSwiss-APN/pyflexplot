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
from pydantic import root_validator
from pydantic import validator

# First-party
from srutils.dict import compress_multival_dicts
from srutils.dict import decompress_multival_dict
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards

# Some plot-specific default values
ENS_THR_AGRMT_THR_DEFAULT: float = 1e-9
ENS_CLOUD_ARRIVAL_TIME_MEM_MIN_DEFAULT = 1
ENS_CLOUD_ARRIVAL_TIME_THR_DEFAULT: float = 1e-9


def setup_repr(obj: "Setup") -> str:
    def fmt(obj):
        if isinstance(obj, str):
            return f"'{obj}'"
        return str(obj)

    s_attrs = ",\n  ".join(f"{k}={fmt(v)}" for k, v in obj.dict().items())
    return f"{type(obj).__name__}(\n  {s_attrs},\n)"


class Setup(BaseModel):
    """
    PyFlexPlot setup.

    Args:
        nageclass: Index of age class (zero-based). Use the format key
            '{nageclass}' to embed it into the output file path.

        combine_species: Sum up over all specified species. Otherwise, each is
            plotted separately.

        deposition_type: Type of deposition. Part of the plot variable name
            that may be embedded in the plot file path with the format key
            '{variable}'. Choices: "tot", "wet", "dry", "none".

        domain: Plot domain. Defaults to 'data', which derives the domain size
            from the input data. Use the format key '{domain}' to embed the
            domain name in the plot file path. Choices": "auto", "ch".

        ens_member_id: Ensemble member ids. Use the format key '{ens_member}'
            to embed it into the input file path. Omit for deterministic
            simulations.

        ens_param_mem_min: Minimum number of ensemble members used to compute
            some ensemble variables. Its precise meaning depends on the
            variable.

        ens_param_thr: Threshold used to compute some ensemble variables. Its
            precise meaning depends on the variable.

        infile: Input file path(s). May contain format keys.

        integrate: Integrate field over time.

        lang: Language. Use the format key '{lang}' to embed it into the plot
            file path. Choices: "en", "de".

        level: Index/indices of vertical level (zero-based, bottom-up). To
            sum up multiple levels, combine their indices with '+'. Use the
            format key '{level}' to embed it in the plot file path.

        noutrel: Index of noutrel (zero-based). Use the format key
            '{noutrel}' to embed it in the plot file path.

        outfile: Output file path. May contain format keys.

        plot_type: Plot type. Choices: "auto", "affected_area",
            "affected_area_mono", "ens_mean", "ens_max", "ens_thr_agrmt".

        numpoint: Index of release point (zero-based).

        simulation_type: Type of the simulation. Choices: "deterministic",
            "ensemble".

        species_id: Species id(s). To sum up multiple species, combine their
            ids with '+'. Use the format key '{species_id}' to embed it in the
            plot file path.

        time: Time step indices (zero-based). Use the format key '{time}'
            to embed one in the plot file path.

        variable: Input variable to be plotted. Choices: "concentration",
            "deposition".

    """

    class Config:  # noqa
        allow_mutation = False
        extra = "forbid"

    nageclass: Tuple[int, ...] = (0,)
    combine_species: bool = False
    # SR_TMP < TODO remove option to specify "tot" as ["dry", "wet"]
    # deposition_type: str = "none"
    deposition_type: Union[str, Tuple[str, str]] = "none"
    # SR_TMP >
    domain: str = "auto"
    ens_member_id: Tuple[Optional[int], ...] = (None,)
    infile: Union[str, Tuple[str, ...]]
    integrate: bool = False
    lang: str = "en"
    noutrel: Tuple[int, ...] = (0,)
    outfile: str
    plot_type: str = "auto"
    numpoint: Tuple[int, ...] = (0,)
    reverse_legend: bool = False
    scale_fact: Optional[float] = None
    simulation_type: str = "deterministic"
    species_id: Union[int, Tuple[int, ...]] = (1,)
    time: Tuple[int, ...] = (0,)
    variable: str = "concentration"

    # Derived parameters
    level: Tuple[Optional[int], ...] = (None,)
    ens_param_mem_min: Optional[int] = None
    ens_param_thr: Optional[float] = None

    @root_validator(pre=True)
    def _to_sequence(cls, values):
        """Ensure that all parameter values constitute a sequence."""
        for param, value in values.items():
            # SR_TMP < TODO Figure out whether/how to pass multiple types
            if param in ["deposition_type"]:
                continue
            # SR_TMP >
            # SR_TMP < Handle one parameter after the other
            if param in [
                "combine_species",
                "deposition_type",
                "domain",
                "integrate",
                "lang",
                "outfile",
                "plot_type",
                "reverse_legend",
                "scale_fact",
                "simulation_type",
                "variable",
                "ens_param_mem_min",
                "ens_param_thr",
            ]:
                continue
            # SR_TMP >
            if not isinstance(value, Sequence) or isinstance(value, str):
                values[param] = [value]
        return values

    @validator("time")
    def _validate_time(cls, value: Tuple[int, ...]):
        if len(value) == 0:
            raise ValueError("missing time")
        return value

    @validator("deposition_type", always=True)
    def _init_deposition_type(cls, value: Union[str, Tuple[str, str]]) -> str:
        if value in ["dry", "wet", "tot", "none"]:
            assert isinstance(value, str)  # for mypy
            return value
        elif set(value) == {"dry", "wet"}:
            return "tot"
        raise ValueError("deposition_type is invalid", value)

    @validator("ens_param_mem_min", always=True)
    def _init_ens_param_mem_min(
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
    def _init_ens_param_thr(
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

    @validator("level", always=True)
    def _init_level(
        cls, value: Tuple[Optional[int], ...], values: Dict[str, Any],
    ) -> Optional[Tuple[int, ...]]:
        if value == (None,):
            if values["variable"] == "concentration":
                return (0,)
            else:
                return None
        elif values["variable"] == "deposition":
            raise ValueError(
                "level must be None for variable", value, values["variable"],
            )
        return value  # type: ignore

    @classmethod
    def as_setup(cls, obj: Union[Mapping[str, Any], "Setup"]) -> "Setup":
        if isinstance(obj, cls):
            return obj
        return cls(**obj)  # type: ignore

    def __repr__(self) -> str:  # type: ignore
        return setup_repr(self)

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
    def derive(self, params: Sequence[Mapping[str, Any]]) -> "SetupCollection":
        ...

    def derive(
        self, params: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    ) -> Union["Setup", "SetupCollection"]:
        """Derive ``Setup`` object(s) with adapted parameters."""
        if isinstance(params, Sequence):
            return SetupCollection([self.derive(params_i) for params_i in params])
        # SR_TMP < TODO find cleaner solution (w/o duplication of logic)
        if (
            self.variable == "concentration"
            and params.get("variable") == "deposition"
            and "level" not in params
        ):
            params["level"] = None  # type: ignore
        # SR_TMP >
        return type(self)(**{**self.dict(), **params})

    @classmethod
    def compress(cls, setups: "SetupCollection") -> "Setup":
        if not setups:
            raise ValueError("missing setups")
        dct = compress_multival_dicts(setups.dicts(), cls_seq=tuple)
        return cls(**dct)

    def decompress(
        self,
        select: Optional[Collection[str]] = None,
        *,
        skip: Optional[Collection[str]] = None,
    ) -> "SetupCollection":
        """Create multiple ``Setup`` objects with one-value parameters only."""

        if skip is None:
            skip = ["infile", "ens_member_id"]

        dct = self.dict()

        # Handle deposition type
        expand_deposition_type = (
            select is None or "deposition_type" in select
        ) and "deposition_type" not in skip
        if expand_deposition_type and dct["deposition_type"] == "tot":
            dct["deposition_type"] = ("dry", "wet")

        # Decompress dict
        dcts = decompress_multival_dict(dct, select=select, skip=skip)

        def create_setup(dct):
            if isinstance(dct["time"], int):
                dct["time"] = [dct["time"]]
            return Setup(**dct)

        return SetupCollection([create_setup(dct) for dct in dcts])


class SetupCollection:
    """A collection of ``Setup`` objects."""

    def __init__(self, setups: Collection[Union[Mapping[str, Any], Setup]]) -> None:
        self._setups: List[Setup] = [Setup.as_setup(obj) for obj in setups]

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
        semi_raw_data = nested_dict_resolve_wildcards(
            raw_data, double_only_to_ends=True,
        )
        data = decompress_nested_dict(
            semi_raw_data, branch_end_criterion=lambda key: not key.startswith("_"),
        )
        setups = SetupCollection(data)
        return setups

    def write(self, *args, **kwargs) -> None:
        """Write the setup to a text file in TOML format."""
        raise NotImplementedError(f"{type(self).__name__}.write")

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
from typing import Type
from typing import Union
from typing import overload

# Third-party
import toml
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic import parse_obj_as
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


def setup_repr(obj: Union["CoreInputSetup", "InputSetup"]) -> str:
    def fmt(obj):
        if isinstance(obj, str):
            return f"'{obj}'"
        return str(obj)

    s_attrs = ",\n  ".join(f"{k}={fmt(v)}" for k, v in obj.dict().items())
    return f"{type(obj).__name__}(\n  {s_attrs},\n)"


class CoreInputSetup(BaseModel):
    """
    PyFlexPlot core setup with exactly one value per parameter.

    See ``InputSetup`` for details on the parameters.

    """

    class Config:  # noqa
        allow_mutation = False
        extra = "forbid"

    # Basics
    infile: str
    outfile: str
    plot_type: str = "auto"
    variable: str = "concentration"

    # Tweaks
    deposition_type: str = "none"
    integrate: bool = False
    combine_species: bool = False

    # Ensemble-related
    simulation_type: str = "deterministic"
    ens_member_id: Optional[int] = None
    ens_param_mem_min: Optional[int] = None
    ens_param_thr: Optional[float] = None

    # Plot appearance
    lang: str = "en"
    domain: str = "auto"

    # Dimensions
    nageclass: int = 0
    noutrel: int = 0
    numpoint: int = 0
    species_id: int = 1
    time: int = 0
    level: Optional[int] = None

    def __repr__(self) -> str:  # type: ignore
        return setup_repr(self)

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "CoreInputSetup":
        return cls(**params)

    @classmethod
    def as_setup(
        cls, obj: Union[Mapping[str, Any], "CoreInputSetup"],
    ) -> "CoreInputSetup":
        if isinstance(obj, cls):
            return obj
        assert isinstance(obj, Mapping)  # mypy
        return cls(**obj)


class InputSetup(BaseModel):
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
        # allow_mutation = False
        extra = "forbid"

    # Basics
    infile: Tuple[str, ...]
    outfile: str
    plot_type: str = "auto"
    variable: str = "concentration"

    # Tweaks
    deposition_type: Union[str, Tuple[str, str]] = "none"
    integrate: bool = False
    combine_species: bool = False

    # Ensemble-related
    simulation_type: str = "deterministic"
    ens_member_id: Optional[Tuple[int, ...]] = None
    ens_param_mem_min: Optional[int] = None
    ens_param_thr: Optional[float] = None

    # Plot appearance
    lang: str = "en"
    domain: str = "auto"

    # Dimensions
    nageclass: Optional[Tuple[int, ...]] = None
    noutrel: Optional[Tuple[int, ...]] = None
    numpoint: Optional[Tuple[int, ...]] = None
    species_id: Optional[Tuple[int, ...]] = None
    time: Optional[Tuple[int, ...]] = None
    level: Optional[Tuple[int, ...]] = None

    @validator("deposition_type", always=True)
    def _init_deposition_type(cls, value: Union[str, Tuple[str, str]]) -> str:
        if value in ["dry", "wet", "tot", "none"]:
            assert isinstance(value, str)  # mypy
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
        cls, value: Optional[Tuple[int, ...]], values: Dict[str, Any],
    ) -> Optional[Tuple[int, ...]]:
        if value is not None and values["variable"] == "deposition":
            raise ValueError(
                "level must be None for variable", value, values["variable"],
            )
        return value

    @classmethod
    def create(cls, params: Dict[str, Any]) -> "InputSetup":
        """Create an instance of ``InputSetup``.

        Args:
            params: Parameters to instatiate ``InputSetup``. In contrast to
                direct instatiation, all ``Tuple`` parameters may be passed
                directly, e.g., as `{"time": 0}` instead of `{"time": (0,)}`.

        """
        for param, value in params.items():
            field = cls.__fields__[param]
            if value is None and field.allow_none:
                continue
            if value == "*" and field.allow_none:
                params[param] = None
                continue
            field_type = field.outer_type_
            try:
                # Try to convert value to the field type
                parse_obj_as(field_type, value)
            except ValidationError as e:
                # Conversion failed, so let's try something else!
                error_type = e.errors()[0]["type"]
                if error_type == "type_error.sequence":
                    try:
                        # Try again, with the value in a sequence
                        parse_obj_as(field_type, [value])
                    except ValidationError:
                        # Still not working; let's give up!
                        raise ValueError(
                            f"value of param {param} with type {type(value).__name__} "
                            f"incompatible with field type {field_type}, both directly "
                            f"and in a sequence"
                        )
                    else:
                        # Now it worked; wrapping value in list and we're good!
                        params[param] = [value]
                else:
                    raise NotImplementedError("unknown ValidationError", error_type, e)
        return cls(**params)

    @classmethod
    def as_setup(cls, obj: Union[Mapping[str, Any], "InputSetup"]) -> "InputSetup":
        if isinstance(obj, cls):
            return obj
        return cls.create(obj)  # type: ignore

    # SR_TODO consider renaming this method (sth. containing 'dimensions')
    def replace_nones(self, meta_data: Mapping[str, Any]) -> List[str]:
        dimensions = meta_data["dimensions"]
        completed = []

        if self.time is None:
            self.time = tuple(range(dimensions["time"]["size"]))
            completed.append("time")

        if self.level is None:
            if self.variable == "concentration":
                if "level" in dimensions:
                    self.level = tuple(range(dimensions["level"]["size"]))
                    completed.append("level")

        if self.species_id is None:
            self.species_id = meta_data["analysis"]["species_ids"]
            completed.append("species_id")

        if self.nageclass is None:
            if "nageclass" in dimensions:
                self.nageclass = tuple(range(dimensions["nageclass"]["size"]))
                completed.append("nageclass")

        if self.noutrel is None:
            if "noutrel" in dimensions:
                self.noutrel = tuple(range(dimensions["noutrel"]["size"]))
                completed.append("nageclass")

        if self.numpoint is None:
            if "numpoint" in dimensions:
                self.numpoint = tuple(range(dimensions["numpoint"]["size"]))
                completed.append("numpoint")

        return completed

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
    def derive(self, params: Mapping[str, Any]) -> "InputSetup":
        ...

    @overload
    def derive(self, params: Sequence[Mapping[str, Any]]) -> "InputSetupCollection":
        ...

    def derive(
        self, params: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    ) -> Union["InputSetup", "InputSetupCollection"]:
        """Derive ``InputSetup`` object(s) with adapted parameters."""
        if isinstance(params, Sequence):
            return InputSetupCollection([self.derive(params_i) for params_i in params])
        # SR_TMP < TODO find cleaner solution (w/o duplication of logic)
        if (
            self.variable == "concentration"
            and params.get("variable") == "deposition"
            and "level" not in params
        ):
            params["level"] = None  # type: ignore
        # SR_TMP >
        params = {**self.dict(), **params}
        return type(self).create(params)

    @classmethod
    def compress(cls, setups: "InputSetupCollection") -> "InputSetup":
        if not setups:
            raise ValueError("missing setups")
        dct = compress_multival_dicts(setups.dicts(), cls_seq=tuple)
        return cls.create(dct)

    def decompress(self) -> "CoreInputSetupCollection":
        return self._decompress(None, None)

    def decompress_partially(
        self, select: Optional[Collection[str]], skip: Optional[Collection[str]] = None,
    ) -> "InputSetupCollection":
        if (select, skip) == (None, None):
            return self._decompress(None, None, InputSetup)
        elif skip is None:
            assert select is not None  # mypy
            return self._decompress(select, None)
        elif select is None:
            assert skip is not None  # mypy
            return self._decompress(None, skip)
        else:
            return self._decompress(select, skip)

    @overload
    def _decompress(
        self,
        select: None,
        skip: None,
        cls_setup: Optional[Type["CoreInputSetup"]] = None,
    ) -> "CoreInputSetupCollection":
        ...

    @overload
    def _decompress(
        self, select: None, skip: None, cls_setup: Type["InputSetup"],
    ) -> "InputSetupCollection":
        ...

    @overload
    def _decompress(
        self,
        select: None,
        skip: Collection[str],
        cls_setup: Optional[Union[Type["CoreInputSetup"], Type["InputSetup"]]] = None,
    ) -> "InputSetupCollection":
        ...

    @overload
    def _decompress(
        self,
        select: Collection[str],
        skip: None,
        cls_setup: Optional[Union[Type["CoreInputSetup"], Type["InputSetup"]]] = None,
    ) -> "InputSetupCollection":
        ...

    @overload
    def _decompress(
        self,
        select: Collection[str],
        skip: Collection[str],
        cls_setup: Optional[Union[Type["CoreInputSetup"], Type["InputSetup"]]] = None,
    ) -> "InputSetupCollection":
        ...

    def _decompress(self, select=None, skip=None, cls_setup=None):
        """Create multiple ``InputSetup`` objects with one-value parameters only."""

        if cls_setup is None:
            if (select, skip) == (None, None):
                cls_setup = CoreInputSetup
            else:
                cls_setup = InputSetup
        if cls_setup is CoreInputSetup:
            cls_setup_collection = CoreInputSetupCollection
        elif cls_setup is InputSetup:
            cls_setup_collection = InputSetupCollection
        else:
            raise ValueError("invalid cls_setup", cls_setup)

        dct = self.dict()

        # Handle deposition type
        expand_deposition_type = (
            select is None or "deposition_type" in select
        ) and "deposition_type" not in (skip or [])
        if expand_deposition_type and dct["deposition_type"] == "tot":
            dct["deposition_type"] = ("dry", "wet")

        # Decompress dict
        dcts = decompress_multival_dict(dct, select=select, skip=skip)

        def create_setup(dct):
            if isinstance(dct["time"], int):
                dct["time"] = [dct["time"]]
            return cls_setup.create(dct)

        return cls_setup_collection([cls_setup.create(dct) for dct in dcts])


# SR_TMP <<< TODO Consider merging with CoreInputSetupCollection (failed due to mypy)
class InputSetupCollection:
    def __init__(self, setups: Collection[InputSetup]) -> None:
        self._setups: List[InputSetup] = [setup for setup in setups]

    @classmethod
    def create(
        cls, setups: Collection[Union[Mapping[str, Any], InputSetup]]
    ) -> "InputSetupCollection":
        setup_objs: List[InputSetup] = []
        for obj in setups:
            setup_obj = InputSetup.as_setup(obj)
            setup_objs.append(setup_obj)
        return cls(setup_objs)

    def __repr__(self) -> str:
        s_setups = "\n  ".join([""] + [str(c) for c in self._setups])
        return f"{type(self).__name__}([{s_setups}\n])"

    def __len__(self) -> int:
        return len(self._setups)

    def __iter__(self) -> Iterator[InputSetup]:
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
        return [setup.dict() for setup in self._setups]

    def replace_nones(
        self,
        meta_data: Mapping[str, Any],
        decompress_skip: Optional[Collection[str]] = None,
    ) -> List[str]:
        orig_setups = [setup for setup in self._setups]
        self._setups.clear()
        for setup in orig_setups:
            completed: List[str] = setup.replace_nones(meta_data)
            select = [dim for dim in completed if dim not in (decompress_skip or [])]
            self._setups.extend(setup.decompress_partially(select))
        return completed


# SR_TMP <<< TODO Consider merging with InputSetupCollection (failed due to mypy)
class CoreInputSetupCollection:
    def __init__(self, setups: Collection[CoreInputSetup]) -> None:
        self._setups: List[CoreInputSetup] = [setup for setup in setups]

    @classmethod
    def create(
        cls, setups: Collection[Union[Mapping[str, Any], CoreInputSetup]]
    ) -> "CoreInputSetupCollection":
        setup_objs: List[CoreInputSetup] = []
        for obj in setups:
            setup_obj = CoreInputSetup.as_setup(obj)
            setup_objs.append(setup_obj)
        return cls(setup_objs)

    # SR_TMP <
    __repr__ = InputSetupCollection.__repr__
    __len__ = InputSetupCollection.__len__
    __iter__ = InputSetupCollection.__iter__
    __eq__ = InputSetupCollection.__eq__
    dicts = InputSetupCollection.dicts
    # SR_TMP >


class InputSetupFile:
    """InputSetup file to be read from and/or written to disk."""

    def __init__(self, path: str) -> None:
        self.path: str = path

    def read(self) -> InputSetupCollection:
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
        setups = InputSetupCollection.create(data)
        return setups

    def write(self, *args, **kwargs) -> None:
        """Write the setup to a text file in TOML format."""
        raise NotImplementedError(f"{type(self).__name__}.write")

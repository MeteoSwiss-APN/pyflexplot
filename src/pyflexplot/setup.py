# -*- coding: utf-8 -*-
"""
Plot setup and setup files.
"""
# Standard library
import dataclasses
import typing
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
from pydantic import root_validator
from pydantic import validator
from pydantic.fields import ModelField

# First-party
from srutils.dict import compress_multival_dicts
from srutils.dict import decompress_multival_dict
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards

# Some plot-specific default values
ENS_PROBABILITY_DEFAULT_PARAM_THR = 1e-8
ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN = 10
ENS_CLOUD_TIME_DEFAULT_PARAM_THR = 1e-7
ENS_CLOUD_PROB_DEFAULT_PARAM_TIME_WIN = 12


# pylint: disable=E0213  # no-self-argument (validators)
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

        ens_param_time_win: Tim window used to compute some ensemble variables.
            Its precise meaning depends on the variable.

        ense_variable: Ensemble variable computed from plot variable.

        infile: Input file path(s). May contain format keys.

        input_variable: Input variable. Choices: "concentration", "deposition".

        integrate: Integrate field over time.

        lang: Language. Use the format key '{lang}' to embed it into the plot
            file path. Choices: "en", "de".

        level: Index/indices of vertical level (zero-based, bottom-up). To
            sum up multiple levels, combine their indices with '+'. Use the
            format key '{level}' to embed it in the plot file path.

        noutrel: Index of noutrel (zero-based). Use the format key
            '{noutrel}' to embed it in the plot file path.

        outfile: Output file path. May contain format keys.

        plot_type: Plot type.

        plot_variable: Variable computed from input variable.

        numpoint: Index of release point (zero-based).

        simulation_type: Type of the simulation (deterministic or ensemble).

        species_id: Species id(s). To sum up multiple species, combine their
            ids with '+'. Use the format key '{species_id}' to embed it in the
            plot file path.

        time: Time step indices (zero-based). Use the format key '{time}'
            to embed one in the plot file path.

    """

    class Config:  # noqa
        # allow_mutation = False
        extra = "forbid"

    # Basics
    infile: str
    outfile: str
    input_variable: str = "concentration"
    plot_variable: str = "auto"
    ens_variable: str = "none"
    plot_type: str = "auto"

    # Tweaks
    deposition_type: Union[str, Tuple[str, str]] = "none"
    integrate: bool = False
    combine_species: bool = False

    # Ensemble-related
    simulation_type: str = "deterministic"
    ens_member_id: Optional[Tuple[int, ...]] = None
    ens_param_mem_min: Optional[int] = None
    ens_param_thr: Optional[float] = None
    ens_param_time_win: Optional[float] = None

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

    @root_validator
    def _check_variables_etc(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        input_variables = ["concentration", "deposition"]
        assert values["input_variable"] in input_variables, values["input_variable"]
        plot_variables = ["auto", "affected_area", "affected_area_mono"]
        assert values["plot_variable"] in plot_variables, values["plot_variable"]
        ens_variables = ["none"]
        assert values["ens_variable"] in ens_variables, values["ens_variable"]
        plot_types = [
            "auto",
            "ensemble_minimum",
            "ensemble_mean",
            "ensemble_median",
            "ensemble_maximum",
            "ensemble_probability",
            "ensemble_cloud_arrival_time",
            "ensemble_cloud_departure_time",
            "ensemble_cloud_occurrence_probability",
        ]
        assert values["plot_type"] in plot_types, values["plot_type"]
        return values

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
        elif values["plot_type"] in [
            "ensemble_cloud_arrival_time",
            "ensemble_cloud_departure_time",
        ]:
            value = ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN
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
        elif values["plot_type"] == "ensemble_probability":
            value = ENS_PROBABILITY_DEFAULT_PARAM_THR
        elif values["plot_type"] in [
            "ensemble_cloud_arrival_time",
            "ensemble_cloud_departure_time",
        ]:
            value = ENS_CLOUD_TIME_DEFAULT_PARAM_THR
        return value

    @validator("ens_param_time_win", always=True)
    def _init_ens_param_time_win(
        cls, value: Optional[float], values: Dict[str, Any],
    ) -> Optional[float]:
        if value is not None:
            if values["simulation_type"] != "ensemble":
                raise ValueError(
                    "ens_param_time_win incompatible with simulation_type",
                    value,
                    values["simulation_type"],
                )
        elif values["plot_type"] == "ensemble_cloud_occurrence_probability":
            value = ENS_CLOUD_PROB_DEFAULT_PARAM_TIME_WIN
        return value

    @validator("level", always=True)
    def _init_level(
        cls, value: Optional[Tuple[int, ...]], values: Dict[str, Any],
    ) -> Optional[Tuple[int, ...]]:
        if value is not None and values["input_variable"] == "deposition":
            raise ValueError(
                "level must be None for variable", value, values["input_variable"],
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
        singles = ["infile"]
        for param, value in params.items():
            if param in singles:
                continue
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

    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        """Cast a parameter to the appropriate type."""
        try:
            field = cls.__fields__[param]
        except KeyError:
            raise ValueError("invalid parameter name", param, sorted(cls.__fields__))
        if isinstance(value, Collection) and not isinstance(value, str):
            try:
                outer_type = field_get_outer_type(field, generic=True)
            except TypeError:
                raise ValueError("invalid parameter value: collection", param, value)
            try:
                outer_type(value)
            except ValueError:
                raise ValueError("invalid parameter value", param, value, outer_type)
            else:
                return [cls.cast(param, val) for val in value]
        if issubclass(field.type_, bool):
            if value in [True, "True"]:
                return True
            elif value in [False, "False"]:
                return False
        else:
            try:
                return field.type_(value)
            except ValueError:
                pass
        raise ValueError("invalid parameter value", param, value, field.type_)

    @classmethod
    def cast_many(
        cls,
        params: Union[Collection[Tuple[str, Any]], Mapping[str, Any]],
        *,
        list_separator: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(params, Mapping):
            params_dct: Dict[str, Any] = {}
            for param, value in params:
                if param in params_dct:
                    raise ValueError("duplicate parameter", param)
                params_dct[param] = value
            return cls.cast_many(params_dct, list_separator=list_separator)
        params_cast = {}
        for param, value in params.items():
            if (
                list_separator is not None
                and isinstance(value, str)
                and list_separator in value
            ):
                value = value.split(list_separator)
            params_cast[param] = cls.cast(param, value)
        return params_cast

    # SR_TODO consider renaming this method (sth. containing 'dimensions')
    # pylint: disable=R0912  # too-many-branches
    def complete_dimensions(self, meta_data: Mapping[str, Any]) -> List[str]:
        dimensions = meta_data["dimensions"]
        completed = []

        if self.time is None:
            self.time = tuple(range(dimensions["time"]["size"]))
            completed.append("time")

        # SR_TMP < does this belong here?
        nts = dimensions["time"]["size"]
        time_new = []
        for its in self.time:
            if its < 0:
                its += nts
                assert 0 <= its < nts
            time_new.append(its)
        self.time = tuple(time_new)
        # SR_TMP >

        if self.level is None:
            if self.input_variable == "concentration":
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
            self.input_variable == "concentration"
            and params.get("input_variable") == "deposition"
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

    @classmethod
    def compress_partially(
        cls, setups: "InputSetupCollection", skip: List[str],
    ) -> "InputSetupCollection":
        dcts: List[Dict[str, Any]] = setups.dicts()
        preserved_params_lst: List[Dict[str, Any]] = []
        for dct in dcts:
            preserved_params = {}
            for param in skip:
                try:
                    preserved_params[param] = dct.pop(param)
                except ValueError:
                    raise ValueError("invalid param", param)
            if preserved_params not in preserved_params_lst:
                preserved_params_lst.append(preserved_params)
        partial_dct = compress_multival_dicts(setups.dicts(), cls_seq=tuple)
        setup_lst: List["InputSetup"] = []
        for preserved_params in preserved_params_lst:
            dct = {**partial_dct, **preserved_params}
            setup_lst.append(cls.create(dct))
        return InputSetupCollection(setup_lst)

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

        return cls_setup_collection([cls_setup.create(dct) for dct in dcts])


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
    input_variable: str = "concentration"
    plot_variable: str = "auto"
    ens_variable: str = "none"
    plot_type: str = "auto"

    # Tweaks
    deposition_type: str = "none"
    integrate: bool = False
    combine_species: bool = False

    # Ensemble-related
    simulation_type: str = "deterministic"
    ens_member_id: Optional[int] = None
    ens_param_mem_min: Optional[int] = None
    ens_param_thr: Optional[float] = None
    ens_param_time_win: Optional[float] = None

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


# SR_TMP <<< TODO Consider merging with CoreInputSetupCollection (failed due to mypy)
class InputSetupCollection:
    def __init__(self, setups: Collection[InputSetup]) -> None:
        if not isinstance(setups, Collection) or (
            setups and not isinstance(next(iter(setups)), InputSetup)
        ):
            raise ValueError("setups is not a collection of InputSetup objects", setups)
        self._setups: List[InputSetup] = list(setups)

    @classmethod
    def create(
        cls, setups: Collection[Union[Mapping[str, Any], InputSetup]]
    ) -> "InputSetupCollection":
        setup_objs: List[InputSetup] = []
        for obj in setups:
            setup_obj = InputSetup.as_setup(obj)
            setup_objs.append(setup_obj)
        return cls(setup_objs)

    def copy(self) -> "InputSetupCollection":
        return type(self)([setup.copy() for setup in self])

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

    def dicts(self) -> List[Dict[str, Any]]:
        return [setup.dict() for setup in self._setups]

    def compress(self) -> InputSetup:
        return InputSetup.compress(self)

    def decompress_partially(
        self, select: Collection[str], skip: Optional[Collection[str]] = None,
    ) -> List["InputSetupCollection"]:
        setups = self._setups
        sub_setup_lst_lst: List[List[InputSetup]] = []
        for setup in setups:
            sub_setups = setup.decompress_partially(select, skip)
            if not sub_setup_lst_lst:
                sub_setup_lst_lst = [[sub_setup] for sub_setup in sub_setups]
            else:
                assert len(sub_setups) == len(sub_setup_lst_lst)
                for idx, sub_setup in enumerate(sub_setups):
                    sub_setup_lst_lst[idx].append(sub_setup)
        return [
            InputSetupCollection(sub_setup_lst) for sub_setup_lst in sub_setup_lst_lst
        ]

    # SR_TMP <<< TODO cleaner solution
    def decompress_grouped_by_time(self) -> List["InputSetupCollection"]:
        # Note: This is what's left-over of FldSpecs, specifically FldSpecs.create
        return self.decompress_two_step(
            select_outer=["time"], select_inner=None, skip=["ens_member_id"],
        )

    def decompress_two_step(
        self,
        select_outer: List[str],
        select_inner: Optional[Collection[str]],
        skip: Optional[Collection[str]] = None,
    ) -> List["InputSetupCollection"]:
        skip = ["ens_member_id"]
        sub_setups_lst: List[InputSetupCollection] = []
        for setup in self._setups:
            for sub_setup in setup.decompress_partially(select_outer, skip):
                sub_setups_lst.append(
                    sub_setup.decompress_partially(select_inner, skip)
                )
        return sub_setups_lst

    def decompress_species_id(self) -> List["InputSetupCollection"]:
        """Decompress species ids depending in whether to combine them."""
        try:
            combine_species: bool = self.collect_equal("combine_species")
        except Exception as e:
            # SR_NOTE This should not happen AFAIK, but in case it does,
            # SR_NOTE catch and raise it here explicitly!
            raise NotImplementedError("sub_setups differing in combine_species", e)
        sub_setups_lst: List["InputSetupCollection"] = (
            self.decompress_partially(["species_id"])
        )
        if combine_species:
            sub_setups_lst = [
                type(self)([setup for setups in sub_setups_lst for setup in setups])
            ]
        return sub_setups_lst

    def collect(self, param: str) -> List[Any]:
        """Collect the values of a parameter for all setups."""
        return [getattr(var_setup, param) for var_setup in self._setups]

    def collect_equal(self, param: str) -> Any:
        """Collect the value of a parameter that is shared by all setups."""
        values = self.collect(param)
        if not all(value == values[0] for value in values[1:]):
            raise Exception("values differ for param", param, values)
        return next(iter(values))

    def group(self, param: str) -> Dict[Any, "InputSetupCollection"]:
        """Group setups by the value of a parameter."""
        grouped_raw: Dict[Any, List[InputSetup]] = {}
        for setup in self:
            try:
                value = getattr(setup, param)
            except AttributeError:
                raise ValueError("invalid input setup parameter", param)
            else:
                if value not in grouped_raw:
                    grouped_raw[value] = []
                grouped_raw[value].append(setup)
        grouped: Dict[Any, "InputSetupCollection"] = {
            value: type(self)(setups) for value, setups in grouped_raw.items()
        }
        return grouped

    def complete_dimensions(
        self,
        meta_data: Mapping[str, Any],
        decompress: bool = False,
        decompress_skip: Optional[Collection[str]] = None,
    ) -> List[str]:
        """Set unconstrained dimensions to all available indices."""
        orig_setups = list(self._setups)
        self._setups.clear()
        completed: List[str] = []
        for setup in orig_setups:
            completed_i: List[str] = setup.complete_dimensions(meta_data)
            if not completed:
                completed = completed_i
            elif completed != completed_i:
                raise Exception("completed dimensions differ", completed, completed_i)
            if not decompress:
                self._setups.append(setup)
            else:
                select = [
                    dim for dim in completed if dim not in (decompress_skip or [])
                ]
                self._setups.extend(setup.decompress_partially(select))
        return completed


# SR_TMP <<< TODO Consider merging with InputSetupCollection (failed due to mypy)
class CoreInputSetupCollection:
    def __init__(self, setups: Collection[CoreInputSetup]) -> None:
        self._setups: List[CoreInputSetup] = list(setups)

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

    @classmethod
    def read_many(
        cls,
        paths: Sequence[str],
        override: Optional[Dict[str, Any]] = None,
        only: Optional[int] = None,
        each_only: Optional[int] = None,
    ) -> InputSetupCollection:
        if only is not None:
            if only < 0:
                raise ValueError("only must not be negative", only)
            each_only = only
        elif each_only is not None:
            if each_only < 0:
                raise ValueError("each_only must not be negative", each_only)
        setup_lst: List[InputSetup] = []
        for path in paths:
            for setup in cls(path).read(override=override, only=each_only):
                if only is not None and len(setup_lst) >= only:
                    break
                if setup not in setup_lst:
                    setup_lst.append(setup)
        return InputSetupCollection(setup_lst)

    def read(
        self, *, override: Optional[Dict[str, Any]] = None, only: Optional[int] = None
    ) -> InputSetupCollection:
        """Read the setup from a text file in TOML format."""
        with open(self.path, "r") as f:
            try:
                raw_data = toml.load(f)
            except Exception as e:
                raise Exception(
                    f"error parsing TOML file {self.path} ({type(e).__name__}: {e})"
                )
        if not raw_data:
            raise ValueError("empty setup file", self.path)
        semi_raw_data = nested_dict_resolve_wildcards(
            raw_data, double_only_to_ends=True,
        )
        params_lst = decompress_nested_dict(
            semi_raw_data, branch_end_criterion=lambda key: not key.startswith("_"),
        )
        if override is not None:
            params_lst, old_params_lst = [], params_lst
            for old_params in old_params_lst:
                params = {**old_params, **override}
                if params not in params_lst:
                    params_lst.append(params)
        setups = InputSetupCollection.create(params_lst)
        if only is not None:
            if only < 0:
                raise ValueError("only must not be negative", only)
            setups = InputSetupCollection(list(setups)[:only])
        return setups

    def write(self, *args, **kwargs) -> None:
        """Write the setup to a text file in TOML format."""
        raise NotImplementedError(f"{type(self).__name__}.write")


def field_get_outer_type(field: ModelField, *, generic: bool = False) -> Type:
    """Obtain the outer type of a pydantic model field."""
    str_ = str(field.outer_type_)
    prefix = "typing."
    if not str_.startswith(prefix):
        raise TypeError(
            "<field>.outer_type_ does not start with '{prefix}'", str_, field,
        )
    str_ = str_[len(prefix) :]
    str_ = str_.split("[")[0]
    try:
        type_ = getattr(typing, str_)
    except AttributeError:
        raise TypeError(
            f"cannot derive type from <field>.outer_type_: typing.{str_} not found",
            field,
        )
    if generic:
        generics = {
            typing.Tuple: tuple,
            typing.List: list,
            typing.Sequence: list,
            typing.Set: set,
            typing.Collection: set,
            typing.Dict: dict,
            typing.Mapping: dict,
        }
        try:
            type_ = generics[type_]
        except KeyError:
            raise NotImplementedError("generic type for tpying type", type_, generics)
    return type_


def setup_repr(obj: Union["CoreInputSetup", "InputSetup"]) -> str:
    def fmt(obj):
        if isinstance(obj, str):
            return f"'{obj}'"
        return str(obj)

    s_attrs = ",\n  ".join(f"{k}={fmt(v)}" for k, v in obj.dict().items())
    return f"{type(obj).__name__}(\n  {s_attrs},\n)"

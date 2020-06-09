# -*- coding: utf-8 -*-
# pylint: disable=C0302  # too-many-lines
"""
Plot setup and setup files.
"""
# Standard library
import dataclasses
import re
from dataclasses import dataclass
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

# Third-party
import toml
from pydantic import BaseModel
from pydantic import root_validator
from pydantic import ValidationError
from pydantic import validator

# First-party
from srutils.dict import compress_multival_dicts
from srutils.dict import decompress_multival_dict
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards
from srutils.str import join_multilines

# Local
from .dimensions import CoreDimensions
from .dimensions import Dimensions
from .exceptions import UnequalInputSetupParamValuesError
from .pydantic import cast_field_value
from .pydantic import prepare_field_value

# Some plot-specific default values
ENS_PROBABILITY_DEFAULT_PARAM_THR = 1e-8
ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN = 10
ENS_CLOUD_TIME_DEFAULT_PARAM_THR = 1e-7
ENS_CLOUD_PROB_DEFAULT_PARAM_TIME_WIN = 12


def is_setup_param(param: str) -> bool:
    return param in CoreInputSetup.__fields__


def is_dimensions_param(param: str) -> bool:
    return param in CoreDimensions.__fields__


# SR_TODO Clean up docstring -- where should format key hints go?
# pylint: disable=E0213  # no-self-argument (validators)
class InputSetup(BaseModel):
    """
    PyFlexPlot setup.

    Args:
        combine_deposition_types: Sum up dry and wet deposition. Otherwise, each
            is plotted separately.

        combine_levels: Sum up over multiple vertical levels. Otherwise, each is
            plotted separately.

        combine_species: Sum up over all specified species. Otherwise, each is
            plotted separately.

        domain: Plot domain. Defaults to 'data', which derives the domain size
            from the input data. Use the format key '{domain}' to embed it in
            ``outfile``. Choices": "auto", "ch".

        ens_member_id: Ensemble member ids. Use the format key '{ens_member}'
            to embed it in ``outfile``. Omit for deterministic simulations.

        ens_param_mem_min: Minimum number of ensemble members used to compute
            some ensemble variables. Its precise meaning depends on the
            variable.

        ens_param_thr: Threshold used to compute some ensemble variables. Its
            precise meaning depends on the variable.

        ens_param_time_win: Tim window used to compute some ensemble variables.
            Its precise meaning depends on the variable.

        ens_variable: Ensemble variable computed from plot variable. Use the
            format key '{ens_variable}' to embed it in ``outfile``.

        infile: Input file path(s). May contain format keys.

        input_variable: Input variable. Choices: "concentration", "deposition".

        integrate: Integrate field over time.

        lang: Language. Use the format key '{lang}' to embed it in ``oufile``.
            Choices: "en", "de".

        multipanel_param: Parameter used to plot multiple panels. Only valid for
            ``plot_type = "multipanel"``. The respective parameter must have one
            value per panel. For example, a four-panel plot with one ensemble
            statistic plot each may be specified with ``multipanel_param =
            "ens_variable"`` and ``ens_variable = ["minimum", "maximum", "meam",
            "median"]``.

        outfile: Output file path. May contain format keys.

        plot_type: Plot type. Use the format key '{plot_type}' to embed it in
            ``outfile``.

        plot_variable: Variable computed from input variable. Use the format key
            '{plot_variable}' to embed it in ``outfile``.

    """

    class Config:  # noqa
        # allow_mutation = False
        arbitrary_types_allowed = True
        extra = "forbid"

    # Basics
    infile: str
    outfile: str
    input_variable: str = "concentration"
    plot_variable: str = "auto"
    ens_variable: Union[str, Tuple[str, ...]] = "none"
    plot_type: str = "auto"
    multipanel_param: Optional[str] = None

    # Tweaks
    integrate: bool = False
    combine_deposition_types: bool = False
    combine_levels: bool = False
    combine_species: bool = False

    # Ensemble-related
    ens_member_id: Optional[Tuple[int, ...]] = None
    ens_param_mem_min: Optional[int] = None
    ens_param_thr: Optional[float] = None
    ens_param_time_win: Optional[float] = None

    # Plot appearance
    lang: str = "en"
    domain: str = "auto"

    # Dimensions
    dimensions: Dimensions = Dimensions()

    @root_validator
    def _check_input_variable(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        value = values["input_variable"]
        choices = ["concentration", "deposition"]
        assert value in choices, value
        return values

    @root_validator
    def _check_plot_variable(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        value = values["plot_variable"]
        choices = ["auto", "affected_area", "affected_area_mono"]
        assert value in choices, value
        return values

    @root_validator
    def _check_ens_variable(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        value = values["ens_variable"]
        choices = [
            "none",
            "probability",
            "minimum",
            "maximum",
            "mean",
            "median",
            "cloud_arrival_time",
            "cloud_departure_time",
            "cloud_occurrence_probability",
        ]
        if isinstance(value, str):
            assert value in choices, value
        else:
            for sub_value in value:
                assert sub_value in choices, sub_value
        return values

    @root_validator
    def _check_plot_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        value = values["plot_type"]
        choices = ["auto", "multipanel"]
        assert value in choices, value
        return values

    @root_validator
    def _check_multipanel_param(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        value = values["multipanel_param"]
        # SR_TODO Consider a generic alternative to the hard-coded list
        multipanel_param_choices = ["ens_variable"]
        if value is None:
            pass
        elif value in multipanel_param_choices:
            if not (
                isinstance(values[value], Sequence)
                and not isinstance(values[value], str)
            ):
                # SR_TMP <  SR_MULTIPANEL
                # raise ValueError(
                #     "multipanel_param parameter must be a sequence",
                #     value,
                #     values[value],
                # )
                # SR_NOTE The exception should be raised when the input is parsed,
                #         but not afterward when the setup objects for the individual
                #         fields are created (which contain only one value each of
                #         the multipanel_param, but still retain plot_type
                #         "multipanel")..
                #         This issue illustrates that InputSetup and Setup should
                #         be separated again in some fashion!
                return values  # SR_TMP
                # SR_TMP >  SR_MULTIPANEL
            # SR_TODO Consider a generic alternative to the hard-coded list
            n_panels_choices = [4]
            n_panels = len(values[value])
            if n_panels not in n_panels_choices:
                raise NotImplementedError(
                    "unexpected number of multipanel_param parameter values",
                    value,
                    n_panels,
                    values[value],
                )
        else:
            raise NotImplementedError(
                f"unknown multipanel_param '{value}'"
                f"; choices: {', '.join(multipanel_param_choices)}"
            )
        return values

    @validator("ens_param_mem_min", always=True)
    def _init_ens_param_mem_min(
        cls, value: Optional[int], values: Dict[str, Any],
    ) -> Optional[int]:
        if value is None:
            if values["ens_variable"] in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                value = ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN
        return value

    @root_validator
    def _init_ens_param_thr(cls, values: Dict[str, Any],) -> Dict[str, Any]:
        value: Optional[float] = values["ens_param_thr"]
        if value is None:
            if values["ens_variable"] == "probability":
                value = ENS_PROBABILITY_DEFAULT_PARAM_THR
            elif values["ens_variable"] in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                value = ENS_CLOUD_TIME_DEFAULT_PARAM_THR
        values["ens_param_thr"] = value
        return values

    @validator("ens_param_time_win", always=True)
    def _init_ens_param_time_win(
        cls, value: Optional[float], values: Dict[str, Any],
    ) -> Optional[float]:
        if value is None:
            if values["ens_variable"] == "cloud_occurrence_probability":
                value = ENS_CLOUD_PROB_DEFAULT_PARAM_TIME_WIN
        return value

    # SR_TMP <<<
    @property
    def deposition_type_str(self) -> str:
        deposition_type = self.dimensions.deposition_type
        if deposition_type is None:
            if self.input_variable == "deposition":
                return "tot"
            return "none"
        elif set(deposition_type) == {"dry", "wet"}:
            return "tot"
        elif deposition_type in ["dry", "wet"]:
            return deposition_type
        else:
            raise NotImplementedError(deposition_type)

    # SR_TMP <<<
    def get_simulation_type(self) -> str:
        if self.ens_member_id:
            return "ensemble"
        return "deterministic"

    @classmethod
    def create(cls, params: Dict[str, Any]) -> "InputSetup":
        """Create an instance of ``InputSetup``.

        Args:
            params: Parameters to instatiate ``InputSetup``.

        """
        params = dict(**params)

        dim_params = params.pop("dimensions", {})
        dimensions = Dimensions.create(dim_params)
        for param, value in dict(**params).items():
            if param in dimensions.params:
                del params[param]
                # dimensions.param = value
                dimensions.update(Dimensions.create({param: value}))  # SR_TMP

        singles = ["infile"]
        for param, value in dict(**params).items():
            if param in singles:
                continue
            field = cls.__fields__[param]
            try:
                params[param] = prepare_field_value(field, value, alias_none=["*"])
            except Exception:
                raise ValueError("invalid parameter value", param, value)

        params["dimensions"] = dimensions
        try:
            return cls(**params)
        except ValidationError as e:
            error = next(iter(e.errors()))
            msg = f"error creating {cls.__name__} object"
            if error["type"] == "value_error.missing":
                param = next(iter(error["loc"]))
                msg += f": missing parameter: {param}"
            raise Exception(msg)

    @classmethod
    def as_setup(cls, obj: Union[Mapping[str, Any], "InputSetup"]) -> "InputSetup":
        if isinstance(obj, cls):
            return obj
        return cls.create(obj)  # type: ignore

    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        """Cast a parameter to the appropriate type."""
        # SR_TMP < TODO move to Dimensions (?)
        if param == "dimensions":
            return {
                dim_param: Dimensions.cast(dim_param, dim_value)
                for dim_param, dim_value in value.items()
            }
        # SR_TMP >
        return cast_field_value(cls, param, value)

    @classmethod
    def cast_many(
        cls, params: Union[Collection[Tuple[str, Any]], Mapping[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(params, Mapping):
            params_dct: Dict[str, Any] = {}
            for param, value in params:
                if param in params_dct:
                    raise ValueError("duplicate parameter", param)
                params_dct[param] = value
            return cls.cast_many(params_dct)
        params_cast = {}
        for param, value in params.items():
            params_cast[param] = cls.cast(param, value)
        return params_cast

    def dict(self, **kwargs) -> Dict[str, Any]:
        # SR_TMP < Catch args that MAY interfere with dimensions
        for arg in ["include", "exclude"]:
            if kwargs.get(arg) is not None:
                raise NotImplementedError(
                    f"{type(self).__name__}.dict with argument '{arg}'", kwargs[arg]
                )
        # SR_TMP >
        return {
            **super().dict(**kwargs),
            "dimensions": self.dimensions.compact_dict(),
        }

    # SR_TODO consider renaming this method (sth. containing 'dimensions')
    # pylint: disable=R0912  # too-many-branches
    # pylint: disable=W0201  # attribute-defined-outside-init (dimensions.*)
    def complete_dimensions(self, meta_data: Mapping[str, Any]) -> "InputSetup":
        """Complete unconstrained dimensions based on available indices."""
        dimensions = meta_data["dimensions"]
        obj = self.copy()

        if obj.dimensions.time is None:
            obj.dimensions.time = tuple(range(dimensions["time"]["size"]))

        # SR_TMP < does this belong here? and what does it do? TODO explain!
        nts = dimensions["time"]["size"]
        time_new = []
        if isinstance(obj.dimensions.time, Sequence):
            times = obj.dimensions.time
        else:
            times = [obj.dimensions.time]
        for its in times:
            if its < 0:
                its += nts
                assert 0 <= its < nts
            time_new.append(its)
        obj.dimensions.time = tuple(time_new)
        # SR_TMP >

        if obj.dimensions.level is None:
            if obj.input_variable == "concentration":
                if "level" in dimensions:
                    obj.dimensions.level = tuple(range(dimensions["level"]["size"]))

        if obj.dimensions.deposition_type is None:
            if obj.input_variable == "deposition":
                obj.dimensions.deposition_type = ("dry", "wet")

        if obj.dimensions.species_id is None:
            obj.dimensions.species_id = meta_data["analysis"]["species_ids"]

        if obj.dimensions.nageclass is None:
            if "nageclass" in dimensions:
                obj.dimensions.nageclass = tuple(range(dimensions["nageclass"]["size"]))

        if obj.dimensions.noutrel is None:
            if "noutrel" in dimensions:
                obj.dimensions.noutrel = tuple(range(dimensions["noutrel"]["size"]))

        if obj.dimensions.numpoint is None:
            if "numpoint" in dimensions:
                obj.dimensions.numpoint = tuple(range(dimensions["numpoint"]["size"]))

        return obj

    def __repr__(self) -> str:  # type: ignore
        return setup_repr(self)

    def copy(self):
        return self.create(self.dict())

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
        dct = {**self.dict(), **params}
        if "dimensions" in params:
            dct["dimensions"] = self.dimensions.derive(params["dimensions"])
        return type(self).create(dct)

    @classmethod
    def compress(cls, setups: "InputSetupCollection") -> "InputSetup":
        # SR_TMP <
        try:
            setups.collect_equal("input_variable")
        except UnequalInputSetupParamValuesError:
            raise ValueError("cannot compress setups: input_variable differs") from None
        # SR_TMP >
        dct = compress_multival_dicts(setups.dicts(), cls_seq=tuple)
        if isinstance(dct["dimensions"], Sequence):
            dct["dimensions"] = compress_multival_dicts(
                dct["dimensions"], cls_seq=tuple
            )
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

    @overload
    def decompress(self, skip: None = None) -> "CoreInputSetupCollection":
        ...

    @overload
    def decompress(self, skip: List[str]) -> "InputSetupCollection":
        ...

    def decompress(self, skip=None):
        return self._decompress(select=None, skip=skip)

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

    # pylint: disable=R0914  # too-many-locals
    def _decompress(self, select=None, skip=None, cls_setup=None):
        """Create multiple ``InputSetup`` objects with one-value parameters only."""

        def get_cls_setup(cls_setup: Type, select: Optional, skip: Optional) -> Type:
            if cls_setup is None:
                if (select, skip) == (None, None):
                    return CoreInputSetup
                else:
                    return InputSetup
            return cls_setup

        def get_cls_setup_collection(cls_setup: Type) -> Type:
            if cls_setup is CoreInputSetup:
                return CoreInputSetupCollection
            elif cls_setup is InputSetup:
                return InputSetupCollection
            else:
                raise ValueError("invalid cls_setup", cls_setup)

        cls_setup = get_cls_setup(cls_setup, select, skip)
        cls_setup_collection = get_cls_setup_collection(cls_setup)

        select_setup, select_dimensions = self._group_params(select)
        skip_setup, skip_dimensions = self._group_params(skip)

        dct = self.dict()

        # SR_TMP < TODO Can this be moved to class Dimensions?!?
        # Handle deposition type
        if self.deposition_type_str == "tot":
            if select is None or "dimensions.deposition_type" in select:
                if "dimensions.deposition_type" not in (skip or []):
                    dct["dimensions"]["deposition_type"] = ("dry", "wet")
        # SR_TMP >

        # Decompress dict
        dcts = []
        for dct_i in decompress_multival_dict(
            dct, select=select_setup, skip=skip_setup
        ):
            if "dimensions" not in dct_i:
                dcts.append(dct_i)
            else:
                for dims_j in decompress_multival_dict(
                    dct_i["dimensions"], select=select_dimensions, skip=skip_dimensions
                ):
                    dct_ij = {**dct_i, "dimensions": dims_j}
                    dcts.append(dct_ij)

        return cls_setup_collection([cls_setup.create(dct) for dct in dcts])

    def _group_params(
        self, params: Optional[Collection[str]]
    ) -> Tuple[Optional[List[str]], ...]:
        if params is None:
            return (None, None)
        params_setup: List[str] = []
        params_dimensions: List[str] = []
        for param in params:
            if is_setup_param(param):
                params_setup.append(param)
                continue
            if param.startswith("dimensions."):
                dims_param = param.split(".", 1)[-1]
                if is_dimensions_param(dims_param):
                    params_dimensions.append(dims_param)
                    continue
            raise ValueError("invalid param", param)
        return (params_setup, params_dimensions)


class CoreInputSetup(BaseModel):
    """
    PyFlexPlot core setup with exactly one value per parameter.

    See ``InputSetup`` for details on the parameters.

    """

    class Config:  # noqa
        arbitrary_types_allowed = True
        allow_mutation = False
        extra = "forbid"

    # Basics
    infile: str
    outfile: str
    input_variable: str = "concentration"
    plot_variable: str = "auto"
    ens_variable: str = "none"
    plot_type: str = "auto"
    multipanel_param: Optional[str] = None

    # Tweaks
    integrate: bool = False
    combine_deposition_types: bool = False
    combine_levels: bool = False
    combine_species: bool = False

    # Ensemble-related
    ens_member_id: Optional[int] = None
    ens_param_mem_min: Optional[int] = None
    ens_param_thr: Optional[float] = None
    ens_param_time_win: Optional[float] = None

    # Plot appearance
    lang: str = "en"
    domain: str = "auto"

    # Dimensions
    # dimensions: CoreDimensions = CoreDimensions()
    dimensions: Dimensions = Dimensions()  # SR_TMP

    # SR_TMP <<<
    @property
    def deposition_type_str(self) -> str:
        if self.dimensions.deposition_type is None:
            return "none"
        else:
            return self.dimensions.deposition_type

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "CoreInputSetup":
        dimensions = Dimensions.create(params.get("dimensions", {}))
        return cls(**{**params, "dimensions": dimensions})

    def __repr__(self) -> str:  # type: ignore
        return setup_repr(self)

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
            raise ValueError(
                "setups is not an InputSetup collection",
                type(setups),
                type(next(iter(setups))),
            )
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

        same: Dict[str, Any] = {}
        diff: Dict[str, Any] = {}

        # Regular params
        for param in InputSetup.__fields__:
            if param == "dimensions":
                continue  # Handled below
            try:
                value = self.collect_equal(param)
            except UnequalInputSetupParamValuesError:
                diff[param] = self.collect(param)
            else:
                same[param] = value

        # Dimensions
        dims_same = {}
        dims_diff = {}
        for param in CoreDimensions.__fields__:
            values = []
            for dims in self.collect("dimensions"):
                values.append(dims.get_compact(param))
            if len(set(values)) == 1:
                dims_same[param] = next(iter(values))
            else:
                dims_diff[param] = values
        if dims_same:
            same["dimensions"] = dims_same
        if dims_diff:
            diff["dimensions"] = dims_diff

        def format_params(params: Dict[str, Any], name: str) -> str:
            lines = []
            for param, value in params.items():
                if param == "dimensions":
                    s_param = format_params(value, "dimensions")
                else:
                    if isinstance(value, str):
                        s_value = f"'{value}'"
                    elif isinstance(value, Sequence):
                        s_value = ", ".join(
                            [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                        )
                    else:
                        s_value = str(value)
                    s_param = f"{param}: {s_value}"
                lines.append(s_param)
            if not lines:
                return f"{name}: --"
            else:
                body = join_multilines(lines, indent=2)
                return f"{name}:\n{body}"

        lines = [
            f"n: {len(self)}",
            format_params(same, "same"),
            format_params(diff, "diff"),
        ]
        body = join_multilines(lines, indent=2)

        return "\n".join([f"{type(self).__name__}[", body, "]"])

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
        return [setup.dict() for setup in self]

    @classmethod
    def merge(
        cls, setups_lst: Sequence["InputSetupCollection"]
    ) -> "InputSetupCollection":
        return cls([setup for setups in setups_lst for setup in setups])

    def compress(self) -> InputSetup:
        return InputSetup.compress(self)

    def decompress(self) -> List["CoreInputSetupCollection"]:
        return self.decompress_partially(select=None, skip=None)

    def derive(self, params: Mapping[str, Any]) -> "InputSetupCollection":
        return type(self)([setup.derive(params) for setup in self])

    @overload
    def decompress_partially(
        self, select: None, skip: None = None
    ) -> List["CoreInputSetupCollection"]:
        ...

    @overload
    def decompress_partially(
        self, select: None, skip: Collection[str]
    ) -> List["InputSetupCollection"]:
        ...

    @overload
    def decompress_partially(
        self, select: Collection[str], skip: Optional[Collection[str]] = None
    ) -> List["InputSetupCollection"]:
        ...

    def decompress_partially(self, select, skip=None):
        if (select, skip) == (None, None):
            return [CoreInputSetupCollection(setup.decompress()) for setup in self]
        sub_setup_lst_lst: List[List[InputSetup]] = []
        for setup in self:
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

    @overload
    def decompress_twice(
        self, outer, skip: None = None,
    ) -> List["CoreInputSetupCollection"]:
        ...

    @overload
    def decompress_twice(
        self, outer, skip: Collection[str]
    ) -> List["InputSetupCollection"]:
        ...

    def decompress_twice(self, outer: str, skip=None):
        sub_setups_lst: List[InputSetupCollection] = []
        for setup in self:
            for sub_setup in setup.decompress_partially([outer], skip):
                sub_sub_setups = sub_setup.decompress(skip)
                sub_setups_lst.append(sub_sub_setups)
        return sub_setups_lst

    def collect(self, param: str) -> List[Any]:
        """Collect the values of a parameter for all setups."""
        if is_setup_param(param):
            return [getattr(var_setup, param) for var_setup in self]
        if param.startswith("dimensions."):
            dims_param = param.split(".", 1)[-1]
            if is_dimensions_param(dims_param):
                values: Set[Any] = set()
                for dimensions in self.collect("dimensions"):
                    value = dimensions.get_compact(dims_param)
                    try:
                        values |= set(value)
                    except TypeError:
                        values.add(value)
                return sorted(values)
        raise ValueError("invalid param", param)

    def collect_equal(self, param: str) -> Any:
        """Collect the value of a parameter that is shared by all setups."""
        values = self.collect(param)
        if not all(value == values[0] for value in values[1:]):
            raise UnequalInputSetupParamValuesError(param, values)
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
        self, meta_data: Mapping[str, Any]
    ) -> "InputSetupCollection":
        """Complete unconstrained dimensions based on available indices."""
        setup_lst = []
        for setup in self:
            setup_lst.append(setup.complete_dimensions(meta_data))
        return type(self)(setup_lst)


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

    def __iter__(self) -> Iterator[CoreInputSetup]:
        return iter(self._setups)

    # SR_TMP < TODO clean this up!!!
    __repr__ = InputSetupCollection.__repr__
    __len__ = InputSetupCollection.__len__
    __eq__ = InputSetupCollection.__eq__
    dicts = InputSetupCollection.dicts
    collect = InputSetupCollection.collect
    collect_equal = InputSetupCollection.collect_equal
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


def setup_repr(obj: Union["CoreInputSetup", "InputSetup"]) -> str:
    def fmt(obj):
        if isinstance(obj, str):
            return f"'{obj}'"
        return str(obj)

    s_attrs = ",\n  ".join(f"{k}={fmt(v)}" for k, v in obj.dict().items())
    return f"{type(obj).__name__}(\n  {s_attrs},\n)"


@dataclass
class FilePathFormatter:
    def __init__(self, previous: Optional[List[str]] = None) -> None:
        self.previous: List[str] = previous if previous is not None else []
        self._setup: Optional[InputSetup] = None

    # pylint: disable=W0102  # dangerous-default-value ([])
    def format(self, setup: InputSetup) -> str:
        self._setup = setup
        assert self._setup is not None  # mypy
        template = setup.outfile
        path = self._format_template(template)
        while path in self.previous:
            template = self.derive_unique_path(self._format_template(template))
        self.previous.append(path)
        self._setup = None
        return path

    def _format_template(self, template: str) -> str:
        assert self._setup is not None  # mypy
        input_variable = self._setup.input_variable
        if self._setup.input_variable == "deposition":
            input_variable += f"_{self._setup.deposition_type_str}"
        kwargs = {
            "nageclass": self._setup.dimensions.nageclass,
            "domain": self._setup.domain,
            "lang": self._setup.lang,
            "level": self._setup.dimensions.level,
            "noutrel": self._setup.dimensions.noutrel,
            "species_id": self._setup.dimensions.species_id,
            "time": self._setup.dimensions.time,
            "input_variable": input_variable,
            "plot_variable": self._setup.plot_variable,
            "plot_type": self._setup.plot_type,
            "ens_variable": self._setup.ens_variable,
        }
        # Format the file path
        # Don't use str.format in order to handle multival elements
        path = self._replace_format_keys(template, kwargs)
        return path

    def _replace_format_keys(self, path: str, kwargs: Mapping[str, Any]) -> str:
        for key, val in kwargs.items():
            if not (isinstance(val, Sequence) and not isinstance(val, str)):
                val = [val]
            # Iterate over relevant format keys
            rxs = r"{" + key + r"(:[^}]*)?}"
            re.finditer(rxs, path)
            for m in re.finditer(rxs, path):

                # Obtain format specifier (if there is one)
                try:
                    f = m.group(1) or ""
                except IndexError:
                    f = ""

                # Format the string that replaces this format key in the path
                formatted_key = "+".join([f"{{{f}}}".format(v) for v in val])

                # Replace format key in the path by the just formatted string
                start, end = path[: m.span()[0]], path[m.span()[1] :]
                path = f"{start}{formatted_key}{end}"

        # Check that all keys have been formatted
        if "{" in path or "}" in path:
            raise Exception(
                "formatted output file path still contains format keys", path,
            )

        return path

    @staticmethod
    def derive_unique_path(path: str) -> str:
        """Add/increment a trailing number to a file path."""

        # Extract suffix
        if path.endswith(".png"):
            suffix = ".png"
        else:
            raise NotImplementedError(f"unknown suffix: {path}")
        path_base = path[: -len(suffix)]

        # Reuse existing numbering if present
        match = re.search(r"-(?P<i>[0-9]+)$", path_base)
        if match:
            i = int(match.group("i")) + 1
            w = len(match.group("i"))
            path_base = path_base[: -w - 1]
        else:
            i = 1
            w = 1

        # Add numbering and suffix
        path = path_base + f"-{{i:0{w}}}{suffix}".format(i=i)

        return path

# pylint: disable=C0302  # too-many-lines (>1000)
"""Plot setup and setup files.

The setup parameters that are exposed in the setup files are all described in
the docstring of the class method ``Setup.create``.

"""
# Standard library
from dataclasses import asdict
from typing import Any
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import toml
from pydantic import BaseModel
from pydantic import root_validator
from pydantic import ValidationError
from pydantic import validator
from typing_extensions import Literal

# First-party
from srutils.dict import compress_multival_dicts
from srutils.dict import decompress_multival_dict
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards
from srutils.str import join_multilines

# Local
from .dimensions import CoreDimensions
from .dimensions import Dimensions
from .utils.exceptions import UnequalSetupParamValuesError
from .utils.pydantic import cast_field_value
from .utils.pydantic import InvalidParameterNameError
from .utils.pydantic import prepare_field_value

# Some plot-specific default values
ENS_PROBABILITY_DEFAULT_PARAM_THR = 0.0
ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN = 1
ENS_CLOUD_TIME_DEFAULT_PARAM_THR = 0.0


# SR_TMP <<< TODO cleaner solution
def is_dimensions_param(param: str) -> bool:
    return param in CoreDimensions.__fields__


# SR_TMP <<< TODO cleaner solution
def is_core_setup_param(param: str) -> bool:
    return param in CoreSetup.__fields__


# SR_TMP <<< TODO cleaner solution
def is_setup_param(param: str) -> bool:
    return param in Setup.__fields__


# SR_TMP <<< TODO cleaner solution
def get_setup_param_value(setup: "Setup", param: str) -> Any:
    if is_setup_param(param):
        return getattr(setup, param)
    elif is_core_setup_param(param):
        return getattr(setup.core, param)
    elif param.startswith("dimensions."):
        dim_param = param.split(".", 1)[-1]
        if is_dimensions_param(dim_param):
            return getattr(setup.core.dimensions, dim_param)
    raise ValueError("invalid input setup parameter", param)


# pylint: disable=R0201  # no-self-use (root validators)
# pylint: disable=E0213  # no-self-argument (validators)
class CoreSetup(BaseModel):
    """PyFlexPlot core setup with exactly one value per parameter.

    See docstring of ``Setup.create`` for a description of the parameters.

    """

    class Config:  # noqa
        arbitrary_types_allowed = True
        allow_mutation = False
        extra = "forbid"

    # Basics
    input_variable: str = "concentration"
    ens_variable: str = "none"
    plot_type: str = "auto"
    multipanel_param: Optional[str] = None

    # Tweaks
    integrate: bool = False
    combine_deposition_types: bool = False
    combine_levels: bool = False
    combine_species: bool = False

    # Ensemble-related
    ens_param_mem_min: Optional[int] = None
    ens_param_pctl: Optional[float] = None
    ens_param_thr: Optional[float] = None

    # Plot appearance
    lang: str = "en"
    domain: str = "full"
    domain_size_lat: Optional[float]
    domain_size_lon: Optional[float]

    # Dimensions
    dimensions_default: str = "all"
    dimensions: Dimensions = Dimensions()

    # SR_TMP <<<
    @property
    def deposition_type_str(self) -> str:
        if self.dimensions.deposition_type is None:
            return "none"
        else:
            return self.dimensions.deposition_type

    @root_validator
    def _check_input_variable(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        value = values["input_variable"]
        choices = [
            "concentration",
            "deposition",
            "affected_area",
        ]
        assert value in choices, value
        return values

    @root_validator
    def _check_ens_variable(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        value = values["ens_variable"]
        choices = [
            "ens_cloud_arrival_time",
            "ens_cloud_departure_time",
            "maximum",
            "mean",
            "median",
            "minimum",
            "none",
            "percentile",
            "probability",
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
                #         This issue illustrates that Setup and Setup should
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
        cls, value: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        if value is None:
            if values["ens_variable"] in [
                "ens_cloud_arrival_time",
                "ens_cloud_departure_time",
            ]:
                value = ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN
        return value

    @validator("ens_param_pctl", always=True)
    def _init_ens_param_pctl(
        cls, value: Optional[float], values: Dict[str, Any]
    ) -> Optional[float]:
        if values["ens_variable"] == "percentile":
            assert value is not None
        return value

    @validator("ens_param_thr", always=True)
    def _init_ens_param_thr(
        cls, value: Optional[float], values: Dict[str, Any]
    ) -> Optional[float]:
        if value is None:
            if values["ens_variable"] in [
                "ens_cloud_arrival_time",
                "ens_cloud_departure_time",
            ]:
                value = ENS_CLOUD_TIME_DEFAULT_PARAM_THR
            elif values["ens_variable"] == "probability":
                value = ENS_PROBABILITY_DEFAULT_PARAM_THR
        return value

    @root_validator()
    def _init_domain_size_lat_lon(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        lat = values["domain_size_lat"]
        lon = values["domain_size_lon"]
        if lat is None and lon is None:
            if values["domain"] in ["release_site", "cloud"]:
                # Lat size derived from lon size and map aspect ratio
                values["domain_size_lon"] = 20.0
        return values

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "CoreSetup":
        params = dict(params)
        dimensions = Dimensions.create(params.pop("dimensions", {}))
        return cls(**{**params, "dimensions": dimensions})

    def dict(self) -> Dict[str, Any]:  # type: ignore
        return {
            **super().dict(),
            "dimensions": self.dimensions.dict(),
        }

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple({**self.dict(), "dimensions": self.dimensions.tuple()}.items())

    def __hash__(self) -> int:
        return hash(self.tuple())

    def __repr__(self) -> str:  # type: ignore
        return setup_repr(self)

    @classmethod
    def as_setup(cls, obj: Union[Mapping[str, Any], "CoreSetup"]) -> "CoreSetup":
        if isinstance(obj, cls):
            return obj
        assert isinstance(obj, Mapping)  # mypy
        return cls(**obj)

    @overload
    def complete_dimensions(
        self, meta_data: Mapping[str, Any], *, inplace: Literal[False] = ...
    ) -> "CoreSetup":
        ...

    @overload
    def complete_dimensions(
        self, meta_data: Mapping[str, Any], *, inplace: Literal[True]
    ) -> None:
        ...

    def complete_dimensions(self, meta_data, *, inplace=False):
        """Complete unconstrained dimensions based on available indices."""
        obj: "CoreSetup" = self if inplace else self.copy()
        obj.dimensions.complete(
            meta_data, self.input_variable, mode=obj.dimensions_default, inplace=True
        )
        return None if inplace else obj


# SR_TODO Clean up docstring -- where should format key hints go?
class Setup(BaseModel):
    """PyFlexPlot setup.

    See docstring of ``Setup.create`` for details on parameters.

    """

    class Config:  # noqa
        # allow_mutation = False
        arbitrary_types_allowed = True
        extra = "forbid"

    infile: str  # = "none"
    outfile: Union[str, Tuple[str, ...]]  # = "none"
    model: str  # = "none"
    outfile_time_format: str = "%Y%m%d%H%M"
    base_time: Optional[int] = None
    ens_member_id: Optional[Tuple[int, ...]] = None
    scale_fact: float = 1.0
    core: "CoreSetup" = CoreSetup()

    # SR_TMP <<<
    @property
    def deposition_type_str(self) -> str:
        deposition_type = self.core.dimensions.deposition_type
        if deposition_type is None:
            if self.core.input_variable == "deposition":
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
    def create(cls, params: Mapping[str, Any]) -> "Setup":
        """Create an instance of ``Setup``.

        Args:
            params: Parameters to create an instance of ``Setup``, including
                parameters to create the ``CoreSetup`` instance ``Setup.core``
                and the ``Dimensions`` instance ``Setup.core.dimensions``. See
                below for a description of each parameter.

        The parameter descriptions are (for now) collected here because they are
        all directly exposed in the setup files without reflecting the internal
        composition hierarchy ``Setup -> CoreSetup -> Dimensions``. This
        docstring thus serves as a single point of reference for all parameters.
        The docstrings of the individual classes, were the parameters should be
        described as arguments, refer to this docstring to avoid duplication.

        Params:
            base_time: Start of the model simulation on which the dispersion
                simulation is based.

            combine_deposition_types: Sum up dry and wet deposition. Otherwise,
                each is plotted separately.

            combine_levels: Sum up over multiple vertical levels. Otherwise,
                each is plotted separately.

            combine_species: Sum up over all specified species. Otherwise, each
                is plotted separately.

            deposition_type: Type(s) of deposition. Part of the plot variable
                name that may be embedded in ``outfile`` with the format key
                '{variable}'. Choices: "none", "dry", "wet" (the latter may can
                be combined).

            dimensions_default: How to complete unspecified dimensions based
                on the values available in the input file. Choices: 'all',
                'first.

            domain: Plot domain. Defaults to 'data', which derives the domain
                size from the input data. Use the format key '{domain}' to embed
                it in ``outfile``. Choices": "auto", "full", "data",
                "release_site", "ch".

            domain_size_lat: Latitudinal extent of domain in degrees. Defaults
                depend on ``domain``.

            domain_size_lon: Longitudinal extent of domain in degrees. Defaults
                depend on ``domain``.

            ens_member_id: Ensemble member ids. Use the format key
                '{ens_member}' to embed it in ``outfile``. Omit for
                deterministic simulations.

            ens_param_mem_min: Minimum number of ensemble members used to
                compute some ensemble variables. Its precise meaning depends on
                the variable.

            ens_param_pctl: Percentile for ``ens_variable = 'percentile'``.

            ens_param_thr: Threshold used to compute some ensemble variables.
                Its precise meaning depends on the variable.

            ens_variable: Ensemble variable computed from plot variable. Use the
                format key '{ens_variable}' to embed it in ``outfile``.

            infile: Input file path(s). May contain format keys.

            input_variable: Input variable. Choices: "concentration",
                "deposition".

            integrate: Integrate field over time.

            lang: Language. Use the format key '{lang}' to embed it in
                ``outfile``. Choices: "en", "de".

            level: Index/indices of vertical level (zero-based, bottom-up). To
                sum up multiple levels, combine their indices with '+'. Use the
                format key '{level}' to embed it in ``outfile``.

            multipanel_param: Parameter used to plot multiple panels. Only valid
                for ``plot_type = "multipanel"``. The respective parameter must
                have one value per panel. For example, a four-panel plot with
                one ensemble statistic plot each may be specified with
                ``multipanel_param = "ens_variable"`` and ``ens_variable =
                ["minimum", "maximum", "meam", "median"]``.

            nageclass: Index of age class (zero-based). Use the format key
                '{nageclass}' to embed it in ``outfile``.

            noutrel: Index of noutrel (zero-based). Use the format key
                '{noutrel}' to embed it in ``outfile``.

            numpoint: Index of release point (zero-based).

            outfile: Output file path(s). May contain format keys.

            outfile_time_format: Format specification (e.g., '%Y%m%d%H%M') for
                time steps (``time_step``, ``base_time``) embedded in
                ``outfile``.

            plot_type: Plot type. Use the format key '{plot_type}' to embed it
                in ``outfile``.

            species_id: Species id(s). To sum up multiple species, combine their
                ids with '+'. Use the format key '{species_id}' to embed it in
                ``outfile``.

            time: Time step indices (zero-based). Use the format key '{time}'
                to embed one in ``outfile``.

        """
        params = dict(**params)
        dim_params = params.pop("dimensions", {})
        dimensions = Dimensions.create(dim_params)
        core_params = {"dimensions": dimensions}
        singles = ["infile"]
        for param, value in dict(**params).items():
            if param in CoreSetup.__fields__:
                core_params[param] = params.pop(param)
                continue
            if param in singles:
                continue
            field = cls.__fields__[param]
            try:
                params[param] = prepare_field_value(field, value, alias_none=["*"])
            except Exception as e:
                raise ValueError("invalid parameter value", param, value) from e
        params["core"] = CoreSetup.create(core_params)
        try:
            return cls(**params)
        except ValidationError as e:
            error = next(iter(e.errors()))
            msg = f"error creating {cls.__name__} object"
            if error["type"] == "value_error.missing":
                param = next(iter(error["loc"]))
                msg += f": missing parameter: {param}"
            raise Exception(msg) from e

    @classmethod
    def as_setup(cls, obj: Union[Mapping[str, Any], "Setup"]) -> "Setup":
        if isinstance(obj, cls):
            return obj
        return cls.create(obj)  # type: ignore

    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        """Cast a parameter to the appropriate type."""
        param_choices = sorted(
            [param for param in cls.__fields__ if param != "core"]
            + [param for param in CoreSetup.__fields__ if param != "dimensions"]
            + list(CoreDimensions.__fields__)
        )
        param_choices_fmtd = ", ".join(map(str, param_choices))
        if param == "dimensions":
            result: Dict[str, Any] = {}
            for dim_param, dim_value in value.items():
                try:
                    result[dim_param] = Dimensions.cast(
                        dim_param, dim_value, many_ok=True
                    )
                except InvalidParameterNameError as e:
                    raise InvalidParameterNameError(
                        f"{dim_param} ({type(dim_value).__name__}: {dim_value})"
                        f"; choices: {param_choices_fmtd}"
                    ) from e
            return result
        try:
            if is_dimensions_param(param):
                return Dimensions.cast(param, value, many_ok=True)
            elif is_core_setup_param(param):
                return cast_field_value(CoreSetup, param, value)
            return cast_field_value(cls, param, value)
        except InvalidParameterNameError as e:
            raise InvalidParameterNameError(
                f"{param} ({type(value).__name__}: {value})"
                f"; choices: {param_choices_fmtd}"
            ) from e

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

    def __repr__(self) -> str:  # type: ignore
        return setup_repr(self)

    # SR_TMP <<< TODO Don't merge core params!
    def dict(self, **kwargs) -> Dict[str, Any]:
        # SR_TMP < Catch args that MAY interfere with dimensions
        for arg in ["include", "exclude"]:
            if kwargs.get(arg) is not None:
                raise NotImplementedError(
                    f"{type(self).__name__}.dict with argument '{arg}'", kwargs[arg]
                )
        # SR_TMP >
        return {
            **{
                param: getattr(self, param)
                for param in self.__fields__
                if param != "core"
            },
            **self.core.dict(),
        }

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        # + return tuple({**self.dict(), "core": self.core.tuple()}.items())
        return tuple(
            {
                **self.dict(),
                "dimensions": self.core.dimensions.tuple(),
                "core": self.core.tuple(),
            }.items()
        )

    def __hash__(self) -> int:
        return hash(self.tuple())

    def copy(self):
        return self.create(self.dict())

    def __len__(self) -> int:
        return len(dict(self))

    def __eq__(self, other: Any) -> bool:
        try:
            other_dict = other.dict()
        except AttributeError:
            try:
                other_dict = dict(other)  # type: ignore
            except TypeError:
                try:
                    other_dict = asdict(other)
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
        self, params: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ) -> Union["Setup", "SetupCollection"]:
        """Derive ``Setup`` object(s) with adapted parameters."""
        if isinstance(params, Sequence):
            return SetupCollection([self.derive(params_i) for params_i in params])
        dct = {**self.dict(), **params}
        if "dimensions" in params:
            dct["dimensions"] = self.core.dimensions.derive(params["dimensions"])
        return type(self).create(dct)

    @classmethod
    def compress(cls, setups: Union["SetupCollection", Sequence["Setup"]]) -> "Setup":
        setups = list(setups)
        # SR_TMP <
        input_variables = [setup.core.input_variable for setup in setups]
        if len(set(input_variables)) != 1:
            raise ValueError(
                "cannot compress setups: input_variable differs", input_variables
            )
        # SR_TMP >
        dcts = [setup.dict() for setup in setups]
        dct = compress_multival_dicts(dcts, cls_seq=tuple)
        if isinstance(dct["dimensions"], Sequence):
            dct["dimensions"] = compress_multival_dicts(
                dct["dimensions"], cls_seq=tuple
            )
        return cls.create(dct)

    @classmethod
    def compress_partially(
        cls, setups: "SetupCollection", skip: List[str]
    ) -> "SetupCollection":
        dcts: List[Dict[str, Any]] = setups.dicts()
        preserved_params_lst: List[Dict[str, Any]] = []
        for dct in dcts:
            preserved_params = {}
            for param in skip:
                try:
                    preserved_params[param] = dct.pop(param)
                except ValueError as e:
                    raise ValueError("invalid param", param) from e
            if preserved_params not in preserved_params_lst:
                preserved_params_lst.append(preserved_params)
        partial_dct = compress_multival_dicts(setups.dicts(), cls_seq=tuple)
        setup_lst: List["Setup"] = []
        for preserved_params in preserved_params_lst:
            dct = {**partial_dct, **preserved_params}
            setup_lst.append(cls.create(dct))
        return SetupCollection(setup_lst)

    def decompress(self, skip: Optional[Collection[str]] = None) -> "SetupCollection":
        return self._decompress(select=None, skip=skip)

    def decompress_partially(
        self, select: Optional[Collection[str]], skip: Optional[Collection[str]] = None
    ) -> "SetupCollection":
        return self._decompress(select, skip)

    # pylint: disable=R0914  # too-many-locals
    def _decompress(
        self,
        select: Optional[Collection[str]] = None,
        skip: Optional[Collection[str]] = None,
    ):
        """Create multiple ``Setup`` objects with one-value parameters only."""
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

        if (
            (select and "input_variable" in select)
            or (not skip or "input_variable" not in skip)
            and dct["input_variable"] == "affected_area"
        ):
            dct["input_variable"] = ("concentration", "deposition")

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
                    # SR_TMP <
                    # Handle affected area
                    if (
                        dct_ij["input_variable"] == "concentration"
                        and dct_ij["dimensions"]["deposition_type"]
                    ):
                        dct_ij["dimensions"] = {
                            **dct_ij["dimensions"],
                            "deposition_type": None,
                        }
                        if dct_ij in dcts:
                            continue
                    # SR_TMP >
                    dcts.append(dct_ij)

        return SetupCollection([Setup.create(dct) for dct in dcts])

    def _group_params(
        self, params: Optional[Collection[str]]
    ) -> Tuple[Optional[List[str]], ...]:
        if params is None:
            return (None, None)
        params_setup: List[str] = []
        params_dimensions: List[str] = []
        for param in params:
            if is_setup_param(param) or is_core_setup_param(param):
                params_setup.append(param)
                continue
            if param.startswith("dimensions."):
                dims_param = param.split(".", 1)[-1]
                if is_dimensions_param(dims_param):
                    params_dimensions.append(dims_param)
                    continue
            raise ValueError("invalid param", param)
        return (params_setup, params_dimensions)


class SetupCollection:
    """A collection of ``Setup`` objects."""

    def __init__(self, setups: Collection[Setup]) -> None:
        """Create an instance of ``SetupCollection``."""
        if not isinstance(setups, Collection) or (
            setups and not isinstance(next(iter(setups)), Setup)
        ):
            raise ValueError(
                "setups is not an collection of Setup objects, but a"
                f" {type(setups).__name__} of {type(next(iter(setups))).__name__}"
            )
        self._setups: List[Setup] = list(setups)

    @classmethod
    def create(
        cls, setups: Collection[Union[Mapping[str, Any], Setup]]
    ) -> "SetupCollection":
        setup_lst: List[Setup] = []
        for obj in setups:
            if isinstance(obj, Setup):
                obj = obj.dict()
            # SR_TMP <
            # dcts = decompress_multival_dict(cast(dict, obj))
            skip = ["dimensions"]
            # SR_TMP <
            skip.append("ens_member_id")
            # SR_TMP >
            dcts = decompress_multival_dict(cast(dict, obj), skip=skip)
            # SR_TMP >
            for dct in dcts:
                setup = Setup.create(dct)
                setup_lst.append(setup)
        return cls(setup_lst)

    @classmethod
    def from_raw_params(
        cls, raw_params_lst: Sequence[Mapping[str, Any]]
    ) -> "SetupCollection":
        params_lst = []
        for raw_params in raw_params_lst:
            params = {}
            for param, value in raw_params.items():
                if not is_dimensions_param(param):
                    params[param] = value
                else:
                    if "dimensions" not in params:
                        params["dimensions"] = {}
                    params["dimensions"][param] = value
            params_lst.append(params)
        return cls.create(params_lst)

    def copy(self) -> "SetupCollection":
        return type(self)([setup.copy() for setup in self])

    # pylint: disable=R0912  # too-many-branches
    # pylint: disable=R0915  # too-many-statements
    def __repr__(self) -> str:

        same: Dict[str, Any] = {}
        diff: Dict[str, Any] = {}

        # Regular params
        for param in Setup.__fields__:
            if param == "core":
                continue  # Handled below
            try:
                value = self.collect_equal(param)
            except UnequalSetupParamValuesError:
                diff[param] = self.collect(param)
            else:
                same[param] = value

        # Core params
        core_same = {}
        core_diff = {}
        for param in CoreSetup.__fields__:
            if param == "dimensions":
                continue  # Handled below
            try:
                value = self.collect_equal(param)
            except UnequalSetupParamValuesError:
                core_diff[param] = self.collect(param)
            else:
                core_same[param] = value
        if core_same:
            same["core"] = core_same
        if core_diff:
            diff["core"] = core_diff

        # Dimensions
        dims_same = {}
        dims_diff = {}
        for param in CoreDimensions.__fields__:
            values = []
            for dims in self.collect("dimensions"):
                values.append(dims.get(param))
            if len(set(values)) == 1:
                dims_same[param] = next(iter(values))
            else:
                dims_diff[param] = values
        if dims_same:
            if "core" not in same:
                same["core"] = {}
            same["core"]["dimensions"] = dims_same
        if dims_diff:
            if "core" not in diff:
                diff["core"] = {}
            diff["core"]["dimensions"] = dims_diff

        def format_params(params: Dict[str, Any], name: str) -> str:
            lines = []
            for param, value in params.items():
                if isinstance(value, dict):
                    s_param = format_params(value, param)
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

    def dicts(self) -> List[Dict[str, Any]]:
        return [setup.dict() for setup in self]

    @classmethod
    def merge(cls, setups_lst: Sequence["SetupCollection"]) -> "SetupCollection":
        return cls([setup for setups in setups_lst for setup in setups])

    def compress(self) -> Setup:
        return Setup.compress(self)

    def compress_partially(self, param: Optional[str] = None) -> "SetupCollection":
        if param == "outfile":
            grouped_setups: Dict[Setup, List[Setup]] = {}
            for setup in self:
                key = setup.derive({"outfile": "none"})
                if key not in grouped_setups:
                    grouped_setups[key] = []
                grouped_setups[key].append(setup)
            new_setup_lst: List[Setup] = []
            for setup_lst_i in grouped_setups.values():
                outfiles: List[str] = []
                for setup in setup_lst_i:
                    if isinstance(setup.outfile, str):
                        outfiles.append(setup.outfile)
                    else:
                        outfiles.extend(setup.outfile)
                new_setup_lst.append(
                    setup_lst_i[0].derive({"outfile": tuple(outfiles)})
                )
            return SetupCollection(new_setup_lst)
        else:
            raise NotImplementedError(f"{type(self).__name__}.compress('{param}')")

    def decompress(self) -> List["SetupCollection"]:
        return self.decompress_partially(select=None, skip=None)

    def derive(self, params: Mapping[str, Any]) -> "SetupCollection":
        return type(self)([setup.derive(params) for setup in self])

    def decompress_partially(
        self, select: Optional[Collection[str]], skip: Optional[Collection[str]] = None
    ) -> List["SetupCollection"]:
        if (select, skip) == (None, None):
            return [setup.decompress() for setup in self]
        sub_setup_lst_lst: List[List[Setup]] = []
        for setup in self:
            sub_setups = setup.decompress_partially(select, skip)
            if not sub_setup_lst_lst:
                sub_setup_lst_lst = [[sub_setup] for sub_setup in sub_setups]
            else:
                # SR_TMP <
                assert len(sub_setups) == len(
                    sub_setup_lst_lst
                ), f"{len(sub_setups)} != {len(sub_setup_lst_lst)}"
                # SR_TMP >
                for idx, sub_setup in enumerate(sub_setups):
                    sub_setup_lst_lst[idx].append(sub_setup)
        return [SetupCollection(sub_setup_lst) for sub_setup_lst in sub_setup_lst_lst]

    def decompress_twice(
        self, outer: str, skip: Optional[Collection[str]] = None
    ) -> List["SetupCollection"]:
        sub_setups_lst: List[SetupCollection] = []
        for setup in self:
            for sub_setup in setup.decompress_partially([outer], skip):
                sub_sub_setups = sub_setup.decompress(skip)
                sub_setups_lst.append(sub_sub_setups)
        return sub_setups_lst

    def collect(
        self, param: str, flatten: bool = False, exclude_nones: bool = False
    ) -> List[Any]:
        """Collect all unique values of a parameter for all setups.

        Args:
            param: Name of parameter.

            flatten (optional): Unpack values that are collection of sub-values.

            exclude_nones (optional): Exclude values -- and, if ``flatten`` is
                true, also sub-values -- that are None.

        """
        values: List[Any] = []
        if is_core_setup_param(param) or is_setup_param(param):
            for var_setup in self:
                if is_core_setup_param(param):
                    value = getattr(var_setup.core, param)
                else:
                    value = getattr(var_setup, param)
                if isinstance(value, Collection) and flatten:
                    for sub_value in value:
                        if exclude_nones and sub_value is None:
                            continue
                        if sub_value not in values:
                            values.append(sub_value)
                else:
                    if exclude_nones and value is None:
                        continue
                    if value not in values:
                        values.append(value)
        elif param.startswith("dimensions."):
            dims_param = param.split(".", 1)[-1]
            if is_dimensions_param(dims_param):
                for dimensions in self.collect("dimensions"):
                    value = dimensions.get(dims_param)
                    for sub_value in (
                        value if isinstance(value, Collection) else [value]
                    ):
                        if exclude_nones and sub_value is None:
                            continue
                        if sub_value not in values:
                            values.append(sub_value)
            else:
                raise ValueError("invalid param", param)
        else:
            raise ValueError("invalid param", param)
        return values

    def collect_equal(self, param: str) -> Any:
        """Collect the value of a parameter that is shared by all setups."""
        values = self.collect(param)
        if not all(value == values[0] for value in values[1:]):
            raise UnequalSetupParamValuesError(param, values)
        return next(iter(values))

    @overload
    def group(self, param: str) -> Dict[Any, "SetupCollection"]:
        ...

    @overload
    def group(self, param: Sequence[str]) -> Dict[Tuple[Any, ...], "SetupCollection"]:
        ...

    def group(self, param):
        """Group setups by the value of one or more parameters."""
        if not isinstance(param, str):
            grouped: Dict[Tuple[Any, ...], "SetupCollection"] = {}
            params: List[str] = list(param)
            for value, sub_setups in self.group(params[0]).items():
                if len(params) == 1:
                    grouped[(value,)] = sub_setups
                elif len(params) > 1:
                    for values, sub_sub_setups in sub_setups.group(params[1:]).items():
                        key = tuple([value] + list(values))
                        grouped[key] = sub_sub_setups
                else:
                    raise NotImplementedError(f"{len(param)} sub_params", param)
            return grouped
        else:
            grouped_raw: Dict[Any, List[Setup]] = {}
            for setup in self:
                value = get_setup_param_value(setup, param)
                if value not in grouped_raw:
                    grouped_raw[value] = []
                grouped_raw[value].append(setup)
            grouped: Dict[Any, "SetupCollection"] = {
                value: type(self)(setups) for value, setups in grouped_raw.items()
            }
            return grouped

    @overload
    def complete_dimensions(
        self, nc_meta_data: Mapping[str, Any], *, inplace: Literal[False] = ...
    ) -> "SetupCollection":
        ...

    @overload
    def complete_dimensions(
        self, nc_meta_data: Mapping[str, Any], *, inplace: Literal[True]
    ) -> None:
        ...

    def complete_dimensions(self, nc_meta_data, *, inplace=False):
        """Complete unconstrained dimensions based on available indices."""
        obj = self if inplace else self.copy()
        for setup in obj:
            setup.core.complete_dimensions(nc_meta_data, inplace=True)
            if setup.base_time is None:
                setup.base_time = nc_meta_data["derived"]["time_steps"][0]
        return None if inplace else obj

    def override_output_suffixes(self, suffix: Union[str, Collection[str]]) -> None:
        """Override output file suffixes one or more times.

        If multiple suffixes are passed, all setups are multiplied as many
        times.

        Args:
            suffix: One or more replacement suffix.

        """
        suffixes: List[str] = list([suffix] if isinstance(suffix, str) else suffix)
        if not suffixes:
            raise ValueError("must pass one or more suffixes")
        new_setups: List[Setup] = []
        for setup in self:
            old_outfiles: List[str] = (
                [setup.outfile]
                if isinstance(setup.outfile, str)
                else list(setup.outfile)
            )
            new_outfiles: List[str] = []
            for old_outfile in list(old_outfiles):
                if any(old_outfile.endswith(f".{suffix}") for suffix in ["png", "pdf"]):
                    old_outfile = ".".join(old_outfile.split(".")[:-1])
                for suffix_i in suffixes:
                    if suffix_i.startswith("."):
                        suffix_i = suffix_i[1:]
                    new_outfiles.append(f"{old_outfile}.{suffix_i}")
            new_setup = setup.derive(
                {
                    "outfile": next(iter(new_outfiles))
                    if len(new_outfiles) == 1
                    else tuple(new_outfiles)
                }
            )
            new_setups.append(new_setup)
        self._setups = new_setups


class SetupFile:
    """Setup file to be read from and/or written to disk."""

    def __init__(self, path: str) -> None:
        """Create an instance of ``SetupFile``."""
        self.path: str = path

    @classmethod
    def read_many(
        cls,
        paths: Sequence[str],
        override: Optional[Dict[str, Any]] = None,
        only: Optional[int] = None,
        each_only: Optional[int] = None,
    ) -> SetupCollection:
        if only is not None:
            if only < 0:
                raise ValueError("only must not be negative", only)
            each_only = only
        elif each_only is not None:
            if each_only < 0:
                raise ValueError("each_only must not be negative", each_only)
        setup_lst: List[Setup] = []
        for path in paths:
            for setup in cls(path).read(override=override, only=each_only):
                if only is not None and len(setup_lst) >= only:
                    break
                if setup not in setup_lst:
                    setup_lst.append(setup)
        return SetupCollection(setup_lst)

    # pylint: disable=R0914  # too-many-locals
    def read(
        self, *, override: Optional[Dict[str, Any]] = None, only: Optional[int] = None
    ) -> SetupCollection:
        """Read the setup from a text file in TOML format."""
        with open(self.path, "r") as f:
            try:
                raw_data = toml.load(f)
            except Exception as e:
                raise Exception(
                    f"error parsing TOML file {self.path} ({type(e).__name__}: {e})"
                ) from e
        if not raw_data:
            raise ValueError("empty setup file", self.path)
        semi_raw_data = nested_dict_resolve_wildcards(
            raw_data, double_criterion=lambda key: key.endswith("+")
        )
        raw_params_lst = decompress_nested_dict(
            semi_raw_data, branch_end_criterion=lambda key: not key.startswith("_")
        )
        if override is not None:
            raw_params_lst, old_raw_params_lst = [], raw_params_lst
            for old_raw_params in old_raw_params_lst:
                raw_params = {**old_raw_params, **override}
                if raw_params not in raw_params_lst:
                    raw_params_lst.append(raw_params)
        setups = SetupCollection.from_raw_params(raw_params_lst)
        if only is not None:
            if only < 0:
                raise ValueError(f"only must not be negative ({only})")
            setups = SetupCollection(list(setups)[:only])
        return setups

    def write(self, *args, **kwargs) -> None:
        """Write the setup to a text file in TOML format."""
        raise NotImplementedError(f"{type(self).__name__}.write")


def setup_repr(obj: Union["CoreSetup", "Setup"]) -> str:
    def fmt(obj):
        if isinstance(obj, str):
            return f"'{obj}'"
        return str(obj)

    s_attrs = ",\n  ".join(f"{k}={fmt(v)}" for k, v in obj.dict().items())
    return f"{type(obj).__name__}(\n  {s_attrs},\n)"

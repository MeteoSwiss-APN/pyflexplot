# pylint: disable=C0302  # too-many-lines (>1000)
"""Plot setup and setup files.

The setup parameters that are exposed in the setup files are all described in
the docstring of the class method ``Setup.create``.

"""
# Standard library
import dataclasses as dc
from pprint import pformat
from typing import Any
from typing import Callable
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
from typing_extensions import Literal

# First-party
from srutils.dict import decompress_multival_dict
from srutils.dict import merge_dicts
from srutils.exceptions import InvalidParameterValueError
from srutils.exceptions import UnexpandableValueError
from srutils.format import sfmt
from srutils.str import join_multilines

# Local
from ..utils.exceptions import UnequalSetupParamValuesError
from ..utils.wrappers import cast_field_value
from .base_setup import BaseSetup
from .dimensions import CoreDimensions
from .dimensions import Dimensions
from .dimensions import is_dimensions_param

# Some plot-specific default values
ENS_PROBABILITY_DEFAULT_PARAM_THR = 0.0
ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN = 1
ENS_CLOUD_TIME_DEFAULT_PARAM_THR = 0.0


# SR_TMP <<< TODO cleaner solution
def is_plot_panel_setup_param(param: str, recursive: bool = False) -> bool:
    if recursive:
        return is_plot_panel_setup_param(param) or is_dimensions_param(param)
    return param in PlotPanelSetup.get_params()


# SR_TMP <<< TODO cleaner solution
def is_ensemble_params_param(param: str, recursive: bool = False) -> bool:
    if recursive:
        raise NotImplementedError("recursive")
    if param.startswith("ens_params."):
        param = param.replace("ens_params.", "")
    return param in EnsembleParams.get_params()


@dc.dataclass
class EnsembleParams(BaseSetup):
    mem_min: Optional[int] = None
    pctl: Optional[float] = None
    thr: Optional[float] = None
    thr_type: str = "lower"


# pylint: disable=R0902  # too-many-instance-attributes (>7)
@dc.dataclass
class PlotPanelSetup(BaseSetup):
    """Setup of an individual panel of a plot.

    See docstring of ``PlotSetup.create`` for a description of the parameters.

    """

    # Basics
    plot_variable: str = "concentration"
    ens_variable: str = "none"

    # Tweaks
    integrate: bool = False
    combine_levels: bool = False
    combine_species: bool = False

    # Ensemble-related
    ens_params: EnsembleParams = dc.field(default_factory=EnsembleParams)

    # Plot appearance
    lang: str = "en"
    domain: str = "full"
    domain_size_lat: Optional[float] = None
    domain_size_lon: Optional[float] = None

    # Dimensions
    dimensions_default: str = "all"
    dimensions: Dimensions = dc.field(default_factory=Dimensions)

    # pylint: disable=R0912  # too-many-branches (>12)
    def __post_init__(self) -> None:

        self._check_types()

        # Check plot_variable
        choices = [
            "concentration",
            "tot_deposition",
            "dry_deposition",
            "wet_deposition",
            "affected_area",
            "cloud_arrival_time",
            "cloud_departure_time",
        ]
        assert self.plot_variable in choices, self.plot_variable

        # Check ens_variable
        choices = [
            "cloud_arrival_time",
            "cloud_departure_time",
            "maximum",
            "mean",
            "med_abs_dev",
            "median",
            "minimum",
            "none",
            "percentile",
            "probability",
            "std_dev",
        ]
        if isinstance(self.ens_variable, str):
            assert self.ens_variable in choices, self.ens_variable
        else:
            for sub_value in self.ens_variable:
                assert sub_value in choices, sub_value
        self._init_defaults()

        # Check combine_levels vs. dimensions.level
        if not self.combine_levels:
            if isinstance(self.dimensions.level, Collection):
                raise ValueError(
                    f"combine_levels=False is inconsistent with multiple levels"
                    f" (dimensions.level={self.dimensions.level})"
                )

        # Check combine_species vs. dimensions.species
        if not self.combine_species:
            if isinstance(self.dimensions.species_id, Collection):
                raise ValueError(
                    f"combine_species=False is inconsistent with multiple species"
                    f" (dimensions.species_id={self.dimensions.species_id})"
                )

        # Check dimensions.variable
        variable = Dimensions.derive_variable(self.plot_variable)
        if self.dimensions.variable is None:
            self.dimensions = self.dimensions.derive({"variable": variable})
        if self.dimensions.variable != variable:
            raise ValueError(
                f"dimensions.variable value '{self.dimensions.variable}' inconsistent"
                f" with plot_variable '{self.plot_variable}'; expecting '{variable}'"
            )

    def collect(self, param: str) -> Any:
        """Collect the value(s) of a parameter."""
        if is_plot_panel_setup_param(param):
            value = getattr(self, param)
        elif is_dimensions_param(param):
            value = self.dimensions.get(param.replace("dimensions.", ""))
        else:
            raise ValueError(f"invalid param '{param}'")
        return value

    @overload
    def complete_dimensions(
        self,
        raw_dimensions: Mapping[str, Any],
        species_ids: Sequence[int],
        *,
        inplace: Literal[False] = ...,
    ) -> "PlotPanelSetup":
        ...

    @overload
    def complete_dimensions(
        self,
        raw_dimensions: Mapping[str, Any],
        species_ids: Sequence[int],
        *,
        inplace: Literal[True],
    ) -> None:
        ...

    def complete_dimensions(self, raw_dimensions, species_ids, *, inplace=False):
        """Complete unconstrained dimensions based on available indices."""
        obj: "PlotPanelSetup" = self if inplace else self.copy()
        obj.dimensions.complete(
            raw_dimensions,
            species_ids,
            self.plot_variable,
            mode=obj.dimensions_default,
            inplace=True,
        )
        return None if inplace else obj

    def decompress(
        self,
        select: Optional[Collection[str]] = None,
        skip: Optional[Collection[str]] = None,
    ) -> "PlotPanelSetupGroup":
        """Create a setup object for each decompressed dimensions object.

        Args:
            select (optional): List of parameter names to select for
                decompression; all others will be skipped; parameters named in
                both ``select`` and ``skip`` will be skipped.

            skip (optional): List of parameter names to skip; if they have list
                values, those are retained as such; parameters named in both
                ``skip`` and ``select`` will be skipped.

        """
        # SR_TMP <
        select_dims = (
            None
            if select is None
            else [
                param.replace("dimensions.", "")
                for param in select
                if is_dimensions_param(param)
            ]
        )
        skip_dims = (
            None
            if skip is None
            else [
                param.replace("dimensions.", "")
                for param in skip
                if is_dimensions_param(param)
            ]
        )
        if (
            select_dims
            and "variable" in select_dims
            and "variable" not in (skip_dims or [])
        ):
            raise ValueError(
                "cannot decompress Dimensions.variable, because it is derived from"
                f" {type(self).__name__}.plot_variable; decompress the Dimensions"
                " object instead which on its own is independent of plot_variable"
            )
        skip_dims = list(skip_dims or []) + ["variable"]
        # SR_TMP >
        setups: List["PlotPanelSetup"] = []
        for dims in self.dimensions.decompress(select=select_dims, skip=skip_dims):
            params = self.dict(rec=False)
            params["dimensions"] = dims
            setup = type(self)(**params)
            setups.append(setup)
        return PlotPanelSetupGroup(setups)

    def derive(self, params: Mapping[str, Any]) -> "PlotPanelSetup":
        params = dict(params)
        self_dct = self.dict()
        # Remove 'dimensions.variable' as it is re-derived from 'plot_variable'
        self_dct["dimensions"].pop("variable")
        return self.create(merge_dicts(self_dct, params, overwrite_seqs=True))

    def dict(self, rec: bool = True) -> Dict[str, Any]:
        """Return the parameter names and values as a dict.

        Args:
            rec (optional): Recursively return sub-objects like ``Dimensions``
                as dicts.

        """
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        return {
            **super().dict(),
            "ens_params": self.ens_params.dict() if rec else self.ens_params,
            "dimensions": self.dimensions.dict() if rec else self.dimensions,
        }

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        dct = self.dict(rec=False)
        ens_params = dct.pop("ens_params")
        dims = dct.pop("dimensions")
        return tuple(
            list(dct.items())
            + [("ens_params", ens_params.tuple()), ("dimensions", dims.tuple())]
        )

    def _check_types(self) -> None:
        for param, value in self.dict(rec=False).items():
            error = ValueError(f"param {param} has invalid value {sfmt(value)}")
            if param == "dimensions":
                if not isinstance(value, Dimensions):
                    raise error
            elif param == "ens_params":
                continue
            else:
                try:
                    cast_field_value(type(self), param, value)
                except InvalidParameterValueError as e:
                    raise error from e

    def _init_defaults(self) -> None:
        """Set some default values depending on other parameters."""
        # Init domain_size_lat_lon
        lat = self.domain_size_lat
        lon = self.domain_size_lon
        if lat is None and lon is None:
            if self.domain in ["release_site", "cloud"]:
                # Lat size derived from lon size and map aspect ratio
                self.domain_size_lon = 20.0

        # Init ens_params
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        if self.ens_params.mem_min is None:
            if self.ens_variable in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                # pylint: disable=E0237  # assigning-non-slot [pylint 2.7.4]
                # (pylint 2.7.4 does not support dataclasses.field)
                self.ens_params.mem_min = ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN
        if self.ens_variable == "percentile":
            # pylint: disable=E1101  # no-member [pylint 2.7.4]
            # (pylint 2.7.4 does not support dataclasses.field)
            assert self.ens_params.pctl is not None
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        if self.ens_params.thr is None:
            if self.ens_variable in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                # pylint: disable=E0237  # assigning-non-slot [pylint 2.7.4]
                # (pylint 2.7.4 does not support dataclasses.field)
                self.ens_params.thr = ENS_CLOUD_TIME_DEFAULT_PARAM_THR
            elif self.ens_variable == "probability":
                # pylint: disable=E0237  # assigning-non-slot [pylint 2.7.4]
                # (pylint 2.7.4 does not support dataclasses.field)
                self.ens_params.thr = ENS_PROBABILITY_DEFAULT_PARAM_THR
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        assert self.ens_params.thr_type in ["lower", "upper"]

    @classmethod
    def as_setup(
        cls, obj: Union[Mapping[str, Any], "PlotPanelSetup"]
    ) -> "PlotPanelSetup":
        if isinstance(obj, cls):
            return obj
        assert isinstance(obj, Mapping)  # mypy
        return cls(**obj)

    @classmethod
    def cast(cls, param: str, value: Any, recursive: bool = False) -> Any:
        if recursive:
            if is_dimensions_param(param):
                return Dimensions.cast(param.replace("dimensions.", ""), value)
        return super().cast(param, value)

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "PlotPanelSetup":
        params = dict(params)
        dims_params = dict(params.pop("dimensions", {}))
        try:
            plot_variable = params["plot_variable"]
        except KeyError:
            plot_variable = cls().plot_variable  # default
        if "variable" in dims_params:
            variable = dims_params["variable"]
            variable_check = Dimensions.derive_variable(plot_variable)
            if variable != variable_check and set(variable) != set(variable_check):
                raise ValueError(
                    f"'dimensions.variable' param value {sfmt(variable)} is"
                    f" incompatible with plot_variable '{plot_variable}'"
                    f": expecting {sfmt(variable_check)}"
                )
        params["dimensions"] = Dimensions.create(
            dims_params, plot_variable=plot_variable
        )
        params["ens_params"] = EnsembleParams.create(params.pop("ens_params", {}))
        params = cls.cast_many(params)
        return cls(**params)


class PlotPanelSetupGroup:
    def __init__(self, panels: Sequence[PlotPanelSetup]) -> None:
        """Create an instance of ``PlotPanelSetupGroup``."""
        self._panels: List[PlotPanelSetup] = list(panels)

    def collect(
        self,
        param: str,
        *,
        flatten: bool = False,
        exclude_nones: bool = False,
        unique: bool = False,
    ) -> List[Any]:
        """Collect all unique values of a parameter for all setups.

        Args:
            param: Name of parameter.

            exclude_nones (optional): Exclude values -- and, if ``flatten`` is
                true, also sub-values -- that are None.

            flatten (optional): Unpack values that are collection of sub-values.

            unique (optional): Return duplicate values only once.

        """
        values: List[Any] = []
        for setup in self:
            value = setup.collect(param)
            if exclude_nones and value is None:
                continue
            if flatten and isinstance(setup, Collection) and not isinstance(setup, str):
                for sub_value in value:
                    if exclude_nones and sub_value is None:
                        continue
                    if not unique or sub_value not in values:
                        values.append(sub_value)
            elif not unique or value not in values:
                values.append(value)
        return values

    def collect_equal(self, param: str) -> Any:
        """Collect the value of a parameter that is shared by all setups."""
        values = self.collect(param, unique=True)
        if not values:
            return None
        if not all(value == values[0] for value in values[1:]):
            raise UnequalSetupParamValuesError(param, values)
        return next(iter(values))

    @overload
    def decompress(
        self,
        select: Optional[Collection[str]] = ...,
        skip: Optional[Collection[str]] = ...,
        *,
        internal: Literal[True] = True,
    ) -> "PlotPanelSetupGroup":
        ...

    @overload
    def decompress(
        self,
        select: Optional[Collection[str]] = ...,
        skip: Optional[Collection[str]] = ...,
        *,
        internal: Literal[False],
    ) -> List["PlotPanelSetupGroup"]:
        ...

    def decompress(
        self,
        select=None,
        skip=None,
        *,
        internal=True,
    ):
        """Create a group object for each decompressed setup object.

        Args:
            select (optional): List of parameter names to select for
                decompression; all others will be skipped; parameters named in
                both ``select`` and ``skip`` will be skipped.

            skip (optional): List of parameter names to skip; if they have list
                values, those are retained as such; parameters named in both
                ``skip`` and ``select`` will be skipped.

            internal (optional): Decompress setup group internally and return
                one group containing the decompressed setup objects; otherwise,
                a separate group is returned for each decompressed setup object.

        """
        if internal:
            sub_setups: List[PlotPanelSetup] = [
                sub_setup
                for setup in self
                for sub_setup in setup.decompress(select=select, skip=skip)
            ]
            return type(self)(sub_setups)
        if skip is None:
            return [
                type(self)([setup]) for setup in self.decompress(select, internal=True)
            ]
        setups_by_idx: Dict[int, List[PlotPanelSetup]] = {}
        for setup in self:
            for idx, sub_setup in enumerate(setup.decompress(select=select, skip=skip)):
                if idx not in setups_by_idx:
                    setups_by_idx[idx] = []
                setups_by_idx[idx].append(sub_setup)
        return [type(self)(setups) for setups in setups_by_idx.values()]

    def dicts(self) -> List[Dict[str, Any]]:
        return [setup.dict() for setup in self]

    def tuple(self) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
        return tuple(map(PlotPanelSetup.tuple, self))

    def __eq__(self, other: Any) -> bool:
        try:
            return self.dicts() == other.dicts()
        except AttributeError:
            return False

    def __iter__(self) -> Iterator[PlotPanelSetup]:
        return iter(self._panels)

    def __len__(self) -> int:
        return len(self._panels)

    def __repr__(self) -> str:
        try:
            return PlotPanelSetupGroupFormatter(self).repr()
        # pylint: disable=W0703  # broad-except
        except Exception as e:
            return (
                f"<{type(self).__name__}.__repr__:"
                f" exception in PlotPanelSetupGroupFormatter(self).repr(): {repr(e)}"
                f"\n{pformat(self.dicts())}>"
            )

    @classmethod
    def create(
        cls,
        params: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
        multipanel_param: Optional[str] = None,
    ) -> "PlotPanelSetupGroup":
        """Prepare and create an instance of ``PlotPanelSetupGroup``.

        Args:
            params: Parameters dict defining one or more ``PlotPanelSetup``
                objects; if ``multipanel_param`` is given, multiple setup
                objects are created, otherwise exactly one.

            multipanel_param (optional): Parameter in ``params`` with multiple
                (i.e., a sequence of) values; for each individual value, a
                separate ``PlotPanelSetup`` object is created; if the value of
                that parameter is not a sequence, an exception is raised; if
                ``multipanel_param`` is None, exactly one setup object will
                becreated based on the unmodified ``params`` dict.

        """
        if isinstance(params, Mapping):
            return cls._create_from_dict(params, multipanel_param)
        elif isinstance(params, Sequence) and not isinstance(params, str):
            return cls._create_from_seq(params, multipanel_param)
        raise ValueError(
            "params must be a params dict or a sequence thereof, not a "
            + type(params).__name__
        )

    @classmethod
    def _create_from_seq(
        cls,
        params: Sequence[Mapping[str, Any]],
        multipanel_param: Optional[str] = None,
    ) -> "PlotPanelSetupGroup":
        def check_consistency(
            mp_param: Optional[str], params: Sequence[Mapping[str, Any]]
        ) -> None:
            def param_missing_error(param: str, params: Mapping[str, Any]) -> Exception:
                return ValueError(
                    f"multipanel_param '{param}' not in params dict:\n{pformat(params)}"
                )

            if mp_param is None:
                if len(params) != 1:
                    raise ValueError(
                        f"must pass exactly one params dict (not {len(params)})"
                        " if multipanel_param is None"
                    )
                return

            for params_i in params:
                fct: Callable[[str], bool]
                for fct, name in [
                    (is_ensemble_params_param, "ens_params"),
                    (is_dimensions_param, "dimensions"),
                    (is_plot_panel_setup_param, ""),
                ]:
                    if not fct(mp_param):
                        continue
                    if name and mp_param.startswith(f"{name}."):
                        mp_param = mp_param.replace(f"{name}.", "")
                    try:
                        value = (params_i[name] if name else params_i)[mp_param]
                    except KeyError as e:
                        raise param_missing_error(mp_param, params_i) from e
                    if isinstance(value, Sequence) and not isinstance(value, str):
                        raise ValueError(
                            "when passing a list of params dicts, multipanel_param"
                            f" values must not be sequences ('{mp_param}': {value})"
                        )
                    break
                else:
                    raise ValueError(f"invalid multipanel_param '{mp_param}'")

        check_consistency(multipanel_param, params)

        setups: List[PlotPanelSetup] = []
        for params_i in params:
            try:
                setup = PlotPanelSetup.create(params_i)
            except ValueError as e:
                raise ValueError(
                    "could not create PlotPanelSetup with params dict:"
                    f"\n{pformat(params_i)}\nmaybe multipanel_param has wrong value"
                    f" ({sfmt(multipanel_param)})?"
                ) from e
            setups.append(setup)
        return cls(setups)

    @classmethod
    def _create_from_dict(
        cls,
        params: Mapping[str, Any],
        multipanel_param: Optional[str] = None,
    ) -> "PlotPanelSetupGroup":
        def handle_sub_params(
            mp_param: str, params: Mapping[str, Any], name: str = ""
        ) -> List[Dict[str, Any]]:
            if name and mp_param.startswith(f"{name}."):
                mp_param = mp_param.replace(f"{name}.", "")
            if not name:
                sub_params = dict(params)
            else:
                try:
                    sub_params = dict(params[name])
                except KeyError as e:
                    raise ValueError(
                        f"params dict missing '{name}' despite multipanel_param"
                        f" '{mp_param}':\n" + pformat(params)
                    ) from e
            try:
                sub_params_lst = decompress_multival_dict(
                    sub_params, select=[mp_param], unexpandable_ok=False
                )
            except UnexpandableValueError as e:
                raise ValueError(
                    f"multipanel_param '{mp_param}' not expandable in params dict:\n"
                    + pformat(sub_params)
                ) from e
            if name:
                return [
                    merge_dicts(params, {name: sub_params}, overwrite_seqs=True)
                    for sub_params in sub_params_lst
                ]
            return [dict(sub_params) for sub_params in sub_params_lst]

        params = dict(params)
        params_lst: List[Dict[str, Any]]
        if multipanel_param is None:
            params_lst = [params]
        elif is_ensemble_params_param(multipanel_param):
            params_lst = handle_sub_params(multipanel_param, params, "ens_params")
        elif is_dimensions_param(multipanel_param):
            params_lst = handle_sub_params(multipanel_param, params, "dimensions")
        elif is_plot_panel_setup_param(multipanel_param):
            params = dict(params)
            try:
                values = params.pop(multipanel_param)
            except KeyError as e:
                raise ValueError(
                    f"multipanel_param '{multipanel_param}' not in params dict: "
                    + pformat(params)
                ) from e
            if not (isinstance(values, Sequence) and not isinstance(values, str)):
                raise ValueError(
                    f"value ({sfmt(values)}) of multipanel_param"
                    f" '{multipanel_param}' is a {type(values).__name__}, not a"
                    " sequence"
                )
            params_lst = [{**params, multipanel_param: value} for value in values]
        else:
            raise ValueError(f"invalid multipanel_param '{multipanel_param}'")
        return cls._create_from_seq(params_lst, multipanel_param)


class SetupGroupFormatter:
    """Format a human-readable representation of a ``*SetupGroup``.

    Parameters with shared values between all setup objects are shown
    separately from those with differing values.

    Note that the representation is human-readable only, i.e., not
    formatted as valid code.

    """

    group_methods: List[str] = []

    def __init__(self, obj: Any) -> None:
        """Create an instance of ``SetupGroupFormatter``.

        Args:
            obj: An object of type ``PlotSetupGroup``, ``PlotPanelSetupGroup``
                or the like.

        """
        if "SetupGroup" not in type(obj).__name__:
            # Note: Proper typing w/o restructuring causes curcular dependencies
            raise ValueError(
                f"obj of type {type(obj).__name__} does not appear to be a setup group"
                " ('SetupGroup' not in type name)"
            )
        self.obj = obj

    def repr(self) -> str:
        """Return a human-readable representation string of ``self.obj``."""
        same: Dict[str, Any] = {}
        diff: Dict[str, Any] = {}
        self._group_params(same, diff)
        lines = [
            f"n: {len(self.obj)}",
            self._format_params(same, "same"),
            self._format_params(diff, "diff"),
        ]
        body = join_multilines(lines, indent=2)
        return "\n".join([f"{type(self.obj).__name__}[", body, "]"])

    def _group_params(self, same: Dict[str, Any], diff: Dict[str, Any]) -> None:
        pass  # override

    def _format_params(self, params: Dict[str, Any], name: str) -> str:
        lines = []
        for param, value in params.items():
            if isinstance(value, dict):
                s_param = self._format_params(value, param)
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


class PlotPanelSetupGroupFormatter(SetupGroupFormatter):
    """Format a human-readable representation of a ``PlotPanelSetupGroup``."""

    def _group_params(self, same: Dict[str, Any], diff: Dict[str, Any]) -> None:
        self._group_panel_params(same, diff)
        self._group_dims_params(same, diff)

    def _group_panel_params(self, same: Dict[str, Any], diff: Dict[str, Any]) -> None:
        for param in PlotPanelSetup.get_params():
            if param == "dimensions":
                continue  # Handled below
            try:
                value = self.obj.collect_equal(param)
            except UnequalSetupParamValuesError:
                diff[param] = self.obj.collect(param)
            else:
                same[param] = value

    def _group_dims_params(self, same: Dict[str, Any], diff: Dict[str, Any]) -> None:
        dims_same: Dict[str, Any] = {}
        dims_diff: Dict[str, Any] = {}
        dimensions = self.obj.collect("dimensions", flatten=True)
        for param in CoreDimensions.get_params():
            values = []
            for dims in dimensions:
                values.append(dims.get(param))
            if len(set(values)) == 1:
                dims_same[param] = next(iter(values))
            else:
                dims_diff[param] = values
        if dims_same:
            same["dimensions"] = dims_same
        if dims_diff:
            diff["dimensions"] = dims_diff

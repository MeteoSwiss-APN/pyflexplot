# pylint: disable=C0302  # too-many-lines (>1000)
"""Plot setup and setup files.

The setup parameters that are exposed in the setup files are all described in
the docstring of the class method ``Setup.create``.

"""
# Standard library
import dataclasses as dc
from pprint import pformat
from typing import Any
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
from srutils.dataclasses import cast_field_value
from srutils.dict import merge_dicts
from srutils.exceptions import InvalidParameterValueError
from srutils.format import nested_repr
from srutils.format import sfmt
from srutils.str import join_multilines

# Local
from ..utils.exceptions import UnequalSetupParamValuesError
from .dimensions import CoreDimensions
from .dimensions import Dimensions
from .dimensions import is_dimensions_param

# Some plot-specific default values
ENS_PROBABILITY_DEFAULT_PARAM_THR = 0.0
ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN = 1
ENS_CLOUD_TIME_DEFAULT_PARAM_THR = 0.0


# SR_TMP <<< TODO cleaner solution
def is_plot_panel_setup_param(param: str) -> bool:
    return param in PlotPanelSetup.get_params()


@dc.dataclass
class EnsembleParams:
    mem_min: Optional[int] = None
    pctl: Optional[float] = None
    thr: Optional[float] = None
    thr_type: str = "lower"

    def dict(self) -> Dict[str, Any]:
        return dc.asdict(self)

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.dict().items())


# pylint: disable=R0902  # too-many-instance-attributes (>7)
@dc.dataclass
class PlotPanelSetup:
    """Setup of an individual panel of a plot.

    See docstring of ``PlotSetup.create`` for a description of the parameters.

    """

    # Basics
    plot_variable: str = "concentration"
    ens_variable: str = "none"

    # Tweaks
    integrate: bool = False
    combine_deposition_types: bool = False
    combine_levels: bool = False
    combine_species: bool = False

    # Ensemble-related
    ens_params: EnsembleParams = EnsembleParams()

    # Plot appearance
    lang: str = "en"
    domain: str = "full"
    domain_size_lat: Optional[float] = None
    domain_size_lon: Optional[float] = None

    # Dimensions
    dimensions_default: str = "all"
    dimensions: Dimensions = Dimensions()

    # pylint: disable=R0912  # too-many-branches (>12)
    def __post_init__(self) -> None:

        self._check_types()

        # Check plot_variable
        choices = [
            "concentration",
            "deposition",
            "affected_area",
            "cloud_arrival_time",
            "cloud_departure_time",
        ]
        assert self.plot_variable in choices, self.plot_variable

        # Check ens_variable
        choices = [
            "ens_cloud_arrival_time",
            "ens_cloud_departure_time",
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

    # SR_TMP <<<
    @property
    def deposition_type_str(self) -> str:
        if self.dimensions.deposition_type is None:
            return "none"
        else:
            return self.dimensions.deposition_type

    def decompress(
        self,
        *,
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
            else [param for param in select if is_dimensions_param(param)]
        )
        skip_dims = (
            None
            if skip is None
            else [param for param in skip if is_dimensions_param(param)]
        )
        # SR_TMP >
        setups: List["PlotPanelSetup"] = []
        for dims in self.dimensions.decompress(select=select_dims, skip=skip_dims):
            params = merge_dicts(
                self.dict(),
                {"dimensions": dims.dict()},
                overwrite_seqs=True,
                overwrite_seq_dicts=True,
            )
            # SR_TMP < Does this logic really belong here? I doubt it...
            if (
                select is not None and "plot_variable" not in select
            ) or "plot_variable" in (skip or []):
                params_lst = [params]
            elif params["plot_variable"] == "affected_area":
                params_lst = [
                    {**params, "plot_variable": "concentration"},
                    {**params, "plot_variable": "deposition"},
                ]
            elif params["plot_variable"] in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                params_lst = [{**params, "plot_variable": "concentration"}]
            else:
                params_lst = [params]
            # SR_TMP >
            for params in params_lst:
                setup = self.create(params)
                setups.append(setup)
        return PlotPanelSetupGroup(setups)

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

    def dict(self, rec: bool = True) -> Dict[str, Any]:
        """Return the parameter names and values as a dict.

        Args:
            rec (optional): Recursively return sub-objects like ``Dimensions``
                as dicts.

        """
        return {
            **dc.asdict(self),
            "ens_params": self.ens_params.dict() if rec else self.ens_params,
            "dimensions": self.dimensions.dict() if rec else self.dimensions,
        }

    def copy(self) -> "PlotPanelSetup":
        return self.create(self.dict())

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
        if self.ens_params.mem_min is None:
            if self.ens_variable in [
                "ens_cloud_arrival_time",
                "ens_cloud_departure_time",
            ]:
                self.ens_params.mem_min = ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN
        if self.ens_variable == "percentile":
            assert self.ens_params.pctl is not None
        if self.ens_params.thr is None:
            if self.ens_variable in [
                "ens_cloud_arrival_time",
                "ens_cloud_departure_time",
            ]:
                self.ens_params.thr = ENS_CLOUD_TIME_DEFAULT_PARAM_THR
            elif self.ens_variable == "probability":
                self.ens_params.thr = ENS_PROBABILITY_DEFAULT_PARAM_THR
        assert self.ens_params.thr_type in ["lower", "upper"]

    def _tuple(self) -> Tuple[Tuple[str, Any], ...]:
        dct = self.dict()
        dims = dct.pop("dimensions")
        return tuple(list(dct.items()) + [("dimensions", tuple(dims.items()))])

    def __hash__(self) -> int:
        return hash(self._tuple())

    def __repr__(self) -> str:  # type: ignore
        return nested_repr(self)

    @classmethod
    def get_params(cls) -> List[str]:
        return list(cls.__dataclass_fields__)  # type: ignore  # pylint: disable=E1101

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "PlotPanelSetup":
        params = dict(params)
        params["dimensions"] = Dimensions.create(params.pop("dimensions", {}))
        params["ens_params"] = EnsembleParams(**params.pop("ens_params", {}))
        return cls(**params)

    @classmethod
    def as_setup(
        cls, obj: Union[Mapping[str, Any], "PlotPanelSetup"]
    ) -> "PlotPanelSetup":
        if isinstance(obj, cls):
            return obj
        assert isinstance(obj, Mapping)  # mypy
        return cls(**obj)


class PlotPanelSetupGroup:
    def __init__(self, panels: Sequence[PlotPanelSetup]) -> None:
        """Create an instance of ``PlotPanelSetupGroup``."""
        self._panels: List[PlotPanelSetup] = list(panels)

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
        for setup in self:
            if is_plot_panel_setup_param(param):
                value = getattr(setup, param)
            elif is_dimensions_param(param):
                value = setup.dimensions.get(param.replace("dimensions.", ""))
            else:
                raise ValueError(f"invalid param '{param}'")
            if exclude_nones and value is None:
                continue
            if flatten and isinstance(setup, Collection) and not isinstance(setup, str):
                for sub_value in value:
                    if exclude_nones and sub_value is None:
                        continue
                    values.append(sub_value)
            else:
                values.append(value)
        return values

    def collect_equal(self, param: str) -> Any:
        """Collect the value of a parameter that is shared by all setups."""
        values = self.collect(param)
        if not values:
            return None
        if not all(value == values[0] for value in values[1:]):
            raise UnequalSetupParamValuesError(param, values)
        return next(iter(values))

    @overload
    def decompress(
        self,
        *,
        select: Optional[Collection[str]] = ...,
        skip: Optional[Collection[str]] = ...,
        internal: Literal[True] = True,
    ) -> "PlotPanelSetupGroup":
        ...

    @overload
    def decompress(
        self,
        *,
        select: Optional[Collection[str]] = ...,
        skip: Optional[Collection[str]] = ...,
        internal: Literal[False],
    ) -> List["PlotPanelSetupGroup"]:
        ...

    def decompress(
        self,
        *,
        select=None,
        skip=None,
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
                one group containing the decompressed setup objectss; otherwise,
                a separate group is returned for each decompressed setup object.

        """
        sub_setups: List[PlotPanelSetup] = [
            sub_setup
            for setup in self
            for sub_setup in setup.decompress(select=select, skip=skip)
        ]
        if internal:
            return type(self)(sub_setups)
        return [type(self)([sub_setup]) for sub_setup in sub_setups]

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
        params: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        multipanel_param: Optional[str] = None,
    ) -> "PlotPanelSetupGroup":
        """Prepare and create an instance of ``PlotPanelSetupGroup``.

        Args:
            params: One or more parameters dicts based on which one or more
                ``PlotPanelSetup`` objects are created each, depending on
                ``multipanel_param``.

            multipanel_param (optional): Parameter in ``params`` based on which
                multiple ``PlotPanelSetup`` objects are created, one for each
                value of the respective parameter; if omitted, exactly one
                object will be created.

        """
        # SR_TMP <
        if not isinstance(params, Mapping):
            if len(params) > 1:
                raise NotImplementedError("multiple params dicts")
            params = next(iter(params))
        assert isinstance(params, Mapping)
        # SR_TMP >
        if multipanel_param is None:
            return cls([PlotPanelSetup.create(params)])
        params = dict(params)
        try:
            values = params.pop(multipanel_param)
        except KeyError as e:
            raise ValueError(
                f"multipanel_param '{multipanel_param}' not among params {list(params)}"
            ) from e
        if not (isinstance(values, Sequence) and not isinstance(values, str)):
            raise ValueError(
                f"value ({sfmt(values)}) of multipanel_param '{multipanel_param}'"
                f" is a {type(values).__name__}, not a sequence"
            )
        panel_setups: List[PlotPanelSetup] = []
        for value in values:
            params_i: Dict[str, Any] = {**params, multipanel_param: value}
            panel_setups.append(PlotPanelSetup.create(params_i))
        return cls(panel_setups)


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
        for param in CoreDimensions.get_params():
            values = []
            for dims in self.obj.collect("dimensions", flatten=True):
                values.append(dims.get(param))
            if len(set(values)) == 1:
                dims_same[param] = next(iter(values))
            else:
                dims_diff[param] = values
        if dims_same:
            same["dimensions"] = dims_same
        if dims_diff:
            diff["dimensions"] = dims_diff

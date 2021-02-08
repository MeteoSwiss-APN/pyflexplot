# pylint: disable=C0302  # too-many-lines (>1000)
"""Plot setup and setup files.

The setup parameters that are exposed in the setup files are all described in
the docstring of the class method ``Setup.create``.

"""
# Standard library
import dataclasses as dc
from typing import Any
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
from srutils.exceptions import InvalidParameterValueError
from srutils.format import nested_repr
from srutils.format import sfmt

# Local
from .dimensions import Dimensions

# Some plot-specific default values
ENS_PROBABILITY_DEFAULT_PARAM_THR = 0.0
ENS_CLOUD_TIME_DEFAULT_PARAM_MEM_MIN = 1
ENS_CLOUD_TIME_DEFAULT_PARAM_THR = 0.0


@dc.dataclass
class EnsembleParams:
    mem_min: Optional[int] = None
    pctl: Optional[float] = None
    thr: Optional[float] = None
    thr_type: str = "lower"


# pylint: disable=R0902  # too-many-instance-attributes (>7)
@dc.dataclass
class PlotPanelSetup:
    """Setup of an individual panel of a plot.

    See docstring of ``PlotSetup.create`` for a description of the parameters.

    """

    # Basics
    input_variable: str = "concentration"
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
        assert isinstance(self.dimensions, Dimensions)  # SR_DBG

        self._check_types()

        # Check input_variable
        choices = [
            "concentration",
            "deposition",
            "affected_area",
            "cloud_arrival_time",
            "cloud_departure_time",
        ]
        assert self.input_variable in choices, self.input_variable

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
            self.input_variable,
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
            "dimensions": self.dimensions.dict() if rec else self.dimensions,
        }

    def copy(self) -> "PlotPanelSetup":
        return self.create(self.dict())

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        dct = self.dict(rec=False)
        dims = dct.pop("dimensions")
        return tuple(list(dct.items()) + [("dimensions", dims.tuple())])

    def _check_types(self) -> None:
        for param, value in self.dict(rec=False).items():
            if param in ["dimensions", "ens_params"]:
                continue
            try:
                cast_field_value(type(self), param, value)
            except InvalidParameterValueError as e:
                raise ValueError(
                    f"param {param} has invalid value {sfmt(value)}"
                ) from e

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
        # SR_TMP <
        if len(panels) > 1:
            raise NotImplementedError("multipanel")
        # SR_TMP >
        self._panels = panels

    def __iter__(self) -> Iterator[PlotPanelSetup]:
        return iter(self._panels)

    def __len__(self) -> int:
        return len(self._panels)

    def __getattr__(self, name: str) -> Any:
        # SR_TMP <
        if len(self._panels) > 1:
            raise NotImplementedError("multipanel")
        assert len(self) == 1
        panel = self._panels[0]
        try:
            return getattr(panel, name)
        except AttributeError as e:
            raise e
        # SR_TMP >

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "PlotPanelSetupGroup":
        # SR_TMP <
        panel = PlotPanelSetup.create(params)
        return cls([panel])
        # SR_TMP >

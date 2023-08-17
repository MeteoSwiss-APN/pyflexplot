# pylint: disable=C0302  # too-many-lines
"""Plot setup and setup files."""
# Standard library
import dataclasses as dc
from pprint import pformat
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
from typing_extensions import Literal

# First-party
from srutils.dict import decompress_multival_dict
from srutils.exceptions import InvalidParameterValueError
from srutils.format import sfmt
from srutils.iter import resolve_indices
from srutils.str import join_multilines

# Local
from ..utils.summarize import summarizable
from ..utils.wrappers import cast_field_value


# SR_TMP <<< TODO cleaner solution
def is_dimensions_param(param: str, recursive: bool = False) -> bool:
    if recursive:
        raise NotImplementedError("recursive")
    return param.replace("dimensions.", "") in Dimensions.get_params()


# SR_TODO Clean up docstring -- where should format key hints go?
@dc.dataclass
class CoreDimensions:
    """Selected dimensions.

    See docstring of ``Setup.create`` (in module ``pyflexplot.setup``) for a
    description of the parameters.

    """

    level: Optional[int] = None
    nageclass: Optional[int] = None
    release: Optional[int] = None
    species_id: Optional[int] = None
    time: Optional[int] = None
    variable: Optional[str] = None
    multiplier: Optional[float] = None

    def __post_init__(self) -> None:
        # Check variable
        choices = [None, "concentration", "dry_deposition", "wet_deposition"]
        if self.variable not in choices:
            raise ValueError(f"variable '{self.variable}' not among {choices}")

    def dict(self) -> Dict[str, Any]:
        return dc.asdict(self)

    @classmethod
    def create(cls, params: Dict[str, Any]) -> "CoreDimensions":
        for param, value in params.items():
            if value == "*":
                value = None
            try:
                params[param] = cls.cast(param, value)
            except InvalidParameterValueError as e:
                raise ValueError(
                    f"cannot cast value of param '{param}' of type"
                    f" {type(value).__name__}: {sfmt(value)}"
                ) from e
        return cls(**params)

    # SR_TMP Identical to ModelSetup.cast
    @classmethod
    def cast(cls, param: str, value: Any, recursive: bool = False) -> Any:
        if recursive:
            raise ValueError(f"recursive cast not implemented for class {cls.__name__}")
        if value is None:
            return None
        return cast_field_value(cls, param, value)

    # SR_TMP Identical to CoreSetup.get_params and ModelSetup.get_params
    @classmethod
    def get_params(cls) -> List[str]:
        return list(cls.__dataclass_fields__)  # type: ignore  # pylint: disable=E1101


@summarizable(summarize=lambda self: self.dict())  # type: ignore
# pylint: disable=R0902  # too-many-instance-attributes
class Dimensions:
    """A collection of Dimensions objects."""

    def __init__(self, core: Optional[Sequence[CoreDimensions]] = None) -> None:
        """Create an instance of ``Dimensions``."""
        if core is None:
            core = [CoreDimensions()]
        assert core is not None  # mypy
        assert all(isinstance(obj, CoreDimensions) for obj in core)
        self._core: List[CoreDimensions] = list(core)

    @overload
    def get(
        self, param: str, *, unpack_single: Literal[False] = ...
    ) -> Tuple[Any, ...]:
        ...

    @overload
    def get(
        self, param: str, *, unpack_single: Literal[True] = ...
    ) -> Optional[Union[Any, Tuple[Any, ...]]]:
        ...

    def get(self, param, *, unpack_single=True):
        """Gather the value(s) of a parameter in compact form.

        The values are ordered, and duplicates and Nones are removed.
        Single values are returned directly, multiple values as a tuple.
        In absence of values, None is returned.

        Args:
            param: Name of parameter.

            unpack_single (optional): Return single values directly, rather than
                as a one-element tuple.

        """
        values: List[Any] = []
        for value in self.get_raw(param):
            if value is not None and value not in values:
                values.append(value)
        if len(values) == 0:
            values = [None]
        if len(values) == 1 and unpack_single:
            return next(iter(values))
        return tuple(sorted(values))

    def get_raw(self, param: str) -> Tuple[Any, ...]:
        """Gather the values of a parameter in raw form.

        The values are neither sorted, nor are duplicates or Nones removed, and
        a tuple is returned regardless of the number of values.

        """
        if param not in self.get_params():
            raise ValueError(param)
        values = []
        for core_dimension in self:
            value = getattr(core_dimension, param)
            values.append(value)
        return tuple(values)

    def set(self, param: str, value: Any) -> None:
        dct = self.dict()
        if isinstance(value, Sequence) and not isinstance(value, str):
            value = tuple(value)
        dct[param] = value
        self._core = list(type(self).create(dct))

    def update(self, other: Union["Dimensions", Mapping[str, Any]]) -> None:
        if isinstance(other, type(self)):
            other = other.dict()
        assert isinstance(other, Mapping)
        for param, value in other.items():
            self.set(param, value)

    def derive(self, params: Mapping[str, Any]) -> "Dimensions":
        """Derive a new ``Dimensions`` object with some changed parameters."""
        return type(self).create({**self.dict(), **params})

    def decompress(
        self,
        select: Optional[Collection[str]] = None,
        skip: Optional[Collection[str]] = None,
    ) -> List["Dimensions"]:
        """Create a dimensions object for each combination of list values.

        Args:
            select (optional): List of parameter names to select for
                decompression; all others will be skipped; parameters named in
                both ``select`` and ``dkip`` will be skipped.

            skip (optional): List of parameter names to skip; if they have list
                values, those are retained as such; parameters named in both
                ``skip`` and ``select`` will be skipped.

        """
        for param in select or []:
            if param not in self.get_params():
                raise ValueError(f"invalid param in select: {param}")
        for param in skip or []:
            if param not in self.get_params():
                raise ValueError(f"invalid param in skip: {param}")
        dcts = decompress_multival_dict(self.dict(), select=select, skip=skip)
        return list(map(self.create, dcts))

    # pylint: disable=R0912  # too-many-branches
    # pylint: disable=R0913  # too-many-arguments (>5)
    # pylint: disable=R0915  # too-many-statements
    # pylint: disable=W0201  # attribute-defined-outside-init
    def complete(
        self,
        raw_dimensions: Mapping[str, Any],
        species_ids: Sequence[int],
        plot_variable: str,
        inplace: bool = False,
        mode: Union[Literal["all"], Literal["first"]] = "all",
    ) -> Optional["Dimensions"]:
        """Complete unconstrained dimensions based on available indices."""
        _name_ = type(self).__name__

        mode_choices = ["all", "first"]
        if mode not in mode_choices:
            raise ValueError(
                f"invalid mode: {mode} (choices: {', '.join(mode_choices)})"
            )

        obj = self if inplace else self.copy()

        if obj.variable is None:
            if plot_variable in ["concentration", "dry_deposition", "wet_deposition"]:
                obj.variable = plot_variable
            elif plot_variable in ["cloud_arrival_time", "cloud_departure_time"]:
                obj.variable = "concentration"
            elif plot_variable == "tot_deposition":
                obj.variable = ("dry_deposition", "wet_deposition")
            elif plot_variable == "affected_area":
                obj.variable = ("concentration", "dry_deposition", "wet_deposition")
            else:
                raise NotImplementedError(
                    f"set {_name_}.variable for plot_variable '{plot_variable}'"
                )

        if obj.species_id is None:
            if mode == "all":
                obj.species_id = tuple(species_ids)
            elif mode == "first":
                obj.species_id = next(iter(species_ids))

        if obj.time is None:
            values = range(raw_dimensions["time"]["size"])
            if mode == "all":
                obj.time = tuple(values)
            elif mode == "first":
                obj.time = next(iter(values))
        else:
            try:
                # If index is negative, make it positive (subtract from end)
                obj.time = resolve_indices(
                    idcs=obj.get("time", unpack_single=False),  # type: ignore
                    n=raw_dimensions["time"]["size"],
                    force_in_range=True,
                )
            except ValueError as e:
                n = raw_dimensions["time"]["size"]
                raise Exception(f"invalid time index (n={n}): {obj.time}") from e

        if obj.level is None:
            if plot_variable == "concentration":
                if "level" in raw_dimensions:
                    values = range(raw_dimensions["level"]["size"])
                elif "height" in raw_dimensions:
                    values = range(raw_dimensions["height"]["size"])
                else:
                    raise Exception(
                        f"missing vertical dimensions: neither 'level' nor 'height'"
                        f" among dimensions ({', '.join(raw_dimensions)})"
                    )
                if mode == "all":
                    obj.level = tuple(values)
                elif mode == "first":
                    obj.level = next(iter(values))

        if obj.nageclass is None:
            if "nageclass" in raw_dimensions:
                values = range(raw_dimensions["nageclass"]["size"])
                if mode == "all":
                    obj.nageclass = tuple(values)
                elif mode == "first":
                    obj.nageclass = next(iter(values))

        if obj.release is None:
            # SR_TMP < TODO Try not to hardcode dimensions names here
            n1: int = raw_dimensions.get("numpoint", {}).get("size", 0)
            n2: int = raw_dimensions.get("noutrel", {}).get("size", 0)
            n3: int = raw_dimensions.get("pointspec", {}).get("size", 0)
            # SR_TMP >
            if any([n1, n2, n3]):
                values = range(max([n1, n2, n3]))
                if mode == "all":
                    obj.release = tuple(values)
                elif mode == "first":
                    obj.release = next(iter(values))

        return None if inplace else obj

    def dict(self) -> Dict[str, Optional[Union[int, Tuple[int, ...]]]]:
        """Return a compact dictionary representation.

        See method ``get`` for information of how the values of each
        parameter are compacted.

        """
        return {param: self.get(param) for param in self.get_params()}

    def raw_dict(self) -> Dict[str, Tuple[Any, ...]]:
        """Return a raw dictionary representation.

        The parameter values are unordered, with duplicates and Nones retained.

        """
        return {param: self.get_raw(param) for param in self.get_params()}

    def copy(self) -> "Dimensions":
        return self.create(self.dict())

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.dict().items())

    def __hash__(self) -> int:
        return hash(self.tuple())

    def __eq__(self, other) -> bool:
        if isinstance(other, dc._MISSING_TYPE):
            return False
        try:
            other_dict = other.dict()
        except AttributeError:
            other_dict = dict(other)
        return self.dict() == other_dict

    def __iter__(self) -> Iterator[CoreDimensions]:
        return iter(self._core)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.get(name)
        except ValueError:
            return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.get_params():
            self.set(name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        head = f"{type(self).__name__}"
        lines = [f"{param}={value}" for param, value in self.dict().items()]
        body = join_multilines(lines, indent=2)
        return f"{head}(\n{body}\n)"

    @classmethod
    def get_params(cls) -> List[str]:
        return CoreDimensions.get_params()

    @classmethod
    def create(
        cls,
        params: Union["Dimensions", Mapping[str, Any]],
        *,
        plot_variable: Optional[str] = None,
    ) -> "Dimensions":
        """Prepare parameters and create an instance of ``Dimensions``.

        Args:
            params: Parameters definint the ``Dimensions`` object.

            plot_variable (optional): Plot variable name from which the
                ``variable`` parameter is derived; mandatory if "variable" is
                missing in ``params``; takes precedence over a "variable" entry
                in ``params``.

        """
        loc_params: Dict[str, Any]
        if isinstance(params, cls):
            loc_params = params.dict()
        else:
            loc_params = dict(params)
        del params
        if plot_variable is not None:
            loc_params["variable"] = cls.derive_variable(plot_variable)
        elif "variable" not in loc_params:
            raise ValueError(
                "add 'variable' to params, or pass plot_variable:"
                f"\n{pformat(loc_params)}"
            )
        n_max = 1
        for param, value in loc_params.items():
            if not isinstance(value, Sequence) or isinstance(value, str):
                assert isinstance(loc_params, MutableMapping)  # mypy
                loc_params[param] = [value]
            else:
                n_max = max(n_max, len(value))
        core_dims_lst: List[CoreDimensions] = []
        for idx in range(n_max):
            core_params = {}
            for param, values in loc_params.items():
                assert isinstance(values, Sequence)  # mypy
                try:
                    core_params[param] = values[idx]
                except IndexError:
                    pass
            core_dims = CoreDimensions.create(core_params)
            core_dims_lst.append(core_dims)
        return cls(core_dims_lst)

    @classmethod
    def cast(cls, param: str, value: Any, recursive: bool = False) -> Any:
        """Cast a parameter to the appropriate type."""
        if recursive:
            raise ValueError(f"recursive cast not implemented for class {cls.__name__}")
        if isinstance(value, Sequence) and not isinstance(value, str):
            sub_values = []
            for sub_value in value:
                sub_values.append(cls.cast(param, sub_value))
            if len(sub_values) == 1:
                return next(iter(sub_values))
            return tuple(sub_values)
        return CoreDimensions.cast(param, value)

    @classmethod
    def merge(cls, objs: Sequence["Dimensions"]) -> "Dimensions":
        return cls([core_setup for obj in objs for core_setup in obj])

    @staticmethod
    def derive_variable(plot_variable: str) -> Union[str, Tuple[str, ...]]:
        if plot_variable in ["concentration", "dry_deposition", "wet_deposition"]:
            return plot_variable
        elif plot_variable == "tot_deposition":
            return ("dry_deposition", "wet_deposition")
        elif plot_variable == "affected_area":
            return ("concentration", "dry_deposition", "wet_deposition")
        elif plot_variable in ["cloud_arrival_time", "cloud_departure_time"]:
            return "concentration"
        raise ValueError(f"invalid plot_variable '{plot_variable}'")

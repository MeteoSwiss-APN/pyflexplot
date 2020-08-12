# -*- coding: utf-8 -*-
# pylint: disable=C0302  # too-many-lines
"""
Plot setup and setup files.
"""
# Standard library
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic import validator

# First-party
from srutils.iter import resolve_negative_indices
from srutils.str import join_multilines

# Local
from .utils.pydantic import cast_field_value
from .utils.pydantic import prepare_field_value
from .utils.summarize import summarizable


# SR_TODO Clean up docstring -- where should format key hints go?
# pylint: disable=E0213  # no-self-argument (validator)
class CoreDimensions(BaseModel):
    """Selected dimensions.

    Args:
        deposition_type: Type(s) of deposition. Part of the plot variable name
            that may be embedded in ``outfile`` with the format key
            '{variable}'. Choices: "none", "dry", "wet" (the latter may can be
            combined).

        level: Index/indices of vertical level (zero-based, bottom-up). To sum
            up multiple levels, combine their indices with '+'. Use the format
            key '{level}' to embed it in ``outfile``.

        nageclass: Index of age class (zero-based). Use the format key
            '{nageclass}' to embed it in ``outfile``.

        noutrel: Index of noutrel (zero-based). Use the format key
            '{noutrel}' to embed it in ``outfile``.

        numpoint: Index of release point (zero-based).

        species_id: Species id(s). To sum up multiple species, combine their
            ids with '+'. Use the format key '{species_id}' to embed it in
            ``outfile``.

        time: Time step indices (zero-based). Use the format key '{time}'
            to embed one in ``outfile``.

    """

    class Config:  # noqa
        # allow_mutation = False
        extra = "forbid"

    deposition_type: Optional[str] = None
    level: Optional[int] = None
    nageclass: Optional[int] = None
    noutrel: Optional[int] = None
    numpoint: Optional[int] = None
    species_id: Optional[int] = None
    time: Optional[int] = None

    @validator("deposition_type", always=True)
    def _check_deposition_type(cls, value: Optional[str]) -> Optional[str]:
        assert value in [None, "dry", "wet"]
        return value

    @classmethod
    def create(cls, params: Dict[str, Any]) -> "CoreDimensions":
        for param, value in params.items():
            field = cls.__fields__[param]
            try:
                params[param] = prepare_field_value(field, value, alias_none=["*"])
            except Exception as error:
                raise ValueError("invalid parameter value", param, value) from error
        try:
            return cls(**params)
        except ValidationError as error:
            msg = f"error creating {cls.__name__} object"
            error_params = next(iter(error.errors()))
            if error_params["type"] == "value_error.missing":
                param = next(iter(error_params["loc"]))
                msg += f": missing parameter: {param}"
            raise Exception(msg)

    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        return cast_field_value(cls, param, value)


@summarizable(summarize=lambda self: self.dict())  # type: ignore
# pylint: disable=R0902  # too-many-instance-attributes
class Dimensions:
    """A collection of Domensions objects."""

    params: Tuple[str, ...] = tuple(CoreDimensions.__fields__)

    def __init__(self, core: Optional[Sequence[CoreDimensions]] = None) -> None:
        """Create an instance of ``Dimensions``."""
        if core is None:
            core = [CoreDimensions()]
        assert core is not None  # mypy
        assert all(isinstance(obj, CoreDimensions) for obj in core)
        self._core: List[CoreDimensions] = list(core)

    def __repr__(self) -> str:
        head = f"{type(self).__name__}"
        lines = [f"{param}={value}" for param, value in self.dict().items()]
        body = join_multilines(lines, indent=2)
        return f"{head}(\n{body}\n)"

    @classmethod
    def create(cls, params: Union["Dimensions", Mapping[str, Any]]) -> "Dimensions":
        if isinstance(params, cls):
            params = params.dict()
        else:
            assert isinstance(params, Mapping)  # mypy
            params = cast(MutableMapping, params)
            params = dict(**params)
        n_max = 1
        for param, value in params.items():
            if not isinstance(value, Sequence) or isinstance(value, str):
                assert isinstance(params, MutableMapping)  # mypy
                params[param] = [value]
            else:
                n_max = max(n_max, len(value))
        core_dims_lst: List[CoreDimensions] = []
        for idx in range(n_max):
            core_params = {}
            for param, values in params.items():
                assert isinstance(values, Sequence)  # mypy
                try:
                    core_params[param] = values[idx]
                except IndexError:
                    pass
            core_dims = CoreDimensions.create(core_params)
            core_dims_lst.append(core_dims)
        return cls(core_dims_lst)

    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        """Cast a parameter to the appropriate type."""
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

    def derive(self, params: Mapping[str, Any]) -> "Dimensions":
        """Derive a new ``Dimensions`` object with some changed parameters."""
        return type(self).create({**self.dict(), **params})

    def get(
        self, param: str, *, unpack_single: bool = True
    ) -> Optional[Union[Any, Tuple[Any, ...]]]:
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
        if param not in self.params:
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

    def dict(self) -> Dict[str, Optional[Union[int, Tuple[int, ...]]]]:
        """Return a compact dictionary representation.

        See method ``get`` for information of how the values of each
        parameter are compacted.

        """
        return {param: self.get(param) for param in self.params}

    def raw_dict(self) -> Dict[str, Tuple[Any, ...]]:
        """Return a raw dictionary representation.

        The parameter values are unordered, with duplicates and Nones retained.

        """
        return {param: self.get_raw(param) for param in self.params}

    # pylint: disable=R0912  # too-many-branches
    def complete(
        self, meta_data: Mapping[str, Any], input_variable: str, inplace: bool = False
    ) -> Optional["Dimensions"]:
        """Complete unconstrained dimensions based on available indices."""
        obj = self if inplace else self.copy()

        raw_dimensions: Mapping[str, Mapping[str, Any]] = meta_data["dimensions"]

        if obj.time is None:
            obj.time = tuple(range(raw_dimensions["time"]["size"]))

        # Make negative (end-relative) time indices positive (absolute)
        obj.time = resolve_negative_indices(
            idcs=obj.get("time", unpack_single=False),  # type: ignore
            n=raw_dimensions["time"]["size"],
        )

        if obj.level is None:
            if input_variable == "concentration":
                if "level" in raw_dimensions:
                    obj.level = tuple(range(raw_dimensions["level"]["size"]))

        if obj.deposition_type is None:
            if input_variable == "deposition":
                obj.deposition_type = ("dry", "wet")

        if obj.species_id is None:
            obj.species_id = meta_data["derived"]["species_ids"]

        if obj.nageclass is None:
            if "nageclass" in raw_dimensions:
                obj.nageclass = tuple(range(raw_dimensions["nageclass"]["size"]))

        if obj.noutrel is None:
            if "noutrel" in raw_dimensions:
                obj.noutrel = tuple(range(raw_dimensions["noutrel"]["size"]))

        if obj.numpoint is None:
            if "numpoint" in raw_dimensions:
                obj.numpoint = tuple(range(raw_dimensions["numpoint"]["size"]))

        return None if inplace else obj

    def __eq__(self, other) -> bool:
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
        if name in self.params:
            self.set(name, value)
        else:
            super().__setattr__(name, value)

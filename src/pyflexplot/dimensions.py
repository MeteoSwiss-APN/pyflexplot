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

# First-party
from srutils.str import join_multilines

# Local
from .pydantic import cast_field_value
from .pydantic import prepare_field_value


# SR_TODO Clean up docstring -- where should format key hints go?
class CoreDimensions(BaseModel):
    """Selected dimensions.

    Args:
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

    nageclass: Optional[int] = None
    noutrel: Optional[int] = None
    numpoint: Optional[int] = None
    species_id: Optional[int] = None
    time: Optional[int] = None
    level: Optional[int] = None

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

    @classmethod
    def create(cls, params: Union["Dimensions", Mapping[str, Any]]) -> "Dimensions":
        if isinstance(params, cls):
            params = params.compact_dict()
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
        return type(self).create({**self.compact_dict(), **params})

    def __repr__(self) -> str:
        head = f"{type(self).__name__}"
        lines = [f"{param}={value}" for param, value in self.compact_dict().items()]
        body = join_multilines(lines, indent=2)
        return f"{head}(\n{body}\n)"

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return self.compact_dict() == other.compact_dict()
        raise NotImplementedError(
            f"comparison of {type(self).__name__} and {type(other).__name__} objects"
        )

    def __iter__(self) -> Iterator[CoreDimensions]:
        return iter(self._core)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.get_compact(name)
        except ValueError:
            return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.params:
            self.set(name, value)
        else:
            super().__setattr__(name, value)

    def update(self, other: Union["Dimensions", Mapping[str, Any]]) -> None:
        if isinstance(other, type(self)):
            other = other.compact_dict()
        assert isinstance(other, Mapping)
        for param, value in other.items():
            self.set(param, value)

    def set(self, param: str, value: Optional[Union[int, Sequence[int]]]) -> None:
        dct = self.compact_dict()
        if isinstance(value, Sequence):
            value = tuple(value)
        dct[param] = value
        self._core = list(type(self).create(dct))

    def get_compact(self, param: str) -> Optional[Union[int, Tuple[int, ...]]]:
        """Gather the value(s) of a parameter in compact form.

        The values are ordered, and duplicates and Nones are removed.
        Single values are returned directly, multiple values as a tuple.
        In absence of values, None is returned.

        """
        values: List[int] = []
        for value in self.get_raw(param):
            if value is not None and value not in values:
                values.append(value)
        if len(values) == 0:
            return None
        elif len(values) == 1:
            return next(iter(values))
        else:
            return tuple(sorted(values))

    def get_raw(self, param: str) -> Tuple[Optional[int], ...]:
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

    def compact_dict(self) -> Dict[str, Optional[Union[int, Tuple[int, ...]]]]:
        """Return a compact dictionary representation.

        See method ``get_compact`` for information of how the values of each
        parameter are compacted.

        """
        return {param: self.get_compact(param) for param in self.params}

    def raw_dict(self) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return a raw dictionary representation.

        The parameter values are unordered, with duplicates and Nones retained.

        """
        return {param: self.get_raw(param) for param in self.params}

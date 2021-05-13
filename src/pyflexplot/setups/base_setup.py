"""Base class for setup classes.

Note that this is design is not clean in the sense that the interfaces of the
subclasses of BaseSetup differ in their attributes, which requires the
occasional isinstance assert for the subclass type to satisfy mypy.

"""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Mapping
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

# First-party
from srutils.dict import merge_dicts
from srutils.format import nested_repr

# Local
from ..utils.wrappers import cast_field_value

SetupT = TypeVar("SetupT", bound="BaseSetup")


class BaseSetup:
    # pylint: disable=W0613  # unused-argument (args, kwargs)
    def __init__(self: SetupT, *args: Any, **kwargs: Any) -> None:
        """Cannot create instances of ``BaseSetup``, only its subclasses."""
        self.__dataclass_fields__: Dict[str, Any] = {}
        raise NotImplementedError(f"{type(self).__name__}.__init__")

    def copy(self: SetupT) -> SetupT:
        return self.create(self.dict())

    def derive(self: SetupT, params: Dict[str, Any]) -> SetupT:
        return type(self).create(merge_dicts(self.dict(), params, overwrite_seqs=True))

    def dict(self: SetupT, rec: bool = False) -> Dict[str, Any]:
        if rec:
            raise NotImplementedError("rec=T")
        return dc.asdict(self)

    def tuple(self: SetupT) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.dict().items())

    def __eq__(self: SetupT, other: Any) -> bool:
        if isinstance(other, dc._MISSING_TYPE):
            return False
        try:
            other_dict = other.dict()
        except AttributeError:
            try:
                other_dict = dict(other)  # type: ignore
            except TypeError:
                try:
                    other_dict = dc.asdict(other)
                except TypeError:
                    return False
        return self.dict() == other_dict

    def __len__(self: SetupT) -> int:
        return len(self.dict())

    def __repr__(self: SetupT) -> str:  # type: ignore
        return nested_repr(self)

    @classmethod
    def cast(cls: Type[SetupT], param: str, value: Any, recursive: bool = False) -> Any:
        if recursive:
            raise ValueError(f"recursive cast not implemented for class {cls.__name__}")
        return cast_field_value(cls, param, value)

    @classmethod
    def cast_many(
        cls: Type[SetupT],
        params: Union[Collection[Tuple[str, Any]], Mapping[str, Any]],
        recursive: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(params, Mapping):
            params_dct: Dict[str, Any] = {}
            for param, value in params:
                if param in params_dct:
                    raise ValueError("duplicate parameter", param)
                params_dct[param] = value
            return cls.cast_many(params_dct, recursive)
        params_cast = {}
        for param, value in params.items():
            params_cast[param] = cls.cast(param, value, recursive)
        return params_cast

    @classmethod
    def create(cls: Type[SetupT], params: Mapping[str, Any]) -> SetupT:
        params = cls._create_mod_params_pre_cast(params)
        params = cls.cast_many(params)
        params = cls._create_mod_params_post_cast(params)
        return cls(**params)

    @classmethod
    def get_params(cls: Type[SetupT]) -> List[str]:
        return list(cls.__dataclass_fields__)  # type: ignore  # pylint: disable=E1101

    @classmethod
    def _create_mod_params_pre_cast(
        cls: Type[SetupT], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Modify params in ``create`` before typecasting."""
        return dict(params)

    @classmethod
    def _create_mod_params_post_cast(
        cls: Type[SetupT], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Modify params in ``create`` after typecasting."""
        return dict(params)

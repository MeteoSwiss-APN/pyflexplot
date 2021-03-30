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
from typing import Union

# First-party
from srutils.dataclasses import cast_field_value
from srutils.dict import merge_dicts
from srutils.format import nested_repr


class BaseSetup:
    # pylint: disable=W0613  # unused-argument (args, kwargs)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Cannot create instances of ``BaseSetup``, only its subclasses."""
        self.__dataclass_fields__: Dict[str, Any] = {}
        raise NotImplementedError(f"{type(self).__name__}.__init__")

    def copy(self) -> "BaseSetup":
        return self.create(self.dict())

    def derive(self, params: Dict[str, Any]) -> "BaseSetup":
        return type(self).create(merge_dicts(self.dict(), params, overwrite_seqs=True))

    def dict(self, rec: bool = False) -> Dict[str, Any]:
        if rec:
            raise NotImplementedError("rec=T")
        return dc.asdict(self)

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.dict().items())

    def __hash__(self) -> int:
        return hash(self.tuple())

    def __repr__(self) -> str:  # type: ignore
        return nested_repr(self)

    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        return cast_field_value(
            cls,
            param,
            value,
            auto_wrap=True,
            bool_mode="intuitive",
            timedelta_unit="hours",
            unpack_str=False,
        )

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

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "BaseSetup":
        params = cls._create_mod_params_pre_cast(params)
        params = cls.cast_many(params)
        params = cls._create_mod_params_post_cast(params)
        return cls(**params)

    @classmethod
    def get_params(cls) -> List[str]:
        return list(cls.__dataclass_fields__)  # type: ignore  # pylint: disable=E1101

    @classmethod
    def _create_mod_params_pre_cast(cls, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Modify params in ``create`` before typecasting."""
        return dict(params)

    @classmethod
    def _create_mod_params_post_cast(cls, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Modify params in ``create`` after typecasting."""
        return dict(params)

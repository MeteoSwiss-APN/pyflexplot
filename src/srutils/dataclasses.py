"""Dataclasses utilities."""
# Standard library
import dataclasses as dc
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import get_type_hints
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

# First-party
from srutils.datetime import derive_datetime_fmt
from srutils.exceptions import IncompatibleTypesError
from srutils.exceptions import InvalidParameterNameError
from srutils.exceptions import InvalidParameterValueError
from srutils.exceptions import UnsupportedTypeError
from srutils.format import sfmt
from srutils.str import split_outside_parens

DataclassT = TypeVar("DataclassT")


def asdict(obj: Any, *, shallow: bool = False) -> Dict[str, Any]:
    """Like ``dataclasses.asdict`` but with option for shallow copy."""
    if shallow:
        return asdict_shallow(obj)
    try:
        return dc.asdict(obj)
    except TypeError as e:
        raise e
    except NotImplementedError as e:
        if "can not be copied" not in str(e):
            raise e
        raise TypeError(
            f"asdict with depcopy of {type(obj).__name__} instance failed"
            "; consider passing shallow=True"
        ) from e


def asdict_shallow(obj: Any) -> Dict[str, Any]:
    """Like ``dateclasses.asdict`` but with shallow instead of deep copy."""
    try:
        fields = obj.__dataclass_fields__
    except AttributeError as e:
        raise TypeError(f"obj of type {type(obj).__name__} is not a dataclass") from e
    else:
        return {name: getattr(obj, name) for name in fields.keys()}


def get_dataclass_fields(obj: DataclassT) -> List[str]:
    if not dc.is_dataclass(obj):
        raise ValueError(f"expected dataclass, not {type(obj).__name__}: {obj}")
    # Use getattr to prevent mypy from complaining
    # pylint: disable=E1101  # no-member
    return getattr(obj, "__dataclass_fields__")


def dataclass_merge(
    objs: Sequence[DataclassT], reduce_equal: bool = False
) -> DataclassT:
    """Merge multiple dataclass objects by merging their values into tuples."""
    obj0 = objs[0]
    cls = type(obj0)
    objs1 = objs[1:]
    if not dc.is_dataclass(obj0):
        raise ValueError(f"first obj is a {cls.__name__}, not a dataclass: {obj0}")
    for obj in objs1:
        if not isinstance(obj, cls):
            raise ValueError(f"classes differ: {cls.__name__} != {type(obj).__name__}")
    kwargs: Dict[str, Any] = {}
    for param in get_dataclass_fields(cls):
        if dc.is_dataclass(type(getattr(obj0, param))):
            kwargs[param] = dataclass_merge(
                [getattr(obj, param) for obj in objs], reduce_equal=reduce_equal
            )
        else:
            values = tuple(getattr(obj, param) for obj in objs)
            if reduce_equal and all(value == values[0] for value in values[1:]):
                kwargs[param] = values[0]
            else:
                kwargs[param] = values
    return cls(**kwargs)  # type: ignore


def dataclass_repr(
    obj: DataclassT,
    indent: int = 2,
    *,
    nested: int = 0,
    fmt: Callable[[Any], str] = repr,
) -> str:
    """Create string representation with one argument per line."""
    body: List[str] = []
    for field in get_dataclass_fields(obj):
        value = getattr(obj, field)
        body.append(f"{field}={fmt(value)},")
    head = f"{type(obj).__name__}("
    foot = f"\n{' ' * indent * nested})"
    return f"\n{' ' * indent * (nested + 1)}".join([head] + body) + foot


def cast_field_value(cls: Type, name: str, value: Any, **kwargs: Any) -> Any:
    """Cast a value to the type of a dataclass field.

    Args:
        cls: A dataclass.

        name: Name of dataclass field.

        value: Value to cast to type of ``field``.

        kwargs (optional): Keyword arguments passed on to ``cast_value``.

    """
    try:
        type_ = get_type_hints(cls)[name]
    except KeyError as e:
        raise InvalidParameterNameError(name) from e
    try:
        return cast_value(type_, value, **kwargs)
    except Exception as e:
        exc: Type[Exception]
        if isinstance(e, IncompatibleTypesError):
            exc = InvalidParameterValueError
            msg = f"value incompatible with type {type_}"
        elif isinstance(e, UnsupportedTypeError):
            exc = InvalidParameterNameError
            msg = f"type {type_} not supported"
        else:
            raise e
        raise exc(f"{msg}: {cls.__name__}.{name} = {sfmt(value)}") from e


# pylint: disable=R0911  # too-many-return-statements (>6)
# pylint: disable=R0912  # too-many-branches (>12)
# pylint: disable=R0914  # too-many-variables (>15)
# pylint: disable=R0915  # too-many-statements (>50)
def cast_value(
    type_: Union[Type, str],
    value: Any,
    *,
    auto_wrap: bool = False,
    bool_mode: str = "native",
    datetime_fmt: Union[str, Tuple[str, ...]] = "auto",
    timedelta_unit: str = "days",
    unpack_str: bool = True,
) -> Any:
    """Cast a value to a type.

    Args:
        type_: Type to cast ``value`` to.

        value: Value to cast to ``type_``.

        auto_wrap (optional): Cast scalars to collection types by wrapping them
            in a list before casting.

        bool_mode (optional): Whether cast the string "False" to True ('native')
            or to False ('intuitive').

        datetime_fmt (optional): One or more format strings to convert datetime
            ints or strings to datetime objects; for 'auto', the format is
            derived from the number of digits assuming a format of the form
            "%Y[%m[%d[%H[%M[%s]]]]]".

        timedelta_unit (optional): Unit of timedelta values: 'weeks', 'days',
            'hours', 'seconds', 'milliseconds' or 'microseconds'.

        unpack_str (optional): Unpack strings, i.e., treat a string as a
            sequence of one-character strings, as is their default behavior.

    """
    try:
        type_ = type_.__name__  # type: ignore
    except AttributeError:
        type_ = str(type_)
    if bool_mode not in ["native", "intuitive"]:
        raise ValueError(f"bool_mode neither 'native' nor 'intuitive': {bool_mode}")
    timedelta_unit_choices = [
        "weeks",
        "days",
        "hours",
        "seconds",
        "milliseconds",
        "microseconds",
    ]
    if timedelta_unit not in timedelta_unit_choices:
        raise ValueError(
            f"invalid timedelta_unit '{timedelta_unit}'; choices: "
            + ", ".join(map("'{}'".format, timedelta_unit_choices))
        )

    # Bundle keyword arguments to pass to recursive calls
    kwargs: Dict[str, Any] = {
        "auto_wrap": auto_wrap,
        "bool_mode": bool_mode,
        "datetime_fmt": datetime_fmt,
        "timedelta_unit": timedelta_unit,
        "unpack_str": unpack_str,
    }

    def error(value: Any, type_: str, msg: str = "") -> Exception:
        msg = f": {msg}" if msg else ""
        return IncompatibleTypesError(
            f"type '{type(value).__name__}' incompatible with '{type_}'{msg}"
        )

    def has_same_type(value: Any, type_: str) -> bool:
        """Check whether the type of a value matches the type string."""
        try:
            equiv_type = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "NoneType": type(None),
                "tuple": tuple,
                "typing.Tuple": tuple,
                "list": list,
                "typing.List": list,
                "datetime": datetime,
                "datetime.datetime": datetime,
                "timedelta": timedelta,
                "datetime.timedelta": timedelta,
            }[type_.split("[")[0]]
        except KeyError as e:
            raise NotImplementedError(f"type check for '{type_}'") from e
        else:
            return isinstance(value, equiv_type)

    def prepare_wrapped_value(
        value: Any, type_: str, cls: Optional[Callable[[Iterable], Sequence]] = None
    ) -> Sequence[Any]:
        if isinstance(value, Sequence) and not isinstance(value, str):
            if cls is not None:
                value = cls(value)
        elif isinstance(value, str):
            if cls is None:
                pass
            elif auto_wrap:
                value = [value]
            elif unpack_str:
                value = list(value)
                if cls is not None:
                    value = cls(value)
            else:
                msg = (
                    "auto-wrap strings with auto_wrap=True"
                    " or unpack them with unpack_str=True"
                )
                raise error(value, type_, msg)
        else:
            if auto_wrap:
                value = [value]
                if cls is not None:
                    value = cls(value)
            else:
                msg = "auto-wrap non-sequences with auto_wrap=True"
                raise error(value, type_, msg)
        return value

    if type_ == "typing.Any":
        return value

    elif type_ == "NoneType":
        if isinstance(value, type(None)) or value == "None":
            return None
        raise error(value, type_)

    elif type_ == "bool":
        if isinstance(value, str):
            if bool_mode == "intuitive" and value == "False":
                return False
            return bool(value)
        elif isinstance(value, Collection):
            raise error(value, type_, "is a collection")
        else:
            return bool(value)

    elif type_ == "str":
        if isinstance(value, Collection) and not isinstance(value, str):
            raise error(value, type_, "is a collection")
        return str(value)

    elif type_ in ["int", "float"]:
        try:
            return {"int": int, "float": float}[type_](value)
        except (TypeError, ValueError) as e:
            raise error(value, type_) from e

    elif type_.startswith("typing.Union["):
        inner_types = split_outside_parens(
            type_[len("typing.Union[") : -len("]")], ", ", parens="[]"
        )
        if "NoneType" in inner_types:
            inner_types.remove("NoneType")
            inner_types.insert(0, "NoneType")
        for inner_type in inner_types:
            if has_same_type(value, inner_type):
                return value
        for inner_type in inner_types:
            try:
                return cast_value(inner_type, value, **kwargs)
            except IncompatibleTypesError:
                pass
        raise error(value, type_, f"no compatible inner type: {inner_types}")

    elif type_ in ["datetime", "datetime.datetime"]:
        if isinstance(datetime_fmt, str):
            datetime_fmt = (datetime_fmt,)  # type: ignore
        assert isinstance(datetime_fmt, Sequence)  # mypy
        assert not isinstance(datetime_fmt, str)  # mypy
        for fmt in datetime_fmt:
            if fmt == "auto":
                try:
                    fmt = derive_datetime_fmt(value)
                except ValueError as e:
                    raise error(value, type_, "cannot derive datetime_fmt") from e
            try:
                return datetime.strptime(str(value), fmt)
            except ValueError:
                pass
        raise error(value, type_, f"no compatible datetime_fmt: {datetime_fmt}")

    elif type_ in ["timedelta", "datetime.timedelta"]:
        try:
            return timedelta(**{timedelta_unit: value})
        except TypeError as e:
            raise error(value, type_, f"timedelta_unit: {timedelta_unit}") from e

    if type_ in ["tuple", "typing.Tuple"]:
        return prepare_wrapped_value(value, type_, tuple)

    elif type_.startswith("typing.Tuple["):
        value = prepare_wrapped_value(value, type_, tuple)
        if type_.endswith(", ...]"):
            inner_type = type_[len("typing.Tuple[") : -len(", ...]")]
            inner_values = [
                cast_value(inner_type, inner_value, **kwargs) for inner_value in value
            ]
            return tuple(inner_values)
        else:
            inner_types = split_outside_parens(
                type_[len("typing.Tuple[") : -len("]")], ", ", parens="[]"
            )
            if len(value) != len(inner_types):
                raise error(value, type_, "wrong length")
            inner_values = [
                cast_value(inner_type, inner_value, **kwargs)
                for inner_type, inner_value in zip(inner_types, value)
            ]
            return tuple(inner_values)

    if type_ in ["list", "typing.List"]:
        return prepare_wrapped_value(value, type_, list)

    elif type_.startswith("typing.List["):
        value = prepare_wrapped_value(value, type_, list)
        inner_type = type_[len("typing.List[") : -len("]")]
        return [cast_value(inner_type, inner_value, **kwargs) for inner_value in value]

    elif type_ == "typing.Sequence":
        return prepare_wrapped_value(value, type_)

    elif type_.startswith("typing.Sequence["):
        value = prepare_wrapped_value(value, type_)
        inner_type = type_[len("typing.Sequence[") : -len("]")]
        cls: Callable[[Iterable], Sequence] = (
            list if isinstance(value, str) else type(value)  # type: ignore
        )
        return cls(
            [cast_value(inner_type, inner_value, **kwargs) for inner_value in value]
        )

    else:
        raise UnsupportedTypeError(f"{type_}")

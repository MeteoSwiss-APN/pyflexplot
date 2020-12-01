"""Tests for function ``pyflexplot.utils.dataclasses.cast_field_value``."""
# Standard library
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import pytest

# First-party
from pyflexplot.utils.dataclasses import cast_field_value
from pyflexplot.utils.exceptions import InvalidParameterNameError
from pyflexplot.utils.exceptions import InvalidParameterValueError

# Abbreviations
T = True
F = False


@dataclass
class Config:
    param: str
    val: Any
    sol: Any = None
    kw: Dict = field(default_factory=dict)


@dataclass
class Scalars:
    str_: str
    int_: int
    float_: float
    bool_: bool


@pytest.mark.parametrize(
    "cfg",
    [
        Config("str_", "foo", "foo"),  # [cfg0]
        Config("str_", 9999, "9999"),  # [cfg1]
        Config("str_", 3.14, "3.14"),  # [cfg2]
        Config("str_", None, "None"),  # [cfg3]
        Config("int_", 99999, 99999),  # [cfg4]
        Config("int_", 999.999, 999),  # [cfg5]
        Config("int_", "9999", 9999),  # [cfg6]
        Config("int_", "00999", 999),  # [cfg7]
        Config("float_", 9999, 9999.0),  # [cfg8]
        Config("float_", 999.999, 999.999),  # [cfg9]
        Config("float_", "9999", 9999.0),  # [cfg10]
        Config("float_", "099.9", 99.9),  # [cfg11]
        Config("float_", "00999", 999.0),  # [cfg12]
        Config("bool_", T, T),  # [cfg13]
        Config("bool_", F, F),  # [cfg14]
        Config("bool_", 1, T),  # [cfg15]
        Config("bool_", 0, F),  # [cfg16]
        Config("bool_", None, F),  # [cfg17]
        Config("bool_", "True", T),  # [cfg18]
        Config("bool_", "False", T),  # [cfg19]
        Config("bool_", "1", T),  # [cfg20]
        Config("bool_", "0", T),  # [cfg21]
        Config("bool_", "", F),  # [cfg22]
        Config("bool_", "None", T),  # [cfg23]
    ],
)
def test_scalars_success(cfg):
    res = cast_field_value(Scalars, cfg.param, cfg.val, **cfg.kw)
    assert res == cfg.sol
    assert isinstance(res, type(cfg.sol))


@pytest.mark.parametrize(
    "cfg",
    [
        Config("asdf", "foo", InvalidParameterNameError),  # [cfg0]
        Config("int_", "foo"),  # [cfg1]
        Config("int_", None),  # [cfg2]
        Config("int_", "999.9"),  # [cfg3]
        Config("int_", "09.9"),  # [cfg4]
        Config("int_", ""),  # [cfg5]
        Config("float_", "foo"),  # [cfg6]
        Config("bool_", []),  # [cfg7]
        Config("str_", ["a", "b"]),  # [cfg8]
    ],
)
def test_scalars_fail(cfg):
    with pytest.raises(cfg.sol or InvalidParameterValueError):
        cast_field_value(Scalars, cfg.param, cfg.val, **cfg.kw)


def test_scalars_bool_mode():
    def cast_bool(value, mode):
        return cast_field_value(Scalars, "bool_", value, bool_mode=mode)

    assert cast_bool("True", "native") is True
    assert cast_bool("False", "native") is True
    assert cast_bool("True", "intuitive") is True
    assert cast_bool("False", "intuitive") is False


@dataclass
class OptionalScalars:
    str_: Optional[str]
    int_: Optional[int]
    float_: Optional[float]
    bool_: Optional[bool]


@pytest.mark.parametrize(
    "cfg",
    [
        Config("str_", "foo", "foo"),  # [cfg0]
        Config("str_", 9999, "9999"),  # [cfg1]
        Config("str_", 3.14, "3.14"),  # [cfg2]
        Config("str_", None, None),  # [cfg3]
        Config("int_", 99999, 99999),  # [cfg4]
        Config("int_", 999.999, 999),  # [cfg5]
        Config("int_", "9999", 9999),  # [cfg6]
        Config("int_", "00999", 999),  # [cfg7]
        Config("float_", 9999, 9999.0),  # [cfg8]
        Config("float_", 999.999, 999.999),  # [cfg9]
        Config("float_", "9999", 9999.0),  # [cfg10]
        Config("float_", "099.9", 99.9),  # [cfg11]
        Config("float_", "00999", 999.0),  # [cfg12]
        Config("bool_", T, T),  # [cfg13]
        Config("bool_", F, F),  # [cfg14]
        Config("bool_", 1, T),  # [cfg15]
        Config("bool_", 0, F),  # [cfg16]
        Config("bool_", None, None),  # [cfg17]
        Config("bool_", "True", T),  # [cfg18]
        Config("bool_", "False", T),  # [cfg19]
        Config("bool_", "1", T),  # [cfg20]
        Config("bool_", "0", T),  # [cfg21]
        Config("bool_", "", F),  # [cfg22]
        Config("bool_", "None", None),  # [cfg23]
    ],
)
def test_optional_scalars_success(cfg):
    res = cast_field_value(OptionalScalars, cfg.param, cfg.val, **cfg.kw)
    assert res == cfg.sol
    assert isinstance(res, type(cfg.sol))


@pytest.mark.parametrize(
    "cfg",
    [
        Config("int_", "foo"),  # [cfg0]
        Config("int_", "999.9"),  # [cfg1]
        Config("int_", "09.9"),  # [cfg2]
        Config("int_", ""),  # [cfg3]
        Config("float_", "foo"),  # [cfg4]
    ],
)
def test_optional_scalars_fail(cfg):
    with pytest.raises(cfg.sol or InvalidParameterValueError):
        cast_field_value(OptionalScalars, cfg.param, cfg.val, **cfg.kw)


@dataclass
class Tuples:
    strs: Tuple[str, ...]
    ints: Tuple[int, ...]
    floats: Tuple[float, ...]
    bools: Tuple[bool, ...]
    opt_ints: Tuple[Optional[int], ...]
    one_str: Tuple[str]
    two_ints: Tuple[int, int]
    three_floats: Tuple[float, float, float]
    four_bools: Tuple[bool, bool, bool, bool]
    mixed: Tuple[str, int, float, bool]


@pytest.mark.parametrize(
    "cfg",
    [
        Config("strs", tuple(), tuple()),  # [cfg0]
        Config("strs", ["x"], ("x",)),  # [cfg1]
        Config("strs", ("foo", "bar"), ("foo", "bar")),  # [cfg2]
        Config("strs", ["x", "y", "z"], ("x", "y", "z")),  # [cfg3]
        Config("strs", (1, 2.2, None), ("1", "2.2", "None")),  # [cfg4]
        Config("ints", [], tuple()),  # [cfg5]
        Config("ints", (0,), (0,)),  # [cfg6]
        Config("ints", (0, 1), (0, 1)),  # [cfg7]
        Config("ints", [8.8, 16, 32.3, "64"], (8, 16, 32, 64)),  # [cfg8]
        Config("floats", tuple(), tuple()),  # [cfg9]
        Config("floats", (1.0,), (1.0,)),  # [cfg10]
        Config("floats", (1, "2", "3.0"), (1.0, 2.0, 3.0)),  # [cfg11]
        Config("bools", tuple(), tuple()),  # [cfg12]
        Config("bools", (T,), (T,)),  # [cfg13]
        Config("bools", (F, 1, "False"), (F, T, T)),  # [cfg14]
        Config("opt_ints", (None, 2.0, "4"), (None, 2, 4)),  # [cfg15]
        Config("one_str", (None,), ("None",)),  # [cfg16]
        Config("two_ints", ["4", 6.6], (4, 6)),  # [cfg17]
        Config("three_floats", (2, "4", 6.0), (2.0, 4.0, 6.0)),  # [cfg18]
        Config("four_bools", (None, "False", 0.0, 1), (F, T, F, T)),  # [cfg19]
        Config("mixed", [F, 6.8, 32, "64"], ("False", 6, 32.0, True)),  # [cfg20]
    ],
)
def test_tuples_success(cfg):
    res = cast_field_value(Tuples, cfg.param, cfg.val, **cfg.kw)
    assert isinstance(res, type(cfg.sol))
    assert len(res) == len(cfg.sol)
    assert all([isinstance(r, type(s)) for r, s in zip(res, cfg.sol)])
    assert res == cfg.sol


@pytest.mark.parametrize(
    "cfg",
    [
        Config("strs", [[None]]),  # [cfg0]
    ],
)
def test_tuples_fail(cfg):
    with pytest.raises(cfg.sol or InvalidParameterValueError):
        cast_field_value(Tuples, cfg.param, cfg.val, **cfg.kw)


@pytest.mark.parametrize(
    "cfg",
    [
        Config("strs", 1, ("1",)),  # [cfg0]
        Config(
            "ints",
            "004",
            (
                0,
                0,
                4,
            ),
        ),  # [cfg1]
        Config("ints", "004", (4,), kw={"unpack_str": False}),  # [cfg2]
        Config("floats", "3", (3.0,)),  # [cfg3]
        Config(
            "bools",
            "False",
            (
                T,
                T,
                T,
                T,
                T,
            ),
        ),  # [cfg4]
        Config("bools", "False", (T,), kw={"unpack_str": False}),  # [cfg5]
    ],
)
def test_tuples_auto_wrap(cfg):
    res = cast_field_value(Tuples, cfg.param, cfg.val, auto_wrap=True, **cfg.kw)
    assert isinstance(res, type(cfg.sol))
    assert len(res) == len(cfg.sol)
    assert all([isinstance(r, type(s)) for r, s in zip(res, cfg.sol)])
    assert res == cfg.sol


@dataclass
class OptionalTuples:
    str_: Optional[Tuple[str]]
    strs: Optional[Tuple[str, ...]]
    int_: Optional[Tuple[int]]
    two_floats: Optional[Tuple[float, float]]
    bools: Optional[Tuple[bool, ...]]


@pytest.mark.parametrize(
    "cfg",
    [
        Config("str_", ["foo"], ("foo",), kw={"unpack_str": False}),  # [cfg0]
        Config("str_", None, None),  # [cfg1]
    ],
)
def test_optional_tuples_success(cfg):
    res = cast_field_value(OptionalTuples, cfg.param, cfg.val, **cfg.kw)
    if isinstance(cfg.sol, Sequence):
        assert isinstance(res, type(cfg.sol))
        assert len(res) == len(cfg.sol)
        assert all([isinstance(r, type(s)) for r, s in zip(res, cfg.sol)])
        assert res == cfg.sol
    elif cfg.sol is None:
        assert res == cfg.sol


@pytest.mark.parametrize(
    "cfg",
    [
        Config("str_", [[None]]),  # [cfg0]
    ],
)
def test_optional_tuples_fail(cfg):
    with pytest.raises(cfg.sol or InvalidParameterValueError):
        cast_field_value(OptionalTuples, cfg.param, cfg.val)


@dataclass
class Unions:
    str_int: Union[str, int]
    int_str: Union[int, str]
    str_tup_strs: Union[str, Tuple[str, ...]]
    tup_ints_tup_strs: Union[Tuple[int, ...], Tuple[str, ...]]
    tup_strs_tup_ints: Union[Tuple[str, ...], Tuple[int, ...]]
    float_bool: Union[float, bool]
    bool_float: Union[bool, float]
    bool_int: Union[bool, int]


@pytest.mark.parametrize(
    "cfg",
    [
        Config("str_int", "1", "1"),  # [cfg0]
        Config("str_int", "foo", "foo"),  # [cfg1]
        Config("str_int", 1, 1),  # [cfg2]
        Config("str_int", 1.0, "1.0"),  # [cfg3]
        Config("int_str", 1.0, 1),  # [cfg4]
        Config("str_tup_strs", "foo", "foo"),  # [cfg5]
        Config("str_tup_strs", ["foo"], ("foo",)),  # [cfg6]
        Config("str_tup_strs", ["foo", "bar"], ("foo", "bar")),  # [cfg7]
        Config("tup_ints_tup_strs", ["foo", 1], ("foo", "1")),  # [cfg8]
        Config("tup_ints_tup_strs", ["1", 2], (1, 2)),  # [cfg9]
        Config("tup_strs_tup_ints", ["1", 2], ("1", "2")),  # [cfg10]
        Config("float_bool", 1, 1.0),  # [cfg11]
        Config("bool_float", 1, True),  # [cfg12]
        Config("bool_int", 1, 1),  # [cfg13]
    ],
)
def test_unions_success(cfg):
    res = cast_field_value(Unions, cfg.param, cfg.val, **cfg.kw)
    if isinstance(cfg.sol, Sequence):
        assert isinstance(res, type(cfg.sol))
        assert len(res) == len(cfg.sol)
        assert all([isinstance(r, type(s)) for r, s in zip(res, cfg.sol)])
        assert res == cfg.sol
    else:
        assert res == cfg.sol


@pytest.mark.parametrize(
    "cfg",
    [
        Config("str_int", [1]),  # [cfg0]
    ],
)
def test_unions_fail(cfg):
    with pytest.raises(cfg.sol or InvalidParameterValueError):
        cast_field_value(Unions, cfg.param, cfg.val)

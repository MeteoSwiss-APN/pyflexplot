"""Tests for function ``pyflexplot.utils.dataclasses.cast_field_value``."""
# Standard library
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import pytest

# First-party
from srutils.dataclasses import cast_field_value
from srutils.exceptions import InvalidParameterNameError
from srutils.exceptions import InvalidParameterValueError

# Abbreviations
T = True
F = False
dt = datetime
td = timedelta


@dataclass
class Cfg:
    param: str
    val: Any
    sol: Any = None
    kw: Dict = field(default_factory=dict)


class TestScalar:
    @dataclass
    class Params:
        str_: str
        int_: int
        float_: float
        bool_: bool

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("str_", "foo", "foo"),  # [cfg0]
            Cfg("str_", 9999, "9999"),  # [cfg1]
            Cfg("str_", 3.14, "3.14"),  # [cfg2]
            Cfg("str_", None, "None"),  # [cfg3]
            Cfg("int_", 99999, 99999),  # [cfg4]
            Cfg("int_", 999.999, 999),  # [cfg5]
            Cfg("int_", "9999", 9999),  # [cfg6]
            Cfg("int_", "00999", 999),  # [cfg7]
            Cfg("float_", 9999, 9999.0),  # [cfg8]
            Cfg("float_", 999.999, 999.999),  # [cfg9]
            Cfg("float_", "9999", 9999.0),  # [cfg10]
            Cfg("float_", "099.9", 99.9),  # [cfg11]
            Cfg("float_", "00999", 999.0),  # [cfg12]
            Cfg("bool_", T, T),  # [cfg13]
            Cfg("bool_", F, F),  # [cfg14]
            Cfg("bool_", 1, T),  # [cfg15]
            Cfg("bool_", 0, F),  # [cfg16]
            Cfg("bool_", None, F),  # [cfg17]
            Cfg("bool_", "True", T),  # [cfg18]
            Cfg("bool_", "False", T),  # [cfg19]
            Cfg("bool_", "1", T),  # [cfg20]
            Cfg("bool_", "0", T),  # [cfg21]
            Cfg("bool_", "", F),  # [cfg22]
            Cfg("bool_", "None", T),  # [cfg23]
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
        assert res == cfg.sol
        assert isinstance(res, type(cfg.sol))

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("asdf", "foo", InvalidParameterNameError),  # [cfg0]
            Cfg("int_", "foo"),  # [cfg1]
            Cfg("int_", None),  # [cfg2]
            Cfg("int_", "999.9"),  # [cfg3]
            Cfg("int_", "09.9"),  # [cfg4]
            Cfg("int_", ""),  # [cfg5]
            Cfg("float_", "foo"),  # [cfg6]
            Cfg("bool_", []),  # [cfg7]
            Cfg("str_", ["a", "b"]),  # [cfg8]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)

    def test_bool_mode(self):
        def cast_bool(value, mode):
            return cast_field_value(self.Params, "bool_", value, bool_mode=mode)

        assert cast_bool("True", "native") is True
        assert cast_bool("False", "native") is True
        assert cast_bool("True", "intuitive") is True
        assert cast_bool("False", "intuitive") is False


class TestOptionalScalar:
    @dataclass
    class Params:
        str_: Optional[str]
        int_: Optional[int]
        float_: Optional[float]
        bool_: Optional[bool]

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("str_", "foo", "foo"),  # [cfg0]
            Cfg("str_", 9999, "9999"),  # [cfg1]
            Cfg("str_", 3.14, "3.14"),  # [cfg2]
            Cfg("str_", None, None),  # [cfg3]
            Cfg("int_", 99999, 99999),  # [cfg4]
            Cfg("int_", 999.999, 999),  # [cfg5]
            Cfg("int_", "9999", 9999),  # [cfg6]
            Cfg("int_", "00999", 999),  # [cfg7]
            Cfg("float_", 9999, 9999.0),  # [cfg8]
            Cfg("float_", 999.999, 999.999),  # [cfg9]
            Cfg("float_", "9999", 9999.0),  # [cfg10]
            Cfg("float_", "099.9", 99.9),  # [cfg11]
            Cfg("float_", "00999", 999.0),  # [cfg12]
            Cfg("bool_", T, T),  # [cfg13]
            Cfg("bool_", F, F),  # [cfg14]
            Cfg("bool_", 1, T),  # [cfg15]
            Cfg("bool_", 0, F),  # [cfg16]
            Cfg("bool_", None, None),  # [cfg17]
            Cfg("bool_", "True", T),  # [cfg18]
            Cfg("bool_", "False", T),  # [cfg19]
            Cfg("bool_", "1", T),  # [cfg20]
            Cfg("bool_", "0", T),  # [cfg21]
            Cfg("bool_", "", F),  # [cfg22]
            Cfg("bool_", "None", None),  # [cfg23]
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
        assert res == cfg.sol
        assert isinstance(res, type(cfg.sol))

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("int_", "foo"),  # [cfg0]
            Cfg("int_", "999.9"),  # [cfg1]
            Cfg("int_", "09.9"),  # [cfg2]
            Cfg("int_", ""),  # [cfg3]
            Cfg("float_", "foo"),  # [cfg4]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)


class TestTuple:
    @dataclass
    class Params:
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
            Cfg("strs", tuple(), tuple()),  # [cfg0]
            Cfg("strs", ["x"], ("x",)),  # [cfg1]
            Cfg("strs", ("foo", "bar"), ("foo", "bar")),  # [cfg2]
            Cfg("strs", ["x", "y", "z"], ("x", "y", "z")),  # [cfg3]
            Cfg("strs", (1, 2.2, None), ("1", "2.2", "None")),  # [cfg4]
            Cfg("ints", [], tuple()),  # [cfg5]
            Cfg("ints", (0,), (0,)),  # [cfg6]
            Cfg("ints", (0, 1), (0, 1)),  # [cfg7]
            Cfg("ints", [8.8, 16, 32.3, "64"], (8, 16, 32, 64)),  # [cfg8]
            Cfg("floats", tuple(), tuple()),  # [cfg9]
            Cfg("floats", (1.0,), (1.0,)),  # [cfg10]
            Cfg("floats", (1, "2", "3.0"), (1.0, 2.0, 3.0)),  # [cfg11]
            Cfg("bools", tuple(), tuple()),  # [cfg12]
            Cfg("bools", (T,), (T,)),  # [cfg13]
            Cfg("bools", (F, 1, "False"), (F, T, T)),  # [cfg14]
            Cfg("opt_ints", (None, 2.0, "4"), (None, 2, 4)),  # [cfg15]
            Cfg("one_str", (None,), ("None",)),  # [cfg16]
            Cfg("two_ints", ["4", 6.6], (4, 6)),  # [cfg17]
            Cfg("three_floats", (2, "4", 6.0), (2.0, 4.0, 6.0)),  # [cfg18]
            Cfg("four_bools", (None, "False", 0.0, 1), (F, T, F, T)),  # [cfg19]
            Cfg("mixed", [F, 6.8, 32, "64"], ("False", 6, 32.0, True)),  # [cfg20]
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
        assert isinstance(res, type(cfg.sol))
        assert len(res) == len(cfg.sol)
        assert all([isinstance(r, type(s)) for r, s in zip(res, cfg.sol)])
        assert res == cfg.sol

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("strs", [[None]]),  # [cfg0]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("strs", 1, ("1",)),  # [cfg0]
            Cfg("ints", "004", (0, 0, 4), kw={"auto_wrap": False}),  # [cfg1]
            Cfg("ints", "004", (4,)),  # [cfg2]
            Cfg("ints", "004", (4,), kw={"unpack_str": False}),  # [cfg3]
            Cfg("floats", "3", (3.0,)),  # [cfg4]
            Cfg("bools", "False", (T, T, T, T, T), kw={"auto_wrap": False}),  # [cfg5]
            Cfg("bools", "False", (T,), kw={"unpack_str": False}),  # [cfg6]
        ],
    )
    def test_auto_wrap(self, cfg):
        kw = {"auto_wrap": True, **cfg.kw}
        res = cast_field_value(self.Params, cfg.param, cfg.val, **kw)
        assert isinstance(res, type(cfg.sol))
        assert len(res) == len(cfg.sol)
        assert all([isinstance(r, type(s)) for r, s in zip(res, cfg.sol)])
        assert res == cfg.sol


class OptionalTuple:
    @dataclass
    class Params:
        str_: Optional[Tuple[str]]
        strs: Optional[Tuple[str, ...]]
        int_: Optional[Tuple[int]]
        two_floats: Optional[Tuple[float, float]]
        bools: Optional[Tuple[bool, ...]]

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("str_", ["foo"], ("foo",), kw={"unpack_str": False}),  # [cfg0]
            Cfg("str_", None, None),  # [cfg1]
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
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
            Cfg("str_", [[None]]),  # [cfg0]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)


class TestList:
    @dataclass
    class Params:
        strs: List[str]
        ints: List[int]
        floats: List[float]
        bools: List[bool]
        tups: List[Tuple]

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("strs", ["foo"], ["foo"]),  # [cfg0]
            Cfg("strs", "foo", ["f", "o", "o"], kw={"auto_wrap": False}),  # [cfg1]
            Cfg("strs", "foo", ["foo"], kw={"auto_wrap": True}),  # [cfg2]
            Cfg(
                "strs", "foo", ["foo"], kw={"auto_wrap": True, "unpack_str": False}
            ),  # [cfg3]
            Cfg("ints", (1, 2), [1, 2]),  # [cfg4]
            Cfg("ints", ["123"], [123]),  # [cfg5]
            Cfg("ints", "123", [1, 2, 3]),  # [cfg6]
            Cfg("ints", "123", [123], kw={"auto_wrap": True}),  # [cfg7]
            Cfg("floats", "123", [1.0, 2.0, 3.0]),  # cfg8
            Cfg("bools", ["False"], [T]),  # cfg9
            Cfg("bools", "False", [T, T, T, T, T]),  # cfg10
            Cfg("bools", 0, [False], kw={"auto_wrap": True}),  # cfg11
            Cfg("tups", [[0]], [(0,)]),  # cfg12
            Cfg("tups", ["foo"], [("f", "o", "o")]),  # cfg13
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
        assert isinstance(res, type(cfg.sol))
        assert len(res) == len(cfg.sol)
        assert all([isinstance(r, type(s)) for r, s in zip(res, cfg.sol)])
        assert res == cfg.sol

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("strs", "123", kw={"unpack_str": False}),  # [cfg0]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)


class TestSequence:
    @dataclass
    class Params:
        blank: Sequence
        anys: Sequence[Any]
        ints: Sequence[int]
        strs: Sequence[str]

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("blank", [], []),  # [cfg0]
            Cfg("blank", "foo", "foo"),  # [cfg1]
            Cfg("blank", 1, [1], kw={"auto_wrap": True}),  # [cfg2]
            Cfg("anys", tuple(), tuple()),  # [cfg3]
            Cfg("anys", (1, "2", (3.0,)), (1, "2", (3.0,))),  # [cfg4]
            Cfg("anys", 1.0, [1.0], kw={"auto_wrap": True}),  # [cfg5]
            Cfg("ints", "123", [1, 2, 3]),  # [cfg6]
            Cfg("strs", "abc", ["a", "b", "c"]),  # [cfg7]
            Cfg("strs", ["abc"], ["abc"]),  # [cfg8]
            Cfg("strs", (1, 2, 3), ("1", "2", "3")),  # [cfg9]
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
        assert isinstance(res, type(cfg.sol))
        assert len(res) == len(cfg.sol)
        assert all([isinstance(r, type(s)) for r, s in zip(res, cfg.sol)])
        assert res == cfg.sol

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("blank", None),  # [cfg0]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)


class TestUnion:
    @dataclass
    class Params:
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
            Cfg("str_int", "1", "1"),  # [cfg0]
            Cfg("str_int", "foo", "foo"),  # [cfg1]
            Cfg("str_int", 1, 1),  # [cfg2]
            Cfg("str_int", 1.0, "1.0"),  # [cfg3]
            Cfg("int_str", 1.0, 1),  # [cfg4]
            Cfg("str_tup_strs", "foo", "foo"),  # [cfg5]
            Cfg("str_tup_strs", ["foo"], ("foo",)),  # [cfg6]
            Cfg("str_tup_strs", ["foo", "bar"], ("foo", "bar")),  # [cfg7]
            Cfg("tup_ints_tup_strs", ["foo", 1], ("foo", "1")),  # [cfg8]
            Cfg("tup_ints_tup_strs", ["1", 2], (1, 2)),  # [cfg9]
            Cfg("tup_strs_tup_ints", ["1", 2], ("1", "2")),  # [cfg10]
            Cfg("float_bool", 1, 1.0),  # [cfg11]
            Cfg("bool_float", 1, True),  # [cfg12]
            Cfg("bool_int", 1, 1),  # [cfg13]
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
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
            Cfg("str_int", [1]),  # [cfg0]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)


class TestDatetime:
    @dataclass
    class Params:
        dt: datetime
        opt_dt: Optional[datetime]
        tup_dts: Tuple[datetime, ...]

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("dt", "202012021308", dt(2020, 12, 2, 13, 8)),  # [cfg0]
            Cfg("dt", 202012021308, dt(2020, 12, 2, 13, 8)),  # [cfg1]
            Cfg(
                "dt", "2020-12-02", dt(2020, 12, 2), kw={"datetime_fmt": "%Y-%m-%d"}
            ),  # [cfg2]
            Cfg("opt_dt", None, None),  # [cfg3]
            Cfg("opt_dt", "202012021308", dt(2020, 12, 2, 13, 8)),  # [cfg4]
            Cfg("opt_dt", 202012021308, dt(2020, 12, 2, 13, 8)),  # [cfg5]
            Cfg("opt_dt", "None", None),  # [cfg6]
            Cfg("tup_dts", ["202012021308"], (dt(2020, 12, 2, 13, 8),)),  # [cfg7]
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
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
            Cfg("dt", "2020-12-02"),  # [cfg0]
            Cfg("dt", "2020-12-02", kw={"datetime_fmt": "auto"}),  # [cfg1]
            Cfg("dt", "202012021308", kw={"datetime_fmt": "%Y-%m-%d"}),  # [cfg2]
            Cfg("tup_dts", "202012021308"),  # [cfg3]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)


class TestTimedelta:
    @dataclass
    class Params:
        td: timedelta
        opt_td: Optional[timedelta]
        tup_tds: Tuple[timedelta, ...]

    @pytest.mark.parametrize(
        "cfg",
        [
            Cfg("td", 0, td(0)),  # [cfg0]
            Cfg("td", 1, td(hours=24)),  # [cfg1]
            Cfg("td", 1, td(1), kw={"timedelta_unit": "days"}),  # [cfg2]
            Cfg("td", 1, td(hours=1), kw={"timedelta_unit": "hours"}),  # [cfg3]
            Cfg("opt_td", "None", None),  # [cfg4]
            Cfg("opt_td", 0, td(0)),  # [cfg5]
            Cfg("opt_td", 7, td(weeks=1)),  # [cfg6]
            Cfg(
                "tup_tds",
                [1, 2, 3],
                (td(1 / 24), td(2 / 24), td(3 / 24)),
                kw={"timedelta_unit": "hours"},
            ),  # [cfg7]
            Cfg("tup_tds", 4, (td(4),), kw={"auto_wrap": True}),  # [cfg8]
        ],
    )
    def test_ok(self, cfg):
        res = cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)
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
            Cfg("td", "foo"),  # [cfg0]
            Cfg("td", 1, ValueError, kw={"timedelta_unit": "hour"}),  # [cfg1]
            Cfg("td", None),  # [cfg2]
            Cfg("tup_tds", 4),  # [cfg3]
        ],
    )
    def test_fail(self, cfg):
        with pytest.raises(cfg.sol or InvalidParameterValueError):
            cast_field_value(self.Params, cfg.param, cfg.val, **cfg.kw)

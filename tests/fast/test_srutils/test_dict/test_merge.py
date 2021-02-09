"""Tests for function ``srutils.dict.merge_dicts``."""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Type

# Third-party
import pytest

# First-party
from srutils.dict import merge_dicts


@dc.dataclass
class Cfg:
    dcts: Sequence[Dict[Any, Any]]
    sol: Dict[Any, Any]
    kwargs: Dict[str, Any] = dc.field(default_factory=dict)


@pytest.mark.parametrize(
    "cfg",
    [
        Cfg(  # [cfg0]
            dcts=[
                {"a": {"b": 1, "c": 2}, "d": 3},
                {"a": {"c": 4, "e": 2}},
            ],
            sol={"a": {"b": 1, "c": 4, "e": 2}, "d": 3},
        ),
        Cfg(  # [cfg1]
            dcts=[
                {"foo": {"bar": [{"baz": "hello"}]}},
                {"foo": {"bar": [{"zab": "world"}]}},
            ],
            sol={"foo": {"bar": [{"zab": "world"}]}},
            kwargs={"rec_seqs": False},
        ),
        Cfg(  # [cfg2]
            dcts=[
                {"foo": {"bar": [{"baz": "hello"}]}},
                {"foo": {"bar": [{"zab": "world"}]}},
            ],
            sol={"foo": {"bar": [{"baz": "hello", "zab": "world"}]}},
            kwargs={"rec_seqs": True},
        ),
        Cfg(  # [cfg3]
            dcts=[
                {"x": [0, 1], "y": [2, 3], "z": [{1: 2, 3: 4}]},
                {"x": [0, [1, 2]], "y": 3, "z": [{2: 4, 3: 6}]},
            ],
            sol={"x": [0, [1, 2]], "y": 3, "z": [{1: 2, 2: 4, 3: 6}]},
            kwargs={"rec_seqs": True, "overwrite_seqs": True},
        ),
        Cfg(  # [cfg4]
            dcts=[
                {0: {"a": [{"b": [[{"c": [1], "d": 4}]]}]}},
                {0: {"a": [{"b": [[{"c": 1}]]}]}},
            ],
            sol={0: {"a": [{"b": [[{"c": 1, "d": 4}]]}]}},
            kwargs={"rec_seqs": True, "overwrite_seqs": True},
        ),
        Cfg(  # [cfg5]
            dcts=[
                {"a": [1, 2, {"b": 3}]},
                {"a": [1, 4, 6]},
            ],
            sol={"a": [1, 4, 6]},
            kwargs={"rec_seqs": True, "overwrite_seq_dicts": True},
        ),
        Cfg(  # [cfg6]
            dcts=[
                {"a": [1, 2, {"b": {"c": [3, {"d": 4}, {"e": 6}]}, "g": 8}]},
                {"a": [1, 3, {"b": {"c": [{"d": 2}, 4, {"f": 7}]}}]},
            ],
            sol={"a": [1, 3, {"b": {"c": [{"d": 2}, 4, {"e": 6, "f": 7}]}, "g": 8}]},
            kwargs={"rec_seqs": True, "overwrite_seq_dicts": True},
        ),
    ],
)
def test(cfg):
    res = merge_dicts(*cfg.dcts, **cfg.kwargs)
    assert res == cfg.sol


@dc.dataclass
class CfgFail:
    dcts: Sequence[Dict[Any, Any]]
    ex: Type[Exception]
    ex_msg: Optional[str] = None
    kwargs: Dict[str, Any] = dc.field(default_factory=dict)


@pytest.mark.parametrize(
    "cfg",
    [
        CfgFail(  # [cfg0]
            dcts=[
                {"x": [0, 1], "y": [2, 3], "z": [{1: 2, 3: 4}]},
                {"x": [0, [1, 2]], "y": 3, "z": [{2: 4, 3: 6}]},
            ],
            ex=TypeError,
            ex_msg=r"^some but not all arguments are sequences.*",
            kwargs={"rec_seqs": True, "overwrite_seqs": False},
        ),
        CfgFail(  # [cfg1]
            dcts=[
                {"a": [{"b": [[{"c": [1], "d": 4}]]}]},
                {"a": [{"b": [[{"c": 1}]]}]},
            ],
            ex=TypeError,
            ex_msg=r"^some but not all arguments are sequences.*",
            kwargs={"rec_seqs": True, "overwrite_seqs": False},
        ),
        CfgFail(  # [cfg2]
            dcts=[
                {"a": [1, 2, {"b": 3}]},
                {"a": [1, 4, 6]},
            ],
            ex=TypeError,
            ex_msg=r"^element #2 is a mapping in some but not all sequences.*",
            kwargs={"rec_seqs": True, "overwrite_seq_dicts": False},
        ),
    ],
)
def test_fail(cfg):
    with pytest.raises(cfg.ex, match=cfg.ex_msg):
        merge_dicts(*cfg.dcts, **cfg.kwargs)

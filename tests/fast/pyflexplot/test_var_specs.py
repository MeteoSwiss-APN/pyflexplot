#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.var_specs``."""
# Third-party
import pytest
from pydantic.dataclasses import dataclass as pydantic_dataclass

# First-party
from pyflexplot.var_specs import MultiVarSpecs
from pyflexplot.var_specs import VarSpecs
from srutils.dict import decompress_multival_dict
from srutils.testing import TestConfBase as _TestConf
from srutils.testing import check_is_list_like
from srutils.various import isiterable


def check_var_specs_lst_lst(var_specs_lst_lst, conf):
    """Check a nested list of var specs objs against a specification dict.

    Because we don't know the order of the expansion, we first produce all
    possible specification dicts. Then, for each var specs object, we check
    in turn all specification dict that have not yet matched a var specs
    object. When the var specs object matches a dict, we remove that dict
    from the list of dicts to be checked, and continue with the next var
    specs object. If an object matches none of the still available dicts,
    the test has failed.

    """
    check_is_list_like(
        var_specs_lst_lst, len_=conf.n, f_children=isiterable,
    )
    for var_specs_lst in var_specs_lst_lst:
        check_var_specs_lst_unordered(var_specs_lst, conf)


def check_var_specs_lst_unordered(var_specs_lst, conf):
    check_is_list_like(var_specs_lst, t_children=VarSpecs)
    var_specs_dct_lst = decompress_multival_dict(conf.dct, 2, flatten=True)
    for var_specs in var_specs_lst:
        exception = None
        for var_specs_dct in var_specs_dct_lst:
            subconf = conf.derive(dct=var_specs_dct)
            try:
                check_var_specs(var_specs, subconf)
            except AssertionError as e:
                exception = e
                continue
            else:
                break
        else:
            raise AssertionError(
                f"no matching solution found for var_specs among var_specs_dct_lst",
                var_specs,
                var_specs_dct_lst,
            ) from exception


def check_var_specs(var_specs, conf):
    """Compare a var specs object with a var specs dict."""

    assert isinstance(var_specs, VarSpecs)

    # Check validity of test data
    mismatches = [k for k in conf.dct if k not in dict(var_specs)]
    if mismatches:
        # Test data is broken, NOT the tested code, so NOT AssertionError!
        raise ValueError(f"invalid solution: key mismatches: {mismatches}")

    assert type(var_specs).__name__ == conf.type_name  # SR_TMP

    sol = set(conf.dct.items())
    res = set(dict(var_specs).items())
    assert sol.issubset(res)

    sol = {**conf.dct, "name": conf.name, "rlon": (None,), "rlat": (None,)}
    res = dict(var_specs)
    assert sol == res


@pydantic_dataclass(frozen=True)
class Conf_Create(_TestConf):
    name: str
    type_name: str  # SR_TMP
    dct: dict
    n: int
    subdct_fail: dict


class _Test_Create_SingleObjDct:
    """Create a single variable specification object."""

    def test(self):
        var_specs_lst_lst = VarSpecs.create(self.c.name, self.c.dct)
        check_is_list_like(var_specs_lst_lst, len_=1, t_children=list)
        var_specs_lst = next(iter(var_specs_lst_lst))
        check_is_list_like(
            var_specs_lst, len_=self.c.n, t_children=VarSpecs,
        )
        var_specs = next(iter(var_specs_lst))
        check_var_specs(var_specs, self.c)

    def test_fail(self):
        var_specs_lst_lst = VarSpecs.create(self.c.name, self.c.dct)
        conf = self.c.derive(dct={**self.c.dct, **self.c.subdct_fail})
        with pytest.raises(AssertionError):
            check_var_specs_lst_lst(var_specs_lst_lst, conf)


class Test_Create_SingleObjDct_Concentration(_Test_Create_SingleObjDct):
    c = Conf_Create(
        name="concentration",
        type_name="VarSpecs_Concentration",  # SR_TMP
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": 3,
            "integrate": False,
            "species_id": 2,
            "level": 1,
        },
        n=1,
        subdct_fail={"time": 4, "level": 0},
    )


class Test_Create_SingleObjDct_Deposition(_Test_Create_SingleObjDct):
    c = Conf_Create(
        name="deposition",
        type_name="VarSpecs_Deposition",  # SR_TMP
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": 3,
            "integrate": False,
            "species_id": 2,
            "deposition": "wet",
        },
        n=1,
        subdct_fail={"time": 4, "species_id": 0},
    )

    def test_tot_fail(self):
        """Creation of ``ValSpecs`` with deposition type "tot" fails."""
        with pytest.raises(ValueError):
            VarSpecs.create(self.c.name, {**self.c.dct, "deposition": "tot"})


class _Test_Create_MultiObjDct:
    """Create multiple variable specification objects."""

    def test(self):
        var_specs_lst_lst = VarSpecs.create(self.c.name, self.c.dct)
        check_var_specs_lst_lst(var_specs_lst_lst, self.c)

    def test_fail(self):
        var_specs_lst_lst = VarSpecs.create(self.c.name, self.c.dct)
        conf = self.c.derive(dct={**self.c.dct, **self.c.subdct_fail})
        with pytest.raises(AssertionError):
            check_var_specs_lst_lst(var_specs_lst_lst, conf)


class Test_Create_MultiObjDct_Concentration(_Test_Create_MultiObjDct):
    c = Conf_Create(
        name="concentration",
        type_name="VarSpecs_Concentration",  # SR_TMP
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": [1, 3],
            "integrate": False,
            "species_id": [1, 2],
            "level": 1,
        },
        n=4,
        subdct_fail={"time": [3, 4], "level": 0},
    )


class Test_Create_MultiObjDct_Deposition(_Test_Create_MultiObjDct):
    c = Conf_Create(
        name="deposition",
        type_name="VarSpecs_Deposition",  # SR_TMP
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": [1, 3],
            "integrate": False,
            "species_id": [1, 2],
            "deposition": "dry",
        },
        n=4,
        subdct_fail={"time": [3, 4], "species_id": 0},
    )


class _Test_Create_MultiObjDctNested:
    """Create multiple variable specification objects (nested list).

    This is relevant because such specs dicts with nested multi-object elements
    are produced by the CLI. The outer nesting is for separate plots, while the
    inner is for multiple variables per plot (e.g., wet and dry deposition).

    """

    def test(self):
        var_specs_lst_lst = VarSpecs.create(self.c.name, self.c.dct)
        check_var_specs_lst_lst(var_specs_lst_lst, self.c)

    def test_fail(self):
        var_specs_lst_lst = VarSpecs.create(self.c.name, self.c.dct)
        conf = self.c.derive(dct={**self.c.dct, **self.c.subdct_fail})
        with pytest.raises(AssertionError):
            check_var_specs_lst_lst(var_specs_lst_lst, conf)


class Test_Create_MultiObjDctNested_Concentration(_Test_Create_MultiObjDctNested):
    c = Conf_Create(
        name="concentration",
        type_name="VarSpecs_Concentration",  # SR_TMP
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": [1, 3],
            "integrate": False,
            "species_id": [(1,), (1, 2)],
            "level": 1,
        },
        n=4,
        subdct_fail={"time": [(3, 4), 2], "level": 0},
    )


class Test_Create_MultiObjDctNested_Deposition(_Test_Create_MultiObjDctNested):
    c = Conf_Create(
        name="deposition",
        type_name="VarSpecs_Deposition",  # SR_TMP
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": [1, 3],
            "integrate": False,
            "species_id": [(1,), (1, 2)],
            "deposition": "wet",
        },
        n=4,
        subdct_fail={"time": [(3, 4), 2], "deposition": "dry"},
    )


@pydantic_dataclass(frozen=True)
class Conf_Multi(_TestConf):
    name: str
    dct: dict
    n: int


class _Test_MultiVarSpecs:
    """Test ``MultiVarSpecs``."""

    def test_create(self, conf=None):

        # Create reference VarSpecs objects (nested list)
        dct = self.c.dct.copy()
        if dct.get("deposition") == "tot":
            # VarSpecs doesn't unterstand "tot" by itself
            dct["deposition"] = ("wet", "dry")
        conf = self.c.derive(dct=dct)
        var_specs_lst_lst = VarSpecs.create(conf.name, conf.dct)
        assert len(var_specs_lst_lst) == self.c.n

        # Create MultVarSpecs objects
        multi_var_specs_lst = MultiVarSpecs.create(self.c.name, self.c.dct)
        assert len(multi_var_specs_lst) == self.c.n

        # Compare VarSpecs objects in MultVarSpecs objects to reference ones
        sol = [
            var_specs
            for var_specs_lst in var_specs_lst_lst
            for var_specs in var_specs_lst
        ]
        res = [
            var_specs
            for multi_var_specs in multi_var_specs_lst
            for var_specs in multi_var_specs
        ]
        assert len(res) == len(sol)
        assert res == sol


class Test_MultiVarSpecs_Concentration(_Test_MultiVarSpecs):
    c = Conf_Multi(
        name="concentration",
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": [1, 3],
            "integrate": False,
            "species_id": [(1,), (1, 2)],
            "level": 1,
        },
        n=4,
    )


class Test_MultiVarSpecs_DepositionDry(_Test_MultiVarSpecs):
    c = Conf_Multi(
        name="deposition",
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": [1, 3],
            "integrate": False,
            "species_id": [(1,), (1, 2)],
            "deposition": "dry",
        },
        n=4,
    )


class Test_MultiVarSpecs_DepositionTot(_Test_MultiVarSpecs):
    c = Conf_Multi(
        name="deposition",
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": 3,
            "integrate": False,
            "species_id": 2,
            "deposition": "tot",
        },
        n=1,
    )

    def test_tot_vs_wet_dry(self):
        """Check that deposition type "tot" is equivalent to ("wet", "dry")."""

        # Deposition type "tot"
        mvs0_lst = MultiVarSpecs.create(self.c.name, self.c.dct)
        assert len(mvs0_lst) == 1
        mvs0 = next(iter(mvs0_lst))

        # Deposition type ("wet", "dry")
        conf1 = self.c.derive(dct={**self.c.dct, "deposition": ("wet", "dry")})
        mvs1_lst = MultiVarSpecs.create(conf1.name, conf1.dct)
        assert len(mvs1_lst) == 1
        mvs1 = next(iter(mvs1_lst))
        assert mvs1 == mvs0  # ("wet", "dry") == "tot"

        # Deposition type "wet"
        conf2 = self.c.derive(dct={**self.c.dct, "deposition": "wet"})
        mvs2_lst = MultiVarSpecs.create(conf2.name, conf2.dct)
        assert len(mvs2_lst) == 1
        mvs2 = next(iter(mvs2_lst))
        assert mvs2 != mvs0  # "tot" != "wet"
        assert mvs2 != mvs1  # "tot" != ("wet", "dry")


class Test_MultiVarSpecs_Interface:
    c = Conf_Multi(
        name="deposition",
        dct={
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "time": 3,
            "integrate": False,
            "species_id": 2,
            "deposition": "tot",
        },
        n=1,
    )

    def create_multi_var_specs(self):
        mvs_lst = MultiVarSpecs.create(self.c.name, self.c.dct)
        assert len(mvs_lst) == 1
        mvs = next(iter(mvs_lst))
        assert len(list(mvs)) == 2
        return mvs

    def test_var_specs(self):
        multi_var_specs = self.create_multi_var_specs()
        var_specs_lst = list(multi_var_specs)

        # Check that there are separate VarSpecs for "wet" and "dry"
        check_is_list_like(var_specs_lst, len_=2, t_children=VarSpecs)
        assert len(var_specs_lst) == 2
        sol = {"wet", "dry"}
        res = {vs.deposition for vs in var_specs_lst}
        assert res == sol

        # Check that apart from "deposition", the two are identical
        # We need to neutralize a few elements that we expect to differ or
        # that are (at the moment still) added during creation of VarSpecs
        neutral = {"name": None, "deposition": None, "rlat": None, "rlon": None}
        sol = {**self.c.dct, **neutral}
        res1 = {**dict(var_specs_lst[0]), **neutral}
        res2 = {**dict(var_specs_lst[1]), **neutral}
        assert res1 == sol
        assert res2 == sol
        assert res1 == res2

    def test_compressed_var_specs(self):
        multi_var_specs = self.create_multi_var_specs()
        var_specs_dct = multi_var_specs.compressed_dct()
        assert isinstance(var_specs_dct, dict)
        # SR_TMP< TODO remove rlon/rlat etc. from dict in VarSpecs
        # + assert var_specs_dct == selc.c.dct
        sol = set(self.c.dct.items())
        res = set(var_specs_dct.items())
        assert res.issuperset(sol)
        # SR_TMP>

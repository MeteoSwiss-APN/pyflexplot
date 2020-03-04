#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.var_specs``.
"""
# Standard library
from dataclasses import dataclass

# Third-party
import pytest

# First-party
from pyflexplot.setup import Setup
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
    var_specs_dct_lst = decompress_multival_dict(conf.dct, depth=2, flatten=True)
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
    mismatches = [k for k in conf.dct if k not in var_specs.dict()]
    if mismatches:
        # Test data is broken, NOT the tested code, so NOT AssertionError!
        raise ValueError(
            f"invalid solution: {len(mismatches)} key mismatches",
            mismatches,
            conf.dct.keys(),
            var_specs.dict().keys(),
        )

    sol = set(conf.dct.items())
    res = set(var_specs.dict().items())
    assert sol.issubset(res)

    sol = conf.dct
    res = var_specs.dict()
    assert sol == res


@dataclass(frozen=True)
class Conf_Create(_TestConf):
    dct: dict
    n: int
    subdct_fail: dict


class _Test_Create_SingleObjDct:
    """Create a single variable specification object."""

    def test(self):
        var_specs_lst_lst = VarSpecs.create_many(self.setups, pre_expand_time=True)
        check_is_list_like(var_specs_lst_lst, len_=1, t_children=list)
        var_specs_lst = next(iter(var_specs_lst_lst))
        check_is_list_like(
            var_specs_lst, len_=self.c.n, t_children=VarSpecs,
        )
        var_specs = next(iter(var_specs_lst))
        check_var_specs(var_specs, self.c)

    def test_fail(self):
        var_specs_lst_lst = VarSpecs.create_many(self.setups, pre_expand_time=True)
        conf = self.c.derive(dct={**self.c.dct, **self.c.subdct_fail})
        with pytest.raises(AssertionError):
            check_var_specs_lst_lst(var_specs_lst_lst, conf)


class Test_Create_SingleObjDct_Concentration(_Test_Create_SingleObjDct):
    c = Conf_Create(
        dct={
            "integrate": False,
            "species_id": 2,
            "level": 1,
            "time": 3,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "deposition": "none",
        },
        n=1,
        subdct_fail={"time": 4, "level": 0},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="concentration",
        integrate=False,
        species_id=2,
        level_idx=1,
        time_idcs=[3],
    )
    setups = [base_setup]


class Test_Create_SingleObjDct_Deposition(_Test_Create_SingleObjDct):
    c = Conf_Create(
        dct={
            "deposition": "wet",
            "species_id": 2,
            "integrate": False,
            "time": 3,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "level": -1,
        },
        n=1,
        subdct_fail={"time": 4, "species_id": 0},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="deposition",
        deposition_type="wet",
        species_id=2,
        integrate=False,
        time_idcs=[3],
    )
    setups = [base_setup]


class _Test_Create_MultiObjDct:
    """Create multiple variable specification objects."""

    def test(self):
        var_specs_lst_lst = VarSpecs.create_many(self.setups, pre_expand_time=True)
        check_var_specs_lst_lst(var_specs_lst_lst, self.c)

    def test_fail(self):
        var_specs_lst_lst = VarSpecs.create_many(self.setups, pre_expand_time=True)
        conf = self.c.derive(dct={**self.c.dct, **self.c.subdct_fail})
        with pytest.raises(AssertionError):
            check_var_specs_lst_lst(var_specs_lst_lst, conf)


class Test_Create_MultiObjDct_Concentration(_Test_Create_MultiObjDct):
    c = Conf_Create(
        dct={
            "species_id": [1, 2],
            "level": 1,
            "integrate": False,
            "time": [1, 3],
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "deposition": "none",
        },
        n=4,
        subdct_fail={"time": [3, 4], "level": 0},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="concentration",
        species_id=1,
        level_idx=1,
        integrate=False,
        time_idcs=[1, 3],
    )
    setups = [base_setup, base_setup.derive({"species_id": 2})]


class Test_Create_MultiObjDct_Deposition(_Test_Create_MultiObjDct):
    c = Conf_Create(
        dct={
            "deposition": "dry",
            "species_id": [1, 2],
            "integrate": False,
            "time": [1, 3],
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "level": -1,
        },
        n=4,
        subdct_fail={"time": [3, 4], "species_id": 0},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="deposition",
        deposition_type="dry",
        species_id=1,
        integrate=False,
        time_idcs=[1, 3],
    )
    setups = [base_setup, base_setup.derive({"species_id": 2})]


class _Test_Create_MultiObjDctNested:
    """Create multiple variable specification objects (nested list).

    This is relevant because such specs dicts with nested multi-object elements
    are produced by the CLI. The outer nesting is for separate plots, while the
    inner is for multiple variables per plot (e.g., wet and dry deposition).

    """

    def test(self):
        var_specs_lst_lst = VarSpecs.create_many(self.setups, pre_expand_time=True)
        check_var_specs_lst_lst(var_specs_lst_lst, self.c)

    def test_fail(self):
        var_specs_lst_lst = VarSpecs.create_many(self.setups, pre_expand_time=True)
        conf = self.c.derive(dct={**self.c.dct, **self.c.subdct_fail})
        with pytest.raises(AssertionError):
            check_var_specs_lst_lst(var_specs_lst_lst, conf)


class Test_Create_MultiObjDctNested_Concentration(_Test_Create_MultiObjDctNested):
    c = Conf_Create(
        dct={
            "species_id": [(1,), (1, 2)],
            "level": 1,
            "integrate": False,
            "time": [1, 3],
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "deposition": "none",
        },
        n=4,
        subdct_fail={"time": [(3, 4), 2], "level": 0},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="concentration",
        species_id=1,
        level_idx=1,
        integrate=False,
        time_idcs=[1, 3],
    )
    setups = [base_setup, base_setup.derive({"species_id": [1, 2]})]


class Test_Create_MultiObjDctNested_Deposition(_Test_Create_MultiObjDctNested):
    c = Conf_Create(
        dct={
            "deposition": "wet",
            "species_id": [(1,), (1, 2)],
            "integrate": False,
            "time": [1, 3],
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "level": -1,
        },
        n=4,
        subdct_fail={"time": [(3, 4), 2], "deposition": "dry"},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="deposition",
        deposition_type="wet",
        species_id=1,
        integrate=False,
        time_idcs=[1, 3],
    )
    setups = [base_setup, base_setup.derive({"species_id": [1, 2]})]


@dataclass(frozen=True)
class Conf_Multi(_TestConf):
    dct: dict
    n: int


class _Test_MultiVarSpecs:
    """Test ``MultiVarSpecs``."""

    def test_create(self, conf=None):

        # Create reference VarSpecs objects (nested list)
        var_specs_lst_lst = VarSpecs.create_many(self.setups, pre_expand_time=True)
        assert len(var_specs_lst_lst) == self.c.n

        # Create MultVarSpecs objects
        multi_var_specs_lst = MultiVarSpecs.create(self.setups)
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
        dct={
            "species_id": [(1,), (1, 2)],
            "level": 1,
            "integrate": False,
            "time": [1, 3],
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "deposition": "none",
        },
        n=4,
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="concentration",
        species_id=1,
        level_idx=1,
        integrate=False,
        time_idcs=[1, 3],
    )
    setups = [base_setup, base_setup.derive({"species_id": [1, 2]})]


class Test_MultiVarSpecs_DepositionDry(_Test_MultiVarSpecs):
    c = Conf_Multi(
        dct={
            "deposition": "dry",
            "species_id": [(1,), (1, 2)],
            "integrate": False,
            "time": [1, 3],
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "level": -1,
        },
        n=4,
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="deposition",
        deposition_type="dry",
        species_id=1,
        integrate=False,
        time_idcs=[1, 3],
    )
    setups = [base_setup, base_setup.derive({"species_id": [1, 2]})]


class Test_MultiVarSpecs_DepositionTot(_Test_MultiVarSpecs):
    c = Conf_Multi(
        dct={
            "deposition": "tot",
            "species_id": 2,
            "integrate": False,
            "time": 3,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "level": -1,
        },
        n=1,
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="deposition",
        deposition_type="tot",
        species_id=2,
        integrate=False,
        time_idcs=[3],
    )
    setups = [base_setup]

    def test_tot_vs_wet_dry(self):
        """Check that deposition type "tot" is equivalent to ("wet", "dry")."""

        # Deposition type "tot"
        setup0 = next(iter(self.setups))
        mvs0_lst = MultiVarSpecs.create(setup0)
        assert len(mvs0_lst) == 1
        mvs0 = next(iter(mvs0_lst))

        # Deposition type ("wet", "dry")
        setup1 = self.setups[0].derive({"deposition_type": ("wet", "dry")})
        mvs1_lst = MultiVarSpecs.create(setup1)
        assert len(mvs1_lst) == 1
        mvs1 = next(iter(mvs1_lst))
        assert mvs1 == mvs0  # ("wet", "dry") == "tot"

        # Deposition type "wet"
        setup2 = self.setups[0].derive({"deposition_type": "wet"})
        mvs2_lst = MultiVarSpecs.create(setup2)
        assert len(mvs2_lst) == 1
        mvs2 = next(iter(mvs2_lst))
        assert mvs2 != mvs0  # "tot" != "wet"
        assert mvs2 != mvs1  # "tot" != ("wet", "dry")


class Test_MultiVarSpecs_Interface:
    c = Conf_Multi(
        dct={
            "deposition": "tot",
            "species_id": 2,
            "integrate": False,
            "time": 3,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "level": -1,
        },
        n=1,
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="deposition",
        deposition_type="tot",
        species_id=2,
        integrate=False,
        time_idcs=[3],
    )
    setups = [base_setup]

    def create_multi_var_specs(self):
        setup = self.setups[0]  # SR_TMP
        mvs_lst = MultiVarSpecs.create(setup)
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
        res = {vs._setup.deposition_type for vs in var_specs_lst}
        assert res == sol

        # Check that apart from "deposition", the two are identical
        # We need to neutralize a few elements that we expect to differ or
        # that are (at the moment still) added during creation of VarSpecs
        neutral = {"deposition": None}
        sol = {**self.c.dct, **neutral}
        res1 = {**var_specs_lst[0].dict(), **neutral}
        res2 = {**var_specs_lst[1].dict(), **neutral}
        assert res1 == sol
        assert res2 == sol
        assert res1 == res2

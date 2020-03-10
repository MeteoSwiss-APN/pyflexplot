#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.var_specs``.
"""
# Standard library
from dataclasses import dataclass

# Third-party
import pytest  # type: ignore

# First-party
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupCollection
from pyflexplot.specs import FldSpecs
from pyflexplot.specs import VarSpecs
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
                {"var_specs": var_specs, "var_specs_dct_lst": var_specs_dct_lst},
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

    # SR_TMP <
    # sol = set(conf.dct.items())
    # SR_TMP -
    dct = conf.dct.copy()
    dct = dct.copy()
    for param in ["species_id"]:
        value = dct[param]
        if not isinstance(value, tuple):
            dct[param] = (value,)
    sol = set(dct.items())
    # SR_TMP >
    res = set(var_specs.dict().items())
    assert sol.issubset(res)

    # SR_TMP <
    # sol = conf.dct
    # SR_TMP -
    sol = dct
    # SR_TMP >
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
            "deposition": "none",
            "integrate": False,
            "level": 1,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": (2,),
            "time": 3,
        },
        n=1,
        subdct_fail={"level": 0, "time": 4},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        integrate=False,
        level=1,
        outfile="dummy.png",
        species_id=2,
        time=[3],
        variable="concentration",
    )
    setups = SetupCollection([base_setup])


class Test_Create_SingleObjDct_Deposition(_Test_Create_SingleObjDct):
    c = Conf_Create(
        dct={
            "deposition": "wet",
            "integrate": False,
            "level": None,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": (2,),
            "time": 3,
        },
        n=1,
        subdct_fail={"species_id": (0,), "time": 4},
    )
    base_setup = Setup(
        deposition_type="wet",
        infiles=["dummy.nc"],
        integrate=False,
        outfile="dummy.png",
        species_id=2,
        time=[3],
        variable="deposition",
    )
    setups = SetupCollection([base_setup])


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
            "deposition": "none",
            "integrate": False,
            "level": 1,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": [1, 2],
            "time": [1, 3],
        },
        n=4,
        subdct_fail={"level": 0, "time": [3, 4]},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        integrate=False,
        level=1,
        outfile="dummy.png",
        species_id=1,
        time=[1, 3],
        variable="concentration",
    )
    setups = SetupCollection([base_setup, base_setup.derive({"species_id": (2,)})])


class Test_Create_MultiObjDct_Deposition(_Test_Create_MultiObjDct):
    c = Conf_Create(
        dct={
            "deposition": "dry",
            "integrate": False,
            "level": None,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": [1, 2],
            "time": [1, 3],
        },
        n=4,
        subdct_fail={"species_id": (0,), "time": [3, 4]},
    )
    base_setup = Setup(
        deposition_type="dry",
        infiles=["dummy.nc"],
        integrate=False,
        outfile="dummy.png",
        species_id=1,
        time=[1, 3],
        variable="deposition",
    )
    setups = SetupCollection([base_setup, base_setup.derive({"species_id": (2,)})])


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
            "deposition": "none",
            "integrate": False,
            "level": 1,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": [(1,), (1, 2)],
            "time": [1, 3],
        },
        n=4,
        subdct_fail={"level": 0, "time": [(3, 4), 2]},
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        integrate=False,
        level=1,
        outfile="dummy.png",
        species_id=1,
        time=[1, 3],
        variable="concentration",
    )
    setups = SetupCollection([base_setup, base_setup.derive({"species_id": [1, 2]})])


class Test_Create_MultiObjDctNested_Deposition(_Test_Create_MultiObjDctNested):
    c = Conf_Create(
        dct={
            "deposition": "wet",
            "integrate": False,
            "level": None,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": [(1,), (1, 2)],
            "time": [1, 3],
        },
        n=4,
        subdct_fail={"deposition": "dry", "time": [(3, 4), 2]},
    )
    base_setup = Setup(
        deposition_type="wet",
        infiles=["dummy.nc"],
        integrate=False,
        outfile="dummy.png",
        species_id=1,
        time=[1, 3],
        variable="deposition",
    )
    setups = SetupCollection([base_setup, base_setup.derive({"species_id": [1, 2]})])


@dataclass(frozen=True)
class Conf_Multi(_TestConf):
    dct: dict
    n: int


class _Test_FldSpecs:
    """Test ``FldSpecs``."""

    def test_create(self, conf=None):

        # Create reference VarSpecs objects (nested list)
        var_specs_lst_lst = VarSpecs.create_many(self.setups, pre_expand_time=True)
        assert len(var_specs_lst_lst) == self.c.n

        # Create MultVarSpecs objects
        fld_specs_lst = FldSpecs.create(self.setups)
        assert len(fld_specs_lst) == self.c.n

        # Compare VarSpecs objects in MultVarSpecs objects to reference ones
        sol = [
            var_specs
            for var_specs_lst in var_specs_lst_lst
            for var_specs in var_specs_lst
        ]
        res = [
            var_specs
            for fld_specs in fld_specs_lst
            for var_specs in fld_specs.var_specs_lst
        ]
        assert len(res) == len(sol)
        assert res == sol


class Test_FldSpecs_Concentration(_Test_FldSpecs):
    c = Conf_Multi(
        dct={
            "deposition": "none",
            "integrate": False,
            "level": 1,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": [(1,), (1, 2)],
            "time": [1, 3],
        },
        n=4,
    )
    base_setup = Setup(
        infiles=["dummy.nc"],
        integrate=False,
        level=1,
        outfile="dummy.png",
        species_id=1,
        time=[1, 3],
        variable="concentration",
    )
    setups = SetupCollection([base_setup, base_setup.derive({"species_id": [1, 2]})])


class Test_FldSpecs_DepositionDry(_Test_FldSpecs):
    c = Conf_Multi(
        dct={
            "deposition": "dry",
            "integrate": False,
            "level": None,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": [(1,), (1, 2)],
            "time": [1, 3],
        },
        n=4,
    )
    base_setup = Setup(
        deposition_type="dry",
        infiles=["dummy.nc"],
        integrate=False,
        outfile="dummy.png",
        species_id=1,
        time=[1, 3],
        variable="deposition",
    )
    setups = SetupCollection([base_setup, base_setup.derive({"species_id": [1, 2]})])


class Test_FldSpecs_DepositionTot(_Test_FldSpecs):
    c = Conf_Multi(
        dct={
            "deposition": "tot",
            "integrate": False,
            "level": None,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": (2,),
            "time": 3,
        },
        n=1,
    )
    base_setup = Setup(
        deposition_type="tot",
        infiles=["dummy.nc"],
        integrate=False,
        outfile="dummy.png",
        species_id=2,
        time=[3],
        variable="deposition",
    )
    setups = SetupCollection([base_setup])

    def test_tot_vs_wet_dry(self):
        """Check that deposition type "tot" is equivalent to ("wet", "dry")."""

        # Deposition type "tot"
        setup0 = next(iter(self.setups))
        fld_specs_0_lst = FldSpecs.create(setup0)
        assert len(fld_specs_0_lst) == 1
        fld_specs_0 = next(iter(fld_specs_0_lst))

        # Deposition type ("wet", "dry")
        setup1 = next(iter(self.setups)).derive({"deposition_type": ("wet", "dry")})
        fld_specs_1_lst = FldSpecs.create(setup1)
        assert len(fld_specs_1_lst) == 1
        fld_specs_1 = next(iter(fld_specs_1_lst))
        assert fld_specs_1 == fld_specs_0  # ("wet", "dry") == "tot"

        # Deposition type "wet"
        setup2 = next(iter(self.setups)).derive({"deposition_type": "wet"})
        fld_specs_2_lst = FldSpecs.create(setup2)
        assert len(fld_specs_2_lst) == 1
        fld_specs_2 = next(iter(fld_specs_2_lst))
        assert fld_specs_2 != fld_specs_0  # "tot" != "wet"
        assert fld_specs_2 != fld_specs_1  # "tot" != ("wet", "dry")


class Test_FldSpecs_Interface:
    c = Conf_Multi(
        dct={
            "deposition": "tot",
            "integrate": False,
            "level": None,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": (2,),
            "time": 3,
        },
        n=1,
    )
    base_setup = Setup(
        deposition_type="tot",
        infiles=["dummy.nc"],
        integrate=False,
        outfile="dummy.png",
        species_id=2,
        time=[3],
        variable="deposition",
    )
    setups = SetupCollection([base_setup])

    def create_fld_specs(self):
        setup = next(iter(self.setups))  # SR_TMP
        fld_specs_lst = FldSpecs.create(setup)
        assert len(fld_specs_lst) == 1
        fld_specs = next(iter(fld_specs_lst))
        assert len(fld_specs.var_specs_lst) == 2
        return fld_specs

    def test_var_specs(self):
        var_specs_lst = self.create_fld_specs().var_specs_lst

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

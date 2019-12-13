#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
import pytest

from dataclasses import dataclass

from pyflexplot.io import FieldSpecs
from pyflexplot.io import VarSpecs

from srutils.dict import dict_mult_vals_product
from srutils.various import isiterable


def not_isiterable(obj):
    return not isiterable(obj, str_ok=False)


# SR_TODO move into some module
def assert_is_list_like(obj, *, len_=None, not_=None, children=None, f_children=None):
    """Assert that an object is list-like, with optional additional checks.

    Args:
        obj (type): Presumably list-like object.

        len_ (int, optional): Length of list-like object. Defaults to None.

        not_ (type or list[type], optional): Type(s) that ``obj`` must not be
            an instance of. Defaults to None.

        children (type or list[type], optional): Type(s) that the elements in
            ``obj`` must be an instance of. Defaults to None.

        f_children (callable, optional): Function used in assert with each
            element in ``obj``. Defaults to None.

    """
    assert isiterable(obj, str_ok=False)

    if len_ is not None:
        assert len(obj) == len_

    if not_ is not None:
        assert not isinstance(obj, not_)

    if children is not None:
        for sub_obj in obj:
            assert isinstance(sub_obj, children)

    if f_children is not None:
        for sub_obj in obj:
            assert f_children(sub_obj)


def check_var_specs_lst_mult(var_specs_lst, conf):
    """Check a list of var specs objs against its's specification dict.

    The specification dict contains elements with multiple values, which
    are combined to create multiple var specs objects, each one based on
    one unique combination of those values. The specification dict thus
    must be expanded into multiple dicts, each of which containing such a
    unique combination of the values in the lists.

    Because we don't know the order of the expansion, we first produce all
    possible specification dicts. Then, for each var specs object, we check
    in turn all specification dict that have not yet matched a var specs
    object. When the var specs object matches a dict, we remove that dict
    from the list of dicts to be checked, and continue with the next var
    specs object. If an object matches none of the still available dicts,
    the test has failed.

    """
    var_specs_dct_lst_todo = dict_mult_vals_product(conf.var_specs_dct)
    for var_specs in var_specs_lst:
        exn = None
        for var_specs_dct in var_specs_dct_lst_todo.copy():
            try:
                check_var_specs(conf.name, conf.type_name, var_specs, var_specs_dct)
            except AssertionError as e:
                exn = e
                continue
            else:
                var_specs_dct_lst_todo.remove(var_specs_dct)
                break
        else:
            raise AssertionError(
                f"no matching solution found for {var_specs} among "
                f"{var_specs_dct_lst_todo}"
            ) from exn


def check_var_specs(name, type_name, var_specs, var_specs_dct):
    """Compare a var specs object with a var specs dict."""

    assert type(var_specs).__name__ == type_name  # SR_TMP

    sol = set(var_specs_dct.items())
    res = set(dict(var_specs).items())
    assert sol.issubset(res)

    sol = {
        **var_specs_dct,
        "name": name,
        "rlon": (None,),
        "rlat": (None,),
    }
    res = dict(var_specs)
    assert sol == res


class _ConfBase:
    def derive(self, **kwargs):
        fields = self.__dataclass_fields__
        fields.update(kwargs)
        return type(self)(**fields)


@dataclass(frozen=True)
class Conf_CreateVarSpecs(_ConfBase):
    name: str
    type_name: str
    var_specs_dct: dict
    kwargs: dict


@dataclass(frozen=True)
class Conf_CreateFieldSpecs(_ConfBase):
    name: str
    type_name: str
    var_specs_dct_base: dict
    var_specs_kwargs: dict


class Test_CreateVarSpecs_SingleObjDct_Concentration:
    """Create a variable specification object for concentration."""

    c = Conf_CreateVarSpecs(
        name="concentration",
        type_name="VarSpecs_Concentration",  # SR_TMP
        var_specs_dct={
            "nageclass": 0,
            "numpoint": 0,
            "time": 3,
            "integrate": False,
            "species_id": 2,
            "level": 1,
        },
        kwargs={"rlon": None, "rlat": None, "lang": None, "words": None},
    )

    def test(self):
        var_specs_lst = VarSpecs.create(
            self.c.name, self.c.var_specs_dct, **self.c.kwargs
        )
        assert_is_list_like(var_specs_lst, len_=1, children=VarSpecs)
        var_specs = next(iter(var_specs_lst))
        check_var_specs(self.c.name, self.c.type_name, var_specs, self.c.var_specs_dct)


class Test_CreateVarSpecs_SingleObjDct_Deposition:
    """Create a variable specification object for deposition."""

    c = Conf_CreateVarSpecs(
        name="deposition",
        type_name="VarSpecs_Deposition",  # SR_TMP
        var_specs_dct={
            "nageclass": 0,
            "numpoint": 0,
            "time": 3,
            "integrate": False,
            "species_id": 2,
            "deposition": "tot",
        },
        kwargs={"rlon": None, "rlat": None, "lang": None, "words": None},
    )

    def test_(self):
        var_specs_lst = VarSpecs.create(
            self.c.name, self.c.var_specs_dct, **self.c.kwargs
        )
        assert_is_list_like(var_specs_lst, len_=1, children=VarSpecs)
        var_specs = next(iter(var_specs_lst))
        check_var_specs(self.c.name, self.c.type_name, var_specs, self.c.var_specs_dct)


class Test_CreateVarSpecs_MultObjDct_Concentration:
    """Create multiple variable specification objects for concentration."""

    c = Conf_CreateVarSpecs(
        name="concentration",
        type_name="VarSpecs_Concentration",  # SR_TMP
        var_specs_dct={
            "nageclass": 0,
            "numpoint": 0,
            "time_lst": [1, 3],
            "integrate": False,
            "species_id_lst": [1, 2],
            "level": 1,
        },
        kwargs={"rlon": None, "rlat": None, "lang": None, "words": None},
    )

    def test(self):
        var_specs_lst = VarSpecs.create(
            self.c.name, self.c.var_specs_dct, **self.c.kwargs,
        )
        assert_is_list_like(var_specs_lst, len_=4, children=VarSpecs)
        check_var_specs_lst_mult(var_specs_lst, self.c)

    def test_fail(self):
        var_specs_lst = VarSpecs.create(
            self.c.name, self.c.var_specs_dct, **self.c.kwargs,
        )
        assert_is_list_like(var_specs_lst, children=VarSpecs)
        conf = self.c.derive(
            var_specs_dct={**self.c.var_specs_dct, "time_lst": [3, 4], "level": 0}
        )
        with pytest.raises(AssertionError):
            check_var_specs_lst_mult(var_specs_lst, conf)


class Test_FieldSpecs_Create_Concentration:

    c = Conf_CreateFieldSpecs(
        name="concentration",
        type_name="FieldSpecs_Concentration",  # SR_TMP
        var_specs_dct_base={
            "nageclass": 0,
            "numpoint": 0,
            "time": 1,
            "integrate": False,
        },
        var_specs_kwargs={"rlon": None, "rlat": None, "lang": None, "words": None},
    )

    def test_single_var_specs_one_fld_one_var(self):
        """Single-value-only var specs, for one field, made of one var."""
        var_specs_dct = {
            **self.c.var_specs_dct_base,
            "species_id": 1,
            "level": 0,
        }
        var_specs_lst = VarSpecs.create(
            self.c.name, var_specs_dct, **self.c.var_specs_kwargs,
        )
        assert_is_list_like(var_specs_lst, len_=1, children=VarSpecs)
        fld_specs_lst = [
            FieldSpecs(self.c.name, [var_specs]) for var_specs in var_specs_lst
        ]
        assert_is_list_like(fld_specs_lst, len_=1, children=FieldSpecs)

        fld_specs = next(iter(fld_specs_lst))
        assert fld_specs.name == self.c.name
        assert len(fld_specs.var_specs_lst) == 1

    def test_mult_var_specs_many_flds_one_var_each(self):
        """Multi-value var specs, for multiple fields, made of one var each."""
        var_specs_dct = {
            **self.c.var_specs_dct_base,
            "species_id_lst": [1, 2],
            "level_lst": [0, 1],
        }
        var_specs_lst = VarSpecs.create(
            self.c.name, var_specs_dct, **self.c.var_specs_kwargs,
        )
        assert_is_list_like(var_specs_lst, len_=4, children=VarSpecs)
        fld_specs_lst = [
            FieldSpecs(self.c.name, [var_specs]) for var_specs in var_specs_lst
        ]
        assert_is_list_like(fld_specs_lst, len_=4, children=FieldSpecs)

        for fld_specs in fld_specs_lst:
            assert fld_specs.name == self.c.name
            assert len(fld_specs.var_specs_lst) == 1

    def test_mult_var_specs_one_fld_many_vars(self):
        """Multi-value var specs, for one field, made of multiple vars."""
        var_specs_dct = {
            **self.c.var_specs_dct_base,
            "species_id_lst": [1, 2],
            "level_lst": [0, 1],
        }
        var_specs_lst = VarSpecs.create(
            self.c.name, var_specs_dct, **self.c.var_specs_kwargs,
        )
        assert_is_list_like(var_specs_lst, len_=4, children=VarSpecs)
        fld_specs_lst = [FieldSpecs(self.c.name, var_specs_lst)]
        assert_is_list_like(fld_specs_lst, len_=1, children=FieldSpecs)

        fld_specs = next(iter(fld_specs_lst))
        assert fld_specs.name == self.c.name
        assert len(fld_specs.var_specs_lst) == 4


class Test_FieldSpecs_Create_Deposition:

    c = Conf_CreateFieldSpecs(
        name="deposition",
        type_name="FieldSpecs_Deposition",  # SR_TMP
        var_specs_dct_base={
            "nageclass": 0,
            "numpoint": 0,
            "integrate": False,
            "deposition": "tot",
        },
        var_specs_kwargs={"rlon": None, "rlat": None, "lang": None, "words": None},
    )

    def test_single_var_specs_one_fld_one_var(self):
        """Single-value-only var specs, for one field, made of one var."""
        var_specs_dct = {
            **self.c.var_specs_dct_base,
            "time": 1,
            "species_id": 1,
        }
        var_specs_lst = VarSpecs.create(
            self.c.name, var_specs_dct, **self.c.var_specs_kwargs,
        )
        assert_is_list_like(var_specs_lst, len_=1, children=VarSpecs)
        fld_specs_lst = [
            FieldSpecs(self.c.name, [var_specs]) for var_specs in var_specs_lst
        ]
        assert_is_list_like(fld_specs_lst, len_=1, children=FieldSpecs)

        fld_specs = next(iter(fld_specs_lst))
        assert fld_specs.name == self.c.name
        assert len(fld_specs.var_specs_lst) == 1

    def test_mult_var_specs_many_flds_one_var_each(self):
        """Multi-value var specs, for multiple fields, made of one var each."""
        var_specs_dct = {
            **self.c.var_specs_dct_base,
            "time_lst": [0, 1, 2],
            "species_id_lst": [1, 2],
        }
        var_specs_lst = VarSpecs.create(
            self.c.name, var_specs_dct, **self.c.var_specs_kwargs,
        )
        assert_is_list_like(var_specs_lst, len_=6, children=VarSpecs)
        fld_specs_lst = [
            FieldSpecs(self.c.name, [var_specs]) for var_specs in var_specs_lst
        ]
        assert_is_list_like(fld_specs_lst, len_=6, children=FieldSpecs)

        for fld_specs in fld_specs_lst:
            assert fld_specs.name == self.c.name
            assert len(fld_specs.var_specs_lst) == 1

    def test_mult_var_specs_one_fld_many_vars(self):
        """Multi-value var specs, for one field, made of multiple vars."""
        var_specs_dct = {
            **self.c.var_specs_dct_base,
            "time_lst": [0, 1, 2],
            "species_id_lst": [1, 2],
        }
        var_specs_lst = VarSpecs.create(
            self.c.name, var_specs_dct, **self.c.var_specs_kwargs,
        )
        assert_is_list_like(var_specs_lst, len_=6, children=VarSpecs)
        fld_specs_lst = [FieldSpecs(self.c.name, var_specs_lst)]
        assert_is_list_like(fld_specs_lst, len_=1, children=FieldSpecs)

        fld_specs = next(iter(fld_specs_lst))
        assert fld_specs.name == self.c.name
        assert len(fld_specs.var_specs_lst) == 6

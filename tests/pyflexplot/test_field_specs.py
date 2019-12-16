#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.specs``."""
import pytest

from dataclasses import dataclass

from pyflexplot.io import VarSpecs
from pyflexplot.io import FieldSpecs

from srutils.testing import assert_is_list_like
from srutils.testing import TestConfBase
from srutils.various import isiterable


@dataclass(frozen=True)
class Conf_CreateFieldSpecs(TestConfBase):
    name: str
    type_name: str
    var_specs_dct_base: dict
    var_specs_kwargs: dict


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

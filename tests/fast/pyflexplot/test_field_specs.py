#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.field_specs``.
"""
# Standard library
from dataclasses import dataclass

# First-party
from pyflexplot.field_specs import FieldSpecs
from pyflexplot.setup import Setup
from pyflexplot.var_specs import MultiVarSpecs
from srutils.testing import TestConfBase as _TestConf
from srutils.testing import check_is_list_like


def check_multi_var_specs_lst(multi_var_specs_lst, len_=None):
    check_is_list_like(multi_var_specs_lst, len_=len_, t_children=MultiVarSpecs)


@dataclass(frozen=True)
class Conf_Create(_TestConf):
    name: str
    n_vs: int


class Test_Create_Concentration:

    c = Conf_Create(name="concentration", n_vs=1,)
    setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="concentration",
        integrate=False,
        time_idcs=[1],
    )

    def test_single_var_specs_one_fld_one_var(self):
        """Single-value-only var specs, for one field, made of one var."""
        setup = self.setup.derive({"species_id": 1, "level_idx": 0})
        multi_var_specs_lst = MultiVarSpecs.from_setup(setup)
        check_multi_var_specs_lst(multi_var_specs_lst, len_=self.c.n_vs)
        fld_specs_lst = [
            FieldSpecs(self.c.name, multi_var_specs)
            for multi_var_specs in multi_var_specs_lst
        ]
        check_is_list_like(fld_specs_lst, len_=self.c.n_vs, t_children=FieldSpecs)

        fld_specs = next(iter(fld_specs_lst))
        assert fld_specs.name == self.c.name
        assert len(fld_specs.multi_var_specs) == 1

    def test_mult_var_specs_many_flds_one_var_each(self):
        """Multi-value var specs, for multiple fields, made of one var each."""
        setups = self.setup.derive(
            [
                {"species_id": 1, "level_idx": 0},
                {"species_id": 1, "level_idx": 1},
                {"species_id": 2, "level_idx": 0},
                {"species_id": 2, "level_idx": 1},
            ]
        )
        multi_var_specs_lst = MultiVarSpecs.from_setups(setups)
        check_multi_var_specs_lst(multi_var_specs_lst, len_=4)
        fld_specs_lst = [
            FieldSpecs(self.c.name, multi_var_specs)
            for multi_var_specs in multi_var_specs_lst
        ]
        check_is_list_like(fld_specs_lst, len_=4, t_children=FieldSpecs)

        for fld_specs in fld_specs_lst:
            assert fld_specs.name == self.c.name
            assert len(fld_specs.multi_var_specs) == 1

    def test_mult_var_specs_one_fld_many_vars(self):
        """Multi-value var specs, for one field, made of multiple vars."""
        setup = self.setup.derive({"species_id": (1, 2), "level_idx": (0, 1)})
        multi_var_specs_lst = MultiVarSpecs.from_setup(setup)
        check_multi_var_specs_lst(multi_var_specs_lst, len_=1)
        fld_specs_lst = [
            FieldSpecs(self.c.name, multi_var_specs)
            for multi_var_specs in multi_var_specs_lst
        ]
        check_is_list_like(fld_specs_lst, len_=1, t_children=FieldSpecs)

        fld_specs = next(iter(fld_specs_lst))
        assert fld_specs.name == self.c.name
        assert len(fld_specs.multi_var_specs) == 4


class Test_Create_Deposition:

    c = Conf_Create(name="deposition", n_vs=1,)
    setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="deposition",
        deposition_type="dry",
        integrate=False,
    )

    def test_single_var_specs_one_fld_one_var(self):
        """Single-value-only var specs, for one field, made of one var."""
        setup = self.setup.derive({"time_idcs": [1], "species_id": 1})
        multi_var_specs_lst = MultiVarSpecs.from_setup(setup)
        check_multi_var_specs_lst(multi_var_specs_lst, len_=1)
        fld_specs_lst = [
            FieldSpecs(self.c.name, multi_var_specs)
            for multi_var_specs in multi_var_specs_lst
        ]
        check_is_list_like(fld_specs_lst, len_=1, t_children=FieldSpecs)

        fld_specs = next(iter(fld_specs_lst))
        assert fld_specs.name == self.c.name
        assert len(fld_specs.multi_var_specs) == self.c.n_vs

    def test_mult_var_specs_many_flds_one_var_each(self):
        """Multi-value var specs, for multiple fields, made of one var each."""
        n = 6
        # SR_TMP <
        setups = self.setup.derive(
            [
                {"time_idcs": [0, 1, 2], "species_id": 1},
                {"time_idcs": [0, 1, 2], "species_id": 2},
            ]
        )
        multi_var_specs_lst = []
        for setup in setups:
            multi_var_specs_lst.extend(MultiVarSpecs.from_setup(setup))
        # SR_TMP >
        check_multi_var_specs_lst(multi_var_specs_lst, len_=n)
        fld_specs_lst = [
            FieldSpecs(self.c.name, multi_var_specs)
            for multi_var_specs in multi_var_specs_lst
        ]
        check_is_list_like(fld_specs_lst, len_=n, t_children=FieldSpecs)

        for fld_specs in fld_specs_lst:
            assert fld_specs.name == self.c.name
            assert len(fld_specs.multi_var_specs) == self.c.n_vs

    def test_mult_var_specs_one_fld_many_vars(self):
        """Multi-value var specs, for one field, made of multiple vars."""
        n_vs = self.c.n_vs * 4
        setup = self.setup.derive({"deposition_type": "tot", "species_id": [1, 2]})
        multi_var_specs_lst = MultiVarSpecs.from_setup(setup)
        check_multi_var_specs_lst(multi_var_specs_lst, len_=1)
        multi_var_specs = next(iter(multi_var_specs_lst))
        fld_specs = FieldSpecs(self.c.name, multi_var_specs)
        assert fld_specs.name == self.c.name
        assert len(fld_specs.multi_var_specs) == n_vs

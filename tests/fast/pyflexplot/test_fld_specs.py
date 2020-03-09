#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.fld_specs``.
"""
# First-party
from pyflexplot.setup import Setup
from pyflexplot.specs import FldSpecs
from srutils.testing import check_is_list_like


class Test_Create_Concentration:

    setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="concentration",
        integrate=False,
        time_idcs=[1],
    )
    n_vs = 1

    def test_single_var_specs_one_fld_one_var(self):
        """Single-value-only var specs, for one field, made of one var."""
        setup = self.setup.derive({"species_id": 1, "level_idx": 0})
        fld_specs_lst = FldSpecs.create(setup)
        check_is_list_like(fld_specs_lst, len_=self.n_vs, t_children=FldSpecs)
        fld_specs = next(iter(fld_specs_lst))
        assert len(fld_specs) == 1

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
        fld_specs_lst = FldSpecs.create(setups)
        check_is_list_like(fld_specs_lst, len_=4, t_children=FldSpecs)
        for fld_specs in fld_specs_lst:
            assert len(fld_specs) == 1

    def test_mult_var_specs_one_fld_many_vars(self):
        """Multi-value var specs, for one field, made of multiple vars."""
        setup = self.setup.derive({"species_id": (1, 2), "level_idx": (0, 1)})
        fld_specs_lst = FldSpecs.create(setup)
        check_is_list_like(fld_specs_lst, len_=1, t_children=FldSpecs)
        fld_specs = next(iter(fld_specs_lst))
        assert len(fld_specs) == 4


class Test_Create_Deposition:

    setup = Setup(
        infiles=["dummy.nc"],
        outfile="dummy.png",
        variable="deposition",
        deposition_type="dry",
        integrate=False,
    )
    n_vs = 1

    def test_single_var_specs_one_fld_one_var(self):
        """Single-value-only var specs, for one field, made of one var."""
        setup = self.setup.derive({"time_idcs": [1], "species_id": 1})
        fld_specs_lst = FldSpecs.create(setup)
        check_is_list_like(fld_specs_lst, len_=1, t_children=FldSpecs)
        fld_specs = next(iter(fld_specs_lst))
        assert len(fld_specs) == self.n_vs

    def test_mult_var_specs_many_flds_one_var_each(self):
        """Multi-value var specs, for multiple fields, made of one var each."""
        n = 6
        setups = self.setup.derive(
            [
                {"time_idcs": [0, 1, 2], "species_id": 1},
                {"time_idcs": [0, 1, 2], "species_id": 2},
            ]
        )
        fld_specs_lst = FldSpecs.create(setups)
        check_is_list_like(fld_specs_lst, len_=n, t_children=FldSpecs)
        for fld_specs in fld_specs_lst:
            assert len(fld_specs) == self.n_vs

    def test_mult_var_specs_one_fld_many_vars(self):
        """Multi-value var specs, for one field, made of multiple vars."""
        n_vs = self.n_vs * 4
        setup = self.setup.derive({"deposition_type": "tot", "species_id": [1, 2]})
        fld_specs_lst = FldSpecs.create(setup)
        assert len(fld_specs_lst) == 1
        fld_specs = next(iter(fld_specs_lst))
        assert len(fld_specs) == n_vs

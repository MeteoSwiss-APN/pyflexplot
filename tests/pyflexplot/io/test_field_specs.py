#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
import pytest

from pyflexplot.io import FieldSpecs

from srutils.dict import dict_mult_vals_product


class TestFieldSpecs_Multiple:
    """Create multiple field specifications."""

    # Dimensions arguments shared by all tests
    dims_shared = {
        'nageclass': 0,
        'numpoint': 0,
        'time_lst': [0, 3, 4, 5, 9],
    }

    # Variables specification arguments shared by all tests
    var_specs_mult_shared = {
        'integrate_lst': [False, True],
        'species_id_lst': [1, 2],
    }

    @property
    def species_id_lst(self):
        return self.var_specs_mult_shared['species_id_lst']

    #------------------------------------------------------------------

    def create_fld_specs_mult_lst_ref(self, cls_fld_specs, vars_specs_mult):

        # Create all variable specifications combinations
        var_specs_lst = dict_mult_vals_product(vars_specs_mult)

        # Create field specifications
        fld_specs_mult_lst = [cls_fld_specs(vs) for vs in var_specs_lst]

        return fld_specs_mult_lst

    #------------------------------------------------------------------

    def test_concentration(self):

        # Create field specifications list
        var_specs_mult = {
            **self.dims_shared,
            **self.var_specs_mult_shared,
            'level_lst': [0, 2],
        }
        fld_specs_mult_lst = FieldSpecs.subclass('concentration').multiple(
            var_specs_mult)

        # Create reference field specifications list
        fld_specs_mult_lst_ref = self.create_fld_specs_mult_lst_ref(
            FieldSpecs.subclass('concentration'), var_specs_mult)

        assert sorted(fld_specs_mult_lst) == sorted(fld_specs_mult_lst_ref)

    def test_deposition(self):

        # Create field specifications list
        var_specs_mult = {
            **self.dims_shared,
            **self.var_specs_mult_shared,
            'deposition_lst': ['wet', 'dry', 'tot'],
        }
        fld_specs_mult_lst = FieldSpecs.subclass('deposition').multiple(
            var_specs_mult)

        # Create reference field specifications list
        fld_specs_mult_lst_ref = self.create_fld_specs_mult_lst_ref(
            FieldSpecs.subclass('deposition'), var_specs_mult)

        assert sorted(fld_specs_mult_lst) == sorted(fld_specs_mult_lst_ref)

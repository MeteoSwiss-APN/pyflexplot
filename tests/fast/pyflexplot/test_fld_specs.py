# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.???``.
"""
# First-party
from pyflexplot.setup import InputSetup
from pyflexplot.setup import InputSetupCollection
from srutils.testing import check_is_list_like

# SR_TODO Turn these into meaningful tests for SetupCollection!!!


class Test_Create_Concentration:

    setup = InputSetup.create(
        {
            "infile": "dummy.nc",
            "outfile": "dummy.png",
            "input_variable": "concentration",
            "integrate": False,
            "dimensions": {"time": 1},
        },
    )
    n_vs = 1

    def test_one_fld_one_var(self):
        """Single-value-only var specs, for one field, made of one var."""
        setup = self.setup.derive({"dimensions": {"species_id": 1, "level": 0}})
        setups = InputSetupCollection([setup])
        var_setups_lst = setups.decompress_twice(
            "dimensions.time", skip=["ens_member_id"]
        )
        check_is_list_like(
            var_setups_lst, len_=self.n_vs, t_children=InputSetupCollection,
        )
        var_setups = next(iter(var_setups_lst))
        assert len(var_setups) == 1

    def test_many_flds_one_var_each(self):
        """Multi-value var specs, for multiple fields, made of one var each."""
        setups = self.setup.derive(
            [
                {"dimensions": {"species_id": 1, "level": 0}},
                {"dimensions": {"species_id": 1, "level": 1}},
                {"dimensions": {"species_id": 2, "level": 0}},
                {"dimensions": {"species_id": 2, "level": 1}},
            ]
        )
        var_setups_lst = setups.decompress_twice(
            "dimensions.time", skip=["ens_member_id"]
        )
        check_is_list_like(var_setups_lst, len_=4, t_children=InputSetupCollection)
        for var_setups in var_setups_lst:
            assert len(var_setups) == 1

    def test_one_fld_many_vars(self):
        """Multi-value var specs, for one field, made of multiple vars."""
        setup = self.setup.derive(
            {"dimensions": {"species_id": (1, 2), "level": (0, 1)}}
        )
        setups = InputSetupCollection([setup])
        var_setups_lst = setups.decompress_twice(
            "dimensions.time", skip=["ens_member_id"]
        )
        check_is_list_like(var_setups_lst, len_=1, t_children=InputSetupCollection)
        var_setups = next(iter(var_setups_lst))
        assert len(var_setups) == 4


class Test_Create_Deposition:

    setup = InputSetup.create(
        {
            "infile": "dummy.nc",
            "outfile": "dummy.png",
            "input_variable": "deposition",
            "deposition_type": "dry",
            "integrate": False,
        },
    )
    n_vs = 1

    def test_one_fld_one_var(self):
        """Single-value-only var specs, for one field, made of one var."""
        setup = self.setup.derive({"dimensions": {"time": 1, "species_id": 1}})
        setups = InputSetupCollection([setup])
        var_setups_lst = setups.decompress_twice(
            "dimensions.time", skip=["ens_member_id"]
        )
        check_is_list_like(var_setups_lst, len_=1, t_children=InputSetupCollection)
        var_setups = next(iter(var_setups_lst))
        assert len(var_setups) == self.n_vs

    def test_many_flds_one_var_each(self):
        """Multi-value var specs, for multiple fields, made of one var each."""
        n = 6
        setups = self.setup.derive(
            [
                {"dimensions": {"time": [0, 1, 2], "species_id": 1}},
                {"dimensions": {"time": [0, 1, 2], "species_id": 2}},
            ]
        )
        var_setups_lst = setups.decompress_twice(
            "dimensions.time", skip=["ens_member_id"]
        )
        check_is_list_like(var_setups_lst, len_=n, t_children=InputSetupCollection)
        for var_setups in var_setups_lst:
            assert len(var_setups) == self.n_vs

    def test_one_fld_many_vars(self):
        """Multi-value var specs, for one field, made of multiple vars."""
        n_vs = self.n_vs * 4
        setup = self.setup.derive(
            {
                "deposition_type": ("dry", "wet"),
                "combine_deposition_types": True,
                "dimensions": {"species_id": [1, 2]},
            }
        )
        setups = InputSetupCollection([setup])
        var_setups_lst = setups.decompress_twice(
            "dimensions.time", skip=["ens_member_id"]
        )
        check_is_list_like(var_setups_lst, len_=1, t_children=InputSetupCollection)
        var_setups = next(iter(var_setups_lst))
        assert len(var_setups) == n_vs

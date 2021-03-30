"""Tests for ``pyflexplot.setups.plot_panel_setup.PlotPanelSetupGroup``."""
# Third-party
import pytest

# First-party
from pyflexplot.setups.plot_panel_setup import PlotPanelSetup
from pyflexplot.setups.plot_panel_setup import PlotPanelSetupGroup
from srutils.dict import merge_dicts
from srutils.testing import assert_is_sub_element
from srutils.testing import assert_nested_equal


class Test_Create:
    class Test_SinglePanel:
        params = {
            "plot_variable": "dry_deposition",
            "dimensions": {
                "species_id": 1,
            },
        }

        def test_params_dict(self):
            group = PlotPanelSetupGroup.create(self.params)
            res = group.dicts()
            sol = [self.params]
            assert_is_sub_element(
                name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
            )

        def test_params_list(self):
            group = PlotPanelSetupGroup.create([self.params])
            assert_is_sub_element(
                name_super="result",
                obj_super=group.dicts(),
                name_sub="solution",
                obj_sub=[self.params],
            )

        def test_multipanel_fail(self):
            with pytest.raises(ValueError, match="multipanel_param"):
                PlotPanelSetupGroup.create(self.params, multipanel_param="species_id")

    class Test_MultiPanel:
        class Test_EnsParamsParam:
            params = {
                "plot_variable": "wet_deposition",
                "ens_variable": "percentile",
                "ens_params": {"pctl": (30, 50, 70, 90)},
            }

            def no_multival_fail(self):
                with pytest.raises(ValueError, match="multipanel_param"):
                    PlotPanelSetupGroup.create(self.params)
                with pytest.raises(ValueError, match="multipanel_param"):
                    PlotPanelSetupGroup.create([self.params])

            def test_params_dict(self):
                group = PlotPanelSetupGroup.create(
                    self.params, multipanel_param="ens_params.pctl"
                )
                res = group.dicts()
                sol = [
                    {"ens_params": {"pctl": 30}},
                    {"ens_params": {"pctl": 50}},
                    {"ens_params": {"pctl": 70}},
                    {"ens_params": {"pctl": 90}},
                ]
                assert_is_sub_element(
                    name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
                )

            def test_params_list_vs_dict(self):
                group_dict = PlotPanelSetupGroup.create(
                    self.params, multipanel_param="ens_params.pctl"
                )
                params_lst = [
                    merge_dicts(
                        self.params,
                        {"ens_params": {"pctl": val}},
                        overwrite_seqs=True,
                    )
                    for val in self.params["ens_params"]["pctl"]
                ]
                group_list = PlotPanelSetupGroup.create(
                    params_lst, multipanel_param="ens_params.pctl"
                )
                res = group_list.dicts()
                sol = group_dict.dicts()
                assert_nested_equal(res, sol)

        class Test_DimensionsParam:
            class Test_SpeciesID:
                params = {
                    "plot_variable": "tot_deposition",
                    "dimensions": {
                        "species_id": (1, 2, 3, 4),
                        "time": (0, 4, 8, 12),
                    },
                }

                def test_params_dict(self):
                    group = PlotPanelSetupGroup.create(
                        self.params, multipanel_param="dimensions.species_id"
                    )
                    res = group.dicts()
                    sol = [
                        {"dimensions": {"species_id": 1, "time": (0, 4, 8, 12)}},
                        {"dimensions": {"species_id": 2, "time": (0, 4, 8, 12)}},
                        {"dimensions": {"species_id": 3, "time": (0, 4, 8, 12)}},
                        {"dimensions": {"species_id": 4, "time": (0, 4, 8, 12)}},
                    ]
                    assert_is_sub_element(
                        name_super="result",
                        obj_super=res,
                        name_sub="solution",
                        obj_sub=sol,
                    )

                def test_params_list_vs_dict(self):
                    group_dict = PlotPanelSetupGroup.create(
                        self.params, multipanel_param="species_id"
                    )
                    params_lst = [
                        merge_dicts(
                            self.params,
                            {"dimensions": {"species_id": val}},
                            overwrite_seqs=True,
                        )
                        for val in self.params["dimensions"]["species_id"]
                    ]
                    group_list = PlotPanelSetupGroup.create(
                        params_lst, multipanel_param="dimensions.species_id"
                    )
                    res = group_list.dicts()
                    sol = group_dict.dicts()
                    assert_nested_equal(res, sol)

            class Test_Time:
                params = {
                    "plot_variable": "concentration",
                    "combine_species": True,
                    "dimensions": {
                        "species_id": (1, 2, 3, 4),
                        "time": (0, 4, 8, 12),
                    },
                }

                def test_params_dict_time(self):
                    group = PlotPanelSetupGroup.create(
                        self.params, multipanel_param="time"
                    )
                    res = group.dicts()
                    sol = [
                        {"dimensions": {"species_id": (1, 2, 3, 4), "time": 0}},
                        {"dimensions": {"species_id": (1, 2, 3, 4), "time": 4}},
                        {"dimensions": {"species_id": (1, 2, 3, 4), "time": 8}},
                        {"dimensions": {"species_id": (1, 2, 3, 4), "time": 12}},
                    ]
                    assert_is_sub_element(
                        name_super="result",
                        obj_super=res,
                        name_sub="solution",
                        obj_sub=sol,
                    )

                def test_params_list_vs_dict(self):
                    group_dict = PlotPanelSetupGroup.create(
                        self.params, multipanel_param="dimensions.time"
                    )
                    params_lst = [
                        merge_dicts(
                            self.params,
                            {"dimensions": {"time": val}},
                            overwrite_seqs=True,
                        )
                        for val in self.params["dimensions"]["time"]
                    ]
                    group_list = PlotPanelSetupGroup.create(
                        params_lst, multipanel_param="time"
                    )
                    res = group_list.dicts()
                    sol = group_dict.dicts()
                    assert_nested_equal(res, sol)

        class Test_PanelParam:
            params = {
                "plot_variable": (
                    "concentration",
                    "tot_deposition",
                    "dry_deposition",
                    "wet_deposition",
                ),
                "combine_species": True,
                "dimensions": {
                    "species_id": (1, 2, 3, 4),
                    "time": (0, 4, 8, 12),
                },
            }

            def test_no_multival_fail(self):
                with pytest.raises(ValueError, match="multipanel_param"):
                    PlotPanelSetupGroup.create(self.params)
                with pytest.raises(ValueError, match="multipanel_param"):
                    PlotPanelSetupGroup.create([self.params])

            def test_params_dict(self):
                group = PlotPanelSetupGroup.create(
                    self.params, multipanel_param="plot_variable"
                )
                res = group.dicts()
                sol = [
                    {"plot_variable": "concentration"},
                    {"plot_variable": "tot_deposition"},
                    {"plot_variable": "dry_deposition"},
                    {"plot_variable": "wet_deposition"},
                ]
                assert_is_sub_element(
                    name_super="result",
                    obj_super=res,
                    name_sub="solution",
                    obj_sub=sol,
                )

            def test_params_list_vs_dict(self):
                group_dict = PlotPanelSetupGroup.create(
                    self.params, multipanel_param="plot_variable"
                )
                params_lst = [
                    {**self.params, "plot_variable": val}
                    for val in self.params["plot_variable"]
                ]
                group_list = PlotPanelSetupGroup.create(
                    params_lst, multipanel_param="plot_variable"
                )
                res = group_list.dicts()
                sol = group_dict.dicts()
                assert_nested_equal(res, sol)


class Test_Collect:
    base = {
        "files": {
            "input": "foo.nc",
            "output": "foo.png",
        },
        "model": {
            "name": "foo",
        },
    }
    md = lambda *dicts: merge_dicts(*dicts, overwrite_seqs=True)  # noqa
    params = {
        "plot_variable": "affected_area",
        "combine_levels": True,
        "combine_species": True,
        "dimensions": {
            "level": (0, 1),
            "species_id": (1, 1, 2, 3),
        },
    }

    @property
    def group(self) -> PlotPanelSetupGroup:
        return PlotPanelSetupGroup.create(
            self.params, multipanel_param="dimensions.species_id"
        )

    def test_variable(self):
        vals = self.group.collect("dimensions.variable")
        assert vals == [("concentration", "dry_deposition", "wet_deposition")] * 4

    def test_variable_fact_flat(self):
        vals = self.group.collect("dimensions.variable", flatten=True)
        assert vals == [("concentration", "dry_deposition", "wet_deposition")] * 4

    def test_variable_flat_unique(self):
        vals = self.group.collect("dimensions.variable", flatten=True, unique=True)
        assert vals == [("concentration", "dry_deposition", "wet_deposition")]

    def test_species_id(self):
        vals = self.group.collect("dimensions.species_id")
        assert vals == [1, 1, 2, 3]

    def test_species_id_flat(self):
        vals = self.group.collect("dimensions.species_id", flatten=True)
        assert vals == [1, 1, 2, 3]

    def test_species_id_flat_unique(self):
        vals = self.group.collect("dimensions.species_id", flatten=True, unique=True)
        assert vals == [1, 2, 3]


class Test_Decompress:
    class Test_Base:
        params = {
            "plot_variable": "concentration",
            "combine_levels": True,
            "combine_species": True,
            "dimensions": {
                "level": (0, 1),
                "nageclass": (0,),
                "noutrel": (0,),
                "numpoint": (0,),
                "species_id": [1, 2],
                "time": [1, 2, 3],
            },
        }

        def test_internal_vs_default(self):
            """The option ``internal`` is true by default."""
            group = PlotPanelSetupGroup.create(self.params)
            group_internal = group.decompress(internal=True)
            group_default = group.decompress()
            assert group_internal == group_default

        def test_internal_vs_external(self):
            """Check types and content with option ``internal`` true or false."""
            group = PlotPanelSetupGroup.create(self.params)
            group_internal = group.decompress(internal=True)
            groups_external = group.decompress(internal=False)
            assert isinstance(group_internal, PlotPanelSetupGroup)
            assert isinstance(groups_external, list)
            assert all(
                isinstance(group, PlotPanelSetupGroup) for group in groups_external
            )
            assert all(len(group) == 1 for group in groups_external)
            dcts_internal = group_internal.dicts()
            dcts_external = [
                setup.dict() for group in groups_external for setup in group
            ]
            assert dcts_internal == dcts_external

        def test_full(self):
            group_in = PlotPanelSetupGroup.create(self.params)
            group_out = group_in.decompress()
            dcts = group_out.dicts()
            sol = [
                {"dimensions": {"level": 0, "species_id": 1, "time": 1}},
                {"dimensions": {"level": 0, "species_id": 1, "time": 2}},
                {"dimensions": {"level": 0, "species_id": 1, "time": 3}},
                {"dimensions": {"level": 0, "species_id": 2, "time": 1}},
                {"dimensions": {"level": 0, "species_id": 2, "time": 2}},
                {"dimensions": {"level": 0, "species_id": 2, "time": 3}},
                {"dimensions": {"level": 1, "species_id": 1, "time": 1}},
                {"dimensions": {"level": 1, "species_id": 1, "time": 2}},
                {"dimensions": {"level": 1, "species_id": 1, "time": 3}},
                {"dimensions": {"level": 1, "species_id": 2, "time": 1}},
                {"dimensions": {"level": 1, "species_id": 2, "time": 2}},
                {"dimensions": {"level": 1, "species_id": 2, "time": 3}},
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_full_internal(self):
            group_in = PlotPanelSetupGroup.create(self.params)
            groups_out = group_in.decompress(internal=False)
            assert isinstance(groups_out, list)
            assert all(isinstance(group, PlotPanelSetupGroup) for group in groups_out)
            group_out = group_in.decompress(internal=True)
            assert isinstance(group_out, PlotPanelSetupGroup)
            assert all(isinstance(setup, PlotPanelSetup) for setup in group_out)

        def test_skip(self):
            group_in = PlotPanelSetupGroup.create(self.params)
            group_out = group_in.decompress(skip=["time"])
            dcts = group_out.dicts()
            sol = [
                {"dimensions": {"level": 0, "species_id": 1, "time": [1, 2, 3]}},
                {"dimensions": {"level": 0, "species_id": 2, "time": [1, 2, 3]}},
                {"dimensions": {"level": 1, "species_id": 1, "time": [1, 2, 3]}},
                {"dimensions": {"level": 1, "species_id": 2, "time": [1, 2, 3]}},
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_skip_all(self):
            group_in = PlotPanelSetupGroup.create(self.params)
            group_out = group_in.decompress(skip=["level", "species_id", "time"])
            dcts = group_out.dicts()
            sol = [
                {
                    "dimensions": {
                        "level": [0, 1],
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                },
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_select(self):
            group_in = PlotPanelSetupGroup.create(self.params)
            group_out = group_in.decompress(select=["level"])
            dcts = group_out.dicts()
            sol = [
                {
                    "dimensions": {
                        "level": 0,
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                },
                {
                    "dimensions": {
                        "level": 1,
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                },
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_select_all(self):
            group_in = PlotPanelSetupGroup.create(self.params)
            group_out = group_in.decompress(select=["level", "species_id", "time"])
            dcts = group_out.dicts()
            sol = [
                {"dimensions": {"level": 0, "species_id": 1, "time": 1}},
                {"dimensions": {"level": 0, "species_id": 1, "time": 2}},
                {"dimensions": {"level": 0, "species_id": 1, "time": 3}},
                {"dimensions": {"level": 0, "species_id": 2, "time": 1}},
                {"dimensions": {"level": 0, "species_id": 2, "time": 2}},
                {"dimensions": {"level": 0, "species_id": 2, "time": 3}},
                {"dimensions": {"level": 1, "species_id": 1, "time": 1}},
                {"dimensions": {"level": 1, "species_id": 1, "time": 2}},
                {"dimensions": {"level": 1, "species_id": 1, "time": 3}},
                {"dimensions": {"level": 1, "species_id": 2, "time": 1}},
                {"dimensions": {"level": 1, "species_id": 2, "time": 2}},
                {"dimensions": {"level": 1, "species_id": 2, "time": 3}},
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_select_skip(self):
            group_in = PlotPanelSetupGroup.create(self.params)
            group_out = group_in.decompress(
                select=["level", "species_id", "time"], skip=["time"]
            )
            dcts = group_out.dicts()
            sol = [
                {"dimensions": {"level": 0, "species_id": 1, "time": [1, 2, 3]}},
                {"dimensions": {"level": 0, "species_id": 2, "time": [1, 2, 3]}},
                {"dimensions": {"level": 1, "species_id": 1, "time": [1, 2, 3]}},
                {"dimensions": {"level": 1, "species_id": 2, "time": [1, 2, 3]}},
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

    class Test_External:
        params = {
            "plot_variable": "concentration",
            "combine_levels": True,
            "combine_species": True,
            "dimensions": {
                "level": [4, 8],
                "nageclass": (0,),
                "noutrel": (0,),
                "numpoint": (0,),
                "species_id": [1, 2],
                "time": [1, 2, 3],
            },
        }

        def test_full(self):
            group = PlotPanelSetupGroup.create(self.params)
            groups = group.decompress(internal=False)
            dcts = [group.dicts() for group in groups]
            sol = [
                [{"dimensions": {"level": 4, "species_id": 1, "time": 1}}],
                [{"dimensions": {"level": 4, "species_id": 1, "time": 2}}],
                [{"dimensions": {"level": 4, "species_id": 1, "time": 3}}],
                [{"dimensions": {"level": 4, "species_id": 2, "time": 1}}],
                [{"dimensions": {"level": 4, "species_id": 2, "time": 2}}],
                [{"dimensions": {"level": 4, "species_id": 2, "time": 3}}],
                [{"dimensions": {"level": 8, "species_id": 1, "time": 1}}],
                [{"dimensions": {"level": 8, "species_id": 1, "time": 2}}],
                [{"dimensions": {"level": 8, "species_id": 1, "time": 3}}],
                [{"dimensions": {"level": 8, "species_id": 2, "time": 1}}],
                [{"dimensions": {"level": 8, "species_id": 2, "time": 2}}],
                [{"dimensions": {"level": 8, "species_id": 2, "time": 3}}],
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_skip(self):
            group = PlotPanelSetupGroup.create(self.params)
            groups = group.decompress(internal=False, skip=["time"])
            dcts = [group.dicts() for group in groups]
            sol = [
                [
                    {
                        "dimensions": {
                            "level": 4,
                            "species_id": 1,
                            "time": [1, 2, 3],
                        }
                    }
                ],
                [
                    {
                        "dimensions": {
                            "level": 4,
                            "species_id": 2,
                            "time": [1, 2, 3],
                        }
                    }
                ],
                [
                    {
                        "dimensions": {
                            "level": 8,
                            "species_id": 1,
                            "time": [1, 2, 3],
                        }
                    }
                ],
                [
                    {
                        "dimensions": {
                            "level": 8,
                            "species_id": 2,
                            "time": [1, 2, 3],
                        }
                    }
                ],
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_skip_all(self):
            group = PlotPanelSetupGroup.create(self.params)
            groups = group.decompress(
                internal=False, skip=["level", "species_id", "time"]
            )
            dcts = [group.dicts() for group in groups]
            sol = [
                [
                    {
                        "dimensions": {
                            "level": [4, 8],
                            "species_id": [1, 2],
                            "time": [1, 2, 3],
                        }
                    }
                ],
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_select(self):
            group = PlotPanelSetupGroup.create(self.params)
            groups = group.decompress(internal=False, select=["level"])
            dcts = [group.dicts() for group in groups]
            sol = [
                [
                    {
                        "dimensions": {
                            "level": 4,
                            "species_id": [1, 2],
                            "time": [1, 2, 3],
                        }
                    }
                ],
                [
                    {
                        "dimensions": {
                            "level": 8,
                            "species_id": [1, 2],
                            "time": [1, 2, 3],
                        }
                    }
                ],
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_select_all(self):
            group = PlotPanelSetupGroup.create(self.params)
            groups = group.decompress(
                internal=False, select=["level", "species_id", "time"]
            )
            dcts = [group.dicts() for group in groups]
            sol = [
                [{"dimensions": {"level": 4, "species_id": 1, "time": 1}}],
                [{"dimensions": {"level": 4, "species_id": 1, "time": 2}}],
                [{"dimensions": {"level": 4, "species_id": 1, "time": 3}}],
                [{"dimensions": {"level": 4, "species_id": 2, "time": 1}}],
                [{"dimensions": {"level": 4, "species_id": 2, "time": 2}}],
                [{"dimensions": {"level": 4, "species_id": 2, "time": 3}}],
                [{"dimensions": {"level": 8, "species_id": 1, "time": 1}}],
                [{"dimensions": {"level": 8, "species_id": 1, "time": 2}}],
                [{"dimensions": {"level": 8, "species_id": 1, "time": 3}}],
                [{"dimensions": {"level": 8, "species_id": 2, "time": 1}}],
                [{"dimensions": {"level": 8, "species_id": 2, "time": 2}}],
                [{"dimensions": {"level": 8, "species_id": 2, "time": 3}}],
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

        def test_select_skip(self):
            group = PlotPanelSetupGroup.create(self.params)
            groups = group.decompress(
                internal=False,
                select=["level", "species_id", "time"],
                skip=["time"],
            )
            dcts = [group.dicts() for group in groups]
            sol = [
                [
                    {
                        "dimensions": {
                            "level": 4,
                            "species_id": 1,
                            "time": [1, 2, 3],
                        }
                    }
                ],
                [
                    {
                        "dimensions": {
                            "level": 4,
                            "species_id": 2,
                            "time": [1, 2, 3],
                        }
                    }
                ],
                [
                    {
                        "dimensions": {
                            "level": 8,
                            "species_id": 1,
                            "time": [1, 2, 3],
                        }
                    }
                ],
                [
                    {
                        "dimensions": {
                            "level": 8,
                            "species_id": 2,
                            "time": [1, 2, 3],
                        }
                    }
                ],
            ]
            assert_is_sub_element(sol, dcts, "solution", "result")

    class Test_Multipanel:
        @property
        def group(self) -> PlotPanelSetupGroup:
            params = {
                "ens_variable": ["minimum", "maximum", "mean", "median"],
                "dimensions": {"time": [1, 2]},
            }
            return PlotPanelSetupGroup.create(params, multipanel_param="ens_variable")

        def test_full_internal(self):
            group = self.group.decompress(internal=True)
            assert isinstance(group, PlotPanelSetupGroup)
            res = group.dicts()
            sol = [
                {"ens_variable": "minimum", "dimensions": {"time": 1}},
                {"ens_variable": "minimum", "dimensions": {"time": 2}},
                {"ens_variable": "maximum", "dimensions": {"time": 1}},
                {"ens_variable": "maximum", "dimensions": {"time": 2}},
                {"ens_variable": "mean", "dimensions": {"time": 1}},
                {"ens_variable": "mean", "dimensions": {"time": 2}},
                {"ens_variable": "median", "dimensions": {"time": 1}},
                {"ens_variable": "median", "dimensions": {"time": 2}},
            ]
            assert_is_sub_element(sol, res, "solution", "result")

        def test_skip_internal(self):
            group = self.group.decompress(skip=["ens_variable"], internal=True)
            assert isinstance(group, PlotPanelSetupGroup)
            res = group.dicts()
            sol = [
                {"ens_variable": "minimum", "dimensions": {"time": 1}},
                {"ens_variable": "minimum", "dimensions": {"time": 2}},
                {"ens_variable": "maximum", "dimensions": {"time": 1}},
                {"ens_variable": "maximum", "dimensions": {"time": 2}},
                {"ens_variable": "mean", "dimensions": {"time": 1}},
                {"ens_variable": "mean", "dimensions": {"time": 2}},
                {"ens_variable": "median", "dimensions": {"time": 1}},
                {"ens_variable": "median", "dimensions": {"time": 2}},
            ]
            assert_is_sub_element(sol, res, "solution", "result")

        def test_full_external(self):
            groups = self.group.decompress(internal=False)
            assert isinstance(groups, list)
            res = [group.dicts() for group in groups]
            sol = [
                [{"ens_variable": "minimum", "dimensions": {"time": 1}}],
                [{"ens_variable": "minimum", "dimensions": {"time": 2}}],
                [{"ens_variable": "maximum", "dimensions": {"time": 1}}],
                [{"ens_variable": "maximum", "dimensions": {"time": 2}}],
                [{"ens_variable": "mean", "dimensions": {"time": 1}}],
                [{"ens_variable": "mean", "dimensions": {"time": 2}}],
                [{"ens_variable": "median", "dimensions": {"time": 1}}],
                [{"ens_variable": "median", "dimensions": {"time": 2}}],
            ]
            assert_is_sub_element(sol, res, "solution", "result")

        def test_skip_external(self):
            groups = self.group.decompress(skip=["ens_variable"], internal=False)
            assert isinstance(groups, list)
            res = [group.dicts() for group in groups]
            sol = [
                [
                    {"ens_variable": "minimum", "dimensions": {"time": 1}},
                    {"ens_variable": "maximum", "dimensions": {"time": 1}},
                    {"ens_variable": "mean", "dimensions": {"time": 1}},
                    {"ens_variable": "median", "dimensions": {"time": 1}},
                ],
                [
                    {"ens_variable": "minimum", "dimensions": {"time": 2}},
                    {"ens_variable": "maximum", "dimensions": {"time": 2}},
                    {"ens_variable": "mean", "dimensions": {"time": 2}},
                    {"ens_variable": "median", "dimensions": {"time": 2}},
                ],
            ]
            assert_is_sub_element(sol, res, "solution", "result")

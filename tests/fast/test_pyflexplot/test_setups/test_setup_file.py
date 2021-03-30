"""Test module ``pyflexplot.setup.SetupFile``."""
# Standard library
from collections.abc import Sequence
from textwrap import dedent

# First-party
from pyflexplot.setups.setup_file import SetupFile
from srutils.dict import merge_dicts
from srutils.testing import assert_is_sub_element

BASE = {
    "files": {
        "input": "foo.nc",
        # "output": "bar.png",
    },
    "outfile": "bar.png",
    "model": {"name": "COSMO-baz"},
}


def fmt_val(val):
    if isinstance(val, str):
        return f'"{val}"'
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, Sequence):
        return f"[{', '.join([fmt_val(v) for v in val])}]"
    else:
        raise NotImplementedError(f"type {type(val).__name__}", val)


def tmp_setup_file(tmp_path, content):
    tmp_file = tmp_path / "setup.toml"
    tmp_file.write_text(dedent(content))
    return tmp_file


class Test_Single:
    def test_minimal_section(self, tmp_path):
        """Read setup file with single minimal section."""
        content = """\
            [plot]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = setups.dicts()
        sol = [BASE]
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_minimal_renamed_section(self, tmp_path):
        """Read setup file with single minimal section with arbitrary name."""
        content = """\
            [foobar]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = setups.dicts()
        sol = [BASE]
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_single_section(self, tmp_path):
        """Read setup file with single non-empty section."""
        content = """\
            [plot]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            plot_variable = "tot_deposition"
            lang = "de"
            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = setups.dicts()
        sol = [
            merge_dicts(
                BASE,
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "level": None,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                            "lang": "de",
                        }
                    ],
                },
                overwrite_seqs=True,
            )
        ]
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )


class Test_Multiple:
    def test_multiple_parallel_empty_sections(self, tmp_path):
        """Read setup file with multiple parallel empty sections."""
        content = """\
            [plot1]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"

            [plot2]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = setups.dicts()
        sol = [BASE] * 2
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_two_nested_empty_sections(self, tmp_path):
        """Read setup file with two nested empty sections."""
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"

            [_base.plot]
            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = setups.dicts()
        sol = [BASE]
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_nested_sections(self, tmp_path):
        """Read setup file with two nested non-empty sections."""
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            lang = "de"

            [_base.con]
            plot_variable = "concentration"

            [_base._dep]
            plot_variable = "tot_deposition"

            [_base._dep._tot._ch.en]
            domain = "ch"
            lang = "en"

            [_base._dep._tot._ch.de]
            domain = "ch"
            lang = "de"

            [_base._dep._tot._full.en]
            domain = "full"
            lang = "en"

            [_base._dep._tot._full.de]
            domain = "full"
            lang = "de"

            [_base._dep.wet]
            plot_variable = "wet_deposition"

            [_base._dep.wet.en]
            lang = "en"

            [_base._dep.wet.de]
            lang = "de"
            """
        sol_base = merge_dicts(
            BASE,
            {
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "dimensions": {
                            "level": None,
                            "variable": ("dry_deposition", "wet_deposition"),
                        },
                        "lang": "de",
                    }
                ],
            },
            overwrite_seqs=True,
        )
        sol_specific = [
            {
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "dimensions": {"variable": "concentration"},
                    }
                ],
            },
            {"panels": [{"domain": "ch", "lang": "en"}]},
            {"panels": [{"domain": "ch", "lang": "de"}]},
            {"panels": [{"domain": "full", "lang": "en"}]},
            {"panels": [{"domain": "full", "lang": "de"}]},
            {
                "panels": [
                    {
                        "plot_variable": "wet_deposition",
                        "dimensions": {"variable": "wet_deposition"},
                    }
                ]
            },
            {
                "panels": [
                    {
                        "plot_variable": "wet_deposition",
                        "dimensions": {"variable": "wet_deposition"},
                        "lang": "en",
                    }
                ]
            },
            {
                "panels": [
                    {
                        "plot_variable": "wet_deposition",
                        "dimensions": {"variable": "wet_deposition"},
                        "lang": "de",
                    }
                ]
            },
        ]
        sol = [
            merge_dicts(sol_base, sol_spc, overwrite_seqs=True)
            for sol_spc in sol_specific
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_override(self, tmp_path):
        """Read setup file and override some parameters."""
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            lang = "de"

            [_base.con]
            plot_variable = "concentration"

            [_base._dep]
            plot_variable = "tot_deposition"

            [_base._dep._tot._ch.en]
            domain = "ch"
            lang = "en"

            [_base._dep._tot._ch.de]
            domain = "ch"
            lang = "de"

            [_base._dep._tot._full.en]
            domain = "full"
            lang = "en"

            [_base._dep._tot._full.de]
            domain = "full"
            lang = "de"

            [_base._dep._wet.en]
            plot_variable = "wet_deposition"
            lang = "en"

            [_base._dep._wet.de]
            plot_variable = "wet_deposition"
            lang = "de"
            """
        override = {"lang": "de"}
        sol_base = merge_dicts(
            BASE,
            {
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "lang": "de",
                        "dimensions": {
                            "level": None,
                            "variable": ("dry_deposition", "wet_deposition"),
                        },
                    }
                ],
            },
            overwrite_seqs=True,
        )
        sol_specific = [
            {
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "dimensions": {"variable": "concentration"},
                    }
                ]
            },
            {"panels": [{"domain": "ch", "lang": "de"}]},
            {"panels": [{"domain": "full", "lang": "de"}]},
            {
                "panels": [
                    {
                        "plot_variable": "wet_deposition",
                        "dimensions": {"variable": "wet_deposition"},
                    }
                ]
            },
        ]
        sol = [
            merge_dicts(sol_base, sol_spc, overwrite_seqs=True)
            for sol_spc in sol_specific
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read(override=override)
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )


class Test_RealCase:
    def test_semi_real(self, tmp_path):
        """Test setup file based on a real case, with some groups indented."""
        content = """\
            # PyFlexPlot setup file to create deterministic NAZ plots

            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"

            [_base._concentration]

                [_base._concentration._full]

                    [_base._concentration._full.en]

                    [_base._concentration._full.de]

                [_base._concentration._ch]

                    [_base._concentration._ch.en]

                    [_base._concentration._ch.de]

                [_base._concentration._integr]

                    [_base._concentration._integr._full.en]

                    [_base._concentration._integr._full.de]

                    [_base._concentration._integr._ch.en]

                    [_base._concentration._integr._ch.de]

            [_base._deposition]

            [_base._deposition._affected_area]

            [_base._deposition._full.en]

            [_base._deposition._full.de]

            [_base._deposition._ch.en]

            [_base._deposition._ch.de]

            [_base._deposition._affected_area._full.en]

            [_base._deposition._affected_area._full.de]

            [_base._deposition._affected_area._ch.en]

            [_base._deposition._affected_area._ch.de]

            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        # Note: Fails with tomlkit, but works with toml (2020-02-18)
        assert len(setups) == 16

    def test_opr_like(self, tmp_path):
        """Setup based on real config for operational cosmo-1 plots."""
        content = """\
            [concentration]
            infile = "data/cosmo1_2019052800.nc"
            outfile = "concentration_{species_id}_{domain}_{lang}_{time:02d}.png"
            model = "COSMO-1"
            species_id = "*"
            plot_variable = "concentration"
            combine_species = false
            integrate = false
            level = 0
            time = "*"
            domain = "full"
            lang = "de"
            # domain = "ch"
            # lang = "de"

            [concentration_integrated]
            infile = "data/cosmo1_2019052800.nc"
            outfile = "integr_concentr_{species_id}_{domain}_{lang}_{time:02d}.png"
            model = "COSMO-1"
            species_id = "*"
            plot_variable = "concentration"
            combine_species = false
            integrate = true
            level = 0
            time = -1
            domain = "full"
            lang = "de"
            # domain = "ch"
            # lang = "de"

            [tot_deposition]
            infile = "data/cosmo1_2019052800.nc"
            model = "COSMO-1"
            species_id = "*"
            outfile = "tot_deposition_{domain}_{lang}_{time:02d}.png"
            plot_variable = "tot_deposition"
            combine_species = true
            integrate = true
            time = -1

            [affected_area]
            outfile = "affected_area_{domain}_{lang}_{time:02d}.png"
            model = "COSMO-1"
            infile = "data/cosmo1_2019052800.nc"
            level = 0
            species_id = "*"
            plot_variable = "affected_area"
            combine_species = true
            integrate = true
            time = -1
            """
        sol = [
            {
                "files": {
                    "input": "data/cosmo1_2019052800.nc",
                    # "output": "concentration_{species_id}_{domain}_{lang}_{time:02d}.png",
                },
                "outfile": "concentration_{species_id}_{domain}_{lang}_{time:02d}.png",
                "panels": [
                    {
                        "combine_levels": False,
                        "combine_species": False,
                        "dimensions": {
                            "level": 0,
                            "variable": "concentration",
                        },
                        "dimensions_default": "all",
                        "domain": "full",
                        "plot_variable": "concentration",
                        "integrate": False,
                        "lang": "de",
                    }
                ],
                "model": {
                    "name": "COSMO-1",
                    "simulation_type": "deterministic",
                },
            },
            {
                "files": {
                    "input": "data/cosmo1_2019052800.nc",
                    # "output": (
                    #     "integr_concentr_{species_id}_{domain}_{lang}_{time:02d}.png"
                    # ),
                },
                "outfile": (
                    "integr_concentr_{species_id}_{domain}_{lang}_{time:02d}.png"
                ),
                "panels": [
                    {
                        "combine_levels": False,
                        "combine_species": False,
                        "dimensions": {
                            "level": 0,
                            "time": -1,
                            "variable": "concentration",
                        },
                        "domain": "full",
                        "plot_variable": "concentration",
                        "integrate": True,
                        "lang": "de",
                    }
                ],
                "model": {
                    "name": "COSMO-1",
                    "simulation_type": "deterministic",
                },
            },
            {
                "files": {
                    "input": "data/cosmo1_2019052800.nc",
                    # "output": "tot_deposition_{domain}_{lang}_{time:02d}.png",
                },
                "outfile": "tot_deposition_{domain}_{lang}_{time:02d}.png",
                "panels": [
                    {
                        "combine_levels": False,
                        "combine_species": True,
                        "dimensions": {
                            "time": -1,
                            "variable": ("dry_deposition", "wet_deposition"),
                        },
                        "dimensions_default": "all",
                        "domain": "full",
                        "plot_variable": "tot_deposition",
                        "integrate": True,
                        "lang": "en",
                    }
                ],
                "model": {
                    "name": "COSMO-1",
                    "simulation_type": "deterministic",
                },
            },
            {
                "files": {
                    "input": "data/cosmo1_2019052800.nc",
                    # "output": "affected_area_{domain}_{lang}_{time:02d}.png",
                },
                "outfile": "affected_area_{domain}_{lang}_{time:02d}.png",
                "panels": [
                    {
                        "combine_levels": False,
                        "combine_species": True,
                        "dimensions": {
                            "level": 0,
                            "time": -1,
                            "variable": (
                                "concentration",
                                "dry_deposition",
                                "wet_deposition",
                            ),
                        },
                        "dimensions_default": "all",
                        "domain": "full",
                        "plot_variable": "affected_area",
                        "integrate": True,
                        "lang": "en",
                    }
                ],
                "model": {
                    "name": "COSMO-1",
                    "simulation_type": "deterministic",
                },
            },
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )


class Test_Wildcards:
    def test_simple(self, tmp_path):
        """Apply a subgroup to multiple groups with a wildcard."""
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"

            [_base._concentration]
            plot_variable = "concentration"

            [_base._deposition]
            plot_variable = "tot_deposition"

            [_base."*".de]
            lang = "de"

            [_base."*".en]
            lang = "en"

            """
        sol = [
            merge_dicts(BASE, dct, overwrite_seqs=True)
            for dct in [
                {
                    "panels": [
                        {
                            "plot_variable": "concentration",
                            "dimensions": {"variable": "concentration"},
                            "lang": "de",
                        }
                    ]
                },
                {
                    "panels": [
                        {
                            "plot_variable": "concentration",
                            "dimensions": {"variable": "concentration"},
                            "lang": "en",
                        }
                    ]
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "level": None,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                            "lang": "de",
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "level": None,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                            "lang": "en",
                        }
                    ],
                },
            ]
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_equal_depth(self, tmp_path):
        """Apply double-star wildcard subdict to an equal-depth nested dict."""
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"

            [_base."_concentration+"]
            plot_variable = "concentration"

            [_base."_deposition+"]
            plot_variable = "tot_deposition"

            ["**"._ch.de]
            domain = "ch"
            lang = "de"

            ["**"._ch.en]
            domain = "ch"
            lang = "en"

            ["**"._full.de]
            domain = "full"
            lang = "de"

            ["**"._full.en]
            domain = "full"
            lang = "en"

            """
        sol = [
            merge_dicts(
                BASE,
                dct,
                {"panels": [{"domain": domain, "lang": lang}]},
                overwrite_seqs=True,
            )
            for dct in [
                {"panels": [{"plot_variable": "concentration", "integrate": False}]},
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "level": None,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                            "integrate": False,
                        }
                    ],
                },
            ]
            for domain in ["ch", "full"]
            for lang in ["de", "en"]
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_double_variable_depth(self, tmp_path):
        """Apply double-star wildcard subdict to variable-depth nested dict."""
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"

            [_base."_concentration"]
            plot_variable = "concentration"

            [_base._concentration."_middle+"]
            time = 5

            [_base._concentration."_end+"]
            time = 10

            [_base."_deposition+"]
            plot_variable = "tot_deposition"

            ["**"._ch.de]
            domain = "ch"
            lang = "de"

            ["**"._ch.en]
            domain = "ch"
            lang = "en"

            ["**"._full.de]
            domain = "full"
            lang = "de"

            ["**"._full.en]
            domain = "full"
            lang = "en"

            """
        sol = [
            merge_dicts(
                BASE,
                dct,
                {"panels": [{"domain": domain, "lang": lang}]},
                overwrite_seqs=True,
            )
            for dct in [
                {
                    "panels": [
                        {"plot_variable": "concentration", "dimensions": {"time": 5}}
                    ]
                },
                {
                    "panels": [
                        {"plot_variable": "concentration", "dimensions": {"time": 10}}
                    ]
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "level": None,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ]
                },
            ]
            for domain in ["ch", "full"]
            for lang in ["de", "en"]
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_combine(self, tmp_path):
        """Apply single- and double-star wildcards in combination."""
        content = """\
            [_base]
            infile = "foo_{ens_member:02d}.nc"
            model = "COSMO-baz"

            [_base."_concentration"]
            plot_variable = "concentration"

            [_base."_deposition"]
            plot_variable = "tot_deposition"

            [_base."*"."_mean+"]
            ens_variable = "mean"
            outfile = "bar_mean_{lang}.png"

            [_base."*"."_max+"]
            ens_variable = "maximum"
            outfile = "bar_maximum_{lang}.png"

            ["**".de]
            lang = "de"

            ["**".en]
            lang = "en"

            """
        sol = [
            merge_dicts(
                {
                    "files": {
                        "input": "foo_{ens_member:02d}.nc",
                        # "output": f"bar_{ens_variable}_{{lang}}.png",
                    },
                    "outfile": f"bar_{ens_variable}_{{lang}}.png",
                    "model": {
                        "name": "COSMO-baz",
                    },
                    "panels": [
                        {
                            "plot_variable": plot_variable,
                            "ens_variable": ens_variable,
                            "lang": lang,
                            "dimensions": {
                                "variable": (
                                    "concentration"
                                    if plot_variable == "concentration"
                                    else ("dry_deposition", "wet_deposition")
                                ),
                            },
                        }
                    ],
                },
                overwrite_seqs=True,
            )
            for plot_variable in ["concentration", "tot_deposition"]
            for ens_variable in ["mean", "maximum"]
            for lang in ["de", "en"]
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )


class Test_IndividualParams_SingleOrMultipleValues:
    def test_species_id(self, tmp_path):
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            combine_species = false

            [_base.single]
            species_id = 1

            [_base.multiple]
            species_id = [1, 2, 3]

            """
        sol = [
            merge_dicts(
                BASE,
                {"panels": [{"dimensions": {"species_id": value}}]},
                overwrite_seqs=True,
            )
            for value in [1, 1, 2, 3]
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_level(self, tmp_path):
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            plot_variable = "concentration"

            [_base.single]
            level = 0

            [_base.multiple]
            level = [1, 2]

            """
        sol = [
            merge_dicts(
                BASE,
                {"panels": [{"dimensions": {"level": value}}]},
                overwrite_seqs=True,
            )
            for value in [0, 1, 2]
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

    def test_level_none(self, tmp_path):
        content = """\
            [base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            plot_variable = "tot_deposition"
            """
        sol = [
            merge_dicts(
                BASE,
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "level": None,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ],
                },
                overwrite_seqs=True,
            )
        ]
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = group.dicts()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )


class Test_Multipanel:
    def test_ens_variable(self, tmp_path):
        """Declare multi-panel plot based on ensemble variables."""
        content = """\
            [plot]
            infile = "foo_{ens_member:02d}.nc"
            outfile = "bar_ens_stats_multipanel.png"
            model = "COSMO-2E"
            ens_member_id = [1, 2, 3]
            plot_type = "multipanel"
            multipanel_param = "ens_variable"
            ens_variable = ["minimum", "maximum", "mean", "median"]
            """
        sol = {
            "layout": {
                "plot_type": "multipanel",
                "multipanel_param": "ens_variable",
            },
            "model": {"ens_member_id": (1, 2, 3)},
            "panels": [
                {"ens_variable": "minimum"},
                {"ens_variable": "maximum"},
                {"ens_variable": "mean"},
                {"ens_variable": "median"},
            ],
        }
        group = SetupFile(tmp_setup_file(tmp_path, content)).read()
        assert len(group) == 1
        res = next(iter(group)).dict()
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )

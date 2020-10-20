"""
Tests for module ``pyflexplot.setup.SetupFile``.
"""
# Standard library
from collections.abc import Sequence
from pprint import pformat
from textwrap import dedent

# Third-party
import pytest

# First-party
from pyflexplot.setup import SetupFile
from srutils.dict import merge_dicts
from srutils.testing import assert_nested_equal

# Local
from .shared import DEFAULT_PARAMS


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


def test_single_minimal_section(tmp_path):
    """Read setup file with single minimal section."""
    content = """\
        [plot]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"
        """
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    assert setups == [{**base, **DEFAULT_PARAMS}]


def test_single_minimal_renamed_section(tmp_path):
    """Read setup file with single minimal section with arbitrary name."""
    content = """\
        [foobar]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"
        """
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    assert setups == [{**base, **DEFAULT_PARAMS}]


def test_single_section(tmp_path):
    """Read setup file with single non-empty section."""
    content = """\
        [plot]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"
        input_variable = "deposition"
        lang = "de"
        """
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    assert len(setups) == 1
    res = next(iter(setups))
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    assert res != {**base, **DEFAULT_PARAMS}
    sol = merge_dicts(
        base,
        DEFAULT_PARAMS,
        {"input_variable": "deposition", "dimensions": {"level": None}, "lang": "de"},
    )
    assert res == sol


def test_multiple_parallel_empty_sections(tmp_path):
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
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    assert setups == [{**base, **DEFAULT_PARAMS}] * 2


def test_two_nested_empty_sections(tmp_path):
    """Read setup file with two nested empty sections."""
    content = """\
        [_base]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"

        [_base.plot]
        """
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    assert setups == [{**base, **DEFAULT_PARAMS}]


def test_multiple_nested_sections(tmp_path):
    """Read setup file with two nested non-empty sections."""
    content = """\
        [_base]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"
        lang = "de"

        [_base.con]
        input_variable = "concentration"

        [_base._dep]
        input_variable = "deposition"
        deposition_type = ["dry", "wet"]
        combine_deposition_types = true

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
        deposition_type = "wet"

        [_base._dep.wet.en]
        lang = "en"

        [_base._dep.wet.de]
        lang = "de"
        """
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    sol_base = merge_dicts(
        base,
        DEFAULT_PARAMS,
        {
            "input_variable": "deposition",
            "combine_deposition_types": True,
            "dimensions": {"level": None, "deposition_type": ("dry", "wet")},
            "lang": "de",
        },
    )
    sol_specific = [
        {
            "input_variable": "concentration",
            "combine_deposition_types": False,
            "dimensions": {"deposition_type": None},
        },
        {"domain": "ch", "lang": "en"},
        {"domain": "ch", "lang": "de"},
        {"domain": "full", "lang": "en"},
        {"domain": "full", "lang": "de"},
        {"dimensions": {"deposition_type": "wet"}},
        {"lang": "en", "dimensions": {"deposition_type": "wet"}},
        {"lang": "de", "dimensions": {"deposition_type": "wet"}},
    ]
    sol = [merge_dicts(sol_base, sol_spc) for sol_spc in sol_specific]
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    assert setups == sol


def test_multiple_override(tmp_path):
    """Read setup file and override some parameters."""
    content = """\
        [_base]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"
        lang = "de"

        [_base.con]
        input_variable = "concentration"

        [_base._dep]
        input_variable = "deposition"
        deposition_type = ["dry", "wet"]
        combine_deposition_types = true

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
        deposition_type = "wet"

        [_base._dep.wet.en]
        lang = "en"

        [_base._dep.wet.de]
        lang = "de"
        """
    override = {"lang": "de"}
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    sol_base = merge_dicts(
        base,
        DEFAULT_PARAMS,
        {
            "input_variable": "deposition",
            "combine_deposition_types": True,
            "dimensions": {"level": None, "deposition_type": ("dry", "wet")},
            "lang": "de",
        },
    )
    sol_specific = [
        {
            "input_variable": "concentration",
            "combine_deposition_types": False,
            "dimensions": {"deposition_type": None},
        },
        {"domain": "ch", "lang": "de"},
        {"domain": "full", "lang": "de"},
        {"dimensions": {"deposition_type": "wet"}},
        {"lang": "de", "dimensions": {"deposition_type": "wet"}},
    ]
    sol = [merge_dicts(sol_base, sol_spc) for sol_spc in sol_specific]
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read(override=override)
    assert setups == sol


def test_semi_realcase(tmp_path):
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


def test_realcase_opr_like(tmp_path):
    """Setup based on real config for operational cosmo-1 plots."""
    content = """\
        [concentration]
        infile = "data/cosmo1_2019052800.nc"
        outfile = "concentration_{species_id}_{domain}_{lang}_{time:02d}.png"
        model = "COSMO-1"
        species_id = "*"
        input_variable = "concentration"
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
        input_variable = "concentration"
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
        input_variable = "deposition"
        deposition_type = ["dry", "wet"]
        combine_deposition_types = true
        combine_species = true
        integrate = true
        time = -1

        [tot_deposition_affected_area]
        outfile = "affected_area_{domain}_{lang}_{time:02d}.png"
        model = "COSMO-1"
        plot_variable = "affected_area_mono"
        infile = "data/cosmo1_2019052800.nc"
        species_id = "*"
        input_variable = "deposition"
        deposition_type = ["dry", "wet"]
        combine_deposition_types = true
        combine_species = true
        integrate = true
        time = -1
        """
    dcts_sol = [
        {
            "base_time": None,
            "combine_deposition_types": False,
            "combine_levels": False,
            "combine_species": False,
            "dimensions": {
                "deposition_type": None,
                "level": 0,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": None,
            },
            "dimensions_default": "all",
            "domain": "full",
            "domain_size_lat": None,
            "domain_size_lon": None,
            "ens_member_id": None,
            "ens_param_mem_min": None,
            "ens_param_pctl": None,
            "ens_param_thr": None,
            "ens_param_time_win": None,
            "ens_variable": "none",
            "infile": "data/cosmo1_2019052800.nc",
            "input_variable": "concentration",
            "integrate": False,
            "lang": "de",
            "model": "COSMO-1",
            "multipanel_param": None,
            "outfile": "concentration_{species_id}_{domain}_{lang}_{time:02d}.png",
            "outfile_time_format": "%Y%m%d%H%M",
            "plot_type": "auto",
            "plot_variable": "auto",
            "scale_fact": 1.0,
        },
        {
            "base_time": None,
            "combine_deposition_types": False,
            "combine_levels": False,
            "combine_species": False,
            "dimensions": {
                "deposition_type": None,
                "level": 0,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": -1,
            },
            "dimensions_default": "all",
            "domain": "full",
            "domain_size_lat": None,
            "domain_size_lon": None,
            "ens_member_id": None,
            "ens_param_mem_min": None,
            "ens_param_pctl": None,
            "ens_param_thr": None,
            "ens_param_time_win": None,
            "ens_variable": "none",
            "infile": "data/cosmo1_2019052800.nc",
            "input_variable": "concentration",
            "integrate": True,
            "lang": "de",
            "model": "COSMO-1",
            "multipanel_param": None,
            "outfile": "integr_concentr_{species_id}_{domain}_{lang}_{time:02d}.png",
            "outfile_time_format": "%Y%m%d%H%M",
            "plot_type": "auto",
            "plot_variable": "auto",
            "scale_fact": 1.0,
        },
        {
            "base_time": None,
            "combine_deposition_types": True,
            "combine_levels": False,
            "combine_species": True,
            "dimensions": {
                "deposition_type": ("dry", "wet"),
                "level": None,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": -1,
            },
            "dimensions_default": "all",
            "domain": "full",
            "domain_size_lat": None,
            "domain_size_lon": None,
            "ens_member_id": None,
            "ens_param_mem_min": None,
            "ens_param_pctl": None,
            "ens_param_thr": None,
            "ens_param_time_win": None,
            "ens_variable": "none",
            "infile": "data/cosmo1_2019052800.nc",
            "input_variable": "deposition",
            "integrate": True,
            "lang": "en",
            "model": "COSMO-1",
            "multipanel_param": None,
            "outfile": "tot_deposition_{domain}_{lang}_{time:02d}.png",
            "outfile_time_format": "%Y%m%d%H%M",
            "plot_type": "auto",
            "plot_variable": "auto",
            "scale_fact": 1.0,
        },
        {
            "base_time": None,
            "combine_deposition_types": True,
            "combine_levels": False,
            "combine_species": True,
            "dimensions": {
                "deposition_type": ("dry", "wet"),
                "level": None,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": -1,
            },
            "dimensions_default": "all",
            "domain": "full",
            "domain_size_lat": None,
            "domain_size_lon": None,
            "ens_member_id": None,
            "ens_param_mem_min": None,
            "ens_param_pctl": None,
            "ens_param_thr": None,
            "ens_param_time_win": None,
            "ens_variable": "none",
            "infile": "data/cosmo1_2019052800.nc",
            "input_variable": "deposition",
            "integrate": True,
            "lang": "en",
            "model": "COSMO-1",
            "multipanel_param": None,
            "outfile": "affected_area_{domain}_{lang}_{time:02d}.png",
            "outfile_time_format": "%Y%m%d%H%M",
            "plot_type": "auto",
            "plot_variable": "affected_area_mono",
            "scale_fact": 1.0,
        },
    ]
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    dcts_res = setups.dicts()
    assert len(dcts_res) == len(dcts_sol)
    assert [d["outfile"] for d in dcts_res] == [d["outfile"] for d in dcts_sol]
    for dct_res, dct_sol in zip(dcts_res, dcts_sol):
        try:
            assert_nested_equal(dct_res, dct_sol, "res", "sol")
        except AssertionError:
            raise AssertionError(
                f"setups differ:\n\n{pformat(dct_res)}\n\n{pformat(dct_sol)}\n"
            )


def test_wildcard_simple(tmp_path):
    """Apply a subgroup to multiple groups with a wildcard."""
    content = """\
        [_base]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"

        [_base._concentration]
        input_variable = "concentration"

        [_base._deposition]
        input_variable = "deposition"

        [_base."*".de]
        lang = "de"

        [_base."*".en]
        lang = "en"

        """
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    sol = [
        merge_dicts(DEFAULT_PARAMS, base, dct)
        for dct in [
            {"input_variable": "concentration", "lang": "de"},
            {"input_variable": "concentration", "lang": "en"},
            {
                "input_variable": "deposition",
                "dimensions": {"level": None},
                "lang": "de",
            },
            {
                "input_variable": "deposition",
                "dimensions": {"level": None},
                "lang": "en",
            },
        ]
    ]
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    assert setups == sol


def test_double_wildcard_equal_depth(tmp_path):
    """Apply a double-star wildcard subdict to an equal-depth nested dict."""
    content = """\
        [_base]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"

        [_base."_concentration+"]
        input_variable = "concentration"

        [_base."_deposition+"]
        input_variable = "deposition"

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
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    sol = [
        merge_dicts(DEFAULT_PARAMS, base, dct, {"domain": domain, "lang": lang})
        for dct in [
            {"input_variable": "concentration", "integrate": False},
            {
                "input_variable": "deposition",
                "dimensions": {"level": None},
                "integrate": False,
            },
        ]
        for domain in ["ch", "full"]
        for lang in ["de", "en"]
    ]
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    assert setups.dicts() == sol


def test_double_wildcard_variable_depth(tmp_path):
    """Apply a double-star wildcard subdict to a variable-depth nested dict."""
    content = """\
        [_base]
        infile = "foo.nc"
        outfile = "bar.png"
        model = "COSMO-baz"

        [_base."_concentration"]
        input_variable = "concentration"

        [_base._concentration."_middle+"]
        time = 5

        [_base._concentration."_end+"]
        time = 10

        [_base."_deposition+"]
        input_variable = "deposition"

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
    base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
    sol = [
        merge_dicts(DEFAULT_PARAMS, base, dct, {"domain": domain, "lang": lang})
        for dct in [
            {"input_variable": "concentration", "dimensions": {"time": 5}},
            {"input_variable": "concentration", "dimensions": {"time": 10}},
            {"input_variable": "deposition", "dimensions": {"level": None}},
        ]
        for domain in ["ch", "full"]
        for lang in ["de", "en"]
    ]
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    res = setups.dicts()
    assert res == sol


def test_combine_wildcards(tmp_path):
    """Apply single- and double-star wildcards in combination."""
    content = """\
        [_base]
        infile = "foo_{ens_member:02d}.nc"
        model = "COSMO-baz"

        [_base."_concentration"]
        input_variable = "concentration"

        [_base."_deposition"]
        input_variable = "deposition"

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
            DEFAULT_PARAMS,
            {
                "infile": "foo_{ens_member:02d}.nc",
                "outfile": f"bar_{ens_variable}_{{lang}}.png",
                "model": "COSMO-baz",
                "input_variable": input_variable,
                "ens_variable": ens_variable,
                "lang": lang,
            },
        )
        for input_variable in ["concentration", "deposition"]
        for ens_variable in ["mean", "maximum"]
        for lang in ["de", "en"]
    ]
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    res = setups.dicts()
    assert len(res) == len(sol)
    assert res == sol


class Test_IndividualParams_SingleOrMultipleValues:
    def test_species_id(self, tmp_path):
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"

            [_base.single]
            species_id = 1

            [_base.multiple]
            species_id = [1, 2, 3]

            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        res = setups.dicts()
        base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
        sol = [
            merge_dicts(DEFAULT_PARAMS, base, {"dimensions": {"species_id": value}})
            for value in [1, (1, 2, 3)]
        ]
        assert res == sol

    def test_level(self, tmp_path):
        content = """\
            [_base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            input_variable = "concentration"

            [_base.single]
            level = 0

            [_base.multiple]
            level = [1, 2]

            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
        sol = [
            merge_dicts(DEFAULT_PARAMS, base, {"dimensions": {"level": value}})
            for value in [0, (1, 2)]
        ]
        assert setups.dicts() == sol

    def test_level_none(self, tmp_path):
        content = """\
            [base]
            infile = "foo.nc"
            outfile = "bar.png"
            model = "COSMO-baz"
            input_variable = "deposition"
            """
        setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
        base = {"infile": "foo.nc", "outfile": "bar.png", "model": "COSMO-baz"}
        sol = [
            merge_dicts(
                DEFAULT_PARAMS,
                base,
                {"input_variable": "deposition", "dimensions": {"level": None}},
            )
        ]
        assert setups.dicts() == sol


@pytest.mark.skip("not quite ready yet")
def test_multipanel_param_ens_variable(tmp_path):
    """Declare multi-panel plot based on ensemble variables."""
    content = """\
        [plot]
        infile = "foo_{ens_member:02d}.nc"
        outfile = "bar_ens_stats_multipanel.png"
        model = "COSMO-baz"
        ens_member_id = [1, 2, 3]
        plot_type = "multipanel"
        multipanel_param = "ens_variable"
        ens_variable = ["minimum", "maximum", "mean", "median"]
        """
    setups = SetupFile(tmp_setup_file(tmp_path, content)).read()
    res = setups.dicts()
    # SR_TODO Figure out what the solution here should be
    sol = []
    # sol = [
    #     merge_dicts(
    #         DUMMY_PARAMS,
    #         DEFAULT_PARAMS,
    #         {
    #             "infile": "data_{ens_member:02d}.nc",
    #             "outfile": "ens_stats_multipanel.png",
    #             "ens_member_id": (1, 2, 3),
    #             "plot_type": "multipanel",
    #             "multipanel_param": "ens_variable",
    #             "ens_variable": ("minimum", "maximum", "mean", "median"),
    #         }
    #     )
    # ]
    assert len(res) == len(sol)
    assert res == sol

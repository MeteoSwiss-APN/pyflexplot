# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.setup.InputSetupFile``.
"""
# Standard library
from collections.abc import Sequence
from pprint import pformat
from textwrap import dedent

# Third-party
import pytest

# First-party
from pyflexplot.setup import InputSetupFile
from srutils.testing import assert_nested_equal

# Local
from .shared import DEFAULT_KWARGS
from .shared import DEFAULT_SETUP


def fmt_val(val):
    if isinstance(val, str):
        return f'"{val}"'
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, Sequence):
        return f"[{', '.join([fmt_val(v) for v in val])}]"
    else:
        raise NotImplementedError(f"type {type(val).__name__}", val)


DEFAULT_TOML = "\n".join([f"{k} = {fmt_val(v)}" for k, v in DEFAULT_KWARGS.items()])


def read_tmp_setup_file(tmp_path, content, **kwargs):
    tmp_file = tmp_path / "setup.toml"
    tmp_file.write_text(dedent(content))
    return InputSetupFile(tmp_file).read(**kwargs)


def test_single_minimal_section(tmp_path):
    """Read setup file with single minimal section."""
    content = f"""\
        [plot]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_SETUP.dict()]


def test_single_minimal_renamed_section(tmp_path):
    """Read setup file with single minimal section with arbitrary name."""
    content = f"""\
        [foobar]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_SETUP.dict()]


def test_single_section(tmp_path):
    """Read setup file with single non-empty section."""
    content = f"""\
        [plot]
        {DEFAULT_TOML}
        input_variable = "deposition"
        lang = "de"
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert len(setups) == 1
    res = next(iter(setups))
    assert res != DEFAULT_SETUP.dict()
    sol = {
        **DEFAULT_SETUP.dict(),
        "input_variable": "deposition",
        "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "level": None},
        "lang": "de",
    }
    assert res == sol


def test_multiple_parallel_empty_sections(tmp_path):
    """Read setup file with multiple parallel empty sections."""
    content = f"""\
        [plot1]
        {DEFAULT_TOML}

        [plot2]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_SETUP.dict()] * 2


def test_two_nested_empty_sections(tmp_path):
    """Read setup file with two nested empty sections."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

        [_base.plot]
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_SETUP.dict()]


def test_multiple_nested_sections(tmp_path):
    """Read setup file with two nested non-empty sections."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}
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

        [_base._dep._tot._auto.en]
        domain = "auto"
        lang = "en"

        [_base._dep._tot._auto.de]
        domain = "auto"
        lang = "de"

        [_base._dep.wet]
        deposition_type = "wet"

        [_base._dep.wet.en]
        lang = "en"

        [_base._dep.wet.de]
        lang = "de"
        """
    sol_base = {
        **DEFAULT_SETUP.dict(),
        "input_variable": "deposition",
        "combine_deposition_types": True,
        "dimensions": {
            **DEFAULT_SETUP.dict()["dimensions"],
            "level": None,
            "deposition_type": ("dry", "wet"),
        },
        "lang": "de",
    }
    sol_specific = [
        {
            "input_variable": "concentration",
            "combine_deposition_types": False,
            "dimensions": {"deposition_type": None},
        },
        {"domain": "ch", "lang": "en"},
        {"domain": "ch", "lang": "de"},
        {"domain": "auto", "lang": "en"},
        {"domain": "auto", "lang": "de"},
        {"dimensions": {"deposition_type": "wet"}},
        {"lang": "en", "dimensions": {"deposition_type": "wet"}},
        {"lang": "de", "dimensions": {"deposition_type": "wet"}},
    ]
    sol = [
        {
            **sol_base,
            **sol_spc,
            "dimensions": {
                **sol_base.get("dimensions", {}),
                **sol_spc.get("dimensions", {}),
            },
        }
        for sol_spc in sol_specific
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == sol


def test_multiple_override(tmp_path):
    """Read setup file and override some parameters."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}
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

        [_base._dep._tot._auto.en]
        domain = "auto"
        lang = "en"

        [_base._dep._tot._auto.de]
        domain = "auto"
        lang = "de"

        [_base._dep.wet]
        deposition_type = "wet"

        [_base._dep.wet.en]
        lang = "en"

        [_base._dep.wet.de]
        lang = "de"
        """
    override = {
        "infile": "foo.nc",
        "lang": "de",
    }
    sol_base = {
        **DEFAULT_SETUP.dict(),
        "infile": "foo.nc",
        "input_variable": "deposition",
        "combine_deposition_types": True,
        "dimensions": {
            **DEFAULT_SETUP.dict()["dimensions"],
            "level": None,
            "deposition_type": ("dry", "wet"),
        },
        "lang": "de",
    }
    sol_specific = [
        {
            "input_variable": "concentration",
            "combine_deposition_types": False,
            "dimensions": {"deposition_type": None},
        },
        {"domain": "ch", "lang": "de"},
        {"domain": "auto", "lang": "de"},
        {"dimensions": {"deposition_type": "wet"}},
        {"lang": "de", "dimensions": {"deposition_type": "wet"}},
    ]
    sol = [
        {
            **sol_base,
            **sol_spc,
            "dimensions": {
                **sol_base.get("dimensions", {}),
                **sol_spc.get("dimensions", {}),
            },
        }
        for sol_spc in sol_specific
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    setups = read_tmp_setup_file(tmp_path, content, override=override)
    assert setups == sol


def test_semi_realcase(tmp_path):
    """Test setup file based on a real case, with some groups indented."""
    content = f"""\
        # PyFlexPlot setup file to create deterministic NAZ plots

        [_base]
        {DEFAULT_TOML}

        [_base._concentration]

            [_base._concentration._auto]

                [_base._concentration._auto.en]

                [_base._concentration._auto.de]

            [_base._concentration._ch]

                [_base._concentration._ch.en]

                [_base._concentration._ch.de]

            [_base._concentration._integr]

                [_base._concentration._integr._auto.en]

                [_base._concentration._integr._auto.de]

                [_base._concentration._integr._ch.en]

                [_base._concentration._integr._ch.de]

        [_base._deposition]

        [_base._deposition._affected_area]

        [_base._deposition._auto.en]

        [_base._deposition._auto.de]

        [_base._deposition._ch.en]

        [_base._deposition._ch.de]

        [_base._deposition._affected_area._auto.en]

        [_base._deposition._affected_area._auto.de]

        [_base._deposition._affected_area._ch.en]

        [_base._deposition._affected_area._ch.de]

        """
    setups = read_tmp_setup_file(tmp_path, content)
    # Note: Fails with tomlkit, but works with toml (2020-02-18)
    assert len(setups) == 16


@pytest.mark.skip("TODO next")
def test_realcase_opr_like(tmp_path):
    """Setup based on real config for operational cosmo1 plots."""
    content = """\
        [concentration]
        infile = "data/cosmo-1_2019052800.nc"
        outfile = "concentration_{species_id}_{domain}_{lang}_{time:02d}.png"
        species_id = "*"
        input_variable = "concentration"
        combine_species = false
        integrate = false
        level = 0
        time = "*"
        domain = "auto"
        lang = "de"
        # domain = "ch"
        # lang = "de"

        [concentration_integrated]
        infile = "data/cosmo-1_2019052800.nc"
        outfile = "integr_concentr_{species_id}_{domain}_{lang}_{time:02d}.png"
        species_id = "*"
        input_variable = "concentration"
        combine_species = false
        integrate = true
        level = 0
        time = -1
        domain = "auto"
        lang = "de"
        # domain = "ch"
        # lang = "de"

        [tot_deposition]
        infile = "data/cosmo-1_2019052800.nc"
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
        plot_variable = "affected_area_mono"
        infile = "data/cosmo-1_2019052800.nc"
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
            "infile": "data/cosmo-1_2019052800.nc",
            "outfile": "concentration_{species_id}_{domain}_{lang}_{time:02d}.png",
            "input_variable": "concentration",
            "plot_variable": "auto",
            "ens_variable": "none",
            "plot_type": "auto",
            "multipanel_param": None,
            "integrate": False,
            "combine_deposition_types": False,
            "combine_levels": False,
            "combine_species": False,
            "ens_member_id": None,
            "ens_param_mem_min": None,
            "ens_param_thr": None,
            "ens_param_time_win": None,
            "lang": "de",
            "domain": "auto",
            "dimensions": {
                "deposition_type": None,
                "level": 0,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": None,
            },
        },
        {
            "infile": "data/cosmo-1_2019052800.nc",
            "outfile": "integr_concentr_{species_id}_{domain}_{lang}_{time:02d}.png",
            "input_variable": "concentration",
            "plot_variable": "auto",
            "ens_variable": "none",
            "plot_type": "auto",
            "multipanel_param": None,
            "integrate": True,
            "combine_deposition_types": False,
            "combine_levels": False,
            "combine_species": False,
            "ens_member_id": None,
            "ens_param_mem_min": None,
            "ens_param_thr": None,
            "ens_param_time_win": None,
            "lang": "de",
            "domain": "auto",
            "dimensions": {
                "deposition_type": None,
                "level": 0,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": -1,
            },
        },
        {
            "infile": "data/cosmo-1_2019052800.nc",
            "outfile": "tot_deposition_{domain}_{lang}_{time:02d}.png",
            "input_variable": "deposition",
            "plot_variable": "auto",
            "ens_variable": "none",
            "plot_type": "auto",
            "multipanel_param": None,
            "integrate": True,
            "combine_deposition_types": True,
            "combine_levels": False,
            "combine_species": True,
            "ens_member_id": None,
            "ens_param_mem_min": None,
            "ens_param_thr": None,
            "ens_param_time_win": None,
            "lang": "en",
            "domain": "auto",
            "dimensions": {
                "deposition_type": ("dry", "wet"),
                "level": None,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": -1,
            },
        },
        {
            "infile": "data/cosmo-1_2019052800.nc",
            "outfile": "affected_area_{domain}_{lang}_{time:02d}.png",
            "input_variable": "deposition",
            "plot_variable": "affected_area_mono",
            "ens_variable": "none",
            "plot_type": "auto",
            "multipanel_param": None,
            "integrate": True,
            "combine_deposition_types": True,
            "combine_levels": False,
            "combine_species": True,
            "ens_member_id": None,
            "ens_param_mem_min": None,
            "ens_param_thr": None,
            "ens_param_time_win": None,
            "lang": "en",
            "domain": "auto",
            "dimensions": {
                "deposition_type": ("dry", "wet"),
                "level": None,
                "nageclass": None,
                "noutrel": None,
                "numpoint": None,
                "species_id": None,
                "time": -1,
            },
        },
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    dcts_res = setups.dicts()
    assert len(dcts_res) == len(dcts_sol)
    assert [d["outfile"] for d in dcts_res] == [d["outfile"] for d in dcts_sol]
    for dct_res, dct_sol in zip(dcts_res, dcts_sol):
        try:
            assert_nested_equal(dct_res, dct_sol)
        except AssertionError:
            raise AssertionError(
                f"setups differ:\n\n{pformat(dct_res)}\n\n{pformat(dct_sol)}\n"
            )


def test_wildcard_simple(tmp_path):
    """Apply a subgroup to multiple groups with a wildcard."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

        [_base._concentration]
        input_variable = "concentration"

        [_base._deposition]
        input_variable = "deposition"

        [_base."*".de]
        lang = "de"

        [_base."*".en]
        lang = "en"

        """
    sol = [
        {**DEFAULT_SETUP.dict(), **dct}
        for dct in [
            {"input_variable": "concentration", "lang": "de"},
            {"input_variable": "concentration", "lang": "en"},
            {
                "input_variable": "deposition",
                "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "level": None},
                "lang": "de",
            },
            {
                "input_variable": "deposition",
                "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "level": None},
                "lang": "en",
            },
        ]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == sol


def test_double_wildcard_equal_depth(tmp_path):
    """Apply a double-star wildcard subdict to an equal-depth nested dict."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

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

        ["**"._auto.de]
        domain = "auto"
        lang = "de"

        ["**"._auto.en]
        domain = "auto"
        lang = "en"

        """
    sol = [
        {**DEFAULT_SETUP.dict(), **dct, "domain": domain, "lang": lang}
        for dct in [
            {"input_variable": "concentration", "integrate": False},
            {
                "input_variable": "deposition",
                "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "level": None},
                "integrate": False,
            },
        ]
        for domain in ["ch", "auto"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups.dicts() == sol


def test_double_wildcard_variable_depth(tmp_path):
    """Apply a double-star wildcard subdict to a variable-depth nested dict."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

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

        ["**"._auto.de]
        domain = "auto"
        lang = "de"

        ["**"._auto.en]
        domain = "auto"
        lang = "en"

        """
    sol = [
        {**DEFAULT_SETUP.dict(), **dct, "domain": domain, "lang": lang}
        for dct in [
            {
                "input_variable": "concentration",
                "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "time": 5},
            },
            {
                "input_variable": "concentration",
                "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "time": 10},
            },
            {
                "input_variable": "deposition",
                "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "level": None},
            },
        ]
        for domain in ["ch", "auto"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    res = setups.dicts()
    assert res == sol


def test_combine_wildcards(tmp_path):
    """Apply single- and double-star wildcards in combination."""
    content = """\
        [_base]
        infile = "data_{ens_member:02d}.nc"

        [_base."_concentration"]
        input_variable = "concentration"

        [_base."_deposition"]
        input_variable = "deposition"

        [_base."*"."_mean+"]
        ens_variable = "mean"
        outfile = "ensemble_mean_{lang}.png"

        [_base."*"."_max+"]
        ens_variable = "maximum"
        outfile = "ensemble_maximum_{lang}.png"

        ["**".de]
        lang = "de"

        ["**".en]
        lang = "en"

        """
    sol = [
        {
            **DEFAULT_SETUP.dict(),
            "infile": "data_{ens_member:02d}.nc",
            "input_variable": input_variable,
            "ens_variable": ens_variable,
            "outfile": f"ensemble_{ens_variable}_{{lang}}.png",
            "lang": lang,
        }
        for input_variable in ["concentration", "deposition"]
        for ens_variable in ["mean", "maximum"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    res = setups.dicts()
    assert len(res) == len(sol)
    assert res == sol


class Test_IndividualParams_SingleOrMultipleValues:
    def test_species_id(self, tmp_path):
        content = f"""\
            [_base]
            {DEFAULT_TOML}

            [_base.single]
            species_id = 1

            [_base.multiple]
            species_id = [1, 2, 3]

            """
        setups = read_tmp_setup_file(tmp_path, content)
        res = setups.dicts()
        sol = [
            {
                **DEFAULT_SETUP.dict(),
                "dimensions": {
                    **DEFAULT_SETUP.dimensions.compact_dict(),
                    "species_id": value,
                },
            }
            for value in [1, (1, 2, 3)]
        ]
        assert res == sol

    def test_level(self, tmp_path):
        content = f"""\
            [_base]
            {DEFAULT_TOML}
            input_variable = "concentration"

            [_base.single]
            level = 0

            [_base.multiple]
            level = [1, 2]

            """
        setups = read_tmp_setup_file(tmp_path, content)
        sol = [
            {
                **DEFAULT_SETUP.dict(),
                "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "level": value},
            }
            for value in [0, (1, 2)]
        ]
        assert setups.dicts() == sol

    def test_level_none(self, tmp_path):
        content = f"""\
            [base]
            {DEFAULT_TOML}
            input_variable = "deposition"

            """
        setups = read_tmp_setup_file(tmp_path, content)
        sol = [
            {
                **DEFAULT_SETUP.dict(),
                "input_variable": "deposition",
                "dimensions": {**DEFAULT_SETUP.dict()["dimensions"], "level": None},
            },
        ]
        assert setups.dicts() == sol


@pytest.mark.skip("TODO multipanel")
def test_multipanel_param_ens_variable(tmp_path):
    """Declare multi-panel plot based on ensemble variables."""
    content = """\
        [plot]
        infile = "data_{ens_member:02d}.nc"
        outfile = "ens_stats_multipanel.png"
        ens_member_id = [1, 2, 3]
        plot_type = "multipanel"
        multipanel_param = "ens_variable"
        ens_variable = ["minimum", "maximum", "mean", "median"]

        """
    setups = read_tmp_setup_file(tmp_path, content)
    res = setups.dicts()
    sol = [
        {
            **DEFAULT_SETUP.dict(),
            "infile": "data_{ens_member:02d}.nc",
            "outfile": "ens_stats_multipanel.png",
            "ens_member_id": (1, 2, 3),
            "plot_type": "multipanel",
            "multipanel_param": "ens_variable",
            "ens_variable": ("minimum", "maximum", "mean", "median"),
        }
    ]
    assert len(res) == len(sol)
    assert res == sol

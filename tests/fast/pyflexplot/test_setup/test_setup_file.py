# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.setup.InputSetupFile``.
"""
# Standard library
from collections.abc import Sequence
from textwrap import dedent

# First-party
from pyflexplot.setup import InputSetupFile

# Local
from .test_setup import DEFAULT_KWARGS
from .test_setup import DEFAULT_SETUP


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


def test_read_single_minimal_section(tmp_path):
    """Read setup file with single minimal section."""
    content = f"""\
        [plot]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_SETUP.dict()]


def test_read_single_minimal_renamed_section(tmp_path):
    """Read setup file with single minimal section with arbitrary name."""
    content = f"""\
        [foobar]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_SETUP.dict()]


def test_read_single_section(tmp_path):
    """Read setup file with single non-empty section."""
    content = f"""\
        [plot]
        {DEFAULT_TOML}
        input_variable = "deposition"
        lang = "de"
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert len(setups) == 1
    assert setups != [DEFAULT_SETUP.dict()]
    sol = [
        {
            **DEFAULT_SETUP.dict(),
            "input_variable": "deposition",
            "level": None,
            "lang": "de",
        },
    ]
    assert setups == sol


def test_read_multiple_parallel_empty_sections(tmp_path):
    """Read setup file with multiple parallel empty sections."""
    content = f"""\
        [plot1]
        {DEFAULT_TOML}

        [plot2]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_SETUP.dict()] * 2


def test_read_two_nested_empty_sections(tmp_path):
    """Read setup file with two nested empty sections."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

        [_base.plot]
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_SETUP.dict()]


def test_read_multiple_nested_sections(tmp_path):
    """Read setup file with two nested non-empty sections."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}
        lang = "de"

        [_base.con]
        input_variable = "concentration"

        [_base._dep]
        input_variable = "deposition"
        deposition_type = "tot"

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
        **DEFAULT_KWARGS,
        "input_variable": "deposition",
        "deposition_type": "tot",
        "level": None,
        "lang": "de",
    }
    sol_specific = [
        {"input_variable": "concentration", "deposition_type": "none"},
        {"domain": "ch", "lang": "en"},
        {"domain": "ch", "lang": "de"},
        {"domain": "auto", "lang": "en"},
        {"domain": "auto", "lang": "de"},
        {"deposition_type": "wet"},
        {"deposition_type": "wet", "lang": "en"},
        {"deposition_type": "wet", "lang": "de"},
    ]
    sol = [{**DEFAULT_SETUP.dict(), **sol_base, **d} for d in sol_specific]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == sol


def test_read_multiple_override(tmp_path):
    """Read setup file and override some parameters."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}
        lang = "de"

        [_base.con]
        input_variable = "concentration"

        [_base._dep]
        input_variable = "deposition"
        deposition_type = "tot"

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
        **DEFAULT_KWARGS,
        "infile": "foo.nc",
        "input_variable": "deposition",
        "deposition_type": "tot",
        "level": None,
        "lang": "de",
    }
    sol_specific = [
        {"input_variable": "concentration", "deposition_type": "none"},
        {"domain": "ch", "lang": "de"},
        {"domain": "auto", "lang": "de"},
        {"deposition_type": "wet"},
        {"deposition_type": "wet", "lang": "de"},
    ]
    sol = [{**DEFAULT_SETUP.dict(), **sol_base, **d} for d in sol_specific]
    setups = read_tmp_setup_file(tmp_path, content, override=override)
    assert setups == sol


def test_read_semi_realcase(tmp_path):
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


def test_read_wildcard_simple(tmp_path):
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
            {"input_variable": "deposition", "level": None, "lang": "de"},
            {"input_variable": "deposition", "level": None, "lang": "en"},
        ]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == sol


def test_read_double_wildcard_equal_depth(tmp_path):
    """Apply a double-star wildcard subdict to an equal-depth nested dict."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

        [_base._concentration]
        input_variable = "concentration"

        [_base._deposition]
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
            {"input_variable": "deposition", "level": None, "integrate": False},
        ]
        for domain in ["ch", "auto"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups.dicts() == sol


def test_read_double_wildcard_variable_depth(tmp_path):
    """Apply a double-star wildcard subdict to a variable-depth nested dict."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

        [_base._concentration]
        input_variable = "concentration"

        [_base._concentration._time10]
        time = 10

        [_base._deposition]
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
            {"input_variable": "concentration", "time": (10,)},
            {"input_variable": "deposition", "level": None},
        ]
        for domain in ["ch", "auto"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups.dicts() == sol


def test_read_combine_wildcards(tmp_path):
    """Apply single- and double-star wildcards in combination."""
    content = """\
        [_base]
        infile = "data_{ens_member:02d}.nc"

        [_base._concentration]
        input_variable = "concentration"

        [_base._deposition]
        input_variable = "deposition"

        [_base."*"._mean]
        plot_type = "ens_mean"
        outfile = "ens_mean_{lang}.png"

        [_base."*"._max]
        plot_type = "ens_max"
        outfile = "ens_max_{lang}.png"

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
            "plot_type": plot_type,
            "outfile": f"{plot_type}_{{lang}}.png",
            "lang": lang,
        }
        for input_variable in ["concentration", "deposition"]
        for plot_type in ["ens_mean", "ens_max"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups.dicts() == sol


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
        sol = [
            {**DEFAULT_SETUP.dict(), "species_id": value} for value in [(1,), (1, 2, 3)]
        ]
        assert setups.dicts() == sol

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
        sol = [{**DEFAULT_SETUP.dict(), "level": value} for value in [(0,), (1, 2)]]
        assert setups.dicts() == sol

    def test_level_none(self, tmp_path):
        content = f"""\
            [base]
            {DEFAULT_TOML}
            input_variable = "deposition"

            """
        setups = read_tmp_setup_file(tmp_path, content)
        sol = [
            {**DEFAULT_SETUP.dict(), "input_variable": "deposition", "level": None},
        ]
        assert setups.dicts() == sol

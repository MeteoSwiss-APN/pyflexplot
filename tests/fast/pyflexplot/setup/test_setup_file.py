#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.setup.SetupFile``.
"""
# Standard library
from collections.abc import Sequence
from textwrap import dedent

# First-party
from pyflexplot.setup import SetupFile

# Local
from .test_setup import CREATE_DEFAULT_SETUP
from .test_setup import DEFAULT_KWARGS


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
    return SetupFile(tmp_file).read(**kwargs)


def test_read_single_minimal_section(tmp_path):
    """Read setup file with single minimal section."""
    content = f"""\
        [plot]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [CREATE_DEFAULT_SETUP().dict()]


def test_read_single_minimal_renamed_section(tmp_path):
    """Read setup file with single minimal section with arbitrary name."""
    content = f"""\
        [foobar]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [CREATE_DEFAULT_SETUP().dict()]


def test_read_single_section(tmp_path):
    """Read setup file with single non-empty section."""
    content = f"""\
        [plot]
        {DEFAULT_TOML}
        variable = "deposition"
        lang = "de"
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert len(setups) == 1
    assert setups != [CREATE_DEFAULT_SETUP().dict()]
    sol = [
        {
            **CREATE_DEFAULT_SETUP().dict(),
            "variable": "deposition",
            "level_idx": None,
            "lang": "de",
        }
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
    assert setups == [CREATE_DEFAULT_SETUP().dict()] * 2


def test_read_two_nested_empty_sections(tmp_path):
    """Read setup file with two nested empty sections."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

        [_base.plot]
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [CREATE_DEFAULT_SETUP().dict()]


def test_read_multiple_nested_sections(tmp_path):
    """Read setup file with two nested non-empty sections."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}
        lang = "de"

        [_base.con]
        variable = "concentration"

        [_base._dep]
        variable = "deposition"
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
        "variable": "deposition",
        "deposition_type": "tot",
        "level_idx": None,
        "lang": "de",
    }
    sol_specific = [
        {"variable": "concentration", "deposition_type": "none", "level_idx": 0},
        {"domain": "ch", "lang": "en"},
        {"domain": "ch", "lang": "de"},
        {"domain": "auto", "lang": "en"},
        {"domain": "auto", "lang": "de"},
        {"deposition_type": "wet"},
        {"deposition_type": "wet", "lang": "en"},
        {"deposition_type": "wet", "lang": "de"},
    ]
    sol = [{**CREATE_DEFAULT_SETUP().dict(), **sol_base, **d} for d in sol_specific]
    setups = read_tmp_setup_file(tmp_path, content)
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
        variable = "concentration"

        [_base._deposition]
        variable = "deposition"

        [_base."*".de]
        lang = "de"

        [_base."*".en]
        lang = "en"

        """
    sol = [
        {**CREATE_DEFAULT_SETUP().dict(), **dct}
        for dct in [
            {"variable": "concentration", "lang": "de"},
            {"variable": "concentration", "lang": "en"},
            {"variable": "deposition", "level_idx": None, "lang": "de"},
            {"variable": "deposition", "level_idx": None, "lang": "en"},
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
        variable = "concentration"

        [_base._deposition]
        variable = "deposition"

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
        {**CREATE_DEFAULT_SETUP().dict(), **dct, "domain": domain, "lang": lang}
        for dct in [
            {"variable": "concentration", "integrate": False},
            {"variable": "deposition", "level_idx": None, "integrate": False},
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
        variable = "concentration"

        [_base._concentration._time10]
        time_idcs = [10]

        [_base._deposition]
        variable = "deposition"

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
        {**CREATE_DEFAULT_SETUP().dict(), **dct, "domain": domain, "lang": lang}
        for dct in [
            {"variable": "concentration", "time_idcs": (10,)},
            {"variable": "deposition", "level_idx": None},
        ]
        for domain in ["ch", "auto"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups.dicts() == sol


def test_read_combine_wildcards(tmp_path):
    """Apply single- and double-star wildcards in combination."""
    content = f"""\
        [_base]
        infiles = ["data_{{ens_member:02d}}.nc"]

        [_base._concentration]
        variable = "concentration"

        [_base._deposition]
        variable = "deposition"

        [_base."*"._mean]
        plot_type = "ens_mean"
        outfile = "ens_mean_{{lang}}.png"

        [_base."*"._max]
        plot_type = "ens_max"
        outfile = "ens_max_{{lang}}.png"

        ["**".de]
        lang = "de"

        ["**".en]
        lang = "en"

        """
    sol = [
        {
            **CREATE_DEFAULT_SETUP().dict(),
            "infiles": ("data_{ens_member:02d}.nc",),
            "variable": variable,
            "level_idx": {"concentration": 0, "deposition": None}[variable],
            "plot_type": plot_type,
            "outfile": f"{plot_type}_{{lang}}.png",
            "lang": lang,
        }
        for variable in ["concentration", "deposition"]
        for plot_type in ["ens_mean", "ens_max"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups.dicts() == sol

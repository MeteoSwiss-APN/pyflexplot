#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.setup``.
"""
# Standard library
from collections.abc import Sequence
from textwrap import dedent

# First-party
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupCollection
from pyflexplot.setup import SetupFile

DEFAULT_KWARGS = {
    "infiles": ("foo.nc",),
    "outfile": "bar.png",
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


DEFAULT_TOML = "\n".join([f"{k} = {fmt_val(v)}" for k, v in DEFAULT_KWARGS.items()])

DEFAULT_CONFIG = {
    **DEFAULT_KWARGS,
    "member_ids": None,
    "variable": "concentration",
    "simulation_type": "deterministic",
    "plot_type": "auto",
    "domain": "auto",
    "lang": "en",
    "age_class_idx": 0,
    "deposition_type": "tot",
    "integrate": False,
    "level_idx": 0,
    "nout_rel_idx": 0,
    "release_point_idx": 0,
    "species_id": 1,
    "time_idx": 0,
    "scale_fact": None,
    "reverse_legend": False,
}


def test_default_setup_dict():
    """Check the default setupuration dict."""
    assert Setup(**DEFAULT_KWARGS) == DEFAULT_CONFIG


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
    assert setups == [DEFAULT_CONFIG]


def test_read_single_minimal_renamed_section(tmp_path):
    """Read setup file with single minimal section with arbitrary name."""
    content = f"""\
        [foobar]
        {DEFAULT_TOML}
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_CONFIG]


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
    assert setups != [DEFAULT_CONFIG]
    sol = [{**DEFAULT_CONFIG, "variable": "deposition", "lang": "de"}]
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
    assert setups == [DEFAULT_CONFIG] * 2


def test_read_two_nested_empty_sections(tmp_path):
    """Read setup file with two nested empty sections."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

        [_base.plot]
        """
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups == [DEFAULT_CONFIG]


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
        "lang": "de",
    }
    sol_specific = [
        {"variable": "concentration"},
        {
            "variable": "deposition",
            "deposition_type": "tot",
            "domain": "ch",
            "lang": "en",
        },
        {
            "variable": "deposition",
            "deposition_type": "tot",
            "domain": "ch",
            "lang": "de",
        },
        {
            "variable": "deposition",
            "deposition_type": "tot",
            "domain": "auto",
            "lang": "en",
        },
        {
            "variable": "deposition",
            "deposition_type": "tot",
            "domain": "auto",
            "lang": "de",
        },
        {"variable": "deposition", "deposition_type": "wet"},
        {"variable": "deposition", "deposition_type": "wet", "lang": "en"},
        {"variable": "deposition", "deposition_type": "wet", "lang": "de"},
    ]
    sol = [{**DEFAULT_CONFIG, **sol_base, **d} for d in sol_specific]
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
        {**DEFAULT_CONFIG, **dct}
        for dct in [
            {"variable": "concentration", "lang": "de"},
            {"variable": "concentration", "lang": "en"},
            {"variable": "deposition", "lang": "de"},
            {"variable": "deposition", "lang": "en"},
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
        {**DEFAULT_CONFIG, **dct, "domain": domain, "lang": lang}
        for dct in [
            {"variable": "concentration", "integrate": False},
            {"variable": "deposition", "integrate": False},
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
        time_idx = 10

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
        {**DEFAULT_CONFIG, **dct, "domain": domain, "lang": lang}
        for dct in [
            {"variable": "concentration", "time_idx": 10},
            {"variable": "deposition"},
        ]
        for domain in ["ch", "auto"]
        for lang in ["de", "en"]
    ]
    setups = read_tmp_setup_file(tmp_path, content)
    assert setups.dicts() == sol


class Test_SetupCollection:
    def create_partial_dicts(self):
        return [
            {**DEFAULT_KWARGS, **dct}
            for dct in [
                {"infiles": ("foo.nc",), "variable": "concentration", "domain": "ch"},
                {"infiles": ("bar.nc",), "variable": "deposition", "lang": "de"},
                {"age_class_idx": 1, "nout_rel_idx": 5, "release_point_idx": 3},
            ]
        ]

    def create_complete_dicts(self):
        return [{**DEFAULT_CONFIG, **dct} for dct in self.create_partial_dicts()]

    def create_setups(self):
        return [Setup(**dct) for dct in self.create_partial_dicts()]

    def test_dicts_setups(self):
        """Check the dicts and Setup objects used in the SetupCollection tests."""
        assert self.create_setups() == self.create_complete_dicts()

    def test_create_empty(self):
        setups = SetupCollection([])
        assert len(setups) == 0

    def test_from_setups(self):
        partial_dicts = self.create_partial_dicts()
        setups = SetupCollection(partial_dicts)
        assert len(setups) == len(partial_dicts)

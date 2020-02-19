#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.config``."""
# Standard library
from collections.abc import Sequence
from textwrap import dedent

# First-party
from pyflexplot.config import Config
from pyflexplot.config import ConfigCollection
from pyflexplot.config import ConfigFile

DEFAULT_KWARGS = {
    "infiles": ["foo.nc"],
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
}


def test_default_config_dict():
    """Check the default configuration dict."""
    assert Config(**DEFAULT_KWARGS) == DEFAULT_CONFIG


def read_tmp_config_file(tmp_path, content, **kwargs):
    tmp_file = tmp_path / "config.toml"
    tmp_file.write_text(dedent(content))
    return ConfigFile(tmp_file).read(**kwargs)


def test_read_single_minimal_section(tmp_path):
    """Read config file with single minimal section."""
    content = f"""\
        [plot]
        {DEFAULT_TOML}
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == [DEFAULT_CONFIG]


def test_read_single_minimal_renamed_section(tmp_path):
    """Read config file with single minimal section with arbitrary name."""
    content = f"""\
        [foobar]
        {DEFAULT_TOML}
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == [DEFAULT_CONFIG]


def test_read_single_section(tmp_path):
    """Read config file with single non-empty section."""
    content = f"""\
        [plot]
        {DEFAULT_TOML}
        variable = "deposition"
        lang = "de"
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert len(configs) == 1
    assert configs != [DEFAULT_CONFIG]
    sol = [{**DEFAULT_CONFIG, "variable": "deposition", "lang": "de"}]
    assert configs == sol


def test_read_multiple_parallel_empty_sections(tmp_path):
    """Read config file with multiple parallel empty sections."""
    content = f"""\
        [plot1]
        {DEFAULT_TOML}

        [plot2]
        {DEFAULT_TOML}
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == [DEFAULT_CONFIG] * 2


def test_read_two_nested_empty_sections(tmp_path):
    """Read config file with two nested empty sections."""
    content = f"""\
        [_base]
        {DEFAULT_TOML}

        [_base.plot]
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == [DEFAULT_CONFIG]


def test_read_multiple_nested_sections(tmp_path):
    """Read config file with two nested non-empty sections."""
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
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == sol


def test_read_semi_realcase(tmp_path):
    """Test config file based on a real case, with some groups indented."""
    content = f"""\
        # PyFlexPlot config file to create deterministic NAZ plots

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
    configs = read_tmp_config_file(tmp_path, content)
    # Note: Fails with tomlkit, but works with toml (2020-02-18)
    assert len(configs) == 16


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
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == sol


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
    configs = read_tmp_config_file(tmp_path, content)
    assert configs.as_dicts() == sol


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
    configs = read_tmp_config_file(tmp_path, content)
    assert configs.as_dicts() == sol


class Test_ConfigCollection:
    def create_partial_dicts(self):
        return [
            {**DEFAULT_KWARGS, **dct}
            for dct in [
                {"infiles": ["foo.nc"], "variable": "concentration", "domain": "ch"},
                {"infiles": ["bar.nc"], "variable": "deposition", "lang": "de"},
                {"age_class_idx": 1, "nout_rel_idx": 5, "release_point_idx": 3},
            ]
        ]

    def create_complete_dicts(self):
        return [{**DEFAULT_CONFIG, **dct} for dct in self.create_partial_dicts()]

    def create_configs(self):
        return [Config(**dct) for dct in self.create_partial_dicts()]

    def test_dicts_configs(self):
        """
        Check the dicts and Config objects used in the ConfigCollection tests.
        """
        assert self.create_configs() == self.create_complete_dicts()

    def test_create_empty(self):
        configs = ConfigCollection([])
        assert len(configs) == 0

    def test_from_configs(self):
        partial_dicts = self.create_partial_dicts()
        configs = ConfigCollection(partial_dicts)
        assert len(configs) == len(partial_dicts)

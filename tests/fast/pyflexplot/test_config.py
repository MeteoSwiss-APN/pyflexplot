#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.config``."""
import pytest

from textwrap import dedent

from pyflexplot.config import Config
from pyflexplot.config import ConfigFile
from pyflexplot.config import ConfigSet


DEFAULT_CONFIG = {
    "infiles": None,
    "member_ids": None,
    "outfile": None,
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
    assert Config() == DEFAULT_CONFIG


def read_tmp_config_file(tmp_path, content):
    tmp_file = tmp_path / "config.toml"
    tmp_file.write_text(dedent(content))
    return ConfigFile(tmp_file).read()


def test_read_single_empty_section(tmp_path):
    """Read config file with single empty section."""
    content = """\
        [plot]
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == [DEFAULT_CONFIG]


def test_read_single_empty_renamed_section(tmp_path):
    """Read config file with single empty section with arbitrary name."""
    content = """\
        [foobar]
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == [DEFAULT_CONFIG]


def test_read_single_section(tmp_path):
    """Read config file with single non-empty section."""
    content = """\
        [plot]
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
    content = """\
        [plot1]

        [plot2]
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == [DEFAULT_CONFIG] * 2


def test_read_two_nested_empty_sections(tmp_path):
    """Read config file with two nested empty sections."""
    content = """\
        [_base]

        [_base.plot]
        """
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == [DEFAULT_CONFIG]


def test_read_multiple_nested_sections(tmp_path):
    """Read config file with two nested non-empty sections."""
    content = """\
        [_base]
        infiles = ["file.nc"]
        lang = "de"

        [_base.con]
        variable = "concentration"

        [_base._dep]
        variable = "deposition"

        [_base._dep.tot]
        deposition_type = "tot"

        [_base._dep.wet]
        deposition_type = "wet"

        [_base._dep.wet.en]
        lang = "en"
        """
    sol_base = {
        "infiles": ["file.nc"],
        "lang": "de",
    }
    sol_specific = [
        {"variable": "concentration"},
        {"variable": "deposition", "deposition_type": "tot"},
        {"variable": "deposition", "deposition_type": "wet"},
        {"variable": "deposition", "deposition_type": "wet", "lang": "en"},
    ]
    sol = [{**DEFAULT_CONFIG, **sol_base, **d} for d in sol_specific]
    configs = read_tmp_config_file(tmp_path, content)
    assert configs == sol


class Test_ConfigSet:
    def create_partial_dicts(self):
        return [
            {"infiles": ["foo.nc"], "variable": "concentration", "domain": "ch"},
            {"infiles": ["bar.nc"], "variable": "deposition", "lang": "de"},
            {"age_class_idx": 1, "nout_rel_idx": 5, "release_point_idx": 3},
        ]

    def create_complete_dicts(self):
        return [{**DEFAULT_CONFIG, **dct} for dct in self.create_partial_dicts()]

    def create_configs(self):
        return [Config(**dct) for dct in self.create_partial_dicts()]

    def test_dicts_configs(self):
        """
        Check the dicts and Config objects used in the ConfigSet tests.
        """
        assert self.create_configs() == self.create_complete_dicts()

    def test_create_empty(self):
        configs = ConfigSet([])
        assert len(configs) == 0

    def test_from_configs(self):
        partial_dicts = self.create_partial_dicts()
        configs = ConfigSet(partial_dicts)
        assert len(configs) == len(partial_dicts)

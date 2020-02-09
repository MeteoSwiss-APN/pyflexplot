#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.config``."""
import pytest

from textwrap import dedent

from pyflexplot.config import Config
from pyflexplot.config import ConfigFile


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
    assert Config().asdict() == DEFAULT_CONFIG


def read_tmp_config_file(tmp_path, content):
    tmp_file = tmp_path / "config.toml"
    tmp_file.write_text(dedent(content))
    return ConfigFile(tmp_file).read()


def test_read_single_empty_section(tmp_path):
    """Read config file with single empty section."""
    content = """\
        [plot]
        """
    named_configs = read_tmp_config_file(tmp_path, content)
    assert named_configs == {"plot": DEFAULT_CONFIG}


def test_read_single_empty_renamed_section(tmp_path):
    """Read config file with single empty section with arbitrary name."""
    content = """\
        [foobar]
        """
    named_configs = read_tmp_config_file(tmp_path, content)
    assert named_configs == {"foobar": DEFAULT_CONFIG}


def test_read_single_section(tmp_path):
    """Read config file with single non-empty section."""
    content = """\
        [plot]
        variable = "deposition"
        lang = "de"
        """
    named_configs = read_tmp_config_file(tmp_path, content)
    assert len(named_configs) == 1
    config = next(iter(named_configs.values()))
    assert config.asdict() != DEFAULT_CONFIG
    sol = {**DEFAULT_CONFIG, "variable": "deposition", "lang": "de"}
    assert config.asdict() == sol


def test_read_multiple_parallel_empty_sections(tmp_path):
    """Read config file with multiple parallel empty sections."""
    content = """\
        [plot1]

        [plot2]
        """
    named_configs = read_tmp_config_file(tmp_path, content)
    assert list(named_configs) == ["plot1", "plot2"]
    config_dicts = [c.asdict() for c in named_configs.values()]
    assert config_dicts == [DEFAULT_CONFIG] * 2


def test_read_two_nested_empty_sections(tmp_path):
    """Read config file with two nested empty sections."""
    content = """\
        [base]

        [base.plot]
        """
    named_configs = read_tmp_config_file(tmp_path, content)
    assert named_configs == {"plot": DEFAULT_CONFIG}

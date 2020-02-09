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


def test_read_single_plot_empty(tmp_path):
    """Read file wit single empty plot definition."""
    content = """\
        [plot]
        """
    config = read_tmp_config_file(tmp_path, content)
    assert config.asdict() == DEFAULT_CONFIG

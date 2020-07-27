# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.data``.
"""
# Third-party
import numpy as np

# First-party
from pyflexplot.data import FieldStats
from srutils.testing import assert_nested_equal


class TestFieldStats:
    arr = np.array([[[0, 1, 1], [0, 2, 4]], [[0, 3, 9], [0, 4, 16]]], np.float32)
    dct = {
        "min": 0,
        "max": 16,
        "mean": 40 / 12,
        "median": 1.5,
        "min_nz": 1,
        "max_nz": 16,
        "mean_nz": 40 / 8,
        "median_nz": 3.5,
    }

    def test_stats(self):
        stats = FieldStats(self.arr)
        assert np.isclose(stats.min, self.dct["min"])
        assert np.isclose(stats.max, self.dct["max"])
        assert np.isclose(stats.mean, self.dct["mean"])
        assert np.isclose(stats.median, self.dct["median"])
        assert np.isclose(stats.min_nz, self.dct["min_nz"])
        assert np.isclose(stats.max_nz, self.dct["max_nz"])
        assert np.isclose(stats.mean_nz, self.dct["mean_nz"])
        assert np.isclose(stats.median_nz, self.dct["median_nz"])

    def test_summarize(self):
        stats = FieldStats(self.arr)
        summary = {"type": "FieldStats", **self.dct}
        assert_nested_equal(stats.summarize(), summary, float_close_ok=True)

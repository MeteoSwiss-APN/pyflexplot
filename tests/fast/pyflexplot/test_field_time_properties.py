# -*- coding: utf-8 -*-
"""
Tests for class ``pyflexplot.data.FieldTimeProperties``.
"""
# Third-party
import numpy as np

# First-party
from pyflexplot.data import FieldTimeProperties
from srutils.testing import assert_nested_equal

N = np.nan


class TestFieldTimeProperties:
    arr = np.array(
        [[[N, 0, N, 1, 1], [0, 2, 4, N, N]], [[0, N, N, 3, 9], [0, 4, N, 16, N]]],
        np.float32,
    )
    mask = np.array([[1, 1, 0, 1, 1], [1, 1, 1, 1, 0]], np.bool)
    mask_nz = np.array([[0, 0, 0, 1, 1], [0, 1, 1, 1, 0]], np.bool)
    stats = {
        "min": 0,
        "max": 16,
        "mean": 40 / 12,
        "median": 1.5,
    }
    stats_nz = {
        "min": 1,
        "max": 16,
        "mean": 40 / 8,
        "median": 3.5,
    }

    def test_stats(self):
        props = FieldTimeProperties(self.arr)
        assert np.isclose(props.stats.min, self.stats["min"])
        assert np.isclose(props.stats.max, self.stats["max"])
        assert np.isclose(props.stats.mean, self.stats["mean"])
        assert np.isclose(props.stats.median, self.stats["median"])

    def test_stats_nonzero(self):
        props = FieldTimeProperties(self.arr)
        assert np.isclose(props.stats_nz.min, self.stats_nz["min"])
        assert np.isclose(props.stats_nz.max, self.stats_nz["max"])
        assert np.isclose(props.stats_nz.mean, self.stats_nz["mean"])
        assert np.isclose(props.stats_nz.median, self.stats_nz["median"])

    def test_summarize(self):
        props = FieldTimeProperties(self.arr)
        summary = {
            "type": "FieldTimeProperties",
            "stats": {"type": "FieldStats", **self.stats},
            "stats_nz": {"type": "FieldStats", **self.stats_nz},
        }
        assert_nested_equal(
            props.summarize(), summary, "res", "sol", float_close_ok=True
        )

    def test_mask(self):
        props = FieldTimeProperties(self.arr)
        assert (props.mask == self.mask).all()

    def test_mask_nonzero(self):
        props = FieldTimeProperties(self.arr)
        assert (props.mask_nz == self.mask_nz).all()

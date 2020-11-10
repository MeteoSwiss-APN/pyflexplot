"""Tests for function ``pyflexplot.data.ensemble_probability``."""
# Third-party
import numpy as np
import pytest  # noqa: F401  # imported but unused

# First-party
from pyflexplot.data import ensemble_probability

# Shorthands for special values
N = np.nan
I = np.inf  # noqa: E741  # ambiguous variable name
J = -np.inf


def a(*args, **kwargs):
    """Shorthand function to create array."""
    return np.array(*args, **kwargs)


class TestEnsembleProbability2D:
    """Fraction of threshold exceedences along a 2D-array axis."""

    arr = np.array(
        [
            [3, 5, 6, 0, 1, 0, 3, 5],
            [6, 3, 4, 4, 4, 1, 6, 9],
            [2, 1, 4, 2, 6, 8, 4, 2],
            [9, 6, 7, 5, 7, 3, 1, 3],
        ]
    )

    thr = 4

    def test(self):
        n = self.arr.shape[0]
        res = ensemble_probability(self.arr, self.thr, n)
        sol = np.array([2, 2, 2, 1, 2, 1, 1, 2]) * 100 / n
        assert np.allclose(res, sol)


class TestEnsembleProbability3D:
    """Fraction of threshold exceedences along a 3D-array axis."""

    arr = np.array(
        [
            [[5, 2, 8], [0, 5, 0], [2, 1, 2], [9, 5, 2], [2, 0, 2]],
            [[5, 6, 7], [0, 0, 1], [2, 1, 9], [4, 5, 2], [0, 3, 9]],
            [[5, 6, 9], [1, 3, 8], [5, 3, 1], [3, 3, 3], [5, 3, 7]],
        ]
    )

    thr = 2

    def test(self):
        n = self.arr.shape[0]
        res = ensemble_probability(self.arr, self.thr, n)
        sol = (
            np.array([[3, 2, 3], [0, 2, 1], [1, 1, 1], [3, 3, 1], [1, 2, 2]]) * 100 / n
        )
        assert np.allclose(res, sol)
